import json
import logging
import os
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from random import shuffle
from typing import Any, Callable, Generator, Sequence

from datasets import Dataset, IterableDataset
from smolagents import Model
from tqdm import tqdm

from screensuite.basebenchmark import AnnotatedContent, EvaluationConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add a stream handler if none exists
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger.addHandler(handler)

TransformSampleCallableType = Callable[
    [Any, EvaluationConfig], AnnotatedContent | Generator[AnnotatedContent, None, None]
]

APPEND_ANSWER_LOCK = threading.Lock()


def append_answer(entry: dict, jsonl_file_str: str | Path) -> None:
    if isinstance(jsonl_file_str, str):
        jsonl_file = Path(jsonl_file_str)  # type: ignore
    else:
        jsonl_file = jsonl_file_str
    jsonl_file.parent.mkdir(parents=True, exist_ok=True)
    with APPEND_ANSWER_LOCK, open(jsonl_file, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(entry, default=str) + "\n")
    assert os.path.exists(jsonl_file), "File not found!"


def process_one_input(
    annotated_input: AnnotatedContent,
    model: Model,
    answers_file_path: Path | None = None,
    **generation_kwargs,
) -> AnnotatedContent | None:
    try:
        model_output = model.generate(annotated_input.messages, **generation_kwargs).content
        logger.debug(f"Generated output: {model_output}")
        if answers_file_path is not None:
            append_answer(
                {
                    "input": annotated_input.messages,
                    "answer": model_output,
                    "ground_truth": annotated_input.ground_truth,
                },
                answers_file_path,
            )
        annotated_input.output = model_output
        return annotated_input

    except Exception as e:
        # Keep per-sample errors concise; full tracebacks at debug level only (otherwise
        # thousands of failed samples flood the logs and stall the run)
        logger.error(f"Error processing generator input: {type(e).__name__}: {e}")
        logger.debug("Traceback for failed input:", exc_info=True)
        return None


def process_sample(
    sample: Any,
    model: Model,
    input_annotation_transform: TransformSampleCallableType,
    evaluation_config: EvaluationConfig,
    answers_file_path: Path | None = None,
    **generation_kwargs,
) -> list[AnnotatedContent | None]:
    model_responses: list[AnnotatedContent | None] = []
    annotated_input: AnnotatedContent | Generator[AnnotatedContent, None, None] = input_annotation_transform(
        sample, evaluation_config
    )
    if isinstance(annotated_input, Generator):
        for annotated_input in annotated_input:
            model_responses.append(process_one_input(annotated_input, model, answers_file_path, **generation_kwargs))
    else:
        model_responses.append(process_one_input(annotated_input, model, answers_file_path, **generation_kwargs))
    return model_responses


def select_examples_to_test(
    dataset: Sequence | Dataset | IterableDataset,
    evaluation_config: EvaluationConfig,
) -> Sequence | Dataset | IterableDataset:
    assert not (evaluation_config.max_samples_to_test and evaluation_config.test_mode), (
        "Cannot set max_samples_to_test in test_mode"
    )
    # If test mode is enabled, we only process a batch of 'evaluation_config.parallel_workers' samples
    if evaluation_config.test_mode:
        if isinstance(dataset, Dataset):
            return dataset.select(range(evaluation_config.parallel_workers))
        elif isinstance(dataset, Sequence):
            return dataset[: evaluation_config.parallel_workers]

    if evaluation_config.max_samples_to_test is not None:
        assert evaluation_config.max_samples_to_test >= 0
        if isinstance(dataset, Dataset):
            return dataset.shuffle(seed=42).select(range(min(evaluation_config.max_samples_to_test, len(dataset))))
        elif isinstance(dataset, Sequence):
            shuffle(list(dataset))
            return dataset[: evaluation_config.max_samples_to_test]
        elif isinstance(dataset, IterableDataset):
            return dataset.shuffle(seed=42).take(min(evaluation_config.max_samples_to_test, len(dataset)))
    return dataset


def get_model_responses(
    dataset: Sequence | Dataset | None,
    model: Model,
    input_annotation_transform: TransformSampleCallableType,
    benchmark_name: str,
    evaluation_config: EvaluationConfig,
    **generation_kwargs,
) -> list[AnnotatedContent | None]:
    """
    Parallelly generate responses from the model and return tuples of (model_generation, ground_truth)

    Returns:
        A list of tuples, where each tuple contains (model_generation, ground_truth)
    """
    run_storage_folder = f"./output/{benchmark_name}/{evaluation_config.run_name}"

    # Bounded in-flight window: only `chunk_size` samples are decoded/submitted at once,
    # otherwise loading a full split (e.g. 8.4k ScreenQA images) exhausts RAM.
    chunk_size = max(evaluation_config.parallel_workers * 8, 64)

    responses: list[AnnotatedContent | None] = []

    assert dataset is not None, "Dataset is required"

    dataset_to_test = select_examples_to_test(dataset, evaluation_config)

    if evaluation_config.run_name is not None:
        print(f"Answers will be logged to {run_storage_folder}")
        answers_file_jsonl = Path(run_storage_folder) / "answers.jsonl"
        if not os.path.exists(answers_file_jsonl):
            os.makedirs(run_storage_folder, exist_ok=True)
            with open(answers_file_jsonl, "w", encoding="utf-8"):
                pass
    else:
        answers_file_jsonl = None  # Don't write output

    total_len = len(dataset_to_test)
    sample_iter = iter(dataset_to_test)
    submitted: list[Any] = []

    def submit_next():
        try:
            sample = next(sample_iter)
        except StopIteration:
            return False
        submitted.append(
            exe.submit(
                process_sample,
                sample,
                model,
                input_annotation_transform,
                evaluation_config,
                answers_file_jsonl,
                **generation_kwargs,
            )
        )
        return True

    with ThreadPoolExecutor(max_workers=evaluation_config.parallel_workers) as exe:
        for _ in range(min(chunk_size, total_len)):
            submit_next()
        with tqdm(total=total_len, desc=f"Processing {benchmark_name}") as pbar:
            while submitted:
                done, remaining = wait(submitted, return_when=FIRST_COMPLETED)
                submitted = list(remaining)
                for f in done:
                    try:
                        result = f.result(timeout=evaluation_config.timeout)
                        responses.extend(result)
                    except Exception as e:
                        logger.error(f"Error processing sample: {e}")
                        responses.append(None)
                    finally:
                        pbar.update(1)
                while len(submitted) < chunk_size:
                    if not submit_next():
                        break

    return responses  # Make sure to return all responses
