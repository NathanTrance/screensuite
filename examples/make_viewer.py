#!/usr/bin/env python
"""Build a local HTML viewer for a ScreenSuite benchmark dataset.

Pulls the whole (subset) dataset, and for every sample generates:
  - the exact prompt text sent to the model
  - the raw image + the resized image (what the model actually sees)
  - question / ground_truth / metadata

Output:
  output/viewer/<benchmark>/viewer.html   (open in a browser)
  output/viewer/<benchmark>/images/*.png

Usage:
  uv run python examples/make_viewer.py --benchmarks screenqa_complex_500 screenqa_short_500
"""
import argparse
import json
import math
import os

from datasets import load_dataset

from screensuite import EvaluationConfig, ImageResizeConfig, get_registry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", nargs="+", default=["screenqa_complex_500"])
    parser.add_argument("--out-dir", default="output/viewer")
    args = parser.parse_args()

    registry = get_registry()
    evaluation_config = EvaluationConfig(
        test_mode=False,
        parallel_workers=1,
        max_samples_to_test=None,
        run_name=None,
        image_resize_config=ImageResizeConfig(),
    )

    for bench_name in args.benchmarks:
        bench = next(b for b in registry.list_all() if b.name == bench_name)
        print(f"=== {bench_name}: pulling {bench.config.hf_repo} ({bench.config.split}) ===")
        stream = load_dataset(bench.config.hf_repo, split=bench.config.split, streaming=True)

        out_dir = os.path.join(args.out_dir, bench_name)
        img_dir = os.path.join(out_dir, "images")
        os.makedirs(img_dir, exist_ok=True)

        records = []
        for row in stream:
            sid = row.get("screen_id", row.get("episode_id", row.get("file_name")))
            annotated = bench._get_annotated_input_from_sample(row, evaluation_config)

            prompt_text = ""
            raw_img = None
            resized_img = None
            for msg in annotated.messages:
                for content in msg["content"]:
                    if content["type"] == "text":
                        prompt_text = content["text"]
                    elif content["type"] == "image":
                        resized_img = content["image"]

            raw_img = row.get("image")
            if raw_img is None:
                raw_img = row.get("screenshot")

            img_path = os.path.join(img_dir, f"{sid}.jpg")
            raw_img.convert("RGB").save(img_path, "JPEG", quality=90)
            resized_path = None
            if resized_img is not None:
                resized_path = f"images/{sid}_resized.jpg"
                resized_img.convert("RGB").save(os.path.join(img_dir, f"{sid}_resized.jpg"), "JPEG", quality=90)

            est_img_tokens = None
            if resized_img is not None:
                w, h = resized_img.size
                est_img_tokens = math.ceil(w / 28) * math.ceil(h / 28) * 256

            records.append({
                "sample_id": str(sid),
                "question": row.get("question", ""),
                "ground_truth": row.get("ground_truth", []),
                "file_name": row.get("file_name", row.get("goal", "")),
                "raw_dims": list(raw_img.size),
                "resized_dims": list(resized_img.size) if resized_img else None,
                "image": f"images/{sid}.png",
                "resized_image": resized_path,
                "prompt": prompt_text,
                "text_chars": len(prompt_text),
                "est_img_tokens": est_img_tokens,
            })

        records.sort(key=lambda r: r["sample_id"])
        html = render_html(bench_name, records)
        with open(os.path.join(out_dir, "viewer.html"), "w", encoding="utf-8") as f:
            f.write(html)
        print(f"  {len(records)} samples -> {os.path.join(out_dir, 'viewer.html')}")


def render_html(bench_name: str, records: list[dict]) -> str:
    data = {r["sample_id"]: r for r in records}
    data_json = json.dumps(data, ensure_ascii=False)
    ids = [r["sample_id"] for r in records]
    ids_json = json.dumps(ids, ensure_ascii=False)

    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__BENCH__ viewer</title>
<style>
  body { font-family: ui-monospace, Consolas, monospace; background:#0f1117; color:#d7dae0; margin:0; }
  header { padding:16px 24px; background:#161a23; border-bottom:1px solid #2a2f3a; }
  h1 { font-size:18px; margin:0 0 8px; }
  input { padding:8px; font-size:14px; background:#0f1117; color:#d7dae0; border:1px solid #2a2f3a; border-radius:6px; width:260px; }
  main { padding:24px; max-width:900px; }
  .card { background:#161a23; border:1px solid #2a2f3a; border-radius:10px; padding:16px 20px; margin-top:16px; }
  .row { display:flex; gap:24px; flex-wrap:wrap; }
  .images img { max-width:340px; border:1px solid #2a2f3a; border-radius:8px; }
  .label { color:#8b93a5; font-size:12px; text-transform:uppercase; letter-spacing:.05em; margin:12px 0 4px; }
  pre { background:#0f1117; border:1px solid #2a2f3a; border-radius:8px; padding:12px; white-space:pre-wrap; }
  button { padding:8px 14px; margin-top:12px; background:#2563eb; color:#fff; border:none; border-radius:6px; cursor:pointer; }
  #count { color:#8b93a5; font-size:12px; margin-top:8px; }
</style>
</head>
<body>
<header>
  <h1>__BENCH__ — sample viewer</h1>
  <input id="search" type="text" placeholder="Type a sample id and press Enter..." autocomplete="off">
  <button onclick="showPrev()">&#9664; Prev</button>
  <button onclick="showNext()">Next &#9654;</button>
  <div id="count"></div>
</header>
<main id="view"></main>
<script>
const DATA = __DATA__;
const IDS = __IDS__;
let current = IDS[0];

function esc(s) { return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

function show(id) {
  if (!(id in DATA)) { document.getElementById('view').innerHTML = '<div class="card">not found</div>'; return; }
  const r = DATA[id];
  document.getElementById('count').textContent = id + ' — ' + (IDS.indexOf(id)+1) + ' / ' + IDS.length;
  document.getElementById('search').value = id;
  current = id;
  const gt = Array.isArray(r.ground_truth) ? r.ground_truth.join(' | ') : r.ground_truth;
  document.getElementById('view').innerHTML =
    '<div class="card">' +
      '<div class="row">' +
        '<div class="images"><div class="label">raw image ' + (r.raw_dims ? r.raw_dims.join('x') : '') + '</div>' +
          "<img src='" + r.image + "'></div>" +
        '<div class="images"><div class="label">resized (what the model sees) ' + (r.resized_dims ? r.resized_dims.join('x') : '') + '</div>' +
          "<img src='" + r.resized_image + "'></div>" +
      '</div>' +
      '<div class="label">screen_id / file</div><div>' + esc(r.sample_id) + ' — ' + esc(r.file_name) + '</div>' +
      '<div class="label">question</div><div>' + esc(r.question) + '</div>' +
      '<div class="label">ground truth</div><div>' + esc(gt) + '</div>' +
      '<div class="label">metadata</div><div>text_chars=' + r.text_chars + ', est_img_tokens=' + r.est_img_tokens + '</div>' +
      '<div class="label">exact prompt sent to the model</div><pre>' + esc(r.prompt) + '</pre>' +
    '</div>';
}

function showPrev() { const i = IDS.indexOf(current); show(IDS[(i-1+IDS.length)%IDS.length]); }
function showNext() { const i = IDS.indexOf(current); show(IDS[(i+1)%IDS.length]); }

document.getElementById('search').addEventListener('keydown', e => {
  if (e.key === 'Enter') show(e.target.value.trim());
});
show(current);
</script>
</body>
</html>"""
    return (
        html.replace("__BENCH__", bench_name)
        .replace("__DATA__", data_json)
        .replace("__IDS__", ids_json)
    )


if __name__ == "__main__":
    main()
