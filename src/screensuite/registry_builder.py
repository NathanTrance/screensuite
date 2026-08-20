"""
Centralized Benchmark Registry

This file contains all benchmark instantiations collected from individual benchmark.py files.
All benchmarks are instantiated here with their original constructor calls and then registered.
"""

# Multistep benchmarks require the optional `osworld` / `android_world` packages (and Docker/KVM).
# They are imported lazily so that offline-only installs (without those submodules) still work.
try:
    from screensuite.benchmarks.multistep.android_world.benchmark import (
        AndroidWorldBenchmark,
    )
    from screensuite.benchmarks.multistep.android_world.config import AndroidWorldConfig
except ImportError:
    AndroidWorldBenchmark = None
    AndroidWorldConfig = None
try:
    from screensuite.benchmarks.multistep.browse_comp.benchmark import BrowseCompBenchmark
    from screensuite.benchmarks.multistep.browse_comp.config import BrowseCompConfig
except ImportError:
    BrowseCompBenchmark = None
    BrowseCompConfig = None
try:
    from screensuite.benchmarks.multistep.gaia.benchmark import GaiaBenchmark
    from screensuite.benchmarks.multistep.gaia.config import GaiaConfig
except ImportError:
    GaiaBenchmark = None
    GaiaConfig = None
try:
    from screensuite.benchmarks.multistep.mind2web.benchmark import Mind2WebBenchmark
    from screensuite.benchmarks.multistep.mind2web.config import Mind2WebConfig
except ImportError:
    Mind2WebBenchmark = None
    Mind2WebConfig = None
try:
    from screensuite.benchmarks.multistep.osworld.benchmark import OSWorldBenchmark
    from screensuite.benchmarks.multistep.osworld.config import OSWorldConfig
except ImportError:
    OSWorldBenchmark = None
    OSWorldConfig = None
from screensuite.benchmarks.perception.screenqa.benchmark import ScreenQABenchmark
from screensuite.benchmarks.perception.screenqa.config import ScreenQaConfig
from screensuite.benchmarks.perception.screenspot.benchmark import ScreenSpotBenchmark
from screensuite.benchmarks.perception.screenspot.config import (
    LocalizationPrompt,
    ScreenSpotConfig,
)
from screensuite.benchmarks.perception.visualwebbench.benchmark import (
    VisualWebBenchBenchmark,
)
from screensuite.benchmarks.perception.visualwebbench.config import VisualWebBenchConfig
from screensuite.benchmarks.perception.websrc.benchmark import WebSrcBenchmark
from screensuite.benchmarks.perception.websrc.config import WebSrcConfig
from screensuite.benchmarks.singlestep.androidcontrol.benchmark import (
    AndroidControlBenchmark,
)
from screensuite.benchmarks.singlestep.androidcontrol.config import AndroidControlConfig
from screensuite.benchmarks.singlestep.mmind2web.benchmark import MMind2WebBenchmark
from screensuite.benchmarks.singlestep.mmind2web.config import MMind2WebConfig
from screensuite.benchmarks.singlestep.showdown_clicks.benchmark import (
    ShowdownClicksBenchmark,
)
from screensuite.benchmarks.singlestep.showdown_clicks.config import (
    ShowdownClicksConfig,
)
from screensuite.registry import BenchmarkRegistry


def get_registry() -> BenchmarkRegistry:
    registry = BenchmarkRegistry()

    # =============================================================================
    # MULTISTEP BENCHMARKS
    # =============================================================================

    # GAIA Benchmark
    gaia_web = None
    if GaiaBenchmark is not None:
        gaia_web = GaiaBenchmark(
            name="gaia_web",
            config=GaiaConfig(),
            tags=["gaia", "multistep", "hf_dataset", "web", "online", "to_evaluate"],
        )

    # Mind2Web Benchmark
    mind2web_live = None
    if Mind2WebBenchmark is not None:
        mind2web_live = Mind2WebBenchmark(
            name="mind2web_live",
            config=Mind2WebConfig(),
            tags=["mind2web", "multistep", "hf_dataset", "web", "online", "to_evaluate"],
        )

    # Browse Comp Benchmark
    browse_comp = None
    if BrowseCompBenchmark is not None:
        browse_comp = BrowseCompBenchmark(
            name="browse_comp",
            config=BrowseCompConfig(),
            tags=["browse_comp", "multistep", "hf_dataset", "web", "online", "to_evaluate"],
        )

    # OSWorld Benchmark
    osworld_benchmark = None
    if OSWorldBenchmark is not None:
        osworld_benchmark = OSWorldBenchmark(
            name="osworld",
            config=OSWorldConfig(),
            tags=["osworld", "multistep", "online", "os", "web", "to_evaluate"],
        )

    # Android World Benchmark
    android_world = None
    if AndroidWorldBenchmark is not None:
        android_world = AndroidWorldBenchmark(
            name="android_world",
            config=AndroidWorldConfig(),
            tags=["android_world", "multistep", "hf_dataset", "online", "mobile", "android", "to_evaluate"],
        )

    # =============================================================================
    # SINGLESTEP BENCHMARKS
    # =============================================================================

    # MMind2Web Benchmark
    mmind2web = MMind2WebBenchmark(
        name="mmind2web",
        config=MMind2WebConfig(),
        tags=["mmind2web", "singlestep", "hf_dataset", "offline", "web", "to_evaluate"],
    )

    # Android Control Benchmark
    android_control = AndroidControlBenchmark(
        name="android_control",
        config=AndroidControlConfig(),
        tags=["android_control", "singlestep", "hf_dataset", "offline", "mobile", "android", "to_evaluate"],
    )

    # Showdown Clicks Benchmark
    showdown_clicks = ShowdownClicksBenchmark(
        name="showdown_clicks",
        config=ShowdownClicksConfig(),
        tags=["showdown_clicks", "singlestep", "hf_dataset", "offline", "web", "to_evaluate"],
    )

    # =============================================================================
    # PERCEPTION BENCHMARKS
    # =============================================================================

    # ScreenQA Benchmarks
    screenqa_short = ScreenQABenchmark(
        name="screenqa_short",
        config=ScreenQaConfig.short(),
        tags=["screenqa", "hf_dataset", "webqa", "short", "mobile", "to_evaluate"],
    )

    screenqa_complex = ScreenQABenchmark(
        name="screenqa_complex",
        config=ScreenQaConfig.complex(),
        tags=["screenqa", "hf_dataset", "webqa", "complex", "mobile", "to_evaluate"],
    )

    # 500-sample fast subsets (pushed to nathantrance/*) - select explicitly with --benchmarks
    screenqa_short_500 = ScreenQABenchmark(
        name="screenqa_short_500",
        config=ScreenQaConfig.short_500(),
        tags=["screenqa", "hf_dataset", "webqa", "short", "mobile", "small"],
    )

    screenqa_complex_500 = ScreenQABenchmark(
        name="screenqa_complex_500",
        config=ScreenQaConfig.complex_500(),
        tags=["screenqa", "hf_dataset", "webqa", "complex", "mobile", "small"],
    )

    # WebSrc Benchmark
    websrc_dev = WebSrcBenchmark(
        name="websrc_dev",
        config=WebSrcConfig.dev(),
        tags=["websrc", "hf_dataset", "webqa", "dev", "web", "to_evaluate"],
    )

    # ScreenSpot Benchmarks
    screenspot_v1_click_prompt = ScreenSpotBenchmark(
        name="screenspot-v1-click-prompt",
        config=ScreenSpotConfig.v1(LocalizationPrompt.CLICK_PROMPT_ABSOLUTE),
        tags=["screenspot", "grounding", "hf_dataset", "v1", "click"],
    )

    screenspot_v1_bounding_box_prompt = ScreenSpotBenchmark(
        name="screenspot-v1-bounding-box-prompt",
        config=ScreenSpotConfig.v1(LocalizationPrompt.BOUNDING_BOX_PROMPT),
        tags=["screenspot", "grounding", "hf_dataset", "v1", "bounding_box"],
    )

    screenspot_v2_click_prompt = ScreenSpotBenchmark(
        name="screenspot-v2-click-prompt",
        config=ScreenSpotConfig.v2(LocalizationPrompt.CLICK_PROMPT_ABSOLUTE),
        tags=["screenspot", "grounding", "hf_dataset", "v2", "click", "to_evaluate"],
    )

    screenspot_v2_bounding_box_prompt = ScreenSpotBenchmark(
        name="screenspot-v2-bounding-box-prompt",
        config=ScreenSpotConfig.v2(LocalizationPrompt.BOUNDING_BOX_PROMPT),
        tags=["screenspot", "grounding", "hf_dataset", "v2", "bounding_box"],
    )

    screenspot_pro_click_prompt = ScreenSpotBenchmark(
        name="screenspot-pro-click-prompt",
        config=ScreenSpotConfig.pro(LocalizationPrompt.CLICK_PROMPT_ABSOLUTE),
        tags=["screenspot", "grounding", "hf_dataset", "pro", "click", "to_evaluate"],
    )

    screenspot_pro_bounding_box_prompt = ScreenSpotBenchmark(
        name="screenspot-pro-bounding-box-prompt",
        config=ScreenSpotConfig.pro(LocalizationPrompt.BOUNDING_BOX_PROMPT),
        tags=["screenspot", "grounding", "hf_dataset", "pro", "bounding_box"],
    )

    # VisualWebBench Benchmark
    visualwebbench = VisualWebBenchBenchmark(
        name="visualwebbench",
        config=VisualWebBenchConfig(),
        tags=["visualwebbench", "hf_dataset", "webqa", "vision", "web", "to_evaluate"],
    )

    # =============================================================================
    # REGISTRY REGISTRATION
    # =============================================================================

    # Register all multistep benchmarks (only those whose optional packages are installed)
    for b in (gaia_web, mind2web_live, browse_comp, osworld_benchmark, android_world):
        if b is not None:
            registry.register(b)

    # Register all singlestep benchmarks
    registry.register(mmind2web)
    registry.register(android_control)
    registry.register(showdown_clicks)

    # Register all perception benchmarks
    registry.register([screenqa_short, screenqa_complex])
    registry.register([screenqa_short_500, screenqa_complex_500])
    registry.register(websrc_dev)
    registry.register(
        [
            screenspot_v1_click_prompt,
            screenspot_v1_bounding_box_prompt,
            screenspot_v2_click_prompt,
            screenspot_v2_bounding_box_prompt,
            screenspot_pro_click_prompt,
            screenspot_pro_bounding_box_prompt,
        ]
    )
    registry.register(visualwebbench)

    return registry
