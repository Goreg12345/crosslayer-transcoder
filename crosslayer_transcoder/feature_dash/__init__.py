"""Feature dashboard for MoLT checkpoints.

Inspired by sae_vis (https://github.com/callummcdougall/sae_vis), trimmed to a
minimal three-panel view: per-transform activation rate, max-activating
examples, and token-level highlighting within those examples.

The corpus used for the qualitative pass is OpenWebText
(`Skylion007/openwebtext`) regardless of what the checkpoint was trained on.
"""

from crosslayer_transcoder.feature_dash.collect import (
    BaseLMRunner,
    FeatureSummary,
    GateCollector,
    collect_features,
    window_example,
    window_feature_summary,
)
from crosslayer_transcoder.feature_dash.bundle import (
    default_bundle_filename,
    make_bundle,
    make_bundle_from_disk,
)
from crosslayer_transcoder.feature_dash.dump import dump_dashboard
from crosslayer_transcoder.feature_dash.load import (
    DEFAULT_HF_REPO,
    MoltCheckpointMetadata,
    infer_molt_arch,
    load_molt,
    load_molt_from_hf,
)
from crosslayer_transcoder.feature_dash.render import copy_render_assets
from crosslayer_transcoder.feature_dash.multilayer import (
    MultiLayerCheckpointMetadata,
    MultiLayerLMRunner,
    collect_multilayer_features,
    find_latest_step,
    load_multilayer_molt,
)
from crosslayer_transcoder.feature_dash.multilayer_bundle import (
    PromptTrace,
    make_multilayer_bundle,
)

__all__ = [
    "BaseLMRunner",
    "DEFAULT_HF_REPO",
    "FeatureSummary",
    "GateCollector",
    "MoltCheckpointMetadata",
    "MultiLayerCheckpointMetadata",
    "MultiLayerLMRunner",
    "PromptTrace",
    "collect_features",
    "collect_multilayer_features",
    "copy_render_assets",
    "default_bundle_filename",
    "dump_dashboard",
    "find_latest_step",
    "infer_molt_arch",
    "load_molt",
    "load_molt_from_hf",
    "load_multilayer_molt",
    "make_bundle",
    "make_bundle_from_disk",
    "make_multilayer_bundle",
    "window_example",
    "window_feature_summary",
]
