"""Copy the static dashboard templates into a dump directory.

The templates are bundled as package data under
`crosslayer_transcoder/feature_dash/templates/`. We copy them rather than
import them as strings so the output is editable / easy to inspect.

After `dump_dashboard` writes `data/`, calling `copy_render_assets(out_dir)`
puts:

    <out_dir>/index.html
    <out_dir>/dashboard.html
    <out_dir>/assets/dashboard.css
    <out_dir>/assets/dashboard.js
    <out_dir>/assets/index.js

next to it. Open `index.html` via a local HTTP server (file:// blocks fetch
in most browsers).
"""

from __future__ import annotations

import shutil
from pathlib import Path

TEMPLATE_DIR = Path(__file__).parent / "templates"

# (source filename, destination relative to out_dir)
_ASSET_MAP: list[tuple[str, str]] = [
    ("index.html", "index.html"),
    ("dashboard.html", "dashboard.html"),
    ("dashboard.css", "assets/dashboard.css"),
    ("dashboard.js", "assets/dashboard.js"),
    ("index.js", "assets/index.js"),
]


def copy_render_assets(out_dir: str | Path) -> Path:
    """Copy the HTML/CSS/JS templates into `out_dir`. Returns out_dir."""
    out_dir = Path(out_dir)
    (out_dir / "assets").mkdir(parents=True, exist_ok=True)
    for src_name, dst_rel in _ASSET_MAP:
        src = TEMPLATE_DIR / src_name
        dst = out_dir / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    return out_dir
