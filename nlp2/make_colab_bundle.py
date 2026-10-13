"""Package what E2 needs for Google Colab into one small zip.

Usage:
    python -m nlp2.make_colab_bundle
Writes outputs/e2_colab_bundle.zip containing nlp2/, requirements.txt and
outputs/processed/conditions/*.csv (no .venv, no raw data). Upload it with
notebooks/e2_colab.ipynb. Re-run after the normaliser is tuned so the bundle
carries the final hist_xh_norm condition.
"""

from __future__ import annotations

import zipfile

from . import config
from .utils import get_logger

log = get_logger("bundle")


def main() -> None:
    out = config.OUT_DIR / "e2_colab_bundle.zip"
    files = sorted((config.ROOT / "nlp2").glob("*.py")) + [config.ROOT / "requirements.txt"]
    files += sorted(config.CONDITIONS_DIR.glob("*.csv"))
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.relative_to(config.ROOT) if f.is_relative_to(config.ROOT) else f"outputs/processed/conditions/{f.name}")
    log.info("Wrote %s (%d files, %.1f MB)", out, len(files), out.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
