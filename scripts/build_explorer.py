#!/usr/bin/env python3
"""Build the interactive explorer HTML by splicing data/explorer.json into the
shipped template.

Run ``python analysis/explorer_data.py`` first to (re)generate
``data/explorer.json``; this script inlines that JSON into
``analysis/nudge_explorer_template.html`` and writes a single self-contained
page to ``$MHYP_FIGDIR/nudge_explorer.html``.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "analysis" / "nudge_explorer_template.html"
DATA = ROOT / "data" / "explorer.json"
OUT = Path(os.environ.get("MHYP_FIGDIR", "figures"))


def main() -> None:
    if not DATA.exists():
        raise SystemExit("data/explorer.json not found -- run "
                         "`python analysis/explorer_data.py` first.")
    html = TEMPLATE.read_text()
    data = DATA.read_text()
    if "__DATA__" not in html:
        raise SystemExit("template has no __DATA__ placeholder")
    if "</script" in data.lower():
        raise SystemExit("data contains a </script> token; cannot inline safely")
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "nudge_explorer.html"
    out.write_text(html.replace("__DATA__", data))
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
