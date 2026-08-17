#!/usr/bin/env bash
# Build the release data archive from this repo's data/ directory.
#
# Run after data/ is populated -- either by `make experiments` (regenerates
# everything on a GPU) or by copying your existing per-cell results into
# data/cells/ (+ data/transfer*/ and the root reasoning/API result JSONs).
# Upload the resulting tarball to your release / Zenodo and point
# scripts/download_data.py at it (set MHYP_DATA_URL to its URL).
#
#   bash scripts/make_data_archive.sh [output.tar.gz]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-mhyp-data.tar.gz}"

paths=()
[ -d data/cells ] && paths+=(data/cells)
for d in data/transfer_found data/transfer data/transfer_ip data/story100; do
  [ -d "$d" ] && paths+=("$d")
done
# root-level derived results (Fig 9/11 reasoning + API cells, etc.); skip the
# regenerable explorer.json and any _scratch files.
for f in data/*.json data/*.tsv; do
  [ -e "$f" ] || continue
  b="$(basename "$f")"
  case "$b" in _*|explorer.json) continue ;; esac
  paths+=("$f")
done

if [ "${#paths[@]}" -eq 0 ]; then
  echo "data/ is empty -- run 'make experiments' or copy results in first." >&2
  exit 1
fi

echo "archiving ${#paths[@]} path(s) -> $OUT"
tar czf "$OUT" --exclude='__pycache__' --exclude='*.pyc' "${paths[@]}"
echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
