#!/bin/bash
# Build the Zenodo archive of trained STEPHY models.
#
# The models are too large for git but are needed by supportingFigures/ to
# regenerate the paper figures. Unpack the archive anywhere and point
# STEPHY_MODELS at the resulting directory (see supportingFigures/_paths.py).
#
# Usage:
#   zenodo/make_archive.sh <path-to-trained_model> [output.tar.zst]
#
# Deliberately excluded, and why:
#   */logs/                 SLURM scheduler output; provenance noise, not results
#   denmark/*/pe            superseded positional-encoding run; the paper uses pe_old
#   denmark/*/pe_old/CBLV-GAT   ablation not reported for the Denmark application
#   *.prebugfix.bak         pre-bugfix backups; publishing them invites confusion
#                           about which artefacts produced the reported numbers

set -euo pipefail

SRC="${1:?Usage: $0 <path-to-trained_model> [output.tar.zst]}"
OUT="${2:-stephy-trained-models.tar.zst}"

[ -d "$SRC" ] || { echo "No such directory: $SRC" >&2; exit 1; }

INCLUDE=()

# Simulation benchmarks: every setting, every pipeline, minus logs.
for setting in "$SRC"/simu/*/; do
    [ -d "$setting" ] || continue
    name=$(basename "$setting")
    for pipeline in "$setting"*/; do
        p=$(basename "$pipeline")
        [ "$p" = "logs" ] && continue
        INCLUDE+=("simu/$name/$p")
    done
done

# Denmark application: pe_old only, and only the head plus the CNN ablation.
for p in stephy CBLV-CNN; do
    [ -d "$SRC/denmark/100k_result/pe_old/$p" ] \
        && INCLUDE+=("denmark/100k_result/pe_old/$p")
done

if [ ${#INCLUDE[@]} -eq 0 ]; then
    echo "Nothing to archive under $SRC" >&2
    exit 1
fi

echo "Including:"
printf '  %s\n' "${INCLUDE[@]}"

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
ROOT="$STAGE/stephy-trained-models"
mkdir -p "$ROOT"

for path in "${INCLUDE[@]}"; do
    mkdir -p "$ROOT/$(dirname "$path")"
    cp -R "$SRC/$path" "$ROOT/$path"
done

# Drop pre-bugfix backups wherever they were copied in.
find "$ROOT" -name '*.prebugfix.bak' -delete

cp "$(dirname "$0")/README.md" "$ROOT/README.md"

tar --use-compress-program='zstd -19 -T0' -cf "$OUT" -C "$STAGE" stephy-trained-models

echo
echo "Wrote $OUT ($(du -h "$OUT" | cut -f1))"
echo "Contents: $(find "$ROOT" -type f | wc -l | tr -d ' ') files, $(du -sh "$ROOT" | cut -f1) uncompressed"
echo
echo "Confirm nothing excluded slipped in (CBLV-GAT is legitimate under simu/,"
echo "so the check is anchored to the Denmark path):"
echo "  tar --use-compress-program=unzstd -tf $OUT \\"
echo "    | grep -E '/logs/|denmark/100k_result/pe/|pe_old/CBLV-GAT/|\\.prebugfix\\.bak' \\"
echo "    || echo '  clean'"
