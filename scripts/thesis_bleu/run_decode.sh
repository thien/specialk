#!/usr/bin/env bash
# Decode one sample of the thesis BLEU test set.
#
# Usage:
#   scripts/thesis_bleu/run_decode.sh NAME ENCODER DECODER VOCAB SRC OUT [LIMIT]
#
# The model/vocabulary paths point into the main checkout because the large
# assets are not duplicated into git worktrees.

set -euo pipefail

NAME=$1
ENCODER=$2
DECODER=$3
VOCAB=$4
SRC=$5
OUT=$6
LIMIT=${7:-}

PYTHON=${PYTHON:-/Users/t/projects/specialk/.venv/bin/python}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

LIMIT_ARGS=()
if [[ -n "$LIMIT" ]]; then
    LIMIT_ARGS=(--limit "$LIMIT")
fi

mkdir -p "$(dirname "$OUT")"
echo "### $NAME -> $OUT" | tee "${OUT}.log"
/usr/bin/time -p "$PYTHON" "$ROOT/scripts/thesis_bleu/decode.py" \
    --encoder "$ENCODER" \
    --decoder "$DECODER" \
    --vocab "$VOCAB" \
    --src "$SRC" \
    --out "$OUT" \
    --beam 5 \
    --batch-size 128 \
    "${LIMIT_ARGS[@]}" 2>&1 | tee -a "${OUT}.log"
