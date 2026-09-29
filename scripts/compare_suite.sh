#!/usr/bin/env sh
# Compare two models evaluated by scripts/eval_suite.sh into the same directory:
# the "All" row of `mt-training compare` per set (chrF++ gain with its paired
# bootstrap 95% CI, BLEU, length ratio, long outputs, repeated words), and a
# JSON file per set with the length buckets.
#
#   ./scripts/compare_suite.sh <baseline name> <candidate name> [dir]
set -eu
if [ $# -lt 2 ]; then
    echo "Usage: $0 <baseline name> <candidate name> [dir]" >&2
    exit 1
fi
BASE=$1
CAND=$2
DIR=${3:-evaluations}
RUNNER=""
if [ "${USE_UV:-1}" = "1" ]; then
    RUNNER="uv run"
fi
for set_name in flores bq-sent bq-para test; do
    if [ ! -f "$DIR/$BASE-$set_name.csv" ] || [ ! -f "$DIR/$CAND-$set_name.csv" ]; then
        echo "== $set_name: missing $DIR/$BASE-$set_name.csv or $DIR/$CAND-$set_name.csv, skipped"
        continue
    fi
    echo "== $set_name: $CAND vs $BASE"
    ${RUNNER} python -m mt_training.compare \
        --baseline "$DIR/$BASE-$set_name.csv" --candidate "$DIR/$CAND-$set_name.csv" \
        --output "$DIR/compare-$CAND-vs-$BASE-$set_name.json" 2>/dev/null |
        grep -E "^bucket|^All"
done
