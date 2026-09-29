#!/usr/bin/env sh
# Evaluate one model on the standard sets, with the settings used in
# .agents/logbooks/experiments.md: FLORES+ devtest, Bouquet test (sentence and
# paragraph level) and the moore-web-parallel test split, references
# normalized for the NLLB vocabulary (--normalize_references). Predictions go
# to OUT_DIR/NAME-{flores,bq-sent,bq-para,test}.csv for scripts/compare_suite.sh.
#
#   ./scripts/eval_suite.sh <model or CT2 dir> <name> [out_dir]
#   SRC_LANG=mos_Latn TGT_LANG=fra_Latn ./scripts/eval_suite.sh <mos-fra model> mf-v2
#
# Extra options for every eval go in EVAL_ARGS (e.g. EVAL_ARGS="--dtype bfloat16").
set -eu
if [ $# -lt 2 ]; then
    echo "Usage: $0 <model> <name> [out_dir]" >&2
    exit 1
fi
MODEL=$1
NAME=$2
OUT=${3:-evaluations}
SRC_LANG=${SRC_LANG:-fra_Latn}
TGT_LANG=${TGT_LANG:-mos_Latn}
if [ "$SRC_LANG" = mos_Latn ]; then
    SRC_FIELD=moore REF_FIELD=french
else
    SRC_FIELD=french REF_FIELD=moore
fi
RUNNER=""
if [ "${USE_UV:-1}" = "1" ]; then
    RUNNER="uv run"
fi
mkdir -p "$OUT"

ev() { # set dataset [options...]
    set_name=$1
    dataset=$2
    shift 2
    printf '%-28s ' "$NAME-$set_name"
    # shellcheck disable=SC2086
    ${RUNNER} python -m mt_training.eval --model "$MODEL" \
        --src_lang "$SRC_LANG" --tgt_lang "$TGT_LANG" --dataset "$dataset" \
        --normalize_references true --show_samples 0 \
        --output "$OUT/$NAME-$set_name.csv" "$@" ${EVAL_ARGS:-} 2>&1 |
        grep -E "BLEU|chrF" | tr -s ' ' | tr '\n' ' '
    echo
}

ev flores openlanguagedata/flores_plus
ev bq-sent facebook/bouquet
ev bq-para facebook/bouquet --bouquet_level paragraph_level --max_new_tokens 384
ev test madoss/moore-web-parallel --split test \
    --src_field "$SRC_FIELD" --ref_field "$REF_FIELD" --max_new_tokens 256
