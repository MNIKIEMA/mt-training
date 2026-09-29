#!/usr/bin/env sh
# Mooré -> French, with exactly the hyperparameters of train.sh (the options
# below override train.sh's; extra arguments are passed through as well).
# Used to backtranslate Mooré monolingual text into French.
exec sh "$(dirname "$0")/train.sh" \
    --source_field moore \
    --target_field french \
    --src_lang mos_Latn \
    --tgt_lang fra_Latn \
    --run_name moore-web-parallel-v1.1.0-mos-fra-bf16 \
    --repo_name nllb-600m-MosFr-mwp-v1 \
    "$@"
