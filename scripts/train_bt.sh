#!/usr/bin/env sh
# French -> Mooré on moore-web-parallel v1.1.0 + the Mooré monolingual text
# backtranslated by nllb-600m-MosFr-mwp-v2, with exactly the hyperparameters
# of train.sh (the options below override train.sh's; extra arguments are
# passed through as well). The dataset is built by `mt-training mix-bt`; see
# .agents/logbooks/synthetic-data.md.
exec sh "$(dirname "$0")/train.sh" \
    --dataset_id madoss/moore-web-parallel-bt \
    --dataset_config mos-fra \
    --dataset_revision 26afc315ad96d70c2d68f203861139933e79c1f7 \
    --run_name moore-web-parallel-v1.1.0-bt-mosfr-v2-fra-mos-bf16 \
    --repo_name nllb-600m-FrMos-mwp-bt-v1 \
    "$@"
