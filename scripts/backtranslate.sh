#!/usr/bin/env sh
# Backtranslate Mooré monolingual sentences (madoss/moore-web-mono v1.1.0) into
# French. Pass --model <Mooré -> French model or CT2 dir>; on Modal the launcher
# adds --output_dir_root /outputs/.
RUNNER=""
if [ "${USE_UV:-0}" = "1" ]; then
    RUNNER="uv run"
fi
${RUNNER} python -m mt_training.backtranslate "$@"
