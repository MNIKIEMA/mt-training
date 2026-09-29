"""Run a training script from scripts/ on a Modal GPU.

The container runs the same shell script as RunPod, with outputs redirected to
a Modal Volume so checkpoints, the final model and the CTranslate2 model
survive the container.

One-time setup (from your own terminal):

    uvx modal setup

Secrets: the workspace-wide `huggingface-secret` (HF_TOKEN) and `wandb-secret`
(WANDB_API_KEY) are attached to the function. Modal injects only the secrets a
function lists, so a new secret must be added to SECRETS below.

Run from the repository root:

    # smoke test (small, ~minutes)
    uvx modal run scripts/modal_train.py --script debug.sh

    # full run: --detach plus --no-wait. The command returns at once and nothing
    # in this terminal can cancel the run. (Waiting with --detach alone is not
    # enough: Ctrl+C in the waiting terminal cancels the call.)
    uvx modal run --detach scripts/modal_train.py --script train.sh --no-wait

    # resume after a timeout or failure, continuing the same W&B run
    uvx modal run --detach scripts/modal_train.py --script train.sh --no-wait \
        --extra-args "--resume_from_checkpoint /outputs/nllb-600m-FrMos/checkpoint-1234" \
        --wandb-run-id <id from the W&B run URL>

    # fetch results
    uvx modal volume ls mt-training-outputs
    uvx modal volume get mt-training-outputs nllb-600m-FrMos-ct2 ./nllb-600m-FrMos-ct2

    # FLORES+ devtest for any Hub model (no training), as the post-training eval
    uvx modal run scripts/modal_train.py --flores-model facebook/nllb-200-distilled-600M
    uvx modal run scripts/modal_train.py --flores-model <mos-fra model> --src-lang mos_Latn --tgt-lang fra_Latn

GPU type: set MODAL_GPU when launching (default A100-80GB), e.g.
MODAL_GPU=A100-40GB uvx modal run ...
"""

import os
import shlex
import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/root/mt-training"
OUTPUTS = "/outputs"
CACHE = "/cache"

image = (
    modal.Image.debian_slim(python_version="3.12")
    # Dependencies from uv.lock (frozen); the project itself is added below so code
    # changes don't rebuild the dependency layer.
    .uv_sync(uv_project_dir=str(ROOT))
    .env(
        {
            "HF_HOME": f"{CACHE}/huggingface",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "PYTHONPATH": f"{REMOTE_ROOT}/src",
        }
    )
    .add_local_dir(ROOT / "src", f"{REMOTE_ROOT}/src")
    .add_local_dir(ROOT / "scripts", f"{REMOTE_ROOT}/scripts", ignore=["*.py"])
)

app = modal.App("mt-training", image=image)
outputs = modal.Volume.from_name("mt-training-outputs", create_if_missing=True)
cache = modal.Volume.from_name("mt-training-hf-cache", create_if_missing=True)
SECRETS = [
    modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"]),
    modal.Secret.from_name("wandb-secret", required_keys=["WANDB_API_KEY"]),
]


@app.function(
    gpu=os.environ.get("MODAL_GPU", "A100-80GB"),
    volumes={OUTPUTS: outputs, CACHE: cache},
    secrets=SECRETS,
    timeout=24 * 60 * 60,
    # A retry would restart training from scratch; resume from a checkpoint instead.
    retries=0,
)
def train(script: str, extra_args: str = "", wandb_run_id: str = "") -> None:
    cmd = [
        "sh",
        f"scripts/{script}",
        # Later occurrences of an option override the script's own values.
        "--output_dir_root",
        f"{OUTPUTS}/",
        *shlex.split(extra_args),
    ]
    env = dict(os.environ)
    if wandb_run_id:
        # Continue an existing W&B run (e.g. after resuming from a checkpoint).
        env |= {"WANDB_RUN_ID": wandb_run_id, "WANDB_RESUME": "must"}
    print("Running:", shlex.join(cmd), flush=True)
    try:
        subprocess.run(cmd, cwd=REMOTE_ROOT, env=env, check=True)
    finally:
        outputs.commit()
        cache.commit()


@app.function(
    gpu=os.environ.get("MODAL_GPU", "A100-80GB"),
    volumes={OUTPUTS: outputs, CACHE: cache},
    secrets=SECRETS,
    timeout=2 * 60 * 60,
)
def evaluate_flores(
    model: str, src_lang: str = "fra_Latn", tgt_lang: str = "mos_Latn", quantization: str = "int8"
) -> dict:
    """FLORES+ devtest for any HF model, exactly as the post-training eval does it."""
    from mt_training.eval import FLORES_DEFAULT_SPLIT, FLORES_PLUS, EvalConfig, run_evaluation
    from mt_training.train import convert_to_ct2

    name = model.replace("/", "--")
    ct2_dir = convert_to_ct2(model, f"{OUTPUTS}/eval/{name}-ct2-{quantization}", quantization)
    direction = (
        "" if (src_lang, tgt_lang) == ("fra_Latn", "mos_Latn") else f"-{src_lang}-{tgt_lang}"
    )
    try:
        metrics, _, _, _ = run_evaluation(
            EvalConfig(
                model=ct2_dir,
                dataset=FLORES_PLUS,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                split=FLORES_DEFAULT_SPLIT,
                output=f"{OUTPUTS}/eval/{name}{direction}-flores_plus-{FLORES_DEFAULT_SPLIT}.csv",
            )
        )
    finally:
        outputs.commit()
        cache.commit()
    return metrics


# One local entrypoint only: with two, `modal run scripts/modal_train.py` needs
# an explicit `::name` and the documented commands break.
@app.local_entrypoint()
def main(
    script: str = "debug.sh",
    extra_args: str = "",
    wandb_run_id: str = "",
    wait: bool = True,
    flores_model: str = "",
    src_lang: str = "fra_Latn",
    tgt_lang: str = "mos_Latn",
) -> None:
    if flores_model:
        # FLORES+ only, no training: --flores-model <hub id> [--src-lang … --tgt-lang …]
        print(flores_model, evaluate_flores.remote(flores_model, src_lang, tgt_lang))
        return
    if not (ROOT / "scripts" / script).is_file():
        raise SystemExit(f"No such script: scripts/{script}")
    if wait:
        # Streams logs; Ctrl+C here cancels the run, even with --detach.
        train.remote(script, extra_args, wandb_run_id)
        return
    # With --detach, a spawned call is not tied to this terminal at all.
    call = train.spawn(script, extra_args, wandb_run_id)
    print(f"Started {call.object_id}.")
    print("Logs: uvx modal app logs mt-training   Stop: uvx modal app stop mt-training")
