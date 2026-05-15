# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU smoke of the CaseOps sweep machinery — D.6b, no pod.

caseops_sweep.py itself can't be smoke-run (RESOURCES is a pod placeholder, the
model/steps are pod-sized, and it pulls the multi-GB FineWeb-EDU download). This
smoke exercises the SAME machinery path — default_tokenize + default_train — for
all 3 tokenizer arms, at nano scale on TinyStories, CPU only.

What it de-risks (the CaseOps-specific integration unknowns):
  1. Does default_tokenize accept the CaseOps trust_remote_code tokenizer at a
     local path? Does Marin's tokenize pipeline propagate trust_remote_code /
     HF_MODULES_CACHE into its workers?  <-- #1 risk, never tested before
  2. Does the executor train a model on the CaseOps-tokenized cache (model init
     at the CaseOps vocab, forward/backward)?
  3. Does eval BPB compute on the CaseOps tokenizer (markers->0-byte accounting
     via byte_length_of_token in the real eval path)?

Arms: baseline (marin-tokenizer ~128K), control (plain_8000), treatment
(caseops_8000_wrapped) — smallest vocab = fastest smoke.

Run (CPU, no pod):
    export WANDB_MODE=offline
    export HF_MODULES_CACHE=/home/frosty40/.cache/hf_modules
    export MARIN_PREFIX=/tmp/marin_caseops_smoke
    uv run experiments/private/caseops/smoke_caseops_sweep.py --prefix /tmp/marin_caseops_smoke
"""

from __future__ import annotations

import os
import pathlib

from fray import ResourceConfig
from levanter.data.text import TextLmDatasetFormat
from marin.execution.executor import executor_main, versioned

from experiments.defaults import default_tokenize, default_train
from experiments.llama import llama_nano
from experiments.marin_models import marin_tokenizer
from experiments.simple_train_config import SimpleTrainConfig

_ARTIFACTS = pathlib.Path(__file__).resolve().parent / "artifacts"

# 3 arms, smallest vocab for speed. Local dirs — Marin's load_tokenizer handles dirs.
ARM_TOKENIZERS = {
    "baseline": marin_tokenizer,
    "control": str(_ARTIFACTS / "plain_8000"),
    "treatment": str(_ARTIFACTS / "caseops_8000_wrapped"),
}

TINYSTORIES = "roneneldan/TinyStories"

# CASEOPS_SMOKE_GPU=1 runs on the local GB10 GPU instead of CPU (still nano scale —
# this just confirms the pipeline runs on the Blackwell GPU).
_SMOKE_RESOURCES = (
    ResourceConfig.with_gpu("GB10", count=1)
    if os.environ.get("CASEOPS_SMOKE_GPU", "0") == "1"
    else ResourceConfig.with_cpu()
)

nano_train_config = SimpleTrainConfig(
    resources=_SMOKE_RESOURCES,
    train_batch_size=4,
    num_train_steps=20,
    learning_rate=6e-4,
    max_eval_batches=2,
)


# CASEOPS_SMOKE_PALOMA=1 exercises the full paloma eval/paloma/c4_en/bpb path
# (paloma re-tokenized with each arm's tokenizer + BPB via byte_length_of_token) —
# this is the risk-#3 validation. Default off keeps the smoke fast (tokenize+train only).
PALOMA_EVAL = os.environ.get("CASEOPS_SMOKE_PALOMA", "0") == "1"


def _arm_steps(arm: str, tokenizer: str):
    """default_tokenize + default_train for one arm — the same machinery caseops_sweep uses."""
    tokenized = default_tokenize(
        name=f"caseops-smoke/{arm}",
        dataset=TINYSTORIES,
        tokenizer=tokenizer,
        format=TextLmDatasetFormat(),
        sample_count=500,
    )
    model = default_train(
        name=f"caseops-smoke-{arm}",
        tokenized=tokenized,
        model_config=versioned(llama_nano),
        train_config=nano_train_config,
        tags=["caseops", "smoke", arm, "private"],
        eval_harness_tasks=[],
        use_default_validation=PALOMA_EVAL,  # True -> exercises eval/paloma/c4_en/bpb (risk #3)
    )
    return [model]


if __name__ == "__main__":
    steps = []
    for arm, tok in ARM_TOKENIZERS.items():
        steps.extend(_arm_steps(arm, tok))
    executor_main(steps=steps)
