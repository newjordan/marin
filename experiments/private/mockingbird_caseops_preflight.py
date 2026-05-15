# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mockingbird-on-CaseOps preflight (Phase 4 GPU verify).

Single-arm short run on TinyStories that exercises the SAME plumbing the real
caseops_sweep.py uses (default_train + executor + checkpoint + eval), but with
Mockingbird as the model and a tiny step budget so we can confirm GPU verify
green without paying for the full FineWeb-EDU triangle.

Once this is green:
 * The Mockingbird port can be plugged into experiments/speedrun/caseops/
   caseops_sweep.py by swapping `MODEL_CONFIG = llama_150m` for
   `MODEL_CONFIG = mockingbird_150m` from this file.

Run on the rented box AFTER `uv sync --all-packages --extra=gpu`:

    WANDB_MODE=offline MARIN_PREFIX=/tmp/marin_mockingbird_preflight \
        uv run experiments/private/mockingbird_caseops_preflight.py \
        --prefix /tmp/marin_mockingbird_preflight
"""

from __future__ import annotations

import os

from fray import ResourceConfig
from levanter.data.text import TextLmDatasetFormat
from levanter.models.mockingbird import MockingbirdConfig
from marin.execution.executor import executor_main, versioned

from experiments.defaults import default_tokenize, default_train
from experiments.marin_models import marin_tokenizer
from experiments.simple_train_config import SimpleTrainConfig

# ─── mockingbird_150m: sized to roughly match llama_150m's compute ──────────
# llama_150m is 6L x 512 dim x 8 heads x mlp_mult=3.5 (intermediate=1792).
# Mockingbird mirrors all of those, with num_kv_heads=4 (PG GQA 8:4) and the
# Phase 1-3 features at default. Loop range / parallel layers fit cleanly:
#   enc = layers 0..1   (n_enc=2)
#   loop = layers 2..3  (n_loop=2, runs num_loops+1=3 times when looping)
#   dec_reg = none      (parallel_start_layer=4 == loop_end+1)
#   dec_par = layers 4..5 (n_dec_par=2, 2-lane parallel blocks)
mockingbird_150m = MockingbirdConfig(
    max_seq_len=1024,
    hidden_dim=512,
    num_layers=6,
    num_heads=8,
    num_kv_heads=4,
    mlp_mult=3.5,         # 1792 / 512 = 3.5, matches llama_150m intermediate_dim
    qk_gain=5.25,
    loop_start=2,
    loop_end=3,
    num_loops=2,
    enable_looping=True,
    parallel_start_layer=4,
)


# ─── Data — TinyStories so we don't need HF auth / FineWeb-EDU download ─────
tinystories_hf_id = "roneneldan/TinyStories"
tinystories_tokenized = default_tokenize(
    name=tinystories_hf_id,
    dataset=tinystories_hf_id,
    tokenizer=marin_tokenizer,
    format=TextLmDatasetFormat(),
    sample_count=20000,
)


# ─── Train config — 200 steps so we get a real loss curve, not just init/fwd ─
NUM_STEPS = int(os.environ.get("MOCKINGBIRD_PREFLIGHT_STEPS", "200"))
gpu_train_config = SimpleTrainConfig(
    resources=ResourceConfig.with_gpu("auto", count=1, cpu=8, ram="32G", disk="32G"),
    train_batch_size=16,
    num_train_steps=NUM_STEPS,
    learning_rate=3e-3,    # caseops_sweep value
    weight_decay=0.1,
    max_eval_batches=4,
)

mockingbird_preflight = default_train(
    name="mockingbird-caseops-preflight",
    tokenized=tinystories_tokenized,
    model_config=versioned(mockingbird_150m),
    train_config=gpu_train_config,
    tags=["mockingbird", "caseops", "preflight", "gpu", "1x", "tinystories", "private"],
    eval_harness_tasks=[],
    use_default_validation=False,
)


if __name__ == "__main__":
    executor_main(steps=[mockingbird_preflight])
