# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mockingbird 2-GPU pipeline smoke on TinyStories.

GPU variant of mockingbird_smoke.py. Same model code, larger size, GPU resources,
50 train steps so the loss curve is visible. Mesh `data: -1` lets Levanter shard
the data axis across all available GPUs; with `count=2` this means FSDP across
both GPUs and you'll see allreduce time in the train_step trace.

Run on the rented 2-GPU box, AFTER:
 1. uv sync --all-packages --extra=gpu        # ~5-10 min
 2. python experiments/private/link_smoke_jax.py   # confirm link first

Then:

    WANDB_MODE=offline MARIN_PREFIX=/tmp/marin_smoke \
        uv run experiments/private/mockingbird_smoke_gpu.py --prefix /tmp/marin_smoke

NOT a leaderboard submission. Sizes are still small (256 dim, 6 layers, ~7M
params before tied-embed reuse). The point is to confirm:
 - Mockingbird forward+backward run on GPU (no kernel issues)
 - FSDP across 2 devices works (the link sync runs every step)
 - The Marin executor pipeline survives a real (non-CPU) device
"""

from fray import ResourceConfig
from levanter.data.text import TextLmDatasetFormat
from levanter.models.mockingbird import MockingbirdConfig
from marin.execution.executor import executor_main, versioned

from experiments.defaults import default_tokenize, default_train
from experiments.marin_models import marin_tokenizer
from experiments.simple_train_config import SimpleTrainConfig

# ---- Model: bigger than CPU smoke so FSDP has something to sync ---------

# 256 dim x 6 layers x 4 heads, mlp_mult=3.75 ~= 7M params (tied embeds).
# Stays well within any consumer / low-tier rental GPU pair.
mockingbird_small = MockingbirdConfig(
    max_seq_len=512,
    hidden_dim=256,
    num_layers=6,
    num_heads=4,
    num_kv_heads=4,
    mlp_mult=3.75,
    qk_gain=5.25,
    # Loop disabled (still TODO in the module). Forced inside valid range
    # so __post_init__ never trips even if enable_looping flipped on.
    loop_start=2,
    loop_end=3,
    enable_looping=False,
)


# ---- Data: same as CPU smoke ---------------------------------------------

tinystories_hf_id = "roneneldan/TinyStories"

tinystories_tokenized = default_tokenize(
    name=tinystories_hf_id,
    dataset=tinystories_hf_id,
    tokenizer=marin_tokenizer,
    format=TextLmDatasetFormat(),
    sample_count=5000,
)


# ---- Train config: 2 GPUs, FSDP across data axis -------------------------

# `gpu_type="auto"` lets fray detect the rented GPU. If detection fails or you
# want a hard pin, replace with "A10G" / "L4" / "A100-80G" / "H100" / etc.
# `count=2` is the link-test point.
gpu_train_config = SimpleTrainConfig(
    resources=ResourceConfig.with_gpu("auto", count=2, cpu=8, ram="32G", disk="32G"),
    # batch=8 splits cleanly to 4 per device.
    train_batch_size=8,
    num_train_steps=50,
    learning_rate=6e-4,
    weight_decay=0.1,
    max_eval_batches=4,
)

mockingbird_gpu_smoke_model = default_train(
    name="mockingbird-smoke-gpu-tinystories",
    tokenized=tinystories_tokenized,
    model_config=versioned(mockingbird_small),
    train_config=gpu_train_config,
    tags=["mockingbird", "smoke", "gpu", "2x", "tinystories", "private"],
    eval_harness_tasks=[],
    use_default_validation=False,
)


if __name__ == "__main__":
    executor_main(steps=[mockingbird_gpu_smoke_model])
