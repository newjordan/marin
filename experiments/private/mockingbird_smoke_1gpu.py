# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mockingbird single-GPU smoke on TinyStories.

Variant of mockingbird_smoke_gpu.py for 1-GPU rentals (e.g. a single RTX PRO 6000).
Skips FSDP — verifies just init/fwd/bwd/loss-decreasing on one device. Use this
to confirm the partial Mockingbird port runs on a real GPU before pushing on the
custom transformer skeleton (Phase 2+).

Run on the rented box, AFTER `uv sync --all-packages --extra=gpu`:

    WANDB_MODE=offline MARIN_PREFIX=/tmp/marin_smoke \
        uv run experiments/private/mockingbird_smoke_1gpu.py --prefix /tmp/marin_smoke
"""

from fray import ResourceConfig
from levanter.data.text import TextLmDatasetFormat
from levanter.models.mockingbird import MockingbirdConfig
from marin.execution.executor import executor_main, versioned

from experiments.defaults import default_tokenize, default_train
from experiments.marin_models import marin_tokenizer
from experiments.simple_train_config import SimpleTrainConfig

# 256 dim x 6 layers x 4 heads, mlp_mult=3.75 ~= 7M params (tied embeds).
mockingbird_small = MockingbirdConfig(
    max_seq_len=512,
    hidden_dim=256,
    num_layers=6,
    num_heads=4,
    num_kv_heads=4,
    mlp_mult=3.75,
    qk_gain=5.25,
    loop_start=2,
    loop_end=3,
    num_loops=2,           # 3 effective reps over loop_layers when enabled
    enable_looping=True,   # Phase 2a — exercise the looped-middle mechanic
)

tinystories_hf_id = "roneneldan/TinyStories"

tinystories_tokenized = default_tokenize(
    name=tinystories_hf_id,
    dataset=tinystories_hf_id,
    tokenizer=marin_tokenizer,
    format=TextLmDatasetFormat(),
    sample_count=5000,
)

gpu_train_config = SimpleTrainConfig(
    resources=ResourceConfig.with_gpu("auto", count=1, cpu=8, ram="32G", disk="32G"),
    train_batch_size=8,
    num_train_steps=50,
    learning_rate=6e-4,
    weight_decay=0.1,
    max_eval_batches=4,
)

mockingbird_1gpu_smoke_model = default_train(
    name="mockingbird-smoke-1gpu-tinystories",
    tokenized=tinystories_tokenized,
    model_config=versioned(mockingbird_small),
    train_config=gpu_train_config,
    tags=["mockingbird", "smoke", "gpu", "1x", "tinystories", "private"],
    eval_harness_tasks=[],
    use_default_validation=False,
)


if __name__ == "__main__":
    executor_main(steps=[mockingbird_1gpu_smoke_model])
