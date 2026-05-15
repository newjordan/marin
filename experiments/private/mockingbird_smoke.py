# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mockingbird CPU smoke run on TinyStories.

This is the Phase B.2 driver: confirm that the partial Mockingbird port (see
lib/levanter/src/levanter/models/mockingbird.py) trains for a handful of steps
on TinyStories without exploding, producing a finite, decreasing loss.

NOT a leaderboard submission. Sizes are tiny (32-dim, 2 layers, 32 batch tokens).
Run:

    wandb offline
    uv run experiments/private/mockingbird_smoke.py --prefix local_store

See docs/tutorials/train-an-lm.md for the executor pattern.
"""

from fray import ResourceConfig
from levanter.data.text import TextLmDatasetFormat
from levanter.models.mockingbird import MockingbirdConfig
from marin.execution.executor import executor_main, versioned

from experiments.defaults import default_tokenize, default_train
from experiments.marin_models import marin_tokenizer
from experiments.simple_train_config import SimpleTrainConfig

# ---- Model ---------------------------------------------------------------

# Tiny config: smoke-only. Mirrors llama_nano shape (32 dim, 2 layers, 2 heads).
# loop_start/loop_end forced inside the valid range for num_layers=2; looping disabled.
mockingbird_nano = MockingbirdConfig(
    max_seq_len=512,
    hidden_dim=32,
    num_layers=2,
    num_heads=2,
    num_kv_heads=2,
    mlp_mult=4.0,  # intermediate ~ 128
    qk_gain=5.25,
    loop_start=0,
    loop_end=0,
    enable_looping=False,
)


# ---- Data ----------------------------------------------------------------

tinystories_hf_id = "roneneldan/TinyStories"

tinystories_tokenized = default_tokenize(
    name=tinystories_hf_id,
    dataset=tinystories_hf_id,
    tokenizer=marin_tokenizer,
    format=TextLmDatasetFormat(),
    sample_count=1000,
)


# ---- Train config --------------------------------------------------------

nano_train_config = SimpleTrainConfig(
    resources=ResourceConfig.with_cpu(),
    train_batch_size=4,
    num_train_steps=20,
    learning_rate=6e-4,
    weight_decay=0.1,
    max_eval_batches=2,
)

mockingbird_smoke_model = default_train(
    name="mockingbird-smoke-tinystories",
    tokenized=tinystories_tokenized,
    model_config=versioned(mockingbird_nano),
    train_config=nano_train_config,
    tags=["mockingbird", "smoke", "tinystories", "private"],
    eval_harness_tasks=[],
    use_default_validation=False,
)


if __name__ == "__main__":
    executor_main(steps=[mockingbird_smoke_model])
