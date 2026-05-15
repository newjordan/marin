# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CaseOps Marin Speedrun submission — the Pareto-triangle sweep.

Three arms, IDENTICAL model architecture + compute budget, only the tokenizer
differs — so the sole variable is the tokenizer:

  baseline   : marin-community/marin-tokenizer (~128K vocab, the speedrun default)
  control    : plain byte-level BPE retrained at VOCAB on untransformed text
  treatment  : CaseOps tokenizer (trust_remote_code wrapper) at VOCAB

Expected result (the "cold improvement"): control sits left of baseline on the
BPB-vs-FLOPs frontier (smaller vocab -> fewer embed/unembed FLOPs) but UP on BPB;
treatment sits left AND recovers the BPB — at/below the baseline line. Training-
free n-gram preflights (DESIGN.md / memory) already show the CaseOps BPB delta
goes monotonically favorable with context depth (unigram +6% -> trigram -1.9%).

All three arms tokenize the SAME raw documents (fineweb_edu_sample_10bt) so the
comparison isn't confounded by a different data selection.

STATUS: D.7-scoped (2026-05-14) — wired for a single A100-80G rental run.
RESOURCES is set, the FineWeb-EDU download is scoped to the sample/10BT subset
(NOT the full multi-TB repo), and tokenization is doc-capped. Run with
WANDB online + an HF token (gated read, for the paloma eval) in the env.

STILL-OPEN ITEMS:
  - AUTHOR: name/affiliation are empty — fill in before opening the submission
    PR (not required to run).
  - Tokenizer references: currently LOCAL paths under experiments/private/caseops/
    artifacts/ (Marin's load_tokenizer handles local dirs). For a portable
    public submission these likely need to become HF repo refs — but the wrapped
    CaseOps tokenizers are currently HF *subfolders*, and Marin's tokenizer
    string may not support subfolders. Resolve before the public PR.
  - The tokenizers are PREFLIGHT (trained on FineWeb-10B sample). The final
    submission tokenizers must be retrained on FineWeb-EDU — DESIGN.md R4.
"""

from __future__ import annotations

import logging
import os
import pathlib

from fray.cluster import ResourceConfig
from marin.datakit.canonical.fineweb_edu import download as fineweb_edu_download
from marin.execution.executor import executor_main

from experiments.defaults import default_tokenize
from experiments.llama import llama3_tokenizer_vocab_size
from experiments.marin_models import marin_tokenizer
from experiments.private.mockingbird_caseops_preflight import mockingbird_150m
from experiments.simple_train_config import SimpleTrainConfig
from experiments.speedrun.caseops.submission_support import Author, SpeedrunConfig, default_speedrun

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Submission config — EDIT AT POD TIME (D.7).
# --------------------------------------------------------------------------

AUTHOR = Author(
    name="",  # TODO: fill in before submitting
    affiliation="",
    url=None,
)

VOCAB = 32000  # CaseOps + control vocab size (baseline is fixed at llama3's ~128K)
MODEL_CONFIG = mockingbird_150m  # Mockingbird-150m (Phase 1-4 ported); identical across arms

# Same documents for every arm. Scoped to the sample/10BT subset (14 parquet
# shards, ~17 GB) — fineweb_edu_download() with no glob would pull the entire
# multi-TB FineWeb-EDU repo.
FINEWEB_EDU_REVISION = "87f0914"
RAW_SOURCE = (
    fineweb_edu_download(revision=FINEWEB_EDU_REVISION, hf_urls_glob=["sample/10BT/*.parquet"])
    .as_executor_step()
    .cd("sample/10BT")
)

# Single rented GPU. fray detects the device via "auto" — note that flops
# accounting in speedrun_results is only honest when fray's DEVICE_FLOPS table
# knows the device (a100-80g, h100, etc.). For exotic cards (e.g. RTX PRO 6000
# Blackwell as of 2026-05-15) the flops field will be zero/default; the loss
# curve and bpb are still honest, just the perf-vs-flops Pareto point isn't.
RESOURCES: ResourceConfig = ResourceConfig.with_gpu("auto", count=1, cpu=8, ram="64G", disk="64G")

TRAIN_BATCH_SIZE = 128
# Env override exists so a tiny preflight and the real run share one file.
NUM_TRAIN_STEPS = int(os.environ.get("CASEOPS_NUM_TRAIN_STEPS", "5000"))
LEARNING_RATE = 3e-3

# Per-shard document cap (sample/10BT has 14 shards) -> ~1.26M docs ~= ~1.3B
# tokens. Training consumes 128 x 5000 x 1024 = 655M tokens, so this is under one
# pass (no repeats) while keeping tokenization bounded for all 3 arms.
TOKENIZE_SAMPLE_COUNT = int(os.environ.get("CASEOPS_TOKENIZE_SAMPLE_COUNT", "90000"))

# Tokenizer references. Pushed to private HF repos as standalone top-level
# tokenizers (NOT subfolders) because Marin's levanter.tokenizers.load_tokenizer
# doesn't accept subfolder syntax. See experiments/private/caseops/push_to_hf.py
# for the multi-variant umbrella repo (Frosty40/caseops-marin-tokenizers); the
# specific 32k variants are split out so any fresh box can pull them directly.
TOKENIZERS = {
    "baseline": marin_tokenizer,                    # the speedrun default, ~128K
    "control":  "Frosty40/marin-plain-32000",       # plain byte-level BPE
    "treatment": "Frosty40/marin-caseops-32000",    # CaseOps lossless-caps wrapped
}
VOCAB_SIZES = {
    "baseline": llama3_tokenizer_vocab_size,
    "control": VOCAB,
    "treatment": VOCAB,
}
ARM_DESCRIPTIONS = {
    "baseline": f"Pareto-triangle baseline: marin-tokenizer (~{llama3_tokenizer_vocab_size} vocab), {MODEL_CONFIG} on fineweb-edu-10BT.",
    "control": f"Pareto-triangle control: plain byte-level BPE at vocab {VOCAB}, {MODEL_CONFIG} on fineweb-edu-10BT.",
    "treatment": f"CaseOps tokenizer at vocab {VOCAB} (lossless case-factoring): {MODEL_CONFIG} on fineweb-edu-10BT.",
}
ARMS = ("baseline", "control", "treatment")
# Each arm's training opens its own wandb run, and multiple concurrent wandb runs
# in one process collide (only one finalizes — the others orphan as `failed`).
# The launch path runs ONE arm per process via CASEOPS_ARM; main() additionally
# caps executor concurrency when >1 arm runs in a single process.
_ARM_FILTER = os.environ.get("CASEOPS_ARM")
if _ARM_FILTER is not None:
    if _ARM_FILTER not in ARMS:
        raise ValueError(f"CASEOPS_ARM={_ARM_FILTER!r} is not one of {ARMS}.")
    ARMS = (_ARM_FILTER,)


def build_config(arm: str) -> tuple[str, SpeedrunConfig]:
    """Build the (name, SpeedrunConfig) for one Pareto-triangle arm."""
    tokenizer = TOKENIZERS[arm]
    tokenized = default_tokenize(
        name=f"caseops/fineweb-edu-10bt-{arm}",
        dataset=RAW_SOURCE,
        tokenizer=tokenizer,
        sample_count=TOKENIZE_SAMPLE_COUNT,
    )
    train = SimpleTrainConfig(
        RESOURCES,
        train_seq_len=MODEL_CONFIG.max_seq_len,
        train_batch_size=TRAIN_BATCH_SIZE,
        num_train_steps=NUM_TRAIN_STEPS,
        learning_rate=LEARNING_RATE,
    )
    config = SpeedrunConfig(
        author=AUTHOR,
        description=ARM_DESCRIPTIONS[arm],
        model_config=MODEL_CONFIG,
        train_config=train,
        tokenized_dataset=tokenized,
        vocab_size=VOCAB_SIZES[arm],  # honest FLOPs/param accounting per arm
    )
    return f"caseops-{arm}-{VOCAB}", config


def main() -> None:
    if os.getenv("CI") is not None:
        logger.info("Skipping speedrun execution on CI (needs HF access + accelerator).")
        return

    steps = []
    for arm in ARMS:
        name, cfg = build_config(arm)
        cfg.print_run_info()
        steps.extend(default_speedrun(name, cfg))

    # >1 arm in one process => serialize so the per-arm wandb runs don't collide
    # and each arm gets the whole GPU. One arm per process needs no cap.
    max_concurrent = 1 if len(ARMS) > 1 else None
    executor_main(
        steps=steps,
        description="CaseOps Pareto-triangle speedrun: baseline vs plain-BPE vs CaseOps.",
        max_concurrent=max_concurrent,
    )


if __name__ == "__main__":
    main()
