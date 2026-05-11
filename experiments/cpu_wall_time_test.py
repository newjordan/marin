#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""CPU wall-time test job for ``iris job cpu-time`` validation.

Submits a Zephyr pipeline with NUM_SHARDS worker tasks, each running a
prime sieve enough times to burn ~SHARD_SECONDS seconds of CPU time.
The total cpu_wall_ms reported by GetJobCpuTime should be approximately
NUM_SHARDS * SHARD_SECONDS (they run in parallel, so elapsed wall clock
for the whole job is just SHARD_SECONDS).

Usage — run under iris (submits the Zephyr job to the cluster):
    MARIN_PREFIX=gs://marin-us-central2/tmp uv run iris \
        --config=lib/iris/examples/marin-dev.yaml job run -- \
        python experiments/cpu_wall_time_test.py

Usage — local smoke test (inline execution, no iris worker tasks):
    MARIN_PREFIX=/tmp/marin-cpu-time-test uv run \
        python experiments/cpu_wall_time_test.py

After the job finishes, collect stats with:
    iris job cpu-time --json <job-id>

And compare with the per-shard elapsed times in the JSONL output.
"""

from __future__ import annotations

import argparse
import logging
import os
import secrets
from collections.abc import Iterator

from fray import ResourceConfig
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from rigging.filesystem import marin_prefix
from rigging.log_setup import configure_logging
from zephyr import Dataset, ZephyrContext

logger = logging.getLogger(__name__)

DEFAULT_NUM_SHARDS = 8


def _sieve(limit: int) -> int:
    """Return count of primes up to limit via sieve of Eratosthenes."""
    is_prime = bytearray([1]) * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i * i :: i] = bytearray(len(is_prime[i * i :: i]))
    return sum(is_prime)


def _flat_map(x: dict) -> Iterator[dict]:
    for i in range(100, 100000):
        yield {"key": i % 10, "value": _sieve(i)}


def _reduce(key: int, records: Iterator[dict]) -> Iterator[dict]:
    total = sum(record["value"] for record in records)
    yield {
        "key": key,
        "total": total,
    }


def _build_step(num_shards: int, run_id: str) -> StepSpec:
    def run(output_path: str) -> None:
        ctx = ZephyrContext(
            max_workers=num_shards,
            resources=ResourceConfig(cpu=1, ram="2g"),
            name="cpu-wall-time-test",
        )
        ctx.execute(
            Dataset.from_list([{"work_id": i} for i in range(num_shards)])
            .flat_map(_flat_map)
            .group_by(
                lambda record: record["key"],
                sort_by=lambda record: record["value"],
                reducer=_reduce,
            )
            .write_jsonl(f"{output_path}/results-{{shard:05d}}-of-{{total:05d}}.jsonl"),
        )

    return StepSpec(
        name="cpu-wall-time-test",
        output_path_prefix=f"{marin_prefix()}/tmp/cpu-wall-time-test-{run_id}",
        fn=run,
    )


if __name__ == "__main__":
    configure_logging(logging.INFO)
    os.environ.setdefault("MARIN_PREFIX", "/tmp/marin-cpu-time-test")

    parser = argparse.ArgumentParser(description="CPU wall-time test job for iris job cpu-time validation.")
    parser.add_argument(
        "--num-shards",
        type=int,
        default=DEFAULT_NUM_SHARDS,
        help=f"Number of Zephyr worker shards (default: {DEFAULT_NUM_SHARDS}).",
    )
    args = parser.parse_args()

    run_id = secrets.token_hex(4)
    logger.info("run_id=%s num_shards=%d", run_id, args.num_shards)
    StepRunner().run([_build_step(num_shards=args.num_shards, run_id=run_id)])
