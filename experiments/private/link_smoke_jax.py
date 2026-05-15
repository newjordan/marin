# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure-JAX 2-device link smoke. NO Marin/Levanter imports.

Goal: confirm both GPUs are visible to JAX AND that a collective (all-reduce)
runs end-to-end across the link. If this fails, the Marin pipeline smoke will
also fail — and the error here will be much easier to diagnose than the
deep-stack one you'd get from inside Levanter's FSDP.

Run on the rented 2-GPU box, AFTER `uv sync --all-packages --extra=gpu`:

    cd /path/to/marin
    .venv/bin/python experiments/private/link_smoke_jax.py

Exits 0 on success, raises on any failure. Tests:
 1. jax.devices() reports >= 2 devices.
 2. A small pmap all-reduce produces the correct sum across devices.
 3. A larger pmap all-reduce (~10MB per device) completes — gives a rough
    bandwidth read.
 4. A matmul under pmap actually runs on each device (uses sharded inputs).

If you only have 1 GPU available, this still passes if you set
`MARIN_SMOKE_REQUIRE_NDEV=1`. Default is 2.
"""

import os
import sys
import time

import jax
import jax.numpy as jnp


def main() -> int:
    required = int(os.environ.get("MARIN_SMOKE_REQUIRE_NDEV", "2"))

    devs = jax.devices()
    print(f"[1/4] jax.devices() returned {len(devs)} device(s):")
    for d in devs:
        print(f"      - {d} (platform={d.platform}, kind={d.device_kind})")
    if len(devs) < required:
        print(f"FAIL: need >= {required} devices, got {len(devs)}.")
        return 1
    if all(d.platform == "cpu" for d in devs):
        print("FAIL: all devices are CPU. GPU jax extras likely not installed.")
        return 1

    n = min(len(devs), required)
    print(f"[2/4] pmap small all-reduce across {n} device(s)...")
    # One scalar per device; psum should yield n on each replica.
    x = jnp.arange(n, dtype=jnp.float32) + 1.0  # [1, 2, ..., n]
    expected = float(x.sum())  # 1+2+...+n

    f = jax.pmap(lambda v: jax.lax.psum(v, axis_name="i"), axis_name="i")
    out = f(x)
    out.block_until_ready()
    got = [float(v) for v in out]
    print(f"      input per device: {[float(v) for v in x]}")
    print(f"      psum output:      {got}")
    if not all(abs(v - expected) < 1e-5 for v in got):
        print(f"FAIL: psum mismatch, expected {expected} on each replica.")
        return 1

    print("[3/4] pmap large all-reduce (10MB per device)...")
    elems = 10 * 1024 * 1024 // 4  # ~10MB of f32 per device
    big = jnp.ones((n, elems), dtype=jnp.float32)
    t0 = time.perf_counter()
    out = f(big)
    out.block_until_ready()
    dt = time.perf_counter() - t0
    expected_sum = float(n)  # sum of 1.0 across n devices
    if not jnp.allclose(out, expected_sum):
        print("FAIL: large psum mismatch.")
        return 1
    bytes_xferred = elems * 4 * n
    bw_mb_s = bytes_xferred / dt / (1024 * 1024)
    print(f"      OK in {dt * 1000:.1f} ms (~{bw_mb_s:.0f} MB/s aggregate, rough)")

    print("[4/4] pmap matmul (verify per-device compute, not just collective)...")
    dim = 1024
    a = jnp.ones((n, dim, dim), dtype=jnp.float32)
    b = jnp.ones((n, dim, dim), dtype=jnp.float32)
    g = jax.pmap(jnp.matmul)
    t0 = time.perf_counter()
    c = g(a, b)
    c.block_until_ready()
    dt = time.perf_counter() - t0
    # Each element of c should be `dim` (sum of dim ones).
    if not jnp.allclose(c[0, 0, 0], float(dim)):
        print(f"FAIL: matmul mismatch, got {float(c[0, 0, 0])}, expected {dim}.")
        return 1
    print(f"      OK in {dt * 1000:.1f} ms ({n} parallel {dim}x{dim} matmuls)")

    print()
    print("LINK SMOKE PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
