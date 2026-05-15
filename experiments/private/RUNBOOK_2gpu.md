# 2-GPU rental smoke runbook

When you spin up the 2-GPU rented host, run these in order. Each step gates the next.

## 0. Pre-flight on the rental host

```bash
# Confirm 2 GPUs, CUDA driver, free disk
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
df -h .
python3 --version   # must be >= 3.11
which uv || curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 1. Get the repo on the rental host

```bash
# If the rental is a fresh box:
gh repo clone newjordan/marin
cd marin
git checkout mockingbird-baseline
git pull --rebase upstream main || true   # optional, only if you want upstream changes
```

> **Note:** Do NOT copy `.venv/` from this DGX Spark to the rental.
> The aarch64 wheels will not work on x86_64 (and vice versa).
> Always run `uv sync` fresh on the rental.

## 2. Install GPU deps (5-10 min)

```bash
uv sync --all-packages --extra=gpu
```

This installs `jax[cuda13]==0.10.0` and pulls CUDA runtime libs. If you get a
CUDA-version mismatch error, check `nvcc --version` — the rental's CUDA must be
>= 13.0. If it's stuck on CUDA 12, you can fall back to `--extra=cpu` for the
Marin pipeline smoke (it'll be slow but proves the pipeline works), and skip the
GPU steps below.

## 3. Link smoke — pure JAX, fast (~30 s)

```bash
.venv/bin/python experiments/private/link_smoke_jax.py
```

Expected output ends with:

```
LINK SMOKE PASSED.
```

If it prints `FAIL: all devices are CPU.`, the GPU extra didn't install
correctly — re-check `uv sync` output and `nvidia-smi`.

If it prints `FAIL: need >= 2 devices, got 1.`, only one GPU is visible to JAX.
Check `CUDA_VISIBLE_DEVICES` and `nvidia-smi` again.

## 4. Marin pipeline smoke — Mockingbird across both GPUs (~3-5 min on A100, longer on smaller GPUs)

```bash
export WANDB_MODE=offline
export MARIN_PREFIX=/tmp/marin_smoke
mkdir -p $MARIN_PREFIX

uv run experiments/private/mockingbird_smoke_gpu.py --prefix $MARIN_PREFIX
```

Expected: tokenization step succeeds, then 50 train steps, then `Step
checkpoints/mockingbird-smoke-gpu-tinystories_... succeeded`. Eval loss should
decrease over the run (init loss ~11.76, expect end loss < 11.0 in 50 steps).

If FSDP misbehaves you'll see it in the first train step. Watch for:
 - `RuntimeError: failed to all-reduce` -> NCCL bring-up issue.
 - Slow per-step time (>> single-GPU baseline) -> link is bandwidth-starved
   (would be a real concern for any larger run).

## 5. If everything passes — decision point

You've confirmed:
 - GPU JAX wheels work on the rental's arch + CUDA version
 - Both GPUs are visible to JAX
 - NCCL collective works
 - Marin/Levanter pipeline runs on GPU
 - FSDP across 2 devices works for Mockingbird
 - Mockingbird itself (the partial port) trains stably at non-trivial scale

Next steps (each is now unblocked):
 - Bump `mockingbird_small` to the actual reference shape (11L x 512 x 8 heads,
   mlp_mult=3.75) and run for ~500 steps to see loss trajectory.
 - Layer in the TODO pieces from `lib/levanter/src/levanter/models/mockingbird.py`
   (per-head QK gain, resid_mix, U-Net skips, looped middle block). Re-run this
   smoke each time to catch regressions.
 - When the full Mockingbird trains cleanly, scale to a real Speedrun compute
   tier on a bigger rental or via Marin Discord cluster access.
