# Mockingbird → Marin/Levanter: faithful architecture port

**Goal:** Port the Mockingbird *architecture* into Levanter as a clean `LmConfig`/`LmHeadModel`,
faithful to the Parameter Golf reference, so it can be the model arm of the CaseOps
speedrun triangle (and a standalone Marin speedrun contribution).

**Reference (ground truth):**
`/home/frosty40/parameter-golf-lab/records/track_10min_16mb/2026-05-01_Mockingbird_8xH100/train_gpt.py`
(3859 lines — a full PG `track_10min_16mb` competition submission).

## Scope: what transfers, what does NOT

The PG reference is a *compress-to-16MB + test-time-training* competition entry. The Marin
speedrun is a pure pretraining-efficiency benchmark (paloma/c4_en BPB at fixed hardware FLOPs).
So:

**Transfers (this port):** the architecture + Muon optimizer + the training recipe (LRs,
schedule, grad clip). Levanter already has Muon (`lib/levanter/src/levanter/optim/muon.py`,
`muonh.py`, `grugmuon.py`).

**Does NOT transfer (different competition — out of scope):** GPTQ 6-bit quant, LQER,
3-phase test-time training, EMA-for-compression, the custom Triton kernels (Levanter has its
own), document packing. The PG headline "1.062 BPB" is a *quantized + TTT-adapted* number —
not comparable to Marin leaderboard BPB. Do not chase it.

## Architecture feature inventory (2026-05-01 run enables ALL of these)

Per-block features (fit a Levanter-style block):
- [x] leaky_relu(0.5)² MLP (non-gated)            — DONE in current mockingbird.py
- [x] per-block per-dim attn_scale / mlp_scale     — DONE
- [x] RMSNorm pre-norm, tied embeddings            — DONE
- [ ] per-head learnable QK gain (Param (num_heads,), init 5.25) — current port uses a scalar
- [ ] QK RMS-norm on q,k before RoPE
- [ ] partial RoPE (rope_dims=16: rotate first 16 head dims only)
- [ ] resid_mix: per-block Param (2,dim), x_in = mix[0]*x + mix[1]*x0  (x0 = post-norm embed)
- [ ] ln_scale: multiply norm output by 1/sqrt(layer_idx+1)
- [ ] sparse attention gate (W_g (num_heads, gate_window=12), scale 0.5, on SDPA output)
- [ ] XSA: orthogonal value projection on attn output (`_xsa_efficient`), all layers

Model-level / structural (do NOT fit the stock `Stacked` scan — need a custom forward):
- [ ] smear gate: x_t += lam*sigmoid(W*x_t[:12]) * x_{t-1}, BOS-masked, right after embed
- [ ] U-Net skips: encoder layers collect outputs; decoder layers pop them with learnable
      skip_weights + sigmoid skip_gates (lerp)
- [ ] looped middle block: layers [loop_start..loop_end] re-run num_loops+1 times; the
      enc/dec split is over the *expanded* index list. (PG toggles it at 45% of training;
      for the port, make it a static config — always-on — unless we add a step schedule.)
- [ ] parallel lanes: decoder layers >= parallel_start_layer (8) run a 2-lane block
      (`_parallel_block`) with parallel_resid_lambdas / parallel_post_lambdas; final hidden
      = mean of lanes
- [ ] logit softcap on the LM head (softcap=30: `cap*tanh(logits/cap)`)

Config fix: reference is `num_kv_heads=4` (current mockingbird.py defaults to 8).

## Plan (phased, CPU-smoke-verified at each step)

**Phase 1 — per-block features** (fit the existing block cleanly):
per-head QK gain, QK RMS-norm, partial RoPE, resid_mix (x0 threaded as scan constant),
ln_scale, logit softcap, num_kv_heads fix. Verify: CPU smoke (init/fwd/bwd, loss decreasing).

**Phase 2 — the custom transformer skeleton:**
Replace the stock `Stacked` forward with a hand-written encoder/loop/decoder forward:
embed → smear gate → encoder (collect skips) → looped middle → decoder (pop skips w/ gates)
→ parallel lanes for layers ≥ psl → final norm. Static index lists (looping always-on).
Verify: CPU smoke + shape/structure checks vs reference.

**Phase 3 — attention extras:** sparse attention gate, XSA. Verify: CPU smoke.

**Phase 4 — wire-up + GPU verify:** plug `MockingbirdConfig` into `caseops_sweep.py` as
`MODEL_CONFIG`; smoke on the rented GPU box (box 1, ssh1:17246). Confirm Muon optimizer
config. Then the CaseOps triangle runs on true Mockingbird.

## Current state (2026-05-15, EOD)
- `lib/levanter/src/levanter/models/mockingbird.py` — Phase 1-4 COMPLETE.
  All architecture features from PG 2026-05-01 reference ported and GPU-verified.
  Branch `mockingbird-baseline` pushed to `origin` (newjordan/marin); HEAD `178ba0761`.
- Phase-4 preflight (mockingbird_150m, 200 steps on TinyStories): bpb 1.17, loss 3.33,
  4.1 it/s on RTX PRO 6000.
- Active box: Texas RTX PRO 6000 instance 36820913 (`ssh -p 20912 root@ssh4.vast.ai`).

## To plug Mockingbird into the actual CaseOps triangle
Edit `experiments/speedrun/caseops/caseops_sweep.py`:

```python
# from experiments.llama import llama_150m
# MODEL_CONFIG = llama_150m
from experiments.private.mockingbird_caseops_preflight import mockingbird_150m
MODEL_CONFIG = mockingbird_150m
```

Optimizer-side: caseops_sweep currently uses SimpleTrainConfig defaults (AdamW).
Muon-on-Mockingbird is a separate follow-up — Levanter has Muon at
`lib/levanter/src/levanter/optim/muon.py`, `muonh.py`, `grugmuon.py`.
