# CaseOps for Marin Speedrun — design

**Status:** private working doc. Submission focus as of 2026-05-14.
**Goal:** a CaseOps tokenizer that Pareto-dominates the plain-BPE / llama3 baseline
on the Marin Speedrun (BPB vs total hardware FLOPs). A cold, understated drop —
let the frontier point speak.

## What CaseOps is

A bijective, ASCII-only text transform applied *before* BPE. Source:
`lossless_caps.py` (copied verbatim from the Parameter Golf SOTA lineage,
sha256 `963626d7…`, matches the Mockingbird run manifest — DO NOT modify it).

The shipped scheme is `lossless_caps_caseops_v1` == `encode_lossless_caps_v2`:
4 PUA marker chars — TITLE (U+E001), ALLCAPS (U+E002), CAPNEXT (U+E003),
ESC (U+E004). Over maximal ASCII alphabetic runs:
- lowercase word → unchanged
- `TitleCase` → `TITLE + lower(word)`
- `ALLCAPS` (len≥2) → `ALLCAPS + lower(word)`
- mixed case → `[TITLE?] + lower[0] + (CAPNEXT + lower[c] per upper c)`
- literal marker in input → `ESC + char`
- **non-ASCII chars: never touched** — this is what makes losslessness airtight
  (sidesteps all Unicode case hell: ß, Turkish i, ligatures, etc.)

Verified: `decode(encode(x)) == x` byte-exact on 30 adversarial cases + 30,000
real FineWeb docs (90 MB). See `test_lossless_roundtrip.py`.

## Why it's a fair, in-scope win (not metric-gaming)

- Marin Speedrun scores `eval/paloma/c4_en/bpb` — **bits per BYTE of raw text**,
  tokenizer-agnostic by construction.
- `default_speedrun()` re-tokenizes the Paloma C4-EN eval with the *training*
  tokenizer (`speedrun_paloma_tokenized(tokenizer=_get_tokenizer_for_train(...))`).
  So the eval automatically follows our tokenizer.
- A better tokenizer that lets the model encode the byte stream in fewer bits is
  a genuine LM-efficiency win. CaseOps de-fragments case statistics → the win is
  real, measurable, and honest.
- Marin's announcement explicitly invites "a new … tokenizer." This is in scope.

## Why it works at all (the BPB-byte-counting compatibility)

Levanter computes BPB with a per-token-**type** byte table
(`eval.py:_calculate_bytes_per_token_type` → `bytes_per_token[token_id]`).
CaseOps is exactly compatible with this:
- marker tokens (TITLE/ALLCAPS/CAPNEXT/ESC) → **0 original bytes**
- every other BPE piece → naive UTF-8 length == original byte count, because the
  only transform on letters is ASCII-upper→ASCII-lower (always 1 byte → 1 byte),
  and `user_defined_symbols` keeps markers atomic so no BPE piece ever straddles
  a marker.
- Only ambiguous case: an ESC-escaped *literal* PUA char in the input. PUA chars
  essentially never occur in real web text — negligible, and verifiable as zero
  on the actual Paloma C4-EN set.
=> BPB stays computable on TRUE original bytes. No per-occurrence sidecar needed
   (unlike the PG `prepare_caseops_data.py` pipeline, which used one because its
   custom train_gpt.py wanted it; Levanter's per-type model is sufficient).

## The strategic framing — it's a Pareto move, not a fixed-vocab win

CaseOps' edge is NOT primarily "lower BPB at fixed vocab." At llama3's ~128K
vocab there is little case-fragmentation pressure — the baseline BPE already
affords separate `The`/`the`/`THE`. CaseOps' real lever:

> **It changes the vocab-size ↔ BPB tradeoff** — lets you shrink the vocab
> (smaller embedding/unembedding → fewer FLOPs/token) while paying a much smaller
> BPB penalty than a plain BPE would, because case is factored into cheap,
> highly-predictable marker tokens instead of fragmenting the vocab.

So the submission is a **frontier-shift**, demonstrated as a small triangle:

| point | tokenizer | vocab | expected position |
|---|---|---|---|
| baseline | llama3 (speedrun default) | ~128K | on the existing frontier |
| control | plain BPE | ~48K | LEFT of baseline (less FLOPs), UP on BPB (worse) |
| treatment | CaseOps BPE | ~48K | LEFT *and* recovers the BPB — at/below the baseline line |

That triangle isolates the variable: plain-48K vs CaseOps-48K is pure "CaseOps
on/off"; both vs llama3-128K shows the FLOPs win. (~48K is a starting guess —
big enough that shrinking from 128K is a meaningful FLOPs save, small enough that
case-fragmentation actually bites. May sweep {32K,48K,64K} later.)

## Integration plan

1. **Transform** — `lossless_caps.py` (done, verified). Apply
   `encode_lossless_caps_v2` to raw text before BPE.
2. **Tokenizer — RESOLVED (D.3).** Ship CaseOps as a **`trust_remote_code`
   custom HF tokenizer directory**. The custom tokenizer class wraps
   (CaseOps transform + a plain BPE trained on CaseOps-transformed text). The 4
   markers are registered as **special tokens**. Then `encode(raw_text)` does
   transform→BPE automatically, so BOTH the training tokenization AND the
   speedrun eval re-tokenization (`speedrun_paloma_tokenized(...)`) work with
   **zero changes to Marin's speedrun code** — everything calls `.encode()`.
   - Why this works: Marin's `load_tokenizer` accepts a local dir and uses
     `AutoTokenizer.from_pretrained(..., trust_remote_code=True)` — custom
     tokenizer classes are already supported. Tokenizers flow as `str`
     (`_get_tokenizer_for_train -> str`); a local dir path satisfies that.
   - Fallback (if reviewers dislike `trust_remote_code`): pre-transform the
     datasets with an ExecutorStep + a plain-BPE tokenizer, AND fork
     `speedrun_paloma_tokenized` to point at pre-transformed Paloma. Messier —
     forks more of the speedrun path. Prefer the custom-tokenizer route.
3. **Caches** — `default_tokenize(dataset=FineWeb-EDU, tokenizer=<caseops dir>)`
   for train; the speedrun's own `speedrun_paloma_tokenized` handles eval.
   [task D.5, needs compute]
4. **BPB byte table — RESOLVED (D.2).** `byte_length_of_token` (in
   `levanter/utils/hf_utils.py`) returns **0 bytes for any token in
   `all_special_ids`**. So registering the 4 markers as special tokens (step 2)
   gives correct marker→0-byte accounting **for free, with zero Levanter
   changes**. Every non-marker token: `decode`→count UTF-8 bytes == its original
   byte count, because the only letter transform is ASCII upper↔lower (1↔1
   byte). BPB is exact by construction.
   - One build-time check: confirm marker positions still receive loss weight
     in the training/eval loss_fn (they MUST — predicting "when to capitalize"
     is real information the model should pay bits for). `all_special_ids`
     membership only affects `byte_length_of_token` and `decode`, not loss
     masking — BOS is special and is normally scored — but verify.
5. **SpeedrunConfig** — if vocab != llama3's, subclass `SpeedrunConfig` to
   override the hard-coded `vocab_size` property so FLOPs/param accounting is
   correct. Subclassing is clean (the "don't modify" warning is about editing
   `speedrun.py` itself). [task D.6]
6. **Runs** — baseline + control + treatment at one model config + compute
   budget; plot BPB vs training_hardware_flops. [task D.7, needs compute]

## Open questions / risks

- **R1 — CaseOps edge shrinks at large vocab.** Mitigated by the Pareto framing:
  we lead with the small-vocab FLOPs win, not a fixed-128K BPB claim.
- **R2 — tokenizer interface.** RESOLVED — see integration step 2.
- **R3 — marker byte accounting in Levanter.** RESOLVED — see integration step 4.
- **R4 — corpus shift.** PG CaseOps was tuned on FineWeb-10B; Marin's speedrun
  trains on FineWeb-EDU. Retrain the BPE on FineWeb-EDU; don't reuse the PG
  SP10240 model directly.
- **R5 — "cold drop" ≠ stealth.** Marin submissions are public PRs with
  OpenReview-style review. "Cold" = understated tone, no hype, let the frontier
  point speak. Develop privately; the drop itself is necessarily public.
- **R6 — loss weight on marker positions.** Build-time verify markers are scored
  (see integration step 4). If Levanter's loss_fn masks special tokens, BPB
  would be understated (markers = 0 bytes, so denominator unaffected, but
  marker-prediction bits would vanish from the numerator) — dishonest. Must
  confirm markers carry loss weight.

## What does NOT transfer from Parameter Golf

The int6 / LQER / per-group quantization and phased-TTT eval machinery — all
dead weight for Speedrun (no artifact-size cap, no quantization in scoring).
Only the tokenizer transform transfers.
