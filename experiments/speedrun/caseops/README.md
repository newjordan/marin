# CaseOps tokenizer — Marin Speedrun submission

A tokenizer submission. The idea: factor ASCII capitalization out of the BPE
vocabulary into four cheap, highly context-predictable marker tokens, so the
tokenizer can run at a smaller vocab without the usual bits-per-byte penalty.
On the Marin Speedrun's BPB-vs-FLOPs frontier, a smaller vocab means a smaller
embedding/unembedding — for small models that dominates FLOPs — so the bet is a
Pareto move: same-or-better BPB at meaningfully fewer FLOPs.

## What CaseOps is

A bijective, **ASCII-only** text transform applied before BPE
(`tokenization_caseops.py`, transform inlined from the Parameter Golf
`lossless_caps` lineage). Over maximal ASCII alphabetic runs:

- lowercase word → unchanged
- `TitleCase` → `TITLE + lower(word)`
- `ALLCAPS` (len ≥ 2) → `ALLCAPS + lower(word)`
- mixed case → `[TITLE?] + lower[0] + (CAPNEXT + lower[c] per upper c)`
- a literal marker char in the input → `ESC + char`
- **every non-ASCII character passes through untouched** — this is what makes
  losslessness airtight (no Unicode case-mapping hazards: ß, Turkish i, …)

`decode(encode(x)) == x` byte-exact, by construction. The 4 markers
(U+E001..U+E004) are registered as special tokens, so they are atomic to BPE
and Levanter's `byte_length_of_token` returns 0 for them — BPB stays computed
on **true original bytes** with no Levanter modification.

## Methodology — the Pareto triangle

`caseops_sweep.py` runs three arms with IDENTICAL model architecture and compute
budget; the only variable is the tokenizer:

| arm | tokenizer | role |
|---|---|---|
| baseline | marin-tokenizer (~128K) | the speedrun default |
| control | plain byte-level BPE at vocab N | isolates "smaller vocab" from "CaseOps" |
| treatment | CaseOps tokenizer at vocab N | the submission |

All three tokenize the **same** raw documents (`fineweb_edu_sample_10bt`), so the
comparison is not confounded by a different data selection. The control arm
isolates the variable: control vs treatment is purely "CaseOps on/off"; both vs
baseline shows the FLOPs axis.

## Files

- `tokenization_caseops.py` — the `CaseOpsTokenizer` (`trust_remote_code` HF
  tokenizer; transform + inner byte-level BPE; markers as special tokens)
- `caseops_sweep.py` — the 3-arm sweep (`build_config` per arm → `default_speedrun`)
- `submission_support.py` — vendored speedrun machinery (`SpeedrunConfig`,
  `default_speedrun`, results step). One edit vs upstream: `vocab_size` is an
  explicit per-arm field, not a hard-coded property, so each arm's FLOPs/param
  accounting is honest.
- `materialize_submission.py` — post-run: aggregates the 3 arms'
  `speedrun_results.json` → `selection_summary.json` + `bpb_vs_flops.png`

## Reproduce

```bash
# 1. set RESOURCES + finalize VOCAB/MODEL in caseops_sweep.py (see its docstring)
# 2. run the 3-arm sweep
HF_TOKEN=<token with gated-repo read>  HF_MODULES_CACHE=<writable dir> \
  uv run -m experiments.speedrun.caseops.caseops_sweep
# 3. aggregate + plot
uv run -m experiments.speedrun.caseops.materialize_submission --results-dir <prefix>/checkpoints/speedrun
```

## Preflight evidence (training-free)

Before any training run, a held-out n-gram conditional-entropy analysis of the
CaseOps vs plain tokenizers showed the BPB delta moving monotonically in
CaseOps' favour as context deepens: unigram +6%, bigram +0.9%, **trigram
−1.9%** (CaseOps ahead), stable across vocab 8K–64K. A real transformer has far
more context than a trigram — the basis for expecting the treatment arm to land
at-or-below the baseline BPB while sitting well left on FLOPs.

## Status

- [x] Transform losslessness verified (30k real docs, byte-exact)
- [x] Tokenizers built + `trust_remote_code` wrappers verified
- [x] Submission machinery validated end-to-end on a CPU smoke
- [x] Hardware-FLOPS accounting verified (incl. GB10)
- [ ] Tokenizers retrained on FineWeb-EDU (preflight tokenizers are FineWeb-10B-sample)
- [ ] The 3-arm runs (D.7) — pending
- [ ] `materialize_submission.py` run on real results
