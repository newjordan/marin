# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Context-aware BPB proxy: held-out n-gram conditional cross-entropy.

The unigram BPB estimate (measure_token_efficiency.py) is structurally unfair to
CaseOps: it charges full context-free entropy for the 4 marker tokens, but
CaseOps' whole thesis is that markers are CHEAP GIVEN CONTEXT (TITLE almost
always follows sentence-end / precedes a proper noun). The unigram proxy cannot
see that. A bigram model can — it has one token of context, enough to make
"predict TITLE after a period" nearly free.

This script builds unigram + bigram models on a CONTEXT corpus and evaluates
held-out cross-entropy on a separate eval corpus, for both the CaseOps and the
plain tokenizer. BPB = total_bits / total_ORIGINAL_bytes (honest denominator).

Identical methodology for both arms — only the caseops-vs-plain DELTA matters,
not the absolute numbers. Bigram is interpolated with unigram (fixed lambda) and
unigram is add-1 smoothed, so unseen contexts/tokens are handled.

Decision use: if the bigram BPB delta is still clearly positive (CaseOps worse),
the unigram result stands and it's a NO-GO. If bigram flips it to <=0 or
near-parity at small vocab, context IS recovering the marker cost — that's the
regime worth a real (pod) run.

Run (CPU):
    .venv/bin/python -m experiments.private.caseops.measure_ngram_entropy \
        --caseops-dir experiments/private/caseops/artifacts/caseops_8000 \
        --plain-dir   experiments/private/caseops/artifacts/plain_8000
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from collections import Counter, defaultdict

from experiments.private.caseops.lossless_caps import encode_lossless_caps_v2

DEFAULT_CORPUS = "/home/frosty40/parameter-golf-lab/data/docs_selected.jsonl"
LAMBDA = 0.9  # bigram interpolation weight: P = lam*P_bigram + (1-lam)*P_unigram
LAMBDA3 = 0.9  # trigram interpolation: P = lam3*P_trigram + (1-lam3)*P_bigram_interp


def _iter_docs(corpus: pathlib.Path, skip: int, n: int):
    seen = 0
    yielded = 0
    with corpus.open("r", encoding="utf-8") as fh:
        for line in fh:
            if yielded >= n:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj["text"] if isinstance(obj, dict) else obj
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            text = text.replace("\x00", " ").strip()
            if not text:
                continue
            seen += 1
            if seen <= skip:
                continue
            yielded += 1
            yield text


def _encode_docs(tok, docs, caseops):
    """Return list of token-id lists (one per doc)."""
    out = []
    for raw in docs:
        feed = encode_lossless_caps_v2(raw) if caseops else raw
        out.append(tok.encode(feed, add_special_tokens=False))
    return out


def _measure(tok, ctx_docs, eval_docs, caseops, vocab_size):
    """Build uni+bigram on ctx, evaluate held-out cross-entropy on eval. Honest bytes."""
    ctx = _encode_docs(tok, ctx_docs, caseops)
    uni = Counter()
    bi_ctx = Counter()  # c(prev)
    bi = defaultdict(Counter)  # c(prev -> tok)
    tri_ctx = Counter()  # c(prev2, prev)
    tri = defaultdict(Counter)  # c((prev2, prev) -> tok)
    for ids in ctx:
        prev2 = prev = None
        for t in ids:
            uni[t] += 1
            if prev is not None:
                bi_ctx[prev] += 1
                bi[prev][t] += 1
            if prev2 is not None:
                tri_ctx[(prev2, prev)] += 1
                tri[(prev2, prev)][t] += 1
            prev2, prev = prev, t
    total_uni = sum(uni.values())
    # add-1 smoothed unigram over the full vocab
    denom_uni = total_uni + vocab_size

    def p_unigram(t):
        return (uni.get(t, 0) + 1) / denom_uni

    def p_bigram(prev, t):
        if prev is None or bi_ctx.get(prev, 0) == 0:
            return p_unigram(t)
        p_bi = bi[prev].get(t, 0) / bi_ctx[prev]
        return LAMBDA * p_bi + (1.0 - LAMBDA) * p_unigram(t)

    def p_trigram(prev2, prev, t):
        p_bi = p_bigram(prev, t)
        key = (prev2, prev)
        if prev2 is None or tri_ctx.get(key, 0) == 0:
            return p_bi
        p_tri = tri[key].get(t, 0) / tri_ctx[key]
        return LAMBDA3 * p_tri + (1.0 - LAMBDA3) * p_bi

    # evaluate held-out
    total_orig_bytes = sum(len(d.encode("utf-8")) for d in eval_docs)
    ev = _encode_docs(tok, eval_docs, caseops)
    bits_uni = 0.0
    bits_bi = 0.0
    bits_tri = 0.0
    n_tok = 0
    for ids in ev:
        prev2 = prev = None
        for t in ids:
            bits_uni += -math.log2(p_unigram(t))
            bits_bi += -math.log2(p_bigram(prev, t))
            bits_tri += -math.log2(p_trigram(prev2, prev, t))
            n_tok += 1
            prev2, prev = prev, t
    return {
        "n_tok": n_tok,
        "orig_bytes": total_orig_bytes,
        "unigram_bpb": bits_uni / max(total_orig_bytes, 1),
        "bigram_bpb": bits_bi / max(total_orig_bytes, 1),
        "trigram_bpb": bits_tri / max(total_orig_bytes, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--caseops-dir", type=pathlib.Path, required=True)
    ap.add_argument("--plain-dir", type=pathlib.Path, required=True)
    ap.add_argument("--corpus", type=pathlib.Path, default=pathlib.Path(DEFAULT_CORPUS))
    ap.add_argument("--skip", type=int, default=150000, help="Docs to skip (the tokenizer training set)")
    ap.add_argument("--ctx-docs", type=int, default=25000, help="Context corpus size (n-gram counts)")
    ap.add_argument("--eval-docs", type=int, default=5000, help="Held-out eval corpus size")
    args = ap.parse_args()

    for d in (args.caseops_dir, args.plain_dir):
        if not d.is_dir():
            print(f"FAIL: tokenizer dir not found: {d}")
            return 1

    from transformers import AutoTokenizer

    caseops_tok = AutoTokenizer.from_pretrained(str(args.caseops_dir))
    plain_tok = AutoTokenizer.from_pretrained(str(args.plain_dir))

    all_docs = list(_iter_docs(args.corpus, args.skip, args.ctx_docs + args.eval_docs))
    ctx_docs = all_docs[: args.ctx_docs]
    eval_docs = all_docs[args.ctx_docs :]
    print(
        f"ctx={len(ctx_docs)} docs, eval={len(eval_docs)} docs "
        f"({sum(len(d.encode('utf-8')) for d in eval_docs) / 1e6:.1f} MB held-out)",
        flush=True,
    )

    plain = _measure(plain_tok, ctx_docs, eval_docs, False, plain_tok.vocab_size)
    caseops = _measure(caseops_tok, ctx_docs, eval_docs, True, caseops_tok.vocab_size)

    def _pct(new, old):
        return (new - old) / old * 100.0

    print()
    print("=" * 58)
    print(f"{'estimate':<22}{'plain':>12}{'caseops':>12}{'delta':>12}")
    print("-" * 58)
    for label, key in [("unigram BPB", "unigram_bpb"), ("bigram BPB", "bigram_bpb"), ("trigram BPB", "trigram_bpb")]:
        print(f"{label:<22}{plain[key]:>12.5f}{caseops[key]:>12.5f}{_pct(caseops[key], plain[key]):>+11.2f}%")
    print("=" * 58)
    print()
    print("READ: watch the delta trend across uni->bi->tri. If it keeps closing")
    print("toward/below zero, deeper (transformer) context recovers the marker")
    print("cost -> GO regime. If it flattens at a positive floor -> weak case.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
