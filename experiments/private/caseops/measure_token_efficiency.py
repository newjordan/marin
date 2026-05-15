# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Training-free leading indicator: does CaseOps actually compress better?

The real proof (lower eval BPB at matched FLOPs) needs a pod. But three
training-free signals, measured on HELD-OUT text, tell us whether it's worth
renting one:

  1. tokens-per-byte / bytes-per-token  — raw compression ratio. Fewer tokens
     for the same original bytes = fewer prediction steps.
  2. unigram BPB estimate  — H(token distribution) / bytes_per_token. A
     context-free lower-bound-ish estimate of the BPB floor each tokenizer
     imposes. NOT the real BPB (the model uses context) but a principled,
     honest training-free proxy. CaseOps lower than plain = real signal.
  3. uppercase-vocab fraction  — what share of each vocab is spent on tokens
     whose surface contains an ASCII capital. This is the *mechanism*: the
     plain BPE burns slots on "The"/"NASA"/" New" case variants; CaseOps
     factors case into 4 markers and frees those slots.

All measured on original-byte denominators so the comparison is honest.

Run (CPU, after build_tokenizer.py has produced the two tokenizer dirs):
    .venv/bin/python -m experiments.private.caseops.measure_token_efficiency \
        --caseops-dir experiments/private/caseops/artifacts/caseops_48k \
        --plain-dir   experiments/private/caseops/artifacts/plain_48k \
        --skip 150000 --eval-docs 10000
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from collections import Counter

from experiments.private.caseops.lossless_caps import encode_lossless_caps_v2

DEFAULT_CORPUS = "/home/frosty40/parameter-golf-lab/data/docs_selected.jsonl"


def _iter_heldout(corpus: pathlib.Path, skip: int, n: int):
    """Yield raw doc strings AFTER the first `skip` docs (held out from training)."""
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


def _uppercase_vocab_fraction(tok) -> tuple[int, int, float]:
    """(n_upper, vocab_size, fraction) — tokens whose decoded surface has ASCII A-Z."""
    vocab_size = tok.vocab_size
    special = set(tok.all_special_ids)
    n_upper = 0
    for tid in range(vocab_size):
        if tid in special:
            continue
        surface = tok.decode([tid])
        if any("A" <= ch <= "Z" for ch in surface):
            n_upper += 1
    return n_upper, vocab_size, n_upper / max(vocab_size, 1)


def _measure(tok, docs: list[str], caseops: bool) -> dict:
    """Encode held-out docs; return token/byte/entropy stats on ORIGINAL bytes."""
    total_tokens = 0
    total_orig_bytes = 0
    freq: Counter[int] = Counter()
    for raw in docs:
        orig_bytes = len(raw.encode("utf-8"))
        feed = encode_lossless_caps_v2(raw) if caseops else raw
        ids = tok.encode(feed, add_special_tokens=False)
        total_tokens += len(ids)
        total_orig_bytes += orig_bytes
        freq.update(ids)

    tokens_per_byte = total_tokens / max(total_orig_bytes, 1)
    bytes_per_token = total_orig_bytes / max(total_tokens, 1)
    # unigram entropy in bits/token
    h = 0.0
    for c in freq.values():
        p = c / total_tokens
        h -= p * math.log2(p)
    unigram_bpb = h / max(bytes_per_token, 1e-9)
    return {
        "total_tokens": total_tokens,
        "total_orig_bytes": total_orig_bytes,
        "tokens_per_byte": tokens_per_byte,
        "bytes_per_token": bytes_per_token,
        "unigram_entropy_bits_per_token": h,
        "unigram_bpb_estimate": unigram_bpb,
        "distinct_tokens_used": len(freq),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--caseops-dir", type=pathlib.Path, required=True)
    ap.add_argument("--plain-dir", type=pathlib.Path, required=True)
    ap.add_argument("--corpus", type=pathlib.Path, default=pathlib.Path(DEFAULT_CORPUS))
    ap.add_argument("--skip", type=int, default=150000, help="Docs to skip (the training set)")
    ap.add_argument("--eval-docs", type=int, default=10000, help="Held-out docs to measure on")
    args = ap.parse_args()

    for d in (args.caseops_dir, args.plain_dir):
        if not d.is_dir():
            print(f"FAIL: tokenizer dir not found: {d}")
            return 1
    if not args.corpus.is_file():
        print(f"FAIL: corpus not found: {args.corpus}")
        return 1

    from transformers import AutoTokenizer

    print("loading tokenizers...", flush=True)
    caseops_tok = AutoTokenizer.from_pretrained(str(args.caseops_dir))
    plain_tok = AutoTokenizer.from_pretrained(str(args.plain_dir))

    print(f"reading {args.eval_docs} held-out docs (skip {args.skip})...", flush=True)
    docs = list(_iter_heldout(args.corpus, args.skip, args.eval_docs))
    print(f"  got {len(docs)} docs, {sum(len(d.encode('utf-8')) for d in docs) / 1e6:.1f} MB", flush=True)

    print("measuring plain...", flush=True)
    plain = _measure(plain_tok, docs, caseops=False)
    print("measuring caseops...", flush=True)
    caseops = _measure(caseops_tok, docs, caseops=True)

    print("vocab case-fragmentation...", flush=True)
    p_up = _uppercase_vocab_fraction(plain_tok)
    c_up = _uppercase_vocab_fraction(caseops_tok)

    def _pct(new, old):
        return (new - old) / old * 100.0

    print()
    print("=" * 64)
    print(f"{'metric':<34}{'plain':>14}{'caseops':>14}")
    print("-" * 64)
    print(f"{'tokens / orig byte':<34}{plain['tokens_per_byte']:>14.5f}{caseops['tokens_per_byte']:>14.5f}")
    print(f"{'orig bytes / token':<34}{plain['bytes_per_token']:>14.4f}{caseops['bytes_per_token']:>14.4f}")
    print(
        f"{'unigram entropy (bits/token)':<34}{plain['unigram_entropy_bits_per_token']:>14.4f}"
        f"{caseops['unigram_entropy_bits_per_token']:>14.4f}"
    )
    print(f"{'UNIGRAM BPB ESTIMATE':<34}{plain['unigram_bpb_estimate']:>14.5f}{caseops['unigram_bpb_estimate']:>14.5f}")
    print(f"{'distinct tokens used':<34}{plain['distinct_tokens_used']:>14d}{caseops['distinct_tokens_used']:>14d}")
    print(f"{'vocab w/ ASCII uppercase':<34}{p_up[0]:>14d}{c_up[0]:>14d}")
    print(f"{'  ... as fraction of vocab':<34}{p_up[2]:>14.4f}{c_up[2]:>14.4f}")
    print("-" * 64)
    print(f"{'tokens/byte delta':<34}{_pct(caseops['tokens_per_byte'], plain['tokens_per_byte']):>+13.2f}%")
    print(f"{'unigram BPB delta':<34}{_pct(caseops['unigram_bpb_estimate'], plain['unigram_bpb_estimate']):>+13.2f}%")
    print("=" * 64)
    print()
    print("READ: negative deltas = CaseOps compresses/encodes better (leading signal).")
    print("This is training-free and context-free — the real BPB proof still needs a pod run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
