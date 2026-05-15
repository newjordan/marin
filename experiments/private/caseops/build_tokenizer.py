# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train a BPE tokenizer for the CaseOps Marin submission (preflight).

Uses `PreTrainedTokenizerFast.train_new_from_iterator` on the marin-community
baseline tokenizer. This keeps the baseline's exact pipeline — the GPT-4-style
Split pre-tokenizer regex, ByteLevel, the BPE model, the decoder — and only
retrains the BPE vocab at the target size. So the control arm is, literally,
"the Marin baseline tokenizer architecture, retrained at vocab N":
the ONLY variable between control and treatment is the CaseOps transform.

  --caseops : apply encode_lossless_caps_v2 to every training doc, and register
              the 4 marker chars (U+E001..U+E004) as special tokens (so BPE
              keeps them atomic AND Levanter's byte_length_of_token returns 0
              for them -> honest BPB, zero Levanter changes).
  (no flag) : plain BPE on untransformed text -- the Pareto-triangle control.

Produces a real HF tokenizer directory (save_pretrained), directly loadable by
Marin's load_tokenizer. NOTE: this trained tokenizer expects ALREADY-TRANSFORMED
text on .encode() for the caseops case -- the raw-text transform is wired in by
the trust_remote_code wrapper class (next task, D.4 part 2 / DESIGN.md step 2).

PREFLIGHT scope: trains on a sample of local FineWeb text (docs_selected.jsonl)
to validate the mechanism and measure token efficiency. The FINAL submission
tokenizer must be retrained on FineWeb-EDU (Marin's speedrun corpus) -- DESIGN R4.

Run (CPU, no GPU, no pod):
    .venv/bin/python -m experiments.private.caseops.build_tokenizer \
        --caseops --vocab 48000 --docs 150000 \
        --out experiments/private/caseops/artifacts/caseops_48k
    .venv/bin/python -m experiments.private.caseops.build_tokenizer \
        --vocab 48000 --docs 150000 \
        --out experiments/private/caseops/artifacts/plain_48k
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from experiments.private.caseops.lossless_caps import (
    DEFAULT_V2_ALLCAPS,
    DEFAULT_V2_CAPNEXT,
    DEFAULT_V2_ESC,
    DEFAULT_V2_TITLE,
    encode_lossless_caps_v2,
)

CASEOPS_MARKERS = [DEFAULT_V2_TITLE, DEFAULT_V2_ALLCAPS, DEFAULT_V2_CAPNEXT, DEFAULT_V2_ESC]
MARKER_NAMES = ["TITLE", "ALLCAPS", "CAPNEXT", "ESC"]
DEFAULT_CORPUS = "/home/frosty40/parameter-golf-lab/data/docs_selected.jsonl"
BASE_TOKENIZER = "marin-community/marin-tokenizer"


def _iter_text(corpus: pathlib.Path, max_docs: int, caseops: bool):
    """Yield (optionally CaseOps-transformed) doc strings from a jsonl corpus."""
    n = 0
    with corpus.open("r", encoding="utf-8") as fh:
        for line in fh:
            if n >= max_docs:
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
            n += 1
            yield encode_lossless_caps_v2(text) if caseops else text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--caseops", action="store_true", help="Apply CaseOps transform + register markers")
    ap.add_argument("--vocab", type=int, default=48000, help="Target vocab size")
    ap.add_argument("--docs", type=int, default=150000, help="Number of docs to train on")
    ap.add_argument("--corpus", type=pathlib.Path, default=pathlib.Path(DEFAULT_CORPUS))
    ap.add_argument("--out", type=pathlib.Path, required=True, help="Output dir for the HF tokenizer")
    args = ap.parse_args()

    if not args.corpus.is_file():
        print(f"FAIL: corpus not found: {args.corpus}")
        return 1

    from transformers import AutoTokenizer

    print(f"loading base tokenizer: {BASE_TOKENIZER}", flush=True)
    base = AutoTokenizer.from_pretrained(BASE_TOKENIZER)

    extra = {}
    if args.caseops:
        extra["new_special_tokens"] = CASEOPS_MARKERS

    print(
        f"training {'CaseOps' if args.caseops else 'plain'} BPE via train_new_from_iterator: "
        f"vocab={args.vocab} docs={args.docs}",
        flush=True,
    )
    new_tok = base.train_new_from_iterator(
        _iter_text(args.corpus, args.docs, args.caseops),
        vocab_size=args.vocab,
        **extra,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    new_tok.save_pretrained(str(args.out))
    print(f"saved HF tokenizer -> {args.out}  (vocab_size={new_tok.vocab_size})", flush=True)

    # Verify markers landed as special tokens (caseops only) -- this is what gives
    # us the free 0-byte BPB accounting (D.2).
    if args.caseops:
        special_ids = set(new_tok.all_special_ids)
        for name, marker in zip(MARKER_NAMES, CASEOPS_MARKERS):
            mid = new_tok.convert_tokens_to_ids(marker)
            is_special = mid in special_ids
            ok = mid is not None and mid != new_tok.unk_token_id and is_special
            print(f"  marker {name} (U+{ord(marker):04X}): id={mid} special={is_special} {'OK' if ok else 'FAIL'}")
            if not ok:
                print("FAIL: a CaseOps marker is missing or not registered as special")
                return 1

    print("build OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
