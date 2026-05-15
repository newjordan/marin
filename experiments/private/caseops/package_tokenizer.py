# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Assemble + verify a trust_remote_code CaseOps wrapper tokenizer directory.

Takes an inner tokenizer dir (a byte-level BPE trained on CaseOps-transformed
text, produced by build_tokenizer.py --caseops) and packages it as a
trust_remote_code HF tokenizer dir:

  <out>/
    tokenizer.json              (inner BPE, verbatim)
    special_tokens_map.json     (inner, verbatim)
    chat_template.jinja         (inner, verbatim, if present)
    tokenization_caseops.py     (the self-contained CaseOpsTokenizer class)
    tokenizer_config.json       (inner config + auto_map + tokenizer_class)

Then verifies the assembled dir:
  - AutoTokenizer.from_pretrained(out, trust_remote_code=True) loads it
  - .encode(raw) == inner.encode(caseops_transform(raw))   [transform wired in]
  - .decode(.encode(raw)) == raw, byte-exact                [round-trips]
  - the 4 markers are in all_special_ids
  - Levanter's byte_length_of_token: 0 for markers, correct >0 for normal tokens

Run (CPU, no pod):
    .venv/bin/python -m experiments.private.caseops.package_tokenizer \
        --inner experiments/private/caseops/artifacts/caseops_48k \
        --out   experiments/private/caseops/artifacts/caseops_48k_wrapped
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

_HERE = pathlib.Path(__file__).resolve().parent
TOKENIZATION_FILE = _HERE / "tokenization_caseops.py"

VERIFY_TEXTS = [
    "The NASA Launch was a SUCCESS.",
    "iPhone, McDonald's, and plain lowercase words.",
    "e=mc^2 AND F=ma -- mixed CONTENT here\nsecond line",
    "accented cafe RESUME, unicode 日本 TEST, emoji \U0001f600 done",
    "ALLCAPS THEN lower then Title Case Words",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inner", type=pathlib.Path, required=True, help="Inner BPE tokenizer dir (caseops-trained)")
    ap.add_argument("--out", type=pathlib.Path, required=True, help="Output wrapper tokenizer dir")
    args = ap.parse_args()

    if not args.inner.is_dir():
        print(f"FAIL: inner tokenizer dir not found: {args.inner}")
        return 1
    if not (args.inner / "tokenizer.json").is_file():
        print(f"FAIL: {args.inner}/tokenizer.json missing")
        return 1
    if not TOKENIZATION_FILE.is_file():
        print(f"FAIL: {TOKENIZATION_FILE} missing")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)

    # 1. Copy inner artifacts verbatim.
    for fname in ("tokenizer.json", "special_tokens_map.json", "chat_template.jinja"):
        src = args.inner / fname
        if src.is_file():
            shutil.copy2(src, args.out / fname)

    # 2. Copy the self-contained custom class.
    shutil.copy2(TOKENIZATION_FILE, args.out / "tokenization_caseops.py")

    # 3. tokenizer_config.json = inner config + auto_map + tokenizer_class.
    cfg_path = args.inner / "tokenizer_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
    cfg["tokenizer_class"] = "CaseOpsTokenizer"
    cfg["auto_map"] = {"AutoTokenizer": [None, "tokenization_caseops.CaseOpsTokenizer"]}
    (args.out / "tokenizer_config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"assembled wrapper tokenizer dir -> {args.out}", flush=True)

    # ============================ VERIFY ============================
    from transformers import AutoTokenizer

    from experiments.private.caseops.lossless_caps import encode_lossless_caps_v2

    print("verifying...", flush=True)
    inner = AutoTokenizer.from_pretrained(str(args.inner))
    wrapped = AutoTokenizer.from_pretrained(str(args.out), trust_remote_code=True)

    fails: list[str] = []

    # type
    if type(wrapped).__name__ != "CaseOpsTokenizer":
        fails.append(f"loaded class is {type(wrapped).__name__}, expected CaseOpsTokenizer")

    # markers special
    from experiments.private.caseops.tokenization_caseops import CASEOPS_MARKERS

    special_ids = set(wrapped.all_special_ids)
    for i, m in enumerate(CASEOPS_MARKERS):
        mid = wrapped.convert_tokens_to_ids(m)
        if mid is None or mid not in special_ids:
            fails.append(f"marker {i} (U+{ord(m):04X}) id={mid} not in all_special_ids")

    # transform wired into encode: wrapped.encode(raw) == inner.encode(transform(raw))
    for t in VERIFY_TEXTS:
        w_ids = wrapped.encode(t, add_special_tokens=False)
        i_ids = inner.encode(encode_lossless_caps_v2(t), add_special_tokens=False)
        if w_ids != i_ids:
            fails.append(f"encode mismatch on {t[:30]!r}: wrapped {len(w_ids)} ids vs inner {len(i_ids)} ids")

    # round-trip: decode(encode(raw)) == raw, byte-exact
    for t in VERIFY_TEXTS:
        rt = wrapped.decode(wrapped.encode(t, add_special_tokens=False), skip_special_tokens=False)
        if rt != t:
            fails.append(f"round-trip mismatch on {t[:30]!r}: got {rt[:40]!r}")

    # batch path (__call__) also applies the transform
    batch = wrapped(VERIFY_TEXTS, add_special_tokens=False)["input_ids"]
    for t, ids in zip(VERIFY_TEXTS, batch):
        if ids != inner.encode(encode_lossless_caps_v2(t), add_special_tokens=False):
            fails.append(f"batch __call__ mismatch on {t[:30]!r}")

    # Levanter byte accounting
    try:
        from levanter.utils.hf_utils import byte_length_of_token

        # markers -> 0 bytes
        for m in CASEOPS_MARKERS:
            mid = wrapped.convert_tokens_to_ids(m)
            bl = byte_length_of_token(wrapped, mid)
            if bl != 0:
                fails.append(f"byte_length_of_token(marker U+{ord(m):04X})={bl}, expected 0")
        # a handful of normal tokens -> positive, sane
        sample_ids = [i for i in range(300, 320)]
        bls = [byte_length_of_token(wrapped, i) for i in sample_ids]
        if not all(b >= 0 for b in bls) or all(b == 0 for b in bls):
            fails.append(f"byte_length_of_token on normal tokens looks wrong: {bls}")
        else:
            print(f"  byte_length_of_token: markers->0 OK, sample normal tokens -> {bls[:8]}...")
    except Exception as exc:
        fails.append(f"byte_length_of_token check errored: {type(exc).__name__}: {exc}")

    print()
    if fails:
        print(f"VERIFY FAILED ({len(fails)}):")
        for f in fails:
            print(f"  - {f}")
        return 1
    print(f"VERIFY PASSED — {args.out} is a working trust_remote_code CaseOps tokenizer.")
    print("  encode(raw) does transform->BPE; decode round-trips byte-exact;")
    print("  markers are special (0-byte in Levanter BPB); __call__/batch paths OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
