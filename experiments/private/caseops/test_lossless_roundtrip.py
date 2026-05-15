# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Losslessness stress test for the CaseOps text transform.

This is the foundational correctness gate for the CaseOps Marin submission.
BPB is scored on TRUE original bytes — if the transform is not perfectly
bijective on real text, the whole submission is invalid (the byte denominator
would be wrong). So before anything else: prove decode(encode(x)) == x,
byte-exact, on adversarial cases AND on a large sample of real FineWeb text.

`lossless_caps.py` is copied verbatim from the Parameter Golf SOTA lineage
(sha256 963626d7... — matches the Mockingbird run's manifest). We do NOT modify
it; we only test it here in the Marin context.

Run (CPU, no GPU, no Marin internals needed):
    .venv/bin/python -m experiments.private.caseops.test_lossless_roundtrip
    # optional: point at a real corpus and set sample size
    CASEOPS_CORPUS=/home/frosty40/parameter-golf-lab/data/docs_selected.jsonl \
    CASEOPS_SAMPLE=20000 \
        .venv/bin/python -m experiments.private.caseops.test_lossless_roundtrip
"""

from __future__ import annotations

import json
import os
import sys

from experiments.private.caseops.lossless_caps import (
    DEFAULT_V2_ALLCAPS,
    DEFAULT_V2_CAPNEXT,
    DEFAULT_V2_ESC,
    DEFAULT_V2_TITLE,
    LOSSLESS_CAPS_CASEOPS_V1,
    decode_lossless_caps_v2,
    encode_lossless_caps_v2,
    get_text_inverse_transform,
    get_text_transform,
    get_text_transform_control_symbols,
)

# The 4 PUA control chars used by the caseops_v1 (== v2) scheme:
# TITLE=U+E001, ALLCAPS=U+E002, CAPNEXT=U+E003, ESC=U+E004.
TITLE = DEFAULT_V2_TITLE
ALLCAPS = DEFAULT_V2_ALLCAPS
CAPNEXT = DEFAULT_V2_CAPNEXT
ESC = DEFAULT_V2_ESC

# Hand-crafted adversarial cases: each must round-trip byte-exact.
ADVERSARIAL: list[str] = [
    "",
    "a",
    "A",
    "the cat",
    "The Cat",
    "THE CAT",
    "tHe CaT",  # mixed case -> capnext path
    "NASA and the FBI",
    "iPhone",  # internal cap
    "McDonald's",  # internal cap + apostrophe
    "ALLCAPS123lower",  # caps run, digits, lower (digits break the alpha run)
    "Hello, World!\nSecond LINE here.",
    "e=mc^2 AND F=ma",
    "  leading and trailing  ",
    "\tTabbed\tAND\tCAPPED",
    # Non-ASCII must pass through untouched:
    "cafe RESUME naive",
    "accented: Éé Üü Ññ MiXeD",
    "Straße ß SS",  # German sharp s
    "日本語 ENGLISH 混在",
    "emoji \U0001f600 TEST \U0001f389 done",
    "Ⅸ Roman NUMERAL Ⅻ",  # unicode numerals
    # Literal control chars in input MUST be escaped and survive:
    f"literal title marker: {TITLE} here",
    f"all four: {TITLE}{ALLCAPS}{CAPNEXT}{ESC}",
    f"{ESC}{ESC}{ESC} escapes",
    f"CAPS then {ALLCAPS} marker then CAPS",
    # Pathological boundaries:
    "A" * 1000,
    "aA" * 500,
    ("The QUICK brown FOX. " * 50),
    "\x00null\x00bytes\x00",  # NUL bytes
    "\r\n\r\nwindows NEWLINES\r\n",
]


def _check(text: str) -> tuple[bool, str]:
    """Return (ok, detail). ok iff decode(encode(text)) == text byte-exact."""
    try:
        enc = encode_lossless_caps_v2(text)
        dec = decode_lossless_caps_v2(enc)
    except Exception as exc:
        return False, f"exception: {type(exc).__name__}: {exc}"
    if dec != text:
        i = next(
            (k for k in range(min(len(dec), len(text))) if dec[k] != text[k]),
            min(len(dec), len(text)),
        )
        return False, f"mismatch at char {i}: orig={text[i : i + 20]!r} got={dec[i : i + 20]!r}"
    if dec.encode("utf-8") != text.encode("utf-8"):
        return False, "str-equal but bytes differ (should be impossible)"
    return True, ""


def _check_registry() -> tuple[bool, str]:
    """The name-based registry must resolve caseops_v1 to the v2 transform pair."""
    fwd = get_text_transform(LOSSLESS_CAPS_CASEOPS_V1)
    inv = get_text_inverse_transform(LOSSLESS_CAPS_CASEOPS_V1)
    syms = get_text_transform_control_symbols(LOSSLESS_CAPS_CASEOPS_V1)
    if syms != [TITLE, ALLCAPS, CAPNEXT, ESC]:
        return False, f"control symbols mismatch: {syms!r}"
    sample = "The QUICK brown FOX jumped"
    if inv(fwd(sample)) != sample:
        return False, "registry fwd/inv round-trip failed"
    return True, ""


def main() -> int:
    failures: list[tuple[str, str]] = []

    ok, detail = _check_registry()
    print(f"[registry] {'ok' if ok else 'FAIL: ' + detail}")
    if not ok:
        failures.append(("<registry>", detail))

    adv_fail = 0
    for case in ADVERSARIAL:
        ok, detail = _check(case)
        if not ok:
            adv_fail += 1
            failures.append((repr(case[:40]), detail))
    print(f"[adversarial] {len(ADVERSARIAL) - adv_fail}/{len(ADVERSARIAL)} passed")

    corpus = os.environ.get("CASEOPS_CORPUS", "/home/frosty40/parameter-golf-lab/data/docs_selected.jsonl")
    sample_n = int(os.environ.get("CASEOPS_SAMPLE", "20000"))
    if os.path.isfile(corpus):
        checked = 0
        corpus_fail = 0
        total_bytes = 0
        with open(corpus, "r", encoding="utf-8") as fh:
            for line in fh:
                if checked >= sample_n:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = obj["text"] if isinstance(obj, dict) else obj
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
                ok, detail = _check(text)
                checked += 1
                total_bytes += len(text.encode("utf-8"))
                if not ok:
                    corpus_fail += 1
                    if corpus_fail <= 5:
                        failures.append((f"<corpus doc {checked}>", detail))
        print(
            f"[corpus] {checked - corpus_fail}/{checked} docs passed "
            f"({total_bytes / 1e6:.1f} MB of real text round-tripped)"
        )
    else:
        print(f"[corpus] SKIPPED - corpus not found at {corpus}")

    print()
    if failures:
        print(f"FAILED: {len(failures)} failing case(s):")
        for name, detail in failures[:20]:
            print(f"  {name}: {detail}")
        return 1
    print("LOSSLESS ROUND-TRIP TEST PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
