# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CaseOpsTokenizer — a trust_remote_code HF tokenizer for the Marin CaseOps run.

This is the KEYSTONE artifact of the CaseOps submission. It wraps:
  - the lossless CaseOps text transform (encode/decode_lossless_caps_v2), and
  - an inner byte-level BPE trained on CaseOps-transformed text (the dir's
    tokenizer.json), with the 4 marker chars registered as special tokens.

so that `.encode(raw_text)` does transform -> BPE automatically and
`.decode(ids)` does BPE -> inverse-transform. Because everything funnels
through standard tokenizer methods, BOTH Marin's training tokenization AND the
speedrun's Paloma eval re-tokenization work with ZERO changes to Marin's
speedrun code.

The CaseOps transform is INLINED below (copied verbatim from lossless_caps.py,
sha256 963626d7..., the Parameter Golf SOTA lineage) so this file is fully
self-contained — required for robust trust_remote_code loading.

BPB accounting: the 4 markers are special tokens, so Levanter's
byte_length_of_token returns 0 for them; every other token's decoded byte
length equals its true original byte count (ASCII upper<->lower is 1<->1 byte).
BPB is exact on TRUE original bytes with no Levanter modification.
"""

from __future__ import annotations

from transformers import PreTrainedTokenizerFast

# ======================================================================
# INLINED: lossless CaseOps transform (verbatim from lossless_caps.py).
# Only touches ASCII A-Z; all non-ASCII passes through untouched.
# decode(encode(x)) == x byte-exact, by construction.
# ======================================================================

DEFAULT_V2_TITLE = ""
DEFAULT_V2_ALLCAPS = ""
DEFAULT_V2_CAPNEXT = ""
DEFAULT_V2_ESC = ""
CASEOPS_MARKERS = [DEFAULT_V2_TITLE, DEFAULT_V2_ALLCAPS, DEFAULT_V2_CAPNEXT, DEFAULT_V2_ESC]


class LosslessCapsError(ValueError):
    """Raised when a transformed string is malformed."""


def _is_ascii_upper(ch: str) -> bool:
    return "A" <= ch <= "Z"


def _is_ascii_lower(ch: str) -> bool:
    return "a" <= ch <= "z"


def _is_ascii_alpha(ch: str) -> bool:
    return _is_ascii_lower(ch) or _is_ascii_upper(ch)


def _validate_distinct_single_chars(*chars: str) -> None:
    if any(len(ch) != 1 for ch in chars):
        raise ValueError("all control characters must be exactly one character")
    if len(set(chars)) != len(chars):
        raise ValueError("control characters must be distinct")


def encode_lossless_caps_v2(
    text: str,
    *,
    title: str = DEFAULT_V2_TITLE,
    allcaps: str = DEFAULT_V2_ALLCAPS,
    capnext: str = DEFAULT_V2_CAPNEXT,
    esc: str = DEFAULT_V2_ESC,
) -> str:
    """Encode ASCII word capitalization with cheap word-level markers."""
    _validate_distinct_single_chars(title, allcaps, capnext, esc)
    controls = {title, allcaps, capnext, esc}
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in controls:
            out.append(esc)
            out.append(ch)
            i += 1
            continue
        if not _is_ascii_alpha(ch):
            out.append(ch)
            i += 1
            continue

        j = i + 1
        while j < n and _is_ascii_alpha(text[j]):
            j += 1
        word = text[i:j]
        lower_word = word.lower()

        if word.islower():
            out.append(word)
        elif len(word) >= 2 and word.isupper():
            out.append(allcaps)
            out.append(lower_word)
        elif _is_ascii_upper(word[0]) and word[1:].islower():
            out.append(title)
            out.append(lower_word)
        else:
            if _is_ascii_upper(word[0]):
                out.append(title)
            out.append(lower_word[0])
            for orig_ch, lower_ch in zip(word[1:], lower_word[1:], strict=True):
                if _is_ascii_upper(orig_ch):
                    out.append(capnext)
                out.append(lower_ch)
        i = j
    return "".join(out)


def decode_lossless_caps_v2(
    text: str,
    *,
    title: str = DEFAULT_V2_TITLE,
    allcaps: str = DEFAULT_V2_ALLCAPS,
    capnext: str = DEFAULT_V2_CAPNEXT,
    esc: str = DEFAULT_V2_ESC,
) -> str:
    """Decode the `lossless_caps_v2` transform back to the original text."""
    _validate_distinct_single_chars(title, allcaps, capnext, esc)
    out: list[str] = []
    pending_escape = False
    pending_word_mode: str | None = None
    active_allcaps = False
    pending_capnext = False
    in_ascii_word = False

    for ch in text:
        if pending_escape:
            if pending_word_mode is not None and not _is_ascii_alpha(ch):
                raise LosslessCapsError("escaped control char cannot satisfy pending word capitalization mode")
            out.append(ch)
            pending_escape = False
            if _is_ascii_alpha(ch):
                in_ascii_word = True
            else:
                in_ascii_word = False
                active_allcaps = False
            continue

        if ch == esc:
            pending_escape = True
            continue
        if ch == title:
            if pending_word_mode is not None or in_ascii_word or pending_capnext:
                raise LosslessCapsError("invalid title marker placement")
            pending_word_mode = "title"
            continue
        if ch == allcaps:
            if pending_word_mode is not None or in_ascii_word or pending_capnext:
                raise LosslessCapsError("invalid allcaps marker placement")
            pending_word_mode = "allcaps"
            continue
        if ch == capnext:
            if pending_capnext:
                raise LosslessCapsError("duplicate capnext marker")
            pending_capnext = True
            continue

        if _is_ascii_alpha(ch):
            at_word_start = not in_ascii_word
            if at_word_start:
                if pending_word_mode == "allcaps":
                    out.append(ch.upper())
                    active_allcaps = True
                elif pending_word_mode == "title":
                    out.append(ch.upper())
                elif pending_capnext:
                    out.append(ch.upper())
                else:
                    out.append(ch)
                pending_word_mode = None
                pending_capnext = False
                in_ascii_word = True
                continue

            if pending_word_mode is not None:
                raise LosslessCapsError("word capitalization marker leaked into the middle of a word")
            if active_allcaps:
                out.append(ch.upper())
            elif pending_capnext:
                out.append(ch.upper())
            else:
                out.append(ch)
            pending_capnext = False
            continue

        if pending_word_mode is not None or pending_capnext:
            raise LosslessCapsError("capitalization marker not followed by an ASCII letter")
        out.append(ch)
        in_ascii_word = False
        active_allcaps = False

    if pending_escape:
        raise LosslessCapsError("dangling escape marker at end of string")
    if pending_word_mode is not None or pending_capnext:
        raise LosslessCapsError("dangling capitalization marker at end of string")
    return "".join(out)


# ======================================================================
# The wrapper tokenizer.
# ======================================================================


def _apply_transform(item):
    """Apply CaseOps to a text item; pass through pairs / pre-tokenized inputs."""
    if isinstance(item, str):
        return encode_lossless_caps_v2(item)
    if isinstance(item, (tuple, list)) and len(item) == 2 and all(isinstance(p, str) for p in item):
        return type(item)((encode_lossless_caps_v2(item[0]), encode_lossless_caps_v2(item[1])))
    # pre-tokenized word lists, etc. — leave untouched
    return item


class CaseOpsTokenizer(PreTrainedTokenizerFast):
    """Byte-level BPE on CaseOps-transformed text, with the transform wired into
    encode/decode so callers pass and receive RAW text.

    Override surface (transformers 4.57): `_encode_plus` funnels into
    `_batch_encode_plus`, so intercepting `_batch_encode_plus` covers
    __call__/encode/encode_plus/batch_encode_plus. `tokenize` is a separate
    path. `_decode` is the egress funnel.
    """

    def _batch_encode_plus(self, batch_text_or_text_pairs, *args, **kwargs):
        if not kwargs.get("is_split_into_words", False):
            batch_text_or_text_pairs = [_apply_transform(x) for x in batch_text_or_text_pairs]
        return super()._batch_encode_plus(batch_text_or_text_pairs, *args, **kwargs)

    def tokenize(self, text, pair=None, add_special_tokens=False, **kwargs):
        text = encode_lossless_caps_v2(text) if isinstance(text, str) else text
        if isinstance(pair, str):
            pair = encode_lossless_caps_v2(pair)
        return super().tokenize(text, pair=pair, add_special_tokens=add_special_tokens, **kwargs)

    def _decode(self, *args, **kwargs):
        decoded = super()._decode(*args, **kwargs)
        if not isinstance(decoded, str):
            return decoded
        # markers may be absent if skip_special_tokens=True — the inverse
        # passes marker-free text through unchanged, so this is always safe.
        # Guard anyway: never let a malformed sequence crash a decode.
        try:
            return decode_lossless_caps_v2(decoded)
        except LosslessCapsError:
            return decoded


__all__ = ["CASEOPS_MARKERS", "CaseOpsTokenizer", "decode_lossless_caps_v2", "encode_lossless_caps_v2"]
