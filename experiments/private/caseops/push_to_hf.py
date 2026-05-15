# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Push CaseOps preflight tokenizer artifacts to a PRIVATE HuggingFace repo.

So a rented pod can `AutoTokenizer.from_pretrained(REPO, subfolder=..., trust_remote_code=True)`
instead of copying directories around.

Pushes (into one private repo, as subfolders):
  - caseops_{8000,16000,32000,48k,64000}_wrapped/  -- the trust_remote_code wrappers
  - plain_{8000,16000,32000,48k,64000}/            -- the Pareto-triangle controls

Does NOT push the intermediate non-wrapped caseops_* inner dirs (subsumed by the
wrapped ones). Idempotent — re-running re-uploads changed files.

NOTE: these are PREFLIGHT tokenizers trained on a 150k-doc sample of FineWeb-10B
(docs_selected.jsonl). The FINAL submission tokenizer must be retrained on
FineWeb-EDU (Marin's speedrun corpus) — DESIGN.md R4. The README in the repo
says so.

Run (CPU, needs HF auth — `Frosty40` is logged in via huggingface_hub):
    .venv/bin/python -m experiments.private.caseops.push_to_hf
    .venv/bin/python -m experiments.private.caseops.push_to_hf --repo newjordan/caseops-marin-tokenizers
"""

from __future__ import annotations

import argparse
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
ARTIFACTS = _HERE / "artifacts"
# HF username is Frosty40 (GitHub is newjordan — different namespaces).
DEFAULT_REPO = "Frosty40/caseops-marin-tokenizers"

PUSH_DIRS = [
    "caseops_8000_wrapped",
    "caseops_16000_wrapped",
    "caseops_32000_wrapped",
    "caseops_48k_wrapped",
    "caseops_64000_wrapped",
    "plain_8000",
    "plain_16000",
    "plain_32000",
    "plain_48k",
    "plain_64000",
]

README = """\
# CaseOps Marin tokenizers (PRIVATE — preflight artifacts)

Private working artifacts for the CaseOps Marin Speedrun submission. Not for release.

## Contents

- `caseops_<vocab>_wrapped/` — `trust_remote_code` CaseOps tokenizers. `encode(raw)`
  applies the lossless CaseOps case-factoring transform, then a byte-level BPE
  trained on transformed text; `decode` inverts it. The 4 marker chars
  (U+E001..U+E004) are registered as special tokens (0-byte in Levanter BPB).
  Load: `AutoTokenizer.from_pretrained(REPO, subfolder="caseops_48k_wrapped", trust_remote_code=True)`
- `plain_<vocab>/` — plain byte-level BPE controls (the Marin baseline tokenizer
  architecture, retrained at vocab N on untransformed text). The Pareto-triangle
  control arm.

Vocab sizes: 8000, 16000, 32000, 48000 ("48k"), 64000.

## IMPORTANT

These are PREFLIGHT tokenizers trained on a 150k-doc sample of FineWeb-10B.
The FINAL submission tokenizer must be retrained on FineWeb-EDU (Marin's speedrun
training corpus). These are for pipeline validation and the Pareto-triangle
methodology, not the final submission artifact.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=DEFAULT_REPO, help="HF repo id (will be created private if missing)")
    ap.add_argument("--dry-run", action="store_true", help="List what would be pushed, do not upload")
    args = ap.parse_args()

    missing = [d for d in PUSH_DIRS if not (ARTIFACTS / d).is_dir()]
    if missing:
        print(f"FAIL: missing artifact dirs: {missing}")
        return 1

    print(f"repo: {args.repo} (private)")
    print(f"pushing {len(PUSH_DIRS)} tokenizer dirs as subfolders:")
    for d in PUSH_DIRS:
        print(f"  - {d}")
    if args.dry_run:
        print("\n[dry-run] nothing uploaded.")
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=True, exist_ok=True)
    print("\nrepo ready (private). uploading...", flush=True)

    # README at repo root
    api.upload_file(
        path_or_fileobj=README.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="model",
    )

    # tokenization_caseops.py at repo ROOT: HF's trust_remote_code module loader
    # resolves the auto_map module relative to the repo root, NOT the subfolder.
    # The class is identical for every caseops_*_wrapped subfolder, so one root
    # copy serves all of them.
    api.upload_file(
        path_or_fileobj=str(_HERE / "tokenization_caseops.py"),
        path_in_repo="tokenization_caseops.py",
        repo_id=args.repo,
        repo_type="model",
    )

    for d in PUSH_DIRS:
        api.upload_folder(
            folder_path=str(ARTIFACTS / d),
            path_in_repo=d,
            repo_id=args.repo,
            repo_type="model",
        )
        print(f"  uploaded {d}", flush=True)

    print(f"\nDONE. Private repo: https://huggingface.co/{args.repo}")
    print("Pod usage:")
    print(f'  AutoTokenizer.from_pretrained("{args.repo}", subfolder="caseops_48k_wrapped", trust_remote_code=True)')
    return 0


if __name__ == "__main__":
    sys.exit(main())
