# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize the CaseOps speedrun submission from the 3 finished arm runs.

Unlike the qwen3 sweep-and-select materializer, the CaseOps submission is a
fixed 3-arm Pareto triangle (baseline / control / treatment) — there is nothing
to "select", all three arms ARE the submission. This script just aggregates the
three speedrun_results.json files that default_speedrun's results step produced,
computes the triangle deltas, and draws the BPB-vs-FLOPs Pareto plot.

Inputs: each arm's speedrun_results.json (default_speedrun writes one per arm at
`<train_step_output>/speedrun_results.json`). Pass the dir that contains them,
or the 3 paths explicitly.

Outputs (written next to this file, the submission dir):
  - selection_summary.json  -- the 3 arm points + treatment-vs-baseline and
    treatment-vs-control deltas on both axes (BPB and training_hardware_flops)
  - bpb_vs_flops.png        -- the Pareto plot; the "cold improvement" is the
    treatment point sitting left-and/or-down of the baseline line

Run (CPU, after D.7 produces the 3 speedrun_results.json files):
    .venv/bin/python -m experiments.speedrun.caseops.materialize_submission \
        --results-dir <prefix>/checkpoints/speedrun
    # or explicitly:
    .venv/bin/python -m experiments.speedrun.caseops.materialize_submission \
        --baseline path/to/baseline/speedrun_results.json \
        --control  path/to/control/speedrun_results.json \
        --treatment path/to/treatment/speedrun_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

os.environ.setdefault("MPLCONFIGDIR", str(pathlib.Path(__file__).resolve().parent / ".matplotlib"))

OUTPUT_DIR = pathlib.Path(__file__).resolve().parent
SUMMARY_PATH = OUTPUT_DIR / "selection_summary.json"
PLOT_PATH = OUTPUT_DIR / "bpb_vs_flops.png"
ARMS = ("baseline", "control", "treatment")
BPB_KEY = "eval/paloma/c4_en/bpb"
FLOPS_KEY = "training_hardware_flops"


def _load_run_info(path: pathlib.Path) -> dict:
    """Load one arm's speedrun_results.json -> its run_info dict."""
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    try:
        return payload["runs"][0]["run_info"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"{path}: not a valid speedrun_results.json ({exc})") from exc


def _resolve_paths(args: argparse.Namespace) -> dict[str, pathlib.Path]:
    """Resolve the 3 arm result paths from --results-dir or explicit flags."""
    if args.results_dir is not None:
        base = pathlib.Path(args.results_dir)
        paths = {}
        for arm in ARMS:
            # default_speedrun names train steps `speedrun/caseops-<arm>-<vocab>`;
            # the results file is at <step output>/speedrun_results.json. Glob for it.
            matches = sorted(base.glob(f"**/caseops-{arm}-*/speedrun_results.json"))
            if not matches:
                matches = sorted(base.glob(f"**/*{arm}*/speedrun_results.json"))
            if not matches:
                raise FileNotFoundError(f"no speedrun_results.json found for arm {arm!r} under {base}")
            paths[arm] = matches[-1]  # most recent if multiple
        return paths
    explicit = {"baseline": args.baseline, "control": args.control, "treatment": args.treatment}
    missing = [a for a, p in explicit.items() if p is None]
    if missing:
        raise ValueError(f"either --results-dir or all of --baseline/--control/--treatment; missing: {missing}")
    return {a: pathlib.Path(p) for a, p in explicit.items()}


def _pct(new: float, old: float) -> float:
    return (new - old) / old * 100.0 if old else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=None, help="Dir containing the 3 arms' speedrun_results.json (globbed)")
    ap.add_argument("--baseline", default=None, help="Explicit path to baseline speedrun_results.json")
    ap.add_argument("--control", default=None, help="Explicit path to control speedrun_results.json")
    ap.add_argument("--treatment", default=None, help="Explicit path to treatment speedrun_results.json")
    args = ap.parse_args()

    paths = _resolve_paths(args)
    arms: dict[str, dict] = {arm: _load_run_info(paths[arm]) for arm in ARMS}

    rows = {}
    for arm, info in arms.items():
        bpb = info.get(BPB_KEY)
        flops = info.get(FLOPS_KEY)
        if bpb is None or flops is None:
            raise ValueError(f"arm {arm!r} missing {BPB_KEY!r} or {FLOPS_KEY!r} — was the run finished?")
        rows[arm] = {
            "bpb": float(bpb),
            "training_hardware_flops": float(flops),
            "vocab_size": info.get("vocab_size"),
            "model_size": info.get("model_size"),
            "model_flops": info.get("model_flops"),
            "training_time": info.get("training_time"),
            "results_path": str(paths[arm]),
            "wandb_run_link": info.get("wandb_run_link"),
        }

    b, c, t = rows["baseline"], rows["control"], rows["treatment"]
    summary = {
        "arms": rows,
        "deltas": {
            "treatment_vs_baseline": {
                "bpb_pct": _pct(t["bpb"], b["bpb"]),
                "flops_pct": _pct(t["training_hardware_flops"], b["training_hardware_flops"]),
            },
            "treatment_vs_control": {
                "bpb_pct": _pct(t["bpb"], c["bpb"]),
                "flops_pct": _pct(t["training_hardware_flops"], c["training_hardware_flops"]),
            },
        },
        # The "cold improvement" claim holds if treatment is left-and/or-down of
        # baseline on the BPB-vs-FLOPs frontier:
        "treatment_pareto_dominates_baseline": (
            t["bpb"] <= b["bpb"] and t["training_hardware_flops"] <= b["training_hardware_flops"]
        ),
    }
    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"wrote {SUMMARY_PATH}")

    # Pareto plot
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))
        colors = {"baseline": "#1a73e8", "control": "#eab308", "treatment": "#14b8a6"}
        for arm in ARMS:
            r = rows[arm]
            ax.scatter(r["training_hardware_flops"], r["bpb"], s=90, color=colors[arm], label=arm, zorder=3)
            ax.annotate(
                f"  {arm}\n  vocab={r['vocab_size']}",
                (r["training_hardware_flops"], r["bpb"]),
                fontsize=8,
                va="center",
            )
        ax.set_xscale("log")
        ax.set_xlabel("training hardware FLOPs")
        ax.set_ylabel("Paloma C4-EN BPB")
        ax.set_title("CaseOps Pareto triangle: BPB vs FLOPs")
        ax.legend()
        ax.grid(True, which="both", ls=":", alpha=0.4)
        fig.tight_layout()
        fig.savefig(PLOT_PATH, dpi=130)
        print(f"wrote {PLOT_PATH}")
    except ImportError:
        print("matplotlib not available — skipped the plot (selection_summary.json still written)")

    print()
    print(f"{'arm':<12}{'BPB':>10}{'hw FLOPs':>14}{'vocab':>9}")
    for arm in ARMS:
        r = rows[arm]
        print(f"{arm:<12}{r['bpb']:>10.5f}{r['training_hardware_flops']:>14.3e}{r['vocab_size']!s:>9}")
    d = summary["deltas"]["treatment_vs_baseline"]
    print(f"\ntreatment vs baseline: BPB {d['bpb_pct']:+.2f}%  |  FLOPs {d['flops_pct']:+.2f}%")
    print(f"treatment Pareto-dominates baseline: {summary['treatment_pareto_dominates_baseline']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
