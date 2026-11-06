"""Static evaluation plots (PNG) for the report.

Each policy keeps a fixed color + marker regardless of which policies are
present, so figures stay comparable across runs.
"""

from __future__ import annotations

import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

STYLE = {  # fixed slot order: blue, orange, aqua, yellow
    "learned": ("#2a78d6", "o"),
    "greedy_preview": ("#eb6834", "s"),
    "greedy_heuristic": ("#1baf7a", "^"),
    "random": ("#eda100", "D"),
}
VALUE_ORDER = ["low", "high", "glossy", "medium", "matte", "frontal", "oblique"]
INK, MUTED, GRID = "#1f1f1d", "#6b6a63", "#e1e0d9"


def _style(p):
    return STYLE.get(p, ("#6250d6", "v"))


def _axes(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def _label(p, preview):
    return f"{p} (preview)" if p in preview else p


def make_plots(out_dir: str, result: Dict, by_cond: List[Dict], recs: List[Dict]) -> None:
    summary, budget = result["summary"], result["budget"]
    preview = set(result.get("preview_baselines", []))
    note = "MOCK — software validation only" if "MOCK" in result.get("note", "") else ""
    pols = list(summary)

    # 1) success and incorrect acceptance vs action budget (two panels, shared scale)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    b = np.arange(budget + 1)
    for p in pols:
        c, m = _style(p)
        axs[0].plot(b, summary[p]["success_vs_budget"], color=c, marker=m, lw=2, ms=7, label=_label(p, preview))
        axs[1].plot(b, summary[p]["incorrect_accept_vs_budget"], color=c, marker=m, lw=2, ms=7, label=_label(p, preview))
    axs[0].set_title("Correct acceptance", color=INK, loc="left")
    axs[1].set_title("Incorrect acceptance", color=INK, loc="left")
    for ax in axs:
        _axes(ax)
        ax.set_xlabel("action budget (movements)")
        ax.set_xticks(b)
        ax.set_ylim(0, 1)
    axs[0].set_ylabel("fraction of episodes")
    axs[0].legend(frameon=False, loc="lower right")
    if note:
        fig.suptitle(note, color=MUTED, fontsize=9, x=0.99, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "success_vs_budget.png"), dpi=150)
    plt.close(fig)

    # 2) success by condition (small multiples, one per condition)
    conds = sorted({r["condition"] for r in by_cond})
    if conds:
        fig, axs = plt.subplots(1, len(conds), figsize=(4 * len(conds), 3.8), sharey=True, squeeze=False)
        for ax, cond in zip(axs[0], conds):
            rows = [r for r in by_cond if r["condition"] == cond]
            present = {r["value"] for r in rows}
            vals = [v for v in VALUE_ORDER if v in present] + sorted(present - set(VALUE_ORDER))
            w = 0.8 / max(1, len(pols))
            for i, p in enumerate(pols):
                ys = [next((r["success_rate"] for r in rows if r["policy"] == p and r["value"] == v), np.nan) for v in vals]
                ax.bar(np.arange(len(vals)) + (i - (len(pols) - 1) / 2) * w, ys, width=w * 0.9,
                       color=_style(p)[0], label=_label(p, preview))
            ax.set_xticks(np.arange(len(vals)))
            ax.set_xticklabels([f"{v}\n(n={next(r['n'] for r in rows if r['value'] == v)})" for v in vals])
            ax.set_title(cond.replace("_", " "), color=INK, loc="left")
            ax.set_ylim(0, 1)
            _axes(ax)
            ax.grid(False, axis="x")
        axs[0][0].set_ylabel("success rate")
        h, l = axs[0][0].get_legend_handles_labels()
        fig.legend(h, l, frameon=False, ncol=len(pols), loc="upper left", fontsize=9)
        if note:
            fig.text(0.99, 0.98, note, color=MUTED, fontsize=9, ha="right", va="top")
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        fig.savefig(os.path.join(out_dir, "success_by_condition.png"), dpi=150)
        plt.close(fig)

    # 3) path length distribution per policy (all valid episodes)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    data = [[r["path_length"] for r in recs if r["policy"] == p and r["outcome"] != "infra_error"] for p in pols]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5)
    for patch, p in zip(bp["boxes"], pols):
        patch.set_facecolor(_style(p)[0])
        patch.set_alpha(0.85)
    for med in bp["medians"]:
        med.set_color(INK)
    ax.set_xticks(range(1, len(pols) + 1))
    ax.set_xticklabels([_label(p, preview) for p in pols], fontsize=8)
    ax.set_ylabel("cumulative camera path (m)")
    _axes(ax)
    if note:
        ax.set_title(note, color=MUTED, fontsize=9, loc="right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "path_length.png"), dpi=150)
    plt.close(fig)
