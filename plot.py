# Key figure: RMSE per model under random and grouped CV, and the paired gap between them.
# Run after baseline.py.

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "results"
MODELS = {"mean": "Mean", "least squares": "Least\nsquares", "ridge": "Ridge", "knn": "kNN"}
COLORS = {"random": "#2a78d6", "grouped": "#eb6834"}
SURFACE, INK, SOFT_INK, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"

summary = pd.read_csv(OUT / "summary.csv").set_index(["scheme", "model"])
paired = pd.read_csv(OUT / "paired.csv").set_index(["a", "b"])
positions = np.arange(len(MODELS))

plt.rcParams.update(
    {
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": SOFT_INK,
        "xtick.color": SOFT_INK,
        "ytick.color": SOFT_INK,
    }
)
fig, (left, right) = plt.subplots(
    1, 2, figsize=(8.2, 4.3), dpi=200, width_ratios=[1.7, 1], facecolor=SURFACE
)

for ax in (left, right):
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_xticks(positions, MODELS.values())

# left: RMSE per model, the two schemes side by side
width = 0.3
for scheme, shift in (("random", -0.16), ("grouped", 0.16)):
    rows = summary.loc[scheme].loc[list(MODELS)]
    x = positions + shift
    left.bar(x, rows.rmse_mean, width, color=COLORS[scheme], label=f"{scheme.capitalize()} folds")
    left.text(x[0], 16, scheme.capitalize(), rotation=90, ha="center", va="center", fontsize=9, color="white")
    left.errorbar(
        x,
        rows.rmse_mean,
        yerr=[rows.rmse_mean - rows.ci_low, rows.ci_high - rows.rmse_mean],
        fmt="none",
        ecolor=INK,
        elinewidth=1,
        capsize=3,
    )
    if scheme == "grouped":
        for xi, mean, top in zip(x, rows.rmse_mean, rows.ci_high):
            left.text(xi, top + 0.7, f"{mean:.1f}", ha="center", fontsize=9)

left.set_ylim(0, 39)
left.set_ylabel("5-fold CV RMSE (K)")
left.set_title("RMSE by model", loc="left", fontsize=11)
left.legend(frameon=False, loc="upper right")

# right: how much worse grouped folds are, paired on the same bootstrap draws
gaps = paired.loc[[(f"grouped {m}", f"random {m}") for m in MODELS]]
right.axhline(0, color=AXIS, linewidth=1)
right.vlines(positions, gaps.ci_low, gaps.ci_high, color=SOFT_INK, linewidth=2)
right.scatter(positions, gaps.diff_mean, s=45, color=SOFT_INK, edgecolor=SURFACE, linewidth=1.5, zorder=3)
for i in (2, 3):
    right.text(i + 0.13, gaps.diff_mean.iloc[i], f"+{gaps.diff_mean.iloc[i]:.2f}", va="center", fontsize=9)
right.set_xlim(-0.5, 3.75)
right.set_ylim(-0.1, 0.85)
right.set_ylabel("Grouped minus random RMSE (K)")
right.set_title("Extra error under grouped folds", loc="left", fontsize=11)

fig.tight_layout()
fig.savefig(OUT / "baseline.png", facecolor=SURFACE)
