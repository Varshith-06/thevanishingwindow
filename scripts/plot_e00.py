"""Hack rate and read-time share V across the run, from the run's own logs."""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, MUTED, GRID, SERIES = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#2a78d6"
df = pd.read_csv("results/e00_logs/v_by_step.csv")
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
fig, axes = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True, facecolor=SURFACE)
panels = [
    ("any_rate", "Hack rate (changed tests or hacked harness)", (0, 1)),
    ("any_V", "Read-time share V (per step; line is a 7-step mean)", (-0.1, 1)),
]
for ax, (col, title, ylim) in zip(axes, panels):
    ax.set_facecolor(SURFACE)
    ax.scatter(df.adapter_step, df[col], s=14, color=SERIES, alpha=0.35, linewidths=0)
    ax.plot(df.adapter_step, df[col].rolling(7, center=True, min_periods=3).mean(), color=SERIES, lw=2)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
axes[1].axhline(0, color=MUTED, lw=0.8)
axes[1].set_xlabel("Adapter step (gradient updates)")
fig.suptitle("Preliminary, logs only: V falls as hacking takes hold, opposite to RQ1", x=0.01, ha="left", fontsize=13, fontweight="bold")
fig.text(0.01, 0.005, "32 tasks x 16 reruns per step; tasks differ by step. Environment label, which includes non-cheating test additions.", fontsize=8.5, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 0.97))
fig.savefig("results/e00_logs/v_by_step.png", dpi=160, facecolor=SURFACE)
