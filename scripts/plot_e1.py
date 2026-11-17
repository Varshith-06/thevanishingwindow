"""Headline figure: V by checkpoint (E1 and E2), then D(t) and C(t) by cut for each resampled checkpoint."""
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
COLORS = {45: "#2a78d6", 55: "#1f9e6e", 70: "#eb6834"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})

vrows = {int(r["adapter"]): r for r in csv.DictReader(open("results/e2_top/v_curve.csv"))}
adapters = sorted(vrows)
cuts = [r for r in csv.DictReader(open("results/e1_cut/d_c_by_cut.csv"))]
resampled = sorted({int(r["adapter"]) for r in cuts})

fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), facecolor=SURFACE)


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


ax = axes[0]
style(ax, "Read-time share V")
v = [float(vrows[a]["V"]) for a in adapters]
lo = [float(vrows[a]["V_lo"]) for a in adapters]
hi = [float(vrows[a]["V_hi"]) for a in adapters]
ax.plot(adapters, v, color=MUTED, lw=2, zorder=1)
ax.vlines(adapters, lo, hi, color=MUTED, lw=2, zorder=1)
ax.scatter(adapters, v, s=60, color=INK, zorder=2, edgecolors=SURFACE, linewidths=2)
for a, val in zip(adapters, v):
    if a in COLORS or a == min(adapters):
        rate = float(vrows[a]["hack_rate"])
        ax.annotate(f"{val:.2f}\nhack {rate:.0%}", (a, val), textcoords="offset points", xytext=(6, 6), fontsize=8, color=MUTED)
ax.set_xticks(adapters)
ax.set_xlim(min(adapters) - 4, max(adapters) + 8)
ax.set_ylim(-0.1, 1)
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_xlabel("Adapter step")

for ax, key, title, ylim in ((axes[1], "D", "Determined share D(t)", (-0.15, 1)), (axes[2], "C", "Decision ceiling C(t), within-task AUC", (0.3, 1))):
    style(ax, title)
    for a in resampled:
        rs = [r for r in cuts if int(r["adapter"]) == a]
        x = [int(r["cut"]) for r in rs]
        y = [float(r[key]) for r in rs]
        ax.fill_between(x, [float(r[f"{key}_lo"]) for r in rs], [float(r[f"{key}_hi"]) for r in rs], color=COLORS[a], alpha=0.12, lw=0)
        ax.plot(x, y, color=COLORS[a], lw=2, marker="o", ms=5, mec=SURFACE, mew=1.5, label=f"adapter {a}")
        ax.plot(x, [float(r[f"{key}_running"]) for r in rs], color=COLORS[a], lw=1.3, ls=(0, (3, 2)), )
    ax.set_xscale("log", base=2)
    ax.set_xticks([32, 128, 256, 512, 768])
    ax.set_xticklabels(["32", "128", "256", "512", "768"])
    ax.set_xlim(24, 1000)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Cut position (generated tokens)")
    if key == "D":
        ax.axhline(0, color=MUTED, lw=0.8)
    else:
        ax.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (3, 3)))
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")

fig.suptitle("As hacking takes hold, the task decides less, and the text decides part of the rest after 256 tokens", x=0.01, ha="left", fontsize=12.5, fontweight="bold")
fig.text(0.01, 0.01, "Qwen3-8B, semantic hack label, 95% task-bootstrap CIs. V: 100 tasks x 8 reruns. D, C: 30 or 50 (adapter 55) mixed tasks x 4 reruns x 4 continuations. Dashed: reruns still running.", fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 0.93))
fig.savefig("results/e1_pilot.png", dpi=160, facecolor=SURFACE)
print("saved results/e1_pilot.png")
