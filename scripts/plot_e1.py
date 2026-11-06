"""E1 headline figure: V by checkpoint, then D(t) and C(t) by cut for each resampled checkpoint."""
import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
COLORS = {45: "#2a78d6", 70: "#eb6834"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})

rel = json.load(open("results/e1_top/relabel.json"))
adapters = sorted(int(k) for k in rel if k.isdigit())
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
v = [rel[str(a)]["hack_sem"]["V"] for a in adapters]
lo = [rel[str(a)]["hack_sem"]["lo"] for a in adapters]
hi = [rel[str(a)]["hack_sem"]["hi"] for a in adapters]
ax.plot(adapters, v, color=MUTED, lw=2, zorder=1)
ax.vlines(adapters, lo, hi, color=MUTED, lw=2, zorder=1)
ax.scatter(adapters, v, s=60, color=INK, zorder=2, edgecolors=SURFACE, linewidths=2)
for a, val in zip(adapters, v):
    rate = rel[str(a)]["hack_sem"]["rate"]
    ax.annotate(f"{val:.2f}\nhack rate {rate:.0%}", (a, val), textcoords="offset points", xytext=(8, 4), fontsize=8.5, color=MUTED)
ax.set_xticks(adapters)
ax.set_xlim(min(adapters) - 6, max(adapters) + 14)
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
        ax.annotate(f"adapter {a}", (x[-1], y[-1]), textcoords="offset points", xytext=(6, 0), fontsize=8.5, color=MUTED, va="center")
    ax.set_xscale("log", base=2)
    ax.set_xticks([32, 128, 256, 512, 768])
    ax.set_xticklabels(["32", "128", "256", "512", "768"])
    ax.set_xlim(24, 1500)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Cut position (generated tokens)")
    if key == "D":
        ax.axhline(0, color=MUTED, lw=0.8)
    else:
        ax.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (3, 3)))
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")

fig.suptitle("E1 pilot: late in training, the hack decision forms while the model writes", x=0.01, ha="left", fontsize=12.5, fontweight="bold")
fig.text(0.01, 0.01, "Qwen3-8B reward-hacking checkpoints, semantic hack label, 95% bootstrap intervals over tasks. V: 100 tasks x 8 reruns. D and C: 30 mixed tasks x 4 reruns x 4 continuations per cut.", fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 0.93))
fig.savefig("results/e1_pilot.png", dpi=160, facecolor=SURFACE)
print("saved results/e1_pilot.png")
