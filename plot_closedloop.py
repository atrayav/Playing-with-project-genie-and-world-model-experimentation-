"""Figure for the closed-loop experiment (reads results_closedloop.json)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plots import INK, style

r = json.load(open("results_closedloop.json"))
modes = [("open_loop", "Open loop", "#eb6834"), ("replan", "Verify + re-plan", "#1baf7a"),
         ("recover", "Verify + re-plan + fix bad frames", "#2a78d6")]
metrics = [("goal_reached", "AI reaches its goal\n(in the generated world)"),
           ("episodes_with_violation", "Episodes with a\nbroken frame"),
           ("steps_rule_consistent", "Steps that match\nthe real game's rules")]

fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), dpi=200, sharey=True)
for ax, (kind, title) in zip(axes, [("peragent", "World model trained on video only"),
                                    ("peragent_ft", "World model + 5% labelled steps")]):
    x = np.arange(len(metrics))
    w = 0.26
    for i, (m, label, col) in enumerate(modes):
        vals = [100 * r[f"{kind}/{m}"][k] for k, _ in metrics]
        bars = ax.bar(x + (i - 1) * w, vals, w - 0.03, color=col, label=label, zorder=2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center", va="bottom",
                    fontsize=7.5, color=INK)
    ax.set_xticks(x, [l for _, l in metrics], fontsize=8)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 25))
    style(ax)
axes[0].set_ylabel("% of 300 episodes / steps")
h, l = axes[0].get_legend_handles_labels()
fig.suptitle("An AI character crossing a generated world while a user gets in the way",
             x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.legend(h, l, frameon=False, fontsize=8, loc="lower center", ncol=3)
fig.savefig("fig_closedloop.png", bbox_inches="tight")
print("wrote fig_closedloop.png")
