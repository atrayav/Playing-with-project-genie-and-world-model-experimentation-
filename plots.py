"""Make figures from results.json plus a GIF comparing rollouts."""
import json

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from env import G
from evaluate import load
from model import step_state

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
COL = {"peragent_ft": "#2a78d6", "genie": "#eb6834", "peragent": "#1baf7a", "oracle": "#8a8984"}
NAME = {"peragent_ft": "Per-agent latent actions + 5% labelled steps",
        "peragent": "Per-agent latent actions, video only",
        "genie": "Genie-style, one latent action, video only",
        "oracle": "True action labels on every step (upper bound)"}
KINDS = ["peragent_ft", "peragent", "genie", "oracle"]
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def control_chart(res):
    groups = [("overall", "All steps"), ("unblocked_steps", "Every move succeeds"),
              ("wall_blocked_steps", "A move hits a wall"), ("collision_steps", "Agents collide")]
    kinds = KINDS
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=200)
    w = 0.2
    x = np.arange(len(groups))
    for i, k in enumerate(kinds):
        vals = [100 * res["control"][k][g] for g, _ in groups]
        bars = ax.bar(x + (i - 1.5) * w, vals, w - 0.03, color=COL[k], label=NAME[k], zorder=2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center",
                    va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, [g for _, g in groups])
    ax.set_ylim(0, 110)
    ax.set_yticks(range(0, 101, 25))
    ax.set_ylabel("Next frame exactly right (%)")
    ax.set_title("Can you steer each agent with the learned actions?", loc="left", color=INK, fontsize=11)
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    fig.tight_layout()
    fig.savefig("fig_control.png")


def drift_chart(res):
    fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=200)
    steps = np.arange(1, len(res["drift"]["oracle"]) + 1)
    for k in KINDS:
        y = 100 * np.array(res["drift"][k])
        ax.plot(steps, y, color=COL[k], lw=2, ls="--" if k == "oracle" else "-", label=NAME[k])
        ax.text(steps[-1] + 0.3, y[-1], f"{y[-1]:.0f}%", va="center", fontsize=8, color=INK)
    ax.set_xlim(1, steps[-1] + 1.5)
    ax.set_ylim(0, 102)
    ax.set_xlabel("Steps the model has run on its own predictions")
    ax.set_ylabel("Rollouts still exactly right (%)")
    ax.set_title("Drift: errors compound when the model feeds on itself", loc="left", color=INK, fontsize=11)
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1, 0.95))
    fig.tight_layout()
    fig.savefig("fig_drift.png")


def confusion_chart(res):
    acts = ["stay", "up", "down", "left", "right"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), dpi=200, gridspec_kw={"width_ratios": [1, 3]})
    A = np.array(res["confusion_peragent_A"])
    J = np.array(res["confusion_genie_joint"])
    for ax, M, title, xl in ((axes[0], A, "Per-agent model, agent A's slot", acts),
                             (axes[1], J, "Genie-style model, one slot for both agents",
                              [f"{a[0].upper()}/{b[0].upper()}" for a in acts for b in acts])):
        M = M / M.sum(0, keepdims=True).clip(1)  # share of each true action going to each code
        ax.imshow(M, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
            "b", ["#fcfcfb", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"]), vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(xl)), xl, rotation=90 if len(xl) > 5 else 0, fontsize=7 if len(xl) > 5 else 8)
        ax.set_yticks(range(8), [f"code {i}" for i in range(8)], fontsize=7)
        ax.set_title(title, loc="left", fontsize=9, color=INK)
        ax.set_xlabel("What actually happened (A/B for agent A / agent B)" if len(xl) > 5 else "What agent A actually did", fontsize=8)
    fig.suptitle("What does each learned code mean? (darker = more of that action maps to the code)",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig("fig_codes.png")


def grid_img(x, scale=28):
    img = np.full((G, G, 3), 252, np.uint8)
    img[x[0] > 0.5] = (82, 81, 78)
    img[x[1] > 0.5] = (42, 120, 214)
    img[x[2] > 0.5] = (235, 104, 52)
    img = img.repeat(scale, 0).repeat(scale, 1)
    img[::scale, :] = 225
    img[:, ::scale] = 225
    return img


def rollout_gif(res, episode=150):
    d = np.load("data_test.npz")
    obs, acts = d["arr_0"], d["arr_1"]
    map_pa = res["peragent_action_to_code"]
    models = {k: load(k) for k in ("peragent_ft", "genie")}
    # genie mapping is not stored; rebuild quickly from calibration episodes
    from evaluate import displacement_action, positions
    T = acts.shape[1]
    x0 = torch.from_numpy(obs[:100, :-1].reshape(-1, 3, G, G))
    x1 = torch.from_numpy(obs[:100, 1:].reshape(-1, 3, G, G))
    eff = displacement_action(positions(x0), positions(x1))
    g = models["genie"].encode(x0, x1)[1][:, 0].numpy()
    common = int(np.bincount(g).argmax())
    map_g = {}
    for u in range(5):
        for v in range(5):
            m = (eff[:, 0] == u) & (eff[:, 1] == v)
            map_g[(u, v)] = int(np.bincount(g[m]).argmax()) if m.any() else common
    states = {"peragent_ft": torch.from_numpy(obs[episode, :1]), "genie": torch.from_numpy(obs[episode, :1])}
    frames = []
    pad = np.full((G * 28, 14, 3), 252, np.uint8)
    for t in range(T + 1):
        row = [grid_img(obs[episode, t]), pad, grid_img(states["peragent_ft"][0].numpy()), pad,
               grid_img(states["genie"][0].numpy())]
        frames.append(np.concatenate(row, 1))
        if t == T:
            break
        u, v = acts[episode, t]
        with torch.no_grad():
            for k, m in models.items():
                codes = torch.tensor([[map_pa[u], map_pa[v]]]) if k == "peragent_ft" else torch.tensor([[map_g[(u, v)]]])
                states[k] = step_state(states[k], m.predict(states[k], m.codes_to_vec(codes)))
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    labels = ["Real game", "Per-agent + 5% labels", "Genie-style (video only)"]
    out = []
    for t, f in enumerate(frames):
        canvas = Image.new("RGB", (f.shape[1], f.shape[0] + 46), (252, 252, 251))
        canvas.paste(Image.fromarray(f), (0, 40))
        dr = ImageDraw.Draw(canvas)
        for i, lab in enumerate(labels):
            dr.text((i * (G * 28 + 14), 4), lab, fill=(11, 11, 11), font=font)
        dr.text((0, 21), f"step {t}   blue = agent A, orange = agent B, gray = wall", fill=(82, 81, 78), font=font)
        out.append(np.array(canvas))
    imageio.mimsave("rollout.gif", out, duration=0.7, loop=0)


if __name__ == "__main__":
    res = json.load(open("results.json"))
    control_chart(res)
    drift_chart(res)
    confusion_chart(res)
    rollout_gif(res)
    print("wrote fig_control.png fig_drift.png fig_codes.png rollout.gif")
