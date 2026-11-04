"""Evaluate the three world models. Writes results.json, used by plots.py.

Metrics (action labels are used ONLY here, for scoring, never for training the
genie / peragent models):
  1. Code purity: does each learned code mean one real action?
  2. Control accuracy: map each real action to a code using a small labelled
     calibration set (Genie does the same to map latent actions to buttons),
     then ask the model for the next frame and compare with the true game.
     Reported overall, on steps where both agents move, and on collision steps.
  3. Leakage: change agent A's code while B's says "stay"; does B wrongly move?
  4. Drift: roll the model forward on its own predictions for 16 steps and
     track how often the whole state is still exactly right.
"""
import json

import numpy as np
import torch

from env import DELTAS, G, N_ACTIONS, step
from model import WorldModel, step_state

torch.set_grad_enabled(False)


def load(kind):
    m = WorldModel("peragent" if kind == "peragent_ft" else kind)
    m.load_state_dict(torch.load(f"ckpt_{kind}.pt"))
    return m.eval()


def positions(x):  # (B,3,G,G) -> (B,2,2) row/col of agents A and B
    flat = x[:, 1:].flatten(2).argmax(-1)
    return torch.stack([flat // G, flat % G], -1)


def displacement_action(p0, p1):
    """Which action's displacement explains the move (0 = no move)."""
    d = (p1 - p0).numpy()
    out = np.zeros(d.shape[:-1], dtype=int)
    for u in range(1, N_ACTIONS):
        out[(d == DELTAS[u]).all(-1)] = u
    return out


def purity(codes, labels):
    tot = 0
    for c in np.unique(codes):
        tot += np.bincount(labels[codes == c]).max()
    return tot / len(codes)


def main():
    d = np.load("data_test.npz")
    obs, acts, coll = d["arr_0"], d["arr_1"], d["arr_2"]
    E, T = acts.shape[:2]
    x0 = torch.from_numpy(obs[:, :-1].reshape(-1, 3, G, G))
    x1 = torch.from_numpy(obs[:, 1:].reshape(-1, 3, G, G))
    a = acts.reshape(-1, 2)
    c = coll.reshape(-1)
    eff = displacement_action(positions(x0), positions(x1))  # observed effect per agent

    calib = slice(0, 100 * T)        # first 100 episodes: labelled calibration
    test = slice(100 * T, E * T)     # remaining 300 episodes: evaluation
    models = {k: load(k) for k in ("oracle", "genie", "peragent", "peragent_ft")}
    res = {}

    # ---- 1. Code purity --------------------------------------------------
    codes = {}
    for k in ("genie", "peragent"):
        _, idx, _, _ = models[k].encode(x0, x1)
        codes[k] = idx.numpy()
    g, pa = codes["genie"][:, 0], codes["peragent"]
    joint = eff[:, 0] * N_ACTIONS + eff[:, 1]
    res["purity"] = {
        "genie_vs_agentA": purity(g, eff[:, 0]),
        "genie_vs_agentB": purity(g, eff[:, 1]),
        "genie_vs_joint": purity(g, joint),
        "peragent_slotA_vs_agentA": purity(pa[:, 0], eff[:, 0]),
        "peragent_slotB_vs_agentB": purity(pa[:, 1], eff[:, 1]),
        "genie_codes_used": int(len(np.unique(g))),
        "peragent_codes_used": int(len(np.unique(pa))),
    }
    # code x effect tables for plotting
    res["confusion_genie_joint"] = np.histogram2d(g, joint, bins=[8, 25], range=[[0, 8], [0, 25]])[0].tolist()
    res["confusion_peragent_A"] = np.histogram2d(pa[:, 0], eff[:, 0], bins=[8, 5], range=[[0, 8], [0, 5]])[0].tolist()

    # ---- action -> code maps from the calibration set --------------------
    def best(code_arr, mask, fallback):
        return int(np.bincount(code_arr[mask]).argmax()) if mask.any() else fallback

    ok = (eff == a)  # the requested move actually happened
    map_pa = [best(pa[calib, 0], (eff[calib, 0] == u), 0) for u in range(N_ACTIONS)]
    common_g = int(np.bincount(g[calib]).argmax())
    map_g = {}
    for ua in range(N_ACTIONS):
        for ub in range(N_ACTIONS):
            m = (eff[calib, 0] == ua) & (eff[calib, 1] == ub)
            map_g[(ua, ub)] = best(g[calib], m, common_g)
    res["genie_distinct_codes_for_25_joint_actions"] = len(set(map_g.values()))
    res["peragent_action_to_code"] = map_pa

    def to_codes(kind, acts_t):
        if kind == "oracle":
            return torch.from_numpy(acts_t)
        if kind.startswith("peragent"):
            return torch.tensor([[map_pa[u], map_pa[v]] for u, v in acts_t])
        return torch.tensor([[map_g[(int(u), int(v))]] for u, v in acts_t])

    # ---- 2. Control accuracy on the held-out episodes --------------------
    tx0, tx1, ta, tc = x0[test], x1[test], a[test], c[test]
    both_move = (ta != 0).all(1)
    teff = eff[test]
    clean = (teff == ta).all(1)                 # every requested move happened
    wall = ~clean & ~tc                         # something hit a wall / edge only
    res["control"] = {}
    preds = {}
    for k, m in models.items():
        logits = m.predict(tx0, m.codes_to_vec(to_codes(k, ta)))
        right = (step_state(tx0, logits) == tx1).flatten(1).all(1).numpy()
        preds[k] = right
        res["control"][k] = {
            "overall": float(right.mean()),
            "both_agents_moving": float(right[both_move].mean()),
            "unblocked_steps": float(right[clean].mean()),
            "wall_blocked_steps": float(right[wall].mean()),
            "collision_steps": float(right[tc].mean()),
        }
    res["n_test_steps"] = int(len(ta))
    res["n_collision_steps"] = int(tc.sum())
    res["n_unblocked_steps"] = int(clean.sum())
    res["n_wall_blocked_steps"] = int(wall.sum())

    # reconstruction: next frame from codes the encoder infers from the real pair
    res["reconstruction_with_inferred_codes"] = {}
    for k in ("genie", "peragent"):
        mm = models[k]
        avec = mm.encode(tx0, tx1)[0]
        right = (step_state(tx0, mm.predict(tx0, avec)) == tx1).flatten(1).all(1)
        res["reconstruction_with_inferred_codes"][k] = float(right.float().mean())

    # ---- 3. Leakage (peragent): vary A, keep B on "stay" ------------------
    m = models["peragent_ft"]
    sx = tx0[:2000]
    pb0 = positions(sx)[:, 1]
    leaks = []
    for u in range(N_ACTIONS):
        cc = torch.tensor([[map_pa[u], map_pa[0]]] * len(sx))
        pb = positions(step_state(sx, m.predict(sx, m.codes_to_vec(cc))))[:, 1]
        leaks.append(float((pb != pb0).any(1).float().mean()))
    res["leakage_B_moves_when_only_A_changes"] = float(np.mean(leaks))

    # ---- 4. Drift: autoregressive rollouts --------------------------------
    eps = range(100, E)
    res["drift"] = {}
    for k, m in models.items():
        alive = np.zeros(T)
        for e in eps:
            s = torch.from_numpy(obs[e, :1])
            for t in range(T):
                s = step_state(s, m.predict(s, m.codes_to_vec(to_codes(k, acts[e, t:t + 1]))))
                if not torch.equal(s[0], torch.from_numpy(obs[e, t + 1])):
                    break
                alive[t] += 1
        res["drift"][k] = (alive / len(eps)).tolist()

    json.dump(res, open("results.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("confusion")}, indent=2))


if __name__ == "__main__":
    main()
