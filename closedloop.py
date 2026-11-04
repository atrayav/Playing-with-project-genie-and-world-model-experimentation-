"""Closed-loop experiment: an AI character, a user, and a learned world.

Here the learned world model IS the world, the way Genie is the world in
Project Genie. There is no game engine underneath at play time.

  * The AI character (agent B) is told "reach the goal cell".
  * The user (agent A) wanders, and often steps into B's path.
  * Each step, both actions go to the world model, which generates the next frame.

Three configurations:
  1. open_loop  - B plans a route once at the start and follows it blindly;
                  every generated frame is accepted as-is.
  2. replan     - a verifier checks every generated frame; B re-plans each step
                  from what the world currently shows (around walls and the user).
  3. recover    - as replan, but when the verifier rejects a frame, the system
                  asks the model for its most likely frame that passes the
                  checks (constrained re-decoding of the same prediction).

The verifier is hand-written rules standing in for the role Gemini plays in
Genie 4 ("reviews incoming frames to ... verify that each step succeeded"):
no agent inside a wall, no agent moving more than one cell, no two agents in
one cell, no swapping through each other, and each agent either did what it
was told or stayed put.

The real game rules are used only for SCORING: after each generated step we
ask what the real game would have done from the same state.
"""
import json
import time
from collections import deque
from itertools import product

import numpy as np
import torch

from env import DELTAS, G, random_layout, render, step
from evaluate import load
from model import step_state

torch.set_grad_enabled(False)
MAX_STEPS = 25
N_EPISODES = 300


def pos_of(x):
    flat = x[0, 1:].flatten(1).argmax(-1).numpy()
    return np.stack([flat // G, flat % G], -1)  # (2, 2): agent A, agent B


def bfs_first_action(walls, start, goal, blocked=None):
    """First action on a shortest path from start to goal (0 if none)."""
    start, goal = tuple(start), tuple(goal)
    if start == goal:
        return 0
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        for u in (1, 2, 3, 4):
            nxt = (cur[0] + DELTAS[u][0], cur[1] + DELTAS[u][1])
            if not (0 <= nxt[0] < G and 0 <= nxt[1] < G) or walls[nxt] or nxt in prev:
                continue
            if blocked is not None and nxt == tuple(blocked) and nxt != goal:
                continue
            prev[nxt] = (cur, u)
            q.append(nxt)
    if goal not in prev:
        return 0
    node = goal
    while prev[node][0] != start:
        node = prev[node][0]
    return prev[node][1]


def bfs_path(walls, start, goal):
    acts, cur = [], np.array(start)
    for _ in range(G * G):
        u = bfs_first_action(walls, cur, goal)
        if u == 0:
            break
        acts.append(u)
        cur = cur + DELTAS[u]
    return acts


def user_policy(rng, a, b, b_next, walls):
    """The user wanders, and 40% of the time steps toward where B is heading."""
    if rng.random() < 0.4:
        return bfs_first_action(walls, a, b_next) or int(rng.integers(5))
    return int(rng.integers(5))


def violations(walls, prev, nxt, acts):
    """Rule checks on a generated frame. Returns a list of problems."""
    probs = []
    for k in range(2):
        p, q = prev[k], nxt[k]
        if walls[q[0], q[1]]:
            probs.append("inside_wall")
        if np.abs(q - p).sum() > 1:
            probs.append("teleport")
        if not ((q == p).all() or (q == p + DELTAS[acts[k]]).all()):
            probs.append("wrong_move")
    if (nxt[0] == nxt[1]).all():
        probs.append("overlap")
    if (nxt[0] == prev[1]).all() and (nxt[1] == prev[0]).all() and not (prev[0] == prev[1]).all():
        probs.append("swap")
    return probs


def constrained_decode(logits, walls, prev, acts):
    """Most likely joint next position that passes every check."""
    lp = torch.log_softmax(logits[0], -1).numpy()  # (2, 64)
    cands = []
    for k in range(2):
        opts = [prev[k], prev[k] + DELTAS[acts[k]]]
        opts = [o for o in opts if 0 <= o[0] < G and 0 <= o[1] < G and not walls[o[0], o[1]]]
        cands.append({tuple(o): o for o in opts}.values())
    best, best_lp = None, -np.inf
    for pa, pb in product(*cands):
        cand = np.stack([pa, pb])
        if violations(walls, prev, cand, acts):
            continue
        score = lp[0, pa[0] * G + pa[1]] + lp[1, pb[0] * G + pb[1]]
        if score > best_lp:
            best, best_lp = cand, score
    return best


def run(model, kind, mode, seed):
    rng = np.random.default_rng(seed)
    map_pa = json.load(open("results.json"))["peragent_action_to_code"]
    # layout with a goal at least 5 steps from B
    while True:
        walls, a, b = random_layout(rng)
        free = np.argwhere(~walls)
        goal = free[rng.integers(len(free))]
        if len(bfs_path(walls, b, goal)) >= 5 and not (goal == a).all():
            break
    state = torch.from_numpy(render(walls, a, b))[None]
    plan = bfs_path(walls, b, goal)
    stats = dict(reached=False, steps=0, violating_steps=0, rejected_frames=0,
                 rule_consistent=0, ms=0.0)
    for t in range(MAX_STEPS):
        prev = pos_of(state)
        if (prev[1] == goal).all():
            stats["reached"] = True
            break
        # AI character's action
        if mode == "open_loop":
            b_act = plan[t] if t < len(plan) else 0
        else:
            b_act = bfs_first_action(walls, prev[1], goal, blocked=prev[0])
        b_next = prev[1] + DELTAS[b_act]
        a_act = user_policy(rng, prev[0], prev[1], b_next, walls)
        acts = [a_act, b_act]

        t0 = time.perf_counter()
        codes = torch.tensor([[map_pa[a_act], map_pa[b_act]]])
        logits = model.predict(state, model.codes_to_vec(codes))
        nxt_state = step_state(state, logits)
        nxt = pos_of(nxt_state)
        if mode != "open_loop":
            probs = violations(walls, prev, nxt, acts)
            if probs and mode == "recover":
                fixed = constrained_decode(logits, walls, prev, acts)
                if fixed is not None:
                    stats["rejected_frames"] += 1
                    nxt = fixed
                    nxt_state = torch.from_numpy(render(walls, nxt[0], nxt[1]))[None]
        stats["ms"] += (time.perf_counter() - t0) * 1000

        # scoring only: what would the real game do from the same state?
        if violations(walls, prev, nxt, acts):
            stats["violating_steps"] += 1
        if not (prev[0] == prev[1]).all():
            ta, tb, _ = step(walls, prev[0], prev[1], a_act, b_act)
            if (ta == nxt[0]).all() and (tb == nxt[1]).all():
                stats["rule_consistent"] += 1
        state = nxt_state
        stats["steps"] += 1
    else:
        stats["reached"] = bool((pos_of(state)[1] == goal).all())
    return stats


def main():
    out = {}
    for kind in ("peragent", "peragent_ft"):
        model = load(kind)
        for mode in ("open_loop", "replan", "recover"):
            S = [run(model, kind, mode, 1000 + s) for s in range(N_EPISODES)]
            steps = sum(s["steps"] for s in S)
            out[f"{kind}/{mode}"] = {
                "goal_reached": float(np.mean([s["reached"] for s in S])),
                "episodes_with_violation": float(np.mean([s["violating_steps"] > 0 for s in S])),
                "steps_violating": sum(s["violating_steps"] for s in S) / steps,
                "steps_rule_consistent": sum(s["rule_consistent"] for s in S) / steps,
                "frames_rejected_and_fixed": sum(s["rejected_frames"] for s in S) / steps,
                "ms_per_step": sum(s["ms"] for s in S) / steps,
                "mean_steps": steps / len(S),
            }
            print(kind, mode, {k: round(v, 3) for k, v in out[f"{kind}/{mode}"].items()}, flush=True)
    json.dump(out, open("results_closedloop.json", "w"), indent=2)


if __name__ == "__main__":
    main()
