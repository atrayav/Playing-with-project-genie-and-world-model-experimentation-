"""Two-agent gridworld used to generate "videos" for the world model.

Observation: 3 x G x G binary grid, channels = [walls, agent A, agent B].
Actions per agent: 0 stay, 1 up, 2 down, 3 left, 4 right.
Both agents move at the same time. A move fails (agent stays) if it hits a wall
or the edge, if both agents target the same cell, if they try to swap places,
or if the target cell is occupied by an agent that is not leaving it.
These collision rules are the "multi-agent interaction" the model has to learn.
"""
import numpy as np

G = 8
N_ACTIONS = 5
ACTION_NAMES = ["stay", "up", "down", "left", "right"]
DELTAS = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]])


def random_layout(rng, wall_frac=0.12):
    walls = rng.random((G, G)) < wall_frac
    free = np.argwhere(~walls)
    a, b = free[rng.choice(len(free), size=2, replace=False)]
    return walls, a.copy(), b.copy()


def render(walls, a, b):
    obs = np.zeros((3, G, G), dtype=np.float32)
    obs[0] = walls
    obs[1, a[0], a[1]] = 1
    obs[2, b[0], b[1]] = 1
    return obs


def _blocked(walls, p):
    return not (0 <= p[0] < G and 0 <= p[1] < G) or walls[p[0], p[1]]


def step(walls, a, b, act_a, act_b):
    """Returns new positions and whether an agent-agent collision blocked a move."""
    ta = a + DELTAS[act_a]
    tb = b + DELTAS[act_b]
    if _blocked(walls, ta):
        ta = a.copy()
    if _blocked(walls, tb):
        tb = b.copy()
    collision = False
    # Resolve agent-agent conflicts until stable.
    for _ in range(3):
        if (ta == tb).all():
            if not (ta == a).all():
                ta = a.copy()
            if not (tb == b).all():
                tb = b.copy()
            collision = True
        if (ta == b).all() and (tb == a).all() and not (ta == a).all():
            ta, tb = a.copy(), b.copy()  # swap attempt
            collision = True
        if (ta == b).all() and (tb == b).all() and not (ta == a).all():
            ta = a.copy()  # B is staying put in the cell A wants
            collision = True
        if (tb == a).all() and (ta == a).all() and not (tb == b).all():
            tb = b.copy()
            collision = True
    return ta, tb, collision


def policy(rng, a, b, prev, chase_p=0.3, repeat_p=0.4):
    """Random policy, sometimes repeating its last move or heading toward the
    other agent so that collisions happen often enough to learn from."""
    acts = []
    for me, other, p in ((a, b, prev[0]), (b, a, prev[1])):
        r = rng.random()
        if r < repeat_p:
            acts.append(p)
        elif r < repeat_p + chase_p:
            d = other - me
            if abs(d[0]) >= abs(d[1]) and d[0] != 0:
                acts.append(1 if d[0] < 0 else 2)
            elif d[1] != 0:
                acts.append(3 if d[1] < 0 else 4)
            else:
                acts.append(0)
        else:
            acts.append(int(rng.integers(N_ACTIONS)))
    return acts


def generate(n_episodes, T, seed):
    rng = np.random.default_rng(seed)
    obs = np.zeros((n_episodes, T + 1, 3, G, G), dtype=np.float32)
    acts = np.zeros((n_episodes, T, 2), dtype=np.int64)
    coll = np.zeros((n_episodes, T), dtype=bool)
    for e in range(n_episodes):
        walls, a, b = random_layout(rng)
        obs[e, 0] = render(walls, a, b)
        prev = [0, 0]
        for t in range(T):
            act = policy(rng, a, b, prev)
            a, b, c = step(walls, a, b, *act)
            obs[e, t + 1] = render(walls, a, b)
            acts[e, t] = act
            coll[e, t] = c
            prev = act
    return obs, acts, coll


if __name__ == "__main__":
    obs, acts, coll = generate(200, 16, 0)
    print("collision rate per step:", coll.mean())
    print("action distribution:", np.bincount(acts.ravel(), minlength=5) / acts.size)
