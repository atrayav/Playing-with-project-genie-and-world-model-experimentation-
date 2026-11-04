"""Generate data and train the three world models.

Usage: python train.py [--steps 4000]
Action labels are only used by the oracle model. The genie and peragent models
learn from frame pairs alone.
"""
import argparse
import time

import numpy as np
import torch

from env import generate
from model import WorldModel


def transitions(obs, acts, coll):
    x0 = torch.from_numpy(obs[:, :-1].reshape(-1, *obs.shape[2:]))
    x1 = torch.from_numpy(obs[:, 1:].reshape(-1, *obs.shape[2:]))
    a = torch.from_numpy(acts.reshape(-1, 2))
    c = torch.from_numpy(coll.reshape(-1))
    return x0, x1, a, c


def train(kind, data, steps, seed=0, bs=256, log=500):
    torch.manual_seed(seed)
    x0, x1, a, _ = data
    m = WorldModel(kind)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    t0 = time.time()
    for s in range(1, steps + 1):
        i = torch.randint(len(x0), (bs,))
        loss, ce, z = m.loss(x0[i], x1[i], a[i])
        opt.zero_grad(); loss.backward(); opt.step()
        if kind != "oracle" and s % 200 == 0 and s <= steps // 2:
            m.vq.restart_dead(z.detach())
        if s % log == 0:
            print(f"[{kind}] step {s:5d}  next-frame CE {ce.item():.4f}  ({time.time()-t0:.0f}s)", flush=True)
    return m


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--kinds", default="oracle,genie,peragent")
    args = p.parse_args()

    tr = generate(3000, 16, seed=1)
    np.savez_compressed("data_test.npz", *generate(400, 16, seed=2))
    data = transitions(*tr)
    print(f"{len(data[0])} training transitions, collision rate {data[3].float().mean():.2f}")
    for kind in args.kinds.split(","):
        m = train(kind, data, args.steps)
        torch.save(m.state_dict(), f"ckpt_{kind}.pt")
