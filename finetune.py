"""Fine-tune the per-agent model's dynamics with a small labelled set.

Finding from evaluate.py: a latent action model trained on video alone learns
what HAPPENED, not what each agent TRIED to do. A blocked move looks exactly
like "stay", so the model never learns collision rules. Here we give it a small
set of transitions with true actions (default 5% of the training data), mapped
to its own codes, and fine-tune only the dynamics model. The encoder and
codebook stay frozen, so the codes keep their meaning.
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from env import G, generate
from evaluate import displacement_action, positions
from model import WorldModel
from train import transitions

p = argparse.ArgumentParser()
p.add_argument("--label_frac", type=float, default=0.05)
p.add_argument("--steps", type=int, default=1500)
args = p.parse_args()
torch.set_grad_enabled(True)  # evaluate.py turns gradients off on import
torch.manual_seed(0)

x0, x1, a, c = transitions(*generate(3000, 16, seed=1))
m = WorldModel("peragent")
m.load_state_dict(torch.load("ckpt_peragent.pt"))
for prm in list(m.enc.parameters()) + list(m.vq.parameters()):
    prm.requires_grad = False

n_lab = int(args.label_frac * len(x0))
lab = torch.randperm(len(x0))[:n_lab]
with torch.no_grad():
    codes = m.encode(x0[lab], x1[lab])[1][:, 0].numpy()
eff = displacement_action(positions(x0[lab]), positions(x1[lab]))[:, 0]
amap = torch.tensor([int(np.bincount(codes[eff == u]).argmax()) for u in range(5)])
print(f"{n_lab} labelled transitions; action->code map {amap.tolist()}")

opt = torch.optim.Adam(m.dyn.parameters(), lr=3e-4)
for s in range(1, args.steps + 1):
    i = lab[torch.randint(n_lab, (128,))]
    tgt = x1[i, 1:].flatten(2).argmax(-1)
    logits = m.predict(x0[i], m.codes_to_vec(amap[a[i]]))
    sup = F.cross_entropy(logits.reshape(-1, G * G), tgt.reshape(-1))
    j = torch.randint(len(x0), (128,))  # keep the video-only objective too
    _, unsup, _ = m.loss(x0[j], x1[j], a[j])
    loss = sup + unsup
    opt.zero_grad(); loss.backward(); opt.step()
    if s % 250 == 0:
        print(f"step {s}  labelled CE {sup.item():.4f}  video CE {unsup.item():.4f}", flush=True)
torch.save(m.state_dict(), "ckpt_peragent_ft.pt")
