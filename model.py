"""Tiny Genie-style latent action models.

Genie 1 has three parts: a video tokenizer, a latent action model (LAM) and a
dynamics model. Our frames are already tiny 8x8 grids, so we skip the tokenizer
and let the LAM's decoder double as the dynamics model.

Three variants:
  * oracle    - dynamics model given the TRUE actions (upper bound, uses labels)
  * genie     - one latent action per transition, 8 codes (as in Genie 1)
  * peragent  - one latent action slot PER AGENT, shared 8-code codebook.
                Each slot's encoder only sees its own agent's channel, so it can
                only describe that agent's motion. No action labels are used.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from env import G, N_ACTIONS


class VQ(nn.Module):
    """Vector quantizer with straight-through gradients and dead-code restarts."""

    def __init__(self, n_codes, dim, beta=0.25):
        super().__init__()
        self.codebook = nn.Embedding(n_codes, dim)
        self.codebook.weight.data.uniform_(-1 / n_codes, 1 / n_codes)
        self.beta = beta
        self.register_buffer("usage", torch.zeros(n_codes))

    def forward(self, z):
        d = (z.pow(2).sum(1, keepdim=True) - 2 * z @ self.codebook.weight.t()
             + self.codebook.weight.pow(2).sum(1))
        idx = d.argmin(1)
        q = self.codebook(idx)
        loss = F.mse_loss(q, z.detach()) + self.beta * F.mse_loss(z, q.detach())
        q = z + (q - z).detach()
        if self.training:
            self.usage.mul_(0.99).index_add_(0, idx, torch.ones_like(idx, dtype=torch.float) * 0.01)
        return q, idx, loss

    @torch.no_grad()
    def restart_dead(self, z):
        dead = (self.usage < 1e-3).nonzero().flatten()
        if len(dead):
            pick = z[torch.randint(len(z), (len(dead),))]
            self.codebook.weight.data[dead] = pick
            self.usage[dead] = 0.01
        return len(dead)


class ActionEncoder(nn.Module):
    """(frame_t, frame_t+1) -> continuous latent action.

    Convolutions followed by a global max-pool make the encoder translation
    invariant: it can see *how* something moved but not *where* it is, so a
    code can't secretly mean "left in this corner, down in that one". This
    was the fix for codes that mixed up actions in the first version.
    """

    def __init__(self, in_ch, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveMaxPool2d(1), nn.Flatten(),
            nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, dim))

    def forward(self, x):
        return self.net(x)


class Dynamics(nn.Module):
    """(frame_t, action embeddings) -> logits over the 64 cells for each agent."""

    def __init__(self, act_ch):
        super().__init__()
        ch = 64
        layers = [nn.Conv2d(3 + act_ch, ch, 3, padding=1), nn.ReLU()]
        for _ in range(4):
            layers += [nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU()]
        layers += [nn.Conv2d(ch, 2, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, a):
        a = a[:, :, None, None].expand(-1, -1, G, G)
        return self.net(torch.cat([x, a], 1)).flatten(2)  # (B, 2, 64)


class WorldModel(nn.Module):
    def __init__(self, kind, n_codes=8, dim=16):
        super().__init__()
        self.kind, self.n_codes, self.dim = kind, n_codes, dim
        if kind == "oracle":
            self.act_emb = nn.Embedding(N_ACTIONS, dim)
            self.dyn = Dynamics(2 * dim)
        elif kind == "genie":
            self.enc = ActionEncoder(4, dim)  # sees both agents, before and after
            self.vq = VQ(n_codes, dim)
            self.dyn = Dynamics(dim)
        elif kind == "peragent":
            self.enc = ActionEncoder(2, dim)  # one agent, before and after; shared across agents
            self.vq = VQ(n_codes, dim)        # shared codebook
            self.dyn = Dynamics(2 * dim)
        else:
            raise ValueError(kind)

    # ----- latent action inference (training / labelling) -----
    def encode(self, x0, x1):
        """Returns quantized action vectors, code indices, VQ loss, pre-quant z."""
        if self.kind == "genie":
            z = self.enc(torch.cat([x0[:, 1:], x1[:, 1:]], 1))
            q, idx, l = self.vq(z)
            return q, idx[:, None], l, z
        # peragent: agent k's encoder sees only its own channel, before and after
        qs, ids, zs, l = [], [], [], 0
        for k in (1, 2):
            inp = torch.cat([x0[:, [k]], x1[:, [k]]], 1)
            z = self.enc(inp)
            q, idx, lk = self.vq(z)
            qs.append(q); ids.append(idx); zs.append(z); l = l + lk
        return torch.cat(qs, 1), torch.stack(ids, 1), l, torch.cat(zs, 0)

    # ----- dynamics given action codes (play time) -----
    def codes_to_vec(self, codes):
        if self.kind == "oracle":
            return torch.cat([self.act_emb(codes[:, 0]), self.act_emb(codes[:, 1])], 1)
        if self.kind == "genie":
            return self.vq.codebook(codes[:, 0])
        return torch.cat([self.vq.codebook(codes[:, 0]), self.vq.codebook(codes[:, 1])], 1)

    def predict(self, x0, avec):
        return self.dyn(x0, avec)

    def loss(self, x0, x1, true_actions):
        tgt = x1[:, 1:].flatten(2).argmax(-1)  # (B, 2) target cell per agent
        if self.kind == "oracle":
            logits = self.predict(x0, self.codes_to_vec(true_actions))
            vq_loss, z = torch.tensor(0.0), None
        else:
            avec, _, vq_loss, z = self.encode(x0, x1)
            logits = self.predict(x0, avec)
        ce = F.cross_entropy(logits.reshape(-1, G * G), tgt.reshape(-1))
        return ce + vq_loss, ce, z


def step_state(x0, logits):
    """Turn predicted logits into the next one-hot grid (walls copied)."""
    x1 = torch.zeros_like(x0)
    x1[:, 0] = x0[:, 0]
    pos = logits.argmax(-1)  # (B, 2)
    for k in range(2):
        x1[:, k + 1].view(len(x0), -1).scatter_(1, pos[:, k:k + 1], 1.0)
    return x1
