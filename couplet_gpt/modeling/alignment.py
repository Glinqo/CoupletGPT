"""GPT plus zero-initialized aligned source residuals; no hard output masks here."""

import torch
from torch import nn
from couplet_gpt.modeling.gpt import GPT, GPTConfig


class AlignedGPT(GPT):
    def __init__(self, c=None):
        super().__init__(c)
        self.alignment = nn.ModuleList(
            [nn.Linear(self.config.width, self.config.width, bias=False) for _ in self.blocks]
        )
        for projection in self.alignment:
            nn.init.zeros_(projection.weight)

    def forward(self, ids, explicit=False):
        b, t = ids.shape
        pos = torch.arange(t, device=ids.device)
        down = (ids == 5).long().argmax(1)
        # At DOWN the next-token target is lower[0]; EOS prediction has no aligned source.
        rel = pos[None, :] - down[:, None]
        n = down - 2
        valid = (down[:, None] > 0) & (rel >= 0) & (rel < n[:, None])
        source = (rel + 2).clamp(0, t - 1)
        h = self.drop(self.tok_emb(ids) + self.pos_emb(pos))
        for block, projection in zip(self.blocks, self.alignment):
            h = block(h, explicit=explicit)
            aligned = h.gather(1, source[:, :, None].expand(-1, -1, h.shape[-1]))
            h = h + projection(aligned) * valid[:, :, None]
        return self.head(self.norm(h))


def load_model_v2(checkpoint, device="cuda"):
    model = (AlignedGPT if checkpoint.get("architecture") == "aligned" else GPT)(
        GPTConfig(**checkpoint["model_config"])
    )
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


def structure_loss(logits, x, labels):
    """Mass loss for learned equality relations, only when the gold obeys that rule."""
    losses = []
    b, t = labels.shape
    down = (x == 5).long().argmax(1)
    length = down - 2
    row = torch.arange(b, device=x.device)
    for j in range(7):
        active = length > j
        if not bool(active.any()):
            continue
        r = row[active]
        d = down[active]
        q = d + j
        n = len(r)
        z = logits[r, q].float()
        p = z.softmax(-1)
        upper = x[r, j + 2]
        gold = labels[r, q]
        # Exclude coincident gold characters; do not make CE and rule losses contradictory.
        invalid = p.gather(1, upper[:, None]).squeeze(1)
        same_ok = gold != upper
        losses.append(-torch.log1p(-invalid[same_ok].clamp(max=1 - 1e-6)))
        if j:
            ux = torch.stack([x[r, k + 2] for k in range(j)], 1)
            past = torch.stack([x[r, d + 1 + k] for k in range(j)], 1)
            eq = ux == upper[:, None]
            has = eq.any(1)
            first = eq.long().argmax(1)
            required = past.gather(1, first[:, None]).squeeze(1)
            good = has & (gold == required)
            if bool(good.any()):
                losses.append(
                    -torch.log(p[good].gather(1, required[good, None]).squeeze(1).clamp_min(1e-8))
                )
            # For a new upper class, earlier lower classes must not be reused.
            no = ~has
            unique = torch.ones_like(past, dtype=torch.bool)
            for k in range(1, j):
                unique[:, k] = ~(past[:, :k] == past[:, k, None]).any(1)
            conflict = (past == gold[:, None]).any(1)
            good = no & ~conflict
            mass = (p.gather(1, past) * unique).sum(1).clamp(max=1 - 1e-6)
            if bool(good.any()):
                losses.append(-torch.log1p(-mass[good]))
    return torch.cat([v.reshape(-1) for v in losses if v.numel()]).mean()
