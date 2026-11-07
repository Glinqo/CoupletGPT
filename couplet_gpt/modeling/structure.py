"""Aligned GPT with optional source-only bidirectional prefix and BMES auxiliary head."""

import math
import torch
from torch import nn
from torch.nn import functional as F
from couplet_gpt.modeling.gpt import GPTConfig, GPT
from couplet_gpt.modeling.alignment import AlignedGPT


def masked_attention(module, x, allowed, explicit=False):
    b, t, d = x.shape
    q, k, v = module.qkv(x).chunk(3, dim=-1)
    q, k, v = [a.reshape(b, t, module.heads, module.head_dim).transpose(1, 2) for a in (q, k, v)]
    if explicit:
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(module.head_dim)
        probs = F.dropout(
            scores.masked_fill(~allowed, -float("inf")).softmax(-1),
            p=module.p,
            training=module.training,
        )
        output = probs @ v
    else:
        output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed, dropout_p=module.p if module.training else 0.0
        )
    output = output.transpose(1, 2).contiguous().reshape(b, t, d)
    return module.resid_drop(module.proj(output))


class StructuredGPTV3(AlignedGPT):
    def __init__(self, c=None, prefix=False):
        super().__init__(c)
        self.prefix = prefix
        self.boundary_head = nn.Linear(self.config.width, 4)
        nn.init.normal_(self.boundary_head.weight, std=0.02)
        nn.init.zeros_(self.boundary_head.bias)

    def forward(self, ids, explicit=False, return_aux=False):
        b, t = ids.shape
        assert t <= self.config.context_length
        pos = torch.arange(t, device=ids.device)
        down = (ids == 5).long().argmax(1)
        rel = pos[None, :] - down[:, None]
        valid = (down[:, None] > 0) & (rel >= 0) & (rel < (down - 2)[:, None])
        source = (rel + 2).clamp(0, t - 1)
        allowed = None
        if self.prefix:
            # Source prefix excludes DOWN; it never reads target tokens.
            causal = pos[:, None] >= pos[None, :]
            src = pos[None, :] < down[:, None]
            allowed = causal[None, :, :] | (src[:, :, None] & src[:, None, :])
            allowed = allowed[:, None, :, :] & (ids != 0)[:, None, None, :]
        h = self.drop(self.tok_emb(ids) + self.pos_emb(pos))
        for block, projection in zip(self.blocks, self.alignment):
            if allowed is None:
                h = block(h, explicit=explicit)
            else:
                h = h + masked_attention(block.attn, block.ln1(h), allowed, explicit)
                h = h + block.ffn(block.ln2(h))
            aligned = h.gather(1, source[:, :, None].expand(-1, -1, h.shape[-1]))
            h = h + projection(aligned) * valid[:, :, None]
        h = self.norm(h)
        logits = self.head(h)
        return (logits, self.boundary_head(h)) if return_aux else logits


class PairCritic(GPT):
    """Complete-pair classifier. It has no access to any reference during inference."""

    def __init__(self, c=None):
        super().__init__(c)
        self.classifier = nn.Linear(self.config.width, 1)
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, ids, explicit=False):
        b, t = ids.shape
        h = self.drop(self.tok_emb(ids) + self.pos_emb(torch.arange(t, device=ids.device)))
        allowed = (ids != 0)[:, None, None, :].expand(-1, 1, t, -1)
        for block in self.blocks:
            h = h + masked_attention(block.attn, block.ln1(h), allowed, explicit)
            h = h + block.ffn(block.ln2(h))
        return self.classifier(self.norm(h[:, 0])).squeeze(-1)


def load_v3(checkpoint, device="cuda"):
    kind = checkpoint["architecture"]
    c = GPTConfig(**checkpoint["model_config"])
    model = (
        PairCritic(c)
        if kind == "pair_critic"
        else StructuredGPTV3(c, prefix=kind == "aligned_prefix_bmes")
    )
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


def fast_structure_loss(logits, x, labels):
    """Vectorized equivalent of v2's mass loss; avoid per-position GPU syncs."""
    r, q = (labels >= 6).nonzero(as_tuple=True)
    down = (x == 5).long().argmax(1)
    j = q - down[r]
    z = logits[r, q].float()
    lse = z.logsumexp(-1)
    gold = labels[r, q]
    upper = x[r, j + 2]
    p_upper = (z.gather(1, upper[:, None]).squeeze(1) - lse).exp()
    same_ok = gold != upper
    first_loss = -torch.log1p(-p_upper.clamp(max=1 - 1e-6))
    ks = torch.arange(7, device=x.device)
    valid = ks[None, :] < j[:, None]
    ux = x[r[:, None], (ks + 2)[None, :].expand(len(r), -1).clamp(max=x.shape[1] - 1)]
    past = x[r[:, None], (down[r, None] + 1 + ks[None, :]).clamp(max=x.shape[1] - 1)]
    eq = (ux == upper[:, None]) & valid
    has = eq.any(1)
    required = past.gather(1, eq.long().argmax(1)[:, None]).squeeze(1)
    good_repeat = has & (gold == required)
    lp_required = z.gather(1, required[:, None]).squeeze(1) - lse
    repeat_loss = -lp_required.clamp_min(math.log(1e-8))
    duplicate = (
        (past[:, :, None] == past[:, None, :]) & (ks[:, None] > ks[None, :])[None, :, :]
    ).any(-1)
    unique = valid & ~duplicate
    pp = (z.gather(1, past) - lse[:, None]).exp()
    mass = (pp * unique).sum(1).clamp(max=1 - 1e-6)
    good_new = (j > 0) & ~has & ~((past == gold[:, None]) & valid).any(1)
    new_loss = -torch.log1p(-mass)
    total = (
        (first_loss * same_ok).sum()
        + (repeat_loss * good_repeat).sum()
        + (new_loss * good_new).sum()
    )
    count = same_ok.sum() + good_repeat.sum() + good_new.sum()
    return total / count.clamp_min(1)
