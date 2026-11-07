"""Small GPT implemented from primitive PyTorch modules, without pretrained weights."""

from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 13232
    context_length: int = 128
    layers: int = 6
    width: int = 256
    heads: int = 4
    dropout: float = 0.1
    norm_eps: float = 1e-5


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads = c.heads
        self.head_dim = c.width // c.heads
        self.p = c.dropout
        self.qkv = nn.Linear(c.width, 3 * c.width, bias=False)
        self.proj = nn.Linear(c.width, c.width)
        self.resid_drop = nn.Dropout(c.dropout)

    def forward(self, x, explicit=False):
        b, t, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [z.reshape(b, t, self.heads, self.head_dim).transpose(1, 2) for z in (q, k, v)]
        if explicit:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            future = torch.ones(t, t, device=x.device, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(future, float("-inf"))
            probs = F.dropout(scores.softmax(dim=-1), p=self.p, training=self.training)
            out = probs @ v
        else:
            # All data is right padded: a valid causal query cannot see any PAD key.
            # PAD query outputs are ignored by CE; no fully masked softmax rows arise.
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.p if self.training else 0.0
            )
        out = out.transpose(1, 2).contiguous().reshape(b, t, d)
        return self.resid_drop(self.proj(out))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.width, eps=c.norm_eps)
        self.attn = Attention(c)
        self.ln2 = nn.LayerNorm(c.width, eps=c.norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(c.width, 4 * c.width),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * c.width, c.width),
            nn.Dropout(c.dropout),
        )

    def forward(self, x, explicit=False):
        x = x + self.attn(self.ln1(x), explicit=explicit)
        return x + self.ffn(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, c=None):
        super().__init__()
        self.config = c or GPTConfig()
        c = self.config
        assert c.width % c.heads == 0
        self.tok_emb = nn.Embedding(c.vocab_size, c.width)
        self.pos_emb = nn.Embedding(c.context_length, c.width)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.layers)])
        self.norm = nn.LayerNorm(c.width, eps=c.norm_eps)
        self.head = nn.Linear(c.width, c.vocab_size, bias=False)
        self.apply(self._init)
        self.head.weight = self.tok_emb.weight
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, std=0.02 / math.sqrt(2 * c.layers))
            nn.init.normal_(block.ffn[2].weight, std=0.02 / math.sqrt(2 * c.layers))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, ids, explicit=False):
        t = ids.shape[1]
        assert t <= self.config.context_length
        h = self.drop(self.tok_emb(ids) + self.pos_emb(torch.arange(t, device=ids.device)))
        for block in self.blocks:
            h = block(h, explicit=explicit)
        return self.head(self.norm(h))

    def optimizer(self, lr=3e-4, weight_decay=0.1):
        decay = []
        no_decay = []
        for name, p in self.named_parameters():
            (decay if p.ndim == 2 and "emb" not in name else no_decay).append(p)
        assert len({id(p) for p in decay + no_decay}) == len(decay) + len(no_decay)
        return torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=lr,
            betas=(0.9, 0.95),
            eps=1e-8,
        )


def loss_sum(logits, labels):
    mask = labels != -100
    return F.cross_entropy(logits[mask].float(), labels[mask], reduction="sum")


@torch.inference_mode()
def generate(model, prompt_ids, max_new=64, temperature=0.0, top_k=20, seed=2027):
    device = next(model.parameters()).device
    ids = list(prompt_ids)
    gen = torch.Generator(device=device).manual_seed(seed)
    for _ in range(min(max_new, model.config.context_length - len(ids))):
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            scores = model(x)[0, -1].float()
        scores[[0, 1, 2, 4, 5]] = float("-inf")
        if temperature == 0:
            token = int(scores.argmax())
        else:
            values, indices = torch.topk(scores / temperature, top_k)
            token = int(indices[torch.multinomial(values.softmax(0), 1, generator=gen)])
        ids.append(token)
        if token == 3:
            break
    return ids[len(prompt_ids) :]
