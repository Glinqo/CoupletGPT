"""Independent, self-built commentary GPT initialized from the course pretraining run."""

import json
from dataclasses import asdict
from pathlib import Path
import torch
from torch.nn import functional as F
from couplet_gpt.data.comment_splits import ROOT, DEST, prompt_text

BASE = ROOT / "runs/pretrain_v1/artifacts/best_model.pt"
# The public model.py has the same SHA256 as the frozen training snapshot.
# Import it directly so cached decoding does not create or read historical runs.
from couplet_gpt.modeling.gpt import GPT, GPTConfig


class Tokenizer:
    def __init__(self):
        self.vocab = json.loads((DEST / "vocab.json").read_text(encoding="utf8"))
        self.reverse = {v: k for k, v in self.vocab.items()}

    def encode(self, text):
        unknown = set(text) - set(self.vocab)
        if unknown:
            raise ValueError("Uncovered characters: " + repr(sorted(unknown)))
        return [self.vocab[c] for c in text]

    def decode(self, ids):
        return "".join(self.reverse[int(i)] for i in ids if int(i) >= 6)


def initialize(seed=2026, device="cuda"):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    tok = Tokenizer()
    ck = torch.load(BASE, map_location="cpu", weights_only=True)
    config = GPTConfig(
        **{**ck["model_config"], "context_length": 768, "vocab_size": len(tok.vocab)}
    )
    model = GPT(config)
    state = {
        k: v
        for k, v in ck["model"].items()
        if k not in ["tok_emb.weight", "head.weight", "pos_emb.weight"]
    }
    missing, extra = model.load_state_dict(state, strict=False)
    assert set(missing) == {"tok_emb.weight", "head.weight", "pos_emb.weight"} and not extra
    with torch.no_grad():
        n = ck["model_config"]["vocab_size"]
        model.tok_emb.weight[:n].copy_(ck["model"]["tok_emb.weight"])
        model.pos_emb.weight[: ck["model_config"]["context_length"]].copy_(
            ck["model"]["pos_emb.weight"]
        )
    assert model.head.weight is model.tok_emb.weight
    return model.to(device), tok, ck


def collate(rows, branch, tok, device="cuda"):
    inputs = []
    labels = []
    for r in rows:
        p = [2] + tok.encode(prompt_text(r, branch))
        target = tok.encode(r["response"]) + [3]
        seq = p + target
        x = seq[:-1]
        y = [-100] * len(x)
        y[len(p) - 1 :] = target
        assert len(x) <= 768 and len(y) == len(x)
        inputs.append(x)
        labels.append(y)
    width = max(map(len, inputs))
    x = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    y = torch.full_like(x, -100)
    for i, (a, b) in enumerate(zip(inputs, labels)):
        x[i, : len(a)] = torch.tensor(a, device=device)
        y[i, : len(b)] = torch.tensor(b, device=device)
    return x, y


@torch.inference_mode()
def evaluate_ce(model, rows, branch, tok, batch=2):
    model.eval()
    total = count = 0
    for start in range(0, len(rows), batch):
        x, y = collate(rows[start : start + batch], branch, tok)
        mask = y != -100
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = model(x)
        total += float(F.cross_entropy(z[mask].float(), y[mask], reduction="sum"))
        count += int(mask.sum())
    return {"output_character_ce": total / count, "supervised_characters": count, "rows": len(rows)}


@torch.inference_mode()
def cached_prefix(model, x, lengths, capacity):
    b, t = x.shape
    c = model.config
    h = model.tok_emb(x) + model.pos_emb(torch.arange(t, device=x.device))
    caches = []
    for block in model.blocks:
        a = block.attn
        z = block.ln1(h)
        q, k, v = a.qkv(z).chunk(3, -1)
        q, k, v = [s.reshape(b, t, a.heads, a.head_dim).transpose(1, 2) for s in (q, k, v)]
        kc = torch.zeros((b, a.heads, capacity, a.head_dim), device=x.device, dtype=k.dtype)
        vc = torch.zeros_like(kc)
        kc[:, :, :t] = k
        vc[:, :, :t] = v
        caches.append((kc, vc))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        h = h + a.proj(out.transpose(1, 2).reshape(b, t, c.width))
        h = h + block.ffn(block.ln2(h))
    last = h[torch.arange(b, device=x.device), lengths - 1]
    return model.head(model.norm(last)), caches


@torch.inference_mode()
def cached_step(model, tokens, positions, newlengths, caches):
    b = len(tokens)
    h = model.tok_emb(tokens[:, None]) + model.pos_emb(positions[:, None])
    width = int(newlengths.max())
    mask = torch.arange(width, device=tokens.device)[None, :] < newlengths[:, None]
    bi = torch.arange(b, device=tokens.device)
    for block, (kc, vc) in zip(model.blocks, caches):
        a = block.attn
        q, k, v = a.qkv(block.ln1(h)).chunk(3, -1)
        q, k, v = [s.reshape(b, 1, a.heads, a.head_dim).transpose(1, 2) for s in (q, k, v)]
        kc[bi, :, positions, :] = k.squeeze(2)
        vc[bi, :, positions, :] = v.squeeze(2)
        out = F.scaled_dot_product_attention(
            q, kc[:, :, :width], vc[:, :, :width], attn_mask=mask[:, None, None, :]
        )
        h = h + a.proj(out.transpose(1, 2).reshape(b, 1, model.config.width))
        h = h + block.ffn(block.ln2(h))
    return model.head(model.norm(h[:, 0]))


@torch.inference_mode()
def decode(model, rows, branch, tok, max_new=260, batch=4, amp=False):
    model.eval()
    results = []
    for start in range(0, len(rows), batch):
        subset = rows[start : start + batch]
        pp = [[2] + tok.encode(prompt_text(r, branch)) for r in subset]
        lengths = torch.tensor([len(p) for p in pp], device="cuda")
        capacity = min(768, max(map(len, pp)) + max_new)
        x = torch.zeros((len(subset), capacity), dtype=torch.long, device="cuda")
        for i, p in enumerate(pp):
            x[i, : len(p)] = torch.tensor(p, device="cuda")
        done = torch.zeros(len(subset), device="cuda", dtype=torch.bool)
        ended = done.clone()
        outs = [[] for r in subset]
        scores, caches = cached_prefix(model, x[:, : int(lengths.max())], lengths, capacity)
        for step in range(max_new):
            active = ~done & (lengths < capacity)
            if not active.any():
                break
            # FP32 KV-cache decoding: no repeated full-context logits or BF16 argmax ties.
            scores[:, [0, 1, 2, 4, 5]] = -float("inf")
            v = scores.argmax(-1)
            active_list = active.cpu().tolist()
            values = v.cpu().tolist()
            for i, token in enumerate(values):
                if active_list[i]:
                    if token == 3:
                        ended[i] = True
                    else:
                        outs[i].append(token)
            positions = lengths.clamp(max=capacity - 1).clone()
            write = active & (v != 3)
            x[write, lengths[write]] = v[write]
            lengths[write] += 1
            done |= active & (v == 3)
            if step + 1 < max_new and ((~done) & (lengths < capacity)).any():
                scores = cached_step(model, v, positions, lengths, caches)
        for r, ids, end in zip(subset, outs, ended.cpu().tolist()):
            results.append(
                {
                    "sample_id": r["sample_id"],
                    "upper": r["upper_display"],
                    "lower": r["lower_display"],
                    "context": r.get("context", ""),
                    "reference": r.get("response", ""),
                    "generated": tok.decode(ids),
                    "ended_naturally": end,
                    "branch": branch,
                    "generation_method": "greedy_unconstrained_except_special_tokens",
                    "max_new_characters": max_new,
                }
            )
    return results
