"""Direct full-poetry SFT, sharing only the unchanged self-built GPT primitives."""

import json, math, random, time
from dataclasses import asdict
import numpy as np
import torch
from torch.nn import functional as F
from couplet_gpt.training.appreciation_common import (
    ROOT,
    BASE,
    GPT,
    GPTConfig,
    hidden,
    dump,
    sha,
    save,
    rows,
    jsonl,
    stable,
    metrics,
)
from couplet_gpt.data.appreciation_prompts import OUT, prompt
from couplet_gpt.resources.prosody_annotation import ToneAnnotator


class Tokenizer:
    def __init__(self):
        self.vocab = json.loads((OUT / "vocab.json").read_text(encoding="utf8"))
        self.reverse = {v: k for k, v in self.vocab.items()}

    def encode(self, s):
        return [self.vocab.get(c, 1) for c in s]

    def decode(self, ids):
        return "".join(self.reverse[int(i)] for i in ids if i >= 6)


class Corpus:
    def __init__(self):
        self.tokens = np.memmap(OUT / "tokens.bin", mode="r", dtype=np.uint16)
        for field in ["offsets", "prefix_lengths", "lengths", "response_lengths", "split_codes"]:
            setattr(self, field, np.load(OUT / (field + ".npy"), mmap_mode="r"))
        self.train_indices = np.flatnonzero(self.split_codes == 0)
        self.tok = Tokenizer()

    def collate(self, ii, device="cuda"):
        width = int(self.lengths[ii].max())
        x = np.zeros((len(ii), width), np.int64)
        y = np.full(x.shape, -100, np.int64)
        for j, i in enumerate(ii):
            seq = self.tokens[int(self.offsets[i]) : int(self.offsets[i + 1])]
            n = len(seq) - 1
            p = int(self.prefix_lengths[i]) - 1
            x[j, :n] = seq[:-1]
            y[j, p:n] = seq[p + 1 :]
        return torch.tensor(x, device=device), torch.tensor(y, device=device)


def initialize():
    torch.manual_seed(20261106)
    torch.cuda.manual_seed_all(20261106)
    ck = torch.load(BASE, map_location="cpu", weights_only=True)
    tok = Tokenizer()
    c = GPTConfig(**{**ck["model_config"], "context_length": 1536, "vocab_size": len(tok.vocab)})
    m = GPT(c)
    missing, extra = m.load_state_dict(
        {
            k: v
            for k, v in ck["model"].items()
            if k not in ["tok_emb.weight", "head.weight", "pos_emb.weight"]
        },
        strict=False,
    )
    assert set(missing) == {"tok_emb.weight", "head.weight", "pos_emb.weight"} and not extra
    with torch.no_grad():
        m.tok_emb.weight[: ck["model_config"]["vocab_size"]].copy_(ck["model"]["tok_emb.weight"])
        m.pos_emb.weight[:128].copy_(ck["model"]["pos_emb.weight"])
    return m.cuda()


def nll(m, x, y):
    valid = y != -100
    h = hidden(m, x)[valid]
    target = y[valid]
    return F.cross_entropy(F.linear(h, m.head.weight).float(), target, reduction="sum")


@torch.inference_mode()
def evaluate(m, data, ii):
    m.eval()
    num = den = unknown = 0
    for j in range(0, len(ii), 4):
        x, y = data.collate(ii[j : j + 4])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = nll(m, x, y)
        num += float(loss)
        den += int((y != -100).sum())
        unknown += int((y == 1).sum())
    return {
        "response_ce": num / den,
        "rows": len(ii),
        "supervised_tokens": den,
        "unknown_target_tokens": unknown,
    }


@torch.inference_mode()
def decode(m, rr, tok, max_new=512, batch=4):
    from couplet_gpt.modeling.cached_decoding import cached_prefix, cached_step

    m.eval()
    out = []
    for j in range(0, len(rr), batch):
        selected = rr[j : j + batch]
        pp = [
            [2] + tok.encode(prompt(r["source_text"], r["program_first_sentence"]))
            for r in selected
        ]
        lens = torch.tensor([len(p) for p in pp], device="cuda")
        capacity = min(m.config.context_length, int(lens.max()) + max_new)
        x = torch.zeros((len(pp), int(lens.max())), device="cuda", dtype=torch.long)
        for i, p in enumerate(pp):
            x[i, : len(p)] = torch.tensor(p, device="cuda")
        scores, cache = cached_prefix(m, x, lens, capacity)
        done = torch.zeros(len(pp), device="cuda", dtype=torch.bool)
        ended = done.clone()
        bodies = [[] for _ in pp]
        for k in range(max_new):
            active = ~done & (lens < capacity)
            if not active.any():
                break
            scores[:, [0, 1, 2, 4, 5]] = -float("inf")
            v = scores.argmax(-1)
            for i, (live, t) in enumerate(zip(active.tolist(), v.tolist())):
                if live and t != 3:
                    bodies[i].append(t)
            ended |= active & (v == 3)
            positions = lens.clamp(max=capacity - 1).clone()
            lens[active & (v != 3)] += 1
            done |= active & (v == 3)
            if k + 1 < max_new and ((~done) & (lens < capacity)).any():
                scores = cached_step(m, v, positions, lens, cache)
        for r, body, finished in zip(selected, bodies, ended.tolist()):
            s = tok.decode(body)
            out.append(
                {
                    **r,
                    "generated_body": s,
                    "display_response": r["program_first_sentence"] + s,
                    "ended_naturally": finished,
                    "generation": "greedy_no_repetition_repair",
                    "program_prefix_is_not_neural_output": True,
                }
            )
    return out


def order(data):
    ix = data.train_indices.copy()
    np.random.default_rng(20261106).shuffle(ix)
    for j in range(0, len(ix), 2048):
        ix[j : j + 2048] = ix[j : j + 2048][
            np.argsort(data.lengths[ix[j : j + 2048]], kind="stable")
        ]
    return ix
