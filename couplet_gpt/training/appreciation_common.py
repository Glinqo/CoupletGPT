"""Independent commentary SFT utilities; original generation models remain unchanged."""

from couplet_gpt.paths import ROOT as PROJECT_ROOT
import gzip, hashlib, json, os, random, re, time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from couplet_gpt.modeling.gpt import GPT, GPTConfig
from couplet_gpt.resources.prosody_annotation import ToneAnnotator, han, text

ROOT = PROJECT_ROOT
DATA = ROOT / "runs/poetry_appreciation_v3"
OUT = ROOT / "runs/appreciation_sft_v3"
BASE = ROOT / "runs/pretrain_v1/artifacts/best_model.pt"


def dump(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(v, ensure_ascii=False, indent=2), encoding="utf8")
    os.replace(tmp, p)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def stable(s):
    return hashlib.sha256(s.encode("utf8")).hexdigest()


def rows(p):
    with (
        gzip.open(p, "rt", encoding="utf8")
        if str(p).endswith(".gz")
        else Path(p).open(encoding="utf8")
    ) as f:
        for s in f:
            if s.strip():
                yield json.loads(s)


def jsonl(p, values):
    with p.open("w", encoding="utf8") as f:
        for v in values:
            f.write(json.dumps(v, ensure_ascii=False, separators=(",", ":")) + "\n")


def save(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, p)


def prompt(r):
    task = "对联" if r["kind"] == "couplet" else "诗文"
    return (
        f"任务：{task}赏析。\n原文："
        + r["source_text"]
        + "\n字音依据："
        + r["program_first_sentence"]
        + "\n请结合原文的具体字词，写一段连贯短评，说明内容与表达结构，不编造出处。\n赏析正文："
    )


class Tokenizer:
    def __init__(self):
        self.vocab = json.loads((OUT / "vocab.json").read_text(encoding="utf8"))
        self.reverse = {v: k for k, v in self.vocab.items()}

    def encode(self, s):
        return [self.vocab.get(c, 1) for c in s]

    def decode(self, ids):
        return "".join(self.reverse[int(i)] for i in ids if int(i) >= 6)


class Corpus:
    def __init__(self):
        self.tokens = np.load(OUT / "tokens.npy", mmap_mode="r")
        self.offsets = np.load(OUT / "offsets.npy", mmap_mode="r")
        self.prefix = np.load(OUT / "prefix_lengths.npy", mmap_mode="r")
        self.lengths = np.diff(self.offsets) - 1
        self.index = list(rows(OUT / "index.jsonl"))
        self.tok = Tokenizer()

    def collate(self, indices, device="cuda"):
        width = int(max(self.lengths[i] for i in indices))
        x = np.zeros((len(indices), width), np.int64)
        y = np.full(x.shape, -100, np.int64)
        for j, i in enumerate(indices):
            seq = self.tokens[self.offsets[i] : self.offsets[i + 1]]
            n = len(seq) - 1
            start = int(self.prefix[i]) - 1
            x[j, :n] = seq[:-1]
            y[j, start:n] = seq[start + 1 :]
        return torch.tensor(x, device=device), torch.tensor(y, device=device)


def initialize(device="cuda"):
    torch.manual_seed(20261106)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20261106)
    ck = torch.load(BASE, map_location="cpu", weights_only=True)
    tok = Tokenizer()
    c = GPTConfig(**{**ck["model_config"], "context_length": 768, "vocab_size": len(tok.vocab)})
    m = GPT(c)
    old = ck["model"]
    missing, extra = m.load_state_dict(
        {
            k: v
            for k, v in old.items()
            if k not in ["tok_emb.weight", "head.weight", "pos_emb.weight"]
        },
        strict=False,
    )
    assert set(missing) == {"tok_emb.weight", "head.weight", "pos_emb.weight"} and not extra
    with torch.no_grad():
        m.tok_emb.weight[: ck["model_config"]["vocab_size"]].copy_(old["tok_emb.weight"])
        m.pos_emb.weight[: ck["model_config"]["context_length"]].copy_(old["pos_emb.weight"])
    assert m.head.weight is m.tok_emb.weight
    return m.to(device)


def hidden(m, x):
    h = m.drop(m.tok_emb(x) + m.pos_emb(torch.arange(x.shape[1], device=x.device)))
    for b in m.blocks:
        h = b(h)
    return m.norm(h)


def nll(m, x, y):
    # Avoid allocating vocabulary logits for prompt positions, whose loss is masked.
    keep = y != -100
    h = hidden(m, x)[keep]
    labels = y[keep]
    return F.cross_entropy(F.linear(h, m.head.weight).float(), labels, reduction="sum")


@torch.inference_mode()
def evaluate(m, data, indices, micro=4):
    m.eval()
    num = den = unknown = 0
    for start in range(0, len(indices), micro):
        x, y = data.collate(indices[start : start + micro])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = nll(m, x, y)
        num += float(loss)
        den += int((y != -100).sum())
        unknown += int((y == 1).sum())
    return {
        "response_ce": num / den,
        "response_tokens": den,
        "rows": len(indices),
        "unknown_target_tokens": unknown,
    }


@torch.inference_mode()
def decode(m, records, tok, max_new=260, batch=4, temperature=0.0, seed=20261106):
    from couplet_gpt.modeling.cached_decoding import cached_prefix, cached_step

    m.eval()
    result = []
    rng = torch.Generator(device="cuda").manual_seed(seed)
    for start in range(0, len(records), batch):
        rr = records[start : start + batch]
        pp = [[2] + tok.encode(prompt(r)) for r in rr]
        lengths = torch.tensor([len(p) for p in pp], device="cuda")
        capacity = min(768, max(map(len, pp)) + max_new)
        x = torch.zeros((len(rr), int(lengths.max())), device="cuda", dtype=torch.long)
        for i, p in enumerate(pp):
            x[i, : len(p)] = torch.tensor(p, device="cuda")
        scores, caches = cached_prefix(m, x, lengths, capacity)
        done = torch.zeros(len(rr), device="cuda", dtype=torch.bool)
        ended = done.clone()
        outs = [[] for _ in rr]
        for step in range(max_new):
            active = ~done & (lengths < capacity)
            if not active.any():
                break
            scores[:, [0, 1, 2, 4, 5]] = -float("inf")
            if temperature:
                vals, ix = torch.topk(scores / temperature, 40, dim=-1)
                v = ix.gather(1, torch.multinomial(vals.softmax(-1), 1, generator=rng)).squeeze(1)
            else:
                v = scores.argmax(-1)
            for i, (live, c) in enumerate(zip(active.tolist(), v.tolist())):
                if live and c != 3:
                    outs[i].append(c)
            ended |= active & (v == 3)
            positions = lengths.clamp(max=capacity - 1).clone()
            lengths[active & (v != 3)] += 1
            done |= active & (v == 3)
            if step + 1 < max_new and ((~done) & (lengths < capacity)).any():
                scores = cached_step(m, v, positions, lengths, caches)
        for r, ids, e in zip(rr, outs, ended.tolist()):
            body = tok.decode(ids)
            result.append(
                {
                    **r,
                    "generated_body": body,
                    "display_response": r["program_first_sentence"] + body,
                    "ended_naturally": e,
                    "generation": "greedy"
                    if temperature == 0
                    else "top40_temperature" + str(temperature),
                    "program_prefix_is_not_neural_output": True,
                }
            )
    return result


def metrics(pred):
    from collections import Counter
    from sacrebleu.metrics import CHRF

    out = {"rows": len(pred)}
    severe = anchored = both = ended = lengthok = 0
    lengths = []
    for r in pred:
        s = r["generated_body"]
        lengths.append(len(s))
        ended += r["ended_naturally"]
        lengthok += 120 <= len(s) <= 220
        grams = [s[j : j + 4] for j in range(max(0, len(s) - 3))]
        c = Counter(grams)
        severe += bool(
            grams
            and (max(c.values()) >= 8 or (max(c.values()) >= 4 and 1 - len(c) / len(grams) > 0.4))
        )
        source = han(r["source_text"])
        anchored += any(source[j : j + 2] in s for j in range(len(source) - 1))
        if r["kind"] == "couplet":
            both += all(
                any(side[j : j + 2] in s for j in range(len(side) - 1))
                for side in [han(r["upper"]), han(r["lower"])]
            )
    nc = sum(r["kind"] == "couplet" for r in pred)
    out.update(
        {
            "natural_EOS_rate": ended / len(pred),
            "body_120_220_rate": lengthok / len(pred),
            "severe_fourgram_repetition_rate": severe / len(pred),
            "literal_source_bigram_anchor_rate": anchored / len(pred),
            "both_couplet_sides_literal_anchor_rate": both / nc if nc else None,
            "median_body_characters": float(np.median(lengths)),
            "chrf_reference_similarity_not_literary_score": CHRF()
            .corpus_score([r["generated_body"] for r in pred], [[r["response_body"] for r in pred]])
            .score,
        }
    )
    return out
