"""Variable-length final generator data and neutral experiment I/O."""

import collections, tarfile
import numpy as np
from couplet_gpt.data.datasets import ROOT, CharacterTokenizer
from couplet_gpt.utils import read, rows, dump, jsonl, sha, stable, normalize
from couplet_gpt.evaluation.metrics import han

OUT = ROOT / "runs/final_couplet_v7"
BASE = ROOT / "runs/syntax_attention_v6/main/baseline_seed2026/best_model.pt"
MAX_LENGTH = 62


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "data_ready.json").exists():
        return
    tok = CharacterTokenizer()
    data = {
        s: list(rows(ROOT / f"data/processed/couplet_{s}.jsonl"))
        for s in ["train", "validation", "test"]
    }
    extra = {"train": [], "test": []}
    reject = collections.Counter()
    seen = set()
    with tarfile.open(ROOT / "data/raw/couplet.tar.gz") as tf:
        names = tf.getnames()
        for split in ["test", "train"]:

            def lines(side):
                return (
                    tf.extractfile(next(n for n in names if n.endswith(f"/{split}/{side}.txt")))
                    .read()
                    .decode()
                    .splitlines()
                )

            for i, (a, b) in enumerate(zip(lines("in"), lines("out")), 1):
                a, b = ["".join(normalize(t).split()).strip("，。！？；：、,.!?;:") for t in [a, b]]
                if len(a) in (5, 7):
                    continue
                if not a or len(a) != len(b) or a == b or not all(han(c) for c in a + b):
                    reject["invalid"] += 1
                    continue
                if len(a) > MAX_LENGTH:
                    reject["context_limit"] += 1
                    continue
                if 1 in tok.encode(a + b):
                    reject["vocabulary_oov"] += 1
                    continue
                if (a, b) in seen:
                    reject["duplicate"] += 1
                    continue
                seen.add((a, b))
                extra[split].append(
                    {"id": f"{split}:{i}", "upper": a, "lower": b, "origin": "raw_length_extension"}
                )
    test_up = {r["upper"] for r in extra["test"]}
    data["test"] += extra["test"]
    for r in extra["train"]:
        if r["upper"] in test_up:
            reject["test_upper_overlap"] += 1
            continue
        s = "validation" if int(stable("final-v7:" + r["upper"])[:8], 16) % 100 < 5 else "train"
        data[s].append(r)
    held = {r["upper"] for s in ["validation", "test"] for r in data[s]}
    data["train"] = [r for r in data["train"] if r["upper"] not in held]
    stats = {}
    for s, rr in data.items():
        jsonl(OUT / f"{s}.jsonl", rr)
        stats[s] = {
            "rows": len(rr),
            "length_counts": dict(sorted(collections.Counter(len(r["upper"]) for r in rr).items())),
        }
    dump(
        OUT / "data_ready.json",
        {
            "status": "ready",
            "counts": stats,
            "rejected_extension": dict(reject),
            "vocab_unchanged": True,
            "model_context": 128,
            "maximum_single_line_characters": MAX_LENGTH,
            "old_splits_preserved": True,
            "extension_tests_may_have_prior_poetry_pretraining_exposure": True,
            "no_claim_of_fresh_independent_test": True,
            "base_sha256": sha(BASE),
        },
    )
    print("VARIABLE DATA", stats, flush=True)


def collate(records, reverse=False, device="cuda"):
    import torch

    tok = CharacterTokenizer()
    xx = []
    yy = []
    for r in records:
        a, b = (r["lower"], r["upper"]) if reverse else (r["upper"], r["lower"])
        a, b = tok.encode(a), tok.encode(b)
        seq = [2, 4] + a + [5] + b + [3]
        x = seq[:-1]
        y = [-100] * len(x)
        y[len(a) + 2 :] = b + [3]
        xx.append(x)
        yy.append(y)
    width = max(map(len, xx))
    x = np.zeros((len(xx), width), np.int64)
    y = np.full_like(x, -100)
    for i, (a, b) in enumerate(zip(xx, yy)):
        x[i, : len(a)] = a
        y[i, : len(b)] = b
    return torch.tensor(x, device=device), torch.tensor(y, device=device)
