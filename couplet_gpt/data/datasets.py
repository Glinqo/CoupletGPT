"""CPU-friendly readers; no model training is started by this module."""

from __future__ import annotations
from couplet_gpt.paths import ROOT as PROJECT_ROOT
import json
from pathlib import Path
import numpy as np

ROOT = PROJECT_ROOT


class CharacterTokenizer:
    def __init__(self, root=ROOT):
        self.vocab = json.loads((Path(root) / "tokenizer/vocab.json").read_text(encoding="utf-8"))
        self.reverse = {v: k for k, v in self.vocab.items()}

    def encode(self, normalized_text):
        """Input must already use the recorded normalization policy."""
        return [self.vocab.get(c, 1) for c in normalized_text]

    def decode(self, ids):
        return "".join(self.reverse[int(i)] for i in ids)


class PretrainDataset:
    def __init__(self, source, split="train", root=ROOT):
        p = Path(root) / "data/pretraining/tokenized" / f"pretrain_{source}_{split}"
        self.tokens = np.memmap(str(p) + ".bin", dtype="<u4", mode="r")
        self.windows = np.load(str(p) + "_windows.npy", mmap_mode="r")
        self.offsets = np.load(str(p) + "_offsets.npy", mmap_mode="r")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        start, n = map(int, self.windows[index])
        x = np.zeros(128, dtype=np.int64)
        y = np.full(128, -100, dtype=np.int64)
        x[:n] = self.tokens[start : start + n]
        y[:n] = self.tokens[start + 1 : start + n + 1]
        return {
            "input_ids": x,
            "labels": y,
            "attention_mask": x != 0,
            "position_ids": np.arange(128, dtype=np.int64),
        }


class CoupletDataset:
    def __init__(self, split="train", root=ROOT):
        self.data = np.load(Path(root) / "data/tokenized" / f"couplet_{split}.npz")
        self.x = self.data["input_ids"]
        self.y = self.data["labels"]
        self.lengths = self.data["lengths"]
        self.all_known = self.data["all_known"]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return {
            "input_ids": self.x[index].astype(np.int64),
            "labels": self.y[index].astype(np.int64),
            "attention_mask": self.x[index] != 0,
            "position_ids": np.arange(128, dtype=np.int64),
        }


class MixedPretrainStream:
    """Sampling with replacement. Correct source probabilities by mean body tokens/window.

    Target 30/30/10/30 is an expectation, not an exact guarantee per finite epoch.
    Report the observed body-token totals during actual training.
    """

    def __init__(self, seed=2026, root=ROOT):
        self.rng = np.random.default_rng(seed)
        cfg = json.loads((Path(root) / "configs/pretraining.json").read_text(encoding="utf-8"))
        stats = json.loads(
            (Path(root) / "data/pretraining/statistics.json").read_text(encoding="utf-8")
        )
        weights = cfg["pretrain_body_token_weights"]
        self.sources = list(weights)
        self.datasets = {s: PretrainDataset(s, root=root) for s in self.sources}
        q = []
        for s in self.sources:
            info = stats[f"pretrain_{s}_train"]
            mean = info["body_tokens"] / info["windows"]
            q.append(weights[s] / mean)
        self.probabilities = np.asarray(q) / sum(q)
        self.observed_body_tokens = {s: 0 for s in self.sources}

    def __iter__(self):
        return self

    def __next__(self):
        source = str(self.rng.choice(self.sources, p=self.probabilities))
        ds = self.datasets[source]
        result = ds[int(self.rng.integers(len(ds)))]
        self.observed_body_tokens[source] += int((result["labels"] >= 6).sum())
        result["source"] = source
        return result


def torch_batch(samples):
    """Example collator for torch.utils.data.DataLoader; leave sources in metadata."""
    import torch

    return {
        k: torch.from_numpy(np.stack([r[k] for r in samples]))
        for k in ("input_ids", "labels", "attention_mask", "position_ids")
    }


if __name__ == "__main__":
    stream = MixedPretrainStream()
    samples = [next(stream) for _ in range(8)]
    print("pretrain batch", {k: tuple(v.shape) for k, v in torch_batch(samples).items()})
    ds = CoupletDataset("test")
    print("couplet batch", {k: tuple(v.shape) for k, v in torch_batch([ds[0], ds[1]]).items()})
