"""Symmetric lexical reverse GPT and frozen final ablation protocol."""

import collections, gc, json, shutil, time
from dataclasses import asdict
from pathlib import Path
import numpy as np, torch
from couplet_gpt.data.context_cache import (
    ROOT,
    DATA,
    Contexts,
    read,
    rows,
    dump,
    jsonl,
    sha,
    collate,
    stable,
)
from couplet_gpt.modeling.couplet import FinalGPT
from couplet_gpt.modeling.alignment import AlignedGPT, load_model_v2
from couplet_gpt.modeling.gpt import GPTConfig
from couplet_gpt.data.datasets import CharacterTokenizer
from couplet_gpt.training.pretrain import seed_all, atomic_save

OUT = ROOT / "runs/final_couplet_v12"
V11 = ROOT / "runs/final_lexicon_v11"
LOCK = ROOT / "runs/data_continuation_v8"
FORWARD = ROOT / "models/couplet/forward.pt"
REVERSE = LOCK / "new_data/reverse/best.pt"
LEX = ROOT / "data/lexicon/model_lexicon.json"
BRANCHES = ("reverse_adapter", "reverse_aligned_control")


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        p = read(OUT / "protocol.json")
        for path, key in [
            (FORWARD, "forward_sha256"),
            (REVERSE, "reverse_sha256"),
            (LEX, "lexicon_sha256"),
            (DATA / "train.jsonl", "original_train_sha256"),
        ]:
            assert sha(path) == p[key]
        return p
    held = {r["lower"] for s in ("validation", "test") for r in rows(DATA / (s + ".jsonl"))}
    tr = [r for r in rows(DATA / "train.jsonl") if r["lower"] not in held]
    jsonl(OUT / "reverse_train.jsonl", tr)
    val = list(rows(LOCK / "validation_locked.jsonl"))
    gg = {}
    for r in val:
        g = gg.setdefault(r["upper"], {"upper": r["upper"], "references": []})
        if r["lower"] not in g["references"]:
            g["references"].append(r["lower"])
    calibration = []
    for n in sorted({len(u) for u in gg}):
        calibration += sorted(
            [g for g in gg.values() if len(g["upper"]) == n],
            key=lambda g: stable("v12-calibration:" + g["upper"]),
        )[: (40 if n in (5, 7) else 15)]
    jsonl(OUT / "calibration_groups.jsonl", calibration)
    p = {
        "status": "locked_before_training",
        "forward_sha256": sha(FORWARD),
        "reverse_sha256": sha(REVERSE),
        "lexicon_sha256": sha(LEX),
        "vocab_sha256": sha(ROOT / "tokenizer/vocab.json"),
        "original_train_sha256": sha(DATA / "train.jsonl"),
        "reverse_train_sha256": sha(OUT / "reverse_train.jsonl"),
        "validation_sha256": sha(LOCK / "validation_locked.jsonl"),
        "calibration_sha256": sha(OUT / "calibration_groups.jsonl"),
        "core_test_sha256": sha(V11 / "system_test_locked.jsonl"),
        "full_test_sha256": sha(LOCK / "test_locked.jsonl"),
        "training_pairs": len(tr),
        "reverse_source_overlap_removed": 944256 - len(tr),
        "reverse_inputs": "original lower only; target original upper",
        "historical_reverse_input_exposure_in_initial_checkpoint_not_ruled_out": True,
        "epochs": 3,
        "batch": 128,
        "lr": 2e-5,
        "weight_decay": 0.01,
        "betas": [0.9, 0.95],
        "clip": 1.0,
        "seed": 2026,
        "sampling": "same replacement orders np7201..7203; non5/7 weight4; local4096 length sorting",
        "loss": "conditional CE only, identical for adapter and aligned control",
        "initialization": "copy all matching old reverse GPT tensors; zero new alignment; adapter donor v11 with lexical_output zero to preserve old logits",
        "adapter_words": 202585,
        "branches": list(BRANCHES),
        "checkpoint_selection": "minimum reverse validation character CE; include epoch0",
        "calibration_grid": [0, 0.1, 0.25, 0.5, 0.75, 1.0],
        "calibration_objective": "corpus chrF on locked validation groups; ties smaller weight",
        "calibration_uses_checkpoint_validation": True,
        "test_for_selection": False,
        "test_historically_reported": True,
        "core_test": "same347 upper groups, all ablations",
        "confirmation_test": "same2617 historical upper groups, selected complete system and calibrated controls",
        "candidate_budget": "beam16+8 deterministic samples",
        "same_pool_for_reranking_ablation": True,
        "training_ablations": [
            "aligned reverse without lexical adapter",
            "full symmetric lexical reverse",
        ],
        "inference_knockouts": [
            "forward lexical fusion off",
            "forward contextual projection zero",
            "forward word residual zero",
            "forward alignment off",
            "forward old8192 matcher",
        ],
        "inference_knockouts_not_retrained": True,
        "single_seed": True,
        "encoder_frozen": True,
    }
    dump(OUT / "protocol.json", p)
    snapshot = OUT / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for path in (ROOT / "scripts").glob("*v12.py"):
        shutil.copy2(path, snapshot / path.name)
    return p


class ReverseContexts(Contexts):
    def __init__(self):
        super().__init__()
        self.lower_index = {}
        cache = ROOT / "data/cache/couplet_reverse"
        if (cache / "context_ready.json").exists():
            self.lower_values = np.load(cache / "reverse_contexts.npy", mmap_mode="r")
            self.lower_valid = np.load(cache / "reverse_context_valid.npy", mmap_mode="r")
            self.lower_index = {
                tuple(r["ids"]): (r["offset"], r["length"])
                for r in rows(cache / "reverse_context_index.jsonl")
            }

    def __call__(self, ids):
        keys = [tuple(r[2 : r.index(5)]) for r in ids.detach().cpu().tolist()]
        missing = list(
            dict.fromkeys(
                k
                for k in keys
                if k not in self.index
                and k not in self.extra
                and k not in self.lower_index
                and k not in self.online
            )
        )
        if missing:
            self.add(missing)
        n = max(map(len, keys))
        v = np.zeros((len(keys), n, 768), np.float16)
        valid = np.zeros((len(keys), n), np.int64)
        for i, k in enumerate(keys):
            if k in self.index:
                p, l = self.index[k]
                a, b = self.values[p : p + l], self.valid[p : p + l]
            elif k in self.extra:
                p, l = self.extra[k]
                a, b = self.extra_values[p : p + l], self.extra_valid[p : p + l]
            elif k in self.lower_index:
                p, l = self.lower_index[k]
                a, b = self.lower_values[p : p + l], self.lower_valid[p : p + l]
            else:
                a, b = self.online[k]
            v[i, : len(k)] = a
            valid[i, : len(k)] = b
        return torch.tensor(v, device=ids.device), torch.tensor(valid, device=ids.device)


def build_reverse_contexts():
    if (OUT / "context_ready.json").exists():
        return
    start = time.perf_counter()
    tok = CharacterTokenizer()
    old = Contexts()
    if not (OUT / "reverse_context_index.jsonl").exists():
        keys = {
            tuple(tok.encode(r["lower"]))
            for path in (OUT / "reverse_train.jsonl", LOCK / "validation_locked.jsonl")
            for r in rows(path)
        }
        missing = sorted(
            (k for k in keys if k not in old.index and k not in old.extra),
            key=lambda k: (len(k), k),
        )
        assert all(1 not in k for k in missing)
        index = []
        offset = 0
        for k in missing:
            index.append({"ids": list(k), "offset": offset, "length": len(k)})
            offset += len(k)
        jsonl(OUT / "reverse_context_index.jsonl", index)
        dump(
            OUT / "context_plan.json",
            {
                "new_sources": len(index),
                "reused_sources": len(keys) - len(index),
                "characters": offset,
                "source": "original training/validation lowers only",
            },
        )
    index = list(rows(OUT / "reverse_context_index.jsonl"))
    plan = read(OUT / "context_plan.json")
    done = (
        read(OUT / "context_progress.json")["completed_sources"]
        if (OUT / "context_progress.json").exists()
        else 0
    )
    mode = "r+" if (OUT / "reverse_contexts.npy").exists() else "w+"
    v = np.lib.format.open_memmap(
        OUT / "reverse_contexts.npy", mode=mode, dtype=np.float16, shape=(plan["characters"], 768)
    )
    masks = np.lib.format.open_memmap(
        OUT / "reverse_context_valid.npy", mode=mode, dtype=np.uint8, shape=(plan["characters"],)
    )
    print("REVERSE CONTEXT PLAN", plan, "resume", done, flush=True)
    for i in range(done, len(index), 128):
        group = index[i : i + 128]
        kk = [tuple(r["ids"]) for r in group]
        old.add(kk)
        for k, r in zip(kk, group):
            a, b = old.online[k]
            p, l = r["offset"], r["length"]
            v[p : p + l] = a
            masks[p : p + l] = b
        old.online.clear()
        if i // 128 % 40 == 0 or i + 128 >= len(index):
            v.flush()
            masks.flush()
            progress = {
                "completed_sources": min(i + 128, len(index)),
                "total_sources": len(index),
                "seconds": time.perf_counter() - start,
            }
            dump(OUT / "context_progress.json", progress)
            print("REVERSE CACHE", progress, flush=True)
    v.flush()
    masks.flush()
    dump(
        OUT / "context_ready.json",
        {
            "status": "complete",
            **plan,
            "index_sha256": sha(OUT / "reverse_context_index.jsonl"),
            "seconds": time.perf_counter() - start,
            "no_target_upper_features": True,
        },
    )
    del old, v, masks
    gc.collect()
    torch.cuda.empty_cache()


def initialize_reverse(branch):
    old = torch.load(REVERSE, map_location="cpu", weights_only=True)
    config = GPTConfig(**old["model_config"])
    if branch == "reverse_adapter":
        donor = torch.load(FORWARD, map_location="cpu", weights_only=True)
        m = FinalGPT(config, donor["lexicon_entries"], "chinesebert")
        state = m.state_dict()
        for k in state:
            if k in donor["model"] and donor["model"][k].shape == state[k].shape:
                state[k] = donor["model"][k].clone()
        for k in state:
            if k.startswith("alignment.") or k == "lexical_output.weight":
                state[k].zero_()
    else:
        m = AlignedGPT(config)
        state = m.state_dict()
    for k, v in old["model"].items():
        assert k in state and state[k].shape == v.shape
        state[k] = v.clone()
    m.load_state_dict(state)
    m = m.cuda()
    if branch == "reverse_adapter":
        m.context_provider = ReverseContexts()
    return m, old


def save_model(m, path, branch, epoch, score):
    atomic_save(
        path,
        {
            "architecture": "symmetric_lexical_reverse_v12"
            if branch == "reverse_adapter"
            else "aligned",
            "model": m.state_dict(),
            "model_config": asdict(m.config),
            "lexicon_entries": getattr(m, "entries", None),
            "direction": "lower_to_upper",
            "feature_source": "chinesebert" if branch == "reverse_adapter" else None,
            "epoch": epoch,
            "validation": score,
            "base_checkpoint": str(REVERSE),
            "base_sha256": sha(REVERSE),
            "max_single_line_length": 62,
            "source_only": True,
        },
    )


def load_reverse(branch="reverse_adapter", path=None):
    ck = torch.load(
        path
        or (
            ROOT / "models/couplet/reverse.pt"
            if branch == "reverse_adapter"
            else OUT / branch / "best.pt"
        ),
        map_location="cpu",
        weights_only=True,
    )
    if ck["architecture"] == "symmetric_lexical_reverse_v12":
        m = FinalGPT(GPTConfig(**ck["model_config"]), ck["lexicon_entries"], "chinesebert")
        m.load_state_dict(ck["model"])
        m = m.cuda()
        m.context_provider = ReverseContexts()
    else:
        m = load_model_v2(ck)
    return m.eval(), ck
