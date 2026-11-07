"""Frozen protocol and compatible loader for expanded lexical final GPT."""

import collections, functools, hashlib, json, shutil
from dataclasses import asdict
import numpy as np
import torch
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
from couplet_gpt.modeling.gpt import GPTConfig
from couplet_gpt.training.pretrain import atomic_save, seed_all

OUT = ROOT / "runs/final_lexicon_v11"
BASE = ROOT / "runs/data_continuation_v8/new_data"
LEX = ROOT / "runs/lexicon_expansion_v10/model_lexicon.json"
LOCK = ROOT / "runs/data_continuation_v8"


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        p = read(OUT / "protocol.json")
        assert sha(LEX) == p["lexicon_sha256"] and sha(BASE / "forward/best.pt") == p["base_sha256"]
        assert sha(DATA / "train.jsonl") == p["training_sha256"]
        return p
    assert read(LEX.parent / "verification.json")["status"] == "pass"
    entries = read(LEX)
    ck = torch.load(BASE / "forward/best.pt", map_location="cpu", weights_only=True)
    old = ck["lexicon_entries"]
    assert entries[: len(old)] == old
    assert [r["id"] for r in entries] == list(range(1, len(entries) + 1))
    assert len({r["word"] for r in entries}) == len(entries)
    # Evaluation sources are fixed before any updated weights exist.
    groups = list(rows(LOCK / "test_locked.jsonl"))
    system = []
    for n in sorted({len(g["upper"]) for g in groups}):
        gg = sorted(
            [g for g in groups if len(g["upper"]) == n],
            key=lambda g: stable("v11-system:" + g["upper"]),
        )
        system += gg[: (50 if n in (5, 7) else 20)]
    jsonl(OUT / "system_test_locked.jsonl", system)
    p = {
        "status": "locked_before_training",
        "base_sha256": sha(BASE / "forward/best.pt"),
        "reverse_sha256": sha(BASE / "reverse/best.pt"),
        "lexicon_sha256": sha(LEX),
        "training_sha256": sha(DATA / "train.jsonl"),
        "validation_sha256": sha(LOCK / "validation_locked.jsonl"),
        "test_sha256": sha(LOCK / "test_locked.jsonl"),
        "system_test_sha256": sha(OUT / "system_test_locked.jsonl"),
        "vocab_sha256": sha(ROOT / "tokenizer/vocab.json"),
        "old_words": len(old),
        "expanded_words": len(entries),
        "epochs": 3,
        "examples_per_epoch": 944256,
        "batch": 128,
        "lr": 2e-5,
        "weight_decay": 0.01,
        "betas": [0.9, 0.95],
        "seed": 2026,
        "sampling": "same np7201..7203 replacement, non5/7 weight4, local4096 length sorting",
        "forward_loss": "CE + .5 structure; gradients clipped1; bf16 autocast",
        "branches": ["expanded", "old_lexicon_control"],
        "reverse_reused": True,
        "encoder_frozen_source_only": True,
        "new_rows_initialization": "zero residual; contextual span feature from frozen ChineseBERT",
        "selection": "minimum locked validation char CE, including epoch0; test never selects checkpoints",
        "same_examples_and_steps_not_same_parameters_or_FLOPs": True,
        "raw_test_groups": len(groups),
        "system_test_groups": len(system),
        "system_decode": "v4 beam16+8sample, forward+same reverse scores, unchanged rules",
        "test_previously_used_for_reporting": True,
        "no_claim_of_pristine_test_or_expert_literary_quality": True,
    }
    dump(OUT / "protocol.json", p)
    snapshot = OUT / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for n in (
        "final_lexicon_v11.py",
        "train_final_lexicon_v11.py",
        "evaluate_final_lexicon_v11.py",
        "generate_final_lexicon_v11.py",
        "model_final_v7.py",
        "model_lexicon_v5.py",
        "build_lexicon_v5.py",
        "train_final_v7.py",
        "generate_final_v7.py",
        "continue_v8_common.py",
    ):
        path = ROOT / "scripts" / n
        if path.exists():
            shutil.copy2(path, snapshot / n)
    return p


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    m = FinalGPT(GPTConfig(**ck["model_config"]), ck["lexicon_entries"], "chinesebert")
    m.load_state_dict(ck["model"])
    m = m.cuda()
    m.context_provider = Contexts()
    return m, ck


def initialize(branch):
    ck = torch.load(BASE / "forward/best.pt", map_location="cpu", weights_only=True)
    if branch == "expanded":
        entries = read(LEX)
        old = ck["lexicon_entries"]
        assert entries[: len(old)] == old
        state = dict(ck["model"])
        previous = state["word_residual.weight"]
        expanded = previous.new_zeros(len(entries) + 1, previous.shape[1])
        expanded[: len(previous)] = previous
        state["word_residual.weight"] = expanded
        for k, v in ck["model"].items():
            assert torch.equal(
                v, state[k][: len(previous)] if k == "word_residual.weight" else state[k]
            )
        ck = {**ck, "model": state, "lexicon_entries": entries}
    m = FinalGPT(GPTConfig(**ck["model_config"]), ck["lexicon_entries"], "chinesebert")
    m.load_state_dict(ck["model"])
    m = m.cuda()
    m.context_provider = Contexts()
    return m, ck


def checkpoint(m, epoch, validation, branch):
    return {
        "architecture": "variable_lexical_gpt_v11",
        "model": m.state_dict(),
        "model_config": asdict(m.config),
        "lexicon_entries": m.entries,
        "feature_source": "chinesebert",
        "epoch": epoch,
        "validation": validation,
        "branch": branch,
        "base_checkpoint": str(BASE / "forward/best.pt"),
        "base_sha256": sha(BASE / "forward/best.pt"),
        "lexicon_sha256": sha(LEX) if branch == "expanded" else None,
        "max_single_line_length": 62,
        "source_only": True,
    }


def release_model(branch="expanded", mode="modern_context"):
    m, _ = load(OUT / branch / "best.pt")
    m.eval()
    from couplet_gpt.modeling.alignment import load_model_v2
    from couplet_gpt.inference.couplet import FinalEngine

    r = load_model_v2(torch.load(BASE / "reverse/best.pt", map_location="cpu", weights_only=True))
    return FinalEngine(m, r, mode)
