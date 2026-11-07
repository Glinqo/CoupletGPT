"""Explicit three-state checks and an optional learned complete-pair scorer."""

import functools
import numpy as np
import torch
from pypinyin import pinyin, Style
from couplet_gpt.inference.search import V2Engine, rule_check


@functools.lru_cache(maxsize=100000)
def context_foot_hint(text):
    if not text:
        return {"reading": None, "tone": "?"}
    reading = pinyin(text, style=Style.TONE3, heteronym=False, neutral_tone_with_five=True)[-1][0]
    return {
        "reading": reading,
        "tone": "P" if reading[-1:] in ("1", "2") else "Z" if reading[-1:] in ("3", "4") else "?",
    }


def rule_check_v3(upper, lower):
    old = rule_check(upper, lower)
    structural = old["equal_han_length"] and old["no_same_position"] and old["repeat_pattern"]
    a, b = old["upper_modern_tone"], old["lower_modern_tone"]
    if a == "Z" and b == "P":
        foot = "pass"
    elif a == "Z" and b == "Z":
        foot = "fail"
    elif a == "P":
        foot = "not_applicable"
    else:
        foot = "unknown"
    status = "fail" if not structural or foot == "fail" else "pass" if foot == "pass" else "unknown"
    return {
        "equal_han_length": old["equal_han_length"],
        "no_same_position": old["no_same_position"],
        "repeat_pattern": old["repeat_pattern"],
        "structural_pass": structural,
        "modern_foot_status": foot,
        "combined_proxy_status": status,
        "upper_single_character_tone": a,
        "lower_single_character_tone": b,
        "context_hint": {
            "upper": context_foot_hint(upper),
            "lower": context_foot_hint(lower),
            "verified": False,
            "used_to_assert_pass": False,
        },
        "legacy_strict_proxy_pass": old["encoded_rules_pass"],
        "full_traditional_rules_verified": False,
    }


@torch.inference_mode()
def pair_scores(model, uppers, lowers, tok):
    result = []
    for start in range(0, len(uppers), 256):
        seqs = [
            [2, 4] + tok.encode(u) + [5] + tok.encode(y) + [3]
            for u, y in zip(uppers[start : start + 256], lowers[start : start + 256])
        ]
        x = torch.zeros((len(seqs), max(map(len, seqs))), device="cuda", dtype=torch.long)
        for i, s in enumerate(seqs):
            x[i, : len(s)] = torch.tensor(s, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s = model(x)
        result.extend(s.float().cpu().tolist())
    return result


def attach_pair_scores(pools, model, tok):
    uu = []
    yy = []
    for g in pools:
        uu.extend([g["upper"]] * len(g["pool"]))
        yy.extend(c["text"] for c in g["pool"])
    scores = pair_scores(model, uu, yy, tok)
    cursor = 0
    for g in pools:
        pp = scores[cursor : cursor + len(g["pool"])]
        cursor += len(pp)
        mu = float(np.mean(pp))
        sd = max(float(np.std(pp)), 1e-6)
        for c, s in zip(g["pool"], pp):
            c["pair_score"] = s
            c["pair_standardized"] = (s - mu) / sd
    return pools


def select_candidate(pool, pair_weight=0.0):
    return max(
        pool,
        key=lambda c: (
            c["forward_score"] + c["reverse_score"] + pair_weight * c.get("pair_standardized", 0.0)
        ),
    )
