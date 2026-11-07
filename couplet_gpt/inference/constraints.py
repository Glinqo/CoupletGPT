"""Train-phrase candidate augmentation, sourced knowledge and evidence-aware prosody.

The neural generator remains the locally pretrained GPT. Auxiliary candidate origins
are explicit. Dictionary/automatic scoring never certifies literary quality.
"""

import collections
import math
import statistics
import numpy as np
import torch
from couplet_gpt.inference.search import V2Engine
from couplet_gpt.inference.checks import rule_check_v3
from couplet_gpt.resources.phrase_statistics import PhraseScorer
from couplet_gpt.resources.special_patterns import KnowledgeSpecial
from couplet_gpt.resources.prosody import ProsodyV4
from couplet_gpt.resources.lexical_semantics import LexicalGrammarV4


def structural_pass(upper, lower):
    c = rule_check_v3(upper, lower)
    return c["structural_pass"]


def enrich_pool(group, phrase, knowledge, prosody, mode="modern_context", lexical=None):
    upper = group["upper"]
    for c in group["pool"]:
        c.setdefault("origin", "gpt_constrained_beam_or_sampling")
        c["phrase_features"] = phrase.features(upper, c["text"])
        c["prosody"] = prosody.inspect(upper, c["text"], mode, profile="even_soft")
        c["radical_check"] = knowledge.radical_check(upper, c["text"])
        c["checks_v3"] = rule_check_v3(upper, c["text"])
        if lexical is not None:
            c["lexical_features"] = lexical.features(upper, c["text"])
    for feature in ["alignment", "fluency", "specificity"]:
        values = [c["phrase_features"][feature] for c in group["pool"]]
        mean = statistics.mean(values)
        sd = max(statistics.pstdev(values), 1e-6)
        for c, v in zip(group["pool"], values):
            c.setdefault("phrase_z", {})[feature] = (v - mean) / sd
    # Preserve the agent's tune-frozen combined-score standardization.
    values = [
        c["phrase_features"]["alignment"]
        + 0.25 * c["phrase_features"]["fluency"]
        + 0.15 * c["phrase_features"]["specificity"]
        for c in group["pool"]
    ]
    mean = statistics.mean(values)
    sd = max(statistics.pstdev(values), 1e-6)
    for c, v in zip(group["pool"], values):
        c["phrase_combined_z"] = (v - mean) / sd
    if lexical is not None:
        values = [c["lexical_features"]["score"] for c in group["pool"]]
        mean = statistics.mean(values)
        sd = max(statistics.pstdev(values), 1e-6)
        for c, v in zip(group["pool"], values):
            c["lexical_z"] = (v - mean) / sd
    return group


def candidate_value(c, phrase_weight=0.5, tone_weight=0.1, teacher_weight=0.0, lexical_weight=0.0):
    return (
        c["forward_score"]
        + c["reverse_score"]
        + phrase_weight * c.get("phrase_combined_z", 0.0)
        + tone_weight * c.get("prosody", {}).get("score", 0.0)
        + teacher_weight * c.get("teacher_joint_logit", 0.0)
        + lexical_weight * c.get("lexical_z", 0.0)
    )


def select(
    group,
    phrase_weight=0.5,
    tone_weight=0.1,
    teacher_weight=0.0,
    special="auto",
    lexical_weight=0.0,
    predicate_profile="soft",
):
    spans = group.get("knowledge", {}).get("shared_radical_spans", [])
    whole = bool(
        len(spans) == 1 and spans[0]["start"] == 0 and spans[0]["stop"] == len(group["upper"])
    )
    require_special = special == "radical" or (special == "auto" and whole)
    candidates = [c for c in group["pool"] if c.get("checks_v3", {}).get("structural_pass", False)]
    # Explicit modern/ancient mode is fixed by caller. Unknown remains reviewable.
    candidates = [c for c in candidates if c.get("prosody", {}).get("foot_status") != "fail"]
    if require_special:
        candidates = [c for c in candidates if c.get("radical_check", {}).get("status") == "pass"]
    if predicate_profile == "verb_to_verb":
        candidates = [
            c
            for c in candidates
            if c.get("lexical_features", {}).get("predicate", {}).get("status")
            not in ("lexical_pos_mismatch", "possible_verb_mismatch")
        ]
    if not candidates:
        return {
            "prediction": None,
            "status": "no_supported_special_candidate"
            if require_special
            else "no_feasible_candidate",
            "full_literary_quality_verified": False,
            "special_constraint_applied": require_special,
        }
    chosen = max(
        candidates,
        key=lambda c: candidate_value(
            c, phrase_weight, tone_weight, teacher_weight, lexical_weight
        ),
    )
    historical = group.get("knowledge", {}).get("historical_lexical_evidence", []) or group.get(
        "knowledge", {}
    ).get("idiom_evidence", [])
    return {
        "prediction": chosen["text"],
        "origin": chosen["origin"],
        "status": "allusion_requires_review"
        if historical
        else "candidate_requires_literary_review",
        "score_components": {
            k: chosen.get(k)
            for k in [
                "forward_score",
                "reverse_score",
                "phrase_features",
                "lexical_features",
                "prosody",
                "teacher_scores",
            ]
        },
        "checks_v3": chosen["checks_v3"],
        "special_constraint_applied": require_special,
        "radical_check": chosen.get("radical_check"),
        "full_literary_quality_verified": False,
        "allusion_correctness_verified": False,
    }


class V4Engine(V2Engine):
    def __init__(self, forward, reverse=None, mode="modern_context"):
        super().__init__(forward, reverse)
        self.phrase = PhraseScorer.load()
        self.knowledge = KnowledgeSpecial()
        self.prosody = ProsodyV4()
        self.mode = mode
        self.lexical = LexicalGrammarV4()
        char_mode = "pingshui" if mode == "pingshui" else "modern_character"
        self.possible_ping = torch.tensor(
            [
                len(self.tok.reverse[i]) == 1
                and self.prosody.ending_tone(self.tok.reverse[i], char_mode) != "Z"
                for i in range(self.V)
            ],
            device="cuda",
        )

    def allowed(self, upper_ids, x, prefix_len, j, need_ping):
        # Remove the inherited modern-character-only filter for actual v4 generation.
        allow = super().allowed(upper_ids, x, prefix_len, j, torch.zeros_like(need_ping))
        uppers = [self.tok.decode(ids) for ids in upper_ids.cpu().tolist()]
        need = torch.tensor(
            [self.prosody.ending_tone(u, self.mode) == "Z" for u in uppers], device="cuda"
        )
        final_class = upper_ids[:, j] == upper_ids[:, -1]
        # Only exclude definite contradictions; contextual polyphony stays feasible.
        require = (need & final_class)[:, None, None]
        return allow & ((~require) | self.possible_ping[None, None, :])

    def scored_pools(self, uppers, pools):
        # A v4-legal ambiguous/contextual candidate need not pass v2's old filter.
        uu = []
        yy = []
        for u, pool in zip(uppers, pools):
            uu.extend([u] * len(pool))
            yy.extend(pool)
        fwd = self.log_scores(self.model, uu, yy)
        rev = (
            self.log_scores(self.reverse, uu, yy, True)
            if self.reverse is not None
            else [0.0] * len(uu)
        )
        result = []
        index = 0
        for u, pool in zip(uppers, pools):
            rows = []
            for y in pool:
                check = rule_check_v3(u, y)
                assert check["structural_pass"]
                rows.append(
                    {
                        "text": y,
                        "forward_score": fwd[index],
                        "reverse_score": rev[index],
                        "checks_v3": check,
                        "origin": "gpt_constrained_beam_or_sampling",
                    }
                )
                index += 1
            result.append(rows)
        return result

    @torch.inference_mode()
    def radical_gpt_candidates(self, upper, k=8, limit_groups=5):
        """Own GPT beam search with an explicit sourced radical plan; never a lookup answer."""
        analysis = self.knowledge.analyze(upper)
        spans = analysis["shared_radical_spans"]
        if not (len(spans) == 1 and spans[0]["start"] == 0 and spans[0]["stop"] == len(upper)):
            return []
        self.knowledge.shared_radical_candidates(upper, limit=1)
        bank = self.knowledge.phrase_bank or {}
        # Most training-attested 2-character material first, excluding the input group.
        groups = sorted(
            (r for r in bank if r != spans[0]["radical"]),
            key=lambda r: sum(p["count"] for p in bank[r].get("2", [])),
            reverse=True,
        )[:limit_groups]
        results = []
        n = len(upper)
        u = torch.tensor([self.tok.encode(upper)], device="cuda")
        prefix = torch.tensor([[2, 4] + self.tok.encode(upper) + [5]], device="cuda")
        p = prefix.shape[1]
        for radical in groups:
            plan = self.knowledge.radical_mask_plan(upper, self.tok.vocab, radical)
            if plan["status"] != "ready":
                continue
            mask = torch.zeros(self.V, dtype=torch.bool, device="cuda")
            mask[plan["position_allowed_ids"][0]] = True
            x = prefix[:, None, :]
            scores = torch.zeros((1, 1), device="cuda")
            for j in range(n):
                old = x.shape[1]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lp = (
                        self.model(x.reshape(old, -1))[:, -1]
                        .float()
                        .log_softmax(-1)
                        .reshape(1, old, self.V)
                    )
                # Keep only structural constraints here; sound assessed on complete words.
                allowed = (
                    self.allowed(u, x, p, j, torch.zeros(1, dtype=torch.bool, device="cuda"))
                    & mask[None, None, :]
                )
                vals, idx = (
                    (scores[:, :, None] + lp.masked_fill(~allowed, -float("inf")))
                    .reshape(1, -1)
                    .topk(k)
                )
                if not bool(torch.isfinite(vals).all()):
                    break
                parent = idx // self.V
                token = idx % self.V
                x = torch.cat(
                    [
                        x.gather(1, parent[:, :, None].expand(-1, -1, x.shape[-1])),
                        token[:, :, None],
                    ],
                    2,
                )
                scores = vals
            if x.shape[-1] != p + n:
                continue
            for ids in x[0, :, p:].cpu().tolist():
                results.append(
                    {
                        "text": self.tok.decode(ids),
                        "origin": "own_gpt_with_dictionary_radical_mask",
                        "target_radical": radical,
                        "constraint_evidence": plan["spans"],
                    }
                )
        return results

    @torch.inference_mode()
    def allusion_gpt_candidates(self, upper, k=6, limit=8):
        if not hasattr(self.knowledge, "allusion_prefix_plans"):
            return []
        plans = self.knowledge.allusion_prefix_plans(upper, limit=limit)
        if isinstance(plans, dict):
            plans = plans.get("plans", [])
        n = len(upper)
        u = torch.tensor([self.tok.encode(upper)], device="cuda")
        result = []
        prefix = [2, 4] + self.tok.encode(upper) + [5]
        p = len(prefix)
        for plan in plans:
            text = plan["prefix"]
            if len(text) >= n or any(c not in self.tok.vocab for c in text):
                continue
            if any(a == b for a, b in zip(upper, text)):
                continue
            if any(
                (upper[i] == upper[j]) != (text[i] == text[j])
                for i in range(len(text))
                for j in range(i)
            ):
                continue
            x = torch.tensor([[prefix + self.tok.encode(text)]], device="cuda")
            scores = torch.zeros((1, 1), device="cuda")
            for j in range(len(text), n):
                old = x.shape[1]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lp = (
                        self.model(x.reshape(old, -1))[:, -1]
                        .float()
                        .log_softmax(-1)
                        .reshape(1, old, self.V)
                    )
                allowed = self.allowed(u, x, p, j, torch.zeros(1, dtype=torch.bool, device="cuda"))
                vals, idx = (
                    (scores[:, :, None] + lp.masked_fill(~allowed, -float("inf")))
                    .reshape(1, -1)
                    .topk(k)
                )
                if not bool(torch.isfinite(vals).all()):
                    break
                parent = idx // self.V
                token = idx % self.V
                x = torch.cat(
                    [
                        x.gather(1, parent[:, :, None].expand(-1, -1, x.shape[-1])),
                        token[:, :, None],
                    ],
                    2,
                )
                scores = vals
            if x.shape[-1] != p + n:
                continue
            for ids in x[0, :, p:].cpu().tolist():
                result.append(
                    {
                        "text": self.tok.decode(ids),
                        "origin": "own_gpt_with_sourced_allusion_prefix",
                        "prefix_plan": plan,
                        "not_raw_unassisted_GPT": True,
                    }
                )
        return result

    @torch.inference_mode()
    def augment(
        self, groups, old_pools=None, phrase_limit=32, include_radical=True, include_allusion=False
    ):
        result = []
        for start in range(0, len(groups), 16):
            gg = groups[start : start + 16]
            if old_pools is None:
                assert len({len(g["upper"]) for g in gg}) == 1, "Group batches by length"
                base = self.scored_pools(
                    [g["upper"] for g in gg], self.candidates([g["upper"] for g in gg])
                )
            else:
                base = [old_pools[start + i]["pool"] for i in range(len(gg))]
            uu = []
            yy = []
            new_rows = []
            for g, original in zip(gg, base):
                u = g["upper"]
                seen = {c["text"] for c in original}
                new = []
                for p in self.phrase.propose_with_evidence(
                    u, [c["text"] for c in original], limit=phrase_limit
                ):
                    p = {**p, "origin": "train_aligned_phrase_edit_not_raw_GPT"}
                    if p["text"] not in seen and structural_pass(u, p["text"]):
                        seen.add(p["text"])
                        new.append(p)
                if include_radical:
                    radical = self.knowledge.shared_radical_candidates(u, limit=48)[
                        "candidates"
                    ] + self.radical_gpt_candidates(u)
                    for p in radical:
                        if (
                            p["text"] not in seen
                            and all(c in self.tok.vocab for c in p["text"])
                            and structural_pass(u, p["text"])
                        ):
                            seen.add(p["text"])
                            new.append(p)
                if include_allusion:
                    for p in self.allusion_gpt_candidates(u):
                        if p["text"] not in seen and structural_pass(u, p["text"]):
                            seen.add(p["text"])
                            new.append(p)
                new_rows.append(new)
                uu.extend([u] * len(new))
                yy.extend(c["text"] for c in new)
            fwd = self.log_scores(self.model, uu, yy) if uu else []
            rev = (
                self.log_scores(self.reverse, uu, yy, True)
                if uu and self.reverse is not None
                else [0.0] * len(uu)
            )
            cursor = 0
            for g, original, new in zip(gg, base, new_rows):
                for c in new:
                    c["forward_score"] = fwd[cursor]
                    c["reverse_score"] = rev[cursor]
                    cursor += 1
                group = {
                    **g,
                    "pool": [dict(c) for c in original] + new,
                    "source_pool_policy": "frozen_v3_legacy_strict"
                    if old_pools is not None
                    else "v4_mode_aware_permissive",
                    "knowledge": self.knowledge.analyze(g["upper"]),
                    "knowledge_examples": self.knowledge.retrieve_train_examples(
                        g["upper"], limit=3
                    ),
                    "phrase_diagnosis": self.phrase.diagnose(g["upper"]),
                }
                enrich_pool(
                    group, self.phrase, self.knowledge, self.prosody, self.mode, self.lexical
                )
                result.append(group)
        return result
