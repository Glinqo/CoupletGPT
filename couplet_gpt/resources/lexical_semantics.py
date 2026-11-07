"""Sourced HowNet POS/sememe lexical lattices. Not a classical-Chinese parser.

Never treats dictionary word classes as expert sentence-quality labels. Default
scores are soft. A strict verb-to-verb profile is an explicit optional constraint.
"""

from couplet_gpt.paths import ROOT as PROJECT_ROOT
import argparse
import collections
import functools
import gzip
import hashlib
import io
import json
import math
import pathlib
import pickle
import re
import statistics
from opencc import OpenCC

ROOT = PROJECT_ROOT
OUT = ROOT / "data/resources/lexical_grammar"


class PrimitiveUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError("Only primitive dictionary data is permitted")


def build():
    src = OUT / "upstream/HowNet_dict_complete"
    raw = PrimitiveUnpickler(io.BytesIO(src.read_bytes())).load()
    cc = OpenCC("t2s")
    index = collections.defaultdict(dict)
    for r in raw.values():
        word = cc.convert(r["ch_word"])
        pos = r["ch_grammar"]
        definition = r["Def"]
        if not 1 <= len(word) <= 4 or not pos or pos in ("char", "pun", "letter", "prefix"):
            continue
        heads = re.findall(r'[A-Za-z][A-Za-z0-9_-]*\|[^\s:},\[\]"=]+', definition)
        # Deduplicate duplicated dictionary senses, preserving lexical ambiguity.
        key = (pos, definition)
        index[word][key] = {
            "pos": pos,
            "head": heads[0] if heads else None,
            "definition": definition,
            "sense_id": r["No"],
        }
    parent = collections.defaultdict(set)
    for line in (
        (OUT / "upstream/sememe_triples_taxonomy.txt").read_text(encoding="utf-8").splitlines()
    ):
        parts = line.split()
        if len(parts) == 3 and parts[1] == "hypernym":
            parent[parts[0]].add(parts[2])
    data = {
        "lexicon": {w: list(v.values()) for w, v in index.items()},
        "parents": {k: sorted(v) for k, v in parent.items()},
    }
    with gzip.open(OUT / "lexicon.json.gz", "wt", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return {
        "original_senses": len(raw),
        "lexical_words": len(index),
        "taxonomy_nodes": len(parent),
        "dictionary_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        "source": "https://github.com/thunlp/OpenHowNet",
        "supervision": "Lexical dictionary, no expert sentence labels",
    }


class LexicalGrammarV4:
    def __init__(self, path=None):
        with gzip.open(path or OUT / "lexicon.json.gz", "rt", encoding="utf-8") as f:
            data = json.load(f)
        self.lexicon = data["lexicon"]
        self.parents = data["parents"]

    @functools.lru_cache(maxsize=100000)
    def lookup(self, word):
        senses = self.lexicon.get(word, [])
        counts = collections.Counter(r["pos"] for r in senses)
        total = sum(counts.values())
        return {
            "word": word,
            "known": bool(senses),
            "pos": sorted(counts),
            "pos_distribution": {k: v / total for k, v in counts.items()},
            "sememe_heads": sorted({r["head"] for r in senses if r["head"]}),
            "sense_ids": [r["sense_id"] for r in senses],
            "lexical_not_context_disambiguated": True,
        }

    @functools.lru_cache(maxsize=16000)
    def ancestors(self, head):
        seen = {head}
        front = {head}
        for _ in range(6):
            nxt = {p for c in front for p in self.parents.get(c, [])} - seen
            if not nxt:
                break
            seen.update(nxt)
            front = nxt
        return frozenset(seen)

    def _sememe(self, a, b):
        if not a["sememe_heads"] or not b["sememe_heads"]:
            return 0.0
        return max(
            len(self.ancestors(x) & self.ancestors(y))
            / max(1, len(self.ancestors(x) | self.ancestors(y)))
            for x in a["sememe_heads"]
            for y in b["sememe_heads"]
        )

    @functools.lru_cache(maxsize=50000)
    def segment(self, text):
        # Maximal dictionary coverage and multi-character lexical evidence; no forced 2/2/3.
        dp = [(-1e9, []) for _ in range(len(text) + 1)]
        dp[0] = (0.0, [])
        for i in range(len(text)):
            for k in range(1, min(3, len(text) - i) + 1):
                word = text[i : i + k]
                known = self.lookup(word)["known"]
                if k > 1 and not known:
                    continue
                value = (k + 0.22 * (k - 1)) if known else -0.5
                candidate = (dp[i][0] + value - 0.05, dp[i][1] + [(i, i + k, word)])
                if candidate[0] > dp[i + k][0]:
                    dp[i + k] = candidate
        return tuple(dp[-1][1])

    @functools.lru_cache(maxsize=100000)
    def predicate_check(self, upper, lower, profile="soft"):
        if not upper or not lower:
            return {"status": "unknown", "reason": "empty"}
        us = self.segment(upper)
        ys = self.segment(lower)
        u = self.lookup(us[-1][2])
        y = self.lookup(ys[-1][2])
        uv = u["pos_distribution"].get("verb", 0.0)
        if not u["known"] or not y["known"]:
            status = "unknown"
        elif uv < 0.6 and profile != "verb_to_verb":
            status = (
                "possible_verb_mismatch"
                if uv > 0 and "verb" not in y["pos"]
                else "not_applicable_source_not_predominantly_verb"
            )
        elif not uv:
            status = "not_applicable_source_has_no_verb_evidence"
        elif "verb" in y["pos"]:
            status = "pass"
        else:
            status = "lexical_pos_mismatch"
        return {
            "status": status,
            "source_final_word": u,
            "target_final_word": y,
            "profile": profile,
            "source_verb_share": uv,
            "strict_profile_pass": status == "pass" if profile == "verb_to_verb" else None,
            "not_universal_literary_rule": True,
            "note": "Ancient adjectival predicates and other valid patterns exist; soft by default.",
        }

    @functools.lru_cache(maxsize=100000)
    def features(self, upper, lower):
        if len(upper) != len(lower) or not upper:
            return {"score": -2.0, "status": "length_mismatch", "coverage": 0.0}
        n = len(upper)
        dp = [(-1e9, []) for _ in range(n + 1)]
        dp[0] = (0.0, [])
        for i in range(n):
            for k in range(1, min(3, n - i) + 1):
                a = self.lookup(upper[i : i + k])
                b = self.lookup(lower[i : i + k])
                if k > 1 and not (a["known"] or b["known"]):
                    continue
                pos = sum(
                    min(v, b["pos_distribution"].get(p, 0.0))
                    for p, v in a["pos_distribution"].items()
                )
                sem = self._sememe(a, b)
                cov = int(a["known"]) + int(b["known"])
                value = k * (0.55 * pos + 0.15 * sem + 0.1 * cov) + 0.08 * (k - 1) * (
                    a["known"] and b["known"]
                )
                row = {
                    "start": i,
                    "end": i + k,
                    "upper": a,
                    "lower": b,
                    "pos_alignment": pos,
                    "sememe_family_similarity": sem,
                }
                candidate = (dp[i][0] + value, dp[i][1] + [row])
                if candidate[0] > dp[i + k][0]:
                    dp[i + k] = candidate
        blocks = dp[-1][1]
        pos = sum((r["end"] - r["start"]) * r["pos_alignment"] for r in blocks) / n
        sem = sum((r["end"] - r["start"]) * r["sememe_family_similarity"] for r in blocks) / n
        cov = (
            sum(
                (r["end"] - r["start"]) * (r["upper"]["known"] and r["lower"]["known"])
                for r in blocks
            )
            / n
        )
        useg = self.segment(upper)
        yseg = self.segment(lower)
        ub = {r[1] for r in useg[:-1]}
        yb = {r[1] for r in yseg[:-1]}
        boundary = 2 * len(ub & yb) / max(1, len(ub) + len(yb)) if ub or yb else 1.0
        pred = self.predicate_check(upper, lower)
        penalty = (
            0.15
            if pred["status"] == "lexical_pos_mismatch"
            else 0.075
            if pred["status"] == "possible_verb_mismatch"
            else 0.0
        )
        value = 0.55 * pos + 0.2 * boundary + 0.15 * sem + 0.1 * cov - penalty
        return {
            "score": value,
            "pos_alignment": pos,
            "sememe_family_similarity": sem,
            "coverage": cov,
            "word_boundary_agreement": boundary,
            "source_segmentation": [r[2] for r in useg],
            "target_segmentation": [r[2] for r in yseg],
            "aligned_blocks": blocks,
            "predicate": pred,
            "status": "lexical_evidence_only",
            "full_syntax_or_semantics_verified": False,
        }

    def score(self, upper, lower):
        return self.features(upper, lower)["score"]

    def compact_features(self, upper, lower):
        f = self.features(upper, lower)
        keep = [
            "score",
            "pos_alignment",
            "sememe_family_similarity",
            "coverage",
            "word_boundary_agreement",
            "source_segmentation",
            "target_segmentation",
            "status",
            "full_syntax_or_semantics_verified",
        ]
        result = {k: f[k] for k in keep if k in f}
        if "predicate" in f:
            result["predicate_status"] = f["predicate"]["status"]
        return result

    def filter_candidates(self, upper, candidates, profile="soft"):
        result = []
        for c in candidates:
            text = c if isinstance(c, str) else c["text"]
            check = self.predicate_check(upper, text, profile)
            if profile == "verb_to_verb" and check["status"] == "lexical_pos_mismatch":
                continue
            result.append(
                {
                    "text": text,
                    "lexical_features": self.features(upper, text),
                    "requires_GPT_rescoring": True,
                }
            )
        return result


def evaluate(model):
    from sacrebleu.metrics import CHRF

    root = ROOT / "runs/optimization_v3"
    metric = CHRF()
    sets = {}
    for name, path in [
        ("tune", root / "tune_candidate_pools.json"),
        ("audit", root / "audit_seg_seed2026/candidate_pools.json"),
    ]:
        groups = json.loads(path.read_text(encoding="utf-8"))
        rows = []
        for g in groups:
            vals = [model.score(g["upper"], p["text"]) for p in g["pool"]]
            mu = statistics.mean(vals)
            sd = max(statistics.pstdev(vals), 1e-6)
            rows.append((g, [(v - mu) / sd for v in vals]))
        sets[name] = rows
    grid = []
    for weight in (0.0, 0.1, 0.25, 0.5, 1.0):
        entry = {"weight": weight}
        for name, rows in sets.items():
            hyp = []
            sent = []
            for g, vals in rows:
                i = max(
                    range(len(vals)),
                    key=lambda i: (
                        g["pool"][i]["forward_score"]
                        + g["pool"][i]["reverse_score"]
                        + weight * vals[i]
                    ),
                )
                y = g["pool"][i]["text"]
                hyp.append(y)
                sent.append(metric.sentence_score(y, g["references"]).score)
            refs = [
                [g["references"][i] if i < len(g["references"]) else None for g, _ in rows]
                for i in range(max(len(g["references"]) for g, _ in rows))
            ]
            entry[name] = {
                "corpus_chrf": metric.corpus_score(hyp, refs).score,
                "mean_sentence_chrf": statistics.mean(sent),
            }
        grid.append(entry)
    best = max(grid, key=lambda r: (r["tune"]["mean_sentence_chrf"], -r["weight"]))
    result = {
        "grid": grid,
        "chosen_by_tune": best,
        "note": "Reused validation lexical proxy, not expert syntax gold.",
    }
    (OUT / "evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if args.build:
        (OUT / "build_info.json").write_text(json.dumps(build(), indent=2), encoding="utf-8")
    model = LexicalGrammarV4()
    assert model.lookup("森")["pos"] == ["adj"]
    assert "verb" in model.lookup("横")["pos"]
    assert (
        model.predicate_check("江河湖海波涛涌", "松柏梧桐枝杈森", "verb_to_verb")["status"]
        == "lexical_pos_mismatch"
    )
    assert (
        model.predicate_check("江河湖海波涛涌", "松柏梧桐枝杈横", "verb_to_verb")["status"]
        == "pass"
    )
    assert model.lookup("㐀㐁㐂")["known"] is False
    checks = {
        "status": "pass",
        "controlled_dictionary_checks": 5,
        "word_class_examples": {w: model.lookup(w) for w in ["森", "横", "涌", "江河", "松柏"]},
        "no_expert_labels": True,
        "source": "https://github.com/thunlp/OpenHowNet",
    }
    (OUT / "checks.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("LEXICAL CHECKS PASS", flush=True)
    if args.evaluate:
        print(json.dumps(evaluate(model), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
