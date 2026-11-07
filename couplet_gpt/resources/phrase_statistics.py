"""Train-only phrase correspondences and collocation scorer; CPU and reference-free at inference.

This is a corpus compatibility aid, not a POS parser or expert semantic judge.
API: PhraseScorer.load(); features(upper,lower); score(upper,lower);
propose(upper, existing_lowers, limit=48). No validation/test references at inference.
"""

from couplet_gpt.paths import ROOT as PROJECT_ROOT
import argparse
import collections
import gzip
import hashlib
import json
import math
import pickle
import statistics
from pathlib import Path

ROOT = PROJECT_ROOT
OUT = ROOT / "runs/optimization_v4/syntax_semantics"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


class PhraseScorer:
    def __init__(self):
        self.pairs = {k: collections.Counter() for k in (1, 2, 3)}
        self.source = {k: collections.Counter() for k in (1, 2, 3)}
        self.target = {k: collections.Counter() for k in (1, 2, 3)}
        self.total = {k: 0 for k in (1, 2, 3)}
        self.lower = collections.Counter()
        self.options = {}

    def fit(self, rows):
        for r in rows:
            u, y = r["upper"], r["lower"]
            self.lower[y] += 1
            for k in (1, 2, 3):
                for j in range(len(u) - k + 1):
                    a, b = u[j : j + k], y[j : j + k]
                    self.pairs[k][a, b] += 1
                    self.source[k][a] += 1
                    self.target[k][b] += 1
                    self.total[k] += 1
        # Only supported phrase replacements, to avoid fitting singleton accidents.
        options = collections.defaultdict(list)
        for k in (2, 3):
            for (a, b), count in self.pairs[k].items():
                if count >= 3 and self.source[k][a] >= 5:
                    value = math.log((count + 0.5) / (self.source[k][a] + 1)) - 0.35 * math.log(
                        (self.target[k][b] + 1) / (self.total[k] + 1)
                    )
                    options[a].append((value, b, count))
        self.options = {a: sorted(v, reverse=True)[:12] for a, v in options.items()}
        return self

    def features(self, u, y):
        if len(u) != len(y):
            return {
                "alignment": -20.0,
                "phrase_support": -20.0,
                "fluency": -20.0,
                "specificity": -20.0,
            }
        # PMI-style source conditioning discounts universally common output phrases.
        conditional = []
        supports = []
        for k in (1, 2, 3):
            scores = []
            support = []
            for j in range(len(u) - k + 1):
                a, b = u[j : j + k], y[j : j + k]
                c = self.pairs[k][a, b]
                s = self.source[k][a]
                t = self.target[k][b]
                back = (t + 0.5) / (self.total[k] + 0.5 * max(1, len(self.target[k])))
                probability = (c + 10 * back) / (s + 10)
                # Weight weak/unseen source phrases toward neutral rather than hallucinate evidence.
                reliability = s / (s + 10)
                scores.append(
                    reliability
                    * (math.log(max(probability, 1e-12)) - 0.35 * math.log(max(back, 1e-12)))
                )
                support.append(math.log1p(c))
            conditional.append(statistics.mean(scores))
            supports.append(statistics.mean(support))
        # Collocation smooths rare spans rather than simply banning novel words.
        flu = []
        for k in (2, 3):
            for j in range(len(y) - k + 1):
                gram = y[j : j + k]
                prefix = gram[:-1]
                flu.append(
                    math.log(
                        (self.target[k][gram] + 0.2)
                        / (self.target[k - 1][prefix] + 0.2 * max(1, len(self.target[1])))
                    )
                )
        specificity = -math.log1p(self.lower[y])
        return {
            "alignment": 0.25 * conditional[0] + 0.5 * conditional[1] + 0.25 * conditional[2],
            "phrase_support": 0.25 * supports[0] + 0.5 * supports[1] + 0.25 * supports[2],
            "fluency": statistics.mean(flu),
            "specificity": specificity,
        }

    def score(self, u, y):
        f = self.features(u, y)
        return f["alignment"] + 0.25 * f["fluency"] + 0.15 * f["specificity"]

    def propose(self, upper, existing_lowers, limit=48):
        """Replace contiguous phrases of actual drafts with train-supported counterparts.

        Does not return retrieved whole training lower sentences. It preserves the
        character equality pattern and same-position avoidance; tone handled upstream.
        """
        return [r["text"] for r in self.propose_with_evidence(upper, existing_lowers, limit)]

    def diagnose(self, upper, lower=None):
        covered = set()
        evidence = []
        for k in (2, 3):
            for j in range(len(upper) - k + 1):
                source = upper[j : j + k]
                if source in self.options:
                    covered.update(range(j, j + k))
                    evidence.append(
                        {
                            "start": j,
                            "source": source,
                            "source_train_count": self.source[k][source],
                            "translation_options": len(self.options[source]),
                        }
                    )
        return {
            "supported_source_character_coverage": len(covered) / len(upper),
            "status": "supported_phrases" if evidence else "abstain_no_supported_source_phrase",
            "source_evidence": evidence,
            "whole_lower_in_train": None if lower is None else lower in self.lower,
            "scope": "Evidence for aligned spans, not expert word segmentation, facts, or allusion.",
        }

    def propose_with_evidence(self, upper, existing_lowers, limit=48):
        def structure(y):
            return (
                len(y) == len(upper)
                and all(a != b for a, b in zip(upper, y))
                and all(
                    (upper[i] == upper[j]) == (y[i] == y[j])
                    for i in range(len(y))
                    for j in range(i)
                )
            )

        original = set(existing_lowers)
        pool = {}
        for draft in list(existing_lowers)[:8]:
            if len(draft) != len(upper):
                continue
            for k in (2, 3):
                for j in range(len(upper) - k + 1):
                    for _, replacement, count in self.options.get(upper[j : j + k], [])[:5]:
                        y = draft[:j] + replacement + draft[j + k :]
                        if y not in original and structure(y):
                            pool.setdefault(y, []).append(
                                {
                                    "draft": draft,
                                    "start": j,
                                    "end": j + k,
                                    "source_phrase": upper[j : j + k],
                                    "replacement": replacement,
                                    "pair_train_count": count,
                                    "source_train_count": self.source[k][upper[j : j + k]],
                                    "target_train_count": self.target[k][replacement],
                                }
                            )
        chosen = sorted(pool, key=lambda y: (self.score(upper, y), y), reverse=True)[:limit]
        return [
            {
                "text": y,
                "phrase_score": self.score(upper, y),
                "edits": pool[y],
                "whole_lower_in_train": y in self.lower,
                "requires_GPT_rescoring": True,
                "expert_quality_verified": False,
            }
            for y in chosen
        ]

    def save(self, path=OUT / "phrase_model.pkl.gz"):
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wb", compresslevel=3) as f:
            pickle.dump({"version": 1, "state": self.__dict__}, f, protocol=5)

    @staticmethod
    def load(path=OUT / "phrase_model.pkl.gz"):
        with gzip.open(path, "rb") as f:
            payload = pickle.load(f)
        assert payload["version"] == 1
        model = PhraseScorer()
        model.__dict__.update(payload["state"])
        return model


def fit():
    source = ROOT / "data/processed/couplet_train.jsonl"
    model = PhraseScorer().fit(json.loads(x) for x in source.open(encoding="utf-8"))
    model.save()
    info = {
        "source": str(source.relative_to(ROOT)),
        "source_sha256": sha(source),
        "rows": sum(model.lower.values()),
        "pair_types": {k: len(v) for k, v in model.pairs.items()},
        "supported_source_phrases": len(model.options),
        "external_pretrained_weights": False,
        "expert_labels_used": False,
        "purpose": "Train-only aligned span compatibility and collocation; not full syntax/semantics certification.",
    }
    (OUT / "training_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(info, flush=True)
    return model


def evaluate(model):
    from sacrebleu.metrics import CHRF

    metric = CHRF()
    root = ROOT / "runs/optimization_v3"
    sets = {
        "tune": root / "tune_candidate_pools.json",
        "audit": root / "audit_seg_seed2026/candidate_pools.json",
    }
    all_results = {}
    rows_by_split = {}
    # At inference only model and candidates used; references appear solely in reporting.
    for split, path in sets.items():
        groups = json.loads(path.read_text(encoding="utf-8"))
        rows = []
        for g in groups:
            features = [model.features(g["upper"], p["text"]) for p in g["pool"]]
            rows.append(
                {
                    **g,
                    "corpus_features": features,
                    "proposals": model.propose(g["upper"], [p["text"] for p in g["pool"]]),
                }
            )
        rows_by_split[split] = rows
    grid = []
    for coefficient in (0.0, 0.1, 0.25, 0.5, 1.0):
        rr = {}
        for split, rows in rows_by_split.items():
            predictions = []
            sentence = []
            for g in rows:
                values = [
                    f["alignment"] + 0.25 * f["fluency"] + 0.15 * f["specificity"]
                    for f in g["corpus_features"]
                ]
                mu = statistics.mean(values)
                sd = max(statistics.pstdev(values), 1e-6)
                i = max(
                    range(len(g["pool"])),
                    key=lambda i: (
                        g["pool"][i]["forward_score"]
                        + g["pool"][i]["reverse_score"]
                        + coefficient * (values[i] - mu) / sd
                    ),
                )
                predictions.append(g["pool"][i]["text"])
                sentence.append(metric.sentence_score(predictions[-1], g["references"]).score)
            refs = [
                [g["references"][i] if i < len(g["references"]) else None for g in rows]
                for i in range(max(len(g["references"]) for g in rows))
            ]
            rr[split] = {
                "corpus_chrf": metric.corpus_score(predictions, refs).score,
                "mean_sentence_chrf": statistics.mean(sentence),
            }
        grid.append({"weight": coefficient, **rr})
    chosen = max(grid, key=lambda r: (r["tune"]["mean_sentence_chrf"], -r["weight"]))
    all_results = {
        "grid": grid,
        "chosen_by_tune": chosen,
        "note": "Audit is reporting only; lexical overlap is not human quality. These are reused validation distributions.",
    }
    for split, rows in rows_by_split.items():
        (OUT / f"{split}_features_proposals.json").write_text(
            json.dumps(rows, ensure_ascii=False), encoding="utf-8"
        )
    (OUT / "evaluation.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(all_results, ensure_ascii=False, indent=2), flush=True)


def ablate():
    from sacrebleu.metrics import CHRF

    sets = {
        s: json.loads((OUT / f"{s}_features_proposals.json").read_text(encoding="utf-8"))
        for s in ("tune", "audit")
    }
    modes = {
        "alignment": lambda f: f["alignment"],
        "fluency": lambda f: f["fluency"],
        "alignment_fluency": lambda f: f["alignment"] + 0.25 * f["fluency"],
        "full": lambda f: f["alignment"] + 0.25 * f["fluency"] + 0.15 * f["specificity"],
    }
    result = {}
    for name, feature in modes.items():
        grid = []
        for weight in (0.0, 0.1, 0.25, 0.5, 1.0):
            row = {"weight": weight}
            for split, groups in sets.items():
                texts = []
                scores = []
                metric = CHRF()
                for g in groups:
                    vv = [feature(f) for f in g["corpus_features"]]
                    mu = statistics.mean(vv)
                    sd = max(statistics.pstdev(vv), 1e-6)
                    i = max(
                        range(len(vv)),
                        key=lambda i: (
                            g["pool"][i]["forward_score"]
                            + g["pool"][i]["reverse_score"]
                            + weight * (vv[i] - mu) / sd
                        ),
                    )
                    texts.append(g["pool"][i]["text"])
                    scores.append(metric.sentence_score(texts[-1], g["references"]).score)
                refs = [
                    [g["references"][i] if i < len(g["references"]) else None for g in groups]
                    for i in range(max(len(g["references"]) for g in groups))
                ]
                row[split] = {
                    "mean_sentence_chrf": statistics.mean(scores),
                    "corpus_chrf": metric.corpus_score(texts, refs).score,
                }
            grid.append(row)
        result[name] = {
            "chosen_by_tune": max(
                grid, key=lambda r: (r["tune"]["mean_sentence_chrf"], -r["weight"])
            ),
            "grid": grid,
        }
    (OUT / "feature_ablation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def verify(model):
    reloaded = PhraseScorer.load()
    groups = json.loads((OUT / "audit_features_proposals.json").read_text(encoding="utf-8"))[:20]
    records = []
    for g in groups:
        drafts = [p["text"] for p in g["pool"]]
        proposals = model.propose_with_evidence(g["upper"], drafts)
        assert [p["text"] for p in proposals] == reloaded.propose(g["upper"], drafts)
        for p in proposals:
            assert p["text"] not in drafts and p["edits"]
            assert all(e["pair_train_count"] >= 3 for e in p["edits"])
        records.append(
            {
                "upper": g["upper"],
                "coverage": model.diagnose(g["upper"]),
                "proposal_count": len(proposals),
                "examples": proposals[:3],
            }
        )
    (OUT / "proposal_verification.json").write_text(
        json.dumps(
            {"status": "pass", "groups": 20, "examples": records, "portable_load": True},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--ablate", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    model = fit() if args.fit or not (OUT / "phrase_model.pkl.gz").exists() else PhraseScorer.load()
    if args.evaluate:
        evaluate(model)
    if args.ablate:
        ablate()
    if args.verify:
        verify(model)


if __name__ == "__main__":
    main()
