"""Validation-only bottleneck probes; legacy test used solely for metric-code audit."""

import collections, functools, hashlib, json, math, pathlib
import numpy as np
import torch
from torch.nn import functional as F
from pypinyin import pinyin, Style
from sacrebleu.metrics import CHRF
from couplet_gpt.data.datasets import ROOT, CoupletDataset, CharacterTokenizer
from couplet_gpt.modeling.gpt import GPT, GPTConfig
from couplet_gpt.training.transfer_comparison import evaluate, batch
from couplet_gpt.evaluation.metrics import decode_all, han
from couplet_gpt.training.pretrain import json_write

OUT = ROOT / "runs/diagnostics_v2"


@functools.lru_cache(None)
def tone(c):
    options = pinyin(c, style=Style.TONE3, heteronym=True, neutral_tone_with_five=True)
    categories = set()
    for reading in options[0] if options else []:
        categories.add(
            "P" if reading[-1:] in ("1", "2") else "Z" if reading[-1:] in ("3", "4") else "?"
        )
    return next(iter(categories)) if len(categories) == 1 and "?" not in categories else "?"


def repeated(s):
    return len(set(s)) < len(s)


def same_pattern(a, b):
    return len(a) == len(b) and all(
        (a[i] == a[j]) == (b[i] == b[j]) for i in range(len(a)) for j in range(i)
    )


def rule_metrics(pairs):
    total = eligible = pass_foot = known_foot = known_opposite = even_pairs = even_opposite = (
        rep
    ) = rep_ok = aligned = 0
    for u, d in pairs:
        total += 1
        a, b = tone(u[-1]), tone(d[-1]) if d else "?"
        if a != "?" and b != "?":
            known_foot += 1
            known_opposite += a != b
        if a == "Z" and b != "?":
            eligible += 1
            pass_foot += b == "P"
        for i in range(1, min(len(u), len(d)), 2):
            x, y = tone(u[i]), tone(d[i])
            if x != "?" and y != "?":
                even_pairs += 1
                even_opposite += x != y
        if repeated(u):
            rep += 1
            rep_ok += same_pattern(u, d)
        aligned += any(x == y for x, y in zip(u, d))
    return {
        "n": total,
        "known_both_foot": known_foot,
        "known_both_foot_coverage": known_foot / max(1, total),
        "known_foot_opposition_rate": known_opposite / max(1, known_foot),
        "upper_Z_lower_known_n": eligible,
        "lower_P_given_upper_Z_rate": pass_foot / max(1, eligible),
        "even_position_known_pairs": even_pairs,
        "even_position_opposition_proxy": even_opposite / max(1, even_pairs),
        "repeated_upper_n": rep,
        "repeated_upper_pattern_match_rate": rep_ok / max(1, rep),
        "any_same_position_character_rate": aligned / max(1, total),
    }


def read_split(s):
    return [
        json.loads(l) for l in (ROOT / f"data/processed/couplet_{s}.jsonl").open(encoding="utf-8")
    ]


def get_probe(rows, each=256):
    grouped = {}
    for i, r in enumerate(rows):
        g = grouped.setdefault(r["upper"], {"upper": r["upper"], "references": [], "index": i})
        if r["lower"] not in g["references"]:
            g["references"].append(r["lower"])
    result = []
    for n in (5, 7):
        candidates = [g for g in grouped.values() if len(g["upper"]) == n]
        result += sorted(
            candidates, key=lambda g: hashlib.sha256(("2032:" + g["upper"]).encode()).hexdigest()
        )[:each]
    return result


def load_model(condition):
    ck = torch.load(
        ROOT / f"runs/couplet_comparison_v1/{condition}_seed2026/best_model.pt",
        map_location="cpu",
        weights_only=True,
    )
    model = GPT(GPTConfig(**ck["model_config"])).cuda().eval()
    model.load_state_dict(ck["model"])
    return model


def metric_audit():
    base = ROOT / "runs/couplet_comparison_v1/pretrained_seed2026"
    pp = [json.loads(l) for l in (base / "test_predictions_raw.jsonl").open(encoding="utf-8")]
    refs = [
        [p["references"][i] if i < len(p["references"]) else None for p in pp]
        for i in range(max(len(p["references"]) for p in pp))
    ]
    c = CHRF()
    score = c.corpus_score([r["prediction"] for r in pp], refs).score
    before = str(c.get_signature())
    for r in pp:
        c.sentence_score(r["prediction"], r["references"])
    after = str(c.get_signature())
    saved = json.loads((base / "test_metrics.json").read_text(encoding="utf-8"))
    first = (
        CHRF().corpus_score([r["prediction"] for r in pp], [[r["references"][0] for r in pp]]).score
    )
    assert abs(score - saved["generation"]["raw"]["chrf"]) < 1e-9
    return {
        "recomputed_multi_ref_score": score,
        "saved_score": saved["generation"]["raw"]["chrf"],
        "signature_immediately_after_corpus": before,
        "signature_after_sentence_scoring": after,
        "first_reference_only_score": first,
        "groups_with_multiple_references": sum(len(r["references"]) > 1 for r in pp),
        "diagnosis": "corpus score used all references; later sentence_score calls overwrite signature state. Metadata timing bug, not loss of references.",
    }


@torch.inference_mode()
def position_probe(model, ds, probe):
    stats = {}
    for n in (5, 7):
        selected = [g for g in probe if len(g["upper"]) == n]
        indices = np.asarray([g["index"] for g in selected])
        x, y = batch(ds, indices, "cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = model(x).float()
        pref = x[:, : n + 3].clone()
        target = torch.stack([y[:, n + 2 + j] for j in range(n)], dim=1)
        entries = []
        for j in range(n):
            teacher = z[:, n + 2 + j]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                free = model(pref)[:, -1].float()
            chosen = free.clone()
            chosen[:, [0, 1, 2, 3, 4, 5]] = float("-inf")
            pred = chosen.argmax(-1)
            entries.append(
                {
                    "position": j + 1,
                    "teacher_char_ce": float(F.cross_entropy(teacher, target[:, j])),
                    "free_prefix_reference_ce": float(F.cross_entropy(free, target[:, j])),
                    "teacher_top1_accuracy": float(
                        (teacher.argmax(-1) == target[:, j]).float().mean()
                    ),
                    "free_top1_reference_accuracy": float((pred == target[:, j]).float().mean()),
                    "teacher_top10_reference_coverage": float(
                        (teacher.topk(10).indices == target[:, j, None]).any(-1).float().mean()
                    ),
                }
            )
            pref = torch.cat([pref, pred[:, None]], 1)
        stats[str(n)] = entries
    return stats


def data_audit(train, validation):
    lowers = collections.Counter(r["lower"] for r in train)
    trainset = set(lowers)
    templates = ["清风", "明月", "风月", "山水", "春风", "秋雨", "人间", "岁月", "风光"]
    summary = {
        "train_reference_rules": rule_metrics((r["upper"], r["lower"]) for r in train),
        "validation_reference_rules": rule_metrics((r["upper"], r["lower"]) for r in validation),
        "train_distinct_lowers": len(lowers),
        "train_lower_distinct_fraction": len(lowers) / len(train),
        "train_most_frequent_lowers": lowers.most_common(20),
        "train_reference_template_line_rates": {
            t: sum(t in r["lower"] for r in train) / len(train) for t in templates
        },
    }
    return summary, trainset


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    json_write(OUT / "metric_audit.json", metric_audit())
    print("METRIC AUDIT COMPLETE", flush=True)
    train = read_split("train")
    validation = read_split("validation")
    probe = get_probe(validation)
    json_write(OUT / "validation_probe_512.json", probe)
    audit, trainset = data_audit(train, validation)
    json_write(OUT / "data_audit.json", audit)
    print("DATA AUDIT COMPLETE", flush=True)
    ds = CoupletDataset("validation")
    original = ds.x
    shuffled = original.copy()
    rng = np.random.default_rng(2031)
    for n in (5, 7):
        idx = np.where(ds.lengths == 2 * n + 3)[0]
        order = rng.permutation(idx)
        shuffled[idx, 2 : 2 + n] = original[order, 2 : 2 + n]
    result = {}
    for condition in ("scratch", "pretrained"):
        model = load_model(condition)
        ds.x = original
        a = evaluate(model, ds)
        ds.x = shuffled
        b = evaluate(model, ds)
        ds.x = original
        generated = decode_all(model, probe, "raw", CharacterTokenizer())
        altered = []
        for n in (5, 7):
            group = [g for g in probe if len(g["upper"]) == n]
            order = np.random.default_rng(2031).permutation(len(group))
            altered += [group[i] for i in order]
        other = decode_all(model, altered, "raw", CharacterTokenizer())
        # Decode_all maps actual upper; align by the known permutation via positional list output.
        unchanged = sum(a["prediction"] == b["prediction"] for a, b in zip(generated, other)) / len(
            probe
        )
        stats = {
            "original_validation": a,
            "same_length_shuffled_validation": b,
            "char_ce_gap": b["char_ce"] - a["char_ce"],
            "generated_unchanged_rate_on_changed_uppers": unchanged,
            "position_probe": position_probe(model, ds, probe),
            "generated_rules": rule_metrics(
                (g["upper"], p["prediction"]) for g, p in zip(probe, generated)
            ),
            "generated_unique_ratio": len({p["prediction"] for p in generated}) / len(generated),
            "generated_lower_seen_in_training_rate": sum(
                p["prediction"] in trainset for p in generated
            )
            / len(generated),
        }
        result[condition] = stats
        json_write(
            OUT / f"{condition}_validation_generated_512.json",
            [{**g, **p} for g, p in zip(probe, generated)],
        )
        print(condition, "conditioning charCE gap", stats["char_ce_gap"], flush=True)
        del model
        torch.cuda.empty_cache()
    json_write(OUT / "conditioning_and_positions.json", result)
    print("READ-ONLY DIAGNOSTICS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
