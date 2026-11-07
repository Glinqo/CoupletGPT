"""Paired locked-source evaluation of original v7, new-data and matched-old-data continuation."""

import argparse, collections, gc, time
import numpy as np, torch
from couplet_gpt.data.context_cache import *
from couplet_gpt.evaluation.generation import raw
from couplet_gpt.evaluation.metrics import score_predictions
from couplet_gpt.training.initial_couplet import evaluate
from couplet_gpt.inference.couplet import FinalEngine
from couplet_gpt.modeling.alignment import load_model_v2
from couplet_gpt.inference.checks import rule_check_v3
from couplet_gpt.inference.search import lexical_score


def paths(branch):
    return (
        (PREVIOUS / "forward/best.pt", PREVIOUS / "reverse/best.pt")
        if branch == "original_v7"
        else (OUT / branch / "forward/best.pt", OUT / branch / "reverse/best.pt")
    )


def groups_from(records):
    g = {}
    for r in records:
        a = g.setdefault(r["upper"], {"upper": r["upper"], "references": []})
        if r["lower"] not in a["references"]:
            a["references"].append(r["lower"])
    return list(g.values())


def stats_for(groups, pred):
    s, detail = score_predictions(groups, pred)
    s["structure_proxy_pass_rate"] = float(
        np.mean(
            [
                rule_check_v3(g["upper"], p["prediction"])["structural_pass"]
                for g, p in zip(groups, pred)
            ]
        )
    )
    s["jieba_pos_boundary_proxy_mean"] = float(
        np.mean([lexical_score(g["upper"], p["prediction"]) for g, p in zip(groups, pred)])
    )
    s["by_length"] = {}
    for n in sorted({len(g["upper"]) for g in groups}):
        ii = [i for i, g in enumerate(groups) if len(g["upper"]) == n]
        s["by_length"][str(n)] = score_predictions([groups[i] for i in ii], [pred[i] for i in ii])[
            0
        ]
    for label, condition in [
        ("five_seven", lambda n: n in (5, 7)),
        ("other_lengths", lambda n: n not in (5, 7)),
        ("eight_or_more", lambda n: n >= 8),
    ]:
        ii = [i for i, g in enumerate(groups) if condition(len(g["upper"]))]
        if ii:
            s[label] = score_predictions([groups[i] for i in ii], [pred[i] for i in ii])[0]
    return s, detail


def system(engine, groups):
    result = {}
    start = time.perf_counter()
    for n in sorted({len(g["upper"]) for g in groups}):
        chosen = [g for g in groups if len(g["upper"]) == n]
        for i in range(0, len(chosen), 8):
            u = [g["upper"] for g in chosen[i : i + 8]]
            pools = engine.scored_pools(u, engine.candidates(u, k=16, nsamples=8))
            for upper, pool in zip(u, pools):
                c = max(pool, key=lambda r: r["forward_score"] + r["reverse_score"])
                result[upper] = {
                    "prediction": c["text"],
                    "ended_naturally": None,
                    "forced_length": True,
                }
        print("DECODE LENGTH", n, len(chosen), round(time.perf_counter() - start, 1), flush=True)
    return [result[g["upper"]] for g in groups]


def evaluate_branch(branch, groups, legacy):
    directory = OUT / "evaluation" / branch
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "metrics.json").exists():
        return read(directory / "metrics.json")
    fpath, rpath = paths(branch)
    m, _ = model(fpath)
    m.eval()
    metrics = {}
    for label, gg in [("locked_test", groups), ("legacy_test", legacy)]:
        p = directory / (label + "_raw.jsonl")
        if p.exists():
            pred = list(rows(p))
        else:
            pred = raw(m, gg)
        s, detail = stats_for(gg, pred)
        jsonl(p, detail)
        metrics[label] = {"raw": s}
        print("RAW", branch, label, s["chrf"], s["pure_han_length"], flush=True)
    metrics["locked_test"]["teacher_forced"] = evaluate(
        m, [{"upper": g["upper"], "lower": y} for g in groups for y in g["references"]]
    )
    rev = load_model_v2(torch.load(rpath, map_location="cpu", weights_only=True))
    engine = FinalEngine(m, rev)
    for label, gg in [("locked_test", groups), ("legacy_test", legacy)]:
        p = directory / (label + "_system.jsonl")
        pred = list(rows(p)) if p.exists() else system(engine, gg)
        s, detail = stats_for(gg, pred)
        jsonl(p, detail)
        metrics[label]["system"] = s
        print("SYSTEM", branch, label, s["chrf"], flush=True)
    dump(directory / "metrics.json", metrics)
    m._dynamic_match.cache_clear()
    del engine, m, rev
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def forward_only(branch):
    torch.set_num_threads(4)
    assert read(OUT / branch / "forward/result.json")["status"] == "complete"
    directory = OUT / "evaluation" / branch
    directory.mkdir(parents=True, exist_ok=True)
    m, _ = model(paths(branch)[0])
    m.eval()
    metrics = {}
    for label, gg in [
        ("locked_test", list(rows(OUT / "test_locked.jsonl"))),
        ("legacy_test", groups_from(rows(PREVIOUS / "test.jsonl"))),
    ]:
        p = directory / (label + "_raw.jsonl")
        pred = list(rows(p)) if p.exists() else raw(m, gg)
        s, detail = stats_for(gg, pred)
        jsonl(p, detail)
        metrics[label] = s
        print("RAW", branch, label, s["chrf"], s["pure_han_length"], flush=True)
    dump(directory / "raw_metrics_interim.json", metrics)


def paired_bootstrap(a, b):
    delta = np.array([x["sentence_chrf"] - y["sentence_chrf"] for x, y in zip(a, b)])
    rng = np.random.default_rng(20261106)
    means = []
    for i in range(1000):
        means.append(float(delta[rng.integers(len(delta), size=len(delta))].mean()))
    return {
        "mean_sentence_chrf_difference": float(delta.mean()),
        "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
        "resamples": 1000,
        "unit": "locked unique upper",
        "training_seeds": 1,
        "scope": "conditional on fixed trained models; sentence chrF, not corpus chrF or expert literary score",
    }


def main():
    torch.set_num_threads(4)
    protocol = read(OUT / "protocol.json")
    assert sha(OUT / "test_locked.jsonl") == protocol["test_sha256"]
    for branch in ("new_data", "old_data_control"):
        for side in ("forward", "reverse"):
            assert read(OUT / branch / side / "result.json")["status"] == "complete"
    groups = list(rows(OUT / "test_locked.jsonl"))
    legacy = groups_from(rows(PREVIOUS / "test.jsonl"))
    results = {}
    for branch in ("original_v7", "new_data", "old_data_control"):
        results[branch] = evaluate_branch(branch, groups, legacy)
    bootstrap = {}
    for test in ("locked_test", "legacy_test"):
        for decode in ("raw", "system"):
            new = list(rows(OUT / "evaluation/new_data" / f"{test}_{decode}.jsonl"))
            for base in ("original_v7", "old_data_control"):
                old = list(rows(OUT / "evaluation" / base / f"{test}_{decode}.jsonl"))
                assert [r["upper"] for r in new] == [r["upper"] for r in old]
                bootstrap[test + "_" + decode + "_new_minus_" + base] = paired_bootstrap(new, old)
    # Fixed, labeled-as-unreviewed examples for semantic inspection; no fabricated ratings.
    a = list(rows(OUT / "evaluation/original_v7/locked_test_system.jsonl"))
    b = list(rows(OUT / "evaluation/new_data/locked_test_system.jsonl"))
    c = list(rows(OUT / "evaluation/old_data_control/locked_test_system.jsonl"))
    review = []
    chosen = sorted(range(len(a)), key=lambda i: stable("blind-review:" + a[i]["upper"]))[:100]
    for i in chosen:
        review.append(
            {
                "upper": a[i]["upper"],
                "references": a[i]["references"],
                "original_v7": a[i]["prediction"],
                "new_data": b[i]["prediction"],
                "old_data_control": c[i]["prediction"],
                "expert_review": "not_performed",
            }
        )
    jsonl(OUT / "review_examples.jsonl", review)
    dump(
        OUT / "comparison.json",
        {
            "status": "complete",
            "results": results,
            "paired_bootstrap": bootstrap,
            "test_historically_exposed": True,
            "single_seed": True,
            "no_expert_literary_quality_claim": True,
            "default_model_not_switched": True,
        },
    )
    print("COMPARISON COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--branch", choices=["original_v7", "new_data", "old_data_control"])
    parser.add_argument("--forward-only", action="store_true")
    args = parser.parse_args()
    if args.forward_only:
        forward_only(args.branch)
    elif args.baseline_only or args.branch:
        branch = args.branch or "original_v7"
        if branch != "original_v7":
            for side in ("forward", "reverse"):
                assert read(OUT / branch / side / "result.json")["status"] == "complete"
        torch.set_num_threads(4)
        evaluate_branch(
            branch,
            list(rows(OUT / "test_locked.jsonl")),
            groups_from(rows(PREVIOUS / "test.jsonl")),
        )
    else:
        main()
