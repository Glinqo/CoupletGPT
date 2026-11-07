"""Locked validation calibration and final model ablations; labels never enter deployed scoring."""

import argparse, contextlib, sys, types
from couplet_gpt.training.reverse_common import *
from couplet_gpt.training.couplet_common import load as load_forward
from couplet_gpt.inference.couplet import FinalEngine
from couplet_gpt.data.lexicon import LexiconMatcher
from couplet_gpt.evaluation.paired_metrics import stats_for, paired_bootstrap
from couplet_gpt.evaluation.generation import raw
from sacrebleu.metrics import CHRF

KNOCKOUTS = (
    "no_lexical_fusion",
    "no_context_projection",
    "no_word_residual",
    "no_alignment",
    "old8192_matcher",
    "relaxed_constraints",
)
POOL = OUT / "pools"
EVAL = OUT / "evaluation"


def groups(split):
    return list(
        rows(
            OUT / "calibration_groups.jsonl"
            if split == "validation"
            else V11 / "system_test_locked.jsonl"
            if split == "core"
            else LOCK / "test_locked.jsonl"
        )
    )


@contextlib.contextmanager
def knockout(m, engine, name):
    hooks = []
    matcher = m.matcher
    allowed = engine.allowed
    try:
        if name == "no_lexical_fusion":
            hooks.append(
                m.lexical_output.register_forward_hook(lambda mod, inp, out: torch.zeros_like(out))
            )
        elif name == "no_context_projection":
            hooks.append(
                m.feature_project.register_forward_hook(lambda mod, inp, out: torch.zeros_like(out))
            )
        elif name == "no_word_residual":
            hooks.append(
                m.word_residual.register_forward_hook(lambda mod, inp, out: torch.zeros_like(out))
            )
        elif name == "no_alignment":
            for projection in m.alignment:
                hooks.append(
                    projection.register_forward_hook(lambda mod, inp, out: torch.zeros_like(out))
                )
        elif name == "old8192_matcher":
            m.matcher = LexiconMatcher(m.entries[:8192])
            m._dynamic_match.cache_clear()
        elif name == "relaxed_constraints":
            engine.allowed = types.MethodType(
                lambda self, upper_ids, x, p, j, need: self.han_mask[None, None, :].expand(
                    x.shape[0], x.shape[1], -1
                ),
                engine,
            )
        yield
    finally:
        for hook in hooks:
            hook.remove()
        m.matcher = matcher
        engine.allowed = allowed
        m._dynamic_match.cache_clear()


def make_pool(engine, gg, path):
    if path.exists():
        return
    result = {}
    start = time.perf_counter()
    for n in sorted({len(g["upper"]) for g in gg}):
        chosen = [g for g in gg if len(g["upper"]) == n]
        for i in range(0, len(chosen), 8):
            bb = chosen[i : i + 8]
            uu = [g["upper"] for g in bb]
            pp = engine.candidates(uu, k=16, nsamples=8)
            flat_u = [u for u, pool in zip(uu, pp) for _ in pool]
            flat_y = [y for pool in pp for y in pool]
            scores = engine.log_scores(engine.model, flat_u, flat_y)
            offset = 0
            for g, pool in zip(bb, pp):
                result[g["upper"]] = {
                    **g,
                    "pool": [
                        {"text": y, "forward_score": scores[offset + j]} for j, y in enumerate(pool)
                    ],
                }
                offset += len(pool)
        print("POOL", path.stem, n, len(chosen), round(time.perf_counter() - start, 1), flush=True)
    jsonl(path, [result[g["upper"]] for g in gg])


def prepare_pools():
    POOL.mkdir(parents=True, exist_ok=True)
    EVAL.mkdir(exist_ok=True)
    m, _ = load_forward(FORWARD)
    m.eval()
    engine = FinalEngine(m, None)
    for split in ("validation", "core", "full"):
        make_pool(engine, groups(split), POOL / (split + "_base.jsonl"))
    # Recompute baseline on identical core batches. Reusing a subset of full-test
    # BF16 outputs changes batch shapes and can flip near-tied greedy decisions.
    core = groups("core")
    _, baseline_detail = stats_for(core, raw(m, core))
    jsonl(EVAL / "core_raw_base.jsonl", baseline_detail)
    for name in KNOCKOUTS:
        with knockout(m, engine, name):
            make_pool(engine, groups("core"), POOL / ("core_" + name + ".jsonl"))
            path = EVAL / ("core_raw_" + name + ".jsonl")
            if not path.exists():
                pred = raw(m, groups("core"))
                _, details = stats_for(groups("core"), pred)
                jsonl(path, details)
    m._dynamic_match.cache_clear()
    del m, engine
    gc.collect()
    torch.cuda.empty_cache()
    print("FORWARD POOLS READY", flush=True)


def preencode(m, records):
    if not hasattr(m, "context_provider"):
        return
    provider = m.context_provider
    tok = CharacterTokenizer()
    keys = {tuple(tok.encode(p["text"])) for r in records for p in r["pool"]}
    missing = sorted(
        (
            k
            for k in keys
            if k not in provider.index
            and k not in provider.extra
            and k not in provider.lower_index
            and k not in provider.online
        ),
        key=lambda k: (len(k), k),
    )
    for i in range(0, len(missing), 128):
        provider.add(missing[i : i + 128])
    if provider.encoder is not None:
        provider.encoder = None
        gc.collect()
        torch.cuda.empty_cache()
    print("REVERSE CANDIDATE CONTEXTS", len(missing), flush=True)


def score_reverse(label, m, pool_path):
    dest = POOL / (pool_path.stem + "_" + label + "_scores.jsonl")
    if dest.exists():
        return
    records = list(rows(pool_path))
    preencode(m, records)
    # log_scores is architecture-independent; a light receiver avoids loading another forward model.
    receiver = types.SimpleNamespace(tok=CharacterTokenizer())
    from couplet_gpt.inference.search import V2Engine

    output = []
    for i in range(0, len(records), 64):
        block = records[i : i + 64]
        uu = [r["upper"] for r in block for p in r["pool"]]
        yy = [p["text"] for r in block for p in r["pool"]]
        scores = V2Engine.log_scores(receiver, m, uu, yy, True)
        offset = 0
        for r in block:
            output.append({"upper": r["upper"], "scores": scores[offset : offset + len(r["pool"])]})
            offset += len(r["pool"])
    jsonl(dest, output)
    print("REVERSE SCORES", label, pool_path.stem, len(records), flush=True)


def all_reverse_scores():
    for label in ("old_reverse", "reverse_adapter", "reverse_aligned_control"):
        if label == "old_reverse":
            m = load_model_v2(torch.load(REVERSE, map_location="cpu", weights_only=True))
        else:
            m, _ = load_reverse(label)
        paths = [POOL / (s + "_base.jsonl") for s in ("validation", "core", "full")]
        if label == "reverse_adapter":
            paths += [POOL / ("core_" + k + ".jsonl") for k in KNOCKOUTS]
        for path in paths:
            score_reverse(label, m, path)
        if hasattr(m, "_dynamic_match"):
            m._dynamic_match.cache_clear()
        del m
        gc.collect()
        torch.cuda.empty_cache()


def predictions(pool_path, label=None, weight=0):
    records = list(rows(pool_path))
    reverse = list(rows(POOL / (pool_path.stem + "_" + label + "_scores.jsonl"))) if label else None
    if reverse:
        assert [r["upper"] for r in records] == [r["upper"] for r in reverse]
    output = []
    for i, r in enumerate(records):
        scores = reverse[i]["scores"] if reverse else [0.0] * len(r["pool"])
        assert len(scores) == len(r["pool"])
        best = max(
            range(len(r["pool"])), key=lambda j: r["pool"][j]["forward_score"] + weight * scores[j]
        )
        output.append(
            {"prediction": r["pool"][best]["text"], "ended_naturally": None, "forced_length": True}
        )
    return output


def calibrate():
    p = read(OUT / "protocol.json")
    metric = CHRF(char_order=6, word_order=0, beta=2)
    gg = groups("validation")
    table = {}
    selected = {}
    for label in ("old_reverse", "reverse_adapter", "reverse_aligned_control"):
        table[label] = []
        for weight in p["calibration_grid"]:
            pred = predictions(POOL / "validation_base.jsonl", label, weight)
            refs = [
                [g["references"][min(i, len(g["references"]) - 1)] for g in gg]
                for i in range(max(len(g["references"]) for g in gg))
            ]
            score = metric.corpus_score([r["prediction"] for r in pred], refs).score
            table[label].append({"weight": weight, "chrf": score})
        selected[label] = max(table[label], key=lambda r: (r["chrf"], -r["weight"]))
    dump(
        OUT / "calibration.json",
        {
            "status": "complete",
            "selection_data": "validation only",
            "selected": selected,
            "grid_results": table,
        },
    )
    return selected


def evaluate_all():
    selected = read(OUT / "calibration.json")["selected"]
    results = {}
    details = {}
    EVAL.mkdir(exist_ok=True)
    for split in ("core", "full"):
        gg = groups(split)
        variants = {
            "forward_only": (None, 0),
            "old_reverse_equal": ("old_reverse", 1),
            "old_reverse_calibrated": ("old_reverse", selected["old_reverse"]["weight"]),
            "aligned_reverse_calibrated": (
                "reverse_aligned_control",
                selected["reverse_aligned_control"]["weight"],
            ),
            "adapter_reverse_equal": ("reverse_adapter", 1),
            "adapter_reverse_calibrated": (
                "reverse_adapter",
                selected["reverse_adapter"]["weight"],
            ),
        }
        for name, (label, weight) in variants.items():
            pred = predictions(POOL / (split + "_base.jsonl"), label, weight)
            score, detail = stats_for(gg, pred)
            results[split + "/" + name] = score
            details[split + "/" + name] = detail
            jsonl(EVAL / (split + "_" + name + ".jsonl"), detail)
        raw_path = (
            EVAL / "core_raw_base.jsonl"
            if split == "core"
            else V11 / "evaluation/expanded/raw.jsonl"
        )
        score, detail = stats_for(gg, list(rows(raw_path)))
        results[split + "/raw"] = score
        details[split + "/raw"] = detail
    for name in KNOCKOUTS:
        gg = groups("core")
        pred = predictions(
            POOL / ("core_" + name + ".jsonl"),
            "reverse_adapter",
            selected["reverse_adapter"]["weight"],
        )
        score, detail = stats_for(gg, pred)
        results["core/knockout_" + name] = score
        details["core/knockout_" + name] = detail
        jsonl(EVAL / ("core_knockout_" + name + ".jsonl"), detail)
        score, detail = stats_for(gg, list(rows(EVAL / ("core_raw_" + name + ".jsonl"))))
        results["core/raw_" + name] = score
    comparisons = {}
    for split in ("core", "full"):
        final = details[split + "/adapter_reverse_calibrated"]
        for name in (
            "forward_only",
            "old_reverse_calibrated",
            "aligned_reverse_calibrated",
            "adapter_reverse_equal",
        ):
            comparisons[split + "/final_minus_" + name] = paired_bootstrap(
                final, details[split + "/" + name]
            )
    for name in KNOCKOUTS:
        comparisons["core/final_minus_" + name] = paired_bootstrap(
            details["core/adapter_reverse_calibrated"], details["core/knockout_" + name]
        )
    dump(
        OUT / "ablation_results.json",
        {
            "status": "complete",
            "results": results,
            "paired_bootstrap": comparisons,
            "calibration": selected,
            "inference_knockouts_not_retrained": True,
            "single_seed": True,
            "no_expert_literary_quality_claim": True,
            "test_historically_reported": True,
        },
    )
    print(
        "FINAL ABLATIONS COMPLETE",
        [(k, round(v["chrf"], 4)) for k, v in results.items()],
        flush=True,
    )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf8")
    torch.set_num_threads(3)
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-pools", action="store_true")
    a = parser.parse_args()
    freeze()
    if a.prepare_pools:
        prepare_pools()
    else:
        for branch in BRANCHES:
            assert read(OUT / branch / "result.json")["status"] == "complete"
        prepare_pools()
        all_reverse_scores()
        calibrate()
        evaluate_all()
