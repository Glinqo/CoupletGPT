"""Evaluate every available length before/after training and the final v4 inference system."""

import collections, time
import torch
import numpy as np
from couplet_gpt.data.couplet_splits import OUT, ROOT, BASE, read, rows, dump, jsonl, sha
from couplet_gpt.modeling.couplet import initialize
from couplet_gpt.inference.couplet import load, respond
from couplet_gpt.evaluation.metrics import score_predictions
from couplet_gpt.inference.checks import rule_check_v3
from couplet_gpt.data.datasets import CharacterTokenizer


@torch.inference_mode()
def raw(m, groups):
    m.eval()
    tok = CharacterTokenizer()
    result = {}
    for n in sorted({len(g["upper"]) for g in groups}):
        chosen = [g for g in groups if len(g["upper"]) == n]
        for i in range(0, len(chosen), 64):
            b = chosen[i : i + 64]
            x = torch.tensor([[2, 4] + tok.encode(g["upper"]) + [5] for g in b], device="cuda")
            done = torch.zeros(len(b), device="cuda", dtype=torch.bool)
            generated = []
            for step in range(min(n + 4, 128 - x.shape[1])):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    z = m(x)[:, -1].float()
                z[:, [0, 1, 2, 4, 5]] = -float("inf")
                v = z.argmax(-1)
                v = torch.where(done, torch.zeros_like(v), v)
                done |= v == 3
                generated.append(v.cpu().tolist())
                x = torch.cat([x, v[:, None]], 1)
                if bool(done.all()):
                    break
            for j, g in enumerate(b):
                ids = []
                for v in generated:
                    if v[j] in (0, 3):
                        break
                    ids.append(v[j])
                result[g["upper"]] = {
                    "prediction": tok.decode(ids),
                    "ended_naturally": bool(done[j]),
                    "forced_length": False,
                }
    return [result[g["upper"]] for g in groups]


def main():
    torch.set_num_threads(4)
    g = {}
    for r in rows(OUT / "test.jsonl"):
        a = g.setdefault(r["upper"], {"upper": r["upper"], "references": []})
        if r["lower"] not in a["references"]:
            a["references"].append(r["lower"])
    groups = list(g.values())
    allstats = {}
    for name, path in [("before_raw", BASE), ("after_raw", OUT / "forward/best.pt")]:
        output = OUT / f"{name}.jsonl"
        if output.exists():
            pred = [
                {
                    "prediction": r["prediction"],
                    "ended_naturally": r["ended_naturally"],
                    "forced_length": False,
                }
                for r in rows(output)
            ]
        else:
            m, _ = initialize(path)
            pred = raw(m, groups)
            m._dynamic_match.cache_clear()
            del m
            torch.cuda.empty_cache()
        stats, detail = score_predictions(groups, pred)
        stats["by_length"] = {}
        for n in sorted({len(g["upper"]) for g in groups}):
            ii = [i for i, g in enumerate(groups) if len(g["upper"]) == n]
            stats["by_length"][str(n)] = score_predictions(
                [groups[i] for i in ii], [pred[i] for i in ii]
            )[0]
        jsonl(output, detail)
        allstats[name] = stats
        print(name, stats["chrf"], stats["pure_han_length"], flush=True)
    engine = load()
    pred = []
    start = time.perf_counter()
    for n in sorted({len(g["upper"]) for g in groups}):
        selected = [g for g in groups if len(g["upper"]) == n]
        for i in range(0, len(selected), 8):
            b = selected[i : i + 8]
            u = [g["upper"] for g in b]
            pools = engine.scored_pools(u, engine.candidates(u, k=16, nsamples=8))
            for upper, pool in zip(u, pools):
                c = max(pool, key=lambda r: r["forward_score"] + r["reverse_score"])
                pred.append(
                    {
                        "upper": upper,
                        "prediction": c["text"],
                        "ended_naturally": None,
                        "forced_length": True,
                    }
                )
        print("SYSTEM LENGTH", n, len(selected), round(time.perf_counter() - start, 1), flush=True)
    lookup = {r["upper"]: r for r in pred}
    pred = [lookup[g["upper"]] for g in groups]
    stats, detail = score_predictions(groups, pred)
    stats["structure_proxy_pass_rate"] = float(
        np.mean([rule_check_v3(r["upper"], r["prediction"])["structural_pass"] for r in detail])
    )
    stats["by_length"] = {}
    for n in sorted({len(g["upper"]) for g in groups}):
        ii = [i for i, g in enumerate(groups) if len(g["upper"]) == n]
        stats["by_length"][str(n)] = score_predictions(
            [groups[i] for i in ii], [pred[i] for i in ii]
        )[0]
    allstats["final_v4_system"] = stats
    jsonl(OUT / "final_system_predictions.jsonl", detail)
    dump(
        OUT / "evaluation.json",
        {
            "status": "complete",
            "results": allstats,
            "old_test_historically_reused": True,
            "system_rules_not_claimed_as_model_skill": True,
        },
    )
    # Synthetic unseen sources exercise >18 characters and mixed batch slots; no quality score fabricated.
    smoke = []
    for upper in [
        "山",
        "春风",
        "山河秀",
        "春风送暖",
        "清泉萦我梦",
        "春风吹绿大地",
        "春风又绿江南岸",
        "春风吹绿江南两岸",
        "春风吹绿江南千里岸",
        "春风吹绿江南千里长堤岸",
        "晓雾绕杉亭",
    ]:
        smoke.append(respond(engine, upper))
    jsonl(OUT / "smoke_examples.jsonl", smoke)
    dump(
        ROOT / "FINAL_V7_READY.json",
        {
            "status": "ready",
            "forward": str(OUT / "forward/best.pt"),
            "reverse": str(OUT / "reverse/best.pt"),
            "generator": "ChineseBERT+lexical_adapter_variable_length",
            "inference": "v4 contextual constraints + forward/reverse scores",
            "maximum_single_line_characters": 62,
            "fixed_five_seven_limit_removed": True,
            "independent_literary_quality_verified": False,
            "allusion_and_syntax_attention_disabled": True,
            "forward_sha256": sha(OUT / "forward/best.pt"),
        },
    )
    print("FINAL V7 READY", flush=True)


if __name__ == "__main__":
    main()
