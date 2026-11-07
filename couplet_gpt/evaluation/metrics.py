"""Final held-out evaluation after all six validation-selected SFT runs complete."""

import collections, json, math, pathlib, time
import numpy as np
import torch
from sacrebleu.metrics import BLEU, CHRF
from couplet_gpt.data.datasets import ROOT, CoupletDataset, CharacterTokenizer
from couplet_gpt.modeling.gpt import GPT, GPTConfig
from couplet_gpt.training.transfer_comparison import RUN, SEEDS, evaluate
from couplet_gpt.training.pretrain import json_write


def han(c):
    n = ord(c)
    return (
        0x3400 <= n <= 0x4DBF
        or 0x4E00 <= n <= 0x9FFF
        or 0xF900 <= n <= 0xFAFF
        or 0x20000 <= n <= 0x2EE5F
        or 0x30000 <= n <= 0x3347F
    )


def test_groups():
    records = [
        json.loads(l) for l in (ROOT / "data/processed/couplet_test.jsonl").open(encoding="utf-8")
    ]
    groups = {}
    for i, r in enumerate(records):
        g = groups.setdefault(
            r["upper"], {"upper": r["upper"], "references": [], "row_indices": []}
        )
        if r["lower"] not in g["references"]:
            g["references"].append(r["lower"])
        g["row_indices"].append(i)
    return records, list(groups.values())


@torch.inference_mode()
def decode_all(model, groups, mode, tokenizer):
    model.eval()
    vocab = tokenizer.vocab
    reverse = tokenizer.reverse
    output = {}
    allowed = torch.zeros(model.config.vocab_size, device="cuda", dtype=torch.bool)
    for c, i in vocab.items():
        if len(c) == 1 and han(c):
            allowed[i] = True
    for n in (5, 7):
        selected = [g for g in groups if len(g["upper"]) == n]
        for start in range(0, len(selected), 128):
            batch = selected[start : start + 128]
            prefix = [[2, 4] + tokenizer.encode(g["upper"]) + [5] for g in batch]
            x = torch.tensor(prefix, device="cuda")
            finished = torch.zeros(len(batch), device="cuda", dtype=torch.bool)
            generated = []
            ended = torch.zeros_like(finished)
            max_steps = n if mode == "length_constrained" else 16
            for step in range(max_steps):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    scores = model(x)[:, -1].float()
                if mode == "length_constrained":
                    scores[:, ~allowed] = float("-inf")
                else:
                    scores[:, [0, 1, 2, 4, 5]] = float("-inf")
                next_id = scores.argmax(-1)
                if mode == "raw":
                    next_id = torch.where(finished, torch.zeros_like(next_id), next_id)
                    new_eos = (next_id == 3) & ~finished
                    ended |= new_eos
                    finished |= new_eos
                generated.append(next_id.cpu().tolist())
                x = torch.cat([x, next_id[:, None]], 1)
                if mode == "raw" and bool(finished.all()):
                    break
            for i, g in enumerate(batch):
                ids = []
                for col in generated:
                    v = col[i]
                    if v in (0, 3):
                        break
                    ids.append(v)
                text = "".join(reverse[v] for v in ids)
                output[g["upper"]] = {
                    "prediction": text,
                    "ids": ids,
                    "ended_naturally": bool(ended[i]) if mode == "raw" else None,
                    "forced_length": mode == "length_constrained",
                }
    return [output[g["upper"]] for g in groups]


def score_predictions(groups, predictions):
    texts = [r["prediction"] for r in predictions]
    refs = [
        [g["references"][i] if i < len(g["references"]) else None for g in groups]
        for i in range(max(len(g["references"]) for g in groups))
    ]
    bleu = BLEU(tokenize="char", smooth_method="exp", effective_order=False)
    chrf = CHRF(char_order=6, word_order=0, beta=2)
    b = bleu.corpus_score(texts, refs)
    c = chrf.corpus_score(texts, refs)
    per = []
    for g, r in zip(groups, predictions):
        text = r["prediction"]
        upper = g["upper"]
        n = len(upper)
        han_len = sum(han(x) for x in text)
        per.append(
            {
                "upper": upper,
                "references": g["references"],
                **r,
                "length_match": han_len == n,
                "pure_han_length": len(text) == n and all(han(x) for x in text),
                "exact_reference_match": text in g["references"],
                "copy_upper": text == upper,
                "position_copy_fraction": sum(a == b for a, b in zip(upper, text)) / n,
                "triple_repetition": any(
                    text[i] == text[i + 1] == text[i + 2] for i in range(max(0, len(text) - 2))
                ),
                "sentence_chrf": chrf.sentence_score(text, g["references"]).score,
            }
        )
    stats = {
        "unique_uppers": len(groups),
        "bleu": b.score,
        "chrf": c.score,
        "bleu_signature": str(bleu.get_signature()),
        "chrf_signature": str(chrf.get_signature()),
    }
    for k in (
        "length_match",
        "pure_han_length",
        "exact_reference_match",
        "copy_upper",
        "position_copy_fraction",
        "triple_repetition",
    ):
        stats[k] = float(np.mean([r[k] for r in per]))
    if predictions[0]["ended_naturally"] is not None:
        stats["natural_eos_rate"] = float(np.mean([r["ended_naturally"] for r in per]))
    return stats, per


def evaluate_one(condition, seed, ds, groups, tokenizer):
    folder = RUN / f"{condition}_seed{seed}"
    path = folder / "test_metrics.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    ck = torch.load(folder / "best_model.pt", map_location="cpu", weights_only=True)
    model = GPT(GPTConfig(**ck["model_config"])).cuda().eval()
    model.load_state_dict(ck["model"])
    teacher, rows = evaluate(model, ds, per_sample=True)
    result = {
        "condition": condition,
        "seed": seed,
        "selected_epoch": ck["epoch"],
        "teacher_forced": teacher,
        "generation": {},
    }
    with (folder / "test_nll.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    for mode in ("raw", "length_constrained"):
        predictions = decode_all(model, groups, mode, tokenizer)
        stats, per = score_predictions(groups, predictions)
        result["generation"][mode] = stats
        result["generation"][mode]["by_length"] = {}
        for n in (5, 7):
            ids = [i for i, g in enumerate(groups) if len(g["upper"]) == n]
            substats, _ = score_predictions([groups[i] for i in ids], [predictions[i] for i in ids])
            result["generation"][mode]["by_length"][str(n)] = substats
        with (folder / f"test_predictions_{mode}.jsonl").open("w", encoding="utf-8") as f:
            for r in per:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json_write(path, result)
    print(
        "TEST",
        condition,
        seed,
        "CE",
        round(teacher["ce"], 4),
        "raw chrF",
        round(result["generation"]["raw"]["chrf"], 3),
        "constrained chrF",
        round(result["generation"]["length_constrained"]["chrf"], 3),
        flush=True,
    )
    del model, ck
    torch.cuda.empty_cache()
    return result


def bootstrap(results, groups):
    # Paired resampling of unique upper groups; models/seeds are held fixed.
    nll = {}
    sentence = {}
    for condition in ("scratch", "pretrained"):
        values = []
        chrs = {"raw": [], "length_constrained": []}
        for seed in SEEDS:
            folder = RUN / f"{condition}_seed{seed}"
            rows = [json.loads(l) for l in (folder / "test_nll.jsonl").open(encoding="utf-8")]
            values.append(
                [float(np.mean([rows[i]["char_ce"] for i in g["row_indices"]])) for g in groups]
            )
            for mode in chrs:
                preds = [
                    json.loads(l)
                    for l in (folder / f"test_predictions_{mode}.jsonl").open(encoding="utf-8")
                ]
                assert [r["upper"] for r in preds] == [g["upper"] for g in groups]
                chrs[mode].append([r["sentence_chrf"] for r in preds])
        nll[condition] = np.mean(values, axis=0)
        sentence[condition] = {mode: np.mean(v, axis=0) for mode, v in chrs.items()}
    deltas = {"char_ce_scratch_minus_pretrained": nll["scratch"] - nll["pretrained"]}
    for mode in ("raw", "length_constrained"):
        deltas[f"{mode}_sentence_chrf_pretrained_minus_scratch"] = (
            sentence["pretrained"][mode] - sentence["scratch"][mode]
        )
    rng = np.random.default_rng(2029)
    n = len(groups)
    draws = rng.integers(0, n, size=(5000, n))
    out = {}
    for name, d in deltas.items():
        means = d[draws].mean(axis=1)
        out[name] = {
            "estimate": float(d.mean()),
            "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
            "bootstrap_replicates": 5000,
            "group": "unique upper",
            "seed": 2029,
            "note": "Conditional on fixed three trained models; not a training-seed confidence interval. Sentence chrF is not corpus chrF.",
        }
    return out


def main():
    torch.set_num_threads(4)
    complete = json.loads((RUN / "training_complete.json").read_text(encoding="utf-8"))
    assert complete["status"] == "complete"
    records, groups = test_groups()
    ds = CoupletDataset("test")
    tokenizer = CharacterTokenizer()
    assert len(ds) == len(records) and len(records) == 2246
    results = []
    for seed in SEEDS:
        for condition in ("pretrained", "scratch"):
            results.append(evaluate_one(condition, seed, ds, groups, tokenizer))
    # Descriptive reference only: the pretrained base has received zero supervised updates.
    base_path = RUN / "pretrained_without_sft.json"
    if not base_path.exists():
        ck = torch.load(
            ROOT / "runs/pretrain_v1/artifacts/best_model.pt", map_location="cpu", weights_only=True
        )
        model = GPT(GPTConfig(**ck["model_config"])).cuda().eval()
        model.load_state_dict(ck["model"])
        base = {"supervised_updates": 0, "teacher_forced": evaluate(model, ds), "generation": {}}
        for mode in ("raw", "length_constrained"):
            pp = decode_all(model, groups, mode, tokenizer)
            ss, per = score_predictions(groups, pp)
            base["generation"][mode] = ss
            with (RUN / f"pretrained_without_sft_{mode}.jsonl").open("w", encoding="utf-8") as f:
                for r in per:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        json_write(base_path, base)
        del model, ck
        torch.cuda.empty_cache()
    boot = bootstrap(results, groups)
    json_write(RUN / "bootstrap.json", boot)
    json_write(
        RUN / "test_results.json",
        {
            "status": "complete",
            "row_count": len(records),
            "unique_uppers": len(groups),
            "runs": results,
            "bootstrap": boot,
        },
    )
    print("ALL FINAL TEST EVALUATIONS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
