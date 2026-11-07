"""Frozen-weight ablation: remove couplet decoding rules, retain search and optional reverse selection."""

import argparse, contextlib, gc, hashlib, json, math, sys, time, types
from pathlib import Path
import numpy as np, torch
from torch.nn import functional as F
from couplet_gpt.training.reverse_common import (
    ROOT,
    OUT as V12,
    FORWARD,
    LEX,
    LOCK,
    V11,
    ReverseContexts,
    load_reverse,
    read,
    rows,
    dump,
    jsonl,
    sha,
    collate,
)
from couplet_gpt.training.couplet_common import load as load_forward
from couplet_gpt.inference.couplet import FinalEngine
from couplet_gpt.evaluation.components import knockout
from couplet_gpt.data.datasets import CharacterTokenizer
from couplet_gpt.evaluation.paired_metrics import stats_for, paired_bootstrap
from couplet_gpt.evaluation.metrics import han

OUT = ROOT / "results/couplet/ablation"
VARIANTS = {
    "with_constraints": "base",
    "unconstrained": "base",
    "without_lexical_fusion": "no_lexical_fusion",
    "without_contextual_features": "no_context_projection",
    "without_word_embeddings": "no_word_residual",
    "with_small_lexicon": "old8192_matcher",
    "without_position_alignment": "no_alignment",
}
MAX_NEW = 63


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    release = {"reverse_checkpoint": str(ROOT / "models/couplet/reverse.pt")}
    if (OUT / "evaluation_protocol.json").exists():
        p = read(OUT / "evaluation_protocol.json")
        assert p["forward_sha256"] == sha(FORWARD) and p["reverse_sha256"] == sha(
            Path(release["reverse_checkpoint"])
        )
        return p
    p = {
        "status": "locked_before_new_evaluation",
        "forward_sha256": sha(FORWARD),
        "reverse_sha256": sha(Path(release["reverse_checkpoint"])),
        "reverse_checkpoint": release["reverse_checkpoint"],
        "vocab_sha256": sha(ROOT / "tokenizer/vocab.json"),
        "lexicon_sha256": sha(LEX),
        "test_sha256": sha(ROOT / "data/couplet/evaluation/test_locked.jsonl"),
        "core_sha256": sha(ROOT / "data/couplet/evaluation/core_test.jsonl"),
        "test_groups": 2617,
        "core_groups": 347,
        "variants": VARIANTS,
        "selection_modes": {"forward_only": 0.0, "bidirectional": 0.25},
        "model_weights_or_architecture_changed": False,
        "training": False,
        "beam": 16,
        "samples": 8,
        "topk": 20,
        "temperature": 0.8,
        "rules_disabled": [
            "equal_length",
            "Han_only",
            "same_position_exclusion",
            "repeat_pattern",
            "foot_tone",
            "structural_filter_before_ranking",
        ],
        "protocol_token_mask": [0, 2, 4, 5],
        "UNK_allowed_and_reported": True,
        "EOS": 3,
        "max_new_tokens": MAX_NEW,
        "beam_end": "finished paths remain in beam with zero-cost EOS self-loop; no expansion after first EOS",
        "beam_pruning": "cumulative forward log probability",
        "terminal_score": "mean token log probability including EOS, even for counterfactual stop score of truncated text",
        "sampling_rng": "legacy per-upper first n steps, independent fixed tail; shared random numbers across variants",
        "score_failures": "any unscorable pool makes bidirectional selection fail for that upper; no hidden fallback or filtering",
        "R0_termination": "length forced; forced EOS is not counted as natural EOS",
        "token_safe_features": "non-Han inputs preserve original token IDs and use character-aligned auxiliary features; UNK gets one replacement glyph only inside encoder, with invalid feature mask",
        "same_width_and_paths_not_equal_FLOPs": True,
        "test_historically_used": True,
        "inference_knockouts_not_retrained": True,
        "single_seed": True,
        "cross_diagnosis": "R1/R2 core subset, same union pools; reverse scores fixed; union budget disclosed",
    }
    dump(OUT / "evaluation_protocol.json", p)
    return p


class TokenSafeContexts(ReverseContexts):
    @torch.inference_mode()
    def add(self, keys):
        tok = CharacterTokenizer()
        normal = [
            k for k in keys if all(len(tok.reverse[t]) == 1 and han(tok.reverse[t]) for t in k)
        ]
        special = [k for k in keys if k not in set(normal)]
        if normal:
            super().add(normal)
        if not special:
            return
        from couplet_gpt.modeling.chinesebert import FrozenChineseBERT, SourceTokenizer
        from pypinyin import pinyin, Style

        if self.encoder is None:
            self.encoder = FrozenChineseBERT().cuda().eval()
            self.tokenizer = SourceTokenizer()
        width = max(map(len, special)) + 2
        ids = np.zeros((len(special), width), np.int64)
        py = np.zeros((len(special), width, 8), np.int64)
        loc = np.full((len(special), width - 2), -1, np.int64)
        for i, k in enumerate(special):
            text = "".join(tok.reverse[t] if len(tok.reverse[t]) == 1 else "\ufffd" for t in k)
            enc = self.tokenizer.tokenizer.encode(text)
            assert len(enc.ids) <= width and len(text) == len(k)
            ids[i, : len(enc.ids)] = enc.ids
            sounds = pinyin(text, style=Style.TONE3, heteronym=True, errors=lambda s: list(s))
            mapping = {}
            assert len(sounds) == len(k)
            for j, candidates in enumerate(sounds):
                sound = candidates[0]
                if sound in self.tokenizer.pytensor:
                    mapping[j] = self.tokenizer.pytensor[sound]
                else:
                    v = [0] * 8
                    if len(sound) <= 8 and all(c in self.tokenizer.pmap["char2idx"] for c in sound):
                        for t, c in enumerate(sound):
                            v[t] = self.tokenizer.pmap["char2idx"][c]
                    mapping[j] = v
            for j, (token, (a, b)) in enumerate(zip(enc.tokens, enc.offsets)):
                if b - a == 1 and token != "[UNK]" and a < len(k) and k[a] != 1:
                    loc[i, a] = j
                    py[i, j] = mapping.get(a, [0] * 8)
        x = torch.tensor(ids, device="cuda")
        pp = torch.tensor(py, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = self.encoder(x, pp, x != 0)
        pos = torch.tensor(loc, device="cuda")
        v = h.gather(1, pos.clamp_min(0)[..., None].expand(-1, -1, 768)) * (pos >= 0)[..., None]
        for k, value, valid in zip(special, v.float().cpu().numpy().astype(np.float16), loc >= 0):
            self.online[k] = (value[: len(k)], valid[: len(k)])


def uniforms(uppers, n, samples=8):
    result = []
    for u in uppers:
        seed = int(hashlib.sha256(("2036:" + u).encode()).hexdigest()[:16], 16)
        prefix = np.random.default_rng(seed).random((samples, n))
        tail_seed = int(hashlib.sha256(("2036:" + u + ":tail").encode()).hexdigest()[:16], 16)
        tail = np.random.default_rng(tail_seed).random((samples, MAX_NEW - n))
        result.append(np.concatenate([prefix, tail], 1))
    return torch.tensor(np.stack(result), device="cuda", dtype=torch.float32)


class Search:
    def __init__(self, model):
        self.model = model.eval()
        self.tok = CharacterTokenizer()
        self.V = len(self.tok.vocab)
        self.rule_engine = FinalEngine(model, None)

    @torch.inference_mode()
    def candidates(self, uppers, rules=False):
        b = len(uppers)
        n = len(uppers[0])
        assert all(len(u) == n for u in uppers)
        source = torch.tensor([self.tok.encode(u) for u in uppers], device="cuda")
        prefix = torch.cat(
            [
                source.new_full((b, 1), 2),
                source.new_full((b, 1), 4),
                source,
                source.new_full((b, 1), 5),
            ],
            1,
        )
        p = prefix.shape[1]

        def allowed(z, x, j, done):
            if rules and j < n:
                mask = self.rule_engine.allowed(
                    source, x, p, j, torch.zeros(b, dtype=torch.bool, device="cuda")
                )
                z = z.masked_fill(~mask, -float("inf"))
            elif rules:
                keep = z[..., 3].clone()
                z = torch.full_like(z, -float("inf"))
                z[..., 3] = keep
            else:
                z[..., list((0, 2, 4, 5))] = -float("inf")
            z = z.masked_fill(done[..., None], -float("inf"))
            z[..., 3] = torch.where(done, torch.zeros_like(z[..., 3]), z[..., 3])
            return z

        def unpack(x, origin):
            output = []
            for group in x.detach().cpu().tolist():
                entries = []
                for seq in group:
                    generated = seq[p:]
                    ended = 3 in generated
                    body = generated[: generated.index(3)] if ended else generated
                    entries.append(
                        {
                            "tokens": body,
                            "text": self.tok.decode(body),
                            "ended_naturally": ended and not rules,
                            "forced_length": rules,
                            "truncated": not ended,
                            "termination": "forced_length"
                            if rules
                            else "EOS"
                            if ended
                            else "token_budget",
                            "origin": [origin],
                        }
                    )
                output.append(entries)
            return output

        x = prefix[:, None, :]
        scores = torch.zeros((b, 1), device="cuda")
        done = torch.zeros((b, 1), device="cuda", dtype=torch.bool)
        beam_steps = 0
        for j in range(n + 1 if rules else MAX_NEW):
            oldk = x.shape[1]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = (
                    self.model(x.reshape(b * oldk, -1))[:, -1]
                    .float()
                    .log_softmax(-1)
                    .reshape(b, oldk, self.V)
                )
            z = allowed(z, x, j, done)
            val, index = (scores[:, :, None] + z).reshape(b, -1).topk(16)
            assert torch.isfinite(val).all()
            parent = index // self.V
            token = index % self.V
            x = torch.cat(
                [x.gather(1, parent[:, :, None].expand(-1, -1, x.shape[-1])), token[:, :, None]], 2
            )
            done = done.gather(1, parent) | (token == 3)
            scores = val
            beam_steps += 1
            if bool(done.all()):
                break
        result = unpack(x, "beam")
        x = prefix[:, None, :].expand(-1, 8, -1).clone()
        done = torch.zeros((b, 8), device="cuda", dtype=torch.bool)
        u = uniforms(uppers, n)
        sample_steps = 0
        for j in range(n + 1 if rules else MAX_NEW):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = self.model(x.reshape(b * 8, -1))[:, -1].float().reshape(b, 8, -1)
            z = allowed(z, x, j, done)
            values, index = (z / 0.8).topk(20)
            cdf = values.softmax(-1).cumsum(-1)
            pos = (u[:, :, j, None] > cdf).sum(-1).clamp(max=19)
            token = index.gather(2, pos[:, :, None]).squeeze(2)
            x = torch.cat([x, token[:, :, None]], 2)
            done |= token == 3
            sample_steps += 1
            if bool(done.all()):
                break
        sampled = unpack(x, "sample")
        for i, pool in enumerate(result):
            unique = {tuple(r["tokens"]): r for r in pool}
            for r in sampled[i]:
                key = tuple(r["tokens"])
                if key in unique:
                    unique[key]["origin"] += r["origin"]
                else:
                    unique[key] = r
            result[i] = list(unique.values())
        return result, {"beam_steps": beam_steps, "sample_steps": sample_steps}


@torch.inference_mode()
def token_scores(model, uppers, bodies, reverse=False, batch=64):
    tok = CharacterTokenizer()
    result = []
    for start in range(0, len(uppers), batch):
        sequences = []
        labels = []
        for u, y in zip(uppers[start : start + batch], bodies[start : start + batch]):
            a, b = (y, tok.encode(u)) if reverse else (tok.encode(u), y)
            s = [2, 4] + list(a) + [5] + list(b) + [3]
            xx = s[:-1]
            yy = [-100] * len(xx)
            yy[len(a) + 2 :] = list(b) + [3]
            assert len(xx) <= 128
            sequences.append(xx)
            labels.append(yy)
        width = max(map(len, sequences))
        x = torch.zeros((len(sequences), width), device="cuda", dtype=torch.long)
        y = torch.full_like(x, -100)
        for i, (xx, yy) in enumerate(zip(sequences, labels)):
            x[i, : len(xx)] = torch.tensor(xx, device="cuda")
            y[i, : len(yy)] = torch.tensor(yy, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = model(x)
        valid = y != -100
        loss = F.cross_entropy(z[valid].float(), y[valid], reduction="none")
        r = valid.nonzero()[:, 0]
        summed = torch.zeros(len(sequences), device="cuda").scatter_add_(0, r, loss)
        values = (-summed / valid.sum(1)).cpu().tolist()
        assert all(math.isfinite(v) for v in values)
        result += values
    return result


def preflight():
    if (OUT / "input_checks.json").exists():
        return
    m, _ = load_forward(FORWARD)
    engine = Search(m)
    base = list(rows(ROOT / "data/couplet/evaluation/core_test.jsonl"))
    sample = [next(r for r in base if len(r["upper"]) == n) for n in (1, 4, 7, 14)]
    for g in sample:
        pools, steps = engine.candidates([g["upper"]], False)
        assert len(pools[0]) <= 24
        for r in pools[0]:
            assert not any(t in (0, 2, 3, 4, 5) for t in r["tokens"]) and len(r["tokens"]) <= 63
    # No scoring-stage rule checker is used; token scoring must accept wrong length and repeats.
    scores = token_scores(m, ["山水秀"] * 3, [[], [m.entries[0]["token_ids"][0]] * 2, [1]])
    assert all(math.isfinite(v) for v in scores)
    m._dynamic_match.cache_clear()
    del m, engine
    gc.collect()
    torch.cuda.empty_cache()
    r, _ = load_reverse()
    r.context_provider = TokenSafeContexts()
    reverse_scores = token_scores(
        r,
        ["山水秀"] * 4,
        [
            [],
            [1],
            [CharacterTokenizer().vocab.get("，", 1)],
            [CharacterTokenizer().vocab["山"]] * 2,
        ],
        True,
    )
    assert all(math.isfinite(v) for v in reverse_scores)
    dump(
        OUT / "input_checks.json",
        {
            "status": "pass",
            "wrong_length_repeats_empty_UNK_and_punctuation_scoreable": True,
            "no_rule_filter_in_scoring": True,
            "forward_scores": scores,
            "reverse_scores": reverse_scores,
        },
    )
    r._dynamic_match.cache_clear()
    del r
    gc.collect()
    torch.cuda.empty_cache()
    print("FREE DECODING PREFLIGHT PASS", flush=True)


def generate_all():
    directory = OUT / "candidates"
    directory.mkdir(exist_ok=True)
    m, _ = load_forward(FORWARD)
    engine = Search(m)
    gg = list(rows(ROOT / "data/couplet/evaluation/test_locked.jsonl"))
    for variant, name in VARIANTS.items():
        path = directory / (variant + ".jsonl")
        old = list(rows(path)) if path.exists() else []
        seen = {r["upper"] for r in old}
        start = time.perf_counter()
        with knockout(m, engine.rule_engine, name):
            with path.open("a", encoding="utf8", buffering=1) as log:
                for n in sorted({len(g["upper"]) for g in gg}):
                    chosen = [g for g in gg if len(g["upper"]) == n and g["upper"] not in seen]
                    for i in range(0, len(chosen), 4):
                        bb = chosen[i : i + 4]
                        uu = [g["upper"] for g in bb]
                        pools, steps = engine.candidates(uu, variant == "with_constraints")
                        flat_u = [u for u, pool in zip(uu, pools) for _ in pool]
                        tokens = [r["tokens"] for pool in pools for r in pool]
                        scores = token_scores(m, flat_u, tokens)
                        offset = 0
                        for g, pool in zip(bb, pools):
                            for j, r in enumerate(pool):
                                r["forward_score"] = scores[offset + j]
                            offset += len(pool)
                            log.write(
                                json.dumps(
                                    {**g, "variant": variant, "pool": pool, "search": steps},
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            seen.add(g["upper"])
                        if len(seen) % 100 < 4:
                            print(
                                "GENERATE",
                                variant,
                                len(seen),
                                len(gg),
                                round(time.perf_counter() - start, 1),
                                flush=True,
                            )
                    if chosen:
                        print(
                            "LENGTH",
                            variant,
                            n,
                            len(seen),
                            round(time.perf_counter() - start, 1),
                            flush=True,
                        )
        assert len(seen) == len(gg)
        dump(
            directory / (variant + "_generation_summary.json"),
            {
                "status": "complete",
                "groups": len(seen),
                "sha256": sha(path),
                "seconds_this_invocation": time.perf_counter() - start,
            },
        )
    m._dynamic_match.cache_clear()
    del m, engine
    gc.collect()
    torch.cuda.empty_cache()


def score_all_reverse():
    model, _ = load_reverse()
    model.context_provider = TokenSafeContexts()
    provider = model.context_provider
    for variant in VARIANTS:
        path = OUT / "reverse_scores" / (variant + ".jsonl")
        path.parent.mkdir(exist_ok=True)
        old = list(rows(path)) if path.exists() else []
        seen = {r["upper"] for r in old}
        records = list(rows(OUT / "candidates" / (variant + ".jsonl")))
        with path.open("a", encoding="utf8", buffering=1) as log:
            for i in range(0, len(records), 32):
                block = [r for r in records[i : i + 32] if r["upper"] not in seen]
                if not block:
                    continue
                keys = {tuple(p["tokens"]) for r in block for p in r["pool"]}
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
                for j in range(0, len(missing), 64):
                    provider.add(missing[j : j + 64])
                uu = [r["upper"] for r in block for p in r["pool"]]
                yy = [p["tokens"] for r in block for p in r["pool"]]
                values = token_scores(model, uu, yy, True)
                offset = 0
                for r in block:
                    log.write(
                        json.dumps(
                            {
                                "upper": r["upper"],
                                "scores": values[offset : offset + len(r["pool"])],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    offset += len(r["pool"])
                    seen.add(r["upper"])
                if len(seen) % 256 < 32:
                    print("REVERSE", variant, len(seen), flush=True)
        assert len(seen) == len(records)
        # Bound online feature memory between variant pools; weights and fixed caches are retained.
        provider.online.clear()
        model._dynamic_match.cache_clear()
        gc.collect()
        torch.cuda.empty_cache()
    del model
    gc.collect()
    torch.cuda.empty_cache()


def summarize():
    (OUT / "predictions").mkdir(exist_ok=True)
    gg = list(rows(ROOT / "data/couplet/evaluation/test_locked.jsonl"))
    core = {r["upper"] for r in rows(ROOT / "data/couplet/evaluation/core_test.jsonl")}
    results = {}
    detail_index = {}
    for variant in VARIANTS:
        pool = {r["upper"]: r for r in rows(OUT / "candidates" / (variant + ".jsonl"))}
        rev = {r["upper"]: r["scores"] for r in rows(OUT / "reverse_scores" / (variant + ".jsonl"))}
        for mode, weight in [("forward_only", 0.0), ("bidirectional", 0.25)]:
            pred = []
            for g in gg:
                r = pool[g["upper"]]
                rr = rev[g["upper"]]
                assert len(rr) == len(r["pool"])
                j = max(
                    range(len(rr)), key=lambda j: r["pool"][j]["forward_score"] + weight * rr[j]
                )
                p = r["pool"][j]
                pred.append(
                    {
                        "prediction": p["text"],
                        "tokens": p["tokens"],
                        "ended_naturally": None
                        if variant == "with_constraints"
                        else p["ended_naturally"],
                        "forced_length": p["forced_length"],
                        "truncated": p["truncated"],
                        "candidate_count": len(rr),
                        "forward_score": p["forward_score"],
                        "reverse_score": rr[j] if weight else None,
                        "reverse_weight": weight,
                        "score_used_reverse": bool(weight),
                        "termination": p["termination"],
                    }
                )
            stats, details = stats_for(gg, pred)
            stats.update(
                truncation_rate=float(np.mean([p["truncated"] for p in pred])),
                UNK_rate=float(np.mean([1 in p["tokens"] for p in pred])),
                empty_rate=float(np.mean([not p["tokens"] for p in pred])),
                mean_candidate_count=float(np.mean([p["candidate_count"] for p in pred])),
                exact_token_length_rate=float(
                    np.mean([len(p["tokens"]) == len(g["upper"]) for g, p in zip(gg, pred)])
                ),
            )
            key = variant + "/" + mode
            results[key] = stats
            detail_index[key] = details
            jsonl(OUT / "predictions" / (variant + "_" + mode + ".jsonl"), details)
            ix = [i for i, g in enumerate(gg) if g["upper"] in core]
            results["core/" + key] = stats_for([gg[i] for i in ix], [pred[i] for i in ix])[0]
    comparisons = {}
    for mode in ("forward_only", "bidirectional"):
        baseline = detail_index["unconstrained/" + mode]
        for variant in VARIANTS:
            if variant != "unconstrained":
                comparisons[variant + "_minus_unconstrained/" + mode] = paired_bootstrap(
                    detail_index[variant + "/" + mode], baseline
                )
    for variant in VARIANTS:
        comparisons[variant + "/reverse_minus_forward"] = paired_bootstrap(
            detail_index[variant + "/bidirectional"], detail_index[variant + "/forward_only"]
        )
    dump(
        OUT / "metrics.json",
        {
            "status": "complete",
            "results": results,
            "paired_bootstrap": comparisons,
            "inference_knockouts_not_retrained": True,
            "test_historical": True,
        },
    )
    print(
        "NO-RULE ABLATIONS COMPLETE",
        [(k, round(v["chrf"], 4)) for k, v in results.items() if not k.startswith("core/")],
        flush=True,
    )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf8")
    torch.set_num_threads(3)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=["all", "generate", "reverse", "summary", "preflight"], default="all"
    )
    a = parser.parse_args()
    freeze()
    if a.phase in ("all", "preflight", "generate"):
        preflight()
    if a.phase in ("all", "generate"):
        generate_all()
    if a.phase in ("all", "reverse"):
        score_all_reverse()
    if a.phase in ("all", "summary"):
        summarize()
