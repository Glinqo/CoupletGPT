"""Fixed-pool lexical diagnosis and audited report; never changes generation weights."""

import argparse, ast, gc, hashlib, json, math, sys
import numpy as np
import torch
from couplet_gpt.evaluation.ablation import *
from couplet_gpt.inference.search import rule_check


def cross():
    if (OUT / "candidate_scoring_metrics.json").exists():
        return
    core = {r["upper"] for r in rows(ROOT / "data/couplet/evaluation/core_test.jsonl")}
    groups = [
        g for g in rows(ROOT / "data/couplet/evaluation/test_locked.jsonl") if g["upper"] in core
    ]
    pools = {
        v: {r["upper"]: r for r in rows(OUT / "candidates" / (v + ".jsonl"))}
        for v in ("unconstrained", "without_lexical_fusion")
    }
    reverse = {
        v: {r["upper"]: r["scores"] for r in rows(OUT / "reverse_scores" / (v + ".jsonl"))}
        for v in pools
    }
    unions = []
    max_reverse_discrepancy = 0.0
    for g in groups:
        u = g["upper"]
        candidates = {}
        indices = {}
        for v in pools:
            indices[v] = []
            for p, r in zip(pools[v][u]["pool"], reverse[v][u]):
                k = tuple(p["tokens"])
                if k in candidates:
                    max_reverse_discrepancy = max(
                        max_reverse_discrepancy, abs(candidates[k]["reverse_score"] - r)
                    )
                else:
                    candidates[k] = {**p, "reverse_score": r}
                indices[v].append(k)
        kk = list(candidates)
        unions.append(
            {
                **g,
                "pool": list(candidates.values()),
                "indices": {v: [kk.index(k) for k in ii] for v, ii in indices.items()},
            }
        )
    m, _ = load_forward(FORWARD)
    engine = Search(m)
    for name in ("base", "no_lexical_fusion"):
        with knockout(m, engine.rule_engine, name):
            for i in range(0, len(unions), 16):
                block = unions[i : i + 16]
                uu = [g["upper"] for g in block for p in g["pool"]]
                yy = [p["tokens"] for g in block for p in g["pool"]]
                values = token_scores(m, uu, yy)
                offset = 0
                for g in block:
                    for p in g["pool"]:
                        p["score_" + name] = values[offset]
                        offset += 1
            print("CROSS FORWARD", name, flush=True)
    del engine, m
    gc.collect()
    torch.cuda.empty_cache()
    jsonl(OUT / "candidate_scoring_inputs.jsonl", unions)
    metrics = {}
    details = {}
    for poolname in ("unconstrained", "without_lexical_fusion", "union"):
        for scorer in ("base", "no_lexical_fusion"):
            for mode, w in [("forward_only", 0.0), ("bidirectional", 0.25)]:
                pred = []
                for g in unions:
                    ii = range(len(g["pool"])) if poolname == "union" else g["indices"][poolname]
                    p = g["pool"][
                        max(
                            ii,
                            key=lambda j: (
                                g["pool"][j]["score_" + scorer] + w * g["pool"][j]["reverse_score"]
                            ),
                        )
                    ]
                    pred.append(
                        {
                            "prediction": p["text"],
                            "ended_naturally": p["ended_naturally"],
                            "forced_length": False,
                            "truncated": p["truncated"],
                        }
                    )
                key = "/".join((poolname, scorer, mode))
                metrics[key], details[key] = stats_for(groups, pred)
    comparisons = {}
    for poolname in ("unconstrained", "without_lexical_fusion", "union"):
        for mode in ("forward_only", "bidirectional"):
            comparisons[poolname + "/" + mode] = paired_bootstrap(
                details[poolname + "/base/" + mode],
                details[poolname + "/no_lexical_fusion/" + mode],
            )
    dump(
        OUT / "candidate_scoring_metrics.json",
        {
            "status": "complete",
            "groups": len(groups),
            "metrics": metrics,
            "full_minus_no_lex_scorer_paired_bootstrap": comparisons,
            "mean_union_size": float(np.mean([len(g["pool"]) for g in unions])),
            "max_union_size": max(len(g["pool"]) for g in unions),
            "not_same_candidate_budget_as_main": True,
            "same_reverse_scores_all_conditions": True,
            "max_repeated_reverse_score_discrepancy": max_reverse_discrepancy,
        },
    )


def audit_and_report():
    result = read(OUT / "metrics.json")
    assert result["status"] == "complete"
    protocol = read(OUT / "evaluation_protocol.json")
    gg = list(rows(ROOT / "data/couplet/evaluation/test_locked.jsonl"))
    tok = CharacterTokenizer()
    audit = {}
    extras = {}
    all_details = {}
    files = {
        "forward": FORWARD,
        "reverse": ROOT / "models/couplet/reverse.pt",
        "vocab": ROOT / "tokenizer/vocab.json",
        "lexicon": LEX,
        "test": ROOT / "data/couplet/evaluation/test_locked.jsonl",
        "core": ROOT / "data/couplet/evaluation/core_test.jsonl",
    }
    hashes = {k: sha(p) for k, p in files.items()}
    for k, v in hashes.items():
        assert v == protocol[k + "_sha256"]
    for variant in VARIANTS:
        pools = {r["upper"]: r for r in rows(OUT / "candidates" / (variant + ".jsonl"))}
        rev = {r["upper"]: r["scores"] for r in rows(OUT / "reverse_scores" / (variant + ".jsonl"))}
        assert len(pools) == len(rev) == len(gg)
        for mode, w in [("forward_only", 0.0), ("bidirectional", 0.25)]:
            details = list(rows(OUT / "predictions" / (variant + "_" + mode + ".jsonl")))
            assert [r["upper"] for r in details] == [g["upper"] for g in gg]
            # Re-select directly from raw numbers, without invoking any rule checker.
            checks = []
            differences = []
            for g, d in zip(gg, details):
                u = g["upper"]
                pp = pools[u]["pool"]
                rr = rev[u]
                assert len(rr) == len(pp)
                assert all(math.isfinite(x) for x in rr)
                winner = max(range(len(pp)), key=lambda j: pp[j]["forward_score"] + w * rr[j])
                p = pp[winner]
                assert d["tokens"] == p["tokens"] and d["prediction"] == tok.decode(p["tokens"])
                assert not any(t in (0, 2, 3, 4, 5) for t in p["tokens"])
                assert d["score_used_reverse"] == bool(w)
                checks.append(rule_check(u, d["prediction"]))
                differences.append(len(p["tokens"]) - len(u))
            applicable = [r for r in checks if r["modern_foot_applicable"]]
            key = variant + "/" + mode
            all_details[key] = details
            extras[key] = {
                "pure_han_rate": float(
                    np.mean(
                        [
                            all(han(c) for c in d["prediction"]) and bool(d["prediction"])
                            for d in details
                        ]
                    )
                ),
                "foot_applicable": len(applicable),
                "foot_pass_rate_applicable": float(
                    np.mean([r["modern_foot_pass"] for r in applicable])
                )
                if applicable
                else None,
                "unknown_lower_tone": sum(r["lower_modern_tone"] == "?" for r in checks),
                "no_same_position_rate": float(np.mean([r["no_same_position"] for r in checks])),
                "repeat_pattern_rate": float(np.mean([r["repeat_pattern"] for r in checks])),
                "shorter_rate": float(np.mean(np.array(differences) < 0)),
                "longer_rate": float(np.mean(np.array(differences) > 0)),
                "score_coverage": 1.0,
                "score_failure_count": 0,
                "mean_output_tokens": float(np.mean([len(d["tokens"]) for d in details])),
            }
            audit[key] = {
                "groups": len(details),
                "selection_without_checker_identical": True,
                "no_control_tokens_in_body": True,
                "finite_reverse_scores": True,
            }
    # The free branch of the search mask must not call the external rule engine.
    source = (ROOT / "couplet_gpt/evaluation/ablation.py").read_text(encoding="utf8")
    tree = ast.parse(source)
    allowed = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "allowed"
    )
    condition = allowed.body[0]
    assert isinstance(condition, ast.If) and "rules" in ast.unparse(condition.test)
    assert "rule_engine.allowed" not in ast.unparse(condition.orelse)
    dump(
        OUT / "reproducibility_checks.json",
        {
            "status": "pass",
            "hashes_unchanged": hashes,
            "selection_audits": audit,
            "free_mask_branch_has_no_rule_engine_call": True,
            "source_sha256": {
                p.name: sha(p)
                for p in (
                    ROOT / "couplet_gpt/evaluation/ablation.py",
                    ROOT / "couplet_gpt/evaluation/report.py",
                )
            },
            "notes": [
                "Offline checker called only after raw-number selection verification.",
                "Peak GPU memory and per-request latency were not instrumented; do not invent values.",
            ],
        },
    )
    dump(OUT / "constraint_metrics.json", extras)
    chosen = sorted(
        gg, key=lambda g: hashlib.sha256(("v13-review:" + g["upper"]).encode()).hexdigest()
    )[:40]
    index = {k: {r["upper"]: r for r in v} for k, v in all_details.items()}
    review = [
        {
            **g,
            "outputs": {k: index[k][g["upper"]]["prediction"] for k in index},
            "expert_review": "not_performed",
        }
        for g in chosen
    ]
    jsonl(OUT / "qualitative_examples.jsonl", review)
    names = {
        "with_constraints": "原规则完整",
        "unconstrained": "无规则完整",
        "without_lexical_fusion": "关闭词汇融合",
        "without_contextual_features": "关闭上下文投影",
        "without_word_embeddings": "关闭词向量残差",
        "with_small_lexicon": "旧8192词匹配",
        "without_position_alignment": "关闭字位对齐",
    }
    s = result["results"]
    boot = result["paired_bootstrap"]
    lines = [
        "# 关闭外部解码规则与反向选择的消融结果 v13",
        "",
        "2026-11-06。全部14组使用同一2617个锁定上联，已完成生成、评分和校验。保留既有ChineseBERT＋词汇适配器前向与反向权重，没有重新训练，也没有替换默认模型或网页。",
        "",
        "## 🧭 结果解读",
        "",
        "1. **可以关闭外部规则作模型能力评测，但结构保证会消失。** 完整模型双向选择的chrF从3.1585降到3.0937，差别较小；结构代理却从100%降至76.88%。无规则等长率仍为100%，因此主要缺口并非长度。原规则组的100%是程序约束的结果，不能归功于模型全部学会。",
        "2. **反向选择有小幅数值收益，稳定性证据有限。** 无规则完整模型由2.9625升到3.0937，结构代理仅从76.73%升到76.88%。逐句chrF差95%区间跨零，不能声称反向选择显著优于仅正向；它也没有代替硬规则。",
        "3. **词汇融合在这次无规则全量评测中有贡献。** 仅正向时关闭融合使chrF从2.9625降到2.5367，双向时从3.0937降到2.7158；两种选择的逐句差区间均未跨零。上下文投影、词向量残差和扩充词库在仅正向时也有正向证据；加入反向后这些单项差异区间跨零，证据变弱。不能据此把小集旧结果概括成“词汇融合无效”。",
        "4. **当前权重非常依赖字位对齐通道。** 屏蔽后双向chrF仅0.6045，约24%为空输出。该结果说明训练好的模型依赖这条通道；不能外推为另一套从头训练的无对齐架构必然失败，也不能记为COLING词性／依存模块的收益。",
        "5. **建议报告以无规则完整模型作为能力主结果，同时列出正向与双向两列。** 有规则版本作为工程增强对照。当前没有足够证据仅凭这轮结果继续修改架构；语义、搭配、词性及文学质量还需人工复核，不能由chrF或结构代理直接认定。",
        "",
        "## 🧪 条件与主要结果",
        "",
        "每池 beam16＋8条采样，top20、温度0.8；“仅正向”仍有搜索。双向选择为平均log概率F＋0.25R，二者都计EOS。每行两种选择严格共用候选池。除R0外均关闭等长、汉字、同位避字、重复模式、平仄及评分前结构过滤，只保留协议Token掩码。EOS自然停止，最多63个新Token，触顶不修复。",
        "",
        "|条件|仅正向 chrF|双向 chrF|反向带来的chrF差|仅正向等长%|双向等长%|仅正向结构代理%|双向结构代理%|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for v in VARIANTS:
        a = s[v + "/forward_only"]
        b = s[v + "/bidirectional"]
        lines.append(
            f"|{names[v]}|{a['chrf']:.4f}|{b['chrf']:.4f}|{b['chrf'] - a['chrf']:+.4f}|{100 * a['exact_token_length_rate']:.2f}|{100 * b['exact_token_length_rate']:.2f}|{100 * a['structure_proxy_pass_rate']:.2f}|{100 * b['structure_proxy_pass_rate']:.2f}|"
        )
    lines += [
        "",
        "等长按原始Token数统计；结构代理由原检查器离线计算，不代表完整传统联律或文学合格。chrF是与参考下联的字符重叠，不能直接替代文学评价。",
        "",
        "## 📊 关键差异与不确定性",
        "",
        "|比较|语料chrF差|逐句chrF平均差|逐句差95%配对区间|",
        "|---|---:|---:|---|",
    ]
    for label, left, right, bk in [
        (
            "关闭规则，仅正向",
            "unconstrained/forward_only",
            "with_constraints/forward_only",
            "with_constraints_minus_unconstrained/forward_only",
        ),
        (
            "关闭规则，双向",
            "unconstrained/bidirectional",
            "with_constraints/bidirectional",
            "with_constraints_minus_unconstrained/bidirectional",
        ),
        (
            "无规则完整，加入反向",
            "unconstrained/bidirectional",
            "unconstrained/forward_only",
            "unconstrained/reverse_minus_forward",
        ),
    ]:
        x = boot[bk]
        sign = -1 if "minus_unconstrained" in bk else 1
        lo, hi = x["ci95"]
        interval = (-hi, -lo) if sign == -1 else (lo, hi)
        lines.append(
            f"|{label}|{s[left]['chrf'] - s[right]['chrf']:+.4f}|{sign * x['mean_sentence_chrf_difference']:+.4f}|[{interval[0]:+.4f}, {interval[1]:+.4f}]|"
        )
    lines += [
        "",
        "区间来自1000次按上联配对重采样，针对逐句chrF均值，并非语料chrF区间，也不覆盖重新训练或多种子不确定性。",
        "",
        "下表是各组件屏蔽相对无规则完整模型的逐句chrF差；负值表示屏蔽后下降。",
        "",
        "|屏蔽条件|仅正向均值差与95%区间|双向均值差与95%区间|",
        "|---|---|---|",
    ]
    for v in list(VARIANTS)[2:]:
        cells = []
        for mode in ("forward_only", "bidirectional"):
            x = boot[v + "_minus_unconstrained/" + mode]
            lo, hi = x["ci95"]
            cells.append(f"{x['mean_sentence_chrf_difference']:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        lines.append("|" + names[v] + "|" + "|".join(cells) + "|")
    lines += [
        "",
        "## 🧾 停止、失败与规则诊断",
        "",
        "|条件／选择|自然EOS%|截断%|空输出%|UNK%|纯汉字%|适用句脚声调通过%|平均Token数|平均候选数|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for v in VARIANTS:
        for mode in ("forward_only", "bidirectional"):
            k = v + "/" + mode
            a = s[k]
            e = extras[k]
            eos = "强制结束" if v == "with_constraints" else f"{100 * a['natural_eos_rate']:.2f}"
            lines.append(
                f"|{names[v]}／{'正向' if mode == 'forward_only' else '双向'}|{eos}|{100 * a['truncation_rate']:.2f}|{100 * a['empty_rate']:.2f}|{100 * a['UNK_rate']:.2f}|{100 * e['pure_han_rate']:.2f}|{100 * e['foot_pass_rate_applicable']:.2f}|{e['mean_output_tokens']:.2f}|{a['mean_candidate_count']:.2f}|"
            )
    lines += [
        "",
        "所有组2617/2617可评分，没有丢弃不合格候选或评分失败后回退。自然EOS也可能产生空串，不能作为质量合格指标。声调仅为旧单字现代音代理，多音字及古入声未获得完整核验；原v4搜索使用上下文和多音字可行性，和此单字代理口径不同，因此有规则组的声调代理也不是100%。适用数和未知项见constraint_metrics.json。",
        "",
        "|完整模型条件|等长%|无同位字%|重复模式一致%|结构代理%|",
        "|---|---:|---:|---:|---:|",
    ]
    for v in ("with_constraints", "unconstrained"):
        for mode in ("forward_only", "bidirectional"):
            k = v + "/" + mode
            e = extras[k]
            a = s[k]
            lines.append(
                f"|{names[v]}／{mode}|{100 * a['exact_token_length_rate']:.2f}|{100 * e['no_same_position_rate']:.2f}|{100 * e['repeat_pattern_rate']:.2f}|{100 * a['structure_proxy_pass_rate']:.2f}|"
            )
    lines += ["", "## 🔀 词汇候选与评分的交叉诊断", ""]
    if (OUT / "candidate_scoring_metrics.json").exists():
        cr = read(OUT / "candidate_scoring_metrics.json")
        lines += [
            "347例固定子集；两种候选池和它们的并集均分别由完整正向与屏蔽词汇正向打分。反向分数始终固定。该表用于拆分候选生成与评分的作用。并集拥有更多候选，不能当作同预算主实验提升。",
            "",
            "|生成池|正向评分器|仅正向 chrF|双向 chrF|",
            "|---|---|---:|---:|",
        ]
        for p in ("unconstrained", "without_lexical_fusion", "union"):
            for scorer in ("base", "no_lexical_fusion"):
                lines.append(
                    f"|{p}|{scorer}|{cr['metrics'][p + '/' + scorer + '/forward_only']['chrf']:.4f}|{cr['metrics'][p + '/' + scorer + '/bidirectional']['chrf']:.4f}|"
                )
        lines += [
            "",
            f"并集平均{cr['mean_union_size']:.2f}个、最多{cr['max_union_size']}个候选；重叠候选的反向分数在不同batch中的最大差为{cr['max_repeated_reverse_score_discrepancy']:.6f}。本诊断统一使用第一次保存的同一反向分数。",
        ]
        lines += [
            "",
            "该347例诊断数值上更支持“词汇信息改善评分判断”，没有显示完整词汇生成池总是优于屏蔽词汇池：固定完整评分器时，两种生成池双向chrF分别为2.9949和2.9905，十分接近；固定任一种池时，完整评分器双向chrF均较高。并集也没有带来稳定的双向改善。候选内容与选句评分的收益不能混为一谈；这里仅是小集诊断，完整配对区间在candidate_scoring_metrics.json，不能外推为全量显著结论。",
        ]
    lines += [
        "",
        "## 🔒 范围、复核与复现",
        "",
        "- 这是固定权重上的推理组件屏蔽，不能证明删除组件后重训一定更好。反向模型自身从未被屏蔽词汇；关闭反向选择仅指其分数不参与选句。",
        "- 测试集历史上使用过；本轮没有用它调反向权重或挑选消融组。347例为全量评测的子集，不是另一份独立测试。",
        "- 4种长度及空串／重复／UNK／标点边界预检通过。14组重新从原始分数选句，与离线统计结果逐字一致；检查器不干预生成或选择。权重、词库、字符表及测试文件哈希与冻结协议一致。",
        "- 保存所有原始候选Token、EOS／截断、F/R分数和逐例输出。固定哈希抽取40例供复核，未编造人工词性、语义或文学评分。",
        "- 各组生成耗时在candidates/*_generation_summary.json。未埋点记录逐请求延迟、完整评分耗时或峰值显存，因此不作这些性能比较；相同beam数并不等于相同FLOPs。",
        "- 执行：`python -m couplet_gpt.evaluation.ablation --phase all`；交叉诊断及报告：`python -m couplet_gpt.evaluation.report`。",
        "",
        "完整机器结果：results/couplet/ablation/metrics.json；校验：reproducibility_checks.json；逐例：predictions/；候选：candidates/；交叉诊断：candidate_scoring_metrics.json。",
        "",
    ]
    (ROOT / "results/couplet/ablation/analysis.md").write_text("\n".join(lines), encoding="utf8")
    dump(
        OUT / "completion.json",
        {
            "status": "complete",
            "groups": 2617,
            "conditions": 14,
            "cross_groups": 347,
            "audit": "pass",
            "default_model_changed": False,
            "report": str(ROOT / "results/couplet/ablation/analysis.md"),
            "result_sha256": sha(OUT / "metrics.json"),
        },
    )
    print("REPORT AND AUDIT COMPLETE", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf8")
    torch.set_num_threads(3)
    cross()
    audit_and_report()
