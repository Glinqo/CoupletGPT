"""Screen every candidate, preserve provenance, split by source components for local SFT."""

import collections, hashlib, json, random, re, unicodedata
from pathlib import Path
from couplet_gpt.data.comment_corpus import ROOT, OUT, han, normalize, stable, dump, jsonl

DEST = OUT / "curated_local_v1"
PATTERNS = {
    "external_factual_claim_requires_separate_verification": r"出自|典出|[《》]|生于|卒于|首创|最早|始于|首开|记载|传说|始建|三大|之最|据载|号称|故乡|\d{3,4}|最深|最大|最高峰|历史上|一千七百年",
    "unverified_technical_assertion": r"平仄|失替|失对|入声|音韵|声调|马蹄韵|名词|动词|形容词|词性|词类|借对|流体力学|通感",
    "missing_discourse_or_comparison": r"上文|下文|上述|以上|下面|前文|前联|后联|前两|后两|上诗|下诗|笔者|我认为|试看|再看|此处不|本文|本期|本栏目|且看|如下|再如|另[一二两]副|另[一二两]首|另一联|举凡|综上|由此可见",
    "poem_excerpt_not_independent_couplet": r"该诗|此诗|此绝|该绝|绝句|五律|七律|颈联|颔联|首联|尾联|这首诗|诗中|首诗",
    "overstated_or_formulaic_claim": r"逻辑闭环|终极解构|哲学本源|首屈一指|独一无二|高质量发展|千古绝唱",
    "unresolved_conclusion_or_relative_date": r"总而言之|还如|今年|去年",
    "publisher_postscript_not_specific_review": r"^跋[：:]",
    "unexplained_latin_artifact": r"[A-Za-z]",
}
INSTRUCTION = "请用一段连贯的中文评论这副对联，结合具体字词说明上下联的含义、对应关系与写法。有不足可以指出，不补造出处。"


def source_context(row):
    """Use only original headings and the closest short label, never target-derived facts."""
    page = json.loads((OUT / row["source_file"]).read_text(encoding="utf8"))
    context = row["context"]
    # Restore a nearby original short heading if the old extractor missed it due to punctuation.
    a, b = row["couplet_span"]
    before = page["body_text"][:a].rstrip().splitlines()
    labels = []
    for s in before[-2:]:
        s = normalize(s)
        if 1 <= len(han(s)) <= 28 and not re.search("[。！？]", s) and s not in context:
            labels.append(s)
    return context + ("；题签：" + "；".join(labels) if labels else ""), page


def display_pair(row):
    raw = normalize(row["raw_couplet"])
    parts = raw.splitlines()
    if len(parts) == 2:
        parts = [s.strip("；。！？ \t") for s in parts]
    elif raw.count("；") == 1:
        parts = [s.strip("；。！？ \t") for s in raw.split("；")]
    else:
        parts = [row["upper"], row["lower"]]
    if len(parts) != 2 or tuple(map(han, parts)) != (row["upper"], row["lower"]):
        return row["upper"], row["lower"]
    return tuple(parts)


def facts(row):
    # No POS/dependency predictions, no target-dependent selection, no global qualified judgement.
    a, b = row["upper_display"], row["lower_display"]

    def lengths(s):
        return "、".join(str(len(han(t))) for t in re.split("[，、,]", s) if han(t))

    return f"两句字数：上{len(han(a))}、下{len(han(b))}。标点分句字数：上{lengths(a)}；下{lengths(b)}。句脚：上{han(a)[-1]}、下{han(b)[-1]}。这些是字面统计，不代表整联合律。"


def prompt_text(row, branch):
    s = INSTRUCTION + "\n上联：" + row["upper_display"] + "\n下联：" + row["lower_display"]
    if row.get("context"):
        s += "\n背景：" + row["context"]
    if branch == "evidence":
        s += "\n核查事实：" + facts(row)
    return s + "\n评论："


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    dump(DEST / "data_ready.json", {"status": "preparing_not_ready"})
    spot_file = DEST / "assistant_spotcheck_decisions_50.jsonl"
    spot = (
        {
            r["sample_id"]: r
            for r in (json.loads(s) for s in spot_file.read_text(encoding="utf8").splitlines())
        }
        if spot_file.exists()
        else {}
    )
    proof_file = DEST / "verified_reference_exceptions.json"
    proofs = json.loads(proof_file.read_text(encoding="utf8")) if proof_file.exists() else {}
    raw = [
        json.loads(s)
        for s in (OUT / "modern_main_candidates.jsonl").read_text(encoding="utf8").splitlines()
    ]
    accepted = []
    ledger = []
    for original in raw:
        r = dict(original)
        t = r["target_comment"]
        reasons = []
        if not 80 <= len(t) <= 400:
            reasons.append("outside_full_target_length_80_400")
        if "duilianzazhi" not in r.get("source_file", ""):
            reasons.append("community_provenance_or_variant_not_fully_checked")
        manual = r.get("assistant_audit", {}).get("assistant_decision")
        if manual not in [None, "aligned_candidate"]:
            reasons.append("previous_assistant_review_has_unresolved_issue")
        if spot.get(r["sample_id"], {}).get("decision") == "hold":
            reasons.append("second_assistant_reading_identified_specific_problem")
        for name, pattern in PATTERNS.items():
            if re.search(pattern, t):
                if (
                    name == "external_factual_claim_requires_separate_verification"
                    and r["sample_id"] in proofs
                ):
                    r["verified_reference_scope"] = proofs[r["sample_id"]]
                else:
                    reasons.append(name)
        anchors = []
        for side in ["upper", "lower"]:
            h = r[side]
            aa = sorted({h[i : i + 2] for i in range(len(h) - 1) if h[i : i + 2] in t})
            anchors.append(aa)
        if not all(anchors):
            reasons.append("no_literal_specific_reference_to_both_sides")
        if re.search(r"[�<>]|\\u[0-9a-fA-F]{4}", t):
            reasons.append("visible_transcription_artifact")
        if any(unicodedata.category(c) == "Co" for c in t + r["raw_couplet"] + r["context"]):
            reasons.append("private_use_font_glyph_not_recoverable_as_character")
        prefixed = [
            m[1]
            for m in re.finditer(r"(?:^|\n)([\u4e00-\u9fff]{2,6})[：:]", t)
            if m[1] not in ["上联", "下联", "全联", "注释", "词义", "解释", "例如", "赏析", "简析"]
        ]
        if len(prefixed) > 1:
            reasons.append("multiple_unresolved_commentator_prefixes")
        # Definitions, biographies and unrelated prose are not silently cut into fluent targets.
        if re.search(r"^作者[：:]|^.{0,8}[，,]字.{1,8}[，,]|\n.{0,18}一生|\n对联是", t):
            reasons.append("biography_or_article_conclusion_mixed_with_review")
        if re.search("上句|下句|词性|平仄", t) and not re.search("上联|下联|此联|全联|联语", t):
            reasons.append("genre_or_technical_scope_unclear")
        r["upper_display"], r["lower_display"] = display_pair(r)
        if not reasons:
            r["context"], page = source_context(r)
            r["article_group"] = "article:" + stable(page["body_text"])[:24]
            if len(prompt_text(r, "evidence")) + len(t) + 2 > 768:
                reasons.append("exceeds_context_768")
        ledger.append(
            {
                "sample_id": r["sample_id"],
                "family_id": r["family_id"],
                "decision": "hold" if reasons else "automatic_screen_pass",
                "reasons": reasons,
                "upper_anchors": anchors[0],
                "lower_anchors": anchors[1],
                "method": "deterministic_full_corpus_screen_with_previous_assistant_reading",
                "human_review": False,
                "all_external_facts_verified": False,
            }
        )
        if not reasons:
            r["quality_tier"] = "published_commentary_assistant_screened_not_expert_gold"
            r["response"] = re.sub(r"^(?:◎)?(?:评语|赏析|简析)[：:]\s*", "", t)
            r["instruction"] = INSTRUCTION
            r["response_format_change"] = (
                "removed_explicit_review_heading_only" if r["response"] != t else "none"
            )
            r["selection_score"] = min(len(set(anchors[0] + anchors[1])), 12) - len(t) / 1000
            accepted.append(r)
    # Do not inflate independent works with multiple reviewers.
    best = {}
    for r in accepted:
        key = r["family_id"]
        if key not in best or r["selection_score"] > best[key]["selection_score"]:
            best[key] = r
    selected = sorted(best.values(), key=lambda r: r["sample_id"])
    # Families are already unique. Whole articles stay together; duplicate body hashes share a group.
    groups = collections.defaultdict(list)
    for r in selected:
        groups[r["article_group"]].append(r)
    keys = sorted(groups)
    random.Random(20261105).shuffle(keys)
    split = {k: [] for k in ["train", "validation", "test"]}
    group_split = {}
    for name in ["test", "validation"]:
        while keys and len(split[name]) < 50:
            k = keys.pop()
            split[name].extend(groups[k])
            group_split[k] = name
    for k in keys:
        split["train"].extend(groups[k])
        group_split[k] = "train"
    assert len(selected) >= 300, (
        f"Only {len(selected)} independent assistant-screened works; do not pad."
    )
    assert min(len(split["validation"]), len(split["test"])) >= 50 and len(split["train"]) >= 180
    for name, rr in split.items():
        for r in rr:
            r["split"] = name
        rr.sort(key=lambda r: r["sample_id"])
        jsonl(DEST / (name + ".jsonl"), rr)
    # Vocabulary extension is shared across matched C1/C2 and never overwrites the generator tokenizer.
    vocab = json.loads((ROOT / "runs/pretrain_v1/artifacts/vocab.json").read_text(encoding="utf8"))
    new_chars = sorted(
        set(
            "".join(
                prompt_text(r, b) + r["response"]
                for r in selected
                for b in ["pair_only", "evidence"]
            )
        )
        - set(vocab)
    )
    original_size = len(vocab)
    for c in new_chars:
        vocab[c] = len(vocab)
    dump(DEST / "vocab.json", vocab)
    dump(
        DEST / "tokenizer_config.json",
        {
            "type": "unicode_codepoint",
            "normalization": "upstream NFC+t2s, preserve punctuation and original targets",
            "base_vocab_size": original_size,
            "vocab_size": len(vocab),
            "added_characters": new_chars,
            "scope": "independent_commentary_model_only",
            "transductive_schema_note": "Only the alphabet is audited across the corpus; no held-out target words/embeddings are pretrained.",
        },
    )
    jsonl(DEST / "quality_decisions_all689.jsonl", ledger)
    jsonl(DEST / "accepted_all.jsonl", selected)
    reasons = collections.Counter(s for r in ledger for s in r["reasons"])
    manifest = {
        "status": "ready_for_local_experimental_sft",
        "raw_candidates": len(raw),
        "screen_pass_rows": len(accepted),
        "independent_selected_works": len(selected),
        "split_rows": {k: len(v) for k, v in split.items()},
        "source_article_groups": len(groups),
        "hold_reasons": dict(reasons),
        "review_mode": "full_automatic_screen_and_assistant_audit_not_human_expert",
        "user_instruction": "请你核查数据集的质量，然后完成微调",
        "authorization_date": "2026-11-05",
        "protocol_change": "Latest user instruction authorizes assistant quality review followed by local SFT; previous wait-for-two-human-review gate is not claimed completed.",
        "rights_status": "publisher_training_terms_unconfirmed_recorded_as_unresolved_not_fabricated_permission",
        "experiment_scope": "local_course_experiment; source permissions have not been certified by this audit",
        "human_review_count": 0,
        "expert_gold": False,
        "context_length": 768,
        "max_epochs": 5,
        "seed": 2026,
        "base_checkpoint": "runs/pretrain_v1/artifacts/best_model.pt",
        "vocab_size": len(vocab),
        "new_chars": new_chars,
        "source_sha256": hashlib.sha256(
            (OUT / "modern_main_candidates.jsonl").read_bytes()
        ).hexdigest(),
        "file_sha256": {
            n: hashlib.sha256((DEST / n).read_bytes()).hexdigest()
            for n in ["train.jsonl", "validation.jsonl", "test.jsonl", "vocab.json"]
        },
    }
    dump(DEST / "data_ready.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
