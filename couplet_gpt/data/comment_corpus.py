"""Traceable pair/comment extraction with an explicit modern-comment SFT data gate."""

from couplet_gpt.paths import ROOT as PROJECT_ROOT
import collections, csv, difflib, hashlib, json, re, unicodedata
from pathlib import Path
from opencc import OpenCC

ROOT = PROJECT_ROOT
OUT = ROOT / "runs/comment_sft_v2"
CC = OpenCC("t2s")


def normalize(s):
    return CC.convert(unicodedata.normalize("NFC", s)).strip()


def stable(s):
    return hashlib.sha256(s.encode("utf8")).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding="utf8"))


def dump(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2), encoding="utf8")


def jsonl(p, rr):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf8") as f:
        for r in rr:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def han(s):
    return "".join(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", s))


def split_pair(text):
    text = text.strip("。！!？? \n")
    if "；" in text:
        halves = text.split("；")
        if len(halves) != 2:
            return None
    else:
        choices = []
        for match in re.finditer("[，,]", text):
            left, right = text[: match.start()], text[match.end() :]
            if len(han(left)) == len(han(right)) and len(han(left)) >= 4:
                choices.append((left, right))
        if len(choices) != 1:
            return None
        halves = choices[0]
    a, b = map(han, halves)
    if not 4 <= len(a) == len(b) <= 64:
        return None
    return a, b


def main():
    candidates = []
    rejected = []
    summary = collections.Counter()
    for path in sorted((OUT / "sources/wikisource").glob("*.json")):
        page = read(path)
        if "/" not in page["title"]:
            continue
        for number, paragraph in enumerate(page["paragraphs"]):
            quotes = list(re.finditer(r"[「“]([^「」“”]{4,320})[」”]", paragraph))
            paired = [(m, split_pair(normalize(m[1]))) for m in quotes]
            paired = [
                (m, p)
                for m, p in paired
                if p
                and re.search(
                    "[联聯楹对對挽题題书書]", paragraph[max(0, m.start() - 60) : m.start()]
                )
            ]
            if not paired:
                continue
            for match, pair in paired:
                sourceid = page["title"] + f":p{number}:q{match.start()}"
                if len(paired) > 1:
                    rejected.append(
                        {
                            "source_id": sourceid,
                            "reason": "multiple_couplets_in_paragraph_require_manual_alignment",
                            "source_paragraph": paragraph,
                        }
                    )
                    continue
                tail = paragraph[match.end() :].strip()
                # A later quotation may refer to another work; never silently attribute it to the current pair.
                if re.search("[「“]", tail):
                    rejected.append(
                        {
                            "source_id": sourceid,
                            "reason": "later_quotation_or_comparison_requires_manual_alignment",
                            "source_paragraph": paragraph,
                        }
                    )
                    continue
                if len(han(tail)) < 30:
                    rejected.append(
                        {
                            "source_id": sourceid,
                            "reason": "insufficient_specific_comment_after_couplet",
                            "comment": tail,
                        }
                    )
                    continue
                if not re.search(
                    "联|聯|句|意|语|語|切|妙|工|佳|高|对|對|趣|笔|筆|境|气|氣|辞|辭|胜|勝", tail
                ):
                    rejected.append(
                        {
                            "source_id": sourceid,
                            "reason": "no_clear_commentary_signal",
                            "comment": tail,
                        }
                    )
                    continue
                if len(normalize(tail)) > 500:
                    rejected.append(
                        {
                            "source_id": sourceid,
                            "reason": "comment_overlength_or_unseparated_anecdote",
                            "comment": tail,
                        }
                    )
                    continue
                before = paragraph[: match.start()].strip()
                comment = normalize(tail)
                row = {
                    "sample_id": "classical:" + stable(sourceid)[:16],
                    "source_id": sourceid,
                    "upper": pair[0],
                    "lower": pair[1],
                    "raw_couplet": match[1],
                    "context": normalize(before[-240:]),
                    "source_comment": tail,
                    "target_comment": comment,
                    "source_paragraph": paragraph,
                    "source_span": [match.end(), len(paragraph)],
                    "couplet_span": [match.start(1), match.end(1)],
                    "source_url": page["url"],
                    "source_revision": page["fixed_revision_url"],
                    "source_author": "梁章钜",
                    "source_license_links": page["license_links"],
                    "rights_status": "public_domain_original_attributed_transcription",
                    "label_origin": "original_classical_human_commentary",
                    "language_style": "classical",
                    "review_status": "automatic_candidate_not_gold",
                    "modern_main_eligible": False,
                    "covers": ["original_comment_scope_not_forced_to_four_aspects"],
                }
                candidates.append(row)
                summary["classical_candidate"] += 1
    meta = read(OUT / "xasoss_source_meta.json")
    revision = meta["sha"]
    for path in sorted((OUT / "sources/xasoss").rglob("content.jsonl")):
        filename = path.relative_to(OUT / "sources/xasoss").as_posix()
        for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines()):
            if not line.strip():
                continue
            r = json.loads(line)
            title = normalize(r.get("title", ""))
            text = normalize(r.get("primary_text", ""))
            comment = normalize(r.get("theme", ""))
            if "联" not in title or "联句" in title:
                continue
            pair = split_pair(text)
            reason = None
            if not pair:
                reason = "cannot_unambiguously_split_standalone_couplet"
            elif len(comment) < 80:
                reason = "modern_comment_too_short_for_main_paragraph_task"
            elif len(comment) > 500:
                reason = "modern_comment_overlength"
            elif re.search("教授|陈文新|蔡义江|周汝昌|俞平伯|刘心武|蒋勋", comment):
                reason = "third_party_modern_commentary_attribution_review"
            elif "世上洞明" in text:
                reason = "couplet_variant_requires_edition_check_not_silent_correction"
            elif "薛定钗" in comment:
                reason = "proper_name_typo_requires_source_check"
            elif ">" in comment:
                reason = "visible_markup_or_transcription_artifact"
            if reason:
                rejected.append(
                    {
                        "source_id": filename + ":" + str(number),
                        "title": title,
                        "reason": reason,
                        "comment": comment,
                    }
                )
                summary["modern_rejected"] += 1
                continue
            sourceid = filename + ":" + str(number)
            candidates.append(
                {
                    "sample_id": "modern:" + stable(sourceid)[:16],
                    "source_id": sourceid,
                    "title": title,
                    "upper": pair[0],
                    "lower": pair[1],
                    "raw_couplet": r["primary_text"],
                    "context": "《红楼梦》相关题联；" + title,
                    "source_comment": r.get("theme", ""),
                    "target_comment": comment,
                    "source_paragraph": r.get("theme", ""),
                    "source_span": [0, len(r.get("theme", ""))],
                    "source_url": "https://huggingface.co/datasets/chanyo111/xasoss",
                    "source_revision": revision,
                    "source_file": filename,
                    "source_author": "xasoss publisher/community compiler",
                    "rights_status": "publisher_declared_cc_by_4.0_with_record_level_provenance_review",
                    "label_origin": "community_original_comment_not_expert_gold",
                    "language_style": "modern",
                    "review_status": "automatic_candidate_not_gold",
                    "modern_main_eligible": True,
                    "covers": ["内容评论"],
                }
            )
            summary["modern_candidate"] += 1
    publisher_path = OUT / "publisher_modern_candidates.jsonl"
    if publisher_path.exists():
        publisher_rows = [
            json.loads(line)
            for line in publisher_path.read_text(encoding="utf8").splitlines()
            if line.strip()
        ]
        candidates.extend(publisher_rows)
        summary["publisher_modern_candidate"] = len(publisher_rows)
        publisher_quarantine = OUT / "publisher_modern_quarantine.jsonl"
        if publisher_quarantine.exists():
            rejected.extend(
                json.loads(line)
                for line in publisher_quarantine.read_text(encoding="utf8").splitlines()
                if line.strip()
            )
    audit_path = OUT / "assistant_audit_round1_50.jsonl"
    audits = (
        {
            r["sample_id"]: r
            for r in (json.loads(s) for s in audit_path.read_text(encoding="utf8").splitlines())
        }
        if audit_path.exists()
        else {}
    )
    retained = []
    for row in candidates:
        if row.get("genre") == "poetry_bell_special_rules_review_required":
            row["modern_main_eligible"] = False
            row["auxiliary_reason"] = (
                "Poetry-bell rules differ from ordinary couplets; keep separate."
            )
        audit = audits.get(row["sample_id"])
        if audit:
            row["assistant_audit"] = {
                k: audit[k]
                for k in [
                    "assistant_decision",
                    "assistant_notes",
                    "reviewer_type",
                    "all_facts_independently_checked",
                ]
            }
            row["review_status"] = (
                "assistant_" + audit["assistant_decision"] + "_not_training_approved"
            )
            if audit["assistant_decision"].startswith("reject_"):
                rejected.append(
                    {
                        "source_id": row["source_id"],
                        "sample_id": row["sample_id"],
                        "reason": "assistant_audit_" + audit["assistant_decision"],
                        "notes": audit["assistant_notes"],
                    }
                )
                continue
            if audit["assistant_decision"] == "genre_auxiliary":
                row["modern_main_eligible"] = False
        retained.append(row)
    candidates = retained
    # Exact/reversed pair, close variants, and exact comment families connected before any future split.
    parents = list(range(len(candidates)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    groups = {}
    for i, row in enumerate(candidates):
        keys = [
            "pair:" + "|".join(sorted([row["upper"], row["lower"]])),
            "comment:" + han(row["target_comment"]),
        ]
        for key in keys:
            if key in groups:
                parents[find(i)] = find(groups[key])
            else:
                groups[key] = i
    # Only compare pairs with many shared bigrams. Variants stay together, never counted as independent data.
    postings = collections.defaultdict(set)
    pair_texts = []
    for i, row in enumerate(candidates):
        pair_text = "|".join(sorted([row["upper"], row["lower"]]))
        pair_texts.append(pair_text)
        grams = {
            pair_text[n : n + 2]
            for n in range(len(pair_text) - 1)
            if "|" not in pair_text[n : n + 2]
        }
        possible = collections.Counter(j for gram in grams for j in postings[gram])
        for j, hits in possible.items():
            other = pair_texts[j]
            if (
                hits >= max(3, int(min(len(pair_text), len(other)) * 0.5))
                and abs(len(pair_text) - len(other)) <= 4
            ):
                if difflib.SequenceMatcher(None, pair_text, other, autojunk=False).ratio() >= 0.90:
                    parents[find(i)] = find(j)
        for gram in grams:
            postings[gram].add(i)
    group_members = collections.defaultdict(list)
    for i, row in enumerate(candidates):
        group_members[find(i)].append(row["sample_id"])
    for i, row in enumerate(candidates):
        row["family_id"] = "family:" + stable(min(group_members[find(i)]))[:16]
    # Retain one identical pair/comment instance; other sources remain in provenance rather than increasing sample count.
    dedup = {}
    for row in candidates:
        key = (row["upper"], row["lower"], row["target_comment"])
        if key in dedup:
            dedup[key].setdefault("additional_sources", []).append(
                {
                    "url": row["source_url"],
                    "revision": row["source_revision"],
                    "sample_id": row["sample_id"],
                }
            )
        else:
            dedup[key] = row
    candidates = list(dedup.values())
    jsonl(OUT / "candidates.jsonl", candidates)
    jsonl(OUT / "quarantine.jsonl", rejected)
    modern = [r for r in candidates if r["language_style"] == "modern"]
    classical = [r for r in candidates if r["language_style"] == "classical"]
    main_modern = [r for r in modern if r["modern_main_eligible"]]
    auxiliary_modern = [r for r in modern if not r["modern_main_eligible"]]
    jsonl(OUT / "modern_candidates.jsonl", modern)
    jsonl(OUT / "classical_candidates.jsonl", classical)
    jsonl(OUT / "modern_main_candidates.jsonl", main_modern)
    jsonl(OUT / "modern_auxiliary_candidates.jsonl", auxiliary_modern)
    # Audit priority puts all modern cases first, then classical cases with substantive comments and clear source spans.
    # Stratified audit: criticism and appreciation, one row per family; max two per article.
    priority = sorted(main_modern, key=lambda r: stable("audit-seed2026:" + r["sample_id"]))
    ordered = []
    seen = set()
    articles = collections.Counter()
    for wanted_criticism, limit in [(True, 20), (False, 50)]:
        for row in priority:
            if len(ordered) >= limit:
                break
            article = row.get("source_file", row["source_url"])
            if row["family_id"] in seen or articles[article] >= 2:
                continue
            if wanted_criticism and not row.get("criticizes_or_revises_signal"):
                continue
            ordered.append(row)
            seen.add(row["family_id"])
            articles[article] += 1
    audit = ordered[:50]
    jsonl(OUT / "review_50_source_records.jsonl", audit)
    fields = [
        "sample_id",
        "upper",
        "lower",
        "language_style",
        "context",
        "target_comment",
        "source_revision",
        "alignment_ok",
        "specificity_ok",
        "fluency_ok",
        "facts_checked",
        "decision",
        "reviewer",
        "notes",
    ]
    review_path = OUT / "review_50.csv"
    # Preserve previously filled human reviews on rebuild; generated blank sheets are disposable.
    if review_path.exists():
        existing = list(csv.DictReader(review_path.open(encoding="utf-8-sig", newline="")))
        if any(r.get("decision") or r.get("reviewer") for r in existing):
            review_path = OUT / "review_50_regenerated_blank.csv"
    with review_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(audit)
    instruction = "请用一段连贯的现代中文评论这副对联，结合具体字词说明句意与上下联的关系，只讨论有依据的特点或不足。"
    jsonl(
        OUT / "instruction_candidates_NOT_TRAIN_READY.jsonl",
        [
            {
                **r,
                "instruction": instruction
                if r["language_style"] == "modern"
                else "请依据给定背景，用文言评述这副对联；保持原评述的语体，不补造事实。",
                "input": {"upper": r["upper"], "lower": r["lower"], "context": r["context"]},
                "response": r["target_comment"],
            }
            for r in candidates
        ],
    )
    modern_families = len({r["family_id"] for r in modern})
    main_families = len({r["family_id"] for r in main_modern})
    gate = {
        "status": "data_review_required",
        "ready_for_modern_sft": False,
        "minimum_modern_independent_families": 300,
        "modern_candidate_rows": len(modern),
        "modern_candidate_families": modern_families,
        "modern_main_candidate_rows": len(main_modern),
        "modern_main_candidate_families": main_families,
        "modern_auxiliary_candidate_rows": len(auxiliary_modern),
        "modern_reviewed_approved_families": 0,
        "classical_candidate_rows": len(classical),
        "total_candidate_rows": len(candidates),
        "review_sheet_rows": len(audit),
        "quarantine_rows": len(rejected),
        "quarantine_reasons": dict(collections.Counter(r["reason"] for r in rejected)),
        "counts_before_dedup": dict(summary),
        "human_or_expert_review_completed": False,
        "modern_candidate_quantity_gate_met": main_families >= 300,
        "modern_source_usage_and_quality_gate_met": False,
        "user_selected_route": "continue_collecting_real_modern_commentary_then_train_after_gate",
        "reason": "Candidate quantity is separate from reviewed/usage-approved data. Publisher usage terms and record-level review are pending; classical originals are not relabeled modern.",
        "training_started": False,
        "generator_changed_by_this_task": False,
    }
    local_result = OUT / "curated_local_v1/TRAINING_COMPLETE.json"
    gate["gate_scope"] = "raw_candidate_pool_not_the_independently_curated_local_experiment"
    if local_result.exists():
        gate["training_started"] = True
        gate["local_experiment"] = {
            "status": read(local_result).get("status"),
            "protocol": "assistant_screened_local_experiment_authorized_by_later_user_instruction",
            "data_directory": "curated_local_v1",
            "quality_status": "see_curated_local_v1/evaluation_summary.json",
            "does_not_mean_raw_pool_or_expert_review_approved": True,
        }
    dump(OUT / "data_gate.json", gate)
    print(json.dumps(gate, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
