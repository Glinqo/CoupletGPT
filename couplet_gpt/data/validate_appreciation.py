"""Validate derived files against source text and make compact transfer candidates.

These files are automatically screened candidates, never expert approved SFT gold.
Run after prepare_poetry_appreciation_v3.py has finished.
"""

import argparse, collections, gzip, json, os, re
from pathlib import Path
import ijson
from couplet_gpt.resources.prosody_annotation import ROOT, OUT, text, han, render
from collect_chinesepoetrylibrary_v3 import dump, hashfile


def rows(path):
    with (
        gzip.open(path, "rt", encoding="utf8")
        if str(path).endswith(".gz")
        else path.open(encoding="utf8")
    ) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def originals(path, kind):
    with path.open("rb") as f:
        stream = (
            ijson.items(f, "item", use_float=True)
            if kind == "ai"
            else (json.loads(s) for s in f if s.strip())
        )
        for r in stream:
            if kind == "ai":
                source, body = text(r.get("content")), text(r.get("appreciation"))
            elif kind == "website":
                source, body = text(r.get("source_text")), text(r.get("comment"))
            else:
                source, body = r["upper_display"] + "；" + r["lower_display"], text(r["response"])
            if source and body and han(source):
                yield source, body


def quantile(hist, q):
    target = max(1, int(sum(hist.values()) * q + 0.999999))
    total = 0
    for length, count in sorted(hist.items()):
        total += count
        if total >= target:
            return length
    return None


def style_flags(body):
    flags = []
    if sum(k in body for k in ["文学手法", "意境营造", "情感表达", "主题思想"]) >= 3:
        flags.append("recurrent_four_facet_formula_needs_style_review")
    if re.search("可能|或许|推测|大概", body):
        flags.append("hedged_interpretation_requires_context_review_not_automatically_wrong")
    return flags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--available", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((OUT / "processing_manifest.json").read_text(encoding="utf8"))
    complete = manifest["status"] == "complete_candidate_preparation_not_training_ready"
    assert complete or args.available, (
        "Wait for all shards to finish, or explicitly audit --available completed files"
    )
    report = {
        "status": "auditing",
        "audit_schema_version": 2,
        "training_started": False,
        "sources": {},
        "purpose": "compact literary appreciation transfer candidates; not couplet-specific gold",
        "source_comparison": "every retained full source/body compared to normalized original input",
        "full_text_normalization": "NFC, HTML unescape/tag removal, outer whitespace strip; raw inputs retained",
        "short_policy": "intact original paragraph, 120..220 characters; no source parenthetical notes; max 3 per original section",
        "unreviewed_AI_and_technical_claims": "retained as explicitly flagged candidates, not certified",
        "family_policy": "same normalized Han work must stay in one split; exact hash is not a near-duplicate guarantee",
    }
    saved = OUT / "derived_validation.json"
    if saved.exists():
        report["sources"] = json.loads(saved.read_text(encoding="utf8")).get("sources", {})
    for key, stat in manifest["sources"].items():
        prior = report["sources"].get(key, {})
        if (
            prior.get("audit_schema_version") == 2
            and prior.get("validated_full_sha256") == stat["full_sha256"]
            and prior.get("validated_input_sha256") == stat["input_sha256"]
            and (ROOT / prior.get("output", "missing")).is_file()
        ):
            print("ALREADY_VALIDATED", key, flush=True)
            continue
        full_path = ROOT / stat["full_output"]
        short_path = ROOT / stat["short_output"]
        full_hist = collections.Counter()
        short_hist = collections.Counter()
        display_hist = collections.Counter()
        prefix_hist = collections.Counter()
        issues = collections.Counter()
        scope = collections.Counter()
        fulls = shorts = selected = classified = annotated = unknown = 0
        shorts_iter = iter(rows(short_path))
        pending = next(shorts_iter, None)
        original_iter = iter(originals(ROOT / stat["input_file"], stat["source_kind"]))
        out_path = OUT / "processed" / (key + ".short_120_220_screened.jsonl.gz")
        part = out_path.with_suffix(".gz.part")
        examples = []
        with gzip.open(part, "wt", encoding="utf8", compresslevel=3) as out:
            for r in rows(full_path):
                fulls += 1
                source, body = next(original_iter)
                assert r["source_text"] == source and r["appreciation_body"] == body, (
                    key,
                    fulls,
                    "original text changed",
                )
                first = r["program_first_sentence"]
                a = r["prosody"]
                assert (
                    first == a["program_first_sentence"]
                    and first.endswith("。")
                    and first.count("。") == 1
                )
                assert (
                    not a["full_meter_verified"]
                    and not a["context_pronunciation_disambiguation_verified"]
                )
                if stat["source_kind"] == "ai":
                    assert r["label_tier"] == "publisher_AI_enriched_collection_not_human_gold"
                assert len(a["patterns"]) in [0, len(a["line_texts"])]
                for s, p in zip(a["line_texts"], a["patterns"]):
                    assert len(s) == len(p)
                full_hist[len(body)] += 1
                prefix_hist[len(first)] += 1
                scope[a["form_scope"]] += 1
                for flag in style_flags(body):
                    issues["full_" + flag] += 1
                classified += a["classified_characters"]
                annotated += a["total_annotated_characters"]
                unknown += len(a["ambiguous_or_uncovered"])
                note = bool(re.search(r"[（(【\[]", source))
                if note:
                    issues["source_parenthetical_or_editorial_note_needs_review"] += 1
                while pending and pending["parent_sample_id"] == r["sample_id"]:
                    c = pending
                    shorts += 1
                    lo, hi = c["comment_body_span"]
                    target = c["response_body"]
                    assert body[lo:hi] == target, (key, shorts, "paragraph span mismatch")
                    assert c["display_response"] == render(a, target)
                    assert c["program_first_sentence"] == first and c["family_id"] == r["family_id"]
                    assert 60 <= len(target) <= 220 and len(c["prompt"]) + len(target) + 2 <= 768
                    short_hist[len(target)] += 1
                    display_hist[len(c["display_response"])] += 1
                    if len(target) < 120:
                        issues["short_under120_auxiliary_only"] += 1
                    elif note:
                        issues["short_with_source_notes_excluded_from_compact"] += 1
                    else:
                        c["target_role"] = "poetry_transfer_candidate_not_couplet_gold"
                        c["content_review_required"] = True
                        c["style_review_flags"] = style_flags(target)
                        for flag in c["style_review_flags"]:
                            issues["compact_" + flag] += 1
                        c["program_prefix_rendered_by_code"] = True
                        c["training_started"] = False
                        out.write(json.dumps(c, ensure_ascii=False, separators=(",", ":")) + "\n")
                        selected += 1
                        if len(examples) < 3:
                            examples.append(c)
                    pending = next(shorts_iter, None)
        assert pending is None, (key, "unmatched short rows")
        assert next(original_iter, None) is None, (key, "source rows omitted")
        assert (
            fulls == stat["records_with_appreciation"]
            and shorts == stat["short_paragraph_candidates"]
        )
        os.replace(part, out_path)
        report["sources"][key] = {
            "audit_schema_version": 2,
            "full_records_validated": fulls,
            "short_records_validated": shorts,
            "compact_transfer_candidates": selected,
            "full_body_median_chars": quantile(full_hist, 0.5),
            "full_body_p90_chars": quantile(full_hist, 0.9),
            "short_body_median_chars": quantile(short_hist, 0.5),
            "short_display_median_chars": quantile(display_hist, 0.5),
            "short_display_p90_chars": quantile(display_hist, 0.9),
            "prefix_median_chars": quantile(prefix_hist, 0.5),
            "tone_characters_classified": classified,
            "tone_characters_annotated": annotated,
            "ambiguous_or_uncovered_characters": unknown,
            "issues": dict(issues),
            "form_scopes": dict(scope),
            "output": str(out_path.relative_to(ROOT)),
            "validated_full_sha256": stat["full_sha256"],
            "validated_input_sha256": stat["input_sha256"],
            "output_bytes": out_path.stat().st_size,
            "output_sha256": hashfile(out_path),
        }
        dump(OUT / (key + ".compact_examples.json"), examples)
        dump(OUT / "derived_validation.json", report)
        print("VALIDATED", key, fulls, shorts, "compact", selected, flush=True)
    report["status"] = (
        "passed_text_integrity_and_mechanical_checks_not_content_approval"
        if complete
        else "partial_available_files_validated"
    )
    keys = [
        "full_records_validated",
        "short_records_validated",
        "compact_transfer_candidates",
        "tone_characters_classified",
        "tone_characters_annotated",
        "ambiguous_or_uncovered_characters",
    ]
    report["totals"] = {k: sum(s[k] for s in report["sources"].values()) for k in keys}
    dump(OUT / "derived_validation.json", report)
    print("VALIDATION_FINISHED", json.dumps(report["totals"]), flush=True)


if __name__ == "__main__":
    main()
