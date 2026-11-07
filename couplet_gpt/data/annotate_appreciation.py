"""Preserve full appreciation, add program prosody, and extract intact short paragraph candidates."""

import argparse, collections, gzip, hashlib, importlib.metadata, json, os, sqlite3, time
from pathlib import Path
import ijson
from couplet_gpt.resources.prosody_annotation import (
    ROOT,
    OUT,
    ToneAnnotator,
    han,
    text,
    family_id,
    short_candidates,
    render,
)
from collect_chinesepoetrylibrary_v3 import NAMES, REVISION, dump, hashfile

SURVEY = ROOT / "runs/poetry_appreciation_survey_20261106"
PROSE_TITLES = {
    "赤壁赋",
    "前赤壁赋",
    "后赤壁赋",
    "岳阳楼记",
    "出师表",
    "后出师表",
    "兰亭集序",
    "师说",
    "劝学",
    "桃花源记",
    "陋室铭",
    "爱莲说",
    "醉翁亭记",
    "小石潭记",
    "阿房宫赋",
    "六国论",
}


def write(f, r):
    f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")


def hint(shard, title, tags):
    if shard == "guwen.json" or title in PROSE_TITLES:
        return "prose"
    if shard == "songci.json":
        return "ci"
    if any(v in str(tags) for v in ["文言文", "散文"]):
        return "prose"
    return "unknown"


class Builder:
    def __init__(self):
        OUT.mkdir(parents=True, exist_ok=True)
        self.tone = ToneAnnotator()
        self.db = sqlite3.connect(OUT / "family_index.sqlite")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS families (family_id TEXT PRIMARY KEY, website_records INTEGER DEFAULT 0, ai_records INTEGER DEFAULT 0, couplet_records INTEGER DEFAULT 0)"
        )
        self.manifest_path = OUT / "processing_manifest.json"
        self.report = (
            json.loads(self.manifest_path.read_text(encoding="utf8"))
            if self.manifest_path.exists()
            else {"status": "building", "sources": {}, "training_started": False}
        )
        self.report["dependencies"] = {
            "ijson": importlib.metadata.version("ijson"),
            "opencc": importlib.metadata.version("OpenCC"),
        }
        self.report["full_text_not_truncated"] = True
        self.report["short_targets_are_source_paragraphs_not_summaries"] = True
        self.report["neural_loss_scope"] = "body_only_program_first_sentence_is_rendered_by_code"
        self.report["meter_claim_scope"] = (
            "dictionary-tone categories only, no context disambiguation or full-meter certificate"
        )

    def process(self, key, iterator, source_path, source_kind):
        if self.report["sources"].get(key, {}).get("status") == "complete":
            return
        folder = OUT / "processed"
        folder.mkdir(exist_ok=True)
        full_path = folder / (key + ".full_annotated.jsonl.gz")
        short_path = folder / (key + ".short_candidates.jsonl.gz")
        full_part = full_path.with_suffix(full_path.suffix + ".part")
        short_part = short_path.with_suffix(short_path.suffix + ".part")
        count = nonempty = shorts = body_chars = 0
        types = collections.Counter()
        lens = collections.Counter()
        issues = collections.Counter()
        examples = []
        # A file is added to the index in one transaction. On failure it is rolled back.
        self.db.execute("BEGIN")
        try:
            with (
                gzip.open(full_part, "wt", encoding="utf8", compresslevel=3) as full,
                gzip.open(short_part, "wt", encoding="utf8", compresslevel=3) as short,
            ):
                for r in iterator:
                    count += 1
                    if source_kind == "ai":
                        source = text(r.get("content"))
                        body = text(r.get("appreciation"))
                        title = str(r.get("title") or "")
                        base = {
                            "sample_id": f"ChinesePoetryLibrary:{key}:{count}",
                            "title": title,
                            "author": r.get("author"),
                            "dynasty": r.get("dynasty"),
                            "source_text": source,
                            "appreciation_body": body,
                            "source_name": "ChinesePoetryLibrary",
                            "upstream_poetry_id": r.get("poetry_id"),
                            "upstream_id": r.get("id"),
                            "source_file": key + ".json",
                            "source_row_1based": count,
                            "source_revision": REVISION,
                            "label_tier": "publisher_AI_enriched_collection_not_human_gold",
                            "interpretation_separate": r.get("interpret"),
                            "alternative_comment_separate": r.get("comment"),
                            "source_usage": "publisher MPL-2.0 declaration and README additional terms retained, not certified as unrestricted",
                        }
                        genre = hint(key + ".json", title, r.get("tags", []))
                        column = "ai_records"
                    elif source_kind == "website":
                        source = text(r.get("source_text"))
                        body = text(r.get("comment"))
                        title = str(r.get("title") or "")
                        base = {
                            "sample_id": r["sample_id"],
                            "title": title,
                            "author": r.get("author"),
                            "dynasty": r.get("dynasty"),
                            "source_text": source,
                            "appreciation_body": body,
                            "source_name": "Papersnake/gushiwen",
                            "source_revision": r.get("dataset_revision"),
                            "source_row_1based": r.get("source_row_1based"),
                            "raw_line_sha256": r.get("raw_line_sha256"),
                            "bibliography": r.get("bibliography"),
                            "label_tier": "website_collected_commentary_not_independently_expert_verified",
                        }
                        genre = hint("", title, [])
                        column = "website_records"
                    else:
                        source = r["upper_display"] + "；" + r["lower_display"]
                        body = text(r["response"])
                        title = r.get("title", "")
                        base = {
                            "sample_id": r["sample_id"],
                            "title": title,
                            "source_text": source,
                            "appreciation_body": body,
                            "source_name": "curated_couplet_v2",
                            "source_url": r.get("source_url"),
                            "source_revision": r.get("source_revision"),
                            "label_tier": r.get("quality_tier"),
                            "upper": r["upper"],
                            "lower": r["lower"],
                        }
                        genre = "couplet"
                        column = "couplet_records"
                    if not source or not body or not han(source):
                        issues["missing_original_or_appreciation"] += 1
                        continue
                    nonempty += 1
                    body_chars += len(body)
                    base["family_id"] = family_id(source)
                    if genre == "couplet":
                        annotation = self.tone.annotate(source, upper=r["upper"], lower=r["lower"])
                        base["modern_tone_reference"] = self.tone.annotate(
                            source,
                            mode="modern_character_dictionary",
                            upper=r["upper"],
                            lower=r["lower"],
                        )
                    else:
                        annotation = self.tone.annotate(source, genre_hint=genre)
                    base["prosody"] = annotation
                    base["program_first_sentence"] = annotation["program_first_sentence"]
                    base["review_status"] = "derived_candidate_not_expert_approved"
                    base["training_started"] = False
                    write(full, base)
                    types[annotation["form_scope"]] += 1
                    if annotation["source_glyph_issue"]:
                        issues["unresolved_private_glyphs"] += 1
                    lens[
                        "under120"
                        if len(body) < 120
                        else "120to220"
                        if len(body) <= 220
                        else "221to400"
                        if len(body) <= 400
                        else "over400"
                    ] += 1
                    choices = short_candidates(source, body, annotation)
                    for j, candidate in enumerate(choices):
                        item = {
                            "sample_id": base["sample_id"] + f":paragraph{j + 1}",
                            "parent_sample_id": base["sample_id"],
                            "family_id": base["family_id"],
                            "source_name": base["source_name"],
                            "label_tier": base["label_tier"],
                            "source_text": source,
                            "title": title,
                            **candidate,
                        }
                        write(short, item)
                        shorts += 1
                        if len(examples) < 12:
                            examples.append(item)
                    self.db.execute(
                        f"INSERT INTO families(family_id,{column}) VALUES (?,1) ON CONFLICT(family_id) DO UPDATE SET {column}={column}+1",
                        (base["family_id"],),
                    )
                    if count % 10000 == 0:
                        print(
                            "ANNOTATE",
                            key,
                            count,
                            "comments",
                            nonempty,
                            "shorts",
                            shorts,
                            flush=True,
                        )
                self.db.commit()
        except Exception:
            self.db.rollback()
            raise
        os.replace(full_part, full_path)
        os.replace(short_part, short_path)
        stat = {
            "status": "complete",
            "source_records_read": count,
            "records_with_appreciation": nonempty,
            "full_body_characters": body_chars,
            "short_paragraph_candidates": shorts,
            "form_scopes": dict(types),
            "full_body_lengths": dict(lens),
            "issues": dict(issues),
            "source_kind": source_kind,
            "input_file": str(source_path.relative_to(ROOT)),
            "input_sha256": hashfile(source_path),
            "full_output": str(full_path.relative_to(ROOT)),
            "full_output_bytes": full_path.stat().st_size,
            "full_sha256": hashfile(full_path),
            "short_output": str(short_path.relative_to(ROOT)),
            "short_output_bytes": short_path.stat().st_size,
            "short_sha256": hashfile(short_path),
        }
        self.report["sources"][key] = stat
        dump(self.manifest_path, self.report)
        dump(OUT / (key + ".examples.json"), examples)
        print("ANNOTATED_COMPLETE", key, count, nonempty, shorts, flush=True)

    def finish(self, complete):
        self.report["status"] = (
            "complete_candidate_preparation_not_training_ready"
            if complete
            else "partial_candidate_preparation"
        )
        self.report["distinct_exact_han_families"] = self.db.execute(
            "SELECT COUNT(*) FROM families"
        ).fetchone()[0]
        self.report["families_shared_website_AI"] = self.db.execute(
            "SELECT COUNT(*) FROM families WHERE website_records>0 AND ai_records>0"
        ).fetchone()[0]
        self.report["totals"] = {
            k: sum(v.get(k, 0) for v in self.report["sources"].values())
            for k in [
                "source_records_read",
                "records_with_appreciation",
                "full_body_characters",
                "short_paragraph_candidates",
            ]
        }
        self.report["completion_date"] = "2026-11-06"
        dump(self.manifest_path, self.report)
        self.db.close()
        print(
            "PREPARATION_FINISHED",
            json.dumps(
                {"status": self.report["status"], "totals": self.report["totals"]},
                ensure_ascii=False,
            ),
            flush=True,
        )


def jsonl(path):
    with path.open(encoding="utf8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch-library", action="store_true")
    args = parser.parse_args()
    b = Builder()
    primary = SURVEY / "hf_full_inspection/appreciation_candidates_NOT_TRAIN_READY.jsonl"
    b.process("gushiwen", jsonl(primary), primary, "website")
    couplets = ROOT / "runs/comment_sft_v2/curated_local_v1/accepted_all.jsonl"
    b.process("couplets", jsonl(couplets), couplets, "couplet")
    raw_dir = OUT / "sources/chinesepoetrylibrary/raw"
    manifest = raw_dir.parent / "collection_manifest.json"
    while True:
        for name in sorted(
            NAMES, key=lambda x: (raw_dir / x).stat().st_size if (raw_dir / x).exists() else 10**20
        ):
            path = raw_dir / name
            key = path.stem
            if path.exists() and b.report["sources"].get(key, {}).get("status") != "complete":
                with path.open("rb") as f:
                    b.process(key, ijson.items(f, "item", use_float=True), path, "ai")
        complete = all(n[:-5] in b.report["sources"] for n in NAMES)
        state = json.loads(manifest.read_text(encoding="utf8")) if manifest.exists() else {}
        if complete or not args.watch_library or state.get("status") == "partial":
            break
        time.sleep(3)
    b.finish(complete)


if __name__ == "__main__":
    main()
