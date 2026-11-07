"""Measure full-text appreciation examples, without loading their metadata into RAM."""

import collections, json, re, time
from couplet_gpt.training.appreciation_common import ROOT, DATA, rows, dump

OUT = ROOT / "runs/poetry_direct_sft_v4"


def prompt(source, first):
    return (
        "任务：古诗文赏析。\n原文："
        + source
        + "\n字音依据："
        + first
        + "\n请结合原文具体字词，写一段连贯赏析，说明内容、情感与表达手法，不编造出处。\n赏析正文："
    )


def first_sentence(r):
    if re.search(r"[（(【\[]", r["source_text"]):
        return "原文含括注或编辑标记，需先区分正文与说明，再核验平仄。"
    return r["program_first_sentence"]


def main():
    report = {"status": "surveying", "sources": {}}
    start = time.perf_counter()
    manifest = json.loads((DATA / "processing_manifest.json").read_text(encoding="utf8"))
    for key, s in manifest["sources"].items():
        if s["source_kind"] == "couplet":
            continue
        n = 0
        fits = collections.Counter()
        chars = collections.Counter()
        issues = collections.Counter()
        for r in rows(ROOT / s["full_output"]):
            n += 1
            source = r["source_text"]
            body = r["appreciation_body"]
            first = first_sentence(r)
            if r["prosody"]["source_glyph_issue"] or any(
                "\ue000" <= c <= "\uf8ff" or c == "\ufffd" for c in body
            ):
                issues["unresolved_glyph"] += 1
                continue
            length = len(prompt(source, first)) + len(body) + 2
            for limit in [768, 1024, 1536, 2048]:
                if length <= limit:
                    fits[limit] += 1
                    chars[limit] += len(body)
        report["sources"][key] = {
            "full_rows": n,
            "fits_context": dict(fits),
            "target_characters_by_context": dict(chars),
            "issues": dict(issues),
        }
        dump(OUT / "length_survey.json", report)
        print("SURVEY", key, n, dict(fits), flush=True)
    report["status"] = "complete"
    report["seconds"] = time.perf_counter() - start
    report["totals"] = {
        str(l): {
            "rows": sum(s["fits_context"].get(l, 0) for s in report["sources"].values()),
            "target_characters": sum(
                s["target_characters_by_context"].get(l, 0) for s in report["sources"].values()
            ),
        }
        for l in [768, 1024, 1536, 2048]
    }
    dump(OUT / "length_survey.json", report)
    print(json.dumps(report["totals"]), flush=True)


if __name__ == "__main__":
    main()
