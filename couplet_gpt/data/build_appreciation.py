"""Prepare a large full-appreciation SFT corpus, with no curated-couplet training."""

import array, collections, json, re, sqlite3, time
from opencc import OpenCC
from couplet_gpt.training.appreciation_common import ROOT, DATA, rows, dump, stable, sha
from couplet_gpt.data.appreciation_prompts import OUT, prompt, first_sentence
from couplet_gpt.resources.prosody_annotation import han
import numpy as np

LIMIT = 1536
BAD = re.compile("[\ue000-\uf8ff\U000f0000-\U000ffffd\U00100000-\U0010fffd\ufffd]")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "DATA_READY.json").exists():
        print("ALREADY_PREPARED")
        return
    db = sqlite3.connect(OUT / "records.sqlite")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA cache_size=-65536")
    assert not db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall(), (
        "Inspect incomplete preparation before retrying"
    )
    db.execute(
        "CREATE TABLE records (id INTEGER PRIMARY KEY, family TEXT, target_hash TEXT, source_name TEXT, source_key TEXT, sample_id TEXT, payload TEXT, split TEXT, component TEXT)"
    )
    db.execute("CREATE UNIQUE INDEX work_target ON records(family,target_hash)")
    cc = OpenCC("t2s")
    parent = {}
    by_target = {}
    forced = collections.defaultdict(set)
    stats = collections.Counter()
    sourcecounts = collections.Counter()
    start = time.perf_counter()

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    previous = ROOT / "runs/appreciation_sft_v3"
    for r in rows(previous / "index.jsonl"):
        union(r["family"], r["family"])
        forced[r["family"]].add(r["split"])
    manifest = json.loads((DATA / "processing_manifest.json").read_text(encoding="utf8"))
    ledger = (OUT / "excluded.jsonl").open("w", encoding="utf8")
    db.execute("BEGIN")
    for key, s in manifest["sources"].items():
        if s["source_kind"] == "couplet":
            continue
        for r in rows(ROOT / s["full_output"]):
            stats["full_source_rows_read"] += 1
            source = r["source_text"]
            body = r["appreciation_body"]
            first = first_sentence(r)
            reason = (
                "unresolved_glyph"
                if r["prosody"]["source_glyph_issue"] or BAD.search(body)
                else "over_context_no_truncation"
                if len(prompt(source, first)) + len(body) + 2 > LIMIT
                else None
            )
            if reason:
                stats[reason] += 1
                ledger.write(
                    json.dumps({"sample_id": r["sample_id"], "reason": reason}, ensure_ascii=False)
                    + "\n"
                )
                continue
            key_family = stable(han(cc.convert(source)))
            find(key_family)
            target_hash = stable(body)
            payload = {
                "sample_id": r["sample_id"],
                "source_text": source,
                "response_body": body,
                "program_first_sentence": first,
                "source_name": r["source_name"],
                "label_tier": r["label_tier"],
                "title": r.get("title"),
                "kind": "poetry",
                "family": key_family,
                "source_key": key,
                "upstream_row": r.get("source_row_1based"),
                "source_revision": r.get("source_revision"),
                "full_appreciation_preserved": True,
            }
            cur = db.execute(
                "INSERT OR IGNORE INTO records(family,target_hash,source_name,source_key,sample_id,payload) VALUES (?,?,?,?,?,?)",
                (
                    key_family,
                    target_hash,
                    r["source_name"],
                    key,
                    r["sample_id"],
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                ),
            )
            if not cur.rowcount:
                stats["same_work_same_comment_duplicates"] += 1
                continue
            if target_hash in by_target:
                union(key_family, by_target[target_hash])
            else:
                by_target[target_hash] = key_family
            stats["retained_full_rows"] += 1
            sourcecounts[key] += 1
        print(
            "FULL_PREP",
            key,
            "retained",
            stats["retained_full_rows"],
            "seconds",
            round(time.perf_counter() - start),
            flush=True,
        )
    db.commit()
    ledger.close()
    components = collections.defaultdict(set)
    for k, v in forced.items():
        components[find(k)].update(v)
    conflicts = {k: sorted(v) for k, v in components.items() if len(v) > 1}
    counts = collections.Counter()
    chars = set()
    splitfamilies = collections.defaultdict(set)
    updates = []
    for rowid, family, payload in db.execute("SELECT id,family,payload FROM records ORDER BY id"):
        root = find(family)
        fs = components.get(root, set())
        if fs:
            split = "test" if "test" in fs else "validation" if "validation" in fs else "train"
        else:
            bucket = int(stable("direct_full20261106:" + root)[:8], 16) % 100
            split = "validation" if bucket == 0 else "test" if bucket == 1 else "train"
        updates.append((split, root, rowid))
        counts[split] += 1
        splitfamilies[split].add(root)
        if split == "train":
            r = json.loads(payload)
            chars.update(prompt(r["source_text"], r["program_first_sentence"]) + r["response_body"])
        if len(updates) >= 5000:
            db.executemany("UPDATE records SET split=?,component=? WHERE id=?", updates)
            updates = []
    if updates:
        db.executemany("UPDATE records SET split=?,component=? WHERE id=?", updates)
    db.commit()
    assert not (
        splitfamilies["train"] & splitfamilies["test"]
        or splitfamilies["train"] & splitfamilies["validation"]
        or splitfamilies["test"] & splitfamilies["validation"]
    )
    vocab = json.loads((ROOT / "runs/pretrain_v1/artifacts/vocab.json").read_text(encoding="utf8"))
    base_size = len(vocab)
    added = sorted(chars - set(vocab))
    for c in added:
        vocab[c] = len(vocab)
    assert len(vocab) < 65536
    dump(OUT / "vocab.json", vocab)
    offsets = array.array("Q", [0])
    prefixes = array.array("H")
    lengths = array.array("H")
    response_lengths = array.array("H")
    splitcodes = array.array("B")
    token_counts = collections.Counter()
    unknown = collections.Counter()
    eval_candidates = {"validation": [], "test": []}
    index = (OUT / "index.jsonl").open("w", encoding="utf8")
    with (OUT / "tokens.bin").open("wb") as f:
        for i, (rowid, split, component, payload) in enumerate(
            db.execute("SELECT id,split,component,payload FROM records ORDER BY id")
        ):
            r = json.loads(payload)
            p = [2] + [
                vocab.get(c, 1) for c in prompt(r["source_text"], r["program_first_sentence"])
            ]
            target = [vocab.get(c, 1) for c in r["response_body"]] + [3]
            seq = p + target
            array.array("H", seq).tofile(f)
            offsets.append(offsets[-1] + len(seq))
            prefixes.append(len(p))
            lengths.append(len(seq) - 1)
            response_lengths.append(len(target))
            splitcodes.append({"train": 0, "validation": 1, "test": 2}[split])
            token_counts[split] += len(target)
            unknown[split] += seq.count(1)
            index.write(
                json.dumps(
                    {
                        "row": i,
                        "db_id": rowid,
                        "sample_id": r["sample_id"],
                        "split": split,
                        "component": component,
                        "family": r["family"],
                        "source_key": r["source_key"],
                        "length": len(seq) - 1,
                        "response_tokens": len(target),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            if split in eval_candidates:
                eval_candidates[split].append((stable(r["sample_id"]), i, rowid))
    index.close()
    for name, a in [
        ("offsets", offsets),
        ("prefix_lengths", prefixes),
        ("lengths", lengths),
        ("response_lengths", response_lengths),
        ("split_codes", splitcodes),
    ]:
        np.save(OUT / (name + ".npy"), np.asarray(a))
    locked = {}
    for split, entries in eval_candidates.items():
        selected = sorted(entries)[:1024]
        locked[split] = [i for _, i, _ in selected]
        with (OUT / (split + "_locked.jsonl")).open("w", encoding="utf8") as f:
            for _, i, rowid in selected:
                r = json.loads(
                    db.execute("SELECT payload FROM records WHERE id=?", (rowid,)).fetchone()[0]
                )
                r.update({"row": i, "split": split})
                f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
    dump(OUT / "evaluation_locked.json", locked)
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    db.execute("PRAGMA journal_mode=DELETE")
    db.close()
    files = [
        "vocab.json",
        "tokens.bin",
        "offsets.npy",
        "prefix_lengths.npy",
        "lengths.npy",
        "response_lengths.npy",
        "split_codes.npy",
        "index.jsonl",
        "evaluation_locked.json",
        "validation_locked.jsonl",
        "test_locked.jsonl",
        "records.sqlite",
    ]
    ready = {
        "status": "ready_for_direct_full_poetry_sft",
        "date": "2026-11-06",
        "context_length": LIMIT,
        "counts": dict(stats),
        "source_counts": dict(sourcecounts),
        "split_counts": dict(counts),
        "target_tokens_per_pass_by_split": dict(token_counts),
        "unknown_tokens": dict(unknown),
        "vocab_size": len(vocab),
        "base_vocab_size": base_size,
        "new_training_characters": added,
        "training_sources": "ChinesePoetryLibrary and Papersnake/gushiwen full appreciation only; curated couplet corpus excluded",
        "AI_repeated_frameworks_retained": True,
        "full_source_and_appreciation_not_truncated": True,
        "family_and_identical_comment_components_disjoint": True,
        "preserve_previous_v3_heldout_families": True,
        "component_conflicts_conservatively_promoted_to_heldout": conflicts,
        "vocabulary_from_train_only": True,
        "historical_pretraining_work_exposure_not_ruled_out": True,
        "source_preparation_sha256": sha(DATA / "BUILD_RECORD.json"),
        "files_sha256": {n: sha(OUT / n) for n in files},
        "seconds": time.perf_counter() - start,
    }
    dump(OUT / "DATA_READY.json", ready)
    print(
        "FULL_DATA_READY",
        json.dumps(
            {
                k: v
                for k, v in ready.items()
                if k
                not in [
                    "new_training_characters",
                    "files_sha256",
                    "component_conflicts_conservatively_promoted_to_heldout",
                ]
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
