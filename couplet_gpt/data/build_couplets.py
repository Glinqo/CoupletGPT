"""Audited multi-source corpus; preserve whole couplets and isolate derived clauses."""

from couplet_gpt.paths import ROOT as PROJECT_ROOT
import collections, csv, functools, gzip, hashlib, itertools, json, re, sqlite3, tarfile, unicodedata
from pathlib import Path
from opencc import OpenCC

ROOT = PROJECT_ROOT
RAW = ROOT / "data/raw/couplet_v8"
OUT = ROOT / "data/processed/couplet_v8"
CC = OpenCC("t2s")
SPLITS = ("train", "validation", "test")
RANK = {"train": 0, "validation": 1, "test": 2}
VOC = json.loads((ROOT / "tokenizer/vocab.json").read_text(encoding="utf8"))


def digest(s):
    return hashlib.sha256(s.encode()).hexdigest()


def dump(path, x):
    path.write_text(json.dumps(x, ensure_ascii=False, indent=2), encoding="utf8")


def line(f, r):
    f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")


def rows(path):
    with path.open(encoding="utf8") as f:
        for s in f:
            if s.strip():
                yield json.loads(s)


def han(c):
    n = ord(c)
    return (
        0x3400 <= n <= 0x4DBF
        or 0x4E00 <= n <= 0x9FFF
        or 0xF900 <= n <= 0xFAFF
        or 0x20000 <= n <= 0x2EE5F
        or 0x30000 <= n <= 0x3347F
    )


@functools.lru_cache(maxsize=2000000)
def convert(s):
    return CC.convert(s)


def norm(s):
    s = "".join(unicodedata.normalize("NFC", str(s)).split())
    s = convert(s).translate(
        str.maketrans({",": "，", ";": "；", ":": "：", ".": "。", "!": "！", "?": "？"})
    )
    return s.strip("，；。！？：、")


def parse(a, b):
    a, b = norm(a), norm(b)
    if not a or not b:
        return None, "empty"
    if not all(han(c) or c in "，；" for c in a + b):
        return None, "unsupported_character_or_internal_punctuation"
    aa = re.split("[，；]", a)
    bb = re.split("[，；]", b)
    if not all(aa + bb):
        return None, "empty_clause"
    if len(aa) != len(bb):
        return None, "clause_count_mismatch"
    if [len(x) for x in aa] != [len(x) for x in bb]:
        return None, "clause_length_mismatch"
    if re.findall("[，；]", a) != re.findall("[，；]", b):
        return None, "separator_structure_mismatch"
    if a == b:
        return None, "identical_lines"
    return (a, b, aa, bb), None


def paired(a, b):
    for i, p in enumerate(itertools.zip_longest(a, b), 1):
        if None in p:
            raise ValueError("Unaligned input/output file row counts")
        yield i, *p


def sources():
    with tarfile.open(ROOT / "data/raw/couplet.tar.gz") as t:
        names = t.getnames()
        for s in ("test", "train"):
            a = (
                t.extractfile(next(n for n in names if n.endswith(f"/{s}/in.txt")))
                .read()
                .decode("utf8")
                .splitlines()
            )
            b = (
                t.extractfile(next(n for n in names if n.endswith(f"/{s}/out.txt")))
                .read()
                .decode("utf8")
                .splitlines()
            )
            for i, x, y in paired(a, b):
                yield "original", s, i, x, y, "couplet.tar.gz::" + s, True
    for s in ("test", "train"):
        with (
            (RAW / f"clean/couplets/{s}/in.txt").open(encoding="utf8") as a,
            (RAW / f"clean/couplets/{s}/out.txt").open(encoding="utf8") as b,
        ):
            for i, x, y in paired(a, b):
                yield "clean", s, i, x, y, f"clean/couplets/{s}", True
    for s in ("test", "train"):
        p = RAW / f"vv/data_set/corpus_{s}_set.txt"
        with p.open(encoding="utf8") as f:
            for i, z in enumerate(f, 1):
                parts = z.rstrip("\r\n").split("==")
                x, y = parts if len(parts) == 2 else ("", "")
                yield "vv", s, i, x, y, str(p.relative_to(RAW)), True
    with (RAW / "anchi.response").open(encoding="utf8") as f:
        for i, z in enumerate(f, 1):
            parts = z.rstrip("\r\n").split("\t")
            x, y = parts if len(parts) == 2 else ("", "")
            yield "anchi", "unspecified", i, x, y, "anchi.response", False
    with gzip.open(RAW / "hf_original/data.csv.gz", "rt", encoding="utf8") as f:
        for i, z in enumerate(csv.DictReader(f), 1):
            yield (
                "hf_original",
                "unspecified",
                i,
                z["up"],
                z["down"],
                "hf_original/data.csv.gz",
                True,
            )
    for s in ("test", "train"):
        p = RAW / f"hf_json/{s}/{s}.json"
        data = json.loads(p.read_text(encoding="utf8"))
        for i, z in enumerate(data, 1):
            x = z["source_text"]
            x = x.split("：", 1)[-1].split("\n", 1)[0]
            yield "hf_json", s, i, x, z["target_text"], str(p.relative_to(RAW)), False
        del data
    import pyarrow.parquet as pq

    for p in sorted((RAW / "hf_100k/data").glob("*.parquet")):
        s = "validation" if p.name.startswith("validation") else "train"
        i = 0
        for batch in pq.ParquetFile(p).iter_batches(batch_size=8192):
            for z in batch.to_pylist():
                i += 1
                yield "hf_100k", s, i, z["input"], z["output"], str(p.relative_to(RAW)), False


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "COMPLETE.json").exists():
        print("Already complete; verify_couplet_v8.py can recheck")
        return
    assert not (OUT / "registry.sqlite3").exists(), (
        "Partial build exists: inspect before rebuilding"
    )
    db = sqlite3.connect(OUT / "registry.sqlite3")
    db.executescript(
        "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; CREATE TABLE pairs(pid TEXT PRIMARY KEY,u TEXT,l TEXT,uk TEXT,aa TEXT,bb TEXT,eligible INTEGER,first_source TEXT); CREATE TABLE refs(pid TEXT,source TEXT,split TEXT,row INTEGER,raw_ref TEXT);"
    )
    counts = collections.defaultdict(collections.Counter)
    audit = (OUT / "filter_log.jsonl").open("w", encoding="utf8")
    last = None
    for src, s, i, a, b, ref, eligible in sources():
        if last != src:
            if last:
                db.commit()
                print("SOURCE COMPLETE", last, dict(counts[last]), flush=True)
            last = src
        c = counts[src]
        c["raw_rows"] += 1
        parsed, reason = parse(a, b)
        if reason:
            c["rejected_" + reason] += 1
            line(audit, {"source": src, "split": s, "row": i, "raw_ref": ref, "reason": reason})
            continue
        u, l, aa, bb = parsed
        uk = "".join(aa)
        pid = digest(uk + "\t" + "".join(bb))
        c["valid_rows"] += 1
        existed = db.execute("SELECT first_source FROM pairs WHERE pid=?", (pid,)).fetchone()
        if existed:
            c["overlap_prior_sources_or_repetition"] += 1
        else:
            c["unique_first_seen"] += 1
            db.execute(
                "INSERT INTO pairs VALUES(?,?,?,?,?,?,?,?)",
                (
                    pid,
                    u,
                    l,
                    uk,
                    json.dumps(aa, ensure_ascii=False),
                    json.dumps(bb, ensure_ascii=False),
                    int(eligible),
                    src,
                ),
            )
        if eligible and existed:
            db.execute("UPDATE pairs SET eligible=1 WHERE pid=? AND eligible=0", (pid,))
        db.execute("INSERT INTO refs VALUES(?,?,?,?,?)", (pid, src, s, i, ref))
        if c["raw_rows"] % 100000 == 0:
            db.commit()
            print("INGEST", src, c["raw_rows"], flush=True)
    audit.close()
    db.commit()
    print("SOURCE COMPLETE", last, dict(counts[last]), flush=True)
    db.executescript(
        "CREATE INDEX refs_pid ON refs(pid); CREATE INDEX pairs_uk ON pairs(uk); CREATE TABLE exports(pid TEXT PRIMARY KEY,split TEXT,reason TEXT); CREATE TABLE singles(sid TEXT PRIMARY KEY,u TEXT,l TEXT,split TEXT); CREATE TABLE derivations(sid TEXT,pid TEXT,clause_index INTEGER); CREATE INDEX derivations_pid ON derivations(pid);"
    )
    dump(OUT / "source_counts.json", counts)
    # Persist source lines independently of exports; mirrors do not multiply sample weight.
    with (OUT / "provenance.jsonl").open("w", encoding="utf8") as f:
        for pid, src, s, i, ref in db.execute("SELECT * FROM refs ORDER BY rowid"):
            line(
                f,
                {
                    "parent_id": pid,
                    "source": src,
                    "source_split": s,
                    "source_row": i,
                    "raw_ref": ref,
                },
            )
    convert.cache_clear()
    old = {}
    old_uppers = {s: set() for s in SPLITS}
    for s in SPLITS:
        for r in rows(ROOT / f"runs/final_couplet_v7/{s}.jsonl"):
            for upper in (r["upper"], norm(r["upper"])):
                old[upper] = max(old.get(upper, -1), RANK[s])
                old_uppers[s].add(upper)
    official_test = {
        pid
        for (pid,) in db.execute(
            "SELECT DISTINCT pid FROM refs WHERE split='test' AND source IN ('original','clean','vv','hf_json')"
        )
    }
    # Full upper is grouped without punctuation; alternate lower references stay together.
    group_rank = {}
    proposals = {}
    quarantine = (OUT / "quarantine.jsonl").open("w", encoding="utf8")
    for pid, u, l, uk, aaj, bbj, eligible, src in db.execute("SELECT * FROM pairs ORDER BY pid"):
        if not eligible:
            line(
                quarantine,
                {
                    "id": pid,
                    "upper": u,
                    "lower": l,
                    "reason": "unique_to_sources_without_clear_provenance_or_data_license",
                },
            )
            continue
        aa = json.loads(aaj)
        bb = json.loads(bbj)
        # Only usable auxiliary clauses participate in the cross-view isolation key.
        use = [
            (j, a, b)
            for j, (a, b) in enumerate(zip(aa, bb))
            if a != b
            and (len(aa) == 1 or len(a) >= 4)
            and len(a) <= 62
            and all(c in VOC for c in a + b)
        ]
        keys = {uk} | {a for _, a, b in use}
        known = [old[k] for k in keys if k in old]
        rank = (
            2
            if pid in official_test
            else max(known)
            if known
            else (
                2
                if int(digest("v8:" + uk)[:8], 16) % 100 < 5
                else 1
                if int(digest("v8:" + uk)[:8], 16) % 100 < 10
                else 0
            )
        )
        group_rank[uk] = max(group_rank.get(uk, -1), rank)
        proposals[pid] = (uk, use)
    # Assign parent groups first, then remove all parents whose full/source-clause key
    # appears in a higher-priority split. Entire parents are removed, never split apart.
    owners = {}
    for pid, (uk, use) in proposals.items():
        rank = group_rank[uk]
        for k in {uk} | {a for _, a, b in use}:
            owners[k] = max(owners.get(k, -1), rank)
    removed = collections.Counter()
    fullstats = {}
    singlestats = {}
    v7pairs = {(r["upper"], r["lower"]) for r in rows(ROOT / "runs/final_couplet_v7/train.jsonl")}
    new_single = 0
    handles = {s: (OUT / "full" / f"{s}.jsonl") for s in SPLITS}
    (OUT / "full").mkdir(exist_ok=True)
    (OUT / "single_clause_v7").mkdir(exist_ok=True)
    streams = {s: p.open("w", encoding="utf8") for s, p in handles.items()}
    fullstats = {
        s: {
            "rows": 0,
            "multi_clause": 0,
            "han_length_counts": collections.Counter(),
            "unknown_character_rows": 0,
            "sequence_128_compatible": 0,
        }
        for s in SPLITS
    }
    for pid, u, l, uk, aaj, bbj, eligible, src in db.execute("SELECT * FROM pairs ORDER BY pid"):
        if pid not in proposals:
            continue
        _, use = proposals[pid]
        rank = group_rank[uk]
        s = SPLITS[rank]
        bad = [k for k in {uk} | {a for _, a, b in use} if owners[k] > rank]
        if bad:
            removed[s] += 1
            db.execute(
                "INSERT INTO exports VALUES(?,?,?)",
                (pid, s, "cross_split_source_or_clause_conflict"),
            )
            line(
                quarantine,
                {
                    "id": pid,
                    "upper": u,
                    "lower": l,
                    "reason": "cross_split_source_or_clause_conflict",
                    "proposed_split": s,
                },
            )
            continue
        aa = json.loads(aaj)
        bb = json.loads(bbj)
        unknown = sorted(set(c for c in u + l if c not in VOC))
        n = len(uk)
        compatible = not unknown and len(u) + len(l) + 3 <= 128
        rec = {
            "id": pid,
            "upper": u,
            "lower": l,
            "upper_clauses": aa,
            "lower_clauses": bb,
            "clause_lengths": list(map(len, aa)),
            "han_length": n,
            "text_length": len(u),
            "split": s,
            "group_id": digest(uk),
            "first_source": src,
            "v7_whole_line_compatible": compatible,
            "unknown_characters": unknown,
            "literary_quality": "not_expert_verified",
        }
        line(streams[s], rec)
        db.execute("INSERT INTO exports VALUES(?,?,?)", (pid, s, "retained"))
        fs = fullstats[s]
        fs["rows"] += 1
        fs["multi_clause"] += len(aa) > 1
        fs["han_length_counts"][n] += 1
        fs["unknown_character_rows"] += bool(unknown)
        fs["sequence_128_compatible"] += compatible
        for j, a, b in use:
            sid = digest(a + "\t" + b)
            db.execute("INSERT OR IGNORE INTO singles VALUES(?,?,?,?)", (sid, a, b, s))
            db.execute("INSERT INTO derivations VALUES(?,?,?)", (sid, pid, j))
    for f in streams.values():
        f.close()
    quarantine.close()
    db.commit()
    db.execute("CREATE INDEX derivations_sid ON derivations(sid)")
    db.commit()
    for s in SPLITS:
        st = {"rows": 0, "length_counts": collections.Counter(), "new_vs_v7_training_pairs": 0}
        singlestats[s] = st
        with (OUT / "single_clause_v7" / f"{s}.jsonl").open("w", encoding="utf8") as f:
            for sid, a, b in db.execute(
                "SELECT sid,u,l FROM singles WHERE split=? ORDER BY sid", (s,)
            ):
                st["rows"] += 1
                st["length_counts"][len(a)] += 1
                if s == "train" and (a, b) not in v7pairs:
                    st["new_vs_v7_training_pairs"] += 1
                line(
                    f,
                    {
                        "id": sid,
                        "upper": a,
                        "lower": b,
                        "split": s,
                        "group_id": digest(a),
                        "length": len(a),
                        "derivation_lookup": "registry.sqlite3::derivations",
                        "literary_quality": "not_expert_verified",
                    },
                )
    with (OUT / "derivations.jsonl").open("w", encoding="utf8") as f:
        for sid, pid, j in db.execute("SELECT * FROM derivations ORDER BY sid,pid,clause_index"):
            line(f, {"id": sid, "parent_id": pid, "clause_index": j})
    stats = {
        "status": "built",
        "full": fullstats,
        "single_clause_v7": singlestats,
        "quarantined_parent_conflicts": dict(removed),
        "unique_normalized_full_pairs": db.execute("SELECT COUNT(*) FROM pairs").fetchone()[0],
        "unique_unknown_source_only_pairs": db.execute(
            "SELECT COUNT(*) FROM pairs WHERE eligible=0"
        ).fetchone()[0],
        "split_seed": "v8:sha256 upper; existing heldout protected; test wins validation wins train",
        "near_duplicate_filter": "none; exact normalized full upper and exported source-clause isolation only",
        "short_derived_clauses": "Multi-clause auxiliary extraction requires >=4 chars; standalone 1-3 character pairs retained",
        "training_started": False,
        "independent_test_claim": False,
        "vocab_unchanged": True,
    }
    dump(OUT / "statistics.json", stats)
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    db.close()
    print("BUILD COMPLETE", json.dumps(stats, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
