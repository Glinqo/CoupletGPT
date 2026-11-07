"""Build a frozen lexicon from archived dictionaries and training text only."""

import collections
import json
import os
import re
import time
import numpy as np
from couplet_gpt.data.datasets import ROOT, CharacterTokenizer, CoupletDataset
from couplet_gpt.evaluation.metrics import han
from couplet_gpt.training.transfer_comparison import digest
from couplet_gpt.training.pretrain import json_write

OUT = ROOT / "runs" / os.environ.get("COUPLET_V5_RUN", "lexicon_gpt_v5")


class LexiconMatcher:
    def __init__(self, entries, max_word_length=None):
        self.entries = entries
        self.trie = {}
        for row in entries:
            node = self.trie
            for token in row["token_ids"]:
                node = node.setdefault(int(token), {})
            node[-1] = row["id"]
        self.rank = {r["id"]: i for i, r in enumerate(entries)}
        self.metadata_dtype = (
            np.int32 if max(self.rank, default=0) > np.iinfo(np.int16).max else np.int16
        )
        self.max_word_length = max_word_length or max(
            (len(r["token_ids"]) for r in entries), default=4
        )

    def match(self, ids, slots=7, k=8):
        ids = list(map(int, ids))
        matches = [[] for _ in ids]
        for start in range(len(ids)):
            node = self.trie
            for end in range(start, min(start + self.max_word_length, len(ids))):
                node = node.get(ids[end])
                if node is None:
                    break
                if -1 in node:
                    for pos in range(start, end + 1):
                        matches[pos].append((node[-1], start, end + 1))
        result = np.zeros((slots, k, 3), dtype=self.metadata_dtype)
        truncated = 0
        for pos, values in enumerate(matches):
            values.sort(key=lambda v: self.rank[v[0]])
            truncated += max(0, len(values) - k)
            if values:
                result[pos, : min(k, len(values))] = values[:k]
        return result, truncated


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "data_ready.json").exists():
        ready = json.loads((OUT / "data_ready.json").read_text(encoding="utf-8"))
        assert digest(OUT / "lexicon.json") == ready["lexicon_sha256"]
        print("REUSE FROZEN LEXICON", ready["lexicon_words"], flush=True)
        return
    started = time.perf_counter()
    tok = CharacterTokenizer()
    sources = ROOT / "runs/optimization_v4/knowledge_special/sources"
    candidates = collections.defaultdict(set)
    for filename, key in [("ci.json", "ci"), ("idiom.json", "word")]:
        rows = json.loads((sources / filename).read_text(encoding="utf-8"))
        for row in rows:
            word = row[key].strip()
            if 2 <= len(word) <= 4 and all(han(c) and c in tok.vocab for c in word):
                candidates[word].add(filename)
    extra_sources = {}
    if os.environ.get("COUPLET_V5_COMPLETE_LEXICON") == "1":
        import jieba

        dictionary = __import__("pathlib").Path(jieba.__file__).parent / "dict.txt"
        with dictionary.open(encoding="utf-8") as f:
            for line in f:
                word = line.split()[0]
                if 2 <= len(word) <= 4 and all(han(c) and c in tok.vocab for c in word):
                    candidates[word].add("jieba-0.42.1/dict.txt")
        extra_sources[str(dictionary)] = digest(dictionary)
    poem, couplet, distinct = collections.Counter(), collections.Counter(), collections.Counter()
    seen = set()
    inputs = [
        ROOT / "data/processed/pretrain_train.jsonl",
        ROOT / "data/processed/couplet_train.jsonl",
    ]
    for path in inputs:
        count = 0
        with path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                texts = [row["text"]] if "text" in row else [row["upper"], row["lower"]]
                target = poem if "text" in row else couplet
                for text in texts:
                    # Punctuation splits the original normalized text; never cross a clause or pair.
                    for clause in re.findall(
                        r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002ee5f\U00030000-\U0003347f]+",
                        text,
                    ):
                        new = clause not in seen
                        if new:
                            seen.add(clause)
                        present = set()
                        for i in range(len(clause) - 1):
                            for length in (2, 3, 4):
                                if i + length > len(clause):
                                    break
                                word = clause[i : i + length]
                                if word in candidates:
                                    target[word] += 1
                                    if new:
                                        present.add(word)
                        distinct.update(present)
                count += 1
                if count % 100000 == 0:
                    print(
                        "COUNT",
                        path.name,
                        count,
                        round(time.perf_counter() - started, 1),
                        flush=True,
                    )
    eligible = [w for w in candidates if distinct[w] >= 2]
    eligible.sort(key=lambda w: (-couplet[w], -poem[w], w))
    entries = [
        {
            "id": i + 1,
            "word": w,
            "token_ids": tok.encode(w),
            "sources": sorted(candidates[w]),
            "couplet_occurrences": couplet[w],
            "pretrain_occurrences": poem[w],
            "distinct_training_clauses": distinct[w],
        }
        for i, w in enumerate(eligible[:8192])
    ]
    json_write(OUT / "lexicon.json", entries)
    matcher = LexiconMatcher(entries)
    coverage = {}
    # Freeze vocabulary before touching validation inputs; no validation/test reference is read here.
    for split in ("train", "validation"):
        ds = CoupletDataset(split)
        mapped = np.zeros((len(ds), 7, 8, 3), dtype=np.int16)
        covered = chars = any_match = truncated = 0
        for i in range(len(ds)):
            n = (int(ds.lengths[i]) - 3) // 2
            mapped[i], removed = matcher.match(ds.x[i, 2 : 2 + n])
            has = (mapped[i, :, :, 0] != 0).any(-1)
            covered += int(has[:n].sum())
            chars += n
            any_match += int(has.any())
            truncated += removed
        np.save(OUT / f"lexical_{split}.npy", mapped)
        coverage[split] = {
            "rows": len(ds),
            "source_characters": chars,
            "covered_characters": covered,
            "character_coverage": covered / chars,
            "any_word_rows": any_match,
            "any_word_rate": any_match / len(ds),
            "truncated_candidates": truncated,
        }
    ready = {
        "status": "ready",
        "lexicon_words": len(entries),
        "eligible_words": len(eligible),
        "dictionary_candidates": len(candidates),
        "unique_training_clauses": len(seen),
        "selection": "distinct clauses>=2; descending train couplet count, pretrain count, then word; cap8192",
        "source_only": True,
        "no_definitions_or_references": True,
        "training_files": {str(p.relative_to(ROOT)): digest(p) for p in inputs},
        "dictionary_files": {n: digest(sources / n) for n in ("ci.json", "idiom.json")},
        "extra_dictionary_files": extra_sources,
        "vocab_sha256": digest(ROOT / "tokenizer/vocab.json"),
        "lexicon_sha256": digest(OUT / "lexicon.json"),
        "coverage": coverage,
        "cache_sha256": {s: digest(OUT / f"lexical_{s}.npy") for s in ("train", "validation")},
        "elapsed_seconds": time.perf_counter() - started,
        "builder_sha256": digest(__import__("pathlib").Path(__file__)),
        "validation_note": "Only already-known source token tensors matched after lexicon freeze; labels never read.",
    }
    json_write(OUT / "data_ready.json", ready)
    print(json.dumps(ready, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    build()
