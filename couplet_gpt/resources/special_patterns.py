"""Sourced idiom evidence, train-only examples, and explicit shared-radical checks.

This auxiliary module never certifies an allusion counterpart or generates an answer.
The caller must keep the from-scratch GPT and report auxiliary contributions separately.
"""

from couplet_gpt.paths import ROOT as PROJECT_ROOT
import argparse
import collections
import functools
import hashlib
import itertools
import math
import re
import json
from pathlib import Path

ROOT = PROJECT_ROOT
OUT = ROOT / "runs/optimization_v4/knowledge_special"


def subsequence(a, b):
    it = iter(b)
    return all(any(c == d for d in it) for c in a)


class KnowledgeSpecial:
    def __init__(self, root=OUT):
        self.root = Path(root)
        self.idioms = json.loads((self.root / "sources/idiom.json").read_text(encoding="utf-8"))
        self.words = json.loads((self.root / "sources/word.json").read_text(encoding="utf-8"))
        self.historical = {}
        self.short_glosses = {}
        ci = self.root / "sources/ci.json"
        if ci.exists():
            for row in json.loads(ci.read_text(encoding="utf-8")):
                explanation = row.get("explanation", "")
                if 2 <= len(row.get("ci", "")) <= 3:
                    self.short_glosses[row["ci"]] = explanation
                if 2 <= len(row.get("ci", "")) <= 7 and re.search(
                    "之典|典故|后因|后用|晋朝|相传|取信于民|世说新语", explanation
                ):
                    self.historical[row["ci"]] = explanation
        self.radicals = {
            r["word"]: r.get("radicals", "") for r in self.words if len(r["word"]) == 1
        }
        self.exact = collections.defaultdict(list)
        self.anchors = collections.defaultdict(list)
        for i, r in enumerate(self.idioms):
            w = r["word"]
            self.exact[w].append(i)
            if len(w) == 4:
                for a in set(w[j : j + 2] for j in range(3)):
                    self.anchors[a].append(i)
        self.inventory = collections.defaultdict(set)
        for c, r in self.radicals.items():
            if r:
                self.inventory[r].add(c)
        self.train_index = None
        self.phrase_bank = None
        self._planning_index = None

    @functools.lru_cache(maxsize=100000)
    def idiom_evidence(self, text):
        matches = {}
        for n in range(2, min(8, len(text)) + 1):
            for p in range(len(text) - n + 1):
                for i in self.exact.get(text[p : p + n], []):
                    matches[i] = "exact_phrase"
        for j in range(len(text) - 1):
            indices = self.anchors.get(text[j : j + 2], [])
            # Rare lexical anchors avoid flagging every common poetic bigram.
            if len(indices) > 4:
                continue
            for i in indices:
                if i in matches:
                    continue
                word = self.idioms[i]["word"]
                if len(set(word) & set(text)) >= 3 and any(
                    subsequence("".join(x), text) for x in itertools.combinations(word, 3)
                ):
                    matches[i] = "rare_anchor_plus_three_char_subsequence"
        rows = []
        for i, kind in matches.items():
            r = self.idioms[i]
            rows.append(
                {
                    "phrase": r["word"],
                    "match_type": kind,
                    "definition": r.get("explanation", ""),
                    "derivation": r.get("derivation", ""),
                    "source": "pwxcoo/chinese-xinhua/data/idiom.json",
                    "dictionary_record_index": i,
                    "primary_source_manually_verified": False,
                }
            )
        return sorted(
            rows, key=lambda x: (x["match_type"] != "exact_phrase", -len(x["phrase"]), x["phrase"])
        )[:12]

    def radical_spans(self, text, min_run=4):
        tags = [self.radicals.get(c) for c in text]
        result = []
        i = 0
        while i < len(text):
            j = i + 1
            while j < len(text) and tags[i] and tags[j] == tags[i]:
                j += 1
            if tags[i] and j - i >= min_run:
                result.append(
                    {
                        "start": i,
                        "stop": j,
                        "text": text[i:j],
                        "radical": tags[i],
                        "scope": "dictionary radical, not recursive IDS or universal literary requirement",
                    }
                )
            i = j
        return result

    def analyze(self, upper):
        evidence = self.idiom_evidence(upper)
        spans = self.radical_spans(upper)
        history = self.historical_evidence(upper)
        return {
            "upper": upper,
            "idiom_evidence": evidence,
            "shared_radical_spans": spans,
            "historical_lexical_evidence": history,
            "knowledge_status": "needs_allusion_review"
            if history
            else "idiom_match_not_allusion_certification"
            if evidence
            else "no_dictionary_match_not_proof_of_no_allusion",
            "special_structure_status": "shared_radical_candidate" if spans else "not_detected",
            "allusion_correctness_verified": False,
            "full_literary_quality_verified": False,
        }

    def historical_evidence(self, text):
        rows = []
        for n in range(min(7, len(text)), 1, -1):
            for i in range(len(text) - n + 1):
                term = text[i : i + n]
                if term in self.historical:
                    rows.append(
                        {
                            "phrase": term,
                            "definition": self.historical[term],
                            "source": "pwxcoo/chinese-xinhua/data/ci.json",
                            "primary_source_manually_verified": False,
                        }
                    )
        return rows[:8]

    def radical_check(self, upper, lower):
        spans = self.radical_spans(upper)
        if not spans:
            return {"status": "not_applicable", "details": []}
        if len(upper) != len(lower):
            return {"status": "fail", "reason": "length_mismatch", "details": []}
        details = []
        for span in spans:
            tt = [self.radicals.get(c) for c in lower[span["start"] : span["stop"]]]
            status = (
                "unknown" if any(not t for t in tt) else "pass" if len(set(tt)) == 1 else "fail"
            )
            details.append(
                {
                    **span,
                    "lower": lower[span["start"] : span["stop"]],
                    "lower_radicals": tt,
                    "status": status,
                }
            )
        status = (
            "fail"
            if any(d["status"] == "fail" for d in details)
            else "unknown"
            if any(d["status"] == "unknown" for d in details)
            else "pass"
        )
        return {
            "status": status,
            "details": details,
            "semantic_correctness_verified": False,
            "note": "Only invoke as an explicit special-mode constraint; ordinary couplets need not share radicals.",
        }

    def radical_mask_plan(self, upper, vocab, target_radical=None):
        """Return per-position allowed IDs for caller's search, or feasible group choices.

        A target group must be explicit: blindly selecting a radical is not semantics.
        """
        spans = self.radical_spans(upper)
        if not spans:
            return {"status": "not_applicable", "position_allowed_ids": {}}
        available = {
            r: sorted(vocab[c] for c in chars if c in vocab) for r, chars in self.inventory.items()
        }
        available = {r: ids for r, ids in available.items() if len(ids) >= 4}
        if target_radical is None:
            return {
                "status": "requires_explicit_radical_plan",
                "spans": spans,
                "available_radical_counts": {r: len(ids) for r, ids in available.items()},
                "position_allowed_ids": {},
            }
        ids = available.get(target_radical, [])
        if not ids:
            return {"status": "unsupported_target_radical", "position_allowed_ids": {}}
        return {
            "status": "ready",
            "target_radical": target_radical,
            "spans": spans,
            "position_allowed_ids": {j: ids for s in spans for j in range(s["start"], s["stop"])},
            "scope": "Syntactic candidate mask only; GPT/semantic validator must still judge the result.",
        }

    def retrieve_train_examples(self, upper, limit=5):
        path = self.root / "train_knowledge_index.json"
        if self.train_index is None:
            self.train_index = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        result = []
        seen = set()
        for e in self.idiom_evidence(upper) + self.historical_evidence(upper):
            for r in self.train_index.get(e["phrase"], []):
                if r["upper"] == upper:
                    continue
                key = (r["upper"], r["lower"])
                if key not in seen:
                    result.append({**r, "matched_phrase": e["phrase"]})
                    seen.add(key)
                if len(result) >= limit:
                    return result
        return result

    def assess_candidate(self, upper, lower):
        analysis = self.analyze(upper)
        radical = self.radical_check(upper, lower)
        if radical["status"] == "fail":
            status = "reject_shared_radical_mismatch"
        elif analysis["historical_lexical_evidence"]:
            status = "needs_allusion_review"
        elif radical["status"] == "unknown":
            status = "needs_radical_review"
        else:
            status = "no_special_violation_detected_not_semantic_certification"
        return {
            "status": status,
            "analysis": analysis,
            "radical_check": radical,
            "allusion_correctness_verified": False,
            "safe_to_claim_full_couplet_quality": False,
        }

    def shared_radical_candidates(self, upper, limit=64, target_radical=None):
        """Compose training-attested character spans; never label them GPT outputs.

        Currently supports a whole-line shared-radical signal. Partial/multiple spans
        return a constraint plan, leaving unrestrained positions to the GPT decoder.
        """
        spans = self.radical_spans(upper)
        if len(spans) != 1 or spans[0]["start"] != 0 or spans[0]["stop"] != len(upper):
            return {
                "status": "use_gpt_with_position_mask",
                "candidates": [],
                "constraint_plan": self.radical_mask_plan(upper, {}),
            }
        if self.phrase_bank is None:
            p = self.root / "radical_train_phrase_bank.json"
            self.phrase_bank = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        n = len(upper)
        source = spans[0]["radical"]
        frames = {5: [(2, 2, 1), (2, 3)], 7: [(2, 2, 2, 1), (2, 2, 3)]}.get(n, [])
        groups = (
            [target_radical] if target_radical else [r for r in self.phrase_bank if r != source]
        )
        found = {}
        for radical in groups:
            bank = self.phrase_bank.get(radical, {})
            for frame in frames:
                beam = [("", 0.0, [])]
                for size in frame:
                    options = bank.get(str(size), [])[:16]
                    expanded = []
                    for text, score, evidence in beam:
                        for option in options:
                            y = text + option["text"]
                            m = len(y)
                            if any(a == b for a, b in zip(upper, y)):
                                continue
                            # Preserve repeated-character equivalence incrementally.
                            if any(
                                (upper[i] == upper[j]) != (y[i] == y[j])
                                for i in range(m)
                                for j in range(i)
                            ):
                                continue
                            expanded.append(
                                (y, score + math.log1p(option["count"]), evidence + [option])
                            )
                    beam = sorted(expanded, key=lambda x: x[1], reverse=True)[:32]
                for y, score, evidence in beam:
                    if len(y) == n and self.radical_check(upper, y)["status"] == "pass":
                        found[y] = {
                            "text": y,
                            "target_radical": radical,
                            "composition_score": score,
                            "origin": "train_phrase_composition_not_raw_GPT",
                            "fragments": evidence,
                            "semantic_and_tonal_validity": "unverified_requires_GPT_and_independent_checker",
                        }
        return {
            "status": "candidates_require_quality_filter"
            if found
            else "unsupported_no_attested_composition",
            "candidates": sorted(
                found.values(), key=lambda x: x["composition_score"], reverse=True
            )[:limit],
            "allusion_correctness_verified": False,
            "fallback": "Do not emit an unsupported draft as a certified couplet.",
        }

    @staticmethod
    def _semantic_gloss(text):
        # Remove quoted narrative and book titles to emphasize the dictionary gloss.
        text = re.sub(r'"[^"]*"|“[^”]*”|《[^》]*》', " ", text)
        text = re.sub(
            "后用|后因|后来|后世|比喻|形容|代指|用以|用来|通常|之典|典故|指|谓|称", "", text
        )
        return re.sub(r"[的了是在其之以于曰云者而也乃为所一二三四五六七八九十\W\d_]+", "", text)

    def _sense_gloss(self, row, upper):
        parts = re.split(
            "[。；;\n①②③④⑤]", re.sub(r'"[^"]*"|“[^”]*”|《[^》]*》', " ", row["definition"])
        )
        purpose = set(upper[-3:]) - set(row["phrase"])
        scored = [(len(set(p) & purpose) / max(4, len(p)), p) for p in parts if p.strip()]
        if not scored:
            return self._semantic_gloss(row["definition"])
        best = max(s[0] for s in scored)
        return self._semantic_gloss(
            "".join(p for score, p in scored if score >= best and best > 0)
        ) or self._semantic_gloss(row["definition"])

    def _build_planning_index(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        records = []
        seen = set()
        for i, r in enumerate(self.idioms):
            w = r["word"]
            derivation = r.get("derivation", "")
            if len(w) != 4 or "《" not in derivation or w in seen:
                continue
            seen.add(w)
            records.append(
                {
                    "prefix": w,
                    "definition": r.get("explanation", ""),
                    "derivation": derivation,
                    "entry_id": f"idiom:{i}",
                    "source": "pwxcoo/chinese-xinhua/data/idiom.json",
                    "construction": "verbatim_four_character_dictionary_entry",
                    "relation_verified": False,
                }
            )
            # A provisional actor+event frame derived from the *same* citation.
            # It is not certified syntax or event extraction; retain the exact quote.
            if re.match("[\u4e00-\u9fff]{4}", w):
                import jieba.posseg as pseg

                verb = list(pseg.cut(w[0]))
                event = w[:2]
                if verb and verb[0].flag.startswith("v") and event in derivation:
                    # Saying/quoting an event does not make someone its actor.
                    # Retain only an explicit intent narrative and a recognized name.
                    actors = set(re.findall(r"([\u4e00-\u9fff]{1}子)(?=欲)", derivation))
                    for actor in actors:
                        person_tags = list(pseg.cut(actor))
                        if not (len(person_tags) == 1 and person_tags[0].flag.startswith("nr")):
                            continue
                        narrative = re.search(re.escape(actor) + r"欲([^。；;]{1,16})", derivation)
                        if not narrative or not set(event) <= set(narrative.group(1)):
                            continue
                        prefix = actor + event
                        if prefix == w or prefix in seen:
                            continue
                        seen.add(prefix)
                        records.append(
                            {
                                "prefix": prefix,
                                "definition": r.get("explanation", ""),
                                "derivation": derivation,
                                "entry_id": f"idiom:{i}:actor_event:{actor}:{event}",
                                "source": "pwxcoo/chinese-xinhua/data/idiom.json",
                                "construction": "citation_actor_event_cooccurrence",
                                "actor": actor,
                                "event": event,
                                "relation_verified": False,
                            }
                        )
        for w, definition in self.historical.items():
            if len(w) == 4 and w not in seen:
                seen.add(w)
                records.append(
                    {
                        "prefix": w,
                        "definition": definition,
                        "derivation": definition,
                        "entry_id": "ci:" + w,
                        "source": "pwxcoo/chinese-xinhua/data/ci.json",
                        "construction": "verbatim_four_character_dictionary_entry",
                        "relation_verified": False,
                    }
                )
        texts = [
            self._semantic_gloss(r["definition"]) * 3 + self._semantic_gloss(r["derivation"])
            for r in records
        ]
        vectorizer = TfidfVectorizer(
            analyzer="char", ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True
        )
        matrix = vectorizer.fit_transform(texts)
        self._planning_index = (records, vectorizer, matrix)

    def allusion_prefix_plans(self, upper, limit=8):
        """Source-gloss TFIDF proposes four-character prefixes, not complete lower lines.

        Every result preserves documentary evidence. A positive retrieval score is
        neither a correctness probability nor a claim that the target is a good pair.
        """
        source = self.historical_evidence(upper)
        if not source:
            return {
                "status": "unknown_no_historical_source_gloss",
                "plans": [],
                "fallback": "Use ordinary GPT or request a specified allusion; do not invent a source.",
            }
        if self._planning_index is None:
            self._build_planning_index()
        records, vectorizer, matrix = self._planning_index
        query = "".join(self._sense_gloss(r, upper) for r in source) * 3 + self._semantic_gloss(
            upper[-3:]
        )
        for size in (2, 3):
            for start in range(max(0, len(upper) - 3), len(upper) - size + 1):
                gloss = self.short_glosses.get(upper[start : start + size])
                if gloss:
                    query += self._semantic_gloss(gloss)
        q = vectorizer.transform([query])
        scores = (matrix @ q.T).toarray().ravel()
        import numpy as np

        order = np.argsort(-scores)[: max(200, limit * 20)]
        results = []
        qfeatures = set(vectorizer.inverse_transform(q)[0])
        source_phrases = {r["phrase"] for r in source} | {
            r["phrase"] for r in self.idiom_evidence(upper)
        }
        source_pairs = {upper[j : j + 2] for j in range(len(upper) - 1)}
        source_citations = {
            re.sub("[^\u4e00-\u9fff]", "", s)
            for r in source
            for s in re.findall("《([^》]+)》", r["definition"])
        }
        source_citations = {s for s in source_citations if len(s) >= 5}
        import jieba.posseg as pseg

        source_person = any(t.flag.startswith("nr") for t in pseg.cut(upper[:2]))
        if source_person:
            actor_indices = [
                i
                for i, r in enumerate(records)
                if r["construction"] == "citation_actor_event_cooccurrence" and scores[i] > 0
            ]
            actor_indices = sorted(actor_indices, key=lambda i: -scores[i])[:2]
            order = list(dict.fromkeys(list(order) + actor_indices))
        try:
            from couplet_gpt.resources.lexical_semantics import LexicalGrammarV4

            grammar = getattr(self, "_planning_grammar", None)
            if grammar is None:
                self._planning_grammar = grammar = LexicalGrammarV4()
        except ImportError:
            grammar = None
        for ix in order:
            row = records[int(ix)]
            prefix = row["prefix"]
            score = float(scores[ix])
            if score <= 0 or prefix in source_phrases or prefix in upper:
                continue
            if any(a == b for a, b in zip(upper, prefix)):
                continue
            # Avoid feeding a source phrase back as a purported different allusion.
            if any(w in prefix for w in source_phrases if len(w) >= 2):
                continue
            if prefix[:2] in source_pairs or prefix[2:] in source_pairs:
                continue
            target_citations = {
                re.sub("[^\u4e00-\u9fff]", "", s)
                for s in re.findall("《([^》]+)》", row["derivation"])
            }
            if source_citations & target_citations:
                continue
            overlap = sorted(
                qfeatures & set(vectorizer.inverse_transform(matrix[int(ix)])[0]),
                key=lambda s: (-len(s), s),
            )[:20]
            confidence = (
                "low_unverified_actor_event_relation"
                if row["construction"] == "citation_actor_event_cooccurrence"
                else "lexical_retrieval_only"
            )
            choices = [(prefix, "identity")]
            import jieba.posseg as pseg

            tags = list(pseg.cut(prefix[:2]))
            person = (
                any(t.flag.startswith("nr") for t in tags)
                or prefix[:2] + "人名" in row["definition"]
            )
            if row["construction"] == "verbatim_four_character_dictionary_entry" and not person:
                choices.append((prefix[2:] + prefix[:2], "swap_two_attested_two_character_blocks"))
            for proposed, transform in choices:
                if any(a == b for a, b in zip(upper, proposed)):
                    continue
                pos_score = None
                if grammar:
                    overlaps = []
                    for a, b in [(upper[:2], proposed[:2]), (upper[2:4], proposed[2:])]:
                        aa = grammar.lookup(a)["pos_distribution"]
                        bb = grammar.lookup(b)["pos_distribution"]
                        if aa and bb:
                            overlaps.append(sum(min(v, bb.get(k, 0.0)) for k, v in aa.items()))
                    pos_score = sum(overlaps) / len(overlaps) if overlaps else None
                actor_frame = source_person and person
                if row["construction"] == "citation_actor_event_cooccurrence":
                    actor_frame = source_person
                planning_score = score + 0.08 * (pos_score or 0.0) + 0.12 * actor_frame
                results.append(
                    {
                        **row,
                        "prefix": proposed,
                        "dictionary_prefix": prefix,
                        "transformation": transform,
                        "source_glosses": source,
                        "selected_source_gloss": query,
                        "retrieval_score": score,
                        "matched_gloss_features": overlap,
                        "prefix_pos_overlap": pos_score,
                        "weak_person_frame_match": actor_frame,
                        "planning_score": planning_score,
                        "confidence": confidence,
                        "word_structure": "unknown_requires_lexical_grammar_check",
                        "allusion_correctness_verified": False,
                        "full_literary_quality_verified": False,
                        "origin": "dictionary_supported_prefix_plan_not_raw_GPT",
                    }
                )
            if len(results) >= limit * 4 and not source_person:
                break
        # POS is a soft rerank; absent/multiple senses never become fabricated gold.
        results.sort(key=lambda r: r["planning_score"], reverse=True)
        # A small structure-diversity quota avoids losing a supported person+event
        # frame solely because short glosses have low lexical overlap.
        actor_plans = (
            [r for r in results if r["construction"] == "citation_actor_event_cooccurrence"][:2]
            if source_person
            else []
        )
        kept = results[:limit]
        for plan in actor_plans:
            if plan not in kept:
                kept = kept[: max(0, limit - 1)] + [
                    dict(plan, selection_reason="source_person_frame_diversity_quota")
                ]
        results = kept
        return {
            "status": "proposed_unverified_prefixes"
            if results
            else "unknown_no_distinct_supported_prefix",
            "plans": results,
            "index_records": len(records),
            "fallback": "GPT may complete these prefixes; reject unsuitable completions rather than certify the plan.",
        }


def build_index(engine):
    index = collections.defaultdict(list)
    count = 0
    phrase_counts = collections.Counter()
    phrase_sources = {}
    path = ROOT / "data/processed/couplet_train.jsonl"
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        hits = engine.idiom_evidence(r["upper"]) + engine.historical_evidence(r["upper"])
        for hit in hits:
            if len(index[hit["phrase"]]) < 8:
                index[hit["phrase"]].append({k: r[k] for k in ["id", "upper", "lower", "raw_ref"]})
        lower = r["lower"]
        for size in (1, 2, 3):
            for start in range(len(lower) - size + 1):
                text = lower[start : start + size]
                radicals = [engine.radicals.get(c) for c in text]
                if radicals[0] and len(set(radicals)) == 1:
                    key = (radicals[0], size, text)
                    phrase_counts[key] += 1
                    if key not in phrase_sources:
                        phrase_sources[key] = {
                            "training_row": r["id"],
                            "raw_ref": r["raw_ref"],
                            "span": [start, start + size],
                        }
        count += 1
    (OUT / "train_knowledge_index.json").write_text(
        json.dumps(index, ensure_ascii=False), encoding="utf-8"
    )
    bank = collections.defaultdict(lambda: collections.defaultdict(list))
    for (radical, size, text), frequency in phrase_counts.items():
        bank[radical][str(size)].append(
            {"text": text, "count": frequency, "source": phrase_sources[(radical, size, text)]}
        )
    for radical in bank:
        for size in bank[radical]:
            bank[radical][size] = sorted(
                bank[radical][size], key=lambda x: (-x["count"], x["text"])
            )[:128]
    (OUT / "radical_train_phrase_bank.json").write_text(
        json.dumps(bank, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "training_rows_scanned": count,
        "indexed_phrases": len(index),
        "radical_phrase_groups": len(bank),
        "training_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def self_check(engine):
    # Programmatic dictionary fixtures: never train on or call these expert gold.
    cases = []
    for radical, chars in sorted(engine.inventory.items()):
        chars = sorted(c for c in chars if "\u4e00" <= c <= "\u9fff")
        if len(chars) < 12:
            continue
        upper = "".join(chars[:5])
        good = "".join(chars[5:10])
        other = next(
            c
            for c, r in engine.radicals.items()
            if r and r != radical and "\u4e00" <= c <= "\u9fff"
        )
        bad = good[:2] + other + good[3:]
        assert engine.radical_check(upper, good)["status"] == "pass"
        assert engine.radical_check(upper, bad)["status"] == "fail"
        cases.append(
            {
                "upper": upper,
                "positive": good,
                "negative": bad,
                "label_origin": "programmatic_dictionary_radical_only",
            }
        )
    exact = []
    for r in engine.idioms[:: max(1, len(engine.idioms) // 100)]:
        if len(r["word"]) == 4:
            assert any(
                h["phrase"] == r["word"] for h in engine.idiom_evidence(r["word"] + "入诗中")
            )
            exact.append(
                {
                    "input": r["word"] + "入诗中",
                    "phrase": r["word"],
                    "label_origin": "dictionary_exact_string_not_human_allusion_judgment",
                }
            )
    out = {
        "radical_cases": cases,
        "dictionary_exact_match_cases": exact,
        "scope": "Component correctness and evidence retrieval, not semantic couplet quality.",
    }
    (OUT / "special_checks.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "radical_positive_cases": len(cases),
        "radical_single_corruption_cases": len(cases),
        "dictionary_exact_cases": len(exact),
        "status": "pass",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--upper")
    parser.add_argument("--lower")
    args = parser.parse_args()
    engine = KnowledgeSpecial()
    if args.build:
        result = {"index": build_index(engine), "tests": self_check(engine)}
        (OUT / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False))
    if args.upper:
        print(
            json.dumps(
                engine.assess_candidate(args.upper, args.lower)
                if args.lower
                else engine.analyze(args.upper),
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
