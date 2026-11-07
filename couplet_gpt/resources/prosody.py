"""Evidence-aware modern phrase and Pingshui prosody, independent of GPT weights.

No default-reading guess counts as disambiguation. Multiple traditional readings
remain unresolved. Only explicit beat positions are strict; even positions are
an optional soft proxy, never a universal full-couplet certificate.
"""

from couplet_gpt.paths import ROOT as PROJECT_ROOT
import argparse
import collections
import functools
import hashlib
import importlib.metadata
import json
import pathlib
import time
import numpy as np
from opencc import OpenCC
from pypinyin import pinyin, Style
from pypinyin.phrases_dict import phrases_dict
from pypinyin.contrib.tone_convert import to_tone3

ROOT = PROJECT_ROOT
OUT = ROOT / "runs/optimization_v4/prosody"
UPSTREAM = ROOT / "data/resources/prosody"


def tone3(value):
    return "P" if value[-1:] in ("1", "2") else "Z" if value[-1:] in ("3", "4") else "?"


@functools.lru_cache(maxsize=24000)
def modern_character(char):
    readings = pinyin(char, style=Style.TONE3, heteronym=True, neutral_tone_with_five=True)[0]
    return tuple(sorted(set(readings))), tuple(sorted({tone3(v) for v in readings}))


def singleton(options):
    values = set(options)
    return next(iter(values)) if len(values) == 1 and "?" not in values else "?"


@functools.lru_cache(maxsize=16000)
def phrase_reading_categories(readings):
    return tuple(sorted({tone3(to_tone3(v)) for v in readings}))


class ProsodyV4:
    def __init__(self, ancient_path=None):
        self.ancient = collections.defaultdict(set)
        path = pathlib.Path(ancient_path) if ancient_path else UPSTREAM / "Pingshui_Rhyme.json"
        self.ancient_available = path.exists()
        self.ancient_path = path
        self.cc = OpenCC("t2s")
        self._mask_cache = collections.OrderedDict()
        self._vocab_refs = {}
        if path.exists():
            book = json.loads(path.read_text(encoding="utf-8"))
            for department, groups in book.items():
                category = "P" if "平" in department else "Z"
                for group, characters in groups.items():
                    for c in characters:
                        # Entries with annotations are not parsed as a bare character.
                        if len(c) != 1:
                            continue
                        for spelling in {c, self.cc.convert(c)}:
                            if len(spelling) == 1:
                                self.ancient[spelling].add((category, department, group))

    @functools.lru_cache(maxsize=8192)
    def analyze(self, text, mode="modern_context"):
        if mode not in ("modern_context", "modern_character", "pingshui"):
            raise ValueError("mode must be modern_context, modern_character or pingshui")
        evidence = [[] for _ in text]
        if mode == "modern_context":
            # Keep ALL longest exact phrase matches covering each character.
            # Agreement is lexical evidence, not an oracle for a classical sentence.
            for start in range(len(text)):
                for end in range(start + 2, min(len(text), start + 12) + 1):
                    word = text[start:end]
                    entry = phrases_dict.get(word)
                    if entry is None or len(entry) != len(word):
                        continue
                    for j, readings in enumerate(entry):
                        vals = tuple(sorted({to_tone3(v) for v in readings}))
                        evidence[start + j].append(
                            {"phrase": word, "start": start, "length": len(word), "readings": vals}
                        )
        result = []
        for i, c in enumerate(text):
            if mode == "pingshui":
                entries = sorted(self.ancient.get(c, set()))
                cats = sorted({r[0] for r in entries}) or ["?"]
                result.append(
                    {
                        "char": c,
                        "tone": singleton(cats),
                        "tone_options": cats,
                        "source": "pingshui_dictionary" if entries else "uncovered",
                        "readings": [
                            {"tone": a, "department": b, "group": g} for a, b, g in entries
                        ],
                        "context_disambiguation_verified": False,
                    }
                )
            else:
                readings, cats = modern_character(c)
                ev = evidence[i]
                if ev:
                    longest = max(e["length"] for e in ev)
                    ev = [e for e in ev if e["length"] == longest]
                    readings = tuple(sorted({v for e in ev for v in e["readings"]}))
                    cats = tuple(sorted({tone3(v) for v in readings}))
                result.append(
                    {
                        "char": c,
                        "tone": singleton(cats),
                        "tone_options": list(cats),
                        "readings": list(readings),
                        "source": "exact_phrase_dictionary" if ev else "all_character_readings",
                        "phrase_evidence": ev,
                        "context_disambiguation_verified": False,
                    }
                )
        return result

    def ending_tone(self, text, mode="modern_context"):
        """Low allocation last-character path, for beam masks and corpus audits."""
        if not text:
            return "?"
        if mode == "pingshui":
            return singleton(r[0] for r in self.ancient.get(text[-1], set()))
        if mode == "modern_context":
            for width in range(min(len(text), 12), 1, -1):
                entry = phrases_dict.get(text[-width:])
                if entry is not None and len(entry) == width:
                    return singleton(phrase_reading_categories(tuple(entry[-1])))
        return singleton(modern_character(text[-1])[1])

    def inspect(self, upper, lower, mode="modern_context", profile="foot", beat_positions=None):
        if profile not in ("foot", "even_soft", "special"):
            raise ValueError("profile must be foot, even_soft or special")
        a = self.analyze(upper, mode)
        b = self.analyze(lower, mode)
        if not upper or not lower or len(upper) != len(lower):
            return {
                "status": "fail",
                "foot_status": "unknown",
                "score": -1.0,
                "reason": "length_or_empty",
                "full_traditional_rules_verified": False,
                "mode": mode,
                "profile": profile,
            }
        u, d = a[-1]["tone"], b[-1]["tone"]
        if profile == "special":
            foot = "not_applicable"
        elif u == "Z" and d == "P":
            foot = "pass"
        elif u == "Z" and d == "Z":
            foot = "fail"
        elif u == "P":
            foot = "not_applicable"
        else:
            foot = "unknown"
        positions = list(beat_positions or [])
        # Optional coarse proxy, not a hard universal rule.
        if profile == "even_soft" and beat_positions is None:
            positions = list(range(1, len(upper) - 1, 2))
        checks = []
        for index in positions:
            if not 0 <= index < len(upper):
                raise ValueError("beat positions use zero-based character indices")
            x, y = a[index]["tone"], b[index]["tone"]
            status = "unknown" if "?" in (x, y) else "pass" if x != y else "fail"
            checks.append(
                {"position_1based": index + 1, "upper_tone": x, "lower_tone": y, "status": status}
            )
        status = foot if foot in ("pass", "fail") else "unknown"
        if beat_positions is not None:
            if any(c["status"] == "fail" for c in checks):
                status = "fail"
            elif any(c["status"] == "unknown" for c in checks):
                status = "unknown"
        score = {"pass": 1.0, "fail": -1.0, "unknown": 0.0, "not_applicable": 0.0}[foot]
        if checks:
            score = 0.7 * score + 0.3 * sum(
                {"pass": 1.0, "fail": -1.0, "unknown": 0.0}[c["status"]] for c in checks
            ) / len(checks)
        return {
            "status": status,
            "foot_status": foot,
            "score": score,
            "mode": mode,
            "profile": profile,
            "upper": a,
            "lower": b,
            "beat_checks": checks,
            "explicit_beat_positions": beat_positions,
            "full_traditional_rules_verified": False,
            "scope": "Dictionary-supported tone proxy; ambiguous readings stay unknown, not a full expert certificate.",
        }

    def candidate_score(self, upper, lower, mode="modern_context", profile="foot"):
        return self.inspect(upper, lower, mode, profile)["score"]

    def allowed_next(
        self,
        upper,
        prefix,
        vocab,
        mode="modern_context",
        strict=False,
        beat_positions=None,
        profile="foot",
    ):
        """CPU bool mask indexed by token ID. Apply structural masks separately.

        In permissive mode exclude only definite tonal contradictions; unknown
        choices remain feasible and are labelled unknown on complete inspection.
        Strict mode accepts only confirmed opposition at explicitly required
        positions, and never silently relaxes an empty feasible mask.
        """
        # Tokenizers are immutable for an engine lifetime; retain references so
        # Python cannot reuse an id. Copy cached masks so caller mutations are safe.
        vid = id(vocab)
        if vid not in self._vocab_refs or self._vocab_refs[vid][1] != len(vocab):
            self._vocab_refs[vid] = (vocab, len(vocab), max(vocab.values()) + 1)
        size = self._vocab_refs[vid][2]
        key = (
            upper,
            prefix,
            mode,
            strict,
            tuple(beat_positions) if beat_positions is not None else None,
            vid,
            len(vocab),
            profile,
        )
        if key in self._mask_cache:
            self._mask_cache.move_to_end(key)
            return self._mask_cache[key].copy()
        mask = np.ones(size, dtype=bool)
        j = len(prefix)
        if j >= len(upper) or profile == "special":
            return mask
        a = self.analyze(upper, mode)
        required = None
        if j == len(upper) - 1 and a[-1]["tone"] == "Z":
            required = "P"
        elif beat_positions is not None and j in beat_positions and a[j]["tone"] != "?":
            required = "Z" if a[j]["tone"] == "P" else "P"
        if required is None:
            return mask
        for c, index in vocab.items():
            if len(c) != 1:
                continue
            tone = self.ending_tone(prefix + c, mode)
            mask[index] = (tone == required) if strict else (tone in (required, "?"))
        self._mask_cache[key] = mask.copy()
        if len(self._mask_cache) > 1024:
            self._mask_cache.popitem(last=False)
        return mask

    def legacy_strict_mask(self, upper, vocab):
        size = max(vocab.values()) + 1
        mask = np.ones(size, dtype=bool)
        if singleton(modern_character(upper[-1])[1]) != "Z":
            return mask
        for c, index in vocab.items():
            mask[index] = len(c) == 1 and singleton(modern_character(c)[1]) == "P"
        return mask


def run_checks():
    OUT.mkdir(parents=True, exist_ok=True)
    p = ProsodyV4()
    assert p.analyze("银行")[-1]["tone"] == "P"
    assert p.analyze("行", mode="modern_character")[-1]["tone"] == "?"
    assert p.analyze("快乐")[-1]["tone"] == "Z"
    assert p.analyze("白", mode="pingshui")[-1]["tone"] == "Z"
    assert p.analyze("白", mode="modern_character")[-1]["tone"] == "P"
    assert p.analyze("月", mode="pingshui")[-1]["tone"] == "Z"
    assert p.analyze("看", mode="pingshui")[-1]["tone"] == "?"
    assert p.inspect("春风又绿江南岸", "秋雨初红塞北天")["foot_status"] == "pass"
    assert p.inspect("春风又绿江南岸", "秋雨初红塞北岸")["foot_status"] == "fail"
    assert p.inspect("春风又绿江南岸", "秋雨初红塞北天", profile="special")["status"] == "unknown"
    vocab = {"天": 0, "岸": 1, "行": 2, "<EOS>": 3}
    m = p.allowed_next("春风又绿江南岸", "秋雨初红塞北", vocab)
    assert m[0] and not m[1] and m[2]
    s = p.allowed_next("春风又绿江南岸", "秋雨初红塞北", vocab, strict=True)
    assert s[0] and not s[1] and not s[2]
    result = {
        "status": "pass",
        "synthetic_checks": 12,
        "ancient_characters": len(p.ancient),
        "pypinyin_version": importlib.metadata.version("pypinyin"),
        "modern_context_vs_character_examples": {
            t: {
                mode: p.analyze(t, mode)
                for mode in ["modern_context", "modern_character", "pingshui"]
            }
            for t in ["银行", "快乐", "白", "月", "看"]
        },
        "scope": "Hand-authored controlled checks, not expert literary evaluation.",
    }
    (OUT / "checks.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return p


def audit(p, splits=("train", "validation")):
    result = {}
    start = time.perf_counter()
    for split in splits:
        counts = {
            mode: collections.Counter()
            for mode in ("modern_character", "modern_context", "pingshui")
        }
        changed = collections.Counter()
        rows = 0
        examples = []
        for line in (ROOT / f"data/processed/couplet_{split}.jsonl").open(encoding="utf-8"):
            r = json.loads(line)
            u, y = r["upper"], r["lower"]
            rows += 1
            statuses = {}
            for mode in counts:
                a = p.ending_tone(u, mode)
                b = p.ending_tone(y, mode)
                s = (
                    "pass"
                    if a == "Z" and b == "P"
                    else "fail"
                    if a == "Z" and b == "Z"
                    else "not_applicable"
                    if a == "P"
                    else "unknown"
                )
                counts[mode][s] += 1
                statuses[mode] = s
            if statuses["modern_context"] != statuses["modern_character"]:
                changed[statuses["modern_character"] + "->" + statuses["modern_context"]] += 1
                if len(examples) < 25:
                    examples.append({"upper": u, "lower": y, "statuses": statuses})
        result[split] = {
            "rows": rows,
            "counts": {k: dict(v) for k, v in counts.items()},
            "modern_context_transitions": dict(changed),
            "first25_transitions": examples,
        }
        print("PROSODY AUDIT", split, rows, {k: dict(v) for k, v in counts.items()}, flush=True)
    result["elapsed_seconds"] = time.perf_counter() - start
    result["note"] = (
        "Reference coverage and internal proxy disagreements, not accuracy against expert phonology labels."
    )
    (OUT / "reference_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    p = run_checks()
    if args.audit:
        audit(p)
    print("PROSODY V4 CHECKS COMPLETE")
