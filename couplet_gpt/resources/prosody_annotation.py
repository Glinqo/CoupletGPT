"""Deterministic prosody first sentence; preserves uncertainty and never certifies full meter."""

from couplet_gpt.paths import ROOT as PROJECT_ROOT
import collections, functools, hashlib, html, json, re, shutil, unicodedata
from pathlib import Path
from opencc import OpenCC

ROOT = PROJECT_ROOT
OUT = ROOT / "data/poetry_appreciation"
HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U000323af]")
UPSTREAM = ROOT / "data/resources/prosody"
DICT_HASH = "fd11164ca1b848eb6b777688d5d6628ce772d72d0a25a0efd12d6b58459c6e2e"


def han(s):
    return "".join(HAN_RE.findall(s))


def text(s):
    if not isinstance(s, str):
        return ""
    return unicodedata.normalize("NFC", html.unescape(re.sub("<[^>]*>", "", s))).strip()


def family_id(s):
    return hashlib.sha256(han(s).encode("utf8")).hexdigest()


class ToneAnnotator:
    def __init__(self):
        dest = OUT / "resources/prosody"
        dest.mkdir(parents=True, exist_ok=True)
        for name in ["Pingshui_Rhyme.json", "LICENSE", "README.md", "manifest.json"]:
            target = dest / name
            if not target.exists():
                shutil.copy2(UPSTREAM / name, target)
        raw = (dest / "Pingshui_Rhyme.json").read_bytes()
        assert hashlib.sha256(raw).hexdigest() == DICT_HASH
        book = json.loads(raw)
        self.entries = collections.defaultdict(set)
        cc = OpenCC("t2s")
        self.skipped = 0
        for department, groups in book.items():
            assert any(k in department for k in ["平", "上声", "去声", "入声"]), department
            category = "平" if "平" in department else "仄"
            for group, characters in groups.items():
                for c in characters:
                    if not isinstance(c, str) or len(c) != 1:
                        self.skipped += 1
                        continue
                    for spelling in {c, cc.convert(c)}:
                        if len(spelling) == 1:
                            self.entries[spelling].add((category, department, group))
        self.table = {
            c: next(iter(cats)) if len(cats := {r[0] for r in refs}) == 1 else "?"
            for c, refs in self.entries.items()
        }
        self.mode = "pingshui_dictionary"

    @functools.lru_cache(maxsize=16000)
    def modern(self, c):
        from pypinyin import pinyin, Style

        readings = pinyin(c, style=Style.TONE3, heteronym=True, neutral_tone_with_five=True)[0]
        cats = {
            "平" if r[-1:] in ["1", "2"] else "仄" if r[-1:] in ["3", "4"] else "?"
            for r in readings
        }
        return next(iter(cats)) if len(cats) == 1 and "?" not in cats else "?"

    def pattern(self, s, mode):
        return "".join(
            self.table.get(c, "?") if mode == "pingshui_dictionary" else self.modern(c)
            for c in han(s)
        )

    def annotate(
        self, source, genre_hint="unknown", mode="pingshui_dictionary", upper=None, lower=None
    ):
        assert mode in ["pingshui_dictionary", "modern_character_dictionary"]
        source = text(source)
        scope = "平水韵字表" if mode == "pingshui_dictionary" else "普通话字典音类"
        if upper is not None and lower is not None:
            lines = [han(upper), han(lower)]
            kind = "couplet"
            applicable = True
        else:
            # Punctuation defines displayed clauses, not automatically verified metrical beats.
            lines = [han(x) for x in re.split("[，,。！？!?；;：:\n\r]+", source) if han(x)]
            lens = [len(x) for x in lines]
            if genre_hint == "prose":
                kind = "prose"
                applicable = False
            elif genre_hint == "ci":
                kind = "ci_without_tune_verification"
                applicable = True
            elif len(lines) in [2, 4, 8] and len(set(lens)) == 1 and lens[0] in [5, 7]:
                kind = "short_equal_length_verse_candidate"
                applicable = True
            else:
                kind = "long_or_irregular_or_genre_unresolved"
                applicable = True
        patterns = [self.pattern(s, mode) for s in lines] if applicable else []
        glyph_issue = any(unicodedata.category(c) == "Co" for c in source)
        if glyph_issue:
            kind = "unresolved_private_use_glyphs"
            applicable = False
            patterns = []
        if not lines:
            kind = "no_annotatable_han_characters"
            applicable = False
            patterns = []
        unknown = [
            {
                "line": i + 1,
                "position": j + 1,
                "character": lines[i][j],
                "reason": "tone_category_ambiguous"
                if lines[i][j] in self.table and mode == "pingshui_dictionary"
                else "uncovered_or_unresolved_reading",
            }
            for i, p in enumerate(patterns)
            for j, c in enumerate(p)
            if c == "?"
        ]
        covered = sum(len(p) - p.count("?") for p in patterns)
        total = sum(map(len, patterns))
        if kind == "unresolved_private_use_glyphs":
            first = "原文含未还原的私用区字形，暂不生成可靠的逐字平仄结论。"
        elif kind == "no_annotatable_han_characters":
            first = "原文没有可识别的汉字，本条不作平仄分析。"
        elif kind == "prose":
            first = "此篇按古文处理，不套用近体诗或对联的平仄格律判断。"
        elif kind == "couplet":
            foot = [p[-1] if p else "?" for p in patterns]
            first = f"按{scope}自动标注，上联为「{patterns[0]}」，下联为「{patterns[1]}」"
            if "?" in foot:
                first += "，句脚尚有未定音类"
            else:
                first += f"，句脚分别为{foot[0]}、{foot[1]}"
            first += ("（?为歧义或未覆盖）" if unknown else "") + "。"
        elif kind == "short_equal_length_verse_candidate":
            first = f"按{scope}自动标注，各分句平仄依次为「" + "／".join(patterns) + "」"
            first += ("（?为歧义或未覆盖）" if unknown else "") + "。"
        elif kind == "ci_without_tune_verification":
            first = f"此词已按{scope}作逐字平仄标注，详见字音表，未据此校验词牌格律。"
        else:
            first = f"按{scope}已标注{len(lines)}个分句，{len(unknown)}处音类未定，详见字音表，不据此作整篇合律结论。"
        result = {
            "program_first_sentence": first,
            "mode": mode,
            "form_scope": kind,
            "patterns": patterns,
            "line_texts": lines,
            "line_lengths": [len(s) for s in lines],
            "ambiguous_or_uncovered": unknown,
            "classified_characters": covered,
            "total_annotated_characters": total,
            "applicable": applicable,
            "source_dictionary_sha256": DICT_HASH,
            "label_origin": "program_dictionary_derived_not_expert_gold",
            "context_pronunciation_disambiguation_verified": False,
            "full_meter_verified": False,
            "punctuation_clause_boundaries_not_verified_beats": True,
        }
        result["source_glyph_issue"] = glyph_issue
        assert first.endswith("。") and first.count("。") == 1
        return result


def render(annotation, body):
    return annotation["program_first_sentence"] + text(body)


def short_candidates(source, body, annotation, max_body=220, max_samples=3):
    """Extract intact source paragraphs; never create a summary or silently truncate a paragraph."""
    body = text(body)
    source = text(source)
    if annotation.get("source_glyph_issue"):
        return []
    if len(han(source)) > 128 or len(annotation["program_first_sentence"]) > 150:
        return []
    chunks = []
    cursor = 0
    for match in re.finditer(r"[^\r\n]+", body):
        para = match[0].strip()
        left = match.start() + len(match[0]) - len(match[0].lstrip())
        if not 60 <= len(para) <= max_body:
            continue
        if not re.search("[。！？!?…]$", para):
            continue
        if re.search("上文|下文|上述|以下|前文|再如|总而言之|综上|本节|如下", para):
            continue
        if any(unicodedata.category(c) == "Co" for c in para):
            continue
        if re.search(r"[�<>]|\\u[0-9a-fA-F]{4}", para):
            continue
        if annotation["form_scope"] == "couplet":
            if not all(
                any(s[i : i + 2] in para for i in range(len(s) - 1))
                for s in annotation["line_texts"]
            ):
                continue
        anchors = {
            source[i : i + 2]
            for i in range(len(source) - 1)
            if source[i : i + 2] in para and len(han(source[i : i + 2])) == 2
        }
        if len(anchors) < 2:
            continue
        rendered = render(annotation, para)
        prompt = (
            "原文："
            + source
            + "\n字音依据："
            + annotation["program_first_sentence"]
            + "\n请结合具体字词赏析，写一段连贯短评："
        )
        if len(prompt) + len(para) + 2 > 768:
            continue
        chunks.append(
            {
                "response_body": para,
                "comment_body_span": [left, left + len(para)],
                "program_first_sentence": annotation["program_first_sentence"],
                "display_response": rendered,
                "prompt": prompt,
                "loss_scope": "response_body_and_EOS_only_program_first_sentence_not_neural_target",
                "selection_score": len(anchors),
                "contains_technical_claims_needing_review": bool(
                    re.search("平仄|合律|对仗|词性|主谓|偏正|动宾", para)
                ),
                "review_status": "automatic_paragraph_candidate_not_summary_or_expert_approved",
            }
        )
    chunks = sorted(chunks, key=lambda r: (-r["selection_score"], r["comment_body_span"][0]))[
        :max_samples
    ]
    for r in chunks:
        r["family_weight"] = 1 / len(chunks)
    return sorted(chunks, key=lambda r: r["comment_body_span"][0])
