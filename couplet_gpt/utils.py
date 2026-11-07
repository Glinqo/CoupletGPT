"""Shared JSON I/O, hashing and Chinese normalization."""

import hashlib
import html
import json
import re
import unicodedata
from pathlib import Path
from opencc import OpenCC
from couplet_gpt.data.datasets import ROOT

CC = OpenCC("t2s")


def normalize(text):
    return CC.convert(
        unicodedata.normalize("NFC", html.unescape(re.sub(r"<[^>]+>", "", str(text))))
    ).strip()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    with Path(path).open(encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def jsonl(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in values:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
