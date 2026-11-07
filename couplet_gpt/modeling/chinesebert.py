"""Faithful frozen ChineseBERT fusion/encoder, adapted to current Transformers APIs.

Reference: ShannonAI/ChineseBert models/{fusion,pinyin,glyph}_embedding.py and
modeling_glycebert.py at 67f45cda50e75053ca3143bb8fadc3748cdfd13e.
All pretrained encoder tensors, including the original glyph table, load strictly.
No generated target or reference lower line is ever encoded.
"""

import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler
from tokenizers import BertWordPieceTokenizer
from pypinyin import pinyin, Style
from couplet_gpt.data.datasets import ROOT, CoupletDataset, CharacterTokenizer
from couplet_gpt.data.lexicon import OUT
import os
from couplet_gpt.training.transfer_comparison import digest
from couplet_gpt.training.pretrain import json_write

EXTERNAL = Path(
    os.environ.get("COUPLET_V5_CHINESEBERT_MODEL", str(ROOT / "models/external/ChineseBERT-base"))
)
CACHE = OUT / "chinesebert_contexts"


class PinyinEmbedding(nn.Module):
    def __init__(self, size, width):
        super().__init__()
        self.embedding = nn.Embedding(size, 128)
        self.conv = nn.Conv1d(128, width, 2)

    def forward(self, ids):
        x = self.embedding(ids)
        b, t, n, d = x.shape
        x = self.conv(x.reshape(-1, n, d).permute(0, 2, 1))
        return x.max(-1).values.reshape(b, t, -1)


class GlyphEmbedding(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        # The original font tensor is already present in the official checkpoint.
        self.embedding = nn.Embedding(vocab, 1728)

    def forward(self, ids):
        return self.embedding(ids)


class FusionEmbeddings(nn.Module):
    def __init__(self, c, pinyin_size):
        super().__init__()
        self.word_embeddings = nn.Embedding(c.vocab_size, c.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(c.max_position_embeddings, c.hidden_size)
        self.token_type_embeddings = nn.Embedding(c.type_vocab_size, c.hidden_size)
        self.pinyin_embeddings = PinyinEmbedding(pinyin_size, c.hidden_size)
        self.glyph_embeddings = GlyphEmbedding(c.vocab_size)
        self.glyph_map = nn.Linear(1728, c.hidden_size)
        self.map_fc = nn.Linear(c.hidden_size * 3, c.hidden_size)
        self.LayerNorm = nn.LayerNorm(c.hidden_size, eps=c.layer_norm_eps)
        self.dropout = nn.Dropout(c.hidden_dropout_prob)
        self.register_buffer("position_ids", torch.arange(c.max_position_embeddings)[None])

    def forward(self, ids, py, pinyin_enabled=True, glyph_enabled=True):
        word = self.word_embeddings(ids)
        sound = self.pinyin_embeddings(py)
        glyph = self.glyph_map(self.glyph_embeddings(ids))
        if not pinyin_enabled:
            sound = sound * 0
        if not glyph_enabled:
            glyph = glyph * 0
        x = self.map_fc(torch.cat([word, sound, glyph], -1))
        x = x + self.position_embeddings(self.position_ids[:, : ids.shape[1]])
        x = x + self.token_type_embeddings(torch.zeros_like(ids))
        return self.dropout(self.LayerNorm(x))


class FrozenChineseBERT(nn.Module):
    def __init__(self, folder=EXTERNAL):
        super().__init__()
        config = BertConfig.from_json_file(str(folder / "config.json"))
        config._attn_implementation = "sdpa"
        self.config = config
        pmap = json.loads((folder / "config/pinyin_map.json").read_text(encoding="utf-8"))
        self.embeddings = FusionEmbeddings(config, len(pmap["idx2char"]))
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        weights = torch.load(folder / "pytorch_model.bin", map_location="cpu", weights_only=True)
        state = {k[len("bert.") :]: v for k, v in weights.items() if k.startswith("bert.")}
        self.load_state_dict(state, strict=True)
        assert set(state) == set(self.state_dict())
        assert all(torch.equal(v, state[k]) for k, v in self.state_dict().items())
        self.load_checks = {
            "strict_encoder_state_loaded": True,
            "loaded_tensors": len(state),
            "ignored_tensors": [k for k in weights if not k.startswith("bert.")],
            "parameters": sum(p.numel() for p in self.parameters()),
            "checkpoint_sha256": digest(folder / "pytorch_model.bin"),
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
        }
        self.requires_grad_(False)
        self.eval()

    def forward(self, ids, py, mask, **feature_flags):
        h = self.embeddings(ids, py, **feature_flags)
        attention = (1.0 - mask[:, None, None, :].to(h.dtype)) * -10000.0
        return self.encoder(h, attention_mask=attention, return_dict=True).last_hidden_state


class SourceTokenizer:
    def __init__(self, folder=EXTERNAL):
        self.tokenizer = BertWordPieceTokenizer(str(folder / "vocab.txt"))
        self.pmap = json.loads((folder / "config/pinyin_map.json").read_text(encoding="utf-8"))
        self.pytensor = json.loads(
            (folder / "config/pinyin2tensor.json").read_text(encoding="utf-8")
        )

    def encode(self, text):
        encoded = self.tokenizer.encode(text)
        sounds = pinyin(
            text, style=Style.TONE3, heteronym=True, errors=lambda x: [["not chinese"] for _ in x]
        )
        sound_ids = {}
        for i, choices in enumerate(sounds):
            reading = choices[0]
            if reading == "not chinese":
                continue
            if reading in self.pytensor:
                sound_ids[i] = self.pytensor[reading]
            else:
                values = [0] * 8
                for j, c in enumerate(reading):
                    if c not in self.pmap["char2idx"] or j >= 8:
                        values = [0] * 8
                        break
                    values[j] = self.pmap["char2idx"][c]
                sound_ids[i] = values
        py = []
        positions = np.full(7, -1, dtype=np.int16)
        for i, (token, offset) in enumerate(zip(encoded.tokens, encoded.offsets)):
            a, b = offset
            py.append(sound_ids.get(a, [0] * 8) if b - a == 1 else [0] * 8)
            if b - a == 1 and token != "[UNK]" and a < len(text):
                positions[a] = i
        assert len(encoded.ids) <= 9 and len(text) <= 7
        return encoded.ids, py, positions


def prepare():
    CACHE.mkdir(parents=True, exist_ok=True)
    if (CACHE / "prepared.json").exists():
        print("REUSE PREPARED SOURCE INPUTS", flush=True)
        return
    sources = []
    seen = set()
    split_sizes = {}
    for split in ("train", "validation"):
        ds = CoupletDataset(split)
        split_sizes[split] = len(ds)
        for i, n in enumerate((ds.lengths - 3) // 2):
            key = tuple(map(int, ds.x[i, 2 : 2 + int(n)]))
            if key not in seen:
                seen.add(key)
                sources.append(list(key))
    # Fixed historical/fresh-review source prompts, no reference lower lines are loaded.
    for name in ("tune_inputs.json", "audit_inputs.json", "blind_inputs.json"):
        rows = json.loads((ROOT / "runs/optimization_v3" / name).read_text(encoding="utf-8"))
        tok = CharacterTokenizer()
        for row in rows:
            key = tuple(tok.encode(row["upper"]))
            if key not in seen:
                sources.append(list(key))
                seen.add(key)
    tokenize_sources(sources, split_sizes)


def tokenize_sources(sources, split_sizes, extra=False):
    started = time.perf_counter()
    tok = CharacterTokenizer()
    tokenizer = SourceTokenizer()
    n = len(sources)
    ids = np.zeros((n, 9), dtype=np.int32)
    py = np.zeros((n, 9, 8), dtype=np.uint8)
    pos = np.full((n, 7), -1, dtype=np.int16)
    for i, source in enumerate(sources):
        # All identical GPT source token sequences must produce identical external features.
        # Original UNKs become a single visible replacement character, not a multi-character special token string.
        text = "".join(tok.reverse[x] if x >= 6 else "\ufffd" for x in source)
        tokens, sounds, positions = tokenizer.encode(text)
        ids[i, : len(tokens)] = tokens
        py[i, : len(tokens)] = sounds
        pos[i] = positions
        if (i + 1) % 50000 == 0:
            print(
                "TOKENIZE SOURCES",
                i + 1,
                "/",
                n,
                round(time.perf_counter() - started, 1),
                flush=True,
            )
    np.save(CACHE / "input_ids.npy", ids)
    np.save(CACHE / "pinyin_ids.npy", py)
    np.save(CACHE / "positions.npy", pos)
    json_write(
        CACHE / "prepared.json",
        {
            "source_token_ids": sources,
            "split_rows": split_sizes,
            "source_only": True,
            "reference_labels_read": False,
            "unique_sources": n,
            "elapsed_seconds": time.perf_counter() - started,
            "extra": extra,
            "pinyin_policy": "Official sentence tokenizer: TONE3 heteronym=True, first context candidate; full known upper only",
            "code_sha256": digest(Path(__file__)),
        },
    )
    print("SOURCE TOKENIZATION COMPLETE", n, flush=True)


@torch.inference_mode()
def encode(batch_size=256):
    if (CACHE / "manifest.json").exists():
        print("REUSE COMPLETE SOURCE CONTEXTS", flush=True)
        return
    prepared = json.loads((CACHE / "prepared.json").read_text(encoding="utf-8"))
    ids = np.load(CACHE / "input_ids.npy", mmap_mode="r")
    py = np.load(CACHE / "pinyin_ids.npy", mmap_mode="r")
    pos = np.load(CACHE / "positions.npy", mmap_mode="r")
    model = FrozenChineseBERT().cuda().eval()
    xx = torch.tensor(np.array(ids[:8]), device="cuda", dtype=torch.long)
    pp = torch.tensor(np.array(py[:8]), device="cuda", dtype=torch.long)
    mm = xx != 0
    a, b = model(xx, pp, mm), model(xx, pp, mm)
    assert torch.equal(a, b)
    disabled_py = model(xx, pp, mm, pinyin_enabled=False)
    disabled_glyph = model(xx, pp, mm, glyph_enabled=False)
    pd = float((a - disabled_py).abs().max())
    gd = float((a - disabled_glyph).abs().max())
    assert pd > 1e-4 and gd > 1e-4
    assert not any(p.requires_grad for p in model.parameters())
    model.load_checks.update(
        status="pass",
        eval_deterministic=True,
        pinyin_branch_ablation_max_delta=pd,
        glyph_branch_ablation_max_delta=gd,
        frozen=True,
        source_only=True,
        references_never_encoded=True,
        code_sha256=digest(Path(__file__)),
    )
    json_write(CACHE / "encoder_checks.json", model.load_checks)
    output = np.lib.format.open_memmap(
        CACHE / "hidden.npy", mode="w+", dtype=np.float16, shape=(len(ids), 7, 768)
    )
    valid = pos >= 0
    np.save(CACHE / "valid.npy", valid.astype(np.uint8))
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for start in range(0, len(ids), batch_size):
        end = min(start + batch_size, len(ids))
        x = torch.tensor(np.array(ids[start:end]), device="cuda", dtype=torch.long)
        p = torch.tensor(np.array(py[start:end]), device="cuda", dtype=torch.long)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = model(x, p, x != 0)
        locations = torch.tensor(np.array(pos[start:end]), device="cuda", dtype=torch.long)
        pooled = hidden.gather(1, locations.clamp_min(0)[..., None].expand(-1, -1, 768))
        pooled = pooled * (locations >= 0)[..., None]
        output[start:end] = pooled.float().cpu().numpy().astype(np.float16)
        if (start // batch_size) % 100 == 0:
            print(
                "ENCODE SOURCE",
                end,
                "/",
                len(ids),
                round(time.perf_counter() - started, 1),
                flush=True,
            )
    output.flush()
    lengths = np.array(list(map(len, prepared["source_token_ids"])))
    manifest = {
        **prepared,
        "status": "complete",
        "hidden_shape": list(output.shape),
        "encoder_checks": model.load_checks,
        "dtype": "float16",
        "covered_characters": int(valid.sum()),
        "total_characters": int(lengths.sum()),
        "character_coverage": float(valid.sum() / lengths.sum()),
        "fully_covered_sources": int((valid.sum(1) == lengths).sum()),
        "encoding_seconds": time.perf_counter() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "hidden_bytes": (CACHE / "hidden.npy").stat().st_size,
        "hidden_sha256": digest(CACHE / "hidden.npy"),
    }
    json_write(CACHE / "manifest.json", manifest)
    print(
        "CONTEXT CACHE COMPLETE",
        {k: v for k, v in manifest.items() if k != "source_token_ids"},
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--encode-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if not args.encode_only:
        prepare()
    if not args.prepare_only:
        encode()


if __name__ == "__main__":
    main()
