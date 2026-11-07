"""Dynamic lexical slots and frozen ChineseBERT source contexts; original GPT architecture retained."""

from couplet_gpt.paths import ROOT as _ROOT

PROJECT_CACHE = _ROOT / "data/cache/couplet_base"
import functools
import numpy as np
import torch
from couplet_gpt.modeling.gpt import GPTConfig
from couplet_gpt.modeling.lexical_adapter import LexicalGPT
from couplet_gpt.data.couplet_splits import OUT, BASE, read, rows


class DynamicChineseBERT:
    def __init__(self):
        self.online = {}
        self.encoder = None
        cache = PROJECT_CACHE
        if (cache / "context_ready.json").exists():
            self.values = np.load(cache / "contexts.npy", mmap_mode="r")
            self.valid = np.load(cache / "context_valid.npy", mmap_mode="r")
            self.index = {
                tuple(r["ids"]): (r["offset"], r["length"])
                for r in rows(cache / "context_index.jsonl")
            }
        else:
            self.index = {}

    @torch.inference_mode()
    def add(self, keys):
        from couplet_gpt.modeling.chinesebert import FrozenChineseBERT, SourceTokenizer
        from couplet_gpt.data.datasets import CharacterTokenizer

        if self.encoder is None:
            self.encoder = FrozenChineseBERT().cuda().eval()
            self.tokenizer = SourceTokenizer()
        tok = CharacterTokenizer()
        width = max(map(len, keys)) + 2
        ids = np.zeros((len(keys), width), np.int64)
        py = np.zeros((len(keys), width, 8), np.int64)
        locations = np.full((len(keys), width - 2), -1, np.int64)
        from pypinyin import pinyin, Style

        for i, key in enumerate(keys):
            text = tok.decode(key)
            enc = self.tokenizer.tokenizer.encode(text)
            sounds = pinyin(text, style=Style.TONE3, heteronym=True)
            mapping = {}
            for j, candidates in enumerate(sounds):
                sound = candidates[0]
                if sound in self.tokenizer.pytensor:
                    mapping[j] = self.tokenizer.pytensor[sound]
                else:
                    v = [0] * 8
                    if len(sound) <= 8 and all(c in self.tokenizer.pmap["char2idx"] for c in sound):
                        for k, c in enumerate(sound):
                            v[k] = self.tokenizer.pmap["char2idx"][c]
                    mapping[j] = v
            ids[i, : len(enc.ids)] = enc.ids
            for j, (token, (a, b)) in enumerate(zip(enc.tokens, enc.offsets)):
                if b - a == 1 and token != "[UNK]":
                    locations[i, a] = j
                    py[i, j] = mapping.get(a, [0] * 8)
        x = torch.tensor(ids, device="cuda")
        pp = torch.tensor(py, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = self.encoder(x, pp, x != 0)
        pos = torch.tensor(locations, device="cuda")
        v = h.gather(1, pos.clamp_min(0)[..., None].expand(-1, -1, 768)) * (pos >= 0)[..., None]
        for key, row, mask in zip(keys, v.float().cpu().numpy().astype(np.float16), locations >= 0):
            self.online[key] = (row[: len(key)], mask[: len(key)])

    def __call__(self, ids):
        keys = [tuple(r[2 : r.index(5)]) for r in ids.detach().cpu().tolist()]
        missing = list(
            dict.fromkeys(k for k in keys if k not in self.index and k not in self.online)
        )
        if missing:
            self.add(missing)
        width = max(map(len, keys))
        v = np.zeros((len(keys), width, 768), np.float16)
        m = np.zeros((len(keys), width), np.int64)
        for i, k in enumerate(keys):
            if k in self.index:
                start, n = self.index[k]
                a, b = self.values[start : start + n], self.valid[start : start + n]
            else:
                a, b = self.online[k]
            v[i, : len(k)] = a
            m[i, : len(k)] = b
        return torch.tensor(v, device=ids.device), torch.tensor(m, device=ids.device)


class FinalGPT(LexicalGPT):
    @functools.lru_cache(maxsize=20000)
    def _dynamic_match(self, key):
        return self.matcher.match(key, slots=len(key))[0]

    def metadata_for(self, ids):
        keys = [tuple(r[2 : r.index(5)]) for r in ids.detach().cpu().tolist()]
        n = max(map(len, keys))
        a = np.zeros((len(keys), n, 8, 3), np.int64)
        for i, k in enumerate(keys):
            a[i, : len(k)] = self._dynamic_match(k)
        return torch.tensor(a, device=ids.device)

    def forward(
        self,
        ids,
        explicit=False,
        return_aux=False,
        lexical_meta=None,
        contexts=None,
        context_valid=None,
    ):
        b, t = ids.shape
        down = (ids == 5).long().argmax(1)
        n = int((down - 2).max())
        if t > self.config.context_length:
            raise ValueError("Input exceeds the trained positional context; no silent truncation.")
        meta = self.metadata_for(ids) if lexical_meta is None else lexical_meta
        if contexts is None:
            contexts, context_valid = self.context_provider(ids)
        pos = torch.arange(t, device=ids.device)
        rel = pos[None, :] - down[:, None]
        valid = (rel >= 0) & (rel < (down - 2)[:, None])
        source = (rel + 2).clamp(0, t - 1)
        h = self.drop(self.tok_emb(ids) + self.pos_emb(pos))
        for layer, (block, projection) in enumerate(zip(self.blocks, self.alignment)):
            h = block(h, explicit=explicit)
            if layer == 0:
                delta = self.lexical_delta(h, meta, contexts, context_valid)
                h = h.clone()
                h[:, 2 : 2 + n] += delta
            aligned = h.gather(1, source[:, :, None].expand(-1, -1, h.shape[-1]))
            h = h + projection(aligned) * valid[:, :, None]
        h = self.norm(h)
        z = self.head(h)
        return (z, self.boundary_head(h)) if return_aux else z


def initialize(path=BASE):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    m = FinalGPT(GPTConfig(**ck["model_config"]), ck["lexicon_entries"], "chinesebert")
    state = {k: v for k, v in ck["model"].items() if not k.startswith("syntax.")}
    m.load_state_dict(state)
    m = m.cuda()
    m.context_provider = DynamicChineseBERT()
    return m, ck
