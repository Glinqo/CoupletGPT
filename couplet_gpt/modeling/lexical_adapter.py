"""Source-only lexical fusion in our from-scratch GPT; no target feature leakage."""

import functools
import json
import os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from couplet_gpt.modeling.gpt import GPTConfig
from couplet_gpt.modeling.structure import StructuredGPTV3
from couplet_gpt.data.lexicon import LexiconMatcher, OUT
from couplet_gpt.data.datasets import CharacterTokenizer

DEFAULT_CONTEXT_ROOT = Path(
    os.environ.get("COUPLET_V5_CONTEXT_CACHE", str(OUT / "chinesebert_contexts"))
)


class LexicalGPT(StructuredGPTV3):
    def __init__(self, config, entries, feature_source="local"):
        super().__init__(config, prefix=False)
        assert feature_source in ("local", "chinesebert")
        self.feature_source = feature_source
        self.entries = entries
        self.matcher = LexiconMatcher(entries)
        self.word_residual = nn.Embedding(len(entries) + 1, 128, padding_idx=0)
        self.feature_project = nn.Linear(256 if feature_source == "local" else 768, 128)
        self.word_transform = nn.Sequential(
            nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, config.width)
        )
        self.word_attention = nn.Parameter(torch.empty(config.width, config.width))
        self.lexical_norm = nn.LayerNorm(config.width, eps=config.norm_eps)
        self.lexical_output = nn.Linear(config.width, config.width, bias=False)
        self.lexical_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "base_vectors",
            torch.zeros(len(entries) + 1, 256)
            if feature_source == "local"
            else torch.empty(0, 256),
        )
        for module in (self.feature_project, self.word_transform, self.lexical_norm):
            module.apply(self._init)
        nn.init.normal_(self.word_attention, std=0.02)
        nn.init.zeros_(self.word_residual.weight)
        nn.init.zeros_(self.lexical_output.weight)
        self.context_provider = None

    @torch.no_grad()
    def initialize_local_vectors(self):
        if self.feature_source == "local":
            for row in self.entries:
                self.base_vectors[row["id"]] = self.tok_emb.weight[row["token_ids"]].mean(0)

    @functools.lru_cache(maxsize=100000)
    def _match_tuple(self, upper):
        return self.matcher.match(upper)[0]

    def metadata_for(self, ids):
        result = []
        for row in ids.detach().cpu().tolist():
            down = row.index(5) if 5 in row else 2
            result.append(self._match_tuple(tuple(row[2:down])))
        return torch.from_numpy(np.stack(result).astype(np.int64)).to(ids.device)

    def lexical_delta(self, h, meta, contexts=None, context_valid=None):
        b, slots, k, _ = meta.shape
        wid, starts, ends = meta.unbind(-1)
        valid = wid != 0
        if self.feature_source == "local":
            features = self.feature_project(self.base_vectors[wid])
        else:
            assert contexts is not None and context_valid is not None
            # Linear projection commutes with span averaging. Project 7 characters, not 56 spans.
            projected = self.feature_project(contexts.to(h.dtype))
            prefix = torch.cat([projected.new_zeros(b, 1, 128), projected.cumsum(1)], 1)
            rows = torch.arange(b, device=h.device)[:, None, None]
            lengths = (ends - starts).clamp_min(1)
            features = (prefix[rows, ends] - prefix[rows, starts]) / lengths[..., None]
            known = torch.cat([context_valid.new_zeros(b, 1), context_valid.cumsum(1)], 1)
            valid = valid & ((known[rows, ends] - known[rows, starts]) == (ends - starts))
        features = features + self.word_residual(wid)
        words = self.word_transform(features)
        queries = h[:, 2 : 2 + slots]
        scores = torch.einsum("btd,df,btkf->btk", queries, self.word_attention, words)
        weights = scores.masked_fill(~valid, -10000.0).softmax(-1) * valid
        value = (weights[..., None] * words).sum(2)
        delta = self.lexical_output(self.lexical_dropout(self.lexical_norm(value)))
        return delta * valid.any(-1)[..., None]

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
        assert t <= self.config.context_length
        if lexical_meta is None:
            lexical_meta = self.metadata_for(ids)
        # Slot count is 7 for paired data; trim only when a short source-only diagnostic is used.
        slots = min(7, t - 2)
        lexical_meta = lexical_meta[:, :slots]
        if self.feature_source == "chinesebert" and contexts is None:
            assert self.context_provider is not None, (
                "Attach a frozen source-context provider for inference"
            )
            contexts, context_valid = self.context_provider(ids)
        if contexts is not None:
            contexts, context_valid = contexts[:, :slots], context_valid[:, :slots]
        pos = torch.arange(t, device=ids.device)
        down = (ids == 5).long().argmax(1)
        rel = pos[None, :] - down[:, None]
        align_valid = (down[:, None] > 0) & (rel >= 0) & (rel < (down - 2)[:, None])
        source = (rel + 2).clamp(0, t - 1)
        h = self.drop(self.tok_emb(ids) + self.pos_emb(pos))
        for layer, (block, projection) in enumerate(zip(self.blocks, self.alignment)):
            h = block(h, explicit=explicit)
            if layer == 0:
                delta = self.lexical_delta(h, lexical_meta, contexts, context_valid)
                h = h.clone()
                h[:, 2 : 2 + slots] = h[:, 2 : 2 + slots] + delta
            aligned = h.gather(1, source[:, :, None].expand(-1, -1, h.shape[-1]))
            h = h + projection(aligned) * align_valid[:, :, None]
        h = self.norm(h)
        logits = self.head(h)
        return (logits, self.boundary_head(h)) if return_aux else logits


def load_lexical_checkpoint(path, device="cuda", provider=None):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    c = GPTConfig(**ck["model_config"])
    if ck["branch"] == "baseline":
        model = StructuredGPTV3(c, prefix=False)
    else:
        model = LexicalGPT(c, ck["lexicon_entries"], ck["feature_source"])
        model.context_provider = provider
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck


class ContextStore:
    """Input-only mmap cache. Unseen uppers require explicit frozen-encoder preparation."""

    def __init__(self, folder=DEFAULT_CONTEXT_ROOT):
        self.folder = folder
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "complete"
        self.index = {tuple(row): i for i, row in enumerate(manifest["source_token_ids"])}
        self.values = np.load(folder / "hidden.npy", mmap_mode="r")
        self.valid = np.load(folder / "valid.npy", mmap_mode="r")

    def __call__(self, ids):
        keys = []
        for row in ids.detach().cpu().tolist():
            down = row.index(5)
            keys.append(tuple(row[2:down]))
        indices = [self.index[k] for k in keys]
        return (
            torch.from_numpy(np.array(self.values[indices])).to(ids.device),
            torch.from_numpy(np.array(self.valid[indices], dtype=np.int64)).to(ids.device),
        )
