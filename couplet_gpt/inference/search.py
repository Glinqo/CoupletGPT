"""Constrained batched search plus forward/reverse and soft lexical critic."""

import functools, hashlib, logging, math
import numpy as np
import torch
from torch.nn import functional as F
import jieba
import jieba.posseg as pseg
from couplet_gpt.data.datasets import CharacterTokenizer
from couplet_gpt.evaluation.diagnostics import tone, same_pattern
from couplet_gpt.evaluation.metrics import han
from couplet_gpt.modeling.alignment import load_model_v2

jieba.setLogLevel(logging.ERROR)


@functools.lru_cache(None)
def lexical_features(text):
    tags = []
    boundaries = set()
    pos = 0
    for word, flag in pseg.cut(text):
        category = flag[:1].upper() if flag[:1] in "nvadmq rpcutfs".replace(" ", "") else "X"
        tags.extend([category] * len(word))
        pos += len(word)
        if pos < len(text):
            boundaries.add(pos)
    return tuple(tags), frozenset(boundaries)


def lexical_score(upper, lower):
    a, ab = lexical_features(upper)
    b, bb = lexical_features(lower)
    matches = [x == y for x, y in zip(a, b) if x != "X" and y != "X"]
    pos = sum(matches) / len(matches) if matches else 0.0
    boundary = 2 * len(ab & bb) / (len(ab) + len(bb)) if ab or bb else 1.0
    return 0.7 * pos + 0.3 * boundary


def rule_check(upper, lower):
    equal = len(upper) == len(lower) and all(han(c) for c in lower)
    copy = any(a == b for a, b in zip(upper, lower))
    pattern = same_pattern(upper, lower)
    u = tone(upper[-1])
    d = tone(lower[-1]) if lower else "?"
    applicable = u == "Z"
    foot = (d == "P") if applicable else None
    return {
        "equal_han_length": equal,
        "no_same_position": not copy,
        "repeat_pattern": pattern,
        "upper_modern_tone": u,
        "lower_modern_tone": d,
        "modern_foot_applicable": applicable,
        "modern_foot_pass": foot,
        "encoded_rules_pass": equal and not copy and pattern and (foot is not False),
        "full_traditional_rules_verified": False,
    }


class V2Engine:
    def __init__(self, forward, reverse=None):
        self.model = forward.eval()
        self.reverse = reverse.eval() if reverse is not None else None
        self.tok = CharacterTokenizer()
        self.V = len(self.tok.vocab)
        self.han_mask = torch.tensor(
            [len(self.tok.reverse[i]) == 1 and han(self.tok.reverse[i]) for i in range(self.V)],
            device="cuda",
        )
        self.ping_mask = torch.tensor(
            [
                len(self.tok.reverse[i]) == 1 and tone(self.tok.reverse[i]) == "P"
                for i in range(self.V)
            ],
            device="cuda",
        )
        self.ids = torch.arange(self.V, device="cuda")

    def allowed(self, upper_ids, x, prefix_len, j, need_ping):
        b, k, t = x.shape
        allow = self.han_mask[None, None, :].expand(b, k, -1).clone()
        allow.scatter_(2, upper_ids[:, j, None, None].expand(-1, k, 1), False)
        for prev in range(j):
            eq = (upper_ids[:, prev] == upper_ids[:, j])[:, None, None]
            candidate_same = self.ids[None, None, :] == x[:, :, prefix_len + prev, None]
            allow &= torch.where(eq, candidate_same, ~candidate_same)
        # Propagate the final-tone requirement to all occurrences of the final upper class.
        ping = (need_ping & (upper_ids[:, j] == upper_ids[:, -1]))[:, None, None]
        allow &= (~ping) | self.ping_mask[None, None, :]
        return allow

    @torch.inference_mode()
    def candidates(self, uppers, k=16, nsamples=8):
        b = len(uppers)
        n = len(uppers[0])
        assert all(len(u) == n for u in uppers)
        upper_ids = torch.tensor([self.tok.encode(u) for u in uppers], device="cuda")
        prefix = torch.cat(
            [
                torch.full((b, 1), 2, device="cuda"),
                torch.full((b, 1), 4, device="cuda"),
                upper_ids,
                torch.full((b, 1), 5, device="cuda"),
            ],
            1,
        )
        p = prefix.shape[1]
        need_ping = torch.tensor([tone(u[-1]) == "Z" for u in uppers], device="cuda")
        x = prefix[:, None, :]
        scores = torch.zeros((b, 1), device="cuda")
        for j in range(n):
            oldk = x.shape[1]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = self.model(x.reshape(b * oldk, -1))[:, -1].float()
            lp = z.log_softmax(-1).reshape(b, oldk, self.V)
            lp = lp.masked_fill(~self.allowed(upper_ids, x, p, j, need_ping), float("-inf"))
            val, idx = (scores[:, :, None] + lp).reshape(b, -1).topk(k)
            assert torch.isfinite(val).all(), "No rule-feasible beam; do not silently relax rules"
            parent = idx // self.V
            token = idx % self.V
            x = torch.cat(
                [x.gather(1, parent[:, :, None].expand(-1, -1, x.shape[-1])), token[:, :, None]], 2
            )
            scores = val
        result = [[self.tok.decode(r[p:]) for r in group] for group in x.cpu().tolist()]
        if nsamples:
            x = prefix[:, None, :].expand(-1, nsamples, -1).clone()
            uniforms = np.stack(
                [
                    np.random.default_rng(
                        int(hashlib.sha256(("2036:" + u).encode()).hexdigest()[:16], 16)
                    ).random((nsamples, n))
                    for u in uppers
                ]
            )
            uniforms = torch.from_numpy(uniforms).float().cuda()
            for j in range(n):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    z = (
                        self.model(x.reshape(b * nsamples, -1))[:, -1]
                        .float()
                        .reshape(b, nsamples, -1)
                    )
                z = z.masked_fill(~self.allowed(upper_ids, x, p, j, need_ping), float("-inf"))
                values, indices = (z / 0.8).topk(20)
                cdf = values.softmax(-1).cumsum(-1)
                pos = (uniforms[:, :, j, None] > cdf).sum(-1).clamp(max=19)
                token = indices.gather(2, pos[:, :, None]).squeeze(2)
                x = torch.cat([x, token[:, :, None]], 2)
            for out, group in zip(result, x.cpu().tolist()):
                out.extend(self.tok.decode(r[p:]) for r in group)
        return [list(dict.fromkeys(group)) for group in result]

    @torch.inference_mode()
    def log_scores(self, model, uppers, lowers, reverse=False):
        result = []
        for start in range(0, len(uppers), 256):
            aa = uppers[start : start + 256]
            bb = lowers[start : start + 256]
            if reverse:
                aa, bb = bb, aa
            seqs = [
                [2, 4] + self.tok.encode(a) + [5] + self.tok.encode(b) + [3] for a, b in zip(aa, bb)
            ]
            width = max(len(s) - 1 for s in seqs)
            x = torch.zeros((len(seqs), width), device="cuda", dtype=torch.long)
            y = torch.full_like(x, -100)
            for i, (s, a) in enumerate(zip(seqs, aa)):
                x[i, : len(s) - 1] = torch.tensor(s[:-1], device="cuda")
                y[i, len(a) + 2 : len(s) - 1] = torch.tensor(s[len(a) + 3 :], device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = model(x)
            mask = y != -100
            loss = F.cross_entropy(z[mask].float(), y[mask], reduction="none")
            row = mask.nonzero()[:, 0]
            summed = torch.zeros(len(seqs), device="cuda").scatter_add_(0, row, loss)
            result.extend((-summed / mask.sum(1)).cpu().tolist())
        return result

    def scored_pools(self, uppers, pools):
        uu = []
        yy = []
        for u, pool in zip(uppers, pools):
            uu.extend([u] * len(pool))
            yy.extend(pool)
        fwd = self.log_scores(self.model, uu, yy)
        rev = (
            self.log_scores(self.reverse, uu, yy, True)
            if self.reverse is not None
            else [0.0] * len(yy)
        )
        result = []
        index = 0
        for u, pool in zip(uppers, pools):
            rows = []
            for y in pool:
                checks = rule_check(u, y)
                assert checks["encoded_rules_pass"]
                rows.append(
                    {
                        "text": y,
                        "forward_score": fwd[index],
                        "reverse_score": rev[index],
                        "lexical_score": lexical_score(u, y),
                        "checks": checks,
                    }
                )
                index += 1
            result.append(rows)
        return result

    @staticmethod
    def select(pool, reverse_weight=0.6, lexical_weight=0.5):
        return max(
            pool,
            key=lambda p: (
                p["forward_score"]
                + reverse_weight * p["reverse_score"]
                + lexical_weight * p["lexical_score"]
            ),
        )

    @torch.inference_mode()
    def polish_candidates(self, uppers, selected, topk=3):
        result = []
        for u, p in zip(uppers, selected):
            y = p["text"]
            prefix = [2, 4] + self.tok.encode(u) + [5]
            x = torch.tensor([prefix + self.tok.encode(y)], device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = self.model(x)[0].float()
            candidates = [y]
            seen = set()
            for j, c in enumerate(u):
                if c in seen:
                    continue
                seen.add(c)
                positions = [i for i, a in enumerate(u) if a == c]
                allowed = self.han_mask.clone()
                allowed[self.tok.vocab[c]] = False
                for i, a in enumerate(y):
                    if i not in positions:
                        allowed[self.tok.vocab[a]] = False
                if tone(u[-1]) == "Z" and u[-1] == c:
                    allowed &= self.ping_mask
                score = z[len(prefix) - 1 + j].masked_fill(~allowed, float("-inf"))
                for v in score.topk(topk).indices.tolist():
                    chars = list(y)
                    for i in positions:
                        chars[i] = self.tok.reverse[v]
                    text = "".join(chars)
                    if rule_check(u, text)["encoded_rules_pass"]:
                        candidates.append(text)
            result.append(list(dict.fromkeys(candidates)))
        return result
