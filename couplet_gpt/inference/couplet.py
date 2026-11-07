"""Final variable-length ChineseBERT generator with v4 constraints and forward/reverse ranking."""

import argparse, json
import torch, numpy as np
from couplet_gpt.data.couplet_splits import OUT, ROOT, read, normalize, MAX_LENGTH
from couplet_gpt.modeling.couplet import initialize
from couplet_gpt.modeling.alignment import load_model_v2
from couplet_gpt.inference.search import V2Engine
from couplet_gpt.inference.constraints import V4Engine
from couplet_gpt.resources.prosody import ProsodyV4
from couplet_gpt.resources.lexical_semantics import LexicalGrammarV4
from couplet_gpt.inference.checks import rule_check_v3
from couplet_gpt.evaluation.metrics import han
from couplet_gpt.modeling.gpt import generate


class FinalEngine(V4Engine):
    def __init__(self, forward, reverse=None, mode="modern_context"):
        V2Engine.__init__(self, forward, reverse)
        self.prosody = ProsodyV4()
        self.mode = mode
        self.lexical = LexicalGrammarV4()
        self.possible_ping = torch.tensor(
            [
                len(self.tok.reverse[i]) == 1
                and self.prosody.ending_tone(
                    self.tok.reverse[i], "pingshui" if mode == "pingshui" else "modern_character"
                )
                != "Z"
                for i in range(self.V)
            ],
            device="cuda",
        )

    allowed = V4Engine.allowed
    scored_pools = V4Engine.scored_pools


def load(mode="modern_context"):
    f, _ = initialize(OUT / "forward/best.pt")
    f.eval()
    ck = torch.load(OUT / "reverse/best.pt", map_location="cpu", weights_only=True)
    r = load_model_v2(ck)
    return FinalEngine(f, r, mode)


def respond(engine, upper, raw=False, beam=16, samples=8):
    upper = normalize(upper).strip()
    n = len(upper)
    if not upper or not all(han(c) for c in upper):
        raise ValueError("请输入纯汉字单句；多分句请逐句提交，不能删除标点后混成一联。")
    if n > MAX_LENGTH:
        raise ValueError(
            f"已解除五七言限制；当前128-token模型每联最多{MAX_LENGTH}字，不会静默截断。"
        )
    if 1 in engine.tok.encode(upper):
        raise ValueError("存在当前字符词表不支持的字，不能静默替换。")
    if raw:
        ids = generate(engine.model, [2, 4] + engine.tok.encode(upper) + [5], max_new=n + 4)
        lower = engine.tok.decode([i for i in ids if i != 3])
        count = 1
    else:
        candidates = engine.candidates([upper], k=beam, nsamples=samples)
        pool = engine.scored_pools([upper], candidates)[0]
        # Stable v4 default: neural forward + reverse scores; no unvalidated phrase/allusion reranking.
        chosen = max(pool, key=lambda r: r["forward_score"] + r["reverse_score"])
        lower = chosen["text"]
        count = len(pool)
    return {
        "upper": upper,
        "lower": lower,
        "length": n,
        "candidate_count": count,
        "forced_length": not raw,
        "model": "ChineseBERT+lexical_adapter_variable_length_final_v7",
        "decoding": "raw_greedy" if raw else "v4_contextual_constraints_forward_reverse_rank",
        "checks": rule_check_v3(upper, lower),
        "prosody": engine.prosody.inspect(upper, lower, mode=engine.mode),
        "lexical_evidence": engine.lexical.compact_features(upper, lower),
        "allusion_generation_disabled": True,
        "syntax_attention_disabled": True,
        "training_length_note": "训练语料覆盖1—14及18字；15—17字及更长长度证据不足，能接受输入不代表质量已验证。",
        "full_literary_quality_verified": False,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--upper", required=True)
    p.add_argument("--raw", action="store_true")
    p.add_argument("--beam", type=int, default=16)
    p.add_argument("--samples", type=int, default=8)
    a = p.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(respond(load(), a.upper, a.raw, a.beam, a.samples), ensure_ascii=False))
