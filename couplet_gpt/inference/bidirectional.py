"""Deployed symmetric adapter pair with validation-calibrated reverse weight."""

import argparse, sys
from couplet_gpt.training.reverse_common import *
from couplet_gpt.training.couplet_common import load as load_forward
from couplet_gpt.inference.couplet import FinalEngine, respond as legacy_respond


class CalibratedEngine(FinalEngine):
    def __init__(self, forward, reverse, mode="modern_context", reverse_weight=1.0):
        super().__init__(forward, reverse, mode)
        self.reverse_weight = reverse_weight

    def share_encoder(self):
        a = self.model.context_provider
        b = self.reverse.context_provider
        if a.encoder is not None and b.encoder is None:
            b.encoder = a.encoder
            b.tokenizer = a.tokenizer
        elif b.encoder is not None and a.encoder is None:
            a.encoder = b.encoder
            a.tokenizer = b.tokenizer

    def scored_pools(self, uppers, pools):
        if self.reverse_weight == 0:
            saved = self.reverse
            self.reverse = None
            try:
                return super().scored_pools(uppers, pools)
            finally:
                self.reverse = saved
        self.share_encoder()
        # After candidate generation the forward provider may have initialized the shared encoder.
        result = super().scored_pools(uppers, pools)
        self.share_encoder()
        # The legacy respondent adds these two fields; scale reverse explicitly here.
        for pool in result:
            for row in pool:
                row["raw_reverse_score"] = row["reverse_score"]
                row["reverse_score"] *= self.reverse_weight
        return result


def load(mode="modern_context"):
    release = read(OUT / "RELEASE.json")
    assert sha(FORWARD) == release["forward_sha256"]
    f, _ = load_forward(FORWARD)
    r, _ = load_reverse(path=Path(release["reverse_checkpoint"]))
    return CalibratedEngine(f, r, mode, release["reverse_weight"])


def respond(engine, upper, raw=False):
    result = legacy_respond(engine, upper, raw)
    result.update(
        model="CoupletGPT_final_v12_symmetric_ChineseBERT_lexical_adapters",
        reverse_weight=engine.reverse_weight,
        reverse_enabled=(not raw and engine.reverse_weight > 0),
        reverse_model="lower_to_upper_ChineseBERT_lexical_adapter_v12",
        training_length_note="正反向均使用词汇适配器；每联支持1—62字纯汉字单句，长联质量未获人工认证。",
    )
    if not raw:
        result["decoding"] = (
            "v4_constraints_validation_calibrated_bidirectional"
            if engine.reverse_weight > 0
            else "v4_constraints_forward_only"
        )
    return result


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf8")
    p = argparse.ArgumentParser()
    p.add_argument("--upper")
    p.add_argument("--raw", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(4)
    upper = a.upper if a.upper is not None else input("请输入上联：").strip()
    print(json.dumps(respond(load(), upper, a.raw), ensure_ascii=False))
