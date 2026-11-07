"""Portable entry point for the three retained models; no archive is required."""

from couplet_gpt.paths import ROOT as PROJECT_ROOT
import argparse, json, os, sys
from pathlib import Path

ROOT = PROJECT_ROOT


def configure_resources():
    os.environ.setdefault(
        "COUPLET_V5_CHINESEBERT_MODEL", str(ROOT / "models/external/ChineseBERT-base")
    )


def couplet(args):
    import torch
    from couplet_gpt.data.datasets import CharacterTokenizer
    from couplet_gpt.training.couplet_common import load as load_forward
    from couplet_gpt.training.reverse_common import load_reverse
    from couplet_gpt.inference.bidirectional import CalibratedEngine, respond
    from couplet_gpt.evaluation.ablation import Search, TokenSafeContexts, token_scores
    from couplet_gpt.evaluation.metrics import han
    from couplet_gpt.inference.checks import rule_check_v3

    upper = args.text.strip()
    tok = CharacterTokenizer()
    if not upper or len(upper) > 62 or not all(han(c) for c in upper):
        raise ValueError("上联须为1—62字纯汉字单句。")
    if 1 in tok.encode(upper):
        raise ValueError("上联含当前字符表未覆盖的字。")
    model, _ = load_forward(ROOT / "models/couplet/forward.pt")
    reverse = None
    weight = 0.0 if args.forward_only else 0.25
    if weight:
        reverse, _ = load_reverse(path=ROOT / "models/couplet/reverse.pt")
    if args.mode == "rules":
        engine = CalibratedEngine(model, reverse, reverse_weight=weight)
        return respond(engine, upper)
    search = Search(model)
    pool = search.candidates([upper], False)[0][0]
    ff = token_scores(model, [upper] * len(pool), [p["tokens"] for p in pool])
    rr = [0.0] * len(pool)
    if reverse is not None:
        reverse.context_provider = TokenSafeContexts()
        if model.context_provider.encoder is not None:
            reverse.context_provider.encoder = model.context_provider.encoder
            reverse.context_provider.tokenizer = model.context_provider.tokenizer
        rr = token_scores(reverse, [upper] * len(pool), [p["tokens"] for p in pool], True)
    j = max(range(len(pool)), key=lambda i: ff[i] + weight * rr[i])
    p = pool[j]
    return {
        "upper": upper,
        "lower": p["text"],
        "mode": "unconstrained_search",
        "reverse_weight": weight,
        "candidate_count": len(pool),
        "ended_naturally": p["ended_naturally"],
        "truncated": p["truncated"],
        "checks": rule_check_v3(upper, p["text"]),
        "checks_used_for_selection": False,
        "full_literary_quality_verified": False,
    }


def poetry(args):
    import torch
    import couplet_gpt.data.appreciation_prompts as survey

    # The vocabulary travels with the public model package, not the ignored corpus.
    survey.OUT = ROOT / "models/poetry_appreciation"
    import couplet_gpt.training.appreciation_runtime as runtime

    runtime.OUT = survey.OUT
    from couplet_gpt.resources.prosody_annotation import ToneAnnotator

    a = ToneAnnotator().annotate(args.text)
    row = {
        "sample_id": "interactive",
        "source_text": args.text,
        "program_first_sentence": a["program_first_sentence"],
        "kind": "poetry",
    }
    row["program_first_sentence"] = survey.first_sentence(row)
    tok = runtime.Tokenizer()
    ck = torch.load(
        ROOT / "models/poetry_appreciation/model.pt", map_location="cpu", weights_only=True
    )
    m = runtime.GPT(runtime.GPTConfig(**ck["model_config"])).cuda().eval()
    m.load_state_dict(ck["model"])
    if (
        len(tok.encode(survey.prompt(args.text, row["program_first_sentence"]))) + 1 + args.max_new
        > m.config.context_length
    ):
        raise ValueError("Input plus generation budget exceeds context.")
    return runtime.decode(m, [row], tok, max_new=args.max_new)[0]


def pretraining(args):
    import torch
    from couplet_gpt.modeling.gpt import GPT, GPTConfig, generate
    from couplet_gpt.data.datasets import CharacterTokenizer

    tok = CharacterTokenizer()
    ids = tok.encode(args.text)
    if 1 in ids or len(ids) > 126:
        raise ValueError("Prompt is too long or has unsupported characters.")
    ck = torch.load(ROOT / "models/pretraining/model.pt", map_location="cpu", weights_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = GPT(GPTConfig(**ck["model_config"])).to(device).eval()
    m.load_state_dict(ck["model"])
    out = generate(m, [2] + ids, max_new=args.max_new)
    return {"prompt": args.text, "continuation": tok.decode([i for i in out if i != 3])}


def main():
    sys.stdout.reconfigure(encoding="utf8")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=["pretraining", "couplet", "poetry"])
    p.add_argument("--text", required=True)
    p.add_argument("--mode", choices=["rules", "unconstrained"], default="unconstrained")
    p.add_argument("--forward-only", action="store_true")
    p.add_argument("--max-new", type=int, default=512)
    args = p.parse_args()
    if args.max_new < 1:
        p.error("--max-new must be positive")
    import torch

    torch.set_num_threads(3)
    configure_resources()
    result = {"pretraining": pretraining, "couplet": couplet, "poetry": poetry}[args.task](args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
