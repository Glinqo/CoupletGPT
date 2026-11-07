"""Verify retained artifacts without loading models or starting training."""

from couplet_gpt.paths import ROOT as PROJECT_ROOT
import argparse, hashlib, json
from pathlib import Path

ROOT = PROJECT_ROOT


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--require-weights", action="store_true")
    a = p.parse_args()
    manifest = json.loads((ROOT / "models/manifest.json").read_text(encoding="utf8"))
    missing = []
    for r in manifest["models"]:
        path = ROOT / r["path"]
        if not path.exists():
            missing.append(r["path"])
            continue
        assert path.stat().st_size == r["bytes"], path
        assert digest(path) == r["sha256"], path
    configs = [
        json.loads((ROOT / f"configs/couplet_{mode}.json").read_text(encoding="utf8"))
        for mode in ("rules", "unconstrained")
    ]
    assert (
        configs[0]["forward"] == configs[1]["forward"]
        and configs[0]["reverse"] == configs[1]["reverse"]
    )
    assert configs[0]["force_equal_length"] and not configs[1]["force_equal_length"]
    assert len(manifest["models"]) == 4
    for name in [
        "couplet_gpt/inference/cli.py",
        "tokenizer/vocab.json",
        "models/poetry_appreciation/vocab.json",
        "data/resources/prosody/Pingshui_Rhyme.json",
        "data/resources/lexical_grammar/lexicon.json.gz",
    ]:
        assert (ROOT / name).exists(), name
    ext = ROOT / "models/external/ChineseBERT-base"
    external_missing = []
    for r in json.loads((ROOT / "models/external/manifest.json").read_text(encoding="utf8"))[
        "files"
    ]:
        path = ext / r["file"]
        if not path.exists():
            external_missing.append(r["file"])
        elif "sha256" in r:
            assert digest(path) == r["sha256"], path
    if a.require_weights:
        assert not missing and not external_missing, (missing, external_missing)
    print(
        json.dumps(
            {
                "status": "pass",
                "local_models_verified": 4 - len(missing),
                "missing_weights": missing,
                "missing_external_files": external_missing,
                "shared_couplet_weights": True,
                "archive_required_for_formal_inference": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
