"""No-weight checks for the public package and retained experimental evidence."""

import ast
import importlib
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class RepositoryTests(unittest.TestCase):
    def test_package_root_and_entrypoint(self):
        paths = importlib.import_module("couplet_gpt.paths")
        cli = importlib.import_module("couplet_gpt.inference.cli")
        self.assertEqual(paths.ROOT, ROOT)
        self.assertTrue(callable(cli.main))

    def test_all_local_imports_resolve(self):
        for p in (ROOT / "couplet_gpt").rglob("*.py"):
            for node in ast.walk(ast.parse(p.read_text(encoding="utf8"))):
                names = (
                    [node.module]
                    if isinstance(node, ast.ImportFrom)
                    else [n.name for n in node.names]
                    if isinstance(node, ast.Import)
                    else []
                )
                for name in names:
                    if name and name.startswith("couplet_gpt."):
                        target = ROOT / Path(*name.split("."))
                        self.assertTrue(
                            target.with_suffix(".py").exists() or (target / "__init__.py").exists(),
                            (p, name),
                        )

    def test_model_modes_share_weights(self):
        configs = [
            json.loads((ROOT / f"configs/couplet_{mode}.json").read_text(encoding="utf8"))
            for mode in ("rules", "unconstrained")
        ]
        self.assertEqual(configs[0]["forward"], configs[1]["forward"])
        self.assertEqual(configs[0]["reverse"], configs[1]["reverse"])
        self.assertTrue(configs[0]["force_equal_length"])
        self.assertFalse(configs[1]["force_equal_length"])

    def test_fourteen_semantically_named_results(self):
        data = json.loads(
            (ROOT / "results/couplet/ablation/metrics.json").read_text(encoding="utf8")
        )["results"]
        keys = [k for k in data if not k.startswith("core/")]
        self.assertEqual(len(keys), 14)
        self.assertFalse(any(re.search(r"R[0-9]_", k) for k in keys))
        for mode in ("forward_only", "bidirectional"):
            self.assertEqual(data["unconstrained/" + mode]["unique_uppers"], 2617)

    def test_reader_document_links(self):
        for p in [ROOT / "README.md", *(ROOT / "docs").glob("*.md")]:
            for target in (
                part.split(")", 1)[0] for part in p.read_text(encoding="utf8").split("](")[1:]
            ):
                if "://" in target or target.startswith("#"):
                    continue
                target = target.split("#")[0]
                self.assertTrue((p.parent / target).exists(), (p, target))


if __name__ == "__main__":
    unittest.main()
