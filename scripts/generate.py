"""Compatibility command; the maintained implementation is in couplet_gpt."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from couplet_gpt.inference.cli import main

if __name__ == "__main__":
    main()
