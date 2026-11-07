"""Verify final model artifacts using the maintained package."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from couplet_gpt.evaluation.verify import main

if __name__ == "__main__":
    main()
