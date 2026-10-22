"""Open a policy and benchmark folder in a cyclotron harness."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/cyclotron"))

from cyclotron.evaluation.cli import entry

if __name__ == "__main__":
    sys.exit(entry(ROOT, viewing=True))
