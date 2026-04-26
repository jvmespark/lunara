"""tests/python/conftest.py
Shared pytest fixtures.
"""

import sys
from pathlib import Path

# Ensure repo root is importable
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
