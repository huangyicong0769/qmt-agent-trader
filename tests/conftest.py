from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATEWAY_SRC = ROOT / "gateway" / "windows_qmt_gateway" / "src"
sys.path.insert(0, str(GATEWAY_SRC))
