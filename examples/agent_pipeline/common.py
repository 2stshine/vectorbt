from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def slugify_symbol(symbol: str) -> str:
    return symbol.replace("/", "_")


def save_json(payload: dict[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    return output_path


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def timestamped_run_dir(output_root: str | Path, symbol: str) -> Path:
    output_root = Path(output_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{slugify_symbol(symbol)}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
