from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def slugify_symbol(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_")


def save_json(payload: dict[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    return output_path


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def update_latest_pointer(
    output_root: str | Path,
    run_dir: str | Path,
    *,
    summary_name: str = "summary_ko.md",
    manifest_name: str = "run_manifest.json",
) -> dict[str, str]:
    output_root = Path(output_root)
    run_dir = Path(run_dir)
    latest_link = output_root / "latest"
    latest_json = output_root / "latest_run.json"

    pointer_payload = {
        "latest": str(run_dir),
        "summary_ko_md": str(run_dir / summary_name),
        "manifest_json": str(run_dir / manifest_name),
    }
    save_json(pointer_payload, latest_json)

    if latest_link.exists() or latest_link.is_symlink():
        if latest_link.is_symlink() or latest_link.is_file():
            latest_link.unlink()
        else:
            return {
                "latest_json": str(latest_json),
            }

    relative_target = os.path.relpath(run_dir, output_root)
    os.symlink(relative_target, latest_link, target_is_directory=True)
    return {
        "latest_symlink": str(latest_link),
        "latest_json": str(latest_json),
    }


def timestamped_run_dir(output_root: str | Path, symbol: str) -> Path:
    output_root = Path(output_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{slugify_symbol(symbol)}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
