from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_DIR = Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    return REPO_DIR


def serialize_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [serialize_json(item) for item in value]
    if isinstance(value, dict):
        return {str(key): serialize_json(val) for key, val in value.items()}
    return value


def get_git_commit_hash() -> str | None:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return output.strip()
    except Exception:
        return None


def run_command(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"[repro] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd or repo_root())


def ensure_file_exists(path: Path) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Expected file was not created: {path}")


def ensure_dir_exists(path: Path) -> None:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Expected directory was not created: {path}")


def write_run_summary(summary_path: Path, args: Any, extra: dict[str, Any] | None = None) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit_hash(),
        "argv": sys.argv,
        "args": serialize_json(vars(args)),
    }
    if extra:
        payload.update(serialize_json(extra))

    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
