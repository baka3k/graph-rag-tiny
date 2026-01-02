import os
from pathlib import Path
from typing import Tuple


def _offline_enabled() -> bool:
    offline = os.getenv("HF_HUB_OFFLINE") or os.getenv("TRANSFORMERS_OFFLINE")
    if offline and offline.strip().lower() not in {"0", "false", "no"}:
        return True
    local_only = os.getenv("EMBEDDING_LOCAL_ONLY")
    if local_only and local_only.strip().lower() not in {"0", "false", "no"}:
        return True
    return False


def resolve_embedding_model(model: str | None, default: str) -> Tuple[str, bool]:
    if model:
        selected = model
    else:
        selected = os.getenv("EMBEDDING_MODEL_PATH") or os.getenv("EMBEDDING_MODEL") or default
    local_files_only = Path(selected).exists()
    if local_files_only or _offline_enabled():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    return selected, local_files_only
