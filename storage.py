import os
import json
from .config import get_settings

S = get_settings()

# Ensure dirs exist
for d in [S.VAR_DIR, S.TEXT_DIR, S.TABLE_DIR]:
    os.makedirs(d, exist_ok=True)


def save_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
