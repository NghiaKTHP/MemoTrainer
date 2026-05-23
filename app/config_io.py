import ast
import dataclasses
import json
from datetime import datetime
from enum import Enum
from pathlib import Path

import yaml


def config_to_dict(cfg) -> dict:
    result = {}
    for f in dataclasses.fields(cfg):
        if f.name.startswith("_"):
            continue
        val = getattr(cfg, f.name)
        if isinstance(val, Enum):
            val = val.name
        elif isinstance(val, tuple):
            val = list(val)
        result[f.name] = val
    return result


def apply_dict_to_config(cfg, data: dict, arch_enum_cls=None):
    for key, val in data.items():
        if not hasattr(cfg, key) or key.startswith("_"):
            continue
        current = getattr(cfg, key)
        if isinstance(current, Enum) or (key == "Architecture" and arch_enum_cls):
            enum_cls = arch_enum_cls if (key == "Architecture" and arch_enum_cls) else type(current)
            if isinstance(val, str):
                try:
                    val = enum_cls[val]
                except KeyError:
                    pass
            elif isinstance(val, int):
                try:
                    val = enum_cls(val)
                except ValueError:
                    pass
        elif isinstance(current, tuple) and isinstance(val, list):
            val = tuple(val)
        elif current is None and isinstance(val, str) and val == "None":
            val = None
        try:
            setattr(cfg, key, val)
        except Exception:
            pass


def save_config(cfg, file_path: str, extra: dict = None):
    data = config_to_dict(cfg)
    if extra:
        data.update(extra)
    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix in (".yaml", ".yml"):
        with open(p, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    else:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def load_config_file(file_path: str) -> dict:
    p = Path(file_path)
    if p.suffix in (".yaml", ".yml"):
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    else:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)


def suggest_filename(model_name: str, dataset_hint: str = "") -> str:
    date_str = datetime.now().strftime("%Y%m%d")
    parts = [p for p in [model_name, dataset_hint, date_str] if p]
    return "_".join(parts) + ".yaml"
