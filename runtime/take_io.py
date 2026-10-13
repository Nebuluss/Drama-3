"""Portable Drama 3 takes: JSON metadata and integer VQ arrays, never pickle."""
import json
from pathlib import Path

import numpy as np


def save_take(path, take):
    arrays = {}
    metadata = {"version": 1, "revision": take["revision"], "turns": [], "voices": []}
    for group in ("turns", "voices"):
        for index, item in enumerate(take.get(group, [])):
            key = f"{group}_{index}"
            codes = np.asarray(item["codes"])
            validate_codes(codes)
            arrays[key] = codes.astype(np.int16)
            metadata[group].append({k: v for k, v in item.items() if k != "codes"} | {"key": key})
    arrays["metadata"] = np.frombuffer(json.dumps(metadata).encode("utf-8"), dtype=np.uint8)
    with Path(path).open("wb") as file:
        np.savez_compressed(file, **arrays)


def validate_codes(codes):
    if codes.ndim != 2 or codes.shape[0] != 10 or not codes.shape[1]:
        raise ValueError("A Drama 3 take requires 10 nonempty VQ codebooks.")
    if not np.issubdtype(codes.dtype, np.integer) or codes.min() < 0 or codes.max() >= 4096:
        raise ValueError("Invalid Drama 3 VQ codes (expected integers 0–4095).")


def load_take(path):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata"].tobytes().decode("utf-8"))
        if metadata["version"] != 1:
            raise ValueError("Unsupported Drama 3 take version.")
        take = {"revision": metadata["revision"]}
        for group in ("turns", "voices"):
            take[group] = []
            for item in metadata[group]:
                item = dict(item)
                codes = archive[item.pop("key")].copy()
                validate_codes(codes)
                take[group].append(item | {"codes": codes})
    return take
