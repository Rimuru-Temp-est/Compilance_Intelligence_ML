"""Fast strict-timestamp behavioral lookup from prepared NumPy arrays."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class BehaviorHistoryIndex:
    """Read-only index of cumulative per-entity behavior at each timestamp."""

    def __init__(self, arrays: dict[str, np.ndarray]):
        self.entity_ids = arrays["entity_ids"].astype(str)
        self._entity_codes = {entity_id: i for i, entity_id in enumerate(self.entity_ids)}
        self._roles: dict[str, dict[str, np.ndarray]] = {}
        for role in ("sender", "receiver"):
            self._roles[role] = {
                name: arrays[f"{name}_{role}"]
                for name in ("offsets", "timestamps", "txns", "received", "paid", "unique_counterparties")
            }
            values = self._roles[role]
            if len(values["offsets"]) != len(self.entity_ids) + 1:
                raise ValueError(f"Malformed {role} behavioral offsets")
            if values["offsets"][-1] != len(values["timestamps"]):
                raise ValueError(f"Malformed {role} behavioral timestamp index")

    @classmethod
    def load(cls, path: Path) -> "BehaviorHistoryIndex":
        if not path.is_file():
            raise FileNotFoundError(f"Prepared behavioral history not found: {path}")
        try:
            with np.load(path, allow_pickle=False) as source:
                arrays = {name: source[name] for name in source.files}
        except Exception as exc:
            raise ValueError(f"Unable to load prepared behavioral history: {path}") from exc
        return cls(arrays)

    def _cumulative(self, role: str, entity_id: str, timestamp_ns: int) -> tuple[float, float, float, float]:
        code = self._entity_codes.get(str(entity_id))
        if code is None:
            return 0.0, 0.0, 0.0, 0.0
        values = self._roles[role]
        start, stop = int(values["offsets"][code]), int(values["offsets"][code + 1])
        if start == stop:
            return 0.0, 0.0, 0.0, 0.0
        local = int(np.searchsorted(values["timestamps"][start:stop], timestamp_ns, side="left"))
        position = start + local - 1
        if position < start:
            return 0.0, 0.0, 0.0, 0.0
        return tuple(float(values[name][position]) for name in ("txns", "received", "paid", "unique_counterparties"))

    def cumulative_for_rows(
        self, entity_ids: np.ndarray, timestamps_ns: np.ndarray, role: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized strict-prior rollup lookup for offline training rows."""
        if role not in self._roles:
            raise ValueError(f"Unknown behavioral role: {role}")
        ids = np.asarray(entity_ids, dtype=str)
        times = np.asarray(timestamps_ns, dtype=np.int64)
        if ids.shape != times.shape:
            raise ValueError("entity_ids and timestamps_ns must have matching shapes")
        codes = np.fromiter(
            (self._entity_codes.get(value, -1) for value in ids),
            dtype=np.int32,
            count=len(ids),
        )
        if (codes < 0).any():
            raise ValueError("Training rows contain entity IDs absent from behavioral history")
        order = np.argsort(codes, kind="stable")
        sorted_codes = codes[order]
        boundaries = np.flatnonzero(np.r_[True, sorted_codes[1:] != sorted_codes[:-1], True])
        outputs = [np.zeros(len(ids), dtype=np.float64) for _ in range(4)]
        history = self._roles[role]
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            code = int(sorted_codes[left])
            start = int(history["offsets"][code])
            stop = int(history["offsets"][code + 1])
            if start == stop:
                continue
            row_positions = order[left:right]
            local = np.searchsorted(
                history["timestamps"][start:stop], times[row_positions], side="left"
            ) - 1
            valid = local >= 0
            if not valid.any():
                continue
            source_positions = start + local[valid]
            target_positions = row_positions[valid]
            for output, name in zip(outputs, ("txns", "received", "paid", "unique_counterparties")):
                output[target_positions] = history[name][source_positions]
        return tuple(outputs)

    def features(self, sender_entity_id: str, receiver_entity_id: str, timestamp: Any) -> dict[str, float]:
        timestamp_ns = int(np.datetime64(timestamp, "ns").astype(np.int64))
        sender = self._cumulative("sender", sender_entity_id, timestamp_ns)
        receiver = self._cumulative("receiver", receiver_entity_id, timestamp_ns)
        return {
            "log_sender_prior_txns": float(np.log1p(sender[0])),
            "log_sender_prior_received": float(np.log1p(sender[1])),
            "log_sender_prior_paid": float(np.log1p(sender[2])),
            "log_receiver_prior_txns": float(np.log1p(receiver[0])),
            "log_receiver_prior_received": float(np.log1p(receiver[1])),
            "log_receiver_prior_paid": float(np.log1p(receiver[2])),
            "log_sender_prior_unique_receivers": float(np.log1p(sender[3])),
            "log_receiver_prior_unique_senders": float(np.log1p(receiver[3])),
        }
