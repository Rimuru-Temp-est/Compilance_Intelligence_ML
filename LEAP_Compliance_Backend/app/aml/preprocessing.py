"""Notebook-aligned transaction feature construction and encoding."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

import numpy as np
import pandas as pd

CATEGORICAL_COLUMNS = ("receiving_currency", "payment_currency", "payment_format")
NUMERIC_COLUMNS = (
    "log_amount_received",
    "log_amount_paid",
    "amount_difference",
    "amounts_equal",
    "log_amount_ratio",
    "hour",
    "day_of_week",
)
BEHAVIORAL_COLUMNS = (
    "log_sender_prior_txns",
    "log_sender_prior_received",
    "log_sender_prior_paid",
    "log_receiver_prior_txns",
    "log_receiver_prior_received",
    "log_receiver_prior_paid",
    "log_sender_prior_unique_receivers",
    "log_receiver_prior_unique_senders",
)


def ordered_feature_names(metadata: Mapping[str, Any]) -> list[str]:
    groups = metadata["feature_groups"]
    names = list(groups["transaction_features"]) + list(groups["behavioral_features"])
    if len(names) != int(groups["total_features"]):
        raise ValueError("Metadata feature groups do not match total_features")
    if len(names) != len(set(names)):
        raise ValueError("Metadata feature names are not unique")
    if len(names) != int(metadata["model_configuration"]["n_features"]):
        raise ValueError("Metadata feature count disagrees with model configuration")
    return names



def derive_numeric_features(amount_received: Any, amount_paid: Any, timestamps: Any) -> dict[str, np.ndarray]:
    """Vectorized numeric transforms shared by training and inference."""
    received = np.asarray(amount_received, dtype=np.float64)
    paid = np.asarray(amount_paid, dtype=np.float64)
    if received.shape != paid.shape:
        raise ValueError("amount_received and amount_paid must have matching shapes")
    if not np.isfinite(received).all() or (received < 0).any():
        raise ValueError("amount_received must be finite and non-negative")
    if not np.isfinite(paid).all() or (paid <= 0).any():
        raise ValueError("amount_paid must be finite and greater than zero for the training ratio transform")
    dt = pd.to_datetime(pd.Series(timestamps), utc=True, errors="raise")
    if len(dt) != len(received):
        raise ValueError("timestamps must align with amount arrays")
    result = {
        "log_amount_received": np.log1p(received),
        "log_amount_paid": np.log1p(paid),
        "amount_difference": received - paid,
        "amounts_equal": (received == paid).astype(np.float64),
        "log_amount_ratio": np.log1p(received / paid),
        "hour": dt.dt.hour.to_numpy(dtype=np.float64),
        "day_of_week": dt.dt.dayofweek.to_numpy(dtype=np.float64),
    }
    return result


def _as_utc_naive(value: Any) -> datetime:
    """Normalize ISO datetimes to UTC-naive, matching the source timeline."""
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is not None:
        from datetime import timezone
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _one_hot_values(
    transaction: Mapping[str, Any], transaction_feature_names: list[str]
) -> list[float]:
    values: list[float] = []
    for column in CATEGORICAL_COLUMNS:
        prefix = f"{column}_"
        categories = [name for name in transaction_feature_names if name.startswith(prefix)]
        if not categories:
            raise ValueError(f"No encoded categories found for {column}")
        selected_name = prefix + str(transaction[column]).strip()
        # OneHotEncoder(handle_unknown='ignore') maps unknown categories to all zeros.
        values.extend(1.0 if name == selected_name else 0.0 for name in categories)
    return values


def build_feature_vector(
    transaction: Mapping[str, Any],
    behavioral_values: Mapping[str, float],
    feature_names: list[str],
) -> np.ndarray:
    """Return the exact 52-wide, training-order float32 feature vector."""
    missing = set(CATEGORICAL_COLUMNS) | {
        "amount_received", "amount_paid", "timestamp"
    }
    missing -= set(transaction)
    if missing:
        raise ValueError(f"Missing transaction feature inputs: {sorted(missing)}")
    missing_behavior = set(BEHAVIORAL_COLUMNS) - set(behavioral_values)
    if missing_behavior:
        raise ValueError(f"Missing behavioral features: {sorted(missing_behavior)}")

    numeric = derive_numeric_features(
        [transaction["amount_received"]],
        [transaction["amount_paid"]],
        [_as_utc_naive(transaction["timestamp"])],
    )
    derived = {name: float(values[0]) for name, values in numeric.items()}
    tx_names = feature_names[:len(feature_names) - len(BEHAVIORAL_COLUMNS)]
    categorical_count = sum(
        sum(name.startswith(f"{column}_") for name in tx_names)
        for column in CATEGORICAL_COLUMNS
    )
    expected_numeric = tx_names[categorical_count:]
    if expected_numeric != list(NUMERIC_COLUMNS):
        raise ValueError(f"Unexpected numeric feature order in metadata: {expected_numeric}")
    values = _one_hot_values(transaction, tx_names)
    values.extend(derived[name] for name in expected_numeric)
    values.extend(float(behavioral_values[name]) for name in BEHAVIORAL_COLUMNS)
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (len(feature_names),):
        raise ValueError(f"Feature vector shape {result.shape} != expected {(len(feature_names),)}")
    if not np.isfinite(result).all():
        raise ValueError("Feature vector contains NaN or infinite values")
    return result
