"""Precompute strict-prior behavioral rollups from the source transactions."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.config import settings
from app.data.prepare_master_data import normalize_bank_id

SOURCE_COLUMNS = {
    "Timestamp", "From Bank", "Account", "To Bank", "Account.1",
    "Amount Received", "Amount Paid",
}
SEPARATOR = "\x1f"


def _normalize_bank_series(values: pd.Series) -> pd.Series:
    values = values.astype("string").str.strip()
    return values.str.replace(r"^0+(?=\d+$)", "", regex=True).replace("", "0")


def _role_rollup(frame: pd.DataFrame, entity_column: str, counterparty_column: str) -> dict[str, np.ndarray]:
    grouped = (
        frame.groupby([entity_column, "timestamp_ns"], sort=True, observed=True)
        .agg(
            txns=(counterparty_column, "size"),
            received=("amount_received", "sum"),
            paid=("amount_paid", "sum"),
            unique_counterparties=(counterparty_column, "nunique"),
        )
        .reset_index()
    )
    # Notebook 3 cumulatively adds each timestamp's distinct counterparties.
    # Preserve that exact behavior, including its per-timestamp aggregation.
    for column in ("txns", "received", "paid", "unique_counterparties"):
        grouped[column] = grouped.groupby(entity_column, sort=False)[column].cumsum()
    codes = grouped[entity_column].to_numpy(dtype=np.int32)
    counts = np.bincount(codes, minlength=frame.attrs["entity_count"])
    offsets = np.empty(len(counts) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    return {
        "offsets": offsets,
        "timestamps": grouped["timestamp_ns"].to_numpy(dtype=np.int64),
        "txns": grouped["txns"].to_numpy(dtype=np.int64),
        "received": grouped["received"].to_numpy(dtype=np.float64),
        "paid": grouped["paid"].to_numpy(dtype=np.float64),
        "unique_counterparties": grouped["unique_counterparties"].to_numpy(dtype=np.int64),
    }


def build_behavior_history(transactions_path: Path, accounts_path: Path) -> dict[str, np.ndarray]:
    if not transactions_path.is_file():
        raise FileNotFoundError(f"Transactions source not found: {transactions_path}")
    accounts = pd.read_csv(
        accounts_path,
        dtype={"bank_id": "string", "account_id": "string", "entity_id": "string"},
        keep_default_na=False,
    )
    account_key = (
        accounts["bank_id"].astype(str).str.strip()
        + SEPARATOR
        + accounts["account_id"].astype(str).str.strip()
    )
    account_map = dict(zip(account_key, accounts["entity_id"].astype(str)))
    entity_ids = np.asarray(sorted(accounts["entity_id"].astype(str).unique()), dtype=str)
    entity_codes = {entity_id: i for i, entity_id in enumerate(entity_ids)}

    transactions = pd.read_csv(
        transactions_path,
        usecols=lambda column: column in SOURCE_COLUMNS,
        dtype={"From Bank": "string", "Account": "string", "To Bank": "string", "Account.1": "string"},
        keep_default_na=False,
    )
    missing = SOURCE_COLUMNS - set(transactions.columns)
    if missing:
        raise ValueError(f"Transactions CSV is missing required columns: {sorted(missing)}")
    transactions["Timestamp"] = pd.to_datetime(transactions["Timestamp"], errors="raise")
    transactions["From Bank"] = _normalize_bank_series(transactions["From Bank"])
    transactions["To Bank"] = _normalize_bank_series(transactions["To Bank"])
    transactions["Account"] = transactions["Account"].astype("string").str.strip()
    transactions["Account.1"] = transactions["Account.1"].astype("string").str.strip()

    sender_key = transactions["From Bank"].astype(str) + SEPARATOR + transactions["Account"].astype(str)
    receiver_key = transactions["To Bank"].astype(str) + SEPARATOR + transactions["Account.1"].astype(str)
    sender_entities = sender_key.map(account_map)
    receiver_entities = receiver_key.map(account_map)
    if sender_entities.isna().any() or receiver_entities.isna().any():
        raise ValueError("Source transactions contain unresolved composite account identities")
    transactions["sender_code"] = sender_entities.map(entity_codes).astype(np.int32)
    transactions["receiver_code"] = receiver_entities.map(entity_codes).astype(np.int32)
    transactions["timestamp_ns"] = transactions["Timestamp"].astype("int64")
    transactions["amount_received"] = pd.to_numeric(transactions["Amount Received"], errors="raise").astype(np.float64)
    transactions["amount_paid"] = pd.to_numeric(transactions["Amount Paid"], errors="raise").astype(np.float64)

    # Only the compact fields needed for the notebook's entity/timestamp rollups remain.
    frame = transactions[["sender_code", "receiver_code", "timestamp_ns", "amount_received", "amount_paid"]]
    frame.attrs["entity_count"] = len(entity_ids)
    sender = _role_rollup(frame, "sender_code", "receiver_code")
    receiver = _role_rollup(frame, "receiver_code", "sender_code")
    result = {"entity_ids": entity_ids}
    result.update({f"{name}_sender": value for name, value in sender.items()})
    result.update({f"{name}_receiver": value for name, value in receiver.items()})
    return result


def persist_behavior_history(arrays: dict[str, np.ndarray], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    arrays = build_behavior_history(
        settings.data_root / "HI-Small_Trans.csv",
        settings.master_table_root / "accounts.csv",
    )
    persist_behavior_history(arrays, settings.behavior_history_path)
    print(f"Entities indexed: {len(arrays['entity_ids']):,}")
    print(f"Sender timestamp rows: {len(arrays['timestamps_sender']):,}")
    print(f"Receiver timestamp rows: {len(arrays['timestamps_receiver']):,}")
    print(f"Behavior history written to: {settings.behavior_history_path}")


if __name__ == "__main__":
    main()
