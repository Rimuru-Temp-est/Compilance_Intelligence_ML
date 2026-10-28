"""Explicit AML Model B training command; never invoked by API startup."""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from app.aml.behavioral import BehaviorHistoryIndex
from app.aml.preprocessing import (
    BEHAVIORAL_COLUMNS,
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    derive_numeric_features,
)
from app.core.config import settings
from app.data.prepare_master_data import normalize_bank_id

RANDOM_STATE = 42
THRESHOLD = 0.90
ACCOUNT_SEPARATOR = "\x1f"


def load_training_transactions(path: Path, accounts_path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Transactions source not found: {path}")
    if not accounts_path.is_file():
        raise FileNotFoundError(
            f"Canonical accounts are missing; run python -m app.data.prepare_master_data first: {accounts_path}"
        )
    dtype = {
        "From Bank": "string", "Account": "string", "To Bank": "string", "Account.1": "string",
        "Receiving Currency": "string", "Payment Currency": "string", "Payment Format": "string",
        "Is Laundering": "int8",
    }
    tx = pd.read_csv(path, dtype=dtype, keep_default_na=False)
    expected = {
        "Timestamp", "From Bank", "Account", "To Bank", "Account.1", "Amount Received",
        "Receiving Currency", "Amount Paid", "Payment Currency", "Payment Format", "Is Laundering",
    }
    missing = expected - set(tx.columns)
    if missing:
        raise ValueError(f"Transactions source is missing required columns: {sorted(missing)}")

    accounts = pd.read_csv(
        accounts_path,
        dtype={"bank_id": "string", "account_id": "string", "entity_id": "string"},
        keep_default_na=False,
    )
    account_keys = (
        accounts["bank_id"].astype(str).str.strip()
        + ACCOUNT_SEPARATOR
        + accounts["account_id"].astype(str).str.strip()
    )
    account_lookup = dict(zip(account_keys, accounts["entity_id"].astype(str)))

    tx["From Bank"] = tx["From Bank"].map(normalize_bank_id)
    tx["To Bank"] = tx["To Bank"].map(normalize_bank_id)
    tx["Account"] = tx["Account"].str.strip()
    tx["Account.1"] = tx["Account.1"].str.strip()
    for column in CATEGORICAL_COLUMNS:
        source_column = {
            "receiving_currency": "Receiving Currency",
            "payment_currency": "Payment Currency",
            "payment_format": "Payment Format",
        }[column]
        tx[source_column] = tx[source_column].str.strip()

    tx["sender_entity_id"] = (
        tx["From Bank"].astype(str) + ACCOUNT_SEPARATOR + tx["Account"].astype(str)
    ).map(account_lookup)
    tx["receiver_entity_id"] = (
        tx["To Bank"].astype(str) + ACCOUNT_SEPARATOR + tx["Account.1"].astype(str)
    ).map(account_lookup)
    if tx["sender_entity_id"].isna().any() or tx["receiver_entity_id"].isna().any():
        raise ValueError("Transactions source contains unresolved composite account identities")

    tx = tx.rename(columns={
        "Timestamp": "timestamp",
        "Amount Received": "amount_received",
        "Amount Paid": "amount_paid",
        "Receiving Currency": "receiving_currency",
        "Payment Currency": "payment_currency",
        "Payment Format": "payment_format",
        "Is Laundering": "is_laundering",
    })
    tx["timestamp"] = pd.to_datetime(tx["timestamp"], errors="raise")
    tx["amount_received"] = pd.to_numeric(tx["amount_received"], errors="raise")
    tx["amount_paid"] = pd.to_numeric(tx["amount_paid"], errors="raise")
    tx["transaction_id"] = [f"TXN_{i:08d}" for i in range(1, len(tx) + 1)]
    tx = tx.sort_values(["timestamp", "transaction_id"], kind="mergesort").reset_index(drop=True)
    return tx


def temporal_masks(timestamps: pd.Series) -> dict[str, np.ndarray]:
    train_end = pd.Timestamp("2022-09-07 23:59:59")
    validation_start = pd.Timestamp("2022-09-08 00:00:00")
    validation_end = pd.Timestamp("2022-09-09 23:59:59")
    test_start = pd.Timestamp("2022-09-10 00:00:00")
    test_end = pd.Timestamp("2022-09-10 23:59:59")
    stress_start = pd.Timestamp("2022-09-11 00:00:00")
    masks = {
        "train": (timestamps <= train_end).to_numpy(),
        "validation": ((timestamps >= validation_start) & (timestamps <= validation_end)).to_numpy(),
        "test": ((timestamps >= test_start) & (timestamps <= test_end)).to_numpy(),
        "stress": (timestamps >= stress_start).to_numpy(),
    }
    membership = np.sum(np.column_stack(list(masks.values())), axis=1)
    if not np.all(membership == 1):
        raise ValueError("Temporal split did not assign every transaction to exactly one partition")
    return masks


def _behavior_matrix(tx: pd.DataFrame, history: BehaviorHistoryIndex) -> np.ndarray:
    timestamps_ns = tx["timestamp"].astype("int64").to_numpy(dtype=np.int64)
    sender = history.cumulative_for_rows(
        tx["sender_entity_id"].astype(str).to_numpy(), timestamps_ns, "sender"
    )
    receiver = history.cumulative_for_rows(
        tx["receiver_entity_id"].astype(str).to_numpy(), timestamps_ns, "receiver"
    )
    columns = {
        "log_sender_prior_txns": np.log1p(sender[0]),
        "log_sender_prior_received": np.log1p(sender[1]),
        "log_sender_prior_paid": np.log1p(sender[2]),
        "log_receiver_prior_txns": np.log1p(receiver[0]),
        "log_receiver_prior_received": np.log1p(receiver[1]),
        "log_receiver_prior_paid": np.log1p(receiver[2]),
        "log_sender_prior_unique_receivers": np.log1p(sender[3]),
        "log_receiver_prior_unique_senders": np.log1p(receiver[3]),
    }
    return np.column_stack([columns[name] for name in BEHAVIORAL_COLUMNS]).astype(np.float32)


def train_aml(
    transactions_path: Path | None = None,
    accounts_path: Path | None = None,
    history_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Train explicitly and write a new model under backend_data (never over the reference model)."""
    transactions_path = transactions_path or settings.data_root / "HI-Small_Trans.csv"
    accounts_path = accounts_path or settings.master_table_root / "accounts.csv"
    history_path = history_path or settings.behavior_history_path
    output_dir = output_dir or settings.project_root / "backend_data" / "processed" / "aml_model"

    tx = load_training_transactions(transactions_path, accounts_path)
    history = BehaviorHistoryIndex.load(history_path)
    masks = temporal_masks(tx["timestamp"])
    targets = tx["is_laundering"].to_numpy(dtype=np.int8)

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
    cat_train = tx.loc[masks["train"], list(CATEGORICAL_COLUMNS)]
    encoder.fit(cat_train)
    category_names = encoder.get_feature_names_out(CATEGORICAL_COLUMNS).tolist()

    numeric = derive_numeric_features(
        tx["amount_received"].to_numpy(), tx["amount_paid"].to_numpy(), tx["timestamp"]
    )
    numeric_matrix = np.column_stack([numeric[name] for name in NUMERIC_COLUMNS]).astype(np.float32)
    behavioral_matrix = _behavior_matrix(tx, history)

    def matrix_for(split: str):
        mask = masks[split]
        categorical = encoder.transform(tx.loc[mask, list(CATEGORICAL_COLUMNS)])
        return hstack(
            [categorical, numeric_matrix[mask], behavioral_matrix[mask]], format="csr"
        )

    X_train = matrix_for("train")
    X_validation = matrix_for("validation")
    X_test = matrix_for("test")
    y_train, y_validation, y_test = (targets[masks[k]] for k in ("train", "validation", "test"))
    negative = int((y_train == 0).sum())
    positive = int((y_train == 1).sum())
    if positive == 0:
        raise ValueError("Training split contains no positive laundering labels")
    scale_pos_weight = negative / positive

    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.1,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_validation, y_validation)], verbose=50)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    performance = {
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "true_positives": int(tp), "false_positives": int(fp),
        "true_negatives": int(tn), "false_negatives": int(fn),
        "test_transactions": int(len(y_test)),
        "predicted_alerts": int(predictions.sum()),
    }

    transaction_feature_names = category_names + list(NUMERIC_COLUMNS)
    feature_names = transaction_feature_names + list(BEHAVIORAL_COLUMNS)
    if len(feature_names) != 52 or model.n_features_in_ != 52:
        raise ValueError(f"Training produced unexpected model feature count: {model.n_features_in_}")
    metadata = {
        "model_name": "AML Model B (backend retrain)",
        "model_object": "xgb_behavioral_model",
        "model_type": "XGBClassifier",
        "purpose": "Transaction-level AML behavioral risk detection",
        "model_configuration": {
            "n_estimators": 500, "learning_rate": 0.1, "max_depth": 6,
            "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8,
            "objective": "binary:logistic", "eval_metric": "aucpr", "tree_method": "hist",
            "scale_pos_weight": float(scale_pos_weight), "random_state": RANDOM_STATE,
            "n_features": 52,
        },
        "decision_threshold": THRESHOLD,
        "feature_groups": {
            "transaction_features": transaction_feature_names,
            "behavioral_features": list(BEHAVIORAL_COLUMNS),
            "total_features": 52,
        },
        "behavioral_feature_definition": {
            "historical_features_use_strictly_prior_transactions": True,
            "historical_features": [
                "sender_prior_txn_count", "sender_prior_total_received", "sender_prior_total_paid",
                "receiver_prior_txn_count", "receiver_prior_total_received", "receiver_prior_total_paid",
            ],
            "historical_amount_features": [
                "sender_prior_total_received", "sender_prior_total_paid",
                "receiver_prior_total_received", "receiver_prior_total_paid",
            ],
            "counterparty_features": ["sender_prior_unique_receivers", "receiver_prior_unique_senders"],
        },
        "temporal_split": {
            "train": "2022-09-01 through 2022-09-07",
            "validation": "2022-09-08 through 2022-09-09",
            "test": "2022-09-10",
            "stress": "2022-09-11 onward",
            "split_strategy": "chronological",
            "future_information_leakage_prevented": True,
        },
        "test_performance": performance,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "xgb_behavioral_model.pkl"
    metadata_path = output_dir / "metadata.json"
    model_tmp = model_path.with_name(f".{model_path.name}.tmp")
    metadata_tmp = metadata_path.with_name(f".{metadata_path.name}.tmp")
    try:
        with model_tmp.open("wb") as stream:
            pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)
        metadata_tmp.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        os.replace(model_tmp, model_path)
        os.replace(metadata_tmp, metadata_path)
    finally:
        model_tmp.unlink(missing_ok=True)
        metadata_tmp.unlink(missing_ok=True)
    print(f"New model written to {model_path}")
    print(f"New metadata written to {metadata_path}")
    print(f"Test performance at threshold {THRESHOLD}: {performance}")
    print("To use the retrained model, set AML_MODEL_PATH and AML_METADATA_PATH explicitly.")
    return metadata


def main() -> None:
    train_aml()


if __name__ == "__main__":
    main()
