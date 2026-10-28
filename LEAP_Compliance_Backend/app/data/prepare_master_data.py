"""Build canonical reference tables using Notebook 2 transformations."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from app.core.config import settings

ACCOUNT_COLUMNS = ["Bank Name", "Bank ID", "Account Number", "Entity ID", "Entity Name"]
IDENTIFIER_DTYPES = {
    "Bank Name": "string",
    "Bank ID": "string",
    "Account Number": "string",
    "Entity ID": "string",
    "Entity Name": "string",
}


def normalize_bank_id(value: object) -> str:
    """Apply Notebook 2's canonical bank ID normalization."""
    bank_id = str(value).strip()
    if bank_id.isdigit():
        return bank_id.lstrip("0") or "0"
    return bank_id


def build_master_tables(accounts_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (accounts, banks, entities) exactly as defined in Notebook 2."""
    if not accounts_path.is_file():
        raise FileNotFoundError(f"Accounts source CSV not found: {accounts_path}")

    raw = pd.read_csv(
        accounts_path,
        dtype=IDENTIFIER_DTYPES,
        keep_default_na=False,
    )
    missing = set(ACCOUNT_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"Accounts CSV is missing required columns: {sorted(missing)}")

    account_rows = raw.copy()
    account_rows["Bank ID"] = account_rows["Bank ID"].map(normalize_bank_id)
    for column in ["Bank Name", "Account Number", "Entity ID", "Entity Name"]:
        account_rows[column] = account_rows[column].str.strip()

    accounts = account_rows.rename(columns={
        "Bank ID": "bank_id",
        "Account Number": "account_id",
        "Entity ID": "entity_id",
    })[["account_id", "bank_id", "entity_id"]].copy()
    for column in ["account_id", "bank_id", "entity_id"]:
        accounts[column] = accounts[column].astype("string")

    if accounts.duplicated(subset=["bank_id", "account_id"]).any():
        raise ValueError("Duplicate canonical (bank_id, account_id) identities in accounts source")

    banks = raw[["Bank ID", "Bank Name"]].copy()
    banks["Bank ID"] = banks["Bank ID"].map(normalize_bank_id)
    banks["Bank Name"] = banks["Bank Name"].str.strip()
    banks = banks.drop_duplicates().sort_values("Bank ID").reset_index(drop=True)
    banks = banks.rename(columns={"Bank ID": "bank_id", "Bank Name": "bank_name"})
    banks["bank_id"] = banks["bank_id"].astype("string")
    banks["bank_name"] = banks["bank_name"].astype("string")
    if banks.groupby("bank_id")["bank_name"].nunique().gt(1).any():
        raise ValueError("A canonical bank_id maps to multiple bank names")
    banks = banks.drop_duplicates(subset=["bank_id"], keep="first").reset_index(drop=True)

    entities = raw[["Entity ID", "Entity Name"]].copy()
    entities["Entity ID"] = entities["Entity ID"].str.strip()
    entities["Entity Name"] = entities["Entity Name"].str.strip()
    entities = entities.drop_duplicates().sort_values("Entity ID").reset_index(drop=True)
    entities = entities.rename(columns={"Entity ID": "entity_id", "Entity Name": "entity_name"})
    entities["entity_id"] = entities["entity_id"].astype("string")
    entities["entity_name"] = entities["entity_name"].astype("string")
    if entities.groupby("entity_id")["entity_name"].nunique().gt(1).any():
        raise ValueError("A canonical entity_id maps to multiple entity names")
    entities = entities.drop_duplicates(subset=["entity_id"], keep="first").reset_index(drop=True)

    account_entities = set(accounts["entity_id"].astype(str))
    master_entities = set(entities["entity_id"].astype(str))
    if account_entities - master_entities:
        raise ValueError("Accounts reference entity IDs missing from the canonical entities table")
    return accounts, banks, entities


def persist_master_tables(
    accounts: pd.DataFrame, banks: pd.DataFrame, entities: pd.DataFrame, output_dir: Path
) -> None:
    """Persist derived outputs only under the backend-controlled artifact directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        "accounts.csv": accounts,
        "banks.csv": banks,
        "entities.csv": entities,
    }
    staged: list[tuple[Path, Path]] = []
    try:
        for filename, frame in frames.items():
            destination = output_dir / filename
            temporary = output_dir / f".{filename}.tmp"
            frame.to_csv(temporary, index=False)
            staged.append((temporary, destination))
        for temporary, destination in staged:
            os.replace(temporary, destination)
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)


def main() -> None:
    accounts, banks, entities = build_master_tables(settings.data_root / "HI-Small_accounts.csv")
    persist_master_tables(accounts, banks, entities, settings.master_table_root)
    print(f"Canonical accounts: {len(accounts):,}")
    print(f"Canonical banks:    {len(banks):,}")
    print(f"Canonical entities: {len(entities):,}")
    print(f"Artifacts written to: {settings.master_table_root}")


if __name__ == "__main__":
    main()
