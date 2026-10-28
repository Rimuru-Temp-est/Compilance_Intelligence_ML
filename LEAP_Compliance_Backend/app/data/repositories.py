"""CSV-backed data access for canonical identities and enrichment tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.data.prepare_master_data import normalize_bank_id


class UnknownBankError(LookupError):
    pass


class UnknownAccountError(LookupError):
    pass


class DataIntegrityError(RuntimeError):
    pass


def _read_csv_required(path: Path, required_columns: set[str], dtypes: dict[str, str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required data artifact not found: {path}")
    frame = pd.read_csv(path, dtype=dtypes, keep_default_na=False)
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")
    return frame


class IdentityRepository:
    """Resolve an entity only by the canonical composite account identity."""

    def __init__(self, master_table_root: Path):
        banks = _read_csv_required(
            master_table_root / "banks.csv",
            {"bank_id", "bank_name"},
            {"bank_id": "string", "bank_name": "string"},
        )
        accounts = _read_csv_required(
            master_table_root / "accounts.csv",
            {"bank_id", "account_id", "entity_id"},
            {"bank_id": "string", "account_id": "string", "entity_id": "string"},
        )
        entities = _read_csv_required(
            master_table_root / "entities.csv",
            {"entity_id", "entity_name"},
            {"entity_id": "string", "entity_name": "string"},
        )
        if banks["bank_id"].duplicated().any():
            raise DataIntegrityError("banks.csv contains duplicate canonical bank IDs")
        if accounts.duplicated(subset=["bank_id", "account_id"]).any():
            raise DataIntegrityError("accounts.csv contains duplicate composite account identities")
        if entities["entity_id"].duplicated().any():
            raise DataIntegrityError("entities.csv contains duplicate entity IDs")

        self._bank_ids = set(banks["bank_id"].astype(str))
        self._account_to_entity = {
            (str(row.bank_id), str(row.account_id)): str(row.entity_id)
            for row in accounts.itertuples(index=False)
        }
        self._entity_ids = set(entities["entity_id"].astype(str))

    def resolve_entity(self, bank_id: str, account_id: str) -> str:
        canonical_bank_id = normalize_bank_id(bank_id)
        canonical_account_id = str(account_id).strip()
        if canonical_bank_id not in self._bank_ids:
            raise UnknownBankError(f"Unknown bank_id: {bank_id}")
        entity_id = self._account_to_entity.get((canonical_bank_id, canonical_account_id))
        if entity_id is None:
            raise UnknownAccountError(
                f"Unknown account for composite identity ({bank_id!r}, {account_id!r})"
            )
        if entity_id not in self._entity_ids:
            raise DataIntegrityError(f"Resolved entity_id is missing from entities.csv: {entity_id}")
        return entity_id


def load_enrichment_tables(data_root: Path) -> dict[str, pd.DataFrame]:
    """Load and minimally validate the three supplied synthetic datasets."""
    specs = {
        "kyc_profiles": (
            {"entity_id", "kyc_status", "customer_type", "country", "occupation", "annual_income_band", "risk_rating", "pep_flag", "kyc_last_review_date"},
            {"entity_id": "string"},
        ),
        "sanctions_entities": (
            {"sanctions_entity_id", "entity_id", "matched_name", "sanctions_program", "match_type", "match_confidence", "sanctions_status"},
            {"sanctions_entity_id": "string", "entity_id": "string"},
        ),
        "investigation_history": (
            {"investigation_id", "entity_id", "account_id", "investigation_date", "domain", "trigger_type", "case_status", "risk_level", "finding", "resolution"},
            {"investigation_id": "string", "entity_id": "string", "account_id": "string"},
        ),
    }
    tables = {}
    for name, (columns, dtypes) in specs.items():
        tables[name] = _read_csv_required(data_root / f"{name}.csv", columns, dtypes)
    if tables["kyc_profiles"]["entity_id"].duplicated().any():
        raise DataIntegrityError("kyc_profiles.csv must have at most one row per entity_id")
    if tables["sanctions_entities"]["sanctions_entity_id"].duplicated().any():
        raise DataIntegrityError("sanctions_entities.csv contains duplicate sanctions_entity_id values")
    if tables["investigation_history"]["investigation_id"].duplicated().any():
        raise DataIntegrityError("investigation_history.csv contains duplicate investigation_id values")
    return tables
