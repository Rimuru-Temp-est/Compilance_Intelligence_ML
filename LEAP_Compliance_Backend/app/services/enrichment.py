"""Entity-level KYC, sanctions, and investigation lookups."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from app.data.repositories import load_enrichment_tables


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _date_value(value: Any) -> str:
    if isinstance(value, (date, pd.Timestamp)):
        return value.date().isoformat() if isinstance(value, pd.Timestamp) else value.isoformat()
    return str(value)


class EnrichmentService:
    def __init__(self, data_root):
        tables = load_enrichment_tables(data_root)
        kyc = tables["kyc_profiles"].copy()
        kyc["entity_id"] = kyc["entity_id"].astype(str).str.strip()
        self._kyc = {row["entity_id"]: row for row in kyc.to_dict(orient="records")}

        sanctions = tables["sanctions_entities"].copy()
        sanctions["entity_id"] = sanctions["entity_id"].astype(str).str.strip()
        self._sanctions = {
            row["entity_id"]: row for row in sanctions.to_dict(orient="records")
        }

        history = tables["investigation_history"].copy()
        history["entity_id"] = history["entity_id"].astype(str).str.strip()
        history["investigation_date"] = pd.to_datetime(history["investigation_date"], errors="raise")
        self._history: dict[str, dict[str, Any]] = {}
        for entity_id, group in history.groupby("entity_id", sort=False):
            ordered = group.sort_values(["investigation_date", "investigation_id"])
            latest = ordered.iloc[-1]
            self._history[str(entity_id)] = {
                "investigation_count": int(len(ordered)),
                "domains": sorted(ordered["domain"].dropna().astype(str).unique().tolist()),
                "trigger_types": sorted(ordered["trigger_type"].dropna().astype(str).unique().tolist()),
                "latest_investigation_date": latest["investigation_date"].date().isoformat(),
                "latest_case_status": str(latest["case_status"]),
                "latest_risk_level": str(latest["risk_level"]),
                "latest_finding": str(latest["finding"]),
                "latest_resolution": str(latest["resolution"]),
                "latest_domain": str(latest["domain"]),
                "latest_trigger_type": str(latest["trigger_type"]),
                "latest_investigation_id": str(latest["investigation_id"]),
            }

    def kyc_profile(self, entity_id: str) -> dict[str, Any] | None:
        row = self._kyc.get(str(entity_id))
        if row is None:
            return None
        return {
            "kyc_status": str(row["kyc_status"]),
            "customer_type": str(row["customer_type"]),
            "country": str(row["country"]),
            "occupation": str(row["occupation"]),
            "annual_income_band": str(row["annual_income_band"]),
            "risk_rating": str(row["risk_rating"]),
            "pep_flag": _bool_value(row["pep_flag"]),
            "kyc_last_review_date": _date_value(row["kyc_last_review_date"]),
        }

    def kyc_evidence_present(self, profile: dict[str, Any] | None) -> bool:
        # Notebook 4: medium/high KYC risk or PEP is evidence; profile presence alone is not.
        return bool(profile and (profile["risk_rating"] in {"Medium", "High"} or profile["pep_flag"]))

    def sanctions_record(self, entity_id: str) -> dict[str, Any] | None:
        row = self._sanctions.get(str(entity_id))
        if row is None:
            return None
        return {
            "sanctions_entity_id": str(row["sanctions_entity_id"]),
            "matched_name": str(row["matched_name"]),
            "sanctions_program": str(row["sanctions_program"]),
            "match_type": str(row["match_type"]),
            "match_confidence": float(row["match_confidence"]),
            "sanctions_status": str(row["sanctions_status"]),
        }

    def historical_record(self, entity_id: str) -> dict[str, Any] | None:
        return self._history.get(str(entity_id))
