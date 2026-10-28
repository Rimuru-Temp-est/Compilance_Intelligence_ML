"""Public request and response models for the analysis API."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class PartyRequest(StrictModel):
    bank_id: str = Field(min_length=1)
    account_id: str = Field(min_length=1)


class AnalyzeRequest(StrictModel):
    transaction_id: str = Field(min_length=1)
    timestamp: datetime
    sender: PartyRequest
    receiver: PartyRequest
    amount_received: float = Field(ge=0, allow_inf_nan=False)
    amount_paid: float = Field(gt=0, allow_inf_nan=False)
    receiving_currency: str = Field(min_length=1)
    payment_currency: str = Field(min_length=1)
    payment_format: str = Field(min_length=1)
    request_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def reject_compliance_payload_fields(cls, value: dict[str, Any]) -> dict[str, Any]:
        forbidden = {
            "is_laundering", "aml_label", "aml_prediction", "model_probability",
            "sender_entity_id", "receiver_entity_id", "shap", "shap_values",
            "behavioral_features", "precomputed_behavioral_features",
        }
        def visit(item: Any) -> None:
            if isinstance(item, dict):
                for key, child in item.items():
                    normalized = str(key).strip().lower()
                    if normalized in forbidden:
                        raise ValueError(f"metadata must not contain backend-owned field: {key}")
                    visit(child)
            elif isinstance(item, list):
                for child in item:
                    visit(child)
        visit(value)
        return value


class PartyIdentity(StrictModel):
    bank_id: str
    account_id: str
    entity_id: str


class TransactionResult(StrictModel):
    transaction_id: str
    timestamp: str
    sender: PartyIdentity
    receiver: PartyIdentity
    amount_received: float
    amount_paid: float
    receiving_currency: str
    payment_currency: str
    payment_format: str


class AMLResult(StrictModel):
    model_probability: float
    decision_threshold: float
    alert: bool


class KYCResult(StrictModel):
    kyc_status: str
    customer_type: str
    country: str
    occupation: str
    annual_income_band: str
    risk_rating: str
    pep_flag: bool
    kyc_last_review_date: date | str


class SanctionsResult(StrictModel):
    sanctions_entity_id: str
    matched_name: str
    sanctions_program: str
    match_type: str
    match_confidence: float
    sanctions_status: str


class HistoricalResult(StrictModel):
    investigation_count: int
    domains: list[str]
    trigger_types: list[str]
    latest_investigation_date: date | str
    latest_case_status: str
    latest_risk_level: str
    latest_finding: str
    latest_resolution: str
    latest_domain: str
    latest_trigger_type: str
    latest_investigation_id: str


class EntityResult(StrictModel):
    entity_id: str
    kyc: KYCResult | None
    sanctions: SanctionsResult | None
    historical: HistoricalResult | None


class CrossDomainSummary(StrictModel):
    kyc_evidence_present: bool
    sanctions_evidence_present: bool
    historical_evidence_present: bool
    additional_evidence_domain_count: int


class EvidenceItem(StrictModel):
    domain: Literal["AML", "Transaction", "KYC", "Sanctions", "Historical"]
    entity_side: Literal["transaction", "sender", "receiver"]
    entity_id: str | None
    observation: str


class AnalyzeResponse(StrictModel):
    request_id: str
    transaction: TransactionResult
    aml: AMLResult
    sender: EntityResult
    receiver: EntityResult
    cross_domain: CrossDomainSummary
    deterministic_evidence: list[EvidenceItem]
