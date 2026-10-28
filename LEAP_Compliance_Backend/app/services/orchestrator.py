"""Coordinates identity, AML scoring, enrichment, and evidence output."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from app.aml.behavioral import BehaviorHistoryIndex
from app.aml.inference import AMLModel
from app.aml.preprocessing import build_feature_vector
from app.api.schemas import AnalyzeRequest, AnalyzeResponse
from app.core.config import settings
from app.data.prepare_master_data import normalize_bank_id
from app.data.repositories import IdentityRepository
from app.services.enrichment import EnrichmentService


def _utc_naive(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _utc_iso(value: datetime) -> str:
    aware = value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    return aware.isoformat().replace("+00:00", "Z")


def _bool_text(value: bool) -> str:
    return "True" if value else "False"


class IntelligenceOrchestrator:
    def __init__(self):
        self.identity = IdentityRepository(settings.master_table_root)
        self.model = AMLModel.load(settings.aml_model_path, settings.aml_metadata_path)
        self.behavior = BehaviorHistoryIndex.load(settings.behavior_history_path)
        self.enrichment = EnrichmentService(settings.data_root)

    def analyze(self, request: AnalyzeRequest) -> AnalyzeResponse:
        sender_entity = self.identity.resolve_entity(request.sender.bank_id, request.sender.account_id)
        receiver_entity = self.identity.resolve_entity(request.receiver.bank_id, request.receiver.account_id)
        timestamp = _utc_naive(request.timestamp)
        behavioral = self.behavior.features(sender_entity, receiver_entity, timestamp)
        transaction_features = {
            "timestamp": timestamp,
            "amount_received": request.amount_received,
            "amount_paid": request.amount_paid,
            "receiving_currency": request.receiving_currency,
            "payment_currency": request.payment_currency,
            "payment_format": request.payment_format,
        }
        vector = build_feature_vector(transaction_features, behavioral, self.model.feature_names)
        probability = self.model.predict_probability(vector)
        alert = probability >= self.model.threshold

        sender_kyc = self.enrichment.kyc_profile(sender_entity)
        receiver_kyc = self.enrichment.kyc_profile(receiver_entity)
        sender_sanctions = self.enrichment.sanctions_record(sender_entity)
        receiver_sanctions = self.enrichment.sanctions_record(receiver_entity)
        sender_history = self.enrichment.historical_record(sender_entity)
        receiver_history = self.enrichment.historical_record(receiver_entity)

        kyc_signal = (
            self.enrichment.kyc_evidence_present(sender_kyc)
            or self.enrichment.kyc_evidence_present(receiver_kyc)
        )
        sanctions_signal = sender_sanctions is not None or receiver_sanctions is not None
        historical_signal = sender_history is not None or receiver_history is not None
        additional_count = sum((kyc_signal, sanctions_signal, historical_signal))
        evidence = self._evidence(
            request, probability, alert,
            sender_entity, receiver_entity,
            sender_kyc, receiver_kyc,
            sender_sanctions, receiver_sanctions,
            sender_history, receiver_history,
        )
        request_id = request.request_id or f"REQ_{uuid4().hex}"
        return AnalyzeResponse.model_validate({
            "request_id": request_id,
            "transaction": {
                "transaction_id": request.transaction_id,
                "timestamp": _utc_iso(request.timestamp),
                "sender": {
                    "bank_id": normalize_bank_id(request.sender.bank_id),
                    "account_id": request.sender.account_id.strip(),
                    "entity_id": sender_entity,
                },
                "receiver": {
                    "bank_id": normalize_bank_id(request.receiver.bank_id),
                    "account_id": request.receiver.account_id.strip(),
                    "entity_id": receiver_entity,
                },
                "amount_received": request.amount_received,
                "amount_paid": request.amount_paid,
                "receiving_currency": request.receiving_currency,
                "payment_currency": request.payment_currency,
                "payment_format": request.payment_format,
            },
            "aml": {
                "model_probability": probability,
                "decision_threshold": self.model.threshold,
                "alert": alert,
            },
            "sender": {"entity_id": sender_entity, "kyc": sender_kyc, "sanctions": sender_sanctions, "historical": sender_history},
            "receiver": {"entity_id": receiver_entity, "kyc": receiver_kyc, "sanctions": receiver_sanctions, "historical": receiver_history},
            "cross_domain": {
                "kyc_evidence_present": kyc_signal,
                "sanctions_evidence_present": sanctions_signal,
                "historical_evidence_present": historical_signal,
                "additional_evidence_domain_count": additional_count,
            },
            "deterministic_evidence": evidence,
        })

    def _evidence(
        self, request, probability, alert, sender_id, receiver_id,
        sender_kyc, receiver_kyc, sender_sanctions, receiver_sanctions,
        sender_history, receiver_history,
    ):
        items = [{
            "domain": "AML", "entity_side": "transaction", "entity_id": None,
            "observation": (
                f"AML model probability is {probability:.4f}; the configured alert threshold is "
                f"{self.model.threshold:.2f}; the transaction "
                f"{'generated' if alert else 'did not generate'} an AML alert."
            ),
        }, {
            "domain": "Transaction", "entity_side": "transaction", "entity_id": None,
            "observation": (
                f"Transaction amount paid is {request.amount_paid:g} {request.payment_currency}; "
                f"payment format is {request.payment_format}."
            ),
        }]
        for side, entity_id, profile, sanctions, history in (
            ("sender", sender_id, sender_kyc, sender_sanctions, sender_history),
            ("receiver", receiver_id, receiver_kyc, receiver_sanctions, receiver_history),
        ):
            if self.enrichment.kyc_evidence_present(profile):
                items.append({
                    "domain": "KYC", "entity_side": side, "entity_id": entity_id,
                    "observation": (
                        f"{side.capitalize()} entity {entity_id} has KYC status {profile['kyc_status']}, "
                        f"risk rating {profile['risk_rating']}, and PEP flag {_bool_text(profile['pep_flag'])}."
                    ),
                })
            if sanctions is not None:
                items.append({
                    "domain": "Sanctions", "entity_side": side, "entity_id": entity_id,
                    "observation": (
                        f"{side.capitalize()} entity {entity_id} has a sanctions record matched under the "
                        f"{sanctions['sanctions_program']} program; match type is {sanctions['match_type']}, "
                        f"confidence is {sanctions['match_confidence']:g}, and status is {sanctions['sanctions_status']}."
                    ),
                })
            if history is not None:
                items.extend([
                    {
                        "domain": "Historical", "entity_side": side, "entity_id": entity_id,
                        "observation": (
                            f"{side.capitalize()} entity {entity_id} has {history['investigation_count']} "
                            f"historical investigation(s); the latest investigation was in the "
                            f"{history['latest_domain']} domain, with case status {history['latest_case_status']} "
                            f"and recorded risk level {history['latest_risk_level']}."
                        ),
                    },
                    {
                        "domain": "Historical", "entity_side": side, "entity_id": entity_id,
                        "observation": f"For {side} entity {entity_id}, the latest recorded finding was: {history['latest_finding']}.",
                    },
                    {
                        "domain": "Historical", "entity_side": side, "entity_id": entity_id,
                        "observation": f"For {side} entity {entity_id}, the latest recorded resolution was: {history['latest_resolution']}.",
                    },
                ])
        return items
