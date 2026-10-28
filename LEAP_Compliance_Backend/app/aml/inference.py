"""Trusted local loading and scoring for the saved AML Model B artifact."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from xgboost import XGBClassifier

from app.aml.preprocessing import ordered_feature_names

EXPECTED_THRESHOLD = 0.90
PARAMETER_KEYS = (
    "n_estimators", "learning_rate", "max_depth", "min_child_weight",
    "subsample", "colsample_bytree", "objective", "eval_metric",
    "tree_method", "scale_pos_weight", "random_state",
)


@dataclass
class AMLModel:
    model: XGBClassifier
    metadata: dict[str, Any]
    feature_names: list[str]
    threshold: float

    @classmethod
    def load(cls, model_path: Path, metadata_path: Path) -> "AMLModel":
        if not model_path.is_file():
            raise FileNotFoundError(f"AML model artifact not found: {model_path}")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"AML metadata artifact not found: {metadata_path}")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Unable to read AML metadata: {metadata_path}") from exc
        if metadata.get("model_object") != "xgb_behavioral_model":
            raise ValueError("Metadata does not identify the expected AML Model B")
        try:
            # Artifacts are loaded only from trusted configured local paths.
            with model_path.open("rb") as stream:
                model = pickle.load(stream)
        except Exception as exc:
            raise ValueError(f"Unable to load configured AML model artifact: {model_path}") from exc
        if not isinstance(model, XGBClassifier):
            raise TypeError(f"Expected XGBClassifier, got {type(model).__name__}")

        names = ordered_feature_names(metadata)
        model_features = int(getattr(model, "n_features_in_", -1))
        if model_features != len(names):
            raise ValueError(f"Model expects {model_features} features; metadata defines {len(names)}")
        config = metadata.get("model_configuration", {})
        actual_params = model.get_params()
        mismatches = {
            key: (actual_params.get(key), config.get(key))
            for key in PARAMETER_KEYS
            if actual_params.get(key) != config.get(key)
        }
        if mismatches:
            raise ValueError(f"Model parameters disagree with metadata: {mismatches}")
        threshold = float(metadata.get("decision_threshold", -1))
        if threshold != EXPECTED_THRESHOLD:
            raise ValueError(f"Expected frozen AML threshold {EXPECTED_THRESHOLD}, got {threshold}")
        return cls(model=model, metadata=metadata, feature_names=names, threshold=threshold)

    def predict_probability(self, features: np.ndarray) -> float:
        matrix = np.asarray(features, dtype=np.float32)
        if matrix.shape != (len(self.feature_names),):
            raise ValueError(f"Expected feature vector shape {(len(self.feature_names),)}, got {matrix.shape}")
        if not np.isfinite(matrix).all():
            raise ValueError("AML features contain NaN or infinity")
        # Training used SciPy CSR matrices. Preserve sparse zero semantics at inference.
        sparse_matrix = csr_matrix(matrix.reshape(1, -1))
        return float(self.model.predict_proba(sparse_matrix)[0, 1])
