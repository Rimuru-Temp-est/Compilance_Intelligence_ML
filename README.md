[README.md](https://github.com/user-attachments/files/34055006/README.md)
# Compilance_Intelligence_ML# AI-Powered Cross-Domain Compliance Intelligence Backend

A local, CSV-backed FastAPI backend for the existing AML Model B and the KYC, sanctions, and historical investigation datasets. It returns deterministic structured evidence; it does not use an LLM to make compliance decisions.

## Architecture

`POST /api/v1/analyze` resolves sender and receiver using `(bank_id, account_id)`, looks up strictly earlier transaction behavior, builds the 52 model features, scores with the saved XGBoost model and threshold `0.90`, retrieves each enrichment domain, and constructs evidence. Data access and enrichment are kept outside the route handler.

The service uses the supplied trained model at `AML_Model/processed/aml_model/`. Explicit retraining writes a separate model to `backend_data/processed/aml_model/` and never overwrites the supplied model. API startup does not train.

## Project layout

- `app/api/`: Pydantic schemas and `/api/v1` routes
- `app/aml/`: feature construction, sparse preprocessing, strict-prior history index, model inference
- `app/data/`: deterministic reference-table preparation and CSV repositories
- `app/services/`: enrichment and request orchestration
- `app/training/`: explicit offline AML retraining command
- `backend_data/master_table/`: generated canonical `accounts.csv`, `banks.csv`, and `entities.csv`
- `backend_data/processed/behavior_history.npz`: generated compact timestamp history index
- `Datasets/`: source CSVs, treated as read-only
- `AML_Model/processed/aml_model/`: supplied model pickle and metadata, treated as read-only

## Requirements and setup

Use Python 3.12, matching the recorded XGBoost 3.4.1 model environment.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
cp .env.example .env
```

The default configuration is project-relative. Set `DATA_ROOT`, `AML_MODEL_PATH`, `AML_METADATA_PATH`, `MASTER_TABLE_ROOT`, or `BEHAVIOR_HISTORY_PATH` in `.env` when moving the project or changing artifact locations.

## Prepare derived data

These commands only read the original source CSVs. They write derived outputs under `backend_data/`.

```bash
python -m app.data.prepare_master_data
python -m app.data.prepare_behavior_history
```

The history index is prepared once and loaded in memory for requests; the 5-million-row transaction CSV is not re-read per request. Re-run the history preparation command only when the source transactions or accounts change.

Bank IDs follow Notebook 2 normalization: digit-only IDs have leading zeros stripped (`"070"` becomes canonical `"70"`). API output uses that canonical string representation. Account IDs and entity IDs remain strings.

The categorical encoder was not separately serialized by the notebook. The exact category columns and their order are reconstructed from the feature names in the supplied metadata; unknown categories map to all-zero one-hot values as in `OneHotEncoder(handle_unknown="ignore")`. Inference converts the feature vector to SciPy CSR because the saved model was trained on sparse matrices; this preserves XGBoost's training-time zero/missing semantics.

The notebook's unique-counterparty calculation cumulatively sums each timestamp's distinct counterparties. The implementation preserves this exact behavior for model parity.

## Run the API

```bash
uvicorn app.main:app --reload
```

- Health: `GET http://127.0.0.1:8000/api/v1/health`
- Analyze: `POST http://127.0.0.1:8000/api/v1/analyze`
- Interactive API docs: `http://127.0.0.1:8000/docs`

Example request:

```json
{
  "transaction_id": "TXN_DEMO_000001",
  "timestamp": "2022-10-02T19:30:00Z",
  "sender": {"bank_id": "070", "account_id": "ACC_001234"},
  "receiver": {"bank_id": "081", "account_id": "ACC_009876"},
  "amount_received": 125000.5,
  "amount_paid": 125000.5,
  "receiving_currency": "Euro",
  "payment_currency": "Euro",
  "payment_format": "Wire",
  "request_id": "REQ_DEMO_0001",
  "metadata": {"source": "demo-ui"}
}
```

The example identifiers are illustrative; the supplied dataset must contain both composite account identities for the request to resolve. The client must not provide entity IDs, labels, SHAP values, or precomputed behavioral features. Unknown banks/accounts return HTTP 404; invalid request fields return HTTP 422. Model or required artifact loading failures return HTTP 503.

The response includes `request_id`, `transaction`, `aml`, `sender`, `receiver`, `cross_domain`, and `deterministic_evidence`. A KYC profile may be present even when `kyc_evidence_present` is false; Notebook 4 defines KYC evidence as medium/high risk or a PEP flag. A sanctions object is `null` when no record is found. Historical enrichment is entity-based, never keyed by account ID alone.

## Explicit training

Training is optional and is not part of API startup. It uses the same numeric feature helper and strict-prior behavior index as inference, applies the notebook's temporal split, and freezes threshold `0.90`.

```bash
python -m app.training.train_aml
```

The resulting artifacts are written under `backend_data/processed/aml_model/`. To serve that newly trained model, set `AML_MODEL_PATH` and `AML_METADATA_PATH` to those output files. Retraining does not modify the supplied model.

## Tests and inference parity

```bash
python -m pytest -q
```

The tests cover the health and analyze endpoints, schema rejection, composite identity, strict timestamp cutoff, model metadata/feature count, and saved notebook probability parity on known transactions. The notebook's AML test-results CSV is a validation reference, not a runtime dependency.

## Packaging

From the project root, create a clean handoff archive with:

```bash
zip -r LEAP_Compliance_Backend.zip \
  app backend_data Datasets \
  AML_Model/processed/aml_model \
  AML_Model/processed/aml_test_results.csv \
  requirements.txt requirements-dev.txt .env.example .gitignore README.md tests scripts \
  -x '*/__pycache__/*' '*/.pytest_cache/*' '*/.venv/*' '*/.DS_Store' '.env'
```

This includes the source data, supplied model, validation reference, and derived artifacts. Do not include notebooks, unrelated frontend archives, `.venv`, or `.env`.
