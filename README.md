# Multimodal Loan Risk Prediction System

## 1) Problem Overview

This project implements a production-style loan approval risk system that predicts applicant outcome using a multimodal model stack and serves predictions through a secured API.  
The focus is system design and operational behavior: deterministic data alignment, explainability, auditability, concurrency-safe serving, and measurable runtime performance.

## 2) System Architecture

The model pipeline combines four branches:

- **Tabular branch (XGBoost + sklearn pipeline):** imputation, scaling, one-hot encoding, and gradient-boosted classification on structured applicant features.
- **Sequence branch (PyTorch LSTM):** deterministic pseudo-temporal financial progression sequences per applicant, encoded into a 32-dim embedding.
- **Graph branch (GraphSAGE):** applicant similarity graph built offline with kNN cosine similarity; GraphSAGE embeddings are precomputed and looked up at inference.
- **Fusion branch (MLP):** concatenates tabular logit (1), LSTM embedding (32), and graph embedding (32) into a 65-dim vector and predicts final approval logit.

### Multimodal Fusion and Alignment

- Each branch is aligned by **`Loan_ID`** (not row position).
- Sequence and graph artifacts include explicit ID arrays / index mappings.
- Fusion input construction validates missing IDs and fails fast on mismatch.
- This prevents cross-modal shuffle errors and leakage from accidental row-order assumptions.

### Design Principles

- Deterministic preprocessing
- No row-order dependency
- No target leakage
- Read-only inference artifacts
- Fail-fast ID validation

## 3) Model Performance

Performance is reported from `artifacts/evaluation_report.json` using hold-out validation.
TEST dataset does not contain labels and was not used for evaluation. All metrics are computed using hold-out validation on TRAIN only.

- **Validation set size:** 123
- **Accuracy:** 0.9512
- **Precision:** 0.9540
- **Recall:** 0.9765
- **F1 Score:** 0.9651
- **ROC-AUC:** 0.9780
- **Confusion Matrix:** `[[34, 4], [2, 83]]` (`[[TN, FP], [FN, TP]]`)

### Validation Methodology

- Dataset: `TRAIN.csv` only
- Split: `test_size=0.2`, stratified by target, `random_state=42`
- No retraining during evaluation; only inference with saved artifacts

## 4) Explainability Strategy

- **Tabular explainability:** SHAP `TreeExplainer` on trained XGBoost model, returning top feature contributions.
- **Sequence explainability:** Integrated Gradients over LSTM input sequence, aggregated to top sequence features.
- **Graph explainability:** simple magnitude-based graph influence score derived from graph-embedding contribution in fusion.
- **Human-readable explanations:** batch inference exports concise explanation text per applicant (`Explanation_Text`) with cleaned feature names.

## 5) Fairness Evaluation

Fairness is evaluated on `TRAIN.csv` labels and saved to `artifacts/fairness_report.json`.

- Group-wise analysis over:
  - `Gender`
  - `Married`
  - `Dependents`
  - `Property_Area`
- Metrics per group:
  - Approval Rate
  - Rejection Rate
  - False Positive Rate (FPR)
  - False Negative Rate (FNR)
- Alert rule:
  - Flag when absolute difference from global metric exceeds **15%** (`0.15`)

Global reference metrics from current run:
- Approval Rate: 0.7590
- FPR: 0.2656
- FNR: 0.0166

## 6) Backend Architecture

- **Framework:** FastAPI
- **Async serving:** `async def` endpoints with executor offload for model-heavy work
- **Executor separation:**
  - `score_executor`: `ThreadPoolExecutor(max_workers=8)`
  - `explain_executor`: `ThreadPoolExecutor(max_workers=2)`
- **Startup initialization:**
  - Artifact/model cache initialization once at startup
  - Database initialization
  - Router/dependency registration

This design keeps the event loop responsive while isolating expensive explanation workloads from scoring traffic.

## 7) Security & Access Control

- JWT authentication with role claims
- Role-based access control:
  - `/score`: authenticated users
  - `/explain`: admin only
- Expected behavior:
  - Missing/invalid token -> 401
  - Non-admin on `/explain` -> 403

## 8) Audit Logging

- Implemented as global FastAPI middleware
- Captures request/response metadata for success and failures, including:
  - 200 / 401 / 403 / 404 / 500 paths
- Normalized endpoint names (`score`, `explain`)
- Non-blocking DB writes via background thread offload
- Audit failures are isolated and do not break API responses

## 9) Concurrency Benchmark Results

### `/score` Endpoint (formal comparison)

| Scenario | Requests | Concurrency | Successful | RPS | Avg Latency (ms) | P95 Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 200 | 1 | 200 | 159.98 | 6.23 | 7.31 |
| Concurrent | 200 | 8 | 200 | 207.50 | 38.18 | 45.81 |

- Throughput improvement factor: **1.297x**
- Scaling efficiency: **16.21%**

### `/explain` Stability (after executor separation)

- Concurrent run (`100` requests, concurrency `8`):
  - Successful: `100/100`
  - RPS: `17.87`
  - P95 latency: `466.15 ms`

Operationally, splitting score/explain executors improved explain-path stability under parallel load (eliminated prior failure-heavy behavior).
Scaling efficiency reflects CPU-bound inference and Python GIL constraints; stability and isolation were prioritized over raw parallel throughput.

## 10) Batch TEST Prediction Output

Batch inference output file:

- `artifacts/test_predictions_with_explanations.csv`

Current columns:

- `Loan_ID`
- `Predicted_Loan_Status`
- `Approval_Probability`
- `Top_Tabular_Features`
- `Top_Sequence_Features`
- `Graph_Influence_Score`
- `Explanation_Text`

`Explanation_Text` is generated as concise, human-readable rationale using top feature contributors and final approval probability.
`Approval_Probability` is computed as `sigmoid(logit)` and represents the model's predicted probability of loan approval.
Decision rule: `Approved` if `Approval_Probability >= 0.5`, otherwise `Rejected`.

## 11) How To Run

### A. Train (end-to-end artifacts)

```bash
# 1) Tabular model
python3 -m training.pipelines.train_tabular

# 2) Sequence features for TRAIN+TEST coverage
python3 scripts/build_sequence_features.py \
  --train-csv data/raw/TRAIN.csv \
  --test-csv data/raw/TEST.csv \
  --output-dir data/processed

# 3) Similarity graph for TRAIN+TEST coverage
python3 scripts/build_similarity_graph.py \
  --train-csv data/raw/TRAIN.csv \
  --test-csv data/raw/TEST.csv \
  --output-dir data/graph \
  --k 10

# 4) LSTM encoder
python3 -m training.pipelines.train_lstm

# 5) GraphSAGE + precomputed node embeddings
python3 -m training.pipelines.train_graphsage

# 6) Fusion model
python3 -m training.pipelines.train_fusion
```

### B. Run API

```bash
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
```

### C. Run Batch Inference on TEST

```bash
python3 -m src.inference.batch_predict_test
```

### D. Run Fairness Evaluation

```bash
python3 -m src.evaluation.fairness_analysis
```

## 12) Project Structure Overview

```text
.
├── data/                  # Raw, processed, and graph inputs
├── artifacts/             # Trained models, embeddings, and reports
├── models/                # Tabular, sequence, graph, and fusion model code
├── training/              # Training pipelines per modality
├── scripts/               # Offline deterministic data/graph builders
├── src/
│   ├── serving/           # FastAPI app, routers, dependencies, middleware
│   ├── explainability/    # SHAP/IG/graph influence unified service
│   ├── inference/         # Batch prediction/export logic
│   ├── evaluation/        # Validation and fairness evaluation scripts
│   ├── benchmark/         # Async load generator + metrics + reporting
│   └── db/                # SQLAlchemy models/session/repositories
├── docker/                # Dockerfile and docker-compose deployment assets
└── requirements/          # Dependency specifications
```

## 13) System Guarantees

- Deterministic inference
- Role-based access isolation
- Audit coverage for 200/401/403/404/500
- No model retraining during serving
- No background mutation of artifacts

