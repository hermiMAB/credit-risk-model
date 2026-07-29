# Credit Scoring Model for BNPL Lending — Bati Bank

A production-ready credit risk model that scores eCommerce customers for a Buy-Now-Pay-Later (BNPL) partnership, using behavioral transaction data in place of traditional credit history.

## Business Problem

Bati Bank is launching a BNPL service with a fast-growing eCommerce partner but has no historical loan default data for these customers. Traditional credit scoring — built on years of repayment history — doesn't work for a "thin-file" user base. Without a reliable risk signal, the bank faces a binary choice that hurts the business either way: approve everyone and absorb high default losses, or decline broadly and lose the growth opportunity entirely.

The model needed to satisfy two constraints at once: strong enough to separate good and bad risk in a genuinely BNPL-relevant way, and interpretable enough to survive Basel II-style regulatory scrutiny — no unexplainable "black box" decisions on live credit approvals.

## Solution Overview

Since no historical default label existed, the target variable was engineered rather than given:

1. **RFM proxy target** — Recency, Frequency, and Monetary features were computed per customer, then clustered with K-Means. The cluster with high recency, low frequency, and low monetary value (the "Ghost Segment") was labeled `is_high_risk = 1`, giving the model a defensible, data-driven stand-in for default risk instead of an arbitrary manual rule.
2. **Feature pipeline** — timestamp decomposition, per-customer behavioral aggregation (total/average/std-dev of transaction amount, transaction count), removal of noisy identifier columns, median imputation + `StandardScaler` for numeric features, and **Weight of Evidence (WoE)** encoding for categorical features (chosen over one-hot encoding specifically for its regulatory interpretability).
3. **Model tournament** — Logistic Regression, Random Forest, and XGBoost were trained with 5-fold stratified cross-validation and grid search, tracked end-to-end in MLflow, with class imbalance handled via `class_weight="balanced"` / `scale_pos_weight`.
4. **Deployment** — the winning model is served through a FastAPI endpoint with Pydantic request validation, containerized with Docker, and wired into a CI/CD pipeline.

## Key Results

- **Random Forest was the champion model**, edging out XGBoost despite XGBoost's typical edge on tabular data:
  - Random Forest — ROC-AUC **0.9995**
  - XGBoost — ROC-AUC **0.9979**
  - Logistic Regression (interpretable baseline) — ROC-AUC **0.9163**
- On the held-out test set, the Random Forest model correctly classified 16,857/16,931 "Good" customers and 2,145/2,202 "Default" customers (57 false negatives, 74 false positives).
- The strongest predictive signals were behavioral, not transactional size: **transaction count and total amount outrank raw transaction value** — confirming that fraud/default risk is non-linear and can't be read off transaction size alone (Spearman correlation between raw amount and target was just 0.07–0.08).
- Estimated business impact: **~15% increase in safe loan approvals** at the current default rate, by replacing a one-size-fits-all approval policy with the model's three-tier risk system (Low / Medium / High).

## Quick Start

```bash
git clone https://github.com/hermiMAB/credit-risk-model
cd credit-risk-model
pip install -r requirements.txt
dvc pull          # pulls the versioned dataset
python src/main.py
```

Run with Docker instead:

```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000/docs` (Swagger UI) for interactive testing of the `/predict` endpoint.

## Project Structure

```
credit-risk-model/
├── .dvc/                  # Data version control metadata
├── .github/workflows/     # CI/CD pipeline (GitHub Actions)
├── data/                  # Raw & processed data (tracked via DVC, not in Git)
├── mlruns/                # MLflow experiment tracking artifacts
├── models/                # Serialized production model (.joblib)
├── notebooks/             # Exploratory data analysis
├── src/                   # Modular pipeline & FastAPI service code
├── tests/                 # Unit tests
├── docker-compose.yml
├── dockerfile
├── dvc.yaml / dvc.lock    # Data & pipeline versioning
├── mlflow.db              # MLflow tracking store
└── requirements.txt
```

## Demo

`POST /predict` — example request/response from the live Swagger UI:

**Request**
```json
{
  "TransactionId": "TXN_999888",
  "CustomerId": "CustomerId_8",
  "ProductCategory": "airtime",
  "Amount": 50000.0,
  "Value": 50000.0,
  "TransactionStartTime": "2024-02-15 03:15:00",
  "PricingStrategy": 2
}
```

**Response**
```json
{
  "transaction_id": "TXN_999888",
  "customer_id": "CustomerId_8",
  "default_probability": 0.025,
  "risk_tier": "LOW_RISK",
  "action": "APPROVE_LOAN"
}
```

Risk-tier policy applied downstream of the score:

| Tier | Probability | Action |
|---|---|---|
| Low Risk | < 5% | Auto-approve, full credit limit, 30-day term |
| Medium Risk | 5–20% | Secondary verification, 50% credit limit, 14-day term |
| High Risk | > 20% | Auto-decline |

## Technical Details

**Data**: 95,662 eCommerce transactions from the Xente Fraud Detection dataset (Zindi/Kaggle), 16 raw features, 100% complete (no missing values), heavily right-skewed transaction amounts requiring robust scaling.

**Target engineering**: RFM behavioral clustering (K-Means) used to derive a proxy `is_high_risk` label in the absence of ground-truth default data.

**Model**: Random Forest classifier (winner of a 3-way tournament vs. XGBoost and Logistic Regression), tuned via `GridSearchCV` over `n_estimators` and `max_depth`, trained on WoE-encoded categorical features and scaled numeric features.

**Evaluation**: 5-fold stratified cross-validation; Precision, Recall, F1, and ROC-AUC used instead of accuracy due to class imbalance in the target.

**MLOps**: Every run's hyperparameters, metrics, and model artifacts are logged to MLflow; the best model by ROC-AUC is auto-promoted to the MLflow Model Registry and exported for the production API.

## Future Improvements

- Replace the RFM-based proxy label with real repayment/delinquency data once available, removing the dependency on a behavioral heuristic
- Incorporate alternative data sources (mobile usage, utility payments) to strengthen signal for thin-file users
- Add SMOTE or cost-sensitive resampling to further improve minority-class (default) detection
- Add adversarial robustness testing to guard against users gaming the behavioral proxy
- Add SHAP-based local explainability to the API response for individual "why was I declined" transparency
- Ongoing fairness auditing of proxy features to catch unintended correlation with protected demographic attributes

## Author

Hermela — [GitHub](https://github.com/hermiMAB)