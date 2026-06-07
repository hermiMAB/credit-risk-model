import logging
import os
from pathlib import Path
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ==========================================
# 1. EVALUATION FUNCTION
# ==========================================

def evaluate_model(model, X: pd.DataFrame, y: pd.Series, split_name: str = "Test") -> dict:
    """Compute and log all evaluation metrics."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else y_pred

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, y_prob),
    }

    logger.info(f"\n{'='*50}\n{split_name} Results\n{'='*50}")
    for k, v in metrics.items():
        logger.info(f"  {k:12s}: {v:.4f}")
            
    logger.info(f"\n{classification_report(y, y_pred, target_names=['Low Risk (0)', 'High Risk (1)'])}")
    return metrics


# ==========================================
# 2. MODEL TRAINING FUNCTIONS (WITH TUNING)
# ==========================================

def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    logger.info("Training Logistic Regression...")
    param_grid = {"C": [0.01, 0.1, 1, 10], "solver": ["lbfgs"], "max_iter": [1000]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(
        LogisticRegression(class_weight="balanced", random_state=42),
        param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0,
    )
    gs.fit(X_train, y_train)
    logger.info(f"Best LR params: {gs.best_params_} | CV AUC: {gs.best_score_:.4f}")
    return gs.best_estimator_

def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    logger.info("Training Random Forest...")
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=42),
        param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0,
    )
    gs.fit(X_train, y_train)
    logger.info(f"Best RF params: {gs.best_params_} | CV AUC: {gs.best_score_:.4f}")
    return gs.best_estimator_

def train_xgboost(X_train, y_train) -> XGBClassifier:
    logger.info("Training XGBoost...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(
        XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric="auc",
            random_state=42,
        ),
        param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0,
    )
    gs.fit(X_train, y_train)
    logger.info(f"Best XGB params: {gs.best_params_} | CV AUC: {gs.best_score_:.4f}")
    return gs.best_estimator_


# ==========================================
# 3. MASTER EXECUTION LOOP (NOW WITH MLFLOW)
# ==========================================

def main():
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / 'data'
    model_output_dir = project_root / 'models'
    os.makedirs(model_output_dir, exist_ok=True)

    logger.info("Loading processed train and test datasets...")
    train_df = pd.read_csv(data_dir / 'train_processed.csv')
    test_df = pd.read_csv(data_dir / 'test_processed.csv')

    X_train = train_df.drop(columns=['is_high_risk'])
    y_train = train_df['is_high_risk']
    
    X_test = test_df.drop(columns=['is_high_risk'])
    y_test = test_df['is_high_risk']

    # Set up the MLflow Experiment Space
    mlflow.set_experiment("Credit_Risk_Prediction")

    # Train Models
    models = {
        "LogisticRegression": train_logistic_regression(X_train, y_train),
        "RandomForest": train_random_forest(X_train, y_train),
        "XGBoost": train_xgboost(X_train, y_train),
    }

    test_results = []
    
    # Evaluate and Log to MLflow
    for name, model in models.items():
        logger.info(f"\n--- Evaluating and Logging {name} ---")
        
        # Start MLflow run for this specific model
        with mlflow.start_run(run_name=name) as run:
            
            # 1. Log Hyperparameters
            mlflow.log_params(model.get_params())
            
            # 2. Evaluate and Log Metrics
            metrics = evaluate_model(model, X_test, y_test, split_name=f"{name} [TEST]")
            mlflow.log_metrics(metrics)
            
            # 3. Log the physical model file into MLflow
            mlflow.sklearn.log_model(model, artifact_path="model")
            
            # Save data to find the winner later
            metrics["model_name"] = name
            metrics["run_id"] = run.info.run_id
            test_results.append(metrics)

    # Find the Tournament Winner
    best = max(test_results, key=lambda x: x["roc_auc"])
    best_model = models[best["model_name"]]
    logger.info(f"\n🏆 TOURNAMENT WINNER: {best['model_name']} (Test AUC = {best['roc_auc']:.4f})")

    # ---> TASK 5: Register the best model in the MLflow Model Registry <---
    model_uri = f"runs:/{best['run_id']}/model"
    mlflow.register_model(model_uri, "Best_Credit_Risk_Model")
    logger.info(f"Winner registered in MLflow Model Registry under 'Best_Credit_Risk_Model'")

    # Also save a standard local copy for your predict.py script
    save_path = model_output_dir / "best_credit_risk_model.joblib"
    joblib.dump(best_model, save_path)
    logger.info(f"Local fallback artifact safely saved to: {save_path}")

if __name__ == "__main__":
    main()