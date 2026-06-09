import logging
import os
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
    roc_curve, auc, ConfusionMatrixDisplay
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ==========================================
# 1. BASE MODULE TRAINING & EVALUATION
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
            
    logger.info(f"\n{classification_report(y, y_pred, target_names=['Good (0)', 'Default (1)'])}")
    return metrics

def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    logger.info("Training Logistic Regression...")
    param_grid = {"C": [0.01, 0.1, 1, 10], "solver": ["lbfgs"], "max_iter": [1000]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(
        LogisticRegression(class_weight="balanced", random_state=42),
        param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0,
    )
    gs.fit(X_train, y_train)
    return gs.best_estimator_

def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    logger.info("Training Random Forest...")
    param_grid = {"n_estimators": [100, 200], "max_depth": [5, 10, None], "min_samples_split": [2, 5]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=42),
        param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0,
    )
    gs.fit(X_train, y_train)
    return gs.best_estimator_

def train_xgboost(X_train, y_train) -> XGBClassifier:
    logger.info("Training XGBoost...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    param_grid = {"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(
        XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric="auc", random_state=42),
        param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0,
    )
    gs.fit(X_train, y_train)
    return gs.best_estimator_


# ==========================================
# 2. EXPORTABLE VISUALIZATION SUBROUTINES
# ==========================================

def plot_tournament_roc(models_dict: dict, X_test: pd.DataFrame, y_test: pd.Series):
    """Plots superimposed ROC curves for all models in the tournament."""
    plt.figure(figsize=(8, 7))
    for name, model in models_dict.items():
        probs = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
        
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.show()

def plot_model_confusion_matrix(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str):
    """Displays the confusion matrix for a specific model."""
    y_pred = model.predict(X_test)
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=['Good', 'Default'],
        cmap='Blues',
    )
    plt.title(f"{model_name} Confusion Matrix on Test Set")
    plt.show()

def plot_lr_coefficients(lr_model, feature_names: list, top_n: int = 10):
    """Plots the top N feature weights from the Logistic Regression model."""
    lr_coefs = pd.DataFrame({
        'feature': feature_names,
        'coefficient': lr_model.coef_[0],
    })
    lr_coefs['abs_coef'] = lr_coefs['coefficient'].abs()
    lr_coefs = lr_coefs.sort_values('abs_coef', ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=lr_coefs, x='coefficient', y='feature', color='steelblue')
    plt.title(f'Top {top_n} Logistic Regression Coefficients')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

def plot_tree_importances(model, feature_names: list, model_name: str, top_n: int = 10):
    """Plots global feature importances for ensemble tree models (RF or XGBoost)."""
    if hasattr(model, 'feature_importances_'):
        imp = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_,
        }).sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=imp, x='importance', y='feature', color='darkorange')
        plt.title(f'Top {top_n} Feature Importances - {model_name}')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()

def plot_shap_explanations(model, X_test: pd.DataFrame):
    """Calculates and renders global SHAP summary values for an model instance."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, plot_type='bar', show=False)
        plt.title('SHAP Feature Importance Summary')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print('SHAP visualization error:', e)


# ==========================================
# 3. EXPORTABLE PORTFOLIO RISK CALCULATOR
# ==========================================

def calculate_portfolio_loss(model, X_test: pd.DataFrame, y_test: pd.Series, df_original: pd.DataFrame):
    """Computes expected losses, evaluates risk tier populations, and prints out metrics."""
    test_probs = model.predict_proba(X_test)[:, 1]
    test_summary = pd.DataFrame({
        "Actual_Default": y_test.values,
        "PD": test_probs,
        "EAD": df_original.loc[X_test.index, "credit_amount"].values,
    })

    test_summary["LGD"] = 0.45
    test_summary["Expected_Loss"] = test_summary["PD"] * test_summary["LGD"] * test_summary["EAD"]

    def get_tier(p):
        if p < 0.20: return "LOW"
        if p < 0.40: return "MEDIUM-LOW"
        if p < 0.60: return "MEDIUM-HIGH"
        return "HIGH"

    test_summary["Risk_Tier"] = test_summary["PD"].apply(get_tier)
    
    portfolio_el = test_summary["Expected_Loss"].sum()
    portfolio_ead = test_summary["EAD"].sum()
    
    print(f"Total Portfolio EAD: {portfolio_ead:,.2f} DM")
    print(f"Total Expected Portfolio Loss (EL): {portfolio_el:,.2f} DM ({portfolio_el/portfolio_ead*100:.2f}%)")
    print("\nRisk Tier Distribution:")
    print(test_summary["Risk_Tier"].value_counts())
    
    return test_summary


def main():
    # Regular isolated script pipeline for terminal runs remains fully standard
    pass

if __name__ == "__main__":
    main()