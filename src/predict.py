import os
from pathlib import Path
from typing import Dict, List, Union

import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin  # <-- Added this import!

# ==========================================
# 0. CUSTOM TRANSFORMERS (REQUIRED TO LOAD PIPELINE)
# ==========================================

class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, time_col='TransactionStartTime'):
        self.time_col = time_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        X_out[self.time_col] = pd.to_datetime(X_out[self.time_col])
        X_out['TransactionHour'] = X_out[self.time_col].dt.hour
        X_out['TransactionDay'] = X_out[self.time_col].dt.day
        X_out['TransactionMonth'] = X_out[self.time_col].dt.month
        X_out['TransactionYear'] = X_out[self.time_col].dt.year
        return X_out.drop(columns=[self.time_col])

class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_cap=None, factor=1.5):
        self.cols_to_cap = cols_to_cap
        self.factor = factor

    def fit(self, X, y=None):
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}
        cols = self.cols_to_cap if self.cols_to_cap else X.select_dtypes(include=np.number).columns
        for col in cols:
            if col in X.columns:
                q1 = X[col].quantile(0.25)
                q3 = X[col].quantile(0.75)
                iqr = q3 - q1
                self.lower_bounds_[col] = q1 - self.factor * iqr
                self.upper_bounds_[col] = q3 + self.factor * iqr
        return self

    def transform(self, X):
        X_out = X.copy()
        cols = self.cols_to_cap if self.cols_to_cap else self.lower_bounds_.keys()
        for col in cols:
            if col in X_out.columns and col in self.lower_bounds_:
                X_out[col] = X_out[col].clip(lower=self.lower_bounds_[col], upper=self.upper_bounds_[col])
        return X_out

class CustomerAggregator(BaseEstimator, TransformerMixin):
    def __init__(self, id_col='CustomerId', target_col='Amount'):
        self.id_col = id_col
        self.target_col = target_col

    def fit(self, X, y=None):
        agg = X.groupby(self.id_col)[self.target_col].agg(
            Total_Amount='sum',
            Median_Amount='median',
            Transaction_Count='count',
            Std_Dev_Amount='std'
        ).fillna(0).reset_index()
        
        # ---> THE FIX: Convert to native Python dictionary to avoid Pandas pickling bugs! <---
        self.agg_dict_ = agg.to_dict('records')
        return self

    def transform(self, X):
        original_index = X.index 
        
        # Rebuild the dataframe on the fly
        agg_df = pd.DataFrame(self.agg_dict_)
        
        X_out = X.merge(agg_df, on=self.id_col, how='left')
        X_out.index = original_index
        new_cols = ['Total_Amount', 'Median_Amount', 'Transaction_Count', 'Std_Dev_Amount']
        X_out[new_cols] = X_out[new_cols].fillna(0)
        return X_out

class DropUselessColumns(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_drop=None):
        self.cols_to_drop = cols_to_drop or [
            'TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId', 'CountryCode'
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=[col for col in self.cols_to_drop if col in X.columns])


# ==========================================
# 1. BUSINESS LOGIC
# ==========================================

def risk_probability_to_tier(prob: float) -> str:
    """Translates raw AI probability into a business risk tier."""
    if prob < 0.10:
        return "LOW RISK"
    elif prob < 0.40:
        return "MODERATE RISK"
    elif prob < 0.70:
        return "HIGH RISK"
    else:
        return "CRITICAL - LIKELY TO DEFAULT"

def get_recommendation(prob: float) -> str:
    """Determines the automated action based on the probability."""
    if prob < 0.40:
        return "APPROVE LOAN"
    elif prob < 0.70:
        return "MANUAL REVIEW REQUIRED"
    else:
        return "DECLINE LOAN"

# ==========================================
# 2. THE INFERENCE ENGINE
# ==========================================

class CreditRiskPredictor:
    """Loads the saved pipeline and model to process live loan applications."""

    def __init__(self):
        project_root = Path(__file__).resolve().parent.parent
        pipeline_path = project_root / 'data' / 'data_pipeline.joblib'
        model_path = project_root / 'models' / 'best_credit_risk_model.joblib'

        if not pipeline_path.exists() or not model_path.exists():
            raise FileNotFoundError(
                "Missing artifacts! Ensure you have run data_processing.py "
                "AND train.py to generate the pipeline and model files."
            )

        print("Loading Data Pipeline and AI Model...")
        self.pipeline = joblib.load(pipeline_path)
        self.model = joblib.load(model_path)

    def predict(self, raw_data: Union[Dict, List[Dict]]) -> List[Dict]:
        if isinstance(raw_data, dict):
            df = pd.DataFrame([raw_data])
        else:
            df = pd.DataFrame(raw_data)

        # 1. Apply Pipeline Math
        X_processed = self.pipeline.transform(df)

        # 2. Get Probability
        probabilities = self.model.predict_proba(X_processed)[:, 1]

        # 3. Format Results
        results = []
        for i, prob in enumerate(probabilities):
            results.append({
                "customer_id": df.iloc[i].get("CustomerId", "UNKNOWN"),
                "default_probability": round(float(prob), 4),
                "risk_tier": risk_probability_to_tier(prob),
                "action": get_recommendation(prob)
            })
            
        return results

# ==========================================
# 3. LIVE TEST
# ==========================================

if __name__ == "__main__":
    sample_transaction = {
        "TransactionId": "TXN_999888",
        "BatchId": "BatchId_111",
        "AccountId": "AccountId_222",
        "SubscriptionId": "SubId_333",
        "CustomerId": "CustomerId_8", 
        "CurrencyCode": "UGX",
        "CountryCode": 256,
        "ProviderId": "ProviderId_6",
        "ProductId": "ProductId_10",
        "ProductCategory": "airtime",
        "ChannelId": "ChannelId_3",
        "Amount": 50000.0,            
        "Value": 50000.0,
        "TransactionStartTime": "2024-02-15 03:15:00", 
        "PricingStrategy": 2,
    }

    try:
        engine = CreditRiskPredictor()
        prediction = engine.predict(sample_transaction)
        
        print("\n" + "="*40)
        print(" LOAN DECISION ENGINE")
        print("="*40)
        for key, value in prediction[0].items():
            print(f" {key.replace('_', ' ').title()}: {value}")
        print("="*40)

    except Exception as e:
        print(f"\nError initializing CreditRiskPredictor: {e}")