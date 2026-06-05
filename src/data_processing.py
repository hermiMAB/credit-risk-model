import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import category_encoders as ce
import pandas as pd
import numpy as np

def generate_iv_report(df: pd.DataFrame, target_col: str, categorical_cols: list) -> pd.DataFrame:
    """Calculates the Information Value (IV) for categorical features."""
    
    # Total counts of fraud (1) and normal (0)
    total_fraud = (df[target_col] == 1).sum()
    total_normal = (df[target_col] == 0).sum()
    
    for col in categorical_cols:
        if col not in df.columns:
            continue
            
        # Group by the category and count frauds vs normals
        stats = df.groupby(col)[target_col].agg(
            fraud_count=lambda x: (x == 1).sum(),
            normal_count=lambda x: (x == 0).sum()
        ).reset_index()
        
        # Calculate distributions (adding 0.0001 to prevent dividing by zero!)
        stats['dist_fraud'] = (stats['fraud_count'] + 0.0001) / (total_fraud + 0.0001)
        stats['dist_normal'] = (stats['normal_count'] + 0.0001) / (total_normal + 0.0001)
        
        # Calculate WoE and IV
        stats['woe'] = np.log(stats['dist_fraud'] / stats['dist_normal'])
        stats['iv'] = (stats['dist_fraud'] - stats['dist_normal']) * stats['woe']
        
        # Sum the IV for the whole column
        total_iv = stats['iv'].sum()
        
        # Grade the feature
        if total_iv < 0.02: grade = "Useless (Drop)"
        elif total_iv < 0.1: grade = "Weak"
        elif total_iv < 0.3: grade = "Medium"
        elif total_iv < 0.5: grade = "Strong"
        else: grade = "Suspiciously Strong (Check for Leakage)"
            
        iv_results.append({'Feature': col, 'IV_Score': round(total_iv, 4), 'Predictive_Power': grade})
        
    # Return a nicely sorted DataFrame
    return pd.DataFrame(iv_results).sort_values(by='IV_Score', ascending=False)
# ==========================================
# 1. CUSTOM TRANSFORMERS FOR FEATURE ENGINEERING
# ==========================================
# (These remain exactly the same as before)

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
        self.agg_df_ = X.groupby(self.id_col)[self.target_col].agg(
            Total_Amount='sum',
            Median_Amount='median',
            Transaction_Count='count',
            Std_Dev_Amount='std'
        ).fillna(0).reset_index()
        return self

    def transform(self, X):
# 1. Save the original shuffled index
        original_index = X.index 
        
        # 2. Perform the merge (which resets the index)
        X_out = X.merge(self.agg_df_, on=self.id_col, how='left')
        
        # 3. Put the original index back!
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
# 2. MASTER PIPELINE ARCHITECTURE (UPDATED)
# ==========================================

def build_data_pipeline(use_woe=True):
    """
    Builds and returns the complete sklearn data processing pipeline.
    Args:
        use_woe (bool): If True, uses Weight of Evidence encoding. 
                        If False, uses One-Hot Encoding.
    """
    numeric_features = [
        'Amount', 'Value', 
        'Total_Amount', 'Median_Amount', 'Transaction_Count', 'Std_Dev_Amount', 
        'TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear'
    ]
    
    categorical_features = ['ProviderId', 'ProductId', 'ProductCategory', 'ChannelId', 'PricingStrategy']

    # Numeric Track
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical Track Toggle Logic
    if use_woe:
        # Use Weight of Evidence
        encoder_step = ('encoder', ce.WOEEncoder())
    else:
        # Use One-Hot Encoding. sparse_output=False is required to output pandas DataFrames nicely.
        # handle_unknown='ignore' prevents crashes if a new category appears in the test set.
        encoder_step = ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        encoder_step
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop' 
    )
    preprocessor.set_output(transform="pandas")

    full_pipeline = Pipeline(steps=[
        ('time_extractor', TimeFeatureExtractor()),
        ('capper', OutlierCapper(cols_to_cap=['Amount', 'Value'])),
        ('aggregator', CustomerAggregator()),
        ('drop_useless', DropUselessColumns()),
        ('preprocessor', preprocessor)
    ])
    
    return full_pipeline


# ==========================================
# 3. EXECUTION AND LEAKAGE PREVENTION
# ==========================================

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    raw_data_path = project_root / 'data' / 'data.csv'
    output_dir = project_root / 'data'
    
    try:
        print(f"Loading raw data from: {raw_data_path}")
        df = pd.read_csv(raw_data_path)
        
        X = df.drop(columns=['FraudResult'])
        y = df['FraudResult']
        
        print("Splitting data to prevent data leakage...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # ---> THE TOGGLE IS HERE <---
        # Try changing this to use_woe=False to test One-Hot Encoding!
        print("Building and fitting the pipeline...")
        pipeline = build_data_pipeline(use_woe=True) 
        
        X_train_processed = pipeline.fit_transform(X_train, y_train)
        X_test_processed = pipeline.transform(X_test)
        
        X_train_processed['FraudResult'] = y_train.values
        X_test_processed['FraudResult'] = y_test.values
        
        os.makedirs(output_dir, exist_ok=True)
        train_path = output_dir / 'train_processed.csv'
        test_path = output_dir / 'test_processed.csv'
        
        X_train_processed.to_csv(train_path, index=False)
        X_test_processed.to_csv(test_path, index=False)
        
        # Show a quick summary of the columns generated
        print(f"\nSuccess! Pipeline ran with {X_train_processed.shape[1] - 1} features generated.")
        print(f"Saved to:\n - {train_path}\n - {test_path}")
        
    except FileNotFoundError:
        print(f"\nERROR: Could not find 'data.csv' at {raw_data_path}")