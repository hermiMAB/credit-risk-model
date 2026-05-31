import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import category_encoders as ce

# ==========================================
# 1. CUSTOM TRANSFORMERS FOR FEATURE ENGINEERING
# ==========================================

class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Converts string timestamps to datetime and extracts time-based features."""
    def __init__(self, time_col='TransactionStartTime'):
        self.time_col = time_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        X_out[self.time_col] = pd.to_datetime(X_out[self.time_col])
        
        # Extract features
        X_out['TransactionHour'] = X_out[self.time_col].dt.hour
        X_out['TransactionDay'] = X_out[self.time_col].dt.day
        X_out['TransactionMonth'] = X_out[self.time_col].dt.month
        X_out['TransactionYear'] = X_out[self.time_col].dt.year
        
        # Drop the original string column
        return X_out.drop(columns=[self.time_col])


class CustomerAggregator(BaseEstimator, TransformerMixin):
    """Groups by CustomerId to create historical behavioral metrics."""
    def __init__(self, id_col='CustomerId', target_col='Amount'):
        self.id_col = id_col
        self.target_col = target_col

    def fit(self, X, y=None):
        # Calculate historical stats during the 'fit' phase
        self.agg_df_ = X.groupby(self.id_col)[self.target_col].agg(
            Total_Amount='sum',
            Average_Amount='mean',
            Transaction_Count='count',
            Std_Dev_Amount='std'
        ).fillna(0).reset_index()
        return self

    def transform(self, X):
        # Merge the historical stats into the dataset during 'transform'
        X_out = X.merge(self.agg_df_, on=self.id_col, how='left')
        
        # If a new customer appears in test data, fill their missing aggregates with 0
        new_cols = ['Total_Amount', 'Average_Amount', 'Transaction_Count', 'Std_Dev_Amount']
        X_out[new_cols] = X_out[new_cols].fillna(0)
        return X_out


class DropUselessColumns(BaseEstimator, TransformerMixin):
    """Drops unique row IDs and columns with zero variance."""
    def __init__(self, cols_to_drop=None):
        # We include CountryCode here because our EDA showed it was 100% constant (256)
        self.cols_to_drop = cols_to_drop or [
            'TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId', 'CountryCode'
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Drop columns if they exist in the dataframe
        return X.drop(columns=[col for col in self.cols_to_drop if col in X.columns])


# ==========================================
# 2. MASTER PIPELINE ARCHITECTURE
# ==========================================

def build_data_pipeline():
    """Builds and returns the complete sklearn data processing pipeline."""
    
    # 1. Define the features that will exist AFTER our custom extractors run
    numeric_features = [
        'Amount', 'Value', 
        'Total_Amount', 'Average_Amount', 'Transaction_Count', 'Std_Dev_Amount', 
        'TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear'
    ]
    
    # We include all repeating IDs and Categories here to be processed by WoE
    categorical_features = ['ProviderId', 'ProductId', 'ProductCategory', 'ChannelId', 'PricingStrategy']

    # 2. Setup the Numeric Transformation Rules
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 3. Setup the Categorical Transformation Rules (Using WoE instead of One-Hot)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('woe', ce.WOEEncoder()) # Calculates risk ratio based on Target variable
    ])

    # 4. Bundle into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop' # Automatically drops anything we forgot to explicitly handle
    )
    preprocessor.set_output(transform="pandas")


    # 5. Create the Master Pipeline
    full_pipeline = Pipeline(steps=[
        ('time_extractor', TimeFeatureExtractor()),
        ('aggregator', CustomerAggregator()),
        ('drop_useless', DropUselessColumns()),
        ('preprocessor', preprocessor)
    ])
    
    # Configure the pipeline to output a Pandas DataFrame instead of a Numpy Array

    return full_pipeline

if __name__ == "__main__":
    import os
    from pathlib import Path
    
    # 1. BULLETPROOF PATHING
    # This finds the exact folder where data_processing.py lives (the src folder)
    # Then it goes up one level (.parent) to find the main project folder.
    project_root = Path(__file__).resolve().parent.parent
    
    # Set the absolute paths
    raw_data_path = project_root / 'data' / 'data.csv'
    output_dir = project_root / 'data'
    processed_data_path = output_dir / 'processed_data.csv'
    
    print(f"Looking for raw data at: {raw_data_path}")
    
    try:
        # 2. RUN THE PIPELINE
        df = pd.read_csv(raw_data_path)
        
        X = df.drop(columns=['FraudResult'])
        y = df['FraudResult']
        
        print("Building and applying the processing pipeline...")
        pipeline = build_data_pipeline()
        X_processed = pipeline.fit_transform(X, y)
        
        X_processed['FraudResult'] = y.values
        
        # 3. SAVE THE OUTPUT
        os.makedirs(output_dir, exist_ok=True)
        X_processed.to_csv(processed_data_path, index=False)
        print(f"Success! Processed data saved to '{processed_data_path}'")
        
    except FileNotFoundError:
        print(f"\nERROR: Could not find 'data.csv' at {raw_data_path}")
        print("Please ensure the 'data' folder exists in your project root and contains 'data.csv'.")