import pandas as pd
import numpy as np
import sys
import os

# This line ensures the test folder can see your src folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing import TimeFeatureExtractor, OutlierCapper

def test_time_feature_extractor():
    """Test that the TimeFeatureExtractor creates the correct date columns."""
    # 1. Create fake data
    df = pd.DataFrame({
        'TransactionStartTime': ['2024-02-15 03:15:00', '2024-12-25 14:30:00']
    })
    
    # 2. Run the function
    extractor = TimeFeatureExtractor(time_col='TransactionStartTime')
    df_transformed = extractor.transform(df)
    
    # 3. Assert (check) that it did its job
    assert 'TransactionHour' in df_transformed.columns
    assert 'TransactionMonth' in df_transformed.columns
    assert df_transformed['TransactionMonth'].iloc[0] == 2  # February
    assert df_transformed['TransactionHour'].iloc[1] == 14  # 2 PM

def test_outlier_capper():
    """Test that the OutlierCapper successfully limits extreme values."""
    # 1. Create fake data with a massive outlier (1,000,000)
    df = pd.DataFrame({
        'Amount': [10, 12, 11, 15, 14, 1,000,000]
    })
    
    # 2. Run the function
    capper = OutlierCapper(cols_to_cap=['Amount'], factor=1.5)
    df_transformed = capper.fit_transform(df)
    
    # 3. Assert that the 1,000,000 was capped to a normal number
    max_amount = df_transformed['Amount'].max()
    assert max_amount < 1000000 
    assert max_amount > 0