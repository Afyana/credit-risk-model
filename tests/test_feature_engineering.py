import pytest
import pandas as pd
import numpy as np
from src.data_processing import CustomerAggregator

def test_customer_aggregation():
    """Test that aggregation returns correct columns"""
    test_data = pd.DataFrame({
        'CustomerId': ['A', 'A', 'B'],
        'Amount': [100, 200, 300],
        'TransactionStartTime': ['2024-01-01', '2024-01-02', '2024-01-01']
    })
    
    aggregator = CustomerAggregator()
    result = aggregator.transform(test_data)
    
    assert 'CustomerId' in result.columns
    assert 'Frequency' in result.columns
    assert 'Monetary_Sum' in result.columns

def test_no_empty_dataframe():
    """Test that empty dataframe raises error"""
    aggregator = CustomerAggregator()
    empty_df = pd.DataFrame()
    
    with pytest.raises(Exception):
        aggregator.transform(empty_df)