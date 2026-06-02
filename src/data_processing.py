"""
Feature Engineering Pipeline for Credit Risk Modeling
Transforms raw transaction data into model-ready customer-level features
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from datetime import datetime, timezone
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CustomerAggregator(BaseEstimator, TransformerMixin):
    """
    Custom transformer to aggregate transaction data to customer level
    Creates RFM (Recency, Frequency, Monetary) and behavioral features
    """
    
    def __init__(self, snapshot_date=None):
        self.snapshot_date = snapshot_date
        
    def fit(self, X, y=None):
        return self
    
    def _get_snapshot_date(self, df):
        """Get appropriate snapshot date handling timezone"""
        if self.snapshot_date is not None:
            snapshot = self.snapshot_date
        else:
            # Get max date from data and handle timezone
            max_date = df['TransactionStartTime'].max()
            
            # If max_date is timezone-aware, make snapshot timezone-aware too
            if max_date.tzinfo is not None:
                snapshot = datetime.now(max_date.tzinfo)
            else:
                snapshot = datetime.now()
        
        return snapshot
    
    def transform(self, X, y=None):
        logger.info("Aggregating transaction data to customer level...")
        
        # Ensure datetime format and handle timezone
        X = X.copy()
        X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])
        
        # Get appropriate snapshot date
        snapshot_date = self._get_snapshot_date(X)
        
        # Group by CustomerId
        customer_features = X.groupby('CustomerId').agg({
            'TransactionId': 'count',
            'Amount': ['sum', 'mean', 'std', 'min', 'max'],
            'FraudResult': 'sum',
            'TransactionStartTime': ['min', 'max']
        }).round(2)
        
        # Flatten column names
        customer_features.columns = [
            'Frequency', 'Monetary_Sum', 'Monetary_Mean', 'Monetary_Std',
            'Monetary_Min', 'Monetary_Max', 'Fraud_Count', 
            'First_Transaction_Date', 'Last_Transaction_Date'
        ]
        
        # Convert date columns to datetime
        customer_features['First_Transaction_Date'] = pd.to_datetime(customer_features['First_Transaction_Date'])
        customer_features['Last_Transaction_Date'] = pd.to_datetime(customer_features['Last_Transaction_Date'])
        
        # Calculate Recency (days since last transaction) - FIXED TIMEZONE ISSUE
        # Make snapshot_date timezone-aware if needed
        if customer_features['Last_Transaction_Date'].iloc[0].tzinfo is not None:
            # If data is tz-aware, make snapshot tz-aware
            if snapshot_date.tzinfo is None:
                snapshot_date = snapshot_date.replace(tzinfo=customer_features['Last_Transaction_Date'].iloc[0].tzinfo)
        else:
            # If data is tz-naive, make snapshot tz-naive
            if snapshot_date.tzinfo is not None:
                snapshot_date = snapshot_date.replace(tzinfo=None)
        
        # Calculate recency
        customer_features['Recency'] = (snapshot_date - customer_features['Last_Transaction_Date']).dt.days
        
        # Calculate customer tenure
        customer_features['Tenure_Days'] = (customer_features['Last_Transaction_Date'] - 
                                            customer_features['First_Transaction_Date']).dt.days
        
        # Handle negative or zero tenure (should not happen, but just in case)
        customer_features['Tenure_Days'] = customer_features['Tenure_Days'].clip(lower=1)
        
        # Calculate average transaction frequency (transactions per day)
        customer_features['Frequency_Daily'] = customer_features['Frequency'] / customer_features['Tenure_Days']
        
        # Coefficient of Variation (risk indicator)
        # Handle cases where Monetary_Mean is 0
        customer_features['Monetary_CV'] = customer_features['Monetary_Std'] / (customer_features['Monetary_Mean'] + 1e-6)
        customer_features['Monetary_CV'] = customer_features['Monetary_CV'].replace([np.inf, -np.inf], 0).fillna(0)
        
        # Fraud rate
        customer_features['Fraud_Rate'] = customer_features['Fraud_Count'] / customer_features['Frequency']
        
        # Additional useful features
        customer_features['Monetary_Range'] = customer_features['Monetary_Max'] - customer_features['Monetary_Min']
        customer_features['Avg_Txn_Value'] = customer_features['Monetary_Sum'] / customer_features['Frequency']
        
        logger.info(f"Created {customer_features.shape[1]} customer-level features for {customer_features.shape[0]} customers")
        logger.info(f"Recency range: {customer_features['Recency'].min()} to {customer_features['Recency'].max()} days")
        
        return customer_features.reset_index()


class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract time-based features from transaction timestamps"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        logger.info("Extracting temporal features...")
        
        X = X.copy()
        X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])
        
        # Extract time components (timezone-naive for consistent extraction)
        # Convert to naive if needed for hour/day extraction
        if X['TransactionStartTime'].dt.tz is not None:
            tz_aware_dates = X['TransactionStartTime']
            # Convert to naive for feature extraction (hours/days don't need timezone)
            X['TransactionStartTime_Naive'] = tz_aware_dates.dt.tz_localize(None)
        else:
            X['TransactionStartTime_Naive'] = X['TransactionStartTime']
        
        # Extract time components from naive datetime
        X['Hour'] = X['TransactionStartTime_Naive'].dt.hour
        X['DayOfWeek'] = X['TransactionStartTime_Naive'].dt.dayofweek
        X['DayOfMonth'] = X['TransactionStartTime_Naive'].dt.day
        X['Month'] = X['TransactionStartTime_Naive'].dt.month
        X['Year'] = X['TransactionStartTime_Naive'].dt.year
        X['Weekend'] = (X['DayOfWeek'] >= 5).astype(int)
        X['BusinessHours'] = ((X['Hour'] >= 9) & (X['Hour'] <= 17)).astype(int)
        
        # Time since previous transaction (per customer)
        X = X.sort_values(['CustomerId', 'TransactionStartTime'])
        
        # Calculate time difference in hours
        X['TimeSincePrevTransaction'] = X.groupby('CustomerId')['TransactionStartTime'].diff().dt.total_seconds() / 3600
        
        # Fill first transaction for each customer with 0
        X['TimeSincePrevTransaction'] = X['TimeSincePrevTransaction'].fillna(0)
        
        # Clean up
        X = X.drop('TransactionStartTime_Naive', axis=1)
        
        logger.info(f"Temporal features extracted: Hour range {X['Hour'].min()}-{X['Hour'].max()}")
        return X


def create_full_feature_pipeline():
    """
    Create complete feature engineering pipeline
    
    Note: This pipeline works at transaction level. For customer-level aggregation,
    use CustomerAggregator separately.
    """
    
    # Numerical features to scale
    numerical_features = [
        'Amount', 'Hour', 'DayOfWeek', 'DayOfMonth', 'Month',
        'TimeSincePrevTransaction'
    ]
    
    # Categorical features to encode
    categorical_features = [
        'ProductCategory', 'ChannelId', 'PricingStrategy'
    ]
    
    # Preprocessing for numerical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Drop columns not specified
    )
    
    # Complete pipeline
    full_pipeline = Pipeline(steps=[
        ('temporal', TemporalFeatureExtractor()),
        ('preprocessor', preprocessor)
    ])
    
    return full_pipeline


def process_full_dataset(raw_data_path, output_path, sample_size=None):
    """
    Complete data processing function that:
    1. Aggregates to customer level
    2. Creates RFM features
    3. Prepares for modeling
    """
    logger.info(f"Loading data from {raw_data_path}")
    
    # Load data
    if sample_size:
        df = pd.read_csv(raw_data_path, nrows=sample_size)
    else:
        df = pd.read_csv(raw_data_path)
    
    logger.info(f"Loaded {len(df)} transactions")
    
    # Step 1: Aggregate to customer level
    aggregator = CustomerAggregator()
    customer_df = aggregator.transform(df)
    
    # Step 2: Save intermediate result
    customer_df.to_csv(output_path.replace('.csv', '_customer_level.csv'), index=False)
    logger.info(f"Customer-level data saved")
    
    return customer_df


if __name__ == "__main__":
    # Test the pipeline
    import os
    
    # Create directories if they don't exist
    os.makedirs('C:/Users/hp/Desktop/credit_risk_data/data/processed', exist_ok=True)
    
    # Load data (use a sample for testing)
    df = pd.read_csv('C:/Users/hp/Desktop/credit_risk_data/data.csv', nrows=50000)  # Sample for testing
    
    # Test Customer Aggregator
    logger.info("\n" + "="*50)
    logger.info("Testing CustomerAggregator")
    logger.info("="*50)
    
    aggregator = CustomerAggregator()
    customer_df = aggregator.transform(df)
    
    print(f"\nCustomer DataFrame shape: {customer_df.shape}")
    print(f"\nFirst 5 rows:\n{customer_df.head()}")
    print(f"\nColumn names: {customer_df.columns.tolist()}")
    
    # Check for any null values
    print(f"\nNull values per column:\n{customer_df.isnull().sum()}")
    
    # Test Temporal Feature Extractor
    logger.info("\n" + "="*50)
    logger.info("Testing TemporalFeatureExtractor")
    logger.info("="*50)
    
    temporal_extractor = TemporalFeatureExtractor()
    df_with_features = temporal_extractor.transform(df.head(1000))
    
    print(f"\nTemporal features added: {[col for col in df_with_features.columns if col in ['Hour', 'DayOfWeek', 'Month', 'Weekend', 'BusinessHours']]}")
    
    # Save processed data
    customer_df.to_csv('C:/Users/hp/Desktop/credit_risk_data/data/processed/processed_customers.csv', index=False)
    logger.info("\nProcessed data saved to C:/Users/hp/Desktop/credit_risk_data/data/processed/processed_customers.csv")
    
    # Save feature list for later use
    feature_list = [col for col in customer_df.columns if col not in ['CustomerId', 'First_Transaction_Date', 'Last_Transaction_Date']]
    with open('C:/Users/hp/Desktop/credit_risk_data/data/processed/feature_columns.txt', 'w') as f:
        for feat in feature_list:
            f.write(f"{feat}\n")
    
    logger.info(f"Feature list saved with {len(feature_list)} features")