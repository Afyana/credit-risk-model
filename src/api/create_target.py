"""
Proxy Target Variable Engineering using RFM Analysis and K-Means Clustering
Identifies high-risk customers based on engagement patterns
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RFMTargetCreator:
    """
    Creates binary target variable using RFM segmentation and K-Means clustering
    """
    
    def __init__(self, random_state=42, n_clusters=3):
        self.random_state = random_state
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        
    def calculate_rfm(self, customer_df):
        """Calculate Recency, Frequency, Monetary values"""
        logger.info("Calculating RFM metrics...")
        
        # RFM values should already be in customer_df from aggregation
        rfm_df = customer_df[['CustomerId', 'Recency', 'Frequency', 'Monetary_Sum']].copy()
        rfm_df.columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary']
        
        # Log transform skewed monetary values
        rfm_df['Monetary_Log'] = np.log1p(rfm_df['Monetary'])
        
        logger.info(f"RFM statistics:\n{rfm_df[['Recency', 'Frequency', 'Monetary']].describe()}")
        
        return rfm_df
    
    def perform_clustering(self, rfm_df):
        """Perform K-Means clustering on RFM features"""
        logger.info("Performing K-Means clustering...")
        
        # Select features for clustering
        features_for_clustering = ['Recency', 'Frequency', 'Monetary_Log']
        X = rfm_df[features_for_clustering].copy()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform clustering
        clusters = self.kmeans.fit_predict(X_scaled)
        rfm_df['Cluster'] = clusters
        
        # Analyze clusters to identify high-risk group
        cluster_profiles = rfm_df.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean',
            'CustomerId': 'count'
        }).round(2)
        
        cluster_profiles.columns = ['Avg_Recency', 'Avg_Frequency', 'Avg_Monetary', 'Count']
        cluster_profiles['Risk_Level'] = cluster_profiles['Avg_Recency'].rank(ascending=False).map(
            lambda x: 'High' if x == cluster_profiles['Avg_Recency'].rank(ascending=False).max() else 'Low'
        )
        
        logger.info(f"Cluster profiles:\n{cluster_profiles}")
        
        # High-risk cluster is the one with highest recency, lowest frequency
        high_risk_cluster = cluster_profiles[cluster_profiles['Risk_Level'] == 'High'].index[0]
        logger.info(f"High-risk cluster identified: Cluster {high_risk_cluster}")
        
        return rfm_df, high_risk_cluster
    
    def create_target_variable(self, rfm_df, high_risk_cluster):
        """Create binary target variable"""
        rfm_df['is_high_risk'] = (rfm_df['Cluster'] == high_risk_cluster).astype(int)
        
        risk_distribution = rfm_df['is_high_risk'].value_counts()
        logger.info(f"Target distribution:\n{risk_distribution}")
        logger.info(f"High-risk customers: {risk_distribution[1]} ({risk_distribution[1]/len(rfm_df)*100:.1f}%)")
        
        return rfm_df[['CustomerId', 'Cluster', 'is_high_risk']]
    
    def visualize_clusters(self, rfm_df, save_path='C:/Users/hp/Desktop/credit_risk_data/plots/cluster_analysis.png'):
        """Create visualization of cluster characteristics"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Prepare data for plotting
        cluster_cols = ['Recency', 'Frequency', 'Monetary']
        
        for idx, col in enumerate(cluster_cols):
            sns.boxplot(data=rfm_df, x='Cluster', y=col, ax=axes[idx])
            axes[idx].set_title(f'{col} by Cluster', fontsize=12, fontweight='bold')
            if col == 'Monetary':
                axes[idx].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Cluster visualization saved to {save_path}")
        plt.show()
        
    def fit_transform(self, customer_df):
        """Complete pipeline to create target variable"""
        # Calculate RFM
        rfm_df = self.calculate_rfm(customer_df)
        
        # Perform clustering
        rfm_df, high_risk_cluster = self.perform_clustering(rfm_df)
        
        # Visualize results
        self.visualize_clusters(rfm_df)
        
        # Create target
        target_df = self.create_target_variable(rfm_df, high_risk_cluster)
        
        # Merge with original customer data
        final_df = customer_df.merge(target_df, on='CustomerId', how='left')
        
        logger.info("Target variable engineering complete")
        return final_df


if __name__ == "__main__":
    # Load processed customer data
    customer_df = pd.read_csv('C:/Users/hp/Desktop/credit_risk_data/data/processed/processed_customers.csv')
    
    # Create target variable
    target_creator = RFMTargetCreator(random_state=42, n_clusters=3)
    final_dataset = target_creator.fit_transform(customer_df)
    
    # Save final dataset
    final_dataset.to_csv('C:/Users/hp/Desktop/credit_risk_data/data/processed/final_training_data.csv', index=False)
    logger.info(f"Final training data saved with shape {final_dataset.shape}")
    
    # Display target distribution
    print(f"\nTarget distribution:\n{final_dataset['is_high_risk'].value_counts()}")
    print(f"\nFinal columns: {final_dataset.columns.tolist()}")