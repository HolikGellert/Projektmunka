import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from .config import Config

class EnergyClusterer:
    def __init__(self, n_clusters=Config.N_CLUSTERS):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=Config.RANDOM_STATE, n_init=10)

    def fit_predict(self, df: pd.DataFrame, features: list) -> pd.Series:
        """Standard KMeans clustering."""
        X = df[features]
        labels = self.model.fit_predict(X)
        return pd.Series(labels, index=df.index, name='cluster')

    def fit_time_based(self, df: pd.DataFrame, features: list, season: str = None) -> pd.Series:
        """
        Performs clustering only on a specific time segment (e.g., 'Winter').
        Returns labels for the whole dataframe (NaN for non-matching rows).
        """
        if season:
            subset = df[df['season_name'] == season].copy()
        else:
            subset = df.copy()
            
        X_subset = subset[features]
        
        # Fit local model for this time slice
        local_kmeans = KMeans(n_clusters=self.n_clusters, random_state=Config.RANDOM_STATE, n_init=10)
        subset_labels = local_kmeans.fit_predict(X_subset)
        
        # Return Series aligned with original index
        full_series = pd.Series(index=df.index, dtype=float)
        full_series.loc[subset.index] = subset_labels
        
        return full_series

    def evaluate(self, df: pd.DataFrame, features: list, labels: pd.Series):
        """Returns clustering metrics."""
        # Filter out NaNs (from time-based clustering)
        valid_idx = labels.dropna().index
        X = df.loc[valid_idx, features]
        y = labels.loc[valid_idx]
        
        # Sampling for large datasets
        if len(X) > 10000:
            X = X.sample(10000, random_state=42)
            y = y.loc[X.index]

        return {
            'silhouette': silhouette_score(X, y),
            'davies_bouldin': davies_bouldin_score(X, y)
        }