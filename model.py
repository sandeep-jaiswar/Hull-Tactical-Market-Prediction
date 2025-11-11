"""
Simple baseline model for Hull Tactical Market Prediction
This demonstrates how to implement a predict function that works with the evaluation framework.
"""

import pandas as pd
import polars as pl
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class SimpleMarketPredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = None
        
    def train(self, train_data_path: str = "train.csv"):
        """Train the model on historical data"""
        print("Loading training data...")
        # Load with polars first, then convert to pandas for sklearn
        train_df = pl.read_csv(train_data_path)
        train_pd = train_df.to_pandas()
        
        # Get feature columns (all except target variables and date_id)
        exclude_cols = ['date_id', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns']
        self.feature_columns = [col for col in train_pd.columns if col not in exclude_cols]
        
        print(f"Using {len(self.feature_columns)} features for training")
        
        # Prepare features and target
        X = train_pd[self.feature_columns].fillna(0)  # Simple fillna for missing values
        y = train_pd['forward_returns'].fillna(0)
        
        # Remove rows where target is missing
        valid_rows = ~train_pd['forward_returns'].isna()
        X = X[valid_rows]
        y = y[valid_rows]
        
        print(f"Training on {len(X)} samples with {X.shape[1]} features")
        
        # Scale features and train model
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        print(f"Model trained successfully!")
        print(f"Model R² score: {self.model.score(X_scaled, y):.4f}")
        
    def predict_batch(self, test_batch):
        """Make predictions for a batch of test data"""
        if not self.is_trained:
            # If not trained, return simple baseline predictions
            print("Warning: Model not trained, using baseline predictions")
            return np.zeros(len(test_batch))
        
        # Convert polars to pandas if needed
        if isinstance(test_batch, pl.DataFrame):
            test_pd = test_batch.to_pandas()
        else:
            test_pd = test_batch
            
        # Extract features (same as training)
        X = test_pd[self.feature_columns].fillna(0)
        
        # Scale and predict
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        return predictions

# Global model instance
predictor = SimpleMarketPredictor()

def predict(test_batch):
    """
    Main prediction function called by the evaluation framework.
    
    Args:
        test_batch: A polars DataFrame containing test features
        
    Returns:
        predictions: A pandas Series or numpy array of predictions
    """
    import pandas as pd
    import numpy as np
    
    try:
        predictions = predictor.predict_batch(test_batch)
        print(f"Generated {len(predictions)} predictions for batch")
        
        # Return as pandas Series (expected by the evaluation framework)
        return pd.Series(predictions, name='prediction')
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        # Return zeros as fallback
        return pd.Series([0.0] * len(test_batch), name='prediction')

def train_model():
    """Train the model - call this before running predictions"""
    predictor.train()

if __name__ == "__main__":
    # Train the model when run directly
    train_model()