"""
Enhanced model with better feature engineering and validation
"""

import pandas as pd
import polars as pl
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

class AdvancedMarketPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = None
        self.feature_importance = None
        
    def engineer_features(self, df):
        """Create additional features from the base features"""
        df_eng = df.copy()
        
        # Get numeric columns (exclude categorical and target columns)
        numeric_cols = [col for col in df.columns if col not in ['date_id']]
        
        # Add rolling statistics for key features (if we have enough data)
        if len(df) > 5:
            for col in ['M1', 'M2', 'E1', 'P1', 'V1', 'S1']:
                if col in df.columns:
                    df_eng[f'{col}_ma3'] = df[col].rolling(window=3, min_periods=1).mean()
                    df_eng[f'{col}_std3'] = df[col].rolling(window=3, min_periods=1).std().fillna(0)
        
        # Add interaction features between key categories
        if 'M1' in df.columns and 'V1' in df.columns:
            df_eng['M1_V1_interaction'] = df['M1'] * df['V1']
        if 'E1' in df.columns and 'P1' in df.columns:
            df_eng['E1_P1_interaction'] = df['E1'] * df['P1']
            
        return df_eng
        
    def train(self, train_data_path: str = "train.csv"):
        """Train the model on historical data with enhanced features"""
        print("Loading training data...")
        train_df = pl.read_csv(train_data_path)
        train_pd = train_df.to_pandas()
        
        print(f"Original training data shape: {train_pd.shape}")
        
        # Remove rows with missing target
        train_pd = train_pd.dropna(subset=['forward_returns'])
        print(f"After removing missing targets: {train_pd.shape}")
        
        # Engineer features
        train_eng = self.engineer_features(train_pd)
        
        # Get feature columns (all except target variables and date_id)
        exclude_cols = ['date_id', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns']
        self.feature_columns = [col for col in train_eng.columns if col not in exclude_cols]
        
        print(f"Using {len(self.feature_columns)} features for training")
        
        # Prepare features and target
        X = train_eng[self.feature_columns].fillna(0)
        y = train_eng['forward_returns']
        
        print(f"Training on {len(X)} samples with {X.shape[1]} features")
        
        # Scale features and train model
        X_scaled = self.scaler.fit_transform(X)
        
        # Cross-validation before final training
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, scoring='r2')
        print(f"Cross-validation R² scores: {cv_scores}")
        print(f"Mean CV R² score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Final training
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Get feature importance
        feature_importance = self.model.feature_importances_
        self.feature_importance = dict(zip(self.feature_columns, feature_importance))
        
        # Show top 10 most important features
        top_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        print("\nTop 10 most important features:")
        for feature, importance in top_features:
            print(f"  {feature}: {importance:.4f}")
        
        train_score = self.model.score(X_scaled, y)
        print(f"\nFinal training R² score: {train_score:.4f}")
        
    def predict_batch(self, test_batch):
        """Make predictions for a batch of test data"""
        if not self.is_trained:
            print("Warning: Model not trained, using baseline predictions")
            return np.zeros(len(test_batch))
        
        # Convert polars to pandas if needed
        if isinstance(test_batch, pl.DataFrame):
            test_pd = test_batch.to_pandas()
        else:
            test_pd = test_batch.copy()
            
        # Engineer features
        test_eng = self.engineer_features(test_pd)
        
        # Extract features (same as training)
        X = test_eng[self.feature_columns].fillna(0)
        
        # Scale and predict
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        return predictions

# Choose which predictor to use
predictor = AdvancedMarketPredictor()  # Use the advanced version

def predict(test_batch):
    """
    Main prediction function called by the evaluation framework.
    """
    try:
        predictions = predictor.predict_batch(test_batch)
        print(f"Generated {len(predictions)} predictions for batch")
        return predictions.tolist()
    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return [0.0] * len(test_batch)

def train_model():
    """Train the model - call this before running predictions"""
    predictor.train()

if __name__ == "__main__":
    train_model()