#!/usr/bin/env python3
"""
Quick test script for Hull Tactical Market Prediction
This is a streamlined version for rapid testing and development.
"""

import sys
import os
import pandas as pd

# Add kaggle_evaluation to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'kaggle_evaluation'))

def quick_test():
    """Run a quick test of the model without the full evaluation framework"""
    print("🚀 Quick Model Test")
    print("=" * 50)
    
    try:
        # Import and train model
        import model
        print("📚 Training model...")
        model.train_model()
        print("✅ Model trained!")
        
        # Load test data
        print("📊 Loading test data...")
        import polars as pl
        test_data = pl.read_csv('test.csv')
        print(f"Test data shape: {test_data.shape}")
        
        # Make predictions on first batch
        print("🔮 Making predictions...")
        first_batch = test_data.head(1)
        predictions = model.predict(first_batch)
        
        print(f"✅ Prediction successful!")
        print(f"Sample prediction: {predictions.iloc[0]:.6f}")
        print(f"Prediction type: {type(predictions)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def full_test():
    """Run the full evaluation framework test"""
    print("🚀 Full Evaluation Test")
    print("=" * 50)
    
    try:
        from kaggle_evaluation.default_inference_server import DefaultInferenceServer
        import model
        
        # Train model
        print("📚 Training model...")
        model.train_model()
        print("✅ Model trained!")
        
        # Run full evaluation
        print("🧪 Running full evaluation...")
        inference_server = DefaultInferenceServer(model.predict)
        inference_server.run_local_gateway()
        
        # Check results
        if os.path.exists('submission.parquet'):
            df = pd.read_parquet('submission.parquet')
            print(f"✅ Success! Generated {len(df)} predictions")
            print(f"Prediction range: [{df['prediction'].min():.6f}, {df['prediction'].max():.6f}]")
            print(f"Mean prediction: {df['prediction'].mean():.6f}")
            return True
        else:
            print("❌ No submission file generated")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function - choose test type"""
    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        success = quick_test()
    else:
        success = full_test()
    
    print("=" * 50)
    if success:
        print("🎉 All tests passed!")
    else:
        print("💥 Tests failed - check errors above")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())