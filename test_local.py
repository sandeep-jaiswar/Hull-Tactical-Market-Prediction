"""
Local test runner for Hull Tactical Market Prediction
This script sets up and runs the evaluation framework locally to test your model.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'kaggle_evaluation'))

from kaggle_evaluation.default_inference_server import DefaultInferenceServer
import model

def main():
    """Run local evaluation test"""
    print("🚀 Starting Hull Tactical Market Prediction Local Test")
    print("=" * 60)
    
    # First, train the model
    print("📚 Training model...")
    try:
        model.train_model()
        print("✅ Model training completed!")
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        return
    
    print("\n🔄 Starting inference server...")
    
    # Create inference server with the predict function
    inference_server = DefaultInferenceServer(model.predict)
    
    # Run local evaluation
    try:
        print("🧪 Running local evaluation...")
        # This will process the test.csv file and generate predictions
        inference_server.run_local_gateway()
        print("✅ Local evaluation completed successfully!")
        print("📄 Check 'submission.parquet' for the generated predictions")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 60)
    print("🏁 Test completed!")

if __name__ == "__main__":
    main()