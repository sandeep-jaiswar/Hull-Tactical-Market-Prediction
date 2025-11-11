#!/bin/bash

# Setup script for Hull Tactical Market Prediction local environment

echo "🚀 Setting up Hull Tactical Market Prediction Environment"
echo "=========================================================="

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️  No virtual environment detected. Consider creating one:"
    echo "   python -m venv venv"
    echo "   source venv/bin/activate"
    echo ""
fi

# Install required packages
echo "📦 Installing required packages..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully!"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Check if data files exist
echo ""
echo "📁 Checking data files..."
if [ -f "train.csv" ]; then
    echo "✅ train.csv found"
else
    echo "❌ train.csv not found in current directory"
fi

if [ -f "test.csv" ]; then
    echo "✅ test.csv found"
else
    echo "❌ test.csv not found in current directory"
fi

# Make the test script executable
chmod +x test_local.py

echo ""
echo "🎯 Setup complete! Next steps:"
echo "1. Make sure train.csv and test.csv are in the project root"
echo "2. Run: python test_local.py"
echo "3. Check the generated submission.parquet file"
echo ""
echo "🔧 To customize your model, edit model.py"
echo "=========================================================="