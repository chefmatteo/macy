#!/bin/bash

# Helper script to run ML models with uv
# Usage: ./run_model.sh <model_name>
# Example: ./run_model.sh BERT-BiLSTM

set -e  # Exit on error

# Add uv to PATH
export PATH="$HOME/.local/bin:$PATH"

# Change to project directory
cd "$(dirname "$0")"

# Check if model name is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_name>"
    echo ""
    echo "Available models:"
    echo "  BERT-BiLSTM     - BERT with BiLSTM model"
    echo "  BERT-CNN        - BERT with CNN model"
    echo "  BERT-TextCNN    - BERT with TextCNN model"
    echo "  BERT_RF         - BERT with Random Forest model"
    echo "  @BERT_RF        - BERT RF from subdirectory"
    echo "  BERT_SVM        - BERT with Support Vector Machine"
    echo "  BERT_LR         - BERT with Logistic Regression"
    echo ""
    echo "Examples:"
    echo "  $0 BERT-BiLSTM"
    echo "  $0 BERT-CNN"
    echo "  $0 BERT_RF"
    echo "  $0 @BERT_RF"
    echo "  $0 BERT_SVM"
    echo "  $0 BERT_LR"
    exit 1
fi

MODEL_NAME="$1"

# Determine the script path
case "$MODEL_NAME" in
    "BERT-BiLSTM")
        SCRIPT_PATH="BERT-BiLSTM.py"
        ;;
    "BERT-CNN")
        SCRIPT_PATH="BERT-CNN.py"
        ;;
    "BERT-TextCNN")
        SCRIPT_PATH="BERT-TextCNN.py"
        ;;
    "BERT_RF")
        SCRIPT_PATH="BERT_RF.py/BERT_RF.py"
        ;;
    "@BERT_RF")
        SCRIPT_PATH="BERT_RF.py/BERT_RF.py"
        ;;
    "BERT_SVM")
        SCRIPT_PATH="BERT_SVM.py/BERT_SVM.py"
        ;;
    "BERT_LR")
        SCRIPT_PATH="BERT_LR/BERT_LR.py"
        ;;
    *)
        echo "Error: Unknown model '$MODEL_NAME'"
        echo "Run '$0' without arguments to see available models."
        exit 1
        ;;
esac

# Check if script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Script '$SCRIPT_PATH' not found!"
    exit 1
fi

echo "🚀 Running $MODEL_NAME model..."
echo "📁 Script: $SCRIPT_PATH"
echo "🕐 Started at: $(date)"
echo "=" * 60

# Run the model using uv
uv run python "$SCRIPT_PATH"

echo ""
echo "=" * 60
echo "✅ Completed at: $(date)"
