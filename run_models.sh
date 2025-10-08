#!/bin/bash
# Convenience script to run ML models

set -e

# Activate virtual environment
source .venv/bin/activate

echo "🤖 Macy ML Project - Model Runner"
echo "================================"

echo "Available models:"
echo "1. BERT + Random Forest (BERT_RF.py)"
echo "2. BERT + Logistic Regression (BERT_LR/BERT_LR.py)"
echo "3. BERT + SVM (BERT_SVM.py/BERT_SVM.py)"
echo "4. New Assignment (new_assignment.py)"

read -p "Select model to run (1-4): " choice

case $choice in
    1)
        echo "Running BERT + Random Forest..."
        cd BERT_RF.py && python3 BERT_RF.py
        ;;
    2)
        echo "Running BERT + Logistic Regression..."
        cd BERT_LR && python3 BERT_LR.py
        ;;
    3)
        echo "Running BERT + SVM..."
        cd BERT_SVM.py && python3 BERT_SVM.py
        ;;
    4)
        echo "Running New Assignment..."
        python3 new_assignment.py
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac
