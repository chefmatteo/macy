#!/bin/bash

# Quick Setup Script for Macy ML Project
# Minimal setup for users who already have uv installed

set -e

echo "⚡ Quick Setup for Macy ML Project"
echo "================================="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ UV package manager not found!"
    echo "Please run ./setup.sh for full installation, or install uv manually:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "✅ UV found: $(uv --version)"

# Navigate to project directory
cd "$(dirname "$0")"

# Create virtual environment and install dependencies
echo "📦 Creating virtual environment and installing dependencies..."
uv venv
source .venv/bin/activate
uv pip install -e .
uv pip install -e ".[dev]"

echo ""
echo "🎉 Quick setup complete!"
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To run models:"
echo "  python3 new_assignment.py"
echo "  cd BERT_RF.py && python3 BERT_RF.py"
echo "  cd BERT_LR && python3 BERT_LR.py"
echo "  cd BERT_SVM.py && python3 BERT_SVM.py"
