#!/bin/bash

# GitHub Repository Setup Script
# This script helps set up the GitHub repository for the suicide risk prediction project

echo "🚀 GitHub Repository Setup for Suicide Risk Prediction Project"
echo "=============================================================="

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "❌ Not in a Git repository. Please run 'git init' first."
    exit 1
fi

echo "📋 Repository Information:"
echo "   Name: suicide-risk-prediction"
echo "   Description: Machine Learning project for suicide risk prediction using acoustic and demographic features"
echo "   Language: Python"
echo "   License: MIT"
echo ""

echo "📝 Manual GitHub Setup Instructions:"
echo "1. Go to https://github.com/new"
echo "2. Repository name: suicide-risk-prediction"
echo "3. Description: Machine Learning project for suicide risk prediction using acoustic and demographic features"
echo "4. Set to Public or Private (your choice)"
echo "5. Do NOT initialize with README, .gitignore, or license (we already have these)"
echo "6. Click 'Create repository'"
echo ""

echo "🔗 After creating the repository, run these commands:"
echo "   git remote add origin https://github.com/YOUR_USERNAME/suicide-risk-prediction.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""

echo "📊 Project Statistics:"
echo "   Total files: $(find . -type f -not -path './.git/*' -not -path './.venv/*' | wc -l)"
echo "   Python files: $(find . -name '*.py' -not -path './.venv/*' | wc -l)"
echo "   Lines of code: $(find . -name '*.py' -not -path './.venv/*' -exec wc -l {} + | tail -1 | awk '{print $1}')"
echo ""

echo "🎯 Key Features to Highlight:"
echo "   ✅ Multiple ML algorithms (SVM, Logistic Regression, Random Forest)"
echo "   ✅ 5-fold cross-validation with statistical testing"
echo "   ✅ PCA dimensionality reduction"
echo "   ✅ Hyperparameter tuning"
echo "   ✅ Individual patient prediction"
echo "   ✅ Comprehensive visualizations"
echo "   ✅ Detailed documentation"
echo ""

echo "⚠️  Important Notes:"
echo "   - This is for research purposes only"
echo "   - Should not replace clinical judgment"
echo "   - Ensure patient data privacy"
echo "   - Consider ethical implications"
echo ""

echo "🔧 Next Steps:"
echo "1. Create GitHub repository (instructions above)"
echo "2. Add remote origin"
echo "3. Push code to GitHub"
echo "4. Add collaborators if needed"
echo "5. Set up GitHub Pages for documentation (optional)"
echo "6. Create issues for future improvements"
echo ""

echo "✅ Setup script complete!"