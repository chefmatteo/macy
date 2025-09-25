### Prerequisites

- **macOS** (Apple Silicon M1/M2/M3 recommended for optimal performance)
- **Python 3.9+**
- **16GB+ RAM** recommended for BERT processing
- **uv package manager** (will be installed automatically)

### Installation

1. **Clone/Download the project**

```bash
cd /path/to/macy
```

2. **Install uv package manager**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

3. **Install dependencies**

```bash
uv venv
uv pip install torch transformers pandas numpy scikit-learn openpyxl matplotlib scipy psutil
```

### Running Models

Use the convenient run script:

```bash
# Logistic Regression (Recommended - Fastest)
./run_model.sh BERT_LR

# Support Vector Machine 
./run_model.sh BERT_SVM

# Random Forest
./run_model.sh BERT_RF
```

Or run directly:

```bash
uv run python BERT_LR/BERT_LR.py
uv run python BERT_SVM.py/BERT_SVM.py
uv run python BERT_RF.py/BERT_RF.py
```

## 📊 Models Overview

### BERT_LR (Logistic Regression) - **Recommended**

- **Speed**: Fastest (~5-8 minutes)
- **Accuracy**: High performance for binary classification
- **Features**: 5-fold CV, Grid Search, PCA dimensionality reduction
- **Best for**: Quick experiments and production deployment

### BERT_SVM (Support Vector Machine)

- **Speed**: Medium (~8-12 minutes)
- **Accuracy**: Excellent for complex decision boundaries
- **Features**: RBF kernel, Grid Search optimization
- **Best for**: Maximum accuracy requirements

### BERT_RF (Random Forest)

- **Speed**: Fast (~3-5 minutes)
- **Accuracy**: Good baseline performance
- **Features**: Simple, interpretable, no hyperparameter tuning
- **Best for**: Quick baseline and feature importance analysis

## 🔧 Model Parameters

### BERT Configuration

- **Model**: `indiejoseph/bert-base-cantonese` (Cantonese BERT)
- **Max Length**: 128 tokens
- **Embedding Size**: 768 → 256 (PCA reduced)
- **Device**: Auto-detects MPS/CUDA/CPU

### Grid Search Parameters

#### BERT_LR (Optimized for Speed)

```python
param_grid = {
    'C': [1, 10],                    # Regularization strength
    'penalty': ['l2'],               # L2 regularization (Ridge)
    'solver': ['lbfgs'],             # Optimization algorithm
    'max_iter': [1000]               # Maximum iterations
}
# Total combinations: 2 × 1 × 1 × 1 = 2
```

#### BERT_SVM (Optimized for Speed)

```python
param_grid = {
    'C': [1, 10],                    # Regularization parameter
    'gamma': ['scale', 0.01],        # RBF kernel coefficient  
    'kernel': ['rbf']                # Radial basis function kernel
}
# Total combinations: 2 × 2 × 1 = 4
```

### Cross-Validation

- **Outer CV**: 5-fold (main evaluation)
- **Inner CV**: 2-fold (hyperparameter tuning)
- **User-level splitting**: Prevents data leakage

## Data Format

- **Text Data**: User ID + Cantonese text + 80+ linguistic features
- **Labels**: Binary classification (0: No risk, 1: Risk present)
- **Total**: ~4,800 text samples from 208 users
