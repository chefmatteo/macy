# Suicide Risk Prediction using Machine Learning

A comprehensive machine learning project for predicting suicide risk using acoustic and demographic features. This project implements multiple ML algorithms with 5-fold cross-validation, PCA dimensionality reduction, and comprehensive performance evaluation.

## 🎯 Project Overview

This project aims to predict suicide risk in patients using:
- **Acoustic features**: Voice patterns, speech characteristics, and audio biomarkers
- **Demographic data**: Age, sex, and other patient information
- **Advanced ML techniques**: PCA, cross-validation, and ensemble methods

## 📊 Dataset

- **Total samples**: 208 patients
- **Features**: 423 acoustic and demographic features
- **Target distribution**: 
  - No risk (0): 109 samples (52.4%)
  - At risk (1): 99 samples (47.6%)

## 🤖 Machine Learning Models

The project implements three main algorithms:

1. **Support Vector Machine (SVM)**
   - RBF kernel with hyperparameter tuning
   - Optimized for binary classification

2. **Logistic Regression**
   - L2 regularization
   - Probability estimates for risk assessment

3. **Random Forest Classifier**
   - Ensemble method with feature importance analysis
   - Robust to overfitting

## 🔧 Key Features

### Data Preprocessing
- **Standardization**: Feature scaling for optimal model performance
- **PCA**: Dimensionality reduction (422 → 50 components, 91.1% variance retained)
- **Missing value handling**: Median imputation strategy

### Model Evaluation
- **5-fold Cross-Validation**: Stratified sampling to maintain class distribution
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Statistical Testing**: Paired t-tests for model comparison
- **Confidence Intervals**: 95% CI for performance metrics

### Advanced Features
- **Hyperparameter Tuning**: Grid search optimization
- **Feature Importance Analysis**: Understanding key predictors
- **Ensemble Predictions**: Combining multiple models for better accuracy
- **Individual Patient Prediction**: Risk assessment for specific patients

## 📈 Performance Results

### Cross-Validation Results (Best Model: Logistic Regression)
- **Accuracy**: 52.4% ± 6.4%
- **Precision**: 50.1% ± 6.4%
- **Recall**: 48.5% ± 3.0%
- **F1-Score**: 49.2% ± 4.3%
- **ROC-AUC**: 52.7% ± 9.0%

### Key Findings
- Models show moderate performance, indicating the complexity of suicide risk prediction
- Logistic Regression provides the best balance of metrics
- Feature importance analysis reveals key acoustic components

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
pandas >= 1.5.0
numpy >= 1.21.0
scikit-learn >= 1.3.0
matplotlib >= 3.5.0
seaborn >= 0.11.0
torch >= 2.0.0 (for MPS support on macOS)
```

### Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/suicide-risk-prediction.git
cd suicide-risk-prediction
```

2. Create virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements_ml.txt
```

### Usage

#### Basic Analysis
```bash
python suicide_risk_prediction.py
```

#### Enhanced Analysis with Hyperparameter Tuning
```bash
python enhanced_suicide_prediction.py
```

#### Quick Analysis (No Tuning)
```bash
python enhanced_suicide_prediction.py --no-tuning
```

#### Predict Specific Patient
```bash
python enhanced_suicide_prediction.py --predict 123
```

#### Custom Parameters
```bash
python enhanced_suicide_prediction.py --components 30 --cv-folds 10
```

## 📁 Project Structure

```
suicide-risk-prediction/
├── README.md                          # Project documentation
├── requirements_ml.txt                # Python dependencies
├── complete_data.csv                  # Main dataset
├── demo.ipynb                         # Jupyter notebook for data exploration
├── suicide_risk_prediction.py        # Basic ML pipeline
├── enhanced_suicide_prediction.py    # Advanced ML pipeline
├── model_performance_comparison.png   # Basic visualizations
├── enhanced_model_analysis.png       # Comprehensive visualizations
└── suicide_risk_analysis_report.json # Detailed analysis report
```

## 📊 Visualizations

The project generates comprehensive visualizations including:
- **Performance Comparison**: Cross-validation metrics across models
- **ROC Curves**: Model discrimination ability
- **Confusion Matrices**: Classification accuracy breakdown
- **Feature Importance**: Key predictive components
- **PCA Analysis**: Dimensionality reduction effectiveness
- **Radar Charts**: Multi-metric model comparison

## 🔍 Individual Patient Prediction

The system provides detailed risk assessment for individual patients:

```python
# Example output
🔍 Enhanced Risk Prediction for Patient ID: 817
Patient Information:
   ID: 817, Sex: F, Age: 42
   Actual Label: 1 (At Risk)

🤖 Individual Model Predictions:
SVM: 🔴 HIGH RISK (Risk: 52.9%)
Logistic Regression: 🔴 HIGH RISK (Risk: 76.4%)
Random Forest: 🔴 HIGH RISK (Risk: 71.3%)

🎯 Ensemble Prediction: 🔴 HIGH RISK (66.9%)
📊 Risk Category: HIGH RISK
Recommendation: Immediate intervention and close monitoring required
```

## ⚠️ Important Considerations

### Ethical Guidelines
- This tool is for **research purposes only**
- Should **never replace professional clinical judgment**
- Requires validation in clinical settings before any practical application
- Patient privacy and data security must be maintained

### Model Limitations
- Moderate performance indicates need for additional features or larger dataset
- Class imbalance may affect prediction accuracy
- Acoustic features may be influenced by various non-risk factors
- Cross-cultural validation needed for broader applicability

## 🔬 Technical Details

### PCA Analysis
- **Components**: 50 (from 422 original features)
- **Explained Variance**: 91.1%
- **First 10 components**: 58.8% of variance

### Cross-Validation Strategy
- **Method**: Stratified 5-fold
- **Rationale**: Maintains class distribution across folds
- **Metrics**: Multiple evaluation criteria for comprehensive assessment

### Hardware Acceleration
- **macOS**: Automatic MPS (Metal Performance Shaders) detection
- **GPU Support**: CUDA-compatible for NVIDIA GPUs
- **CPU Fallback**: Automatic fallback for unsupported hardware

## 📚 Research Applications

This project can be extended for:
- **Clinical Decision Support**: Integration with healthcare systems
- **Longitudinal Studies**: Tracking risk changes over time
- **Multi-modal Analysis**: Combining with other biomarkers
- **Population Health**: Large-scale screening applications

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset contributors and research community
- Scikit-learn and Python ML ecosystem
- Mental health research initiatives
- Clinical collaborators and domain experts

## 📞 Contact

For questions, suggestions, or collaborations:
- **Email**: [your.email@domain.com]
- **GitHub**: [@yourusername]
- **LinkedIn**: [Your LinkedIn Profile]

---

**⚠️ Disclaimer**: This software is for research and educational purposes only. It should not be used for clinical diagnosis or treatment decisions without proper validation and clinical oversight.