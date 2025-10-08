"""
Suicide Risk Prediction using Machine Learning
=============================================

This script implements a comprehensive machine learning pipeline to predict suicide risk
using acoustic and demographic features. The pipeline includes:
- Data preprocessing and feature engineering
- Principal Component Analysis (PCA) for dimensionality reduction
- 5-fold cross-validation
- Three ML models: SVM, Logistic Regression, Random Forest
- Comprehensive performance evaluation
- Individual patient prediction capability

Author: ML Team
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report, roc_curve)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# For MacBook Pro MPS support
import torch
if torch.backends.mps.is_available():
    print("MPS (Metal Performance Shaders) is available for acceleration")
    device = "mps"
else:
    print("MPS not available, using CPU")
    device = "cpu"

class SuicideRiskPredictor:
    """
    A comprehensive machine learning pipeline for suicide risk prediction.
    
    This class handles data preprocessing, model training, evaluation, and prediction
    for suicide risk assessment using multiple ML algorithms.
    """
    
    def __init__(self, data_path):
        """
        Initialize the predictor with data loading and basic setup.
        
        Args:
            data_path (str): Path to the CSV file containing the dataset
        """
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None
        self.scaler = StandardScaler()
        self.pca = None
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.results = {}
        self.cv_scores = {}
        
        print("🚀 Initializing Suicide Risk Prediction System")
        print("=" * 60)
        
    def load_and_explore_data(self):
        """
        Load the dataset and perform initial exploration.
        """
        print("\n📊 Loading and exploring data...")
        
        # Load the data
        self.data = pd.read_csv(self.data_path)
        
        print(f"Dataset shape: {self.data.shape}")
        print(f"Features: {self.data.shape[1] - 1}")  # Excluding target variable
        print(f"Samples: {self.data.shape[0]}")
        
        # Display basic information about the target variable
        if 'Label' in self.data.columns:
            target_counts = self.data['Label'].value_counts()
            print(f"\nTarget distribution:")
            print(f"  No risk (0): {target_counts.get(0, 0)} samples ({target_counts.get(0, 0)/len(self.data)*100:.1f}%)")
            print(f"  At risk (1): {target_counts.get(1, 0)} samples ({target_counts.get(1, 0)/len(self.data)*100:.1f}%)")
        
        # Check for missing values
        missing_values = self.data.isnull().sum().sum()
        print(f"Missing values: {missing_values}")
        
        return self.data
    
    def preprocess_data(self, n_components=50):
        """
        Preprocess the data including feature engineering and PCA.
        
        Args:
            n_components (int): Number of PCA components to retain
        """
        print(f"\n🔧 Preprocessing data with PCA ({n_components} components)...")
        
        # Separate features and target
        feature_columns = [col for col in self.data.columns if col not in ['ID', 'Label']]
        
        # Handle categorical variables (Sex)
        data_processed = self.data.copy()
        if 'Sex' in data_processed.columns:
            data_processed['Sex'] = self.label_encoder.fit_transform(data_processed['Sex'])
        
        # Extract features and target
        self.X = data_processed[feature_columns].values
        self.y = data_processed['Label'].values
        
        # Handle missing values by filling with median
        if np.isnan(self.X).any():
            print("⚠️  Handling missing values...")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            self.X = imputer.fit_transform(self.X)
        
        # Standardize features
        print("📏 Standardizing features...")
        self.X = self.scaler.fit_transform(self.X)
        
        # Apply PCA for dimensionality reduction
        print(f"🎯 Applying PCA (reducing from {self.X.shape[1]} to {n_components} features)...")
        self.pca = PCA(n_components=n_components, random_state=42)
        self.X = self.pca.fit_transform(self.X)
        
        # Print explained variance ratio
        explained_variance = self.pca.explained_variance_ratio_.sum()
        print(f"📈 PCA explained variance: {explained_variance:.3f} ({explained_variance*100:.1f}%)")
        
        print(f"✅ Preprocessing complete. Final feature shape: {self.X.shape}")
        
    def setup_models(self):
        """
        Initialize the machine learning models with optimized parameters.
        """
        print("\n🤖 Setting up machine learning models...")
        
        # SVM with RBF kernel
        self.models['SVM'] = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,  # Enable probability estimates
            random_state=42
        )
        
        # Logistic Regression with L2 regularization
        self.models['Logistic Regression'] = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
            solver='liblinear'  # Good for small datasets
        )
        
        # Random Forest with balanced parameters
        self.models['Random Forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1  # Use all available cores
        )
        
        print(f"✅ Initialized {len(self.models)} models: {list(self.models.keys())}")
    
    def perform_cross_validation(self, cv_folds=5):
        """
        Perform k-fold cross-validation for all models.
        
        Args:
            cv_folds (int): Number of cross-validation folds
        """
        print(f"\n🔄 Performing {cv_folds}-fold cross-validation...")
        
        # Initialize stratified k-fold to maintain class distribution
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Metrics to evaluate
        scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        for model_name, model in self.models.items():
            print(f"\n📊 Evaluating {model_name}...")
            
            self.cv_scores[model_name] = {}
            
            for metric in scoring_metrics:
                scores = cross_val_score(model, self.X, self.y, cv=skf, scoring=metric)
                self.cv_scores[model_name][metric] = {
                    'scores': scores,
                    'mean': scores.mean(),
                    'std': scores.std()
                }
                
                print(f"  {metric.capitalize()}: {scores.mean():.4f} (±{scores.std():.4f})")
    
    def train_and_evaluate_models(self):
        """
        Train models on full dataset and evaluate performance.
        """
        print("\n🎯 Training models on full dataset...")
        
        for model_name, model in self.models.items():
            print(f"\n🔧 Training {model_name}...")
            
            # Train the model
            model.fit(self.X, self.y)
            
            # Make predictions
            y_pred = model.predict(self.X)
            y_pred_proba = model.predict_proba(self.X)[:, 1]  # Probability of positive class
            
            # Calculate metrics
            self.results[model_name] = {
                'accuracy': accuracy_score(self.y, y_pred),
                'precision': precision_score(self.y, y_pred),
                'recall': recall_score(self.y, y_pred),
                'f1': f1_score(self.y, y_pred),
                'roc_auc': roc_auc_score(self.y, y_pred_proba),
                'confusion_matrix': confusion_matrix(self.y, y_pred),
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"✅ {model_name} training complete")
    
    def display_results(self):
        """
        Display comprehensive results comparison.
        """
        print("\n" + "="*80)
        print("📈 MODEL PERFORMANCE COMPARISON")
        print("="*80)
        
        # Create results DataFrame for easy comparison
        results_df = pd.DataFrame({
            model_name: {
                'CV Accuracy': self.cv_scores[model_name]['accuracy']['mean'],
                'CV Precision': self.cv_scores[model_name]['precision']['mean'],
                'CV Recall': self.cv_scores[model_name]['recall']['mean'],
                'CV F1-Score': self.cv_scores[model_name]['f1']['mean'],
                'CV ROC-AUC': self.cv_scores[model_name]['roc_auc']['mean'],
                'Train Accuracy': self.results[model_name]['accuracy'],
                'Train Precision': self.results[model_name]['precision'],
                'Train Recall': self.results[model_name]['recall'],
                'Train F1-Score': self.results[model_name]['f1'],
                'Train ROC-AUC': self.results[model_name]['roc_auc']
            }
            for model_name in self.models.keys()
        }).round(4)
        
        print("\n📊 Cross-Validation Results:")
        print(results_df.iloc[:5])  # CV results
        
        print("\n📊 Training Set Results:")
        print(results_df.iloc[5:])  # Training results
        
        # Find best model based on CV F1-score
        best_model = max(self.cv_scores.keys(), 
                        key=lambda x: self.cv_scores[x]['f1']['mean'])
        print(f"\n🏆 Best performing model: {best_model}")
        print(f"   CV F1-Score: {self.cv_scores[best_model]['f1']['mean']:.4f}")
        
        return results_df
    
    def plot_results(self):
        """
        Create visualizations for model comparison.
        """
        print("\n📊 Creating performance visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Suicide Risk Prediction - Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. CV Accuracy Comparison
        models = list(self.models.keys())
        cv_accuracies = [self.cv_scores[model]['accuracy']['mean'] for model in models]
        cv_std = [self.cv_scores[model]['accuracy']['std'] for model in models]
        
        axes[0, 0].bar(models, cv_accuracies, yerr=cv_std, capsize=5, alpha=0.7, color=['skyblue', 'lightgreen', 'salmon'])
        axes[0, 0].set_title('Cross-Validation Accuracy')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        
        # 2. Multiple Metrics Comparison
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        x = np.arange(len(metrics))
        width = 0.25
        
        for i, model in enumerate(models):
            values = [self.cv_scores[model][metric]['mean'] for metric in metrics]
            axes[0, 1].bar(x + i*width, values, width, label=model, alpha=0.7)
        
        axes[0, 1].set_title('Cross-Validation Metrics Comparison')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_xlabel('Metrics')
        axes[0, 1].set_xticks(x + width)
        axes[0, 1].set_xticklabels(metrics)
        axes[0, 1].legend()
        axes[0, 1].set_ylim(0, 1)
        
        # 3. Confusion Matrix for best model
        best_model = max(self.cv_scores.keys(), 
                        key=lambda x: self.cv_scores[x]['f1']['mean'])
        cm = self.results[best_model]['confusion_matrix']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_title(f'Confusion Matrix - {best_model}')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        
        # 4. ROC Curves
        for model_name in models:
            fpr, tpr, _ = roc_curve(self.y, self.results[model_name]['probabilities'])
            auc_score = self.results[model_name]['roc_auc']
            axes[1, 1].plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
        
        axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[1, 1].set_title('ROC Curves')
        axes[1, 1].set_xlabel('False Positive Rate')
        axes[1, 1].set_ylabel('True Positive Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizations saved as 'model_performance_comparison.png'")
    
    def predict_patient_risk(self, patient_id):
        """
        Predict suicide risk for a specific patient ID.
        
        Args:
            patient_id (int): The patient ID to predict
            
        Returns:
            dict: Predictions from all models
        """
        print(f"\n🔍 Predicting suicide risk for Patient ID: {patient_id}")
        print("-" * 50)
        
        # Find patient data
        patient_data = self.data[self.data['ID'] == patient_id]
        
        if patient_data.empty:
            print(f"❌ Patient ID {patient_id} not found in dataset")
            return None
        
        # Get patient info
        patient_info = patient_data.iloc[0]
        print(f"Patient Info:")
        print(f"  ID: {patient_info['ID']}")
        print(f"  Sex: {patient_info['Sex']}")
        print(f"  Age: {patient_info['Age']}")
        print(f"  Actual Label: {patient_info['Label']} ({'At Risk' if patient_info['Label'] == 1 else 'No Risk'})")
        
        # Prepare patient features (same preprocessing as training)
        feature_columns = [col for col in self.data.columns if col not in ['ID', 'Label']]
        patient_features = patient_data[feature_columns].copy()
        
        # Handle categorical variables
        if 'Sex' in patient_features.columns:
            patient_features['Sex'] = self.label_encoder.transform(patient_features['Sex'])
        
        # Apply same preprocessing
        patient_features = self.scaler.transform(patient_features.values)
        patient_features = self.pca.transform(patient_features)
        
        # Make predictions with all models
        predictions = {}
        print(f"\n🤖 Model Predictions:")
        
        for model_name, model in self.models.items():
            # Binary prediction
            binary_pred = model.predict(patient_features)[0]
            # Probability prediction
            prob_pred = model.predict_proba(patient_features)[0]
            
            predictions[model_name] = {
                'binary_prediction': binary_pred,
                'risk_probability': prob_pred[1],  # Probability of being at risk
                'no_risk_probability': prob_pred[0]  # Probability of no risk
            }
            
            risk_level = "HIGH RISK" if binary_pred == 1 else "LOW RISK"
            print(f"  {model_name}:")
            print(f"    Binary Prediction: {binary_pred} ({risk_level})")
            print(f"    Risk Probability: {prob_pred[1]:.3f} ({prob_pred[1]*100:.1f}%)")
            print(f"    No Risk Probability: {prob_pred[0]:.3f} ({prob_pred[0]*100:.1f}%)")
            print()
        
        return predictions
    
    def run_complete_analysis(self, n_components=50, cv_folds=5):
        """
        Run the complete machine learning pipeline.
        
        Args:
            n_components (int): Number of PCA components
            cv_folds (int): Number of cross-validation folds
        """
        print("🚀 Starting Complete Suicide Risk Prediction Analysis")
        print("=" * 80)
        
        # Step 1: Load and explore data
        self.load_and_explore_data()
        
        # Step 2: Preprocess data
        self.preprocess_data(n_components=n_components)
        
        # Step 3: Setup models
        self.setup_models()
        
        # Step 4: Perform cross-validation
        self.perform_cross_validation(cv_folds=cv_folds)
        
        # Step 5: Train and evaluate models
        self.train_and_evaluate_models()
        
        # Step 6: Display results
        results_df = self.display_results()
        
        # Step 7: Create visualizations
        self.plot_results()
        
        print("\n" + "="*80)
        print("✅ ANALYSIS COMPLETE!")
        print("="*80)
        
        return results_df

def main():
    """
    Main function to run the suicide risk prediction analysis.
    """
    # Initialize the predictor
    predictor = SuicideRiskPredictor('complete_data.csv')
    
    # Run complete analysis
    results = predictor.run_complete_analysis(n_components=50, cv_folds=5)
    
    # Example: Predict risk for specific patients
    print("\n" + "="*80)
    print("🔍 INDIVIDUAL PATIENT PREDICTIONS")
    print("="*80)
    
    # Get some example patient IDs
    sample_ids = predictor.data['ID'].sample(3).tolist()
    
    for patient_id in sample_ids:
        predictor.predict_patient_risk(patient_id)
        print("\n" + "-"*50)

if __name__ == "__main__":
    main()
