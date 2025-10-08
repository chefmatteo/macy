"""
Enhanced Suicide Risk Prediction System
=======================================

This enhanced version includes:
- Interactive patient ID prediction
- Model comparison with statistical significance
- Feature importance analysis
- Hyperparameter tuning
- Detailed reporting
- Export functionality

Usage:
    python enhanced_suicide_prediction.py
    
Or for specific patient prediction:
    python enhanced_suicide_prediction.py --predict 123
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report, roc_curve)
from sklearn.pipeline import Pipeline
from scipy import stats
import argparse
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EnhancedSuicideRiskPredictor:
    """
    Enhanced suicide risk prediction system with advanced features.
    """
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None
        self.scaler = StandardScaler()
        self.pca = None
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.best_models = {}
        self.results = {}
        self.cv_scores = {}
        self.feature_importance = {}
        
        print("🚀 Enhanced Suicide Risk Prediction System")
        print("=" * 60)
        
    def load_and_explore_data(self):
        """Load and explore the dataset with detailed statistics."""
        print("\n📊 Loading and exploring data...")
        
        self.data = pd.read_csv(self.data_path)
        
        print(f"Dataset shape: {self.data.shape}")
        print(f"Features: {self.data.shape[1] - 1}")
        print(f"Samples: {self.data.shape[0]}")
        
        # Target distribution
        if 'Label' in self.data.columns:
            target_counts = self.data['Label'].value_counts()
            print(f"\nTarget distribution:")
            print(f"  No risk (0): {target_counts.get(0, 0)} samples ({target_counts.get(0, 0)/len(self.data)*100:.1f}%)")
            print(f"  At risk (1): {target_counts.get(1, 0)} samples ({target_counts.get(1, 0)/len(self.data)*100:.1f}%)")
            
            # Class balance analysis
            imbalance_ratio = target_counts.max() / target_counts.min()
            print(f"  Class imbalance ratio: {imbalance_ratio:.2f}")
            if imbalance_ratio > 1.5:
                print("  ⚠️  Dataset is imbalanced - consider using balanced models")
        
        # Demographic analysis
        if 'Sex' in self.data.columns and 'Age' in self.data.columns:
            print(f"\nDemographic distribution:")
            sex_dist = self.data['Sex'].value_counts()
            print(f"  Sex: {dict(sex_dist)}")
            print(f"  Age: Mean={self.data['Age'].mean():.1f}, Std={self.data['Age'].std():.1f}")
            print(f"       Range: {self.data['Age'].min()}-{self.data['Age'].max()}")
        
        # Missing values analysis
        missing_values = self.data.isnull().sum().sum()
        print(f"Missing values: {missing_values}")
        
        return self.data
    
    def preprocess_data(self, n_components=50):
        """Enhanced preprocessing with feature analysis."""
        print(f"\n🔧 Enhanced preprocessing with PCA ({n_components} components)...")
        
        # Separate features and target
        feature_columns = [col for col in self.data.columns if col not in ['ID', 'Label']]
        
        # Handle categorical variables
        data_processed = self.data.copy()
        if 'Sex' in data_processed.columns:
            data_processed['Sex'] = self.label_encoder.fit_transform(data_processed['Sex'])
        
        # Extract features and target
        self.X = data_processed[feature_columns].values
        self.y = data_processed['Label'].values
        
        # Handle missing values
        if np.isnan(self.X).any():
            print("⚠️  Handling missing values...")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            self.X = imputer.fit_transform(self.X)
        
        # Feature scaling analysis
        print("📏 Analyzing feature distributions before scaling...")
        feature_means = np.mean(self.X, axis=0)
        feature_stds = np.std(self.X, axis=0)
        print(f"   Feature means range: {feature_means.min():.3f} to {feature_means.max():.3f}")
        print(f"   Feature stds range: {feature_stds.min():.3f} to {feature_stds.max():.3f}")
        
        # Standardize features
        print("📏 Standardizing features...")
        self.X = self.scaler.fit_transform(self.X)
        
        # PCA analysis
        print(f"🎯 Applying PCA (reducing from {self.X.shape[1]} to {n_components} features)...")
        self.pca = PCA(n_components=n_components, random_state=42)
        self.X = self.pca.fit_transform(self.X)
        
        # Detailed PCA analysis
        explained_variance = self.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        print(f"📈 PCA Analysis:")
        print(f"   Total explained variance: {explained_variance.sum():.3f} ({explained_variance.sum()*100:.1f}%)")
        print(f"   First 10 components explain: {cumulative_variance[9]:.3f} ({cumulative_variance[9]*100:.1f}%)")
        print(f"   Components needed for 95% variance: {np.argmax(cumulative_variance >= 0.95) + 1}")
        
        print(f"✅ Preprocessing complete. Final feature shape: {self.X.shape}")
        
    def setup_models_with_tuning(self):
        """Setup models with hyperparameter tuning."""
        print("\n🤖 Setting up models with hyperparameter tuning...")
        
        # Define parameter grids for tuning
        param_grids = {
            'SVM': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'kernel': ['rbf', 'linear']
            },
            'Logistic Regression': {
                'C': [0.1, 1, 10, 100],
                'solver': ['liblinear', 'lbfgs'],
                'max_iter': [1000]
            },
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5, 10]
            }
        }
        
        # Base models
        base_models = {
            'SVM': SVC(probability=True, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1)
        }
        
        # Perform grid search for each model
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Reduced folds for speed
        
        for model_name, base_model in base_models.items():
            print(f"🔍 Tuning {model_name}...")
            
            grid_search = GridSearchCV(
                base_model, 
                param_grids[model_name], 
                cv=cv, 
                scoring='f1',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(self.X, self.y)
            
            self.best_models[model_name] = grid_search.best_estimator_
            self.models[model_name] = grid_search.best_estimator_
            
            print(f"   Best parameters: {grid_search.best_params_}")
            print(f"   Best CV F1-score: {grid_search.best_score_:.4f}")
        
        print(f"✅ Model tuning complete for {len(self.models)} models")
    
    def perform_enhanced_cross_validation(self, cv_folds=5):
        """Enhanced cross-validation with statistical analysis."""
        print(f"\n🔄 Performing enhanced {cv_folds}-fold cross-validation...")
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        for model_name, model in self.models.items():
            print(f"\n📊 Evaluating {model_name}...")
            
            self.cv_scores[model_name] = {}
            
            for metric in scoring_metrics:
                scores = cross_val_score(model, self.X, self.y, cv=skf, scoring=metric)
                self.cv_scores[model_name][metric] = {
                    'scores': scores,
                    'mean': scores.mean(),
                    'std': scores.std(),
                    'ci_lower': scores.mean() - 1.96 * scores.std() / np.sqrt(len(scores)),
                    'ci_upper': scores.mean() + 1.96 * scores.std() / np.sqrt(len(scores))
                }
                
                print(f"  {metric.capitalize()}: {scores.mean():.4f} (±{scores.std():.4f}) "
                      f"[CI: {self.cv_scores[model_name][metric]['ci_lower']:.4f}-"
                      f"{self.cv_scores[model_name][metric]['ci_upper']:.4f}]")
    
    def analyze_feature_importance(self):
        """Analyze feature importance for applicable models."""
        print("\n🔍 Analyzing feature importance...")
        
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                # Random Forest feature importance
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                self.feature_importance[model_name] = {
                    'importances': importances,
                    'indices': indices,
                    'top_10': indices[:10]
                }
                
                print(f"\n{model_name} - Top 10 Most Important Features:")
                for i, idx in enumerate(indices[:10]):
                    print(f"  {i+1}. PCA Component {idx+1}: {importances[idx]:.4f}")
            
            elif hasattr(model, 'coef_'):
                # Logistic Regression coefficients
                coef = np.abs(model.coef_[0])
                indices = np.argsort(coef)[::-1]
                
                self.feature_importance[model_name] = {
                    'coefficients': coef,
                    'indices': indices,
                    'top_10': indices[:10]
                }
                
                print(f"\n{model_name} - Top 10 Most Important Features (by |coefficient|):")
                for i, idx in enumerate(indices[:10]):
                    print(f"  {i+1}. PCA Component {idx+1}: {coef[idx]:.4f}")
    
    def train_and_evaluate_models(self):
        """Train and evaluate models with detailed metrics."""
        print("\n🎯 Training and evaluating models...")
        
        for model_name, model in self.models.items():
            print(f"\n🔧 Training {model_name}...")
            
            # Train the model
            model.fit(self.X, self.y)
            
            # Make predictions
            y_pred = model.predict(self.X)
            y_pred_proba = model.predict_proba(self.X)[:, 1]
            
            # Calculate comprehensive metrics
            self.results[model_name] = {
                'accuracy': accuracy_score(self.y, y_pred),
                'precision': precision_score(self.y, y_pred),
                'recall': recall_score(self.y, y_pred),
                'f1': f1_score(self.y, y_pred),
                'roc_auc': roc_auc_score(self.y, y_pred_proba),
                'confusion_matrix': confusion_matrix(self.y, y_pred),
                'classification_report': classification_report(self.y, y_pred),
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"✅ {model_name} training complete")
    
    def statistical_comparison(self):
        """Perform statistical comparison between models."""
        print("\n📊 Statistical Model Comparison...")
        
        models = list(self.models.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        print("\nPairwise t-test comparisons (p-values):")
        print("=" * 50)
        
        for metric in metrics:
            print(f"\n{metric.upper()}:")
            for i, model1 in enumerate(models):
                for j, model2 in enumerate(models):
                    if i < j:  # Only compare each pair once
                        scores1 = self.cv_scores[model1][metric]['scores']
                        scores2 = self.cv_scores[model2][metric]['scores']
                        
                        # Paired t-test
                        t_stat, p_value = stats.ttest_rel(scores1, scores2)
                        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                        
                        print(f"  {model1} vs {model2}: p={p_value:.4f} {significance}")
    
    def create_enhanced_visualizations(self):
        """Create comprehensive visualizations."""
        print("\n📊 Creating enhanced visualizations...")
        
        # Create a larger figure with more subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. CV Performance Comparison
        ax1 = plt.subplot(3, 3, 1)
        models = list(self.models.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        x = np.arange(len(metrics))
        width = 0.25
        colors = ['skyblue', 'lightgreen', 'salmon']
        
        for i, model in enumerate(models):
            values = [self.cv_scores[model][metric]['mean'] for metric in metrics]
            errors = [self.cv_scores[model][metric]['std'] for metric in metrics]
            ax1.bar(x + i*width, values, width, label=model, alpha=0.7, 
                   color=colors[i], yerr=errors, capsize=3)
        
        ax1.set_title('Cross-Validation Performance Comparison', fontweight='bold')
        ax1.set_ylabel('Score')
        ax1.set_xlabel('Metrics')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(metrics)
        ax1.legend()
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # 2. ROC Curves
        ax2 = plt.subplot(3, 3, 2)
        for i, model_name in enumerate(models):
            fpr, tpr, _ = roc_curve(self.y, self.results[model_name]['probabilities'])
            auc_score = self.results[model_name]['roc_auc']
            ax2.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})', 
                    color=colors[i], linewidth=2)
        
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax2.set_title('ROC Curves Comparison', fontweight='bold')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Confusion Matrices
        for i, model_name in enumerate(models):
            ax = plt.subplot(3, 3, 3 + i)
            cm = self.results[model_name]['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
            ax.set_title(f'{model_name}\nConfusion Matrix', fontweight='bold')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        # 6. PCA Explained Variance
        ax6 = plt.subplot(3, 3, 6)
        cumulative_var = np.cumsum(self.pca.explained_variance_ratio_)
        ax6.plot(range(1, len(cumulative_var) + 1), cumulative_var, 'bo-', linewidth=2)
        ax6.axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95% Variance')
        ax6.set_title('PCA Cumulative Explained Variance', fontweight='bold')
        ax6.set_xlabel('Number of Components')
        ax6.set_ylabel('Cumulative Explained Variance')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Feature Importance (if available)
        if 'Random Forest' in self.feature_importance:
            ax7 = plt.subplot(3, 3, 7)
            rf_importance = self.feature_importance['Random Forest']
            top_features = rf_importance['top_10'][:10]
            importances = rf_importance['importances'][top_features]
            
            ax7.barh(range(len(top_features)), importances, color='lightgreen', alpha=0.7)
            ax7.set_yticks(range(len(top_features)))
            ax7.set_yticklabels([f'PC{i+1}' for i in top_features])
            ax7.set_title('Random Forest\nTop 10 Feature Importance', fontweight='bold')
            ax7.set_xlabel('Importance')
            ax7.grid(True, alpha=0.3)
        
        # 8. Model Performance Radar Chart
        ax8 = plt.subplot(3, 3, 8, projection='polar')
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        
        for i, model in enumerate(models):
            values = [self.cv_scores[model][metric]['mean'] for metric in metrics]
            values = np.concatenate((values, [values[0]]))
            ax8.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
            ax8.fill(angles, values, alpha=0.1, color=colors[i])
        
        ax8.set_xticks(angles[:-1])
        ax8.set_xticklabels(metrics)
        ax8.set_ylim(0, 1)
        ax8.set_title('Model Performance Radar Chart', fontweight='bold', pad=20)
        ax8.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # 9. Training vs CV Performance
        ax9 = plt.subplot(3, 3, 9)
        cv_f1 = [self.cv_scores[model]['f1']['mean'] for model in models]
        train_f1 = [self.results[model]['f1'] for model in models]
        
        x_pos = np.arange(len(models))
        width = 0.35
        
        ax9.bar(x_pos - width/2, cv_f1, width, label='CV F1-Score', alpha=0.7, color='lightblue')
        ax9.bar(x_pos + width/2, train_f1, width, label='Train F1-Score', alpha=0.7, color='lightcoral')
        
        ax9.set_title('Training vs Cross-Validation F1-Score', fontweight='bold')
        ax9.set_ylabel('F1-Score')
        ax9.set_xlabel('Models')
        ax9.set_xticks(x_pos)
        ax9.set_xticklabels(models, rotation=45)
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        
        plt.suptitle('Enhanced Suicide Risk Prediction - Comprehensive Analysis', 
                     fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig('enhanced_model_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Enhanced visualizations saved as 'enhanced_model_analysis.png'")
    
    def predict_patient_risk_enhanced(self, patient_id):
        """Enhanced patient risk prediction with confidence intervals."""
        print(f"\n🔍 Enhanced Risk Prediction for Patient ID: {patient_id}")
        print("=" * 60)
        
        # Find patient data
        patient_data = self.data[self.data['ID'] == patient_id]
        
        if patient_data.empty:
            print(f"❌ Patient ID {patient_id} not found in dataset")
            available_ids = sorted(self.data['ID'].unique())
            print(f"Available IDs: {available_ids[:10]}... (showing first 10)")
            return None
        
        # Patient information
        patient_info = patient_data.iloc[0]
        print(f"📋 Patient Information:")
        print(f"   ID: {patient_info['ID']}")
        print(f"   Sex: {patient_info['Sex']}")
        print(f"   Age: {patient_info['Age']}")
        print(f"   Actual Label: {patient_info['Label']} ({'At Risk' if patient_info['Label'] == 1 else 'No Risk'})")
        
        # Prepare features
        feature_columns = [col for col in self.data.columns if col not in ['ID', 'Label']]
        patient_features = patient_data[feature_columns].copy()
        
        if 'Sex' in patient_features.columns:
            patient_features['Sex'] = self.label_encoder.transform(patient_features['Sex'])
        
        patient_features = self.scaler.transform(patient_features.values)
        patient_features = self.pca.transform(patient_features)
        
        # Model predictions with ensemble
        predictions = {}
        risk_probabilities = []
        
        print(f"\n🤖 Individual Model Predictions:")
        print("-" * 40)
        
        for model_name, model in self.models.items():
            binary_pred = model.predict(patient_features)[0]
            prob_pred = model.predict_proba(patient_features)[0]
            
            predictions[model_name] = {
                'binary_prediction': binary_pred,
                'risk_probability': prob_pred[1],
                'no_risk_probability': prob_pred[0]
            }
            
            risk_probabilities.append(prob_pred[1])
            
            risk_level = "🔴 HIGH RISK" if binary_pred == 1 else "🟢 LOW RISK"
            confidence = max(prob_pred)
            
            print(f"{model_name}:")
            print(f"   Prediction: {risk_level}")
            print(f"   Risk Probability: {prob_pred[1]:.3f} ({prob_pred[1]*100:.1f}%)")
            print(f"   Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
            print()
        
        # Ensemble prediction
        ensemble_risk_prob = np.mean(risk_probabilities)
        ensemble_prediction = 1 if ensemble_risk_prob > 0.5 else 0
        ensemble_std = np.std(risk_probabilities)
        
        print("🎯 Ensemble Prediction:")
        print("-" * 30)
        ensemble_level = "🔴 HIGH RISK" if ensemble_prediction == 1 else "🟢 LOW RISK"
        print(f"   Final Prediction: {ensemble_level}")
        print(f"   Ensemble Risk Probability: {ensemble_risk_prob:.3f} ({ensemble_risk_prob*100:.1f}%)")
        print(f"   Model Agreement (std): {ensemble_std:.3f}")
        
        # Risk interpretation
        print(f"\n📊 Risk Assessment:")
        print("-" * 25)
        if ensemble_risk_prob < 0.3:
            risk_category = "🟢 LOW RISK"
            recommendation = "Continue regular monitoring"
        elif ensemble_risk_prob < 0.7:
            risk_category = "🟡 MODERATE RISK"
            recommendation = "Increased monitoring and intervention consideration"
        else:
            risk_category = "🔴 HIGH RISK"
            recommendation = "Immediate intervention and close monitoring required"
        
        print(f"   Risk Category: {risk_category}")
        print(f"   Recommendation: {recommendation}")
        print(f"   Model Consensus: {'High' if ensemble_std < 0.1 else 'Moderate' if ensemble_std < 0.2 else 'Low'}")
        
        return {
            'patient_id': patient_id,
            'patient_info': dict(patient_info),
            'individual_predictions': predictions,
            'ensemble_prediction': ensemble_prediction,
            'ensemble_probability': ensemble_risk_prob,
            'model_agreement': ensemble_std,
            'risk_category': risk_category,
            'recommendation': recommendation
        }
    
    def generate_report(self):
        """Generate a comprehensive analysis report."""
        print("\n📄 Generating comprehensive analysis report...")
        
        report = {
            'analysis_date': datetime.now().isoformat(),
            'dataset_info': {
                'total_samples': int(len(self.data)),
                'total_features': int(self.data.shape[1] - 1),
                'target_distribution': {str(k): int(v) for k, v in self.data['Label'].value_counts().items()},
                'pca_components': int(self.X.shape[1]),
                'explained_variance': float(self.pca.explained_variance_ratio_.sum())
            },
            'model_performance': {},
            'best_model': None,
            'recommendations': []
        }
        
        # Add model performance
        best_f1 = 0
        best_model_name = None
        
        for model_name in self.models.keys():
            cv_scores = self.cv_scores[model_name]
            train_scores = self.results[model_name]
            
            report['model_performance'][model_name] = {
                'cv_scores': {metric: {
                    'mean': float(scores['mean']),
                    'std': float(scores['std']),
                    'ci_lower': float(scores['ci_lower']),
                    'ci_upper': float(scores['ci_upper'])
                } for metric, scores in cv_scores.items()},
                'train_scores': {
                    'accuracy': float(train_scores['accuracy']),
                    'precision': float(train_scores['precision']),
                    'recall': float(train_scores['recall']),
                    'f1': float(train_scores['f1']),
                    'roc_auc': float(train_scores['roc_auc'])
                }
            }
            
            if cv_scores['f1']['mean'] > best_f1:
                best_f1 = cv_scores['f1']['mean']
                best_model_name = model_name
        
        report['best_model'] = best_model_name
        
        # Add recommendations
        if best_f1 < 0.6:
            report['recommendations'].append("Consider collecting more data or engineering additional features")
        if best_f1 < 0.7:
            report['recommendations'].append("Model performance is moderate - consider ensemble methods")
        
        # Save report with custom JSON encoder for numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)
        
        with open('suicide_risk_analysis_report.json', 'w') as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder)
        
        print("✅ Report saved as 'suicide_risk_analysis_report.json'")
        return report
    
    def run_enhanced_analysis(self, n_components=50, cv_folds=5, tune_hyperparameters=True):
        """Run the complete enhanced analysis pipeline."""
        print("🚀 Starting Enhanced Suicide Risk Prediction Analysis")
        print("=" * 80)
        
        # Step 1: Load and explore data
        self.load_and_explore_data()
        
        # Step 2: Preprocess data
        self.preprocess_data(n_components=n_components)
        
        # Step 3: Setup models (with or without tuning)
        if tune_hyperparameters:
            self.setup_models_with_tuning()
        else:
            # Use default models for faster execution
            self.models = {
                'SVM': SVC(probability=True, random_state=42),
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1)
            }
        
        # Step 4: Perform enhanced cross-validation
        self.perform_enhanced_cross_validation(cv_folds=cv_folds)
        
        # Step 5: Train and evaluate models
        self.train_and_evaluate_models()
        
        # Step 6: Analyze feature importance
        self.analyze_feature_importance()
        
        # Step 7: Statistical comparison
        self.statistical_comparison()
        
        # Step 8: Create enhanced visualizations
        self.create_enhanced_visualizations()
        
        # Step 9: Generate report
        report = self.generate_report()
        
        print("\n" + "="*80)
        print("✅ ENHANCED ANALYSIS COMPLETE!")
        print("="*80)
        print(f"🏆 Best performing model: {report['best_model']}")
        print(f"📊 Best CV F1-Score: {self.cv_scores[report['best_model']]['f1']['mean']:.4f}")
        print("\n📁 Generated files:")
        print("   - enhanced_model_analysis.png")
        print("   - suicide_risk_analysis_report.json")
        
        return report

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Enhanced Suicide Risk Prediction System')
    parser.add_argument('--predict', type=int, help='Predict risk for specific patient ID')
    parser.add_argument('--no-tuning', action='store_true', help='Skip hyperparameter tuning for faster execution')
    parser.add_argument('--components', type=int, default=50, help='Number of PCA components (default: 50)')
    parser.add_argument('--cv-folds', type=int, default=5, help='Number of CV folds (default: 5)')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = EnhancedSuicideRiskPredictor('complete_data.csv')
    
    if args.predict:
        # Quick prediction mode - load data and predict
        predictor.load_and_explore_data()
        predictor.preprocess_data(n_components=args.components)
        
        # Use pre-trained models or train quickly
        predictor.models = {
            'SVM': SVC(probability=True, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1)
        }
        
        # Quick training
        for model in predictor.models.values():
            model.fit(predictor.X, predictor.y)
        
        # Predict
        result = predictor.predict_patient_risk_enhanced(args.predict)
        
    else:
        # Full analysis mode
        tune_hyperparameters = not args.no_tuning
        report = predictor.run_enhanced_analysis(
            n_components=args.components,
            cv_folds=args.cv_folds,
            tune_hyperparameters=tune_hyperparameters
        )
        
        # Example predictions
        print("\n" + "="*80)
        print("🔍 SAMPLE PATIENT PREDICTIONS")
        print("="*80)
        
        sample_ids = predictor.data['ID'].sample(2).tolist()
        for patient_id in sample_ids:
            predictor.predict_patient_risk_enhanced(patient_id)
            print("\n" + "-"*60)

if __name__ == "__main__":
    main()
