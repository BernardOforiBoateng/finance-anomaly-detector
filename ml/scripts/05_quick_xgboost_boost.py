#!/usr/bin/env python3
"""
Quick XGBoost Performance Boost
Focus: Improve current XGBoost results with minimal parameter tuning
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import joblib
from datetime import datetime

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the final processed datasets"""
    data_dir = Path(__file__).parent.parent / 'data' / 'processed'
    
    X_train = np.load(data_dir / 'X_train_final.npy')
    X_test = np.load(data_dir / 'X_test_final.npy')
    
    y_train = np.load(data_dir / 'y_train.npy')
    y_val = np.load(data_dir / 'y_val.npy')
    y_test = np.load(data_dir / 'y_test.npy')
    
    y_train_combined = np.hstack([y_train, y_val])
    
    with open(data_dir / 'final_features.json', 'r') as f:
        final_features = json.load(f)
    
    print("Data loaded for XGBoost improvement:")
    print(f"Training: {len(X_train):,} samples, {len(final_features)} features")
    print(f"Test: {len(X_test):,} samples")
    print(f"Fraud rate: {y_train_combined.mean():.3%}")
    
    return X_train, X_test, y_train_combined, y_test, final_features

def test_improved_xgboost_configs(X_train, y_train):
    """Test a few key XGBoost configurations to improve performance"""
    print("\n" + "="*60)
    print("TESTING IMPROVED XGBOOST CONFIGURATIONS")
    print("="*60)
    
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Test configurations based on common good practices for imbalanced data
    configs = {
        'Current_Best': {
            'learning_rate': 0.1,
            'max_depth': 3,
            'n_estimators': 100,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'subsample': 1.0,
            'colsample_bytree': 1.0
        },
        'More_Trees': {
            'learning_rate': 0.05,  # Lower learning rate
            'max_depth': 3,
            'n_estimators': 300,    # More trees to compensate
            'reg_alpha': 0,
            'reg_lambda': 1,
            'subsample': 1.0,
            'colsample_bytree': 1.0
        },
        'Deeper_Trees': {
            'learning_rate': 0.1,
            'max_depth': 6,         # Deeper trees
            'n_estimators': 200,
            'reg_alpha': 0.1,       # Some L1 regularization
            'reg_lambda': 1,
            'subsample': 0.8,       # Some subsampling
            'colsample_bytree': 0.8
        },
        'Regularized': {
            'learning_rate': 0.1,
            'max_depth': 4,
            'n_estimators': 200,
            'reg_alpha': 0.3,       # More regularization
            'reg_lambda': 0.3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 1              # Minimum split loss
        }
    }
    
    results = {}
    
    for name, params in configs.items():
        print(f"\nTesting {name} configuration...")
        
        model = xgb.XGBClassifier(
            random_state=42,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            n_jobs=-1,
            tree_method='hist',
            verbosity=0,
            **params
        )
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
        
        results[name] = {
            'cv_roc_auc': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'params': params,
            'model': model
        }
        
        print(f"  ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # Find best configuration
    best_config = max(results.keys(), key=lambda x: results[x]['cv_roc_auc'])
    best_model = results[best_config]['model']
    best_score = results[best_config]['cv_roc_auc']
    
    print(f"\nBest configuration: {best_config}")
    print(f"Best CV ROC-AUC: {best_score:.4f}")
    
    return best_model, best_config, results

def final_evaluation(model, config_name, X_train, y_train, X_test, y_test):
    """Final evaluation on test set"""
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60)
    
    # Train on full training set
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Key metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    print(f"FRAUD DETECTION PERFORMANCE ({config_name}):")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    
    # Confusion matrix analysis
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Business metrics
    fraud_detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    print(f"\nCONFUSION MATRIX:")
    print(f"True Negatives:  {tn:,}")
    print(f"False Positives: {fp:,}")
    print(f"False Negatives: {fn:,}")
    print(f"True Positives:  {tp:,}")
    
    print(f"\nBUSINESS IMPACT:")
    print(f"Fraud Detection Rate: {fraud_detection_rate:.1%}")
    print(f"False Alarm Rate:     {false_alarm_rate:.1%}")
    print(f"Precision:            {precision:.1%}")
    
    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'fraud_detection_rate': fraud_detection_rate,
        'false_alarm_rate': false_alarm_rate,
        'precision': precision,
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
    }

def save_improved_model(model, config_name, test_results, feature_names):
    """Save the improved model"""
    print("\n" + "="*60)
    print("SAVING IMPROVED XGBOOST MODEL")
    print("="*60)
    
    models_dir = Path(__file__).parent.parent / 'models' / 'improved'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_filename = f"xgboost_improved_{config_name.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    joblib.dump(model, models_dir / model_filename)
    
    # Save metadata
    metadata = {
        'model_name': f'Improved XGBoost ({config_name})',
        'model_file': model_filename,
        'improvement_date': datetime.now().isoformat(),
        'features': feature_names,
        'test_performance': test_results,
        'config_used': config_name
    }
    
    with open(models_dir / 'improved_model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"Improved model saved: {models_dir / model_filename}")
    print(f"Metadata saved: {models_dir / 'improved_model_metadata.json'}")

def main():
    """Run quick XGBoost improvement"""
    
    print("QUICK XGBOOST PERFORMANCE IMPROVEMENT")
    print("="*60)
    print("GOAL: Improve current XGBoost results with focused tuning")
    
    # Load data
    X_train, X_test, y_train, y_test, feature_names = load_data()
    
    # Test improved configurations
    best_model, best_config, all_results = test_improved_xgboost_configs(X_train, y_train)
    
    # Final evaluation
    test_results = final_evaluation(best_model, best_config, X_train, y_train, X_test, y_test)
    
    # Save improved model
    save_improved_model(best_model, best_config, test_results, feature_names)
    
    # Summary
    print("\n" + "="*60)
    print("XGBOOST IMPROVEMENT COMPLETE!")
    print("="*60)
    print(f"Best Configuration: {best_config}")
    print(f"CV ROC-AUC: {all_results[best_config]['cv_roc_auc']:.4f}")
    print(f"Test ROC-AUC: {test_results['roc_auc']:.4f}")
    print(f"Test PR-AUC: {test_results['pr_auc']:.4f}")
    print(f"Fraud Detection Rate: {test_results['fraud_detection_rate']:.1%}")
    print("Improved model ready for use!")

if __name__ == "__main__":
    main()