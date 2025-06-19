#!/usr/bin/env python3
"""
Focused Model Training for Fraud Detection
AIM: Detect fraudulent transactions with high precision and recall
KEY METRICS: ROC-AUC (primary), PR-AUC (secondary) for imbalanced data
Models: Random Forest, XGBoost, LightGBM with built-in class imbalance handling
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import joblib
from datetime import datetime

from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_val_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, f1_score, precision_score, recall_score,
    accuracy_score
)
import xgboost as xgb
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available, will install if needed")
import warnings
warnings.filterwarnings('ignore')

def load_final_data(use_subset=True, subset_size=50000):
    """Load the final processed datasets with optional subset for faster training"""
    data_dir = Path(__file__).parent.parent / 'data' / 'processed'
    
    X_train = np.load(data_dir / 'X_train_final.npy')
    X_test = np.load(data_dir / 'X_test_final.npy')
    
    # Load original y data
    y_train = np.load(data_dir / 'y_train.npy')
    y_val = np.load(data_dir / 'y_val.npy')
    y_test = np.load(data_dir / 'y_test.npy')
    
    # Combine train and validation
    y_train_combined = np.hstack([y_train, y_val])
    
    with open(data_dir / 'final_features.json', 'r') as f:
        final_features = json.load(f)
    
    # SPEED OPTIMIZATION: Use stratified subset for hyperparameter tuning
    if use_subset and len(X_train) > subset_size:
        print(f"Using stratified subset of {subset_size:,} samples for faster training...")
        
        from sklearn.model_selection import train_test_split
        X_train_subset, _, y_train_subset, _ = train_test_split(
            X_train, y_train_combined, 
            train_size=subset_size, 
            stratify=y_train_combined,
            random_state=42
        )
        X_train = X_train_subset
        y_train_combined = y_train_subset
        
        print(f"Subset fraud rate: {y_train_combined.mean():.4f}")
    
    print("Final data loaded:")
    print(f"Training samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")
    print(f"Features: {len(final_features)}")
    print(f"Fraud rate: {y_train_combined.mean():.4f}")
    
    return X_train, X_test, y_train_combined, y_test, final_features

def setup_cross_validation():
    """Setup stratified cross-validation for imbalanced data"""
    print("\n" + "="*50)
    print("CROSS-VALIDATION SETUP")
    print("="*50)
    
    # Use 5-fold stratified CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("Using 5-fold Stratified Cross-Validation")
    print("Ensures balanced class distribution in each fold")
    
    return cv

def fast_model_comparison_and_tuning(X_train, y_train):
    """Fast comparison and tuning of RF, XGBoost, and LightGBM with speed optimizations"""
    print("\n" + "="*50)
    print("FAST MODEL TRAINING: RF, XGBOOST, LIGHTGBM")
    print("="*50)
    
    # Calculate scale_pos_weight for tree-based models
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Class imbalance ratio: {scale_pos_weight:.1f}:1")
    
    # Use 3-fold CV for speed, then validate with 5-fold for best model
    cv_fast = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    models_configs = {
        'Random Forest': {
            'base_model': RandomForestClassifier(
                random_state=42,
                class_weight='balanced',
                n_estimators=100,  # Start moderate
                n_jobs=-1  # Use all cores
            ),
            'param_grid': {
                'n_estimators': [100, 200],
                'max_depth': [10, 15],
                'min_samples_split': [2, 5]
            }
        },
        'XGBoost': {
            'base_model': xgb.XGBClassifier(
                random_state=42,
                scale_pos_weight=scale_pos_weight,
                eval_metric='logloss',
                n_jobs=-1,  # Use all cores
                tree_method='hist'  # Faster training
            ),
            'param_grid': {
                'n_estimators': [100, 200],
                'max_depth': [3, 6],
                'learning_rate': [0.1, 0.2]
            }
        }
    }
    
    # Add LightGBM if available (naturally fast)
    if LIGHTGBM_AVAILABLE:
        models_configs['LightGBM'] = {
            'base_model': lgb.LGBMClassifier(
                random_state=42,
                scale_pos_weight=scale_pos_weight,
                verbosity=-1,
                n_jobs=-1,  # Use all cores
                boosting_type='gbdt'  # Default, fast
            ),
            'param_grid': {
                'n_estimators': [100, 200],
                'max_depth': [3, 6],
                'learning_rate': [0.1, 0.2]
            }
        }
    
    results = {}
    
    for name, config in models_configs.items():
        print(f"\nTraining and tuning {name}...")
        
        # Use GridSearchCV with reduced CV for speed
        grid_search = GridSearchCV(
            config['base_model'], 
            config['param_grid'],
            cv=cv_fast,  # 3-fold for speed
            scoring='roc_auc',
            n_jobs=1,  # Let individual models use n_jobs=-1
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        results[name] = {
            'best_model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'cv_roc_auc': grid_search.best_score_,
            'cv_folds': 3
        }
        
        print(f"  Best ROC-AUC (3-fold): {grid_search.best_score_:.4f}")
        print(f"  Best params: {grid_search.best_params_}")
    
    # Select best model and validate with 5-fold CV
    best_model_name = max(results.keys(), key=lambda x: results[x]['cv_roc_auc'])
    best_model = results[best_model_name]['best_model']
    
    print(f"\nBest model: {best_model_name}")
    print(f"Validating with 5-fold CV...")
    
    cv_5fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    final_cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv_5fold, scoring='roc_auc')
    
    results[best_model_name]['cv_roc_auc_5fold'] = final_cv_scores.mean()
    results[best_model_name]['cv_std_5fold'] = final_cv_scores.std()
    
    print(f"5-fold CV ROC-AUC: {final_cv_scores.mean():.4f} (+/- {final_cv_scores.std()*2:.4f})")
    
    return best_model, best_model_name, results

def hyperparameter_tuning(X_train, y_train, cv, baseline_results):
    """Focused hyperparameter tuning with minimal but effective search"""
    print("\n" + "="*50)
    print("HYPERPARAMETER TUNING")
    print("="*50)
    
    tuned_results = {}
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    # 1. Random Forest - focused grid
    print("\nTuning Random Forest...")
    rf_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15],
        'min_samples_split': [2, 5]
    }
    
    rf_base = RandomForestClassifier(random_state=42, class_weight='balanced')
    rf_search = GridSearchCV(
        rf_base, rf_param_grid, 
        cv=cv, scoring='roc_auc', n_jobs=-1
    )
    rf_search.fit(X_train, y_train)
    
    tuned_results['Random Forest'] = {
        'best_params': rf_search.best_params_,
        'best_score': rf_search.best_score_,
        'best_model': rf_search.best_estimator_
    }
    
    print(f"Best RF ROC-AUC: {rf_search.best_score_:.4f}")
    print(f"Best RF params: {rf_search.best_params_}")
    
    # 2. XGBoost - focused grid
    print("\nTuning XGBoost...")
    xgb_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6],
        'learning_rate': [0.1, 0.2]
    }
    
    xgb_base = xgb.XGBClassifier(
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss'
    )
    xgb_search = GridSearchCV(
        xgb_base, xgb_param_grid,
        cv=cv, scoring='roc_auc', n_jobs=-1
    )
    xgb_search.fit(X_train, y_train)
    
    tuned_results['XGBoost'] = {
        'best_params': xgb_search.best_params_,
        'best_score': xgb_search.best_score_,
        'best_model': xgb_search.best_estimator_
    }
    
    print(f"Best XGB ROC-AUC: {xgb_search.best_score_:.4f}")
    print(f"Best XGB params: {xgb_search.best_params_}")
    
    # 3. LightGBM - if available
    if LIGHTGBM_AVAILABLE:
        print("\nTuning LightGBM...")
        lgb_param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'learning_rate': [0.1, 0.2]
        }
        
        lgb_base = lgb.LGBMClassifier(
            random_state=42,
            scale_pos_weight=scale_pos_weight,
            verbosity=-1
        )
        lgb_search = GridSearchCV(
            lgb_base, lgb_param_grid,
            cv=cv, scoring='roc_auc', n_jobs=-1
        )
        lgb_search.fit(X_train, y_train)
        
        tuned_results['LightGBM'] = {
            'best_params': lgb_search.best_params_,
            'best_score': lgb_search.best_score_,
            'best_model': lgb_search.best_estimator_
        }
        
        print(f"Best LGB ROC-AUC: {lgb_search.best_score_:.4f}")
        print(f"Best LGB params: {lgb_search.best_params_}")
    
    return tuned_results

def select_final_model(baseline_results, tuned_results):
    """Select the best performing model"""
    print("\n" + "="*50)
    print("FINAL MODEL SELECTION")
    print("="*50)
    
    # Compare all results
    all_results = []
    
    # Baseline results
    for name, results in baseline_results.items():
        all_results.append({
            'Model': f"{name} (Baseline)",
            'ROC-AUC': results['roc_auc_mean'],
            'PR-AUC': results['pr_auc_mean'],
            'Type': 'Baseline'
        })
    
    # Tuned results
    for name, results in tuned_results.items():
        all_results.append({
            'Model': f"{name} (Tuned)",
            'ROC-AUC': results['best_score'],
            'PR-AUC': 0,  # Not calculated during tuning
            'Type': 'Tuned'
        })
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame(all_results)
    comparison_df = comparison_df.sort_values('ROC-AUC', ascending=False)
    
    print("Model Performance Comparison:")
    print(comparison_df.round(4).to_string(index=False))
    
    # Select best model
    best_overall = comparison_df.iloc[0]
    best_score = best_overall['ROC-AUC']
    
    if best_overall['Type'] == 'Tuned':
        model_name = best_overall['Model'].split(' (')[0]
        final_model = tuned_results[model_name]['best_model']
    else:
        model_name = best_overall['Model'].split(' (')[0]
        final_model = baseline_results[model_name]['model']
    
    print(f"\nFINAL SELECTED MODEL: {model_name}")
    print(f"Cross-validation ROC-AUC: {best_score:.4f}")
    
    return final_model, model_name, comparison_df

def evaluate_on_test_set(final_model, model_name, X_train, y_train, X_test, y_test):
    """Final evaluation on held-out test set - focus on fraud detection metrics"""
    print("\n" + "="*50)
    print("FINAL TEST SET EVALUATION")
    print("="*50)
    
    # Train final model on full training set
    print(f"Training final {model_name} on full training set...")
    final_model.fit(X_train, y_train)
    
    # Predictions on test set
    y_pred = final_model.predict(X_test)
    y_pred_proba = final_model.predict_proba(X_test)[:, 1]
    
    # Key fraud detection metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    print(f"FRAUD DETECTION PERFORMANCE:")
    print(f"ROC-AUC: {roc_auc:.4f} (Higher = Better overall discrimination)")
    print(f"PR-AUC:  {pr_auc:.4f} (Higher = Better for imbalanced data)")
    
    # Confusion matrix analysis
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate derived metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall/True Positive Rate
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0    # Positive Predictive Value
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0          # Negative Predictive Value
    
    print(f"\nCONFUSION MATRIX ANALYSIS:")
    print(f"True Negatives (Correct Normal):   {tn:,}")
    print(f"False Positives (False Alarms):    {fp:,}")
    print(f"False Negatives (Missed Fraud):    {fn:,}")
    print(f"True Positives (Caught Fraud):     {tp:,}")
    
    print(f"\nDETAILED METRICS:")
    print(f"Sensitivity (Recall): {sensitivity:.4f} - How many frauds we catch")
    print(f"Specificity:          {specificity:.4f} - How many normals we correctly identify")
    print(f"Precision:            {precision:.4f} - Of flagged transactions, how many are actual fraud")
    print(f"F1-Score:             {f1_score(y_test, y_pred):.4f} - Balance of precision and recall")
    
    # Business impact metrics
    fraud_detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    print(f"\nBUSINESS IMPACT:")
    print(f"Fraud Detection Rate: {fraud_detection_rate:.1%} of frauds caught")
    print(f"False Alarm Rate:     {false_alarm_rate:.1%} of normal transactions flagged")
    
    test_results = {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': {
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
        },
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1_score(y_test, y_pred),
        'fraud_detection_rate': fraud_detection_rate,
        'false_alarm_rate': false_alarm_rate
    }
    
    return test_results

def save_final_results(final_model, model_name, test_results, comparison_df, 
                      training_results, final_features):
    """Save the final model and all results"""
    print("\n" + "="*50)
    print("SAVING FINAL RESULTS")
    print("="*50)
    
    # Create directories
    models_dir = Path(__file__).parent.parent / 'models' / 'final'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    experiments_dir = Path(__file__).parent.parent / 'experiments'
    
    # Save final model
    model_filename = f"final_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    joblib.dump(final_model, models_dir / model_filename)
    
    # Save model metadata
    model_metadata = {
        'model_name': model_name,
        'model_file': model_filename,
        'features': final_features,
        'feature_count': len(final_features),
        'test_performance': test_results,
        'training_date': datetime.now().isoformat(),
        'training_results': training_results
    }
    
    with open(models_dir / 'model_metadata.json', 'w') as f:
        json.dump(model_metadata, f, indent=2, default=str)
    
    # Save performance comparison (if available)
    if comparison_df is not None:
        comparison_df.to_csv(experiments_dir / 'model_comparison_final.csv', index=False)
    
    # Save test results
    with open(experiments_dir / 'final_test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    
    print(f"Final model saved: {models_dir / model_filename}")
    print(f"Model metadata saved: {models_dir / 'model_metadata.json'}")
    print(f"Performance results saved to {experiments_dir}")

def main():
    """Run optimized model training pipeline for RF, XGBoost, LightGBM"""
    
    print("OPTIMIZED FRAUD DETECTION MODEL TRAINING")
    print("="*60)
    print("FOCUS: Random Forest, XGBoost, LightGBM")
    print("OPTIMIZATIONS: Subset training, 3-fold CV, parallel processing")
    
    # Load final data with subset for faster training
    X_train, X_test, y_train, y_test, final_features = load_final_data(
        use_subset=True, subset_size=50000
    )
    
    # Fast model comparison and tuning
    best_model, model_name, training_results = fast_model_comparison_and_tuning(X_train, y_train)
    
    # Retrain best model on full dataset for final evaluation
    print(f"\nRetraining {model_name} on full dataset...")
    X_train_full, X_test_full, y_train_full, y_test_full, _ = load_final_data(use_subset=False)
    
    # Evaluate on test set
    test_results = evaluate_on_test_set(
        best_model, model_name, X_train_full, y_train_full, X_test_full, y_test_full
    )
    
    # Save results
    save_final_results(
        best_model, model_name, test_results, None, 
        training_results, final_features
    )
    
    print("\n" + "="*60)
    print("OPTIMIZED TRAINING COMPLETE!")
    print("="*60)
    print(f"Best Model: {model_name}")
    print(f"CV ROC-AUC (5-fold): {training_results[model_name].get('cv_roc_auc_5fold', 'N/A'):.4f}")
    print(f"Test ROC-AUC: {test_results['roc_auc']:.4f}")
    print(f"Test PR-AUC: {test_results['pr_auc']:.4f}")
    print("Model ready for deployment!")

if __name__ == "__main__":
    main()