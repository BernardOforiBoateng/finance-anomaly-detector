#!/usr/bin/env python3
"""
XGBoost Optimization for Fraud Detection
Focus: Maximize ROC-AUC and PR-AUC for imbalanced fraud detection
Techniques: Advanced hyperparameter tuning, early stopping, feature importance analysis
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import joblib
from datetime import datetime

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix
import xgboost as xgb
from scipy.stats import uniform, randint
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
    
    print("Data loaded for XGBoost optimization:")
    print(f"Training: {len(X_train):,} samples, {len(final_features)} features")
    print(f"Test: {len(X_test):,} samples")
    print(f"Fraud rate: {y_train_combined.mean():.3%}")
    
    return X_train, X_test, y_train_combined, y_test, final_features

def xgboost_baseline(X_train, y_train):
    """Establish XGBoost baseline with current best parameters"""
    print("\n" + "="*60)
    print("XGBOOST BASELINE PERFORMANCE")
    print("="*60)
    
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    baseline_model = xgb.XGBClassifier(
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        n_jobs=-1,
        tree_method='hist',
        # Current best parameters from previous training
        learning_rate=0.1,
        max_depth=3,
        n_estimators=100
    )
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    roc_scores = cross_val_score(baseline_model, X_train, y_train, cv=cv, scoring='roc_auc')
    pr_scores = cross_val_score(baseline_model, X_train, y_train, cv=cv, scoring='average_precision')
    
    print(f"Baseline ROC-AUC: {roc_scores.mean():.4f} (+/- {roc_scores.std()*2:.4f})")
    print(f"Baseline PR-AUC:  {pr_scores.mean():.4f} (+/- {pr_scores.std()*2:.4f})")
    
    return baseline_model, roc_scores.mean(), pr_scores.mean()

def advanced_xgboost_tuning(X_train, y_train):
    """Advanced hyperparameter tuning for XGBoost"""
    print("\n" + "="*60)
    print("ADVANCED XGBOOST HYPERPARAMETER TUNING")
    print("="*60)
    
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    # Expanded parameter space for better optimization
    param_distributions = {
        # Core boosting parameters
        'n_estimators': randint(100, 500),
        'learning_rate': uniform(0.01, 0.29),  # 0.01 to 0.3
        'max_depth': randint(3, 12),
        
        # Regularization parameters
        'reg_alpha': uniform(0, 1),      # L1 regularization
        'reg_lambda': uniform(0, 1),     # L2 regularization
        'gamma': uniform(0, 5),          # Minimum split loss
        
        # Sampling parameters for better generalization
        'subsample': uniform(0.6, 0.4),        # 0.6 to 1.0
        'colsample_bytree': uniform(0.6, 0.4), # 0.6 to 1.0
        'colsample_bylevel': uniform(0.6, 0.4), # 0.6 to 1.0
        
        # Tree structure
        'min_child_weight': randint(1, 10),
        'max_delta_step': randint(0, 10)
    }
    
    base_model = xgb.XGBClassifier(
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        n_jobs=-1,
        tree_method='hist',
        verbosity=0
    )
    
    # Use RandomizedSearchCV for efficient exploration
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("Running randomized search (25 iterations for speed)...")
    random_search = RandomizedSearchCV(
        base_model,
        param_distributions,
        n_iter=25,  # Reduced for faster execution
        cv=cv,
        scoring='roc_auc',
        n_jobs=1,  # Let XGBoost use parallelization
        random_state=42,
        verbose=1
    )
    
    random_search.fit(X_train, y_train)
    
    print(f"\nBest ROC-AUC: {random_search.best_score_:.4f}")
    print(f"Best parameters:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")
    
    return random_search.best_estimator_, random_search.best_params_, random_search.best_score_

def fine_tune_best_model(best_model, best_params, X_train, y_train):
    """Fine-tune around the best parameters"""
    print("\n" + "="*60)
    print("FINE-TUNING AROUND BEST PARAMETERS")
    print("="*60)
    
    # Create a refined search space around best parameters
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    # Fine-tune key parameters around best values
    n_est = best_params['n_estimators']
    lr = best_params['learning_rate']
    depth = best_params['max_depth']
    
    fine_tune_params = {
        'n_estimators': [max(100, n_est-50), n_est, n_est+50, n_est+100],
        'learning_rate': [lr*0.8, lr*0.9, lr, lr*1.1, lr*1.2],
        'max_depth': [max(3, depth-1), depth, depth+1],
        'reg_alpha': [best_params.get('reg_alpha', 0)*0.5, 
                     best_params.get('reg_alpha', 0), 
                     best_params.get('reg_alpha', 0)*1.5],
        'reg_lambda': [best_params.get('reg_lambda', 0)*0.5, 
                      best_params.get('reg_lambda', 0), 
                      best_params.get('reg_lambda', 0)*1.5]
    }
    
    # Clean up parameter ranges
    fine_tune_params = {k: [v for v in vals if v >= 0] for k, vals in fine_tune_params.items()}
    
    base_model = xgb.XGBClassifier(
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        n_jobs=-1,
        tree_method='hist',
        verbosity=0,
        # Set other best parameters
        **{k: v for k, v in best_params.items() if k not in fine_tune_params}
    )
    
    from sklearn.model_selection import GridSearchCV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("Running fine-tuning grid search...")
    grid_search = GridSearchCV(
        base_model,
        fine_tune_params,
        cv=cv,
        scoring='roc_auc',
        n_jobs=1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Fine-tuned ROC-AUC: {grid_search.best_score_:.4f}")
    print(f"Improvement: {grid_search.best_score_ - best_params.get('cv_score', 0):+.4f}")
    print(f"Final best parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

def analyze_feature_importance(model, feature_names):
    """Analyze XGBoost feature importance"""
    print("\n" + "="*60)
    print("XGBOOST FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    # Get different types of importance
    importance_gain = model.feature_importances_
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_gain': importance_gain
    }).sort_values('importance_gain', ascending=False)
    
    print("Top 10 Most Important Features:")
    for i, row in importance_df.head(10).iterrows():
        print(f"{row.name+1:2d}. {row['feature']:25s}: {row['importance_gain']:.4f}")
    
    # Feature importance insights
    total_importance = importance_gain.sum()
    top5_importance = importance_df.head(5)['importance_gain'].sum()
    top10_importance = importance_df.head(10)['importance_gain'].sum()
    
    print(f"\nFeature Importance Insights:")
    print(f"Top 5 features contribute: {top5_importance/total_importance:.1%} of importance")
    print(f"Top 10 features contribute: {top10_importance/total_importance:.1%} of importance")
    
    return importance_df

def comprehensive_evaluation(model, X_train, y_train, X_test, y_test, model_name="Optimized XGBoost"):
    """Comprehensive evaluation of the optimized model"""
    print("\n" + "="*60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*60)
    
    # Train on full training set
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Key metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    print(f"FRAUD DETECTION PERFORMANCE:")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    
    # Detailed confusion matrix analysis
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate all metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    fraud_detection_rate = sensitivity
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    print(f"\nCONFUSION MATRIX:")
    print(f"True Negatives:  {tn:,}")
    print(f"False Positives: {fp:,}")
    print(f"False Negatives: {fn:,}")
    print(f"True Positives:  {tp:,}")
    
    print(f"\nPERFORMANCE METRICS:")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity:          {specificity:.4f}")
    print(f"Precision:            {precision:.4f}")
    print(f"Negative Pred Value:  {npv:.4f}")
    
    print(f"\nBUSINESS IMPACT:")
    print(f"Fraud Detection Rate: {fraud_detection_rate:.1%}")
    print(f"False Alarm Rate:     {false_alarm_rate:.1%}")
    
    results = {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'npv': npv,
        'fraud_detection_rate': fraud_detection_rate,
        'false_alarm_rate': false_alarm_rate
    }
    
    return results

def save_optimized_model(model, final_params, cv_score, test_results, importance_df, feature_names):
    """Save the optimized XGBoost model"""
    print("\n" + "="*60)
    print("SAVING OPTIMIZED XGBOOST MODEL")
    print("="*60)
    
    models_dir = Path(__file__).parent.parent / 'models' / 'optimized'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    experiments_dir = Path(__file__).parent.parent / 'experiments'
    
    # Save model
    model_filename = f"xgboost_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    joblib.dump(model, models_dir / model_filename)
    
    # Save comprehensive metadata
    metadata = {
        'model_name': 'Optimized XGBoost Fraud Detector',
        'model_file': model_filename,
        'optimization_date': datetime.now().isoformat(),
        'features': feature_names,
        'feature_count': len(feature_names),
        'final_hyperparameters': final_params,
        'cv_roc_auc': cv_score,
        'test_performance': test_results,
        'model_purpose': 'Financial transaction fraud detection with optimized performance'
    }
    
    with open(models_dir / 'optimized_model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    # Save feature importance
    importance_df.to_csv(experiments_dir / 'xgboost_feature_importance.csv', index=False)
    
    # Save test results
    with open(experiments_dir / 'xgboost_optimized_results.json', 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    
    print(f"Optimized model saved: {models_dir / model_filename}")
    print(f"Metadata saved: {models_dir / 'optimized_model_metadata.json'}")
    print(f"Feature importance saved: {experiments_dir / 'xgboost_feature_importance.csv'}")

def main():
    """Run complete XGBoost optimization pipeline"""
    
    print("XGBOOST OPTIMIZATION FOR FRAUD DETECTION")
    print("="*60)
    print("GOAL: Maximize fraud detection performance")
    print("APPROACH: Advanced hyperparameter tuning + feature analysis")
    
    # Load data
    X_train, X_test, y_train, y_test, feature_names = load_data()
    
    # Baseline performance
    baseline_model, baseline_roc, baseline_pr = xgboost_baseline(X_train, y_train)
    
    # Advanced hyperparameter tuning
    best_model, best_params, best_score = advanced_xgboost_tuning(X_train, y_train)
    
    # Fine-tuning around best parameters
    final_model, final_params, final_score = fine_tune_best_model(best_model, best_params, X_train, y_train)
    
    # Feature importance analysis
    importance_df = analyze_feature_importance(final_model, feature_names)
    
    # Comprehensive evaluation
    test_results = comprehensive_evaluation(final_model, X_train, y_train, X_test, y_test)
    
    # Save optimized model
    save_optimized_model(final_model, final_params, final_score, test_results, importance_df, feature_names)
    
    # Summary
    print("\n" + "="*60)
    print("XGBOOST OPTIMIZATION COMPLETE!")
    print("="*60)
    print(f"Baseline ROC-AUC:  {baseline_roc:.4f}")
    print(f"Optimized ROC-AUC: {final_score:.4f}")
    print(f"Improvement:       {final_score - baseline_roc:+.4f}")
    print(f"Test ROC-AUC:      {test_results['roc_auc']:.4f}")
    print(f"Test PR-AUC:       {test_results['pr_auc']:.4f}")
    print(f"Fraud Detection:   {test_results['fraud_detection_rate']:.1%}")
    print("Model ready for deployment!")

if __name__ == "__main__":
    main()