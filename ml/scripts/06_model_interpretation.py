#!/usr/bin/env python3
"""
Model Interpretation & Analysis
GOAL: Understand how the fraud detection model makes decisions
APPROACH: SHAP values, feature importance, decision boundaries
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, precision_recall_curve
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Try to import SHAP for model explanation
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available - install with: pip install shap")

def load_model_and_data():
    """Load the best model and test data"""
    print("Loading optimized model and data...")
    
    # Load model
    models_dir = Path(__file__).parent.parent / 'models' / 'improved'
    
    # Find the most recent improved model
    model_files = list(models_dir.glob("xgboost_improved_*.pkl"))
    if not model_files:
        # Fallback to final model
        models_dir = Path(__file__).parent.parent / 'models' / 'final'
        model_files = list(models_dir.glob("final_model_*.pkl"))
    
    if not model_files:
        raise FileNotFoundError("No trained model found!")
    
    latest_model_file = max(model_files, key=lambda x: x.stat().st_mtime)
    model = joblib.load(latest_model_file)
    
    # Load data
    data_dir = Path(__file__).parent.parent / 'data' / 'processed'
    X_test = np.load(data_dir / 'X_test_final.npy')
    y_test = np.load(data_dir / 'y_test.npy')
    
    with open(data_dir / 'final_features.json', 'r') as f:
        feature_names = json.load(f)
    
    print(f"Model loaded: {latest_model_file.name}")
    print(f"Test data: {len(X_test):,} samples, {len(feature_names)} features")
    
    return model, X_test, y_test, feature_names

def analyze_feature_importance(model, feature_names):
    """Analyze XGBoost feature importance"""
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    # Get feature importance
    importance = model.feature_importances_
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print("Top 10 Most Important Features:")
    for i, row in importance_df.head(10).iterrows():
        print(f"{i+1:2d}. {row['feature']:25s}: {row['importance']:.4f}")
    
    # Feature importance insights
    total_imp = importance.sum()
    top5_imp = importance_df.head(5)['importance'].sum()
    top10_imp = importance_df.head(10)['importance'].sum()
    
    print(f"\nFeature Concentration:")
    print(f"Top 5 features: {top5_imp/total_imp:.1%} of total importance")
    print(f"Top 10 features: {top10_imp/total_imp:.1%} of total importance")
    
    # Save feature importance plot
    plt.figure(figsize=(10, 8))
    top_features = importance_df.head(15)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 15 Feature Importance (XGBoost)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    plots_dir = Path(__file__).parent.parent / 'experiments' / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return importance_df

def analyze_model_performance_curves(model, X_test, y_test):
    """Analyze ROC and Precision-Recall curves"""
    print("\n" + "="*60)
    print("PERFORMANCE CURVE ANALYSIS")
    print("="*60)
    
    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # ROC Curve
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba)
    
    # Precision-Recall Curve
    precision, recall, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    # Find optimal thresholds
    # ROC: maximize TPR - FPR
    roc_optimal_idx = np.argmax(tpr - fpr)
    roc_optimal_threshold = roc_thresholds[roc_optimal_idx]
    
    # PR: maximize F1 score
    f1_scores = 2 * (precision * recall) / (precision + recall)
    f1_scores = np.nan_to_num(f1_scores)  # Handle division by zero
    pr_optimal_idx = np.argmax(f1_scores)
    pr_optimal_threshold = pr_thresholds[pr_optimal_idx] if len(pr_thresholds) > pr_optimal_idx else 0.5
    
    print(f"Optimal Thresholds:")
    print(f"ROC-based: {roc_optimal_threshold:.4f} (TPR={tpr[roc_optimal_idx]:.3f}, FPR={fpr[roc_optimal_idx]:.3f})")
    print(f"PR-based:  {pr_optimal_threshold:.4f} (Precision={precision[pr_optimal_idx]:.3f}, Recall={recall[pr_optimal_idx]:.3f})")
    
    # Create performance plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # ROC Curve
    ax1.plot(fpr, tpr, linewidth=2, label=f'ROC Curve')
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    ax1.scatter(fpr[roc_optimal_idx], tpr[roc_optimal_idx], 
               color='red', s=100, label=f'Optimal (threshold={roc_optimal_threshold:.3f})')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    ax2.plot(recall, precision, linewidth=2, label=f'PR Curve')
    ax2.axhline(y=y_test.mean(), color='k', linestyle='--', alpha=0.5, 
               label=f'Random (baseline={y_test.mean():.4f})')
    ax2.scatter(recall[pr_optimal_idx], precision[pr_optimal_idx], 
               color='red', s=100, label=f'Optimal (threshold={pr_optimal_threshold:.3f})')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plots_dir = Path(__file__).parent.parent / 'experiments' / 'plots'
    plt.savefig(plots_dir / 'performance_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'roc_optimal_threshold': roc_optimal_threshold,
        'pr_optimal_threshold': pr_optimal_threshold,
        'roc_optimal_tpr': tpr[roc_optimal_idx],
        'roc_optimal_fpr': fpr[roc_optimal_idx],
        'pr_optimal_precision': precision[pr_optimal_idx],
        'pr_optimal_recall': recall[pr_optimal_idx]
    }

def shap_analysis(model, X_test, feature_names, sample_size=1000):
    """SHAP analysis for model interpretability"""
    if not SHAP_AVAILABLE:
        print("\nSHAP analysis skipped - install shap package for model explanations")
        return None
    
    print("\n" + "="*60)
    print("SHAP MODEL INTERPRETABILITY ANALYSIS")
    print("="*60)
    
    # Use a sample for SHAP analysis (computational efficiency)
    if len(X_test) > sample_size:
        sample_idx = np.random.choice(len(X_test), sample_size, replace=False)
        X_sample = X_test[sample_idx]
        print(f"Using {sample_size} samples for SHAP analysis...")
    else:
        X_sample = X_test
        print(f"Using all {len(X_test)} samples for SHAP analysis...")
    
    try:
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # Convert to DataFrame for easier analysis
        shap_df = pd.DataFrame(shap_values, columns=feature_names)
        
        # Global feature importance (mean absolute SHAP values)
        global_importance = np.abs(shap_values).mean(0)
        global_importance_df = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': global_importance
        }).sort_values('mean_abs_shap', ascending=False)
        
        print("Top 10 Features by SHAP Importance:")
        for i, row in global_importance_df.head(10).iterrows():
            print(f"{i+1:2d}. {row['feature']:25s}: {row['mean_abs_shap']:.4f}")
        
        # Create SHAP plots
        plots_dir = Path(__file__).parent.parent / 'experiments' / 'plots'
        
        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        plt.savefig(plots_dir / 'shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Feature importance plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                         plot_type="bar", show=False)
        plt.savefig(plots_dir / 'shap_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"SHAP plots saved to {plots_dir}")
        
        return global_importance_df
        
    except Exception as e:
        print(f"SHAP analysis failed: {e}")
        return None

def threshold_analysis(model, X_test, y_test):
    """Analyze model performance at different thresholds"""
    print("\n" + "="*60)
    print("THRESHOLD ANALYSIS FOR BUSINESS DECISIONS")
    print("="*60)
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Test different thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    results = []
    for threshold in thresholds:
        y_pred_thresh = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        tp = ((y_pred_thresh == 1) & (y_test == 1)).sum()
        fp = ((y_pred_thresh == 1) & (y_test == 0)).sum()
        tn = ((y_pred_thresh == 0) & (y_test == 0)).sum()
        fn = ((y_pred_thresh == 0) & (y_test == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        fraud_detection_rate = recall
        false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'fraud_detection_rate': fraud_detection_rate,
            'false_alarm_rate': false_alarm_rate,
            'flagged_transactions': tp + fp
        })
    
    threshold_df = pd.DataFrame(results)
    
    print("Threshold Analysis Results:")
    print(threshold_df.round(4).to_string(index=False))
    
    # Find business-optimal threshold (balance fraud detection and false alarms)
    # Maximize fraud detection while keeping false alarm rate reasonable (<15%)
    viable_thresholds = threshold_df[threshold_df['false_alarm_rate'] <= 0.15]
    if len(viable_thresholds) > 0:
        business_optimal = viable_thresholds.loc[viable_thresholds['fraud_detection_rate'].idxmax()]
        print(f"\nRecommended Business Threshold: {business_optimal['threshold']}")
        print(f"  Fraud Detection Rate: {business_optimal['fraud_detection_rate']:.1%}")
        print(f"  False Alarm Rate: {business_optimal['false_alarm_rate']:.1%}")
        print(f"  Precision: {business_optimal['precision']:.1%}")
    
    # Save threshold analysis
    experiments_dir = Path(__file__).parent.parent / 'experiments'
    threshold_df.to_csv(experiments_dir / 'threshold_analysis.csv', index=False)
    
    return threshold_df

def save_interpretation_results(feature_importance_df, shap_importance_df, threshold_df, 
                               performance_metrics, feature_names):
    """Save all interpretation results"""
    print("\n" + "="*60)
    print("SAVING INTERPRETATION RESULTS")
    print("="*60)
    
    experiments_dir = Path(__file__).parent.parent / 'experiments'
    
    # Comprehensive interpretation report
    interpretation_report = {
        'model_type': 'XGBoost Fraud Detector',
        'analysis_date': pd.Timestamp.now().isoformat(),
        'total_features': len(feature_names),
        'performance_metrics': performance_metrics,
        'feature_importance': feature_importance_df.to_dict('records'),
        'threshold_analysis': threshold_df.to_dict('records')
    }
    
    if shap_importance_df is not None:
        interpretation_report['shap_importance'] = shap_importance_df.to_dict('records')
    
    # Save comprehensive report
    with open(experiments_dir / 'model_interpretation_report.json', 'w') as f:
        json.dump(interpretation_report, f, indent=2, default=str)
    
    # Save individual CSVs
    feature_importance_df.to_csv(experiments_dir / 'feature_importance_detailed.csv', index=False)
    
    if shap_importance_df is not None:
        shap_importance_df.to_csv(experiments_dir / 'shap_importance_detailed.csv', index=False)
    
    print(f"Interpretation results saved to {experiments_dir}")
    print("Generated files:")
    print("  - model_interpretation_report.json (comprehensive report)")
    print("  - feature_importance_detailed.csv")
    print("  - threshold_analysis.csv")
    if shap_importance_df is not None:
        print("  - shap_importance_detailed.csv")
    print("  - plots/feature_importance.png")
    print("  - plots/performance_curves.png")
    if shap_importance_df is not None:
        print("  - plots/shap_summary.png")
        print("  - plots/shap_importance.png")

def main():
    """Run complete model interpretation analysis"""
    print("MODEL INTERPRETATION & ANALYSIS")
    print("="*60)
    print("GOAL: Understand how the fraud detection model works")
    
    # Load model and data
    model, X_test, y_test, feature_names = load_model_and_data()
    
    # Feature importance analysis
    feature_importance_df = analyze_feature_importance(model, feature_names)
    
    # Performance curve analysis
    performance_metrics = analyze_model_performance_curves(model, X_test, y_test)
    
    # SHAP analysis (if available)
    shap_importance_df = shap_analysis(model, X_test, feature_names)
    
    # Threshold analysis
    threshold_df = threshold_analysis(model, X_test, y_test)
    
    # Save all results
    save_interpretation_results(
        feature_importance_df, shap_importance_df, threshold_df,
        performance_metrics, feature_names
    )
    
    print("\n" + "="*60)
    print("MODEL INTERPRETATION COMPLETE!")
    print("="*60)
    print("Key insights:")
    print(f"- Model uses {len(feature_names)} features effectively")
    print(f"- Top feature: {feature_importance_df.iloc[0]['feature']}")
    print(f"- Recommended threshold: Check threshold_analysis.csv")
    print("- Visual explanations saved in experiments/plots/")
    print("Model interpretation ready for business stakeholders!")

if __name__ == "__main__":
    main()