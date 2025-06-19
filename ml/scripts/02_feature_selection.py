#!/usr/bin/env python3
"""
Step 2: Simple, Effective Feature Selection
Based on EDA insights:
- Use only the 13 statistically significant features from EDA
- Add only 2-3 high-value interactions we know work
- Focus on quality over quantity
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json

def load_data_and_eda_results():
    """Load data and EDA results"""
    data_dir = Path(__file__).parent.parent / 'data' / 'processed'
    exp_dir = Path(__file__).parent.parent / 'experiments'
    
    # Load data
    X_train = np.load(data_dir / 'X_train.npy')
    X_val = np.load(data_dir / 'X_val.npy')
    X_test = np.load(data_dir / 'X_test.npy')
    y_train = np.load(data_dir / 'y_train.npy')
    y_val = np.load(data_dir / 'y_val.npy')
    y_test = np.load(data_dir / 'y_test.npy')
    
    with open(data_dir / 'feature_names.json', 'r') as f:
        feature_names = json.load(f)
    
    # Load EDA significance results
    significance_df = pd.read_csv(exp_dir / 'feature_significance.csv')
    
    # Combine train and validation
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.hstack([y_train, y_val])
    
    print("Data loaded:")
    print(f"Training samples: {len(X_combined):,}")
    print(f"Test samples: {len(X_test):,}")
    print(f"Total features: {len(feature_names)}")
    
    return X_combined, X_test, y_combined, y_test, feature_names, significance_df

def select_best_features(feature_names, significance_df):
    """Select only the most important features based on EDA"""
    print("\n" + "="*50)
    print("SELECTING HIGH-VALUE FEATURES")
    print("="*50)
    
    # Get statistically significant features (p < 0.05)
    significant_features = significance_df[significance_df['significant']]['feature'].tolist()
    
    print(f"Statistically significant features: {len(significant_features)}")
    
    # Top features by effect size (Cohen's d > 0.2)
    top_features = significance_df[significance_df['effect_size'] > 0.2]['feature'].tolist()
    
    print(f"Features with strong effect size (d > 0.2): {len(top_features)}")
    
    # Combine and prioritize
    selected_features = list(set(significant_features + top_features))
    
    print(f"Total selected features: {len(selected_features)}")
    print("\nSelected features:")
    for i, feature in enumerate(selected_features, 1):
        effect_size = significance_df[significance_df['feature'] == feature]['effect_size'].iloc[0]
        p_value = significance_df[significance_df['feature'] == feature]['p_value'].iloc[0]
        print(f"{i:2d}. {feature:25s} (d={effect_size:.3f}, p={p_value:.2e})")
    
    return selected_features

def add_minimal_interactions(X, feature_names, selected_features):
    """Add only 2-3 proven valuable interactions"""
    print("\n" + "="*50)
    print("ADDING MINIMAL HIGH-VALUE INTERACTIONS")
    print("="*50)
    
    df = pd.DataFrame(X, columns=feature_names)
    
    # Only add interactions for the top 3 most significant features
    # From EDA: V3 (d=1.315), v_std (d=1.210), v_max (d=1.130)
    
    interactions_added = 0
    
    # Interaction 1: V1 * V3 (we know this is significant from EDA)
    if 'V1' in df.columns and 'V3' in df.columns:
        df['V1_V3_interaction'] = df['V1'] * df['V3']
        selected_features.append('V1_V3_interaction')
        interactions_added += 1
        print("Added V1_V3_interaction")
    
    # Interaction 2: Amount with top V feature
    if 'Amount' in df.columns and 'V3' in df.columns:
        df['Amount_V3_interaction'] = df['Amount'] * df['V3']
        selected_features.append('Amount_V3_interaction')
        interactions_added += 1
        print("Added Amount_V3_interaction")
    
    print(f"\nTotal interactions added: {interactions_added}")
    print(f"Final feature count: {len(selected_features)}")
    
    return df.values, df.columns.tolist(), selected_features

def create_final_datasets(X_train, X_test, all_features, selected_features):
    """Create final training and test datasets with selected features only"""
    print("\n" + "="*50)
    print("CREATING FINAL DATASETS")
    print("="*50)
    
    # Get indices of selected features
    feature_indices = [all_features.index(f) for f in selected_features if f in all_features]
    
    # Select features
    X_train_final = X_train[:, feature_indices]
    X_test_final = X_test[:, feature_indices]
    
    # Get final feature names (in case some interactions couldn't be created)
    final_features = [all_features[i] for i in feature_indices]
    
    print(f"Original features: {len(all_features)}")
    print(f"Selected features: {len(final_features)}")
    print(f"Reduction: {(1 - len(final_features)/len(all_features))*100:.1f}%")
    
    print(f"\nFinal feature set ({len(final_features)} features):")
    for i, feature in enumerate(final_features, 1):
        print(f"{i:2d}. {feature}")
    
    return X_train_final, X_test_final, final_features

def save_results(X_train_final, X_test_final, final_features, significance_df):
    """Save the final datasets and feature information"""
    print("\n" + "="*50)
    print("SAVING RESULTS")
    print("="*50)
    
    # Create directories
    processed_dir = Path(__file__).parent.parent / 'data' / 'processed'
    experiments_dir = Path(__file__).parent.parent / 'experiments'
    
    # Save final datasets
    np.save(processed_dir / 'X_train_final.npy', X_train_final)
    np.save(processed_dir / 'X_test_final.npy', X_test_final)
    
    # Save final feature list
    with open(processed_dir / 'final_features.json', 'w') as f:
        json.dump(final_features, f, indent=2)
    
    # Save feature selection summary
    summary = {
        'total_original_features': 41,
        'selected_features_count': len(final_features),
        'reduction_percentage': (1 - len(final_features)/41)*100,
        'selection_criteria': 'Statistical significance (p < 0.05) + Effect size (d > 0.2)',
        'interactions_added': 2,
        'final_features': final_features
    }
    
    with open(experiments_dir / 'feature_selection_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save final feature importance
    final_significance = significance_df[significance_df['feature'].isin(final_features)].copy()
    final_significance = final_significance.sort_values('effect_size', ascending=False)
    final_significance.to_csv(experiments_dir / 'final_feature_importance.csv', index=False)
    
    print(f"Final datasets saved to {processed_dir}")
    print(f"Feature analysis saved to {experiments_dir}")
    
    return summary

def main():
    """Run focused feature selection"""
    
    # Load data and EDA results
    X_train, X_test, y_train, y_test, feature_names, significance_df = load_data_and_eda_results()
    
    # Select best features based on EDA
    selected_features = select_best_features(feature_names, significance_df)
    
    # Add minimal high-value interactions
    X_train_enhanced, enhanced_features, selected_features = add_minimal_interactions(
        X_train, feature_names, selected_features.copy()
    )
    X_test_enhanced, _, _ = add_minimal_interactions(
        X_test, feature_names, selected_features.copy()
    )
    
    # Create final datasets
    X_train_final, X_test_final, final_features = create_final_datasets(
        X_train_enhanced, X_test_enhanced, enhanced_features, selected_features
    )
    
    # Save results
    summary = save_results(X_train_final, X_test_final, final_features, significance_df)
    
    print("\n" + "="*50)
    print("FEATURE SELECTION COMPLETE")
    print("="*50)
    print(f"SUCCESS: Focused on {len(final_features)} high-quality features")
    print(f"SUCCESS: {summary['reduction_percentage']:.1f}% reduction in feature space")
    print(f"SUCCESS: Ready for model training")

if __name__ == "__main__":
    main()