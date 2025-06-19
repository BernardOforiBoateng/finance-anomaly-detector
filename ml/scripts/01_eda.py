#!/usr/bin/env python3
"""
Step 1: Exploratory Data Analysis
- Load and examine the dataset
- Understand class distribution
- Analyze feature distributions
- Identify patterns and anomalies
- Generate insights for feature engineering
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the processed data"""
    data_dir = Path(__file__).parent.parent / 'data' / 'processed'
    
    X_train = np.load(data_dir / 'X_train.npy')
    X_val = np.load(data_dir / 'X_val.npy')
    X_test = np.load(data_dir / 'X_test.npy')
    y_train = np.load(data_dir / 'y_train.npy')
    y_val = np.load(data_dir / 'y_val.npy')
    y_test = np.load(data_dir / 'y_test.npy')
    
    with open(data_dir / 'feature_names.json', 'r') as f:
        feature_names = json.load(f)
    
    # Combine train and validation for EDA
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.hstack([y_train, y_val])
    
    return X_combined, X_test, y_combined, y_test, feature_names

def basic_dataset_info(X, y, feature_names):
    """Print basic dataset information"""
    print("="*60)
    print("DATASET OVERVIEW")
    print("="*60)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Number of samples: {X.shape[0]:,}")
    
    print(f"\nClass distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count:,} ({count/len(y)*100:.2f}%)")
    
    imbalance_ratio = counts[0] / counts[1]
    print(f"\nClass imbalance ratio: {imbalance_ratio:.1f}:1")
    
    return imbalance_ratio

def analyze_feature_types(feature_names):
    """Categorize features by type"""
    print("\n" + "="*60)
    print("FEATURE ANALYSIS")
    print("="*60)
    
    v_features = [f for f in feature_names if f.startswith('V')]
    amount_features = [f for f in feature_names if 'amount' in f.lower()]
    time_features = [f for f in feature_names if any(t in f.lower() for t in ['time', 'hour', 'day'])]
    interaction_features = [f for f in feature_names if 'interaction' in f.lower()]
    other_features = [f for f in feature_names if f not in v_features + amount_features + time_features + interaction_features]
    
    print(f"V features (PCA): {len(v_features)}")
    print(f"Amount features: {len(amount_features)}")
    print(f"Time features: {len(time_features)}")
    print(f"Interaction features: {len(interaction_features)}")
    print(f"Other features: {len(other_features)}")
    
    return {
        'v_features': v_features,
        'amount_features': amount_features,
        'time_features': time_features,
        'interaction_features': interaction_features,
        'other_features': other_features
    }

def analyze_class_separation(X, y, feature_names):
    """Analyze how well features separate classes"""
    print("\n" + "="*60)
    print("CLASS SEPARATION ANALYSIS")
    print("="*60)
    
    df = pd.DataFrame(X, columns=feature_names)
    df['Class'] = y
    
    # Calculate mean differences between classes
    fraud_means = df[df['Class'] == 1].drop('Class', axis=1).mean()
    normal_means = df[df['Class'] == 0].drop('Class', axis=1).mean()
    
    mean_diff = abs(fraud_means - normal_means)
    mean_diff_sorted = mean_diff.sort_values(ascending=False)
    
    print("Top 15 features with highest class separation (mean difference):")
    for i, (feature, diff) in enumerate(mean_diff_sorted.head(15).items(), 1):
        fraud_mean = fraud_means[feature]
        normal_mean = normal_means[feature]
        print(f"{i:2d}. {feature:25s}: {diff:8.4f} (Fraud: {fraud_mean:7.3f}, Normal: {normal_mean:7.3f})")
    
    return mean_diff_sorted

def statistical_analysis(X, y, feature_names):
    """Perform statistical tests for feature significance"""
    print("\n" + "="*60)
    print("STATISTICAL SIGNIFICANCE ANALYSIS")
    print("="*60)
    
    df = pd.DataFrame(X, columns=feature_names)
    
    # Perform t-tests for each feature
    fraud_data = df[y == 1]
    normal_data = df[y == 0]
    
    p_values = []
    effect_sizes = []
    
    for feature in feature_names:
        # Welch's t-test (unequal variances)
        t_stat, p_val = stats.ttest_ind(
            fraud_data[feature], 
            normal_data[feature], 
            equal_var=False
        )
        
        # Cohen's d (effect size)
        pooled_std = np.sqrt(
            ((len(fraud_data) - 1) * fraud_data[feature].var() + 
             (len(normal_data) - 1) * normal_data[feature].var()) /
            (len(fraud_data) + len(normal_data) - 2)
        )
        
        cohens_d = abs(fraud_data[feature].mean() - normal_data[feature].mean()) / pooled_std
        
        p_values.append(p_val)
        effect_sizes.append(cohens_d)
    
    # Create significance dataframe
    significance_df = pd.DataFrame({
        'feature': feature_names,
        'p_value': p_values,
        'effect_size': effect_sizes,
        'significant': np.array(p_values) < 0.05
    }).sort_values('effect_size', ascending=False)
    
    print(f"Statistically significant features (p < 0.05): {significance_df['significant'].sum()}")
    print(f"Total features: {len(feature_names)}")
    
    print("\nTop 15 features by effect size (Cohen's d):")
    for i, row in significance_df.head(15).iterrows():
        sig_marker = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
        print(f"{i+1:2d}. {row['feature']:25s}: d={row['effect_size']:6.3f} (p={row['p_value']:.2e}) {sig_marker}")
    
    return significance_df

def correlation_analysis(X, feature_names):
    """Analyze feature correlations"""
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    
    df = pd.DataFrame(X, columns=feature_names)
    
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    # Find high correlations
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.8:  # High correlation threshold
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_val
                })
    
    high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('correlation', key=abs, ascending=False)
    
    print(f"Feature pairs with |correlation| > 0.8: {len(high_corr_df)}")
    
    if len(high_corr_df) > 0:
        print("\nTop 10 highly correlated feature pairs:")
        for i, row in high_corr_df.head(10).iterrows():
            print(f"{row['feature1']:20s} - {row['feature2']:20s}: {row['correlation']:6.3f}")
    else:
        print("No highly correlated feature pairs found.")
    
    return high_corr_df, corr_matrix

def missing_values_analysis(X, feature_names):
    """Check for missing values and data quality"""
    print("\n" + "="*60)
    print("DATA QUALITY ANALYSIS")
    print("="*60)
    
    df = pd.DataFrame(X, columns=feature_names)
    
    # Missing values
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df)) * 100
    
    if missing_counts.sum() > 0:
        print("Features with missing values:")
        for feature, count in missing_counts[missing_counts > 0].items():
            print(f"  {feature}: {count} ({missing_pct[feature]:.2f}%)")
    else:
        print("No missing values found.")
    
    # Check for infinite values
    inf_counts = np.isinf(X).sum(axis=0)
    if inf_counts.sum() > 0:
        print(f"\nInfinite values found in {np.sum(inf_counts > 0)} features")
    else:
        print("No infinite values found.")
    
    # Basic statistics
    print(f"\nDataset statistics:")
    print(f"  Mean feature value: {X.mean():.4f}")
    print(f"  Std feature value: {X.std():.4f}")
    print(f"  Min feature value: {X.min():.4f}")
    print(f"  Max feature value: {X.max():.4f}")

def feature_distribution_analysis(X, y, feature_names, top_features):
    """Analyze distribution of top features"""
    print("\n" + "="*60)
    print("FEATURE DISTRIBUTION ANALYSIS")
    print("="*60)
    
    df = pd.DataFrame(X, columns=feature_names)
    
    # Analyze top 5 most important features
    top_5_features = top_features.head(5).index.tolist()
    
    for feature in top_5_features:
        fraud_values = df[y == 1][feature]
        normal_values = df[y == 0][feature]
        
        print(f"\n{feature}:")
        print(f"  Normal - Mean: {normal_values.mean():8.3f}, Std: {normal_values.std():8.3f}")
        print(f"  Fraud  - Mean: {fraud_values.mean():8.3f}, Std: {fraud_values.std():8.3f}")
        
        # Skewness and kurtosis
        print(f"  Normal - Skew: {stats.skew(normal_values):7.3f}, Kurt: {stats.kurtosis(normal_values):7.3f}")
        print(f"  Fraud  - Skew: {stats.skew(fraud_values):7.3f}, Kurt: {stats.kurtosis(fraud_values):7.3f}")

def generate_insights(imbalance_ratio, feature_types, significance_df, high_corr_df):
    """Generate insights for next steps"""
    print("\n" + "="*60)
    print("INSIGHTS & RECOMMENDATIONS")
    print("="*60)
    
    insights = []
    
    # Class imbalance
    if imbalance_ratio > 100:
        insights.append(f"WARNING: Severe class imbalance ({imbalance_ratio:.0f}:1) - Use balanced sampling, cost-sensitive learning")
    elif imbalance_ratio > 10:
        insights.append(f"WARNING: Moderate class imbalance ({imbalance_ratio:.0f}:1) - Consider class weights")
    
    # Feature significance
    sig_features = significance_df['significant'].sum()
    total_features = len(significance_df)
    sig_pct = (sig_features / total_features) * 100
    
    if sig_pct > 80:
        insights.append(f"GOOD: High feature quality: {sig_pct:.0f}% features are statistically significant")
    elif sig_pct > 50:
        insights.append(f"WARNING: Moderate feature quality: {sig_pct:.0f}% features are statistically significant")
    else:
        insights.append(f"ISSUE: Low feature quality: Only {sig_pct:.0f}% features are statistically significant")
    
    # Correlation issues
    if len(high_corr_df) > 20:
        insights.append(f"WARNING: High multicollinearity: {len(high_corr_df)} highly correlated pairs - Consider PCA or feature selection")
    elif len(high_corr_df) > 5:
        insights.append(f"WARNING: Some multicollinearity: {len(high_corr_df)} highly correlated pairs")
    else:
        insights.append("GOOD: Low multicollinearity detected")
    
    # V features analysis
    v_feature_count = len(feature_types['v_features'])
    if v_feature_count > 20:
        insights.append(f"ANALYSIS: {v_feature_count} PCA features available - Good for dimensionality")
    
    print("Key Insights:")
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")
    
    print("\nRecommendations for next steps:")
    print("1. Feature Engineering: Create interaction terms for top significant features")
    print("2. Feature Selection: Use top significant features with high effect sizes")
    print("3. Sampling Strategy: Address class imbalance with SMOTE or class weights")
    print("4. Model Selection: Try ensemble methods (RF, XGBoost) for imbalanced data")
    print("5. Evaluation: Use ROC-AUC, PR-AUC, and F1-score as primary metrics")

def save_eda_results(imbalance_ratio, feature_types, significance_df, high_corr_df):
    """Save EDA results for future reference"""
    results_dir = Path(__file__).parent.parent / 'experiments'
    results_dir.mkdir(exist_ok=True)
    
    # Save significance analysis
    significance_df.to_csv(results_dir / 'feature_significance.csv', index=False)
    
    # Save correlation analysis
    if len(high_corr_df) > 0:
        high_corr_df.to_csv(results_dir / 'high_correlations.csv', index=False)
    
    # Save EDA summary
    eda_summary = {
        'imbalance_ratio': float(imbalance_ratio),
        'significant_features_count': int(significance_df['significant'].sum()),
        'total_features': len(significance_df),
        'high_correlation_pairs': len(high_corr_df),
        'feature_types': {k: len(v) for k, v in feature_types.items()},
        'top_features_by_effect_size': significance_df.head(20)['feature'].tolist()
    }
    
    with open(results_dir / 'eda_summary.json', 'w') as f:
        json.dump(eda_summary, f, indent=2)
    
    print(f"\nEDA results saved to {results_dir}")

def main():
    """Run comprehensive EDA"""
    
    # Load data
    X, X_test, y, y_test, feature_names = load_data()
    
    # Basic dataset info
    imbalance_ratio = basic_dataset_info(X, y, feature_names)
    
    # Feature type analysis
    feature_types = analyze_feature_types(feature_names)
    
    # Class separation analysis
    class_separation = analyze_class_separation(X, y, feature_names)
    
    # Statistical significance analysis
    significance_df = statistical_analysis(X, y, feature_names)
    
    # Correlation analysis
    high_corr_df, corr_matrix = correlation_analysis(X, feature_names)
    
    # Data quality check
    missing_values_analysis(X, feature_names)
    
    # Feature distribution analysis
    feature_distribution_analysis(X, y, feature_names, class_separation)
    
    # Generate insights
    generate_insights(imbalance_ratio, feature_types, significance_df, high_corr_df)
    
    # Save results
    save_eda_results(imbalance_ratio, feature_types, significance_df, high_corr_df)
    
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()