#!/usr/bin/env python3
"""
Data preprocessing pipeline for financial anomaly detection
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import joblib
import json
from datetime import datetime

class FinancialDataPreprocessor:
    """Preprocess financial transaction data for anomaly detection"""
    
    def __init__(self, scaler_type='robust'):
        """
        Initialize preprocessor
        
        Args:
            scaler_type: 'standard' or 'robust' (robust is better for outliers)
        """
        self.scaler_type = scaler_type
        self.scaler = RobustScaler() if scaler_type == 'robust' else StandardScaler()
        self.feature_columns = None
        self.preprocessing_stats = {}
        
    def load_data(self, file_path):
        """Load transaction data from CSV"""
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df):,} transactions")
        return df
    
    def create_features(self, df):
        """Create additional features for better anomaly detection"""
        df = df.copy()
        
        # Time-based features
        df['hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24) if 'Hour' in df else 0
        df['hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24) if 'Hour' in df else 0
        
        # Amount features
        if 'Amount' in df:
            df['amount_log'] = np.log1p(df['Amount'])
            
            # Rolling statistics (if we had user ID, would group by user)
            df['amount_rolling_mean'] = df['Amount'].rolling(window=10, min_periods=1).mean()
            df['amount_rolling_std'] = df['Amount'].rolling(window=10, min_periods=1).std().fillna(0)
            
        # V1-V28 feature interactions (top PCA components often interact)
        if 'V1' in df and 'V2' in df:
            df['V1_V2_interaction'] = df['V1'] * df['V2']
            df['V1_V3_interaction'] = df['V1'] * df['V3'] if 'V3' in df else 0
            
        # Statistical features
        v_cols = [col for col in df.columns if col.startswith('V')]
        if v_cols:
            df['v_mean'] = df[v_cols].mean(axis=1)
            df['v_std'] = df[v_cols].std(axis=1)
            df['v_max'] = df[v_cols].max(axis=1)
            df['v_min'] = df[v_cols].min(axis=1)
            
        return df
    
    def remove_outliers(self, df, contamination=0.001):
        """Remove extreme outliers (optional step)"""
        from sklearn.ensemble import IsolationForest
        
        # Only for training data
        if 'Class' in df:
            normal_data = df[df['Class'] == 0].copy()
            
            # Fit isolation forest on normal transactions
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            features = [col for col in normal_data.columns if col not in ['Class', 'Time']]
            
            predictions = iso_forest.fit_predict(normal_data[features])
            normal_data = normal_data[predictions == 1]
            
            # Combine back with fraud data
            fraud_data = df[df['Class'] == 1]
            df = pd.concat([normal_data, fraud_data], ignore_index=True)
            
            print(f"Removed {len(predictions) - sum(predictions == 1)} outliers")
            
        return df
    
    def split_data(self, df, test_size=0.2, val_size=0.1):
        """Split data into train, validation, and test sets"""
        
        # Separate features and target
        target_col = 'Class'
        feature_cols = [col for col in df.columns if col != target_col]
        
        X = df[feature_cols]
        y = df[target_col]
        
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Second split: train and val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )
        
        print(f"\nData Split:")
        print(f"Training: {len(X_train):,} samples ({y_train.sum()} frauds)")
        print(f"Validation: {len(X_val):,} samples ({y_val.sum()} frauds)")
        print(f"Test: {len(X_test):,} samples ({y_test.sum()} frauds)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def fit_transform(self, X_train):
        """Fit scaler and transform training data"""
        self.feature_columns = X_train.columns.tolist()
        
        # Don't scale time-based features
        scale_columns = [col for col in self.feature_columns 
                        if col not in ['Time', 'Hour', 'Day']]
        
        X_scaled = X_train.copy()
        X_scaled[scale_columns] = self.scaler.fit_transform(X_train[scale_columns])
        
        # Store preprocessing statistics
        self.preprocessing_stats = {
            'scaled_columns': scale_columns,
            'feature_columns': self.feature_columns,
            'scaler_type': self.scaler_type,
            'n_features': len(self.feature_columns)
        }
        
        return X_scaled
    
    def transform(self, X):
        """Transform new data using fitted scaler"""
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_transform first.")
            
        scale_columns = self.preprocessing_stats['scaled_columns']
        
        X_scaled = X.copy()
        X_scaled[scale_columns] = self.scaler.transform(X[scale_columns])
        
        return X_scaled
    
    def save_preprocessor(self, path):
        """Save preprocessor and scaler"""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save scaler
        joblib.dump(self.scaler, save_dir / 'scaler.pkl')
        
        # Save preprocessing stats
        with open(save_dir / 'preprocessing_stats.json', 'w') as f:
            json.dump(self.preprocessing_stats, f, indent=2)
            
        print(f"Preprocessor saved to {save_dir}")
    
    def load_preprocessor(self, path):
        """Load preprocessor and scaler"""
        load_dir = Path(path)
        
        # Load scaler
        self.scaler = joblib.load(load_dir / 'scaler.pkl')
        
        # Load preprocessing stats
        with open(load_dir / 'preprocessing_stats.json', 'r') as f:
            self.preprocessing_stats = json.load(f)
            
        self.feature_columns = self.preprocessing_stats['feature_columns']
        
        print(f"Preprocessor loaded from {load_dir}")


def main():
    """Run preprocessing pipeline"""
    
    # Initialize preprocessor
    preprocessor = FinancialDataPreprocessor(scaler_type='robust')
    
    # Load data
    data_path = Path(__file__).parent.parent / 'data' / 'raw' / 'creditcard_2023.csv'
    df = preprocessor.load_data(data_path)
    
    # Create features
    print("\nCreating enhanced features...")
    df = preprocessor.create_features(df)
    
    # Optional: Remove extreme outliers
    # df = preprocessor.remove_outliers(df, contamination=0.001)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(df)
    
    # Scale features
    print("\nScaling features...")
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_val_scaled = preprocessor.transform(X_val)
    X_test_scaled = preprocessor.transform(X_test)
    
    # Save processed data
    processed_dir = Path(__file__).parent.parent / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as numpy arrays for faster loading
    np.save(processed_dir / 'X_train.npy', X_train_scaled)
    np.save(processed_dir / 'X_val.npy', X_val_scaled)
    np.save(processed_dir / 'X_test.npy', X_test_scaled)
    np.save(processed_dir / 'y_train.npy', y_train)
    np.save(processed_dir / 'y_val.npy', y_val)
    np.save(processed_dir / 'y_test.npy', y_test)
    
    # Save feature names
    with open(processed_dir / 'feature_names.json', 'w') as f:
        json.dump(preprocessor.feature_columns, f)
    
    # Save preprocessor
    preprocessor.save_preprocessor(processed_dir / 'preprocessor')
    
    print("\nPreprocessing complete!")
    print(f"Processed data saved to {processed_dir}")
    
    # Print class distribution
    print(f"\nClass Distribution:")
    print(f"Training - Normal: {(y_train == 0).sum():,}, Fraud: {(y_train == 1).sum():,}")
    print(f"Validation - Normal: {(y_val == 0).sum():,}, Fraud: {(y_val == 1).sum():,}")
    print(f"Test - Normal: {(y_test == 0).sum():,}, Fraud: {(y_test == 1).sum():,}")


if __name__ == "__main__":
    main()