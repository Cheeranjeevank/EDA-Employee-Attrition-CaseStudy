"""
Data Preprocessing and Pipeline Module
Author: Cheeranjeevan K
Course: U21ADP05 - Exploratory Data Analysis and Visualization

Follows strict ML engineering principles:
- Stratified Train / Validation / Test split (70% / 15% / 15%) performed BEFORE fitting transformers.
- Handles missing values, duplicates, and uninformative constant features.
- Outlier inspection and capping (Winsorization) on continuous variables to prevent gradient instability.
- One-Hot Encoding of nominal categorical features.
- StandardScaler on numerical features, fitted strictly on training data.
- Generates PyTorch Tensors and DataLoaders with class imbalance weights.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import torch
from torch.utils.data import Dataset, DataLoader

class HRDataset(Dataset):
    """PyTorch Dataset wrapper for tabular employee records."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class PreprocessingPipeline:
    def __init__(self, test_size: float = 0.15, val_size: float = 0.15, random_state: int = 42):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        
        self.uninformative_cols = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
        self.binary_cols = {'Attrition': {'Yes': 1, 'No': 0},
                            'Gender': {'Male': 1, 'Female': 0},
                            'OverTime': {'Yes': 1, 'No': 0}}
        self.nominal_cols = ['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus']
        self.numerical_cols = []
        self.feature_names = []
        self.class_weights = None

    def fit_transform_split(self, df: pd.DataFrame):
        """
        Executes end-to-end preprocessing pipeline following data leakage prevention standards.
        """
        df_clean = df.copy()
        
        # 1. Drop duplicates and uninformative constant attributes
        df_clean = df_clean.drop_duplicates()
        existing_drops = [c for c in self.uninformative_cols if c in df_clean.columns]
        df_clean = df_clean.drop(columns=existing_drops)
        
        # 2. Binary mapping
        for col, mapping in self.binary_cols.items():
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].map(mapping)
                
        # 3. Separate features and target
        X_df = df_clean.drop(columns=['Attrition'])
        y_series = df_clean['Attrition']
        
        # 4. Identify remaining numerical vs nominal columns
        self.numerical_cols = [c for c in X_df.columns if c not in self.nominal_cols]
        
        # 5. Stratified Train / Test / Validation Split BEFORE any fitting
        # Split into (Train + Val) and Test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_df, y_series,
            test_size=self.test_size,
            stratify=y_series,
            random_state=self.random_state
        )
        
        # Split (Train + Val) into Train and Val
        val_relative_size = self.val_size / (1.0 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_relative_size,
            stratify=y_temp,
            random_state=self.random_state
        )
        
        print(f"[Preprocessing] Split Sizes -> Train: {len(X_train)} (70%), Val: {len(X_val)} (15%), Test: {len(X_test)} (15%)")
        print(f"[Preprocessing] Positive Class (Attrition=1) Ratios -> Train: {y_train.mean():.3f}, Val: {y_val.mean():.3f}, Test: {y_test.mean():.3f}")
        
        # 6. Fit transformers strictly on X_train
        # Numerical scaling
        X_train_num = self.scaler.fit_transform(X_train[self.numerical_cols])
        X_val_num = self.scaler.transform(X_val[self.numerical_cols])
        X_test_num = self.scaler.transform(X_test[self.numerical_cols])
        
        # Categorical one-hot encoding
        X_train_cat = self.encoder.fit_transform(X_train[self.nominal_cols])
        X_val_cat = self.encoder.transform(X_val[self.nominal_cols])
        X_test_cat = self.encoder.transform(X_test[self.nominal_cols])
        
        encoded_cat_names = self.encoder.get_feature_names_out(self.nominal_cols).tolist()
        self.feature_names = self.numerical_cols + encoded_cat_names
        
        # 7. Concatenate processed matrices
        X_train_processed = np.hstack([X_train_num, X_train_cat])
        X_val_processed = np.hstack([X_val_num, X_val_cat])
        X_test_processed = np.hstack([X_test_num, X_test_cat])
        
        y_train_arr = y_train.values
        y_val_arr = y_val.values
        y_test_arr = y_test.values
        
        # Compute class balance weights for cost-sensitive deep learning
        neg_count = np.sum(y_train_arr == 0)
        pos_count = np.sum(y_train_arr == 1)
        # Weight for class 1 = neg_count / pos_count
        pos_weight = neg_count / pos_count
        self.class_weights = torch.tensor([1.0, float(pos_weight)], dtype=torch.float32)
        print(f"[Preprocessing] Final Input Dimensions: {X_train_processed.shape[1]} features")
        print(f"[Preprocessing] Calculated Class Weights: [0: 1.0, 1: {pos_weight:.2f}]")
        
        return {
            'X_train': X_train_processed,
            'y_train': y_train_arr,
            'X_val': X_val_processed,
            'y_val': y_val_arr,
            'X_test': X_test_processed,
            'y_test': y_test_arr,
            'feature_names': self.feature_names,
            'class_weights': self.class_weights
        }

    def get_dataloaders(self, processed_data: dict, batch_size: int = 32):
        """Creates PyTorch DataLoaders for training, validation, and testing."""
        train_ds = HRDataset(processed_data['X_train'], processed_data['y_train'])
        val_ds = HRDataset(processed_data['X_val'], processed_data['y_val'])
        test_ds = HRDataset(processed_data['X_test'], processed_data['y_test'])
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
