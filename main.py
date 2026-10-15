"""
Main Orchestration Script for EDA & Deep Learning Case Study
Author: Cheeranjeevan K
Department of Artificial Intelligence and Data Science
KPR Institute of Engineering and Technology
Course: U21ADP05 - Exploratory Data Analysis and Visualization
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch

from src.eda_visualization import (
    stage_1_acquire, stage_2_parse, stage_3_filter, stage_4_mine,
    stage_5_and_6_represent_and_refine, stage_7_interact
)
from src.preprocessing import PreprocessingPipeline
from src.dl_model import (
    train_mlp_model, tune_hyperparameters, compute_permutation_importance,
    evaluate_model
)
from src.evaluate import evaluate_and_generate_all

def main():
    print("=" * 80)
    print("EXPLORATORY DATA ANALYSIS & DEEP LEARNING CASE STUDY PIPELINE")
    print("IBM HR Analytics Employee Attrition & Performance Case Study")
    print("Course: U21ADP05 | Academic Year: 2026-2027")
    print("=" * 80)
    
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'employee_attrition.csv')
    figures_dir = os.path.join(base_dir, 'figures')
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # -------------------------------------------------------------
    # 1. Ben Fry's 7 Stages of Data Visualization
    # -------------------------------------------------------------
    print("\n>>> STEP 1 & 2 & 3: BEN FRY'S 7 STAGES OF DATA VISUALIZATION")
    df_raw = stage_1_acquire(data_path)
    schema_info = stage_2_parse(df_raw)
    df_filtered = stage_3_filter(df_raw)
    mined_stats = stage_4_mine(df_filtered)
    
    eda_figures = stage_5_and_6_represent_and_refine(df_filtered, figures_dir)
    stage_7_interact(df_filtered, figures_dir)
    print(">>> Finished EDA Visualizations successfully.")
    
    # -------------------------------------------------------------
    # 2. Preprocessing & Data Leakage Prevention
    # -------------------------------------------------------------
    print("\n>>> STEP 2 (Cont.): PREPROCESSING & STRATIFIED SPLITTING")
    pipeline = PreprocessingPipeline(test_size=0.15, val_size=0.15, random_state=42)
    processed = pipeline.fit_transform_split(df_raw)
    train_loader, val_loader, test_loader = pipeline.get_dataloaders(processed, batch_size=32)
    
    # -------------------------------------------------------------
    # 3. Model Building & Hyperparameter Tuning
    # -------------------------------------------------------------
    print("\n>>> STEP 4: DEEP LEARNING MODEL BUILDING & TUNING")
    in_features = processed['X_train'].shape[1]
    best_cfg, tuning_results = tune_hyperparameters(
        train_loader, val_loader, in_features, processed['class_weights']
    )
    
    # Train production model with best configuration
    print("\n>>> Training Final Model with Best Hyperparameters...")
    model_save_path = os.path.join(models_dir, 'employee_attrition_mlp.pt')
    final_model, history = train_mlp_model(
        train_loader, val_loader, in_features, processed['class_weights'],
        hidden_dims=best_cfg['hidden'], dropout_rates=best_cfg['dropout'],
        lr=best_cfg['lr'], weight_decay=best_cfg['wd'], epochs=50,
        save_path=model_save_path
    )
    
    # -------------------------------------------------------------
    # 4. Model Evaluation Visualizations & Explainability
    # -------------------------------------------------------------
    print("\n>>> STEP 5: MODEL EVALUATION VISUALIZATIONS & FEATURE IMPORTANCE")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.CrossEntropyLoss(weight=processed['class_weights'].to(device))
    
    # Evaluate on held-out test set
    test_loss, test_acc, test_auc, y_probs, y_preds, y_true = evaluate_model(
        final_model, test_loader, criterion, device
    )
    
    print(f"\n[Test Set Results] Loss: {test_loss:.4f} | Accuracy: {test_acc:.3f} | ROC-AUC: {test_auc:.4f}")
    
    # Compute Permutation Feature Importance
    print(">>> Computing Permutation Feature Importance (5 repetitions per feature)...")
    importances = compute_permutation_importance(
        final_model, processed['X_test'], processed['y_test'],
        processed['feature_names'], n_repeats=5
    )
    
    eval_metrics = evaluate_and_generate_all(
        history, y_true, y_preds, y_probs, importances, figures_dir
    )
    
    # Save metrics summary to JSON
    summary_path = os.path.join(base_dir, 'report', 'evaluation_metrics.json')
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(eval_metrics, f, indent=4)
        
    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION COMPLETE!")
    print(f"- Processed records: {len(df_raw)}")
    print(f"- Total features: {df_raw.shape[1]} -> Preprocessed: {in_features}")
    print(f"- Test Accuracy: {eval_metrics['accuracy']:.4f}")
    print(f"- Test Balanced Accuracy: {eval_metrics['balanced_accuracy']:.4f}")
    print(f"- Test ROC-AUC Score: {eval_metrics['roc_auc']:.4f}")
    print(f"- Test Average Precision (PR-AUC): {eval_metrics['average_precision']:.4f}")
    print(f"- Saved figures: {figures_dir}")
    print(f"- Saved model: {model_save_path}")
    print("=" * 80)

if __name__ == '__main__':
    main()
