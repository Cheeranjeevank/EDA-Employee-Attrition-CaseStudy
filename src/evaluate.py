"""
Model Evaluation and Result Visualization Module
Author: Cheeranjeevan K
Course: U21ADP05 - Exploratory Data Analysis and Visualization

Generates comprehensive evaluation visualizations:
1. Training Loss vs Validation Loss over Epochs
2. Training Accuracy vs Validation Accuracy over Epochs
3. Confusion Matrix (Raw counts and Normalized percentages)
4. Receiver Operating Characteristic (ROC) Curve with AUC Score
5. Precision-Recall (PR) Curve with Average Precision (AP)
6. Neural Network Permutation Feature Importance Ranking
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score, classification_report, accuracy_score,
    balanced_accuracy_score, f1_score
)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

def plot_training_curves(history: dict, output_dir: str):
    """Plots and saves Loss vs Epoch and Accuracy vs Epoch charts."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1. Loss vs Epoch
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.plot(epochs, history['train_loss'], label='Training Loss', color='#2b5c8f', linewidth=2.2)
    ax.plot(epochs, history['val_loss'], label='Validation Loss', color='#d9534f', linewidth=2.2, linestyle='--')
    ax.set_title("Model Evaluation 1: Cross-Entropy Loss vs. Epochs", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    loss_path = os.path.join(output_dir, "eval_01_loss_epochs.png")
    fig.savefig(loss_path, dpi=300)
    plt.close(fig)
    print(f"[Evaluation] Saved loss curve to {loss_path}")
    
    # 2. Accuracy vs Epoch
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.plot(epochs, history['train_acc'], label='Training Accuracy', color='#2b5c8f', linewidth=2.2)
    ax.plot(epochs, history['val_acc'], label='Validation Accuracy', color='#2ca02c', linewidth=2.2, linestyle='--')
    ax.set_title("Model Evaluation 2: Classification Accuracy vs. Epochs", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(0.5, 1.0)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    acc_path = os.path.join(output_dir, "eval_02_accuracy_epochs.png")
    fig.savefig(acc_path, dpi=300)
    plt.close(fig)
    print(f"[Evaluation] Saved accuracy curve to {acc_path}")
    
    return loss_path, acc_path

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_dir: str):
    """Plots dual raw count and normalized confusion matrices."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    labels = ['Retained (No)', 'Attrition (Yes)']
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False,
                xticklabels=labels, yticklabels=labels, annot_kws={"size": 13, "weight": "bold"})
    axes[0].set_title("Confusion Matrix (Absolute Counts)", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Predicted Label", fontsize=11)
    axes[0].set_ylabel("Actual Ground Truth", fontsize=11)
    
    # Normalized percentages
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', ax=axes[1], cbar=True,
                xticklabels=labels, yticklabels=labels, annot_kws={"size": 13, "weight": "bold"})
    axes[1].set_title("Normalized Confusion Matrix (Recall per Class)", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Predicted Label", fontsize=11)
    axes[1].set_ylabel("Actual Ground Truth", fontsize=11)
    
    fig.suptitle("Model Evaluation 3: Confusion Matrix Analysis on Test Set", fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "eval_03_confusion_matrix.png")
    fig.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[Evaluation] Saved confusion matrix to {cm_path}")
    return cm_path

def plot_roc_curve(y_true: np.ndarray, y_probs: np.ndarray, output_dir: str):
    """Plots ROC Curve with AUC metric."""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.plot(fpr, tpr, color='#2b5c8f', linewidth=2.5, label=f'Deep MLP (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='#888888', linestyle='--', linewidth=1.5, label='Random Chance Baseline (AUC = 0.500)')
    
    ax.fill_between(fpr, tpr, color='#2b5c8f', alpha=0.1)
    ax.set_title("Model Evaluation 4: Receiver Operating Characteristic (ROC) Curve", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower right", frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    roc_path = os.path.join(output_dir, "eval_04_roc_auc_curve.png")
    fig.savefig(roc_path, dpi=300)
    plt.close(fig)
    print(f"[Evaluation] Saved ROC curve to {roc_path}")
    return roc_path, roc_auc

def plot_precision_recall_curve(y_true: np.ndarray, y_probs: np.ndarray, output_dir: str):
    """Plots Precision-Recall curve with average precision score."""
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    ap_score = average_precision_score(y_true, y_probs)
    pos_rate = np.mean(y_true)
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.plot(recall, precision, color='#d9534f', linewidth=2.5, label=f'Deep MLP (AP = {ap_score:.3f})')
    ax.axhline(pos_rate, color='#888888', linestyle='--', linewidth=1.5, label=f'Imbalance Baseline ({pos_rate:.1%})')
    
    ax.fill_between(recall, precision, color='#d9534f', alpha=0.1)
    ax.set_title("Model Evaluation 5: Precision-Recall Curve (Cost-Sensitive Evaluation)", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Recall (True Positive Rate)", fontsize=11)
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=11)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="upper right", frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    pr_path = os.path.join(output_dir, "eval_05_pr_curve.png")
    fig.savefig(pr_path, dpi=300)
    plt.close(fig)
    print(f"[Evaluation] Saved PR curve to {pr_path}")
    return pr_path, ap_score

def plot_feature_importance(importance_tuples: list, output_dir: str, top_n: int = 12):
    """Plots top N permutation feature importance drivers."""
    top_items = importance_tuples[:top_n]
    names = [x[0] for x in reversed(top_items)]
    scores = [x[1] for x in reversed(top_items)]
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    bars = ax.barh(names, scores, color='#2b5c8f', edgecolor='#1b3b5f', height=0.6)
    
    for bar in bars:
        width = bar.get_width()
        ax.annotate(f"{width:+.4f}",
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5 if width >= 0 else -35, 0), textcoords="offset points",
                    ha='left' if width >= 0 else 'right', va='center', fontsize=9, fontweight='bold')
        
    ax.set_title("Model Evaluation 6: Top Permutation Feature Importance Drivers (Neural MLP)", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Mean Drop in ROC-AUC Metric Upon Feature Shuffling", fontsize=11)
    ax.set_ylabel("Feature Name", fontsize=11)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    fi_path = os.path.join(output_dir, "eval_06_feature_importance.png")
    fig.savefig(fi_path, dpi=300)
    plt.close(fig)
    print(f"[Evaluation] Saved feature importance plot to {fi_path}")
    return fi_path

def evaluate_and_generate_all(history: dict, y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray,
                              importance_tuples: list, output_dir: str):
    """Generates all model evaluation figures and computes complete metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    loss_p, acc_p = plot_training_curves(history, output_dir)
    cm_p = plot_confusion_matrix(y_true, y_pred, output_dir)
    roc_p, roc_auc = plot_roc_curve(y_true, y_probs, output_dir)
    pr_p, ap = plot_precision_recall_curve(y_true, y_probs, output_dir)
    fi_p = plot_feature_importance(importance_tuples, output_dir)
    
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=['Retained (No)', 'Attrition (Yes)'], output_dict=True)
    
    metrics = {
        'accuracy': float(acc),
        'balanced_accuracy': float(bal_acc),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'average_precision': float(ap),
        'classification_report': report,
        'figures': {
            'loss_curve': loss_p,
            'accuracy_curve': acc_p,
            'confusion_matrix': cm_p,
            'roc_curve': roc_p,
            'pr_curve': pr_p,
            'feature_importance': fi_p
        }
    }
    return metrics
