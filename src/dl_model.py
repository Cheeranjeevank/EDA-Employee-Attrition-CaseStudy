"""
Deep Learning Model Architecture, Training, and Hyperparameter Tuning Module
Author: Cheeranjeevan K
Course: U21ADP05 - Exploratory Data Analysis and Visualization

Deep Learning Model: Multi-Layer Perceptron (MLP) for Tabular Attrition Prediction
Features:
- Modular PyTorch MLP with Batch Normalization, Dropout regularization, and LeakyReLU activations.
- Class-weighted CrossEntropyLoss to mitigate severe class imbalance (84% No / 16% Yes).
- Hyperparameter tuning across learning rates, hidden dimensions, and regularization strengths.
- Model checkpointing based on Validation ROC-AUC.
- Permutation Feature Importance module for model explainability.
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score

class EmployeeAttritionMLP(nn.Module):
    def __init__(self, in_features: int, hidden_dims: list = [128, 64, 32], dropout_rates: list = [0.3, 0.2, 0.1]):
        super(EmployeeAttritionMLP, self).__init__()
        
        layers = []
        current_dim = in_features
        
        # Build hidden layers with BatchNorm, Activation, and Dropout
        for i, (h_dim, p_drop) in enumerate(zip(hidden_dims, dropout_rates)):
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(p_drop))
            current_dim = h_dim
            
        # Final classification output layer (2 logits for CrossEntropyLoss)
        layers.append(nn.Linear(current_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []
    
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * len(y_batch)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(y_batch.cpu().numpy())
        
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    return epoch_loss, epoch_acc

def evaluate_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_probs, all_targets = [], [], []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            
            running_loss += loss.item() * len(y_batch)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(y_batch.cpu().numpy())
            
    val_loss = running_loss / len(loader.dataset)
    val_acc = accuracy_score(all_targets, all_preds)
    try:
        val_auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        val_auc = 0.5
        
    return val_loss, val_acc, val_auc, np.array(all_probs), np.array(all_preds), np.array(all_targets)

def train_mlp_model(
    train_loader, val_loader, in_features: int, class_weights: torch.Tensor,
    hidden_dims: list = [128, 64, 32], dropout_rates: list = [0.3, 0.2, 0.1],
    lr: float = 0.001, weight_decay: float = 1e-4, epochs: int = 50,
    save_path: str = None
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EmployeeAttritionMLP(in_features, hidden_dims, dropout_rates).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'val_auc': []
    }
    
    best_val_auc = -1.0
    best_model_weights = None
    
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc, v_auc, _, _, _ = evaluate_model(model, val_loader, criterion, device)
        scheduler.step()
        
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(v_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(v_acc)
        history['val_auc'].append(v_auc)
        
        if v_auc > best_val_auc:
            best_val_auc = v_auc
            best_model_weights = copy.deepcopy(model.state_dict())
            
        if epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {tr_loss:.4f}, Val Loss: {v_loss:.4f} | Train Acc: {tr_acc:.3f}, Val Acc: {v_acc:.3f}, Val AUC: {v_auc:.4f}")
            
    # Load best weights
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'state_dict': model.state_dict(),
            'in_features': in_features,
            'hidden_dims': hidden_dims,
            'dropout_rates': dropout_rates,
            'best_val_auc': best_val_auc
        }, save_path)
        print(f"[Model Checkpoint] Saved best model (Val AUC: {best_val_auc:.4f}) to {save_path}")
        
    return model, history

def tune_hyperparameters(train_loader, val_loader, in_features: int, class_weights: torch.Tensor):
    """
    Evaluates 3 diverse configurations to identify the optimal learning rate, 
    capacity, and regularization strength.
    """
    configs = [
        {'name': 'Config 1 (Standard LR, Medium Capacity)', 'lr': 0.005, 'hidden': [128, 64, 32], 'dropout': [0.2, 0.2, 0.1], 'wd': 1e-4},
        {'name': 'Config 2 (Low LR, High Regularization)', 'lr': 0.001, 'hidden': [128, 64, 32], 'dropout': [0.3, 0.2, 0.1], 'wd': 1e-3},
        {'name': 'Config 3 (Deep Capacity, High Dropout)', 'lr': 0.001, 'hidden': [256, 128, 64], 'dropout': [0.4, 0.3, 0.2], 'wd': 1e-4}
    ]
    
    print("\n" + "="*70)
    print("STEP 4: HYPERPARAMETER TUNING EXPERIMENT")
    print("="*70)
    
    results = []
    best_config = None
    highest_auc = -1.0
    
    for cfg in configs:
        print(f"\nEvaluating: {cfg['name']} (lr={cfg['lr']}, hidden={cfg['hidden']}, wd={cfg['wd']})")
        model, hist = train_mlp_model(
            train_loader, val_loader, in_features, class_weights,
            hidden_dims=cfg['hidden'], dropout_rates=cfg['dropout'],
            lr=cfg['lr'], weight_decay=cfg['wd'], epochs=40
        )
        max_auc = max(hist['val_auc'])
        min_loss = min(hist['val_loss'])
        results.append({
            'config_name': cfg['name'],
            'best_val_auc': max_auc,
            'min_val_loss': min_loss,
            'config': cfg
        })
        print(f"--> Result: Peak Val AUC: {max_auc:.4f}, Min Val Loss: {min_loss:.4f}")
        
        if max_auc > highest_auc:
            highest_auc = max_auc
            best_config = cfg
            
    print(f"\n[Hyperparameter Tuning] Selected Best Configuration: {best_config['name']} with Val AUC: {highest_auc:.4f}")
    return best_config, results

def compute_permutation_importance(model, X_test: np.ndarray, y_test: np.ndarray, feature_names: list, n_repeats: int = 5):
    """
    Computes permutation feature importance for neural network explainability.
    Measures mean drop in ROC-AUC score when each feature column is shuffled.
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Baseline test AUC
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        base_logits = model(X_t)
        base_probs = torch.softmax(base_logits, dim=1)[:, 1].cpu().numpy()
        base_auc = roc_auc_score(y_test, base_probs)
        
    importances = {}
    for i, col_name in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            X_shuffled = X_test.copy()
            np.random.shuffle(X_shuffled[:, i])
            with torch.no_grad():
                X_shuf_t = torch.tensor(X_shuffled, dtype=torch.float32).to(device)
                shuf_logits = model(X_shuf_t)
                shuf_probs = torch.softmax(shuf_logits, dim=1)[:, 1].cpu().numpy()
                shuf_auc = roc_auc_score(y_test, shuf_probs)
                drops.append(base_auc - shuf_auc)
        importances[col_name] = float(np.mean(drops))
        
    sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    return sorted_importances
