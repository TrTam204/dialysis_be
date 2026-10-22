"""
Training Pipeline for Hemodialysis Intradialytic Hypotension (IDH) LSTM Model.
Calculates Accuracy, Precision, Recall, F1-score, and Loss across Train/Validation/Test sets,
and saves model weights and metadata into artifacts/ directory.
"""
from typing import Dict, Any
import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from idh_ai_service.dataset_generator import prepare_dataset_splits
from idh_ai_service.lstm_model import IDHLSTMNet


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_dataset(
    model: nn.Module,
    criterion: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 64,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Calculate Accuracy, Precision, Recall, F1, Loss, and ROC-AUC on a dataset split."""
    model.eval()
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(1))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_logits = []
    all_targets = []
    total_loss = 0.0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    all_logits = np.vstack(all_logits)
    all_targets = np.vstack(all_targets).flatten()
    probabilities = 1.0 / (1.0 + np.exp(-all_logits)).flatten()
    predictions = (probabilities >= threshold).astype(int)

    mean_loss = total_loss / len(X)
    acc = accuracy_score(all_targets, predictions)
    prec = precision_score(all_targets, predictions, zero_division=0)
    rec = recall_score(all_targets, predictions, zero_division=0)
    f1 = f1_score(all_targets, predictions, zero_division=0)
    try:
        auc = roc_auc_score(all_targets, probabilities)
    except Exception:
        auc = 0.5

    cm = confusion_matrix(all_targets, predictions).tolist()

    return {
        'loss': round(float(mean_loss), 4),
        'accuracy': round(float(acc), 4),
        'precision': round(float(prec), 4),
        'recall': round(float(rec), 4),
        'f1_score': round(float(f1), 4),
        'roc_auc': round(float(auc), 4),
        'confusion_matrix': cm,
        'sample_count': int(len(X)),
    }


def train_idh_lstm(
    n_sessions: int = 1500,
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    hidden_dim: int = 64,
    num_layers: int = 2,
    seed: int = 42,
    artifacts_dir: str = None,
) -> Dict[str, Any]:
    """Full training pipeline with checkpointing and evaluation metrics export."""
    set_seed(seed)

    if artifacts_dir is None:
        artifacts_dir = os.path.join(os.path.dirname(__file__), 'artifacts')
    os.makedirs(artifacts_dir, exist_ok=True)

    print(f"[1/4] Generating hemodialysis time-series data ({n_sessions} sessions)...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler_dict = prepare_dataset_splits(
        n_sessions=n_sessions, seed=seed
    )

    print(f"      Train samples: {len(X_train)} | Val samples: {len(X_val)} | Test samples: {len(X_test)}")
    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    pos_weight_val = neg_count / max(1, pos_count)
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32)
    print(f"      IDH Class distribution in Train: Positive={pos_count}, Negative={neg_count} (Weight={pos_weight_val:.2f})")

    # Save scaler parameters
    scaler_path = os.path.join(artifacts_dir, 'scaler_params.json')
    with open(scaler_path, 'w', encoding='utf-8') as f:
        json.dump(scaler_dict, f, indent=2)
    print(f"      Saved scaler params to: {scaler_path}")

    # Prepare PyTorch datasets & dataloaders
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    input_dim = X_train.shape[2]
    model = IDHLSTMNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.2,
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    eval_criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    best_val_f1 = -1.0
    best_state_dict = None
    best_epoch = 0

    print(f"[2/4] Training LSTM network ({epochs} epochs)...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)

        # Validate each epoch
        val_metrics = evaluate_dataset(model, eval_criterion, X_val, y_val, batch_size=batch_size)

        if val_metrics['f1_score'] > best_val_f1:
            best_val_f1 = val_metrics['f1_score']
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

        if epoch % 5 == 0 or epoch == epochs:
            print(f"      Epoch {epoch:02d}/{epochs:02d} - Loss: {epoch_loss/len(X_train):.4f} - Val F1: {val_metrics['f1_score']:.4f} - Val Acc: {val_metrics['accuracy']:.4f}")

    # Load best model weights
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    print(f"[3/4] Evaluating best model (from Epoch {best_epoch})...")
    train_metrics = evaluate_dataset(model, eval_criterion, X_train, y_train, batch_size=batch_size)
    val_metrics = evaluate_dataset(model, eval_criterion, X_val, y_val, batch_size=batch_size)
    test_metrics = evaluate_dataset(model, eval_criterion, X_test, y_test, batch_size=batch_size)

    print(f"      TEST RESULTS:")
    print(f"      - Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"      - Precision: {test_metrics['precision']:.4f}")
    print(f"      - Recall:    {test_metrics['recall']:.4f}")
    print(f"      - F1-Score:  {test_metrics['f1_score']:.4f}")
    print(f"      - Loss:      {test_metrics['loss']:.4f}")
    print(f"      - ROC-AUC:   {test_metrics['roc_auc']:.4f}")

    # [4/4] Save model artifacts
    model_path = os.path.join(artifacts_dir, 'lstm_idh_model.pt')
    torch.save({
        'state_dict': model.state_dict(),
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'best_epoch': best_epoch,
    }, model_path)

    metrics_path = os.path.join(artifacts_dir, 'evaluation_metrics.json')
    evaluation_report = {
        'model_name': 'IDHLSTMNet',
        'framework': 'PyTorch 2.x',
        'architecture': '2-Layer Stacked LSTM with LayerNorm and Dropout',
        'input_features': scaler_dict['feature_names'],
        'sequence_length': int(X_train.shape[1]),
        'metrics': {
            'train': train_metrics,
            'validation': val_metrics,
            'test': test_metrics,
        },
        'training_hyperparameters': {
            'epochs': epochs,
            'best_epoch': best_epoch,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'seed': seed,
        },
    }

    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_report, f, indent=2)

    config_path = os.path.join(artifacts_dir, 'model_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump({
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'sequence_length': int(X_train.shape[1]),
            'feature_names': scaler_dict['feature_names'],
        }, f, indent=2)

    print(f"[4/4] Saved artifacts successfully into {artifacts_dir}")
    return evaluation_report


if __name__ == '__main__':
    train_idh_lstm()
