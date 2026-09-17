"""
Unified CNN Training Loop - Train any CNN architecture with consistent interface.
Saves best checkpoint based on validation accuracy.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from typing import Dict, Any


def train_cnn_model(model: nn.Module,
                    train_loader: DataLoader,
                    val_loader: DataLoader,
                    epochs: int = 20,
                    learning_rate: float = 0.001,
                    device: str = 'cuda',
                    model_name: str = 'resnet50',
                    checkpoint_dir: str = 'cnn_results/checkpoints') -> Dict[str, Any]:
    """
    train a CNN model and save the best checkpoint.
    args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on ('cuda' or 'cpu')
        model_name: Name of the model (for checkpoint naming)
        checkpoint_dir: Directory to save checkpoints
        
    returns:
        Dictionary with training history containing:
        - train_loss: List of train losses per epoch
        - val_loss: List of validation losses per epoch
        - train_acc: List of train accuracies per epoch
        - val_acc: List of validation accuracies per epoch
        - epoch_times: List of epoch durations in seconds
    """
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'epoch_times': []
    }
    
    best_val_acc = 0
    checkpoint_path = f"{checkpoint_dir}/best_{model_name}.pt"
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # handle models with auxiliary outputs (e.g., Inception-V3)
            if hasattr(model, 'training') and model.training:
                outputs = model(images)
                if isinstance(outputs, tuple):  # Inception-V3 returns (output, aux_output)
                    logits = outputs[0]
                else:
                    logits = outputs
            else:
                logits = model(images)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * labels.size(0)
            _, predicted = logits.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                if isinstance(outputs, tuple):  
                    logits = outputs[0]
                else:
                    logits = outputs
                
                loss = criterion(logits, labels)
                
                val_loss += loss.item() * labels.size(0)
                _, predicted = logits.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        epoch_time = time.time() - epoch_start
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['epoch_times'].append(epoch_time)
        
        # save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"Time: {epoch_time:.2f}s | Best checkpoint saved!")
        else:
            print(f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"Time: {epoch_time:.2f}s")
        
        scheduler.step()
    
    return history
