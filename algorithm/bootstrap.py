import numpy as np
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
import torchvision.transforms as transforms

DATASET_KEY_MAP = {
    "Fash03": "FashionMNIST0.3.npz",
    "Fash06": "FashionMNIST0.6.npz",
    "CIFAR": "CIFAR.npz"
}

class CNN_Model(nn.Module):
    """CNN model for image classification"""
    def __init__(self, num_classes=10, T=None, input_channels=3, input_size=32):
        super(CNN_Model, self).__init__()

        if T is not None:
            self.T = torch.FloatTensor(T)
        else:
            self.T = None
        
        self.input_channels = input_channels
        self.input_size = input_size
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calculate the size after pooling layers
        # 3 pooling layers: size / 2 / 2 / 2
        pooled_size = input_size // 8
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * pooled_size * pooled_size, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Conv block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Conv block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten - use reshape instead of view for compatibility
        x = x.reshape(x.size(0), -1)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def import_data(data="CIFAR"):
    data_path = './data/'
    if data == "CIFAR":
        Data = np.load(data_path + 'CIFAR.npz')
    elif data == "Fash03":
        Data = np.load(data_path + 'FashionMNIST0.3.npz')
    elif data == "Fash06":
        Data = np.load(data_path + 'FashionMNIST0.6.npz')

    X_tr = Data['Xtr']
    S_tr = Data['Str']
    X_ts = Data['Xts']
    Y_ts = Data['Yts']

    return X_tr, S_tr, X_ts, Y_ts

def preprocess_data(X, is_cifar):
    """Preprocess data with proper normalization for both datasets"""
    if is_cifar:
        # CIFAR: RGB images (H, W, C) -> (C, H, W) and normalize
        X_tensor = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2)
        # Normalize to [0, 1]
        X_tensor = X_tensor / 255.0
        # Standardize with approximate ImageNet stats (commonly used for CIFAR)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        X_tensor = (X_tensor - mean) / std
    else:
        # FashionMNIST: Grayscale images
        X_reshaped = X.reshape(-1, 28, 28)
        X_tensor = torch.tensor(X_reshaped, dtype=torch.float32).unsqueeze(1)
        # Normalize to [0, 1]
        X_tensor = X_tensor / 255.0
        # Standardize (mean and std of FashionMNIST)
        X_tensor = (X_tensor - 0.5) / 0.5
    
    return X_tensor

def transition_matrix(data="CIFAR"):
    if data == "Fash03":
        T = np.array([
            [0.7, 0.3, 0.0],
            [0.0, 0.7, 0.3],
            [0.3, 0.0, 0.7]
        ])
    elif data == "Fash06":
        T = np.array([
            [0.4, 0.3, 0.3],
            [0.3, 0.4, 0.3],
            [0.3, 0.3, 0.4]
        ])
    else:
        T = None
    return T

def transition_matrix_estimator(model, X_train, y_train, num_classes, is_cifar, device, anchor_percentile=95):
    """Estimate transition matrix using anchor points (high confidence predictions)"""
    model.eval()
    
    X_tensor = preprocess_data(X_train, is_cifar)
    
    batch_size = 256
    all_probs = []
    
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size].to(device)
            logits = model(batch)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu())
    
    all_probs = torch.cat(all_probs, dim=0).numpy()
    
    T_estimate = np.zeros((num_classes, num_classes))
    
    print("\nFinding anchor points for each class...")
    
    for true_class in range(num_classes):
        class_probs = all_probs[:, true_class]
        threshold = np.percentile(class_probs, anchor_percentile)
        anchor_indices = np.where(class_probs >= threshold)[0]
        
        if len(anchor_indices) > 0:
            anchor_noisy_labels = y_train[anchor_indices]
            
            for noisy_class in range(num_classes):
                T_estimate[true_class, noisy_class] = np.sum(anchor_noisy_labels == noisy_class) / len(anchor_indices)

        else:
            T_estimate[true_class, true_class] = 1.0
            print(f"Class {true_class}: No anchor points found, assuming no noise")

    print(f"Estimated transition matrix:\n{np.round(T_estimate, 2)}")
    
    return T_estimate

def bootstrapping(X_train, y_train, num_classes, is_cifar, device, 
                         num_epochs=20, batch_size=128, lr=0.001, K=0.1, tau=0.99, val_split=0.2):

    
    input_size = 32 if is_cifar else 28
    input_channels = 3 if is_cifar else 1
    
    # Create CNN model
    model = CNN_Model(num_classes=num_classes, input_channels=input_channels, input_size=input_size)
    model = model.to(device)
    
    # Split into train/val with randomization
    num_samples = len(X_train)
    indices = np.random.permutation(num_samples)
    num_train = int((1 - val_split) * num_samples)
    train_idx, val_idx = indices[:num_train], indices[num_train:]
    
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    
    print(f"\nStage 1 - Train/Val Split: {len(X_tr)} train, {len(X_val)} val")
    
    # Preprocess data
    X_tr_tensor = preprocess_data(X_tr, is_cifar)
    y_tr_tensor = torch.tensor(y_tr, dtype=torch.long)
    X_val_tensor = preprocess_data(X_val, is_cifar)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    
    train_dataset = torch.utils.data.TensorDataset(X_tr_tensor, y_tr_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Early stopping training
    print("\nEarly stopping training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Validation accuracy
        if (epoch + 1) % 5 == 0:
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logits = model(inputs)
                    _, predicted = torch.max(logits, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_acc = 100 * val_correct / val_total
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Val Acc: {val_acc:.2f}%')
    
    # Generate predictions with dropout enabled on training set
    print("\nGenerating predictions for clean/noisy separation...")
    X_tensor = preprocess_data(X_train, is_cifar)
    model.eval()
    
    # Enable dropout during evaluation
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    
    all_predictions = []
    num_augmentations = 25
    
    with torch.no_grad():
        for _ in range(num_augmentations):
            batch_preds = []
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size].to(device)
                logits = model(batch)
                probs = F.softmax(logits, dim=1)
                batch_preds.append(probs.cpu())
            all_predictions.append(torch.cat(batch_preds, dim=0))
    
    # Average predictions
    avg_predictions = torch.stack(all_predictions).mean(dim=0).numpy()
    confidences = np.max(avg_predictions, axis=1)
    predicted_labels = np.argmax(avg_predictions, axis=1)
    
    # Estimate noise transition matrix using top 90% confident predictions per class
    print("\nEstimating noise transition matrix...")
    T_est = np.zeros((num_classes, num_classes))
    
    for c in range(num_classes):
        class_mask = predicted_labels == c
        if np.sum(class_mask) > 0:
            class_confidences = confidences[class_mask]
            threshold = np.percentile(class_confidences, 10)  # Top 90%
            high_conf_mask = class_mask & (confidences >= threshold)
            
            if np.sum(high_conf_mask) > 0:
                noisy_labels_for_class = y_train[high_conf_mask]
                for noisy_c in range(num_classes):
                    T_est[c, noisy_c] = np.sum(noisy_labels_for_class == noisy_c) / np.sum(high_conf_mask)
    
    print(f"Estimated transition matrix:\n{np.round(T_est, 2)}")
    
    # Noise-transition sample balancing
    print("\nPerforming noise-transition sample balancing...")
    clean_indices = []
    
    # Select K * |Y| * T[i,j] most confident samples from each noise transition
    for pred_class in range(num_classes):
        for noisy_class in range(num_classes):
            if T_est[pred_class, noisy_class] > 0:
                # Find samples where prediction=pred_class and noisy_label=noisy_class
                mask = (predicted_labels == pred_class) & (y_train == noisy_class)
                indices = np.where(mask)[0]
                
                if len(indices) > 0:
                    # Select top K * |Y| * T[i,j] confident samples
                    num_select = int(K * num_classes * T_est[pred_class, noisy_class] * len(y_train))
                    num_select = max(1, min(num_select, len(indices)))
                    
                    confs = confidences[indices]
                    top_indices = indices[np.argsort(-confs)[:num_select]]
                    clean_indices.extend(top_indices.tolist())
    
    # Also add samples with confidence > tau
    high_conf_indices = np.where(confidences > tau)[0]
    clean_indices = list(set(clean_indices) | set(high_conf_indices.tolist()))
    
    noisy_indices = list(set(range(len(y_train))) - set(clean_indices))
    
    print(f"\nClean set size: {len(clean_indices)}")
    print(f"Noisy set size: {len(noisy_indices)}")
    
    return model, clean_indices, noisy_indices, predicted_labels

def semi_supervised(model, X_train, y_train, clean_indices, noisy_indices, 
                          predicted_labels, num_classes, is_cifar, device,
                          num_epochs=100, batch_size=64, lr=0.001, tau_threshold=0.95, val_split=0.2):
    
    X_tensor = preprocess_data(X_train, is_cifar)
    
    # Prepare clean and noisy sets
    X_clean = X_tensor[clean_indices]
    y_clean = predicted_labels[clean_indices]  # Use predicted labels as clean labels
    
    X_noisy = X_tensor[noisy_indices]
    y_noisy_labels = y_train[noisy_indices]
    
    # Split clean data into train/val with randomization
    num_clean = len(X_clean)
    clean_perm = np.random.permutation(num_clean)
    num_clean_train = int((1 - val_split) * num_clean)
    clean_train_idx, clean_val_idx = clean_perm[:num_clean_train], clean_perm[num_clean_train:]
    
    X_clean_train = X_clean[clean_train_idx]
    y_clean_train = y_clean[clean_train_idx]
    X_clean_val = X_clean[clean_val_idx]
    y_clean_val = y_clean[clean_val_idx]
    
    print(f"\nStage 2 - Clean Train/Val Split: {len(X_clean_train)} train, {len(X_clean_val)} val")
    print(f"Noisy samples for semi-supervised learning: {len(X_noisy)}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Training loop
    print("\nSemi-supervised training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Sample mini-batches
        num_batches = max(len(X_clean_train) // batch_size, 1)
        
        for batch_idx in range(num_batches):
            # Sample clean batch
            clean_batch_idx = np.random.choice(len(X_clean_train), size=min(batch_size, len(X_clean_train)), replace=False)
            inputs_clean = X_clean_train[clean_batch_idx].to(device)
            labels_clean = torch.tensor(y_clean_train[clean_batch_idx], dtype=torch.long).to(device)
            
            # Sample noisy batch (3x size as per paper)
            noisy_batch_size = min(3 * batch_size, len(X_noisy))
            noisy_batch_idx = np.random.choice(len(X_noisy), size=noisy_batch_size, replace=False)
            inputs_noisy = X_noisy[noisy_batch_idx].to(device)
            
            optimizer.zero_grad()
            
            # Supervised loss on clean samples
            logits_clean = model(inputs_clean)
            loss_supervised = criterion(logits_clean, labels_clean)
            
            # Unsupervised loss on noisy samples (pseudo-labeling)
            with torch.no_grad():
                logits_noisy_weak = model(inputs_noisy)
                probs_noisy = F.softmax(logits_noisy_weak, dim=1)
                max_probs, pseudo_labels = torch.max(probs_noisy, dim=1)
                mask = max_probs > tau_threshold
            
            if mask.sum() > 0:
                logits_noisy_strong = model(inputs_noisy)
                loss_unsupervised = criterion(logits_noisy_strong[mask], pseudo_labels[mask])
                loss = loss_supervised + loss_unsupervised
            else:
                loss = loss_supervised
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Validation on clean validation set
        if (epoch + 1) % 20 == 0:
            model.eval()
            val_correct = 0
            val_total = 0
            
            val_batch_size = 256
            with torch.no_grad():
                for i in range(0, len(X_clean_val), val_batch_size):
                    batch_end = min(i + val_batch_size, len(X_clean_val))
                    inputs = X_clean_val[i:batch_end].to(device)
                    labels = torch.tensor(y_clean_val[i:batch_end], dtype=torch.long).to(device)
                    
                    logits = model(inputs)
                    _, predicted = torch.max(logits, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_acc = 100 * val_correct / val_total if val_total > 0 else 0
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/num_batches:.4f}, Val Acc: {val_acc:.2f}%')
    
    # Relabel all samples
    print("\nRelabeling all samples...")
    model.eval()
    all_predictions = []
    
    batch_size_relabel = 256
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size_relabel):
            batch = X_tensor[i:i+batch_size_relabel].to(device)
            logits = model(batch)
            probs = F.softmax(logits, dim=1)
            all_predictions.append(probs.cpu())
    
    all_predictions = torch.cat(all_predictions, dim=0)
    relabeled = torch.argmax(all_predictions, dim=1).numpy()
    
    return model, relabeled

def final_training(X_train, y_train, relabeled, num_classes, is_cifar, device,
                         num_epochs=50, batch_size=128, lr=0.001, val_split=0.2):

    print("\n" + "="*70)
    print("STAGE 3: FINAL TRAINING")
    print("="*70)
    
    input_size = 32 if is_cifar else 28
    input_channels = 3 if is_cifar else 1
    
    # Create fresh CNN model
    model = CNN_Model(num_classes=num_classes, input_channels=input_channels, input_size=input_size)
    model = model.to(device)
    
    # Split into train/val with randomization
    num_samples = len(X_train)
    indices = np.random.permutation(num_samples)
    num_train = int((1 - val_split) * num_samples)
    train_idx, val_idx = indices[:num_train], indices[num_train:]
    
    X_tr, relabeled_tr = X_train[train_idx], relabeled[train_idx]
    X_val, relabeled_val = X_train[val_idx], relabeled[val_idx]
    
    print(f"Stage 3 - Train/Val Split: {len(X_tr)} train, {len(X_val)} val")
    
    # Use relabeled data
    X_tr_tensor = preprocess_data(X_tr, is_cifar)
    y_tr_tensor = torch.tensor(relabeled_tr, dtype=torch.long)
    
    X_val_tensor = preprocess_data(X_val, is_cifar)
    y_val_tensor = torch.tensor(relabeled_val, dtype=torch.long)
    
    train_dataset = torch.utils.data.TensorDataset(X_tr_tensor, y_tr_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Training with MixUp
    print("\nFinal training with MixUp...")
    alpha = 1.0  # MixUp parameter
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Apply MixUp
            lam = np.random.beta(alpha, alpha)
            batch_size_actual = inputs.size(0)
            index = torch.randperm(batch_size_actual).to(device)
            
            mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
            labels_a, labels_b = labels, labels[index]
            
            optimizer.zero_grad()
            logits = model(mixed_inputs)
            
            loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        scheduler.step()
        
        # Validation accuracy
        if (epoch + 1) % 10 == 0:
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logits = model(inputs)
                    _, predicted = torch.max(logits, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_acc = 100 * val_correct / val_total
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Val Acc: {val_acc:.2f}%')
    
    return model

def forward_correction(X_train, y_train, X_test, y_test, T, num_classes, is_cifar, device,
                      num_epochs=50, batch_size=128, lr=0.001):
    
    input_size = 32 if is_cifar else 28
    input_channels = 3 if is_cifar else 1
    
    # Use CNN model
    model = CNN_Model(num_classes=num_classes, T=T, input_channels=input_channels, input_size=input_size)
    model = model.to(device)
    
    # Split into train/val
    num_train = int(0.8 * len(X_train))
    indices = np.random.permutation(len(X_train))
    train_idx, val_idx = indices[:num_train], indices[num_train:]
    
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    
    X_tr_tensor = preprocess_data(X_tr, is_cifar)
    X_val_tensor = preprocess_data(X_val, is_cifar)
    X_test_tensor = preprocess_data(X_test, is_cifar)
    
    y_tr_tensor = torch.tensor(y_tr, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    train_dataset = torch.utils.data.TensorDataset(X_tr_tensor, y_tr_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    if T is None:
        only_estimate_T = True
    else:
        T_tensor = torch.FloatTensor(T)
        only_estimate_T = False
    
    print("\nTraining with forward correction...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(inputs)
            probs = F.softmax(logits, dim=1)
            if only_estimate_T:
                probs = probs
            else:
                probs = torch.matmul(probs, T_tensor.to(device))
            log_probs = torch.log(probs + 1e-8)
            loss = criterion(log_probs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            # Validation accuracy
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logits = model(inputs)
                    probs = F.softmax(logits, dim=1)
                    if only_estimate_T:
                        probs = probs
                    else:
                        probs = torch.matmul(probs, T_tensor.to(device))
                    _, predicted = torch.max(probs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_acc = 100 * val_correct / val_total
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Val Acc: {val_acc:.2f}%')
    
    # Test accuracy
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            _, predicted = torch.max(logits, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    test_acc = 100 * test_correct / test_total
    print(f'\nFinal Test Accuracy: {test_acc:.2f}%')
    
    return model, test_acc

def run_experiment(data="CIFAR", algorithm="bootstrap", output_path="./results/", est_t = False, runs=5, num_epochs_stage1=20,
                  num_epochs_stage2=100, num_epochs_stage3=50, batch_size=128, lr=0.001):

    X_train, y_train, X_test, y_test = import_data(data)
    num_classes = len(np.unique(y_train))
    is_cifar = (data == "CIFAR")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\nUsing device: {device}")
    

    if est_t:
        T = None
    else:
        T = transition_matrix(data)

    if T is not None:
        print(f"Using known transition matrix T:\n{np.round(T, 2)}")
    else:
        print("No known transition matrix T available.")
        T = None
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {data} - {algorithm.upper()} ALGORITHM")
    print(f"{'='*70}")
    print(f"Dataset: {data} ({'RGB' if is_cifar else 'Grayscale'})")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    print(f"Number of classes: {num_classes}")
    
    all_test_accuracies = []
    transition_matrix_estimates = []
    
    for run in range(runs):
        print(f"\n{'='*70}")
        print(f"RUN {run + 1}/{runs}")
        print(f"{'='*70}")
        
        # Set random seed for reproducibility but different for each run
        np.random.seed(42 + run)
        torch.manual_seed(42 + run)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42 + run)
        
        if algorithm == "bootstrap":
            # Stage 1: Bootstrapping
            model, clean_indices, noisy_indices, predicted_labels = bootstrapping(
                X_train, y_train, num_classes, is_cifar, device,
                num_epochs=num_epochs_stage1, batch_size=batch_size, lr=lr
            )
            
            # Stage 2: Semi-Supervised Learning
            model, relabeled = semi_supervised(
                model, X_train, y_train, clean_indices, noisy_indices,
                predicted_labels, num_classes, is_cifar, device,
                num_epochs=num_epochs_stage2, batch_size=batch_size//2, lr=lr
            )
            
            # Stage 3: Final Training
            model = final_training(
                X_train, y_train, relabeled, num_classes, is_cifar, device,
                num_epochs=num_epochs_stage3, batch_size=batch_size, lr=lr
            )
            
            # Evaluate
            X_test_tensor = preprocess_data(X_test, is_cifar)
            y_test_tensor = torch.tensor(y_test, dtype=torch.long)
            test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            model.eval()
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logits = model(inputs)
                    _, predicted = torch.max(logits, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()
            
            test_acc = 100 * test_correct / test_total  
            print(f'\nFinal Test Accuracy: {test_acc:.2f}%')
            all_test_accuracies.append(test_acc)

            T_estimate = transition_matrix_estimator(model, X_train, y_train, num_classes, is_cifar, device)
            transition_matrix_estimates.append(T_estimate)
            
        elif algorithm == "forward" and T is not None:
            
            model, test_acc = forward_correction(
                X_train, y_train, X_test, y_test, T, num_classes, is_cifar, device,
                num_epochs=num_epochs_stage3, batch_size=batch_size, lr=lr
            )

            T_estimate = transition_matrix_estimator(model, X_train, y_train, num_classes, is_cifar, device)
            transition_matrix_estimates.append(T_estimate)
            all_test_accuracies.append(test_acc)

        elif algorithm == "forward" and T is None:

            model , test_acc = forward_correction(
                X_train, y_train, X_test, y_test, T, num_classes, is_cifar, device,
                num_epochs=10, batch_size=batch_size, lr=lr
            )

            T_estimate = transition_matrix_estimator(model, X_train, y_train, num_classes, is_cifar, device)
            transition_matrix_estimates.append(T_estimate)

            model , test_acc = forward_correction(
                X_train, y_train, X_test, y_test, T_estimate, num_classes, is_cifar, device,
                num_epochs=num_epochs_stage3, batch_size=batch_size, lr=lr
            )
            all_test_accuracies.append(test_acc)

    mean_acc = float(np.mean(all_test_accuracies)) if all_test_accuracies else 0.0
    std_acc = float(np.std(all_test_accuracies)) if all_test_accuracies else 0.0
    if transition_matrix_estimates:
        stacked_T = np.stack(transition_matrix_estimates)
        mean_T = stacked_T.mean(axis=0)
        std_T = stacked_T.std(axis=0)
    else:
        mean_T = np.zeros((num_classes, num_classes))
        std_T = np.zeros((num_classes, num_classes))
    
    print(f"\n{'='*70}")
    print(f"RESULTS - {data} - {algorithm.upper()}")
    print(f"{'='*70}")
    print(f"Mean Test Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"Estimated Transition Matrix (averaged over runs):\n{np.round(mean_T, 2)}")
    print(f"Individual runs: {[f'{acc:.2f}%' for acc in all_test_accuracies]}")
    print(f"{'='*70}\n")

    return {
        'dataset': data,
        'algorithm': algorithm,
        'mean_acc': mean_acc,
        'std_acc': std_acc,
        'all_accs': all_test_accuracies,
        'mean_T': mean_T,
        'std_T': std_T
    }

def run_bootstrap_experiments(num_runs=3, output_path="./results/",datasets=["Fash03", "Fash06","CIFAR"]):
    results = []
    summary_results = {}
    os.makedirs(output_path, exist_ok=True)
    for dataset in datasets:
        output_path_dataset = f"{output_path}/{dataset}"
        # os.makedirs(output_path_dataset, exist_ok=True)



    for dataset in datasets:
        result = run_experiment(
            data=dataset,
            algorithm="bootstrap",
            output_path=output_path,
            runs=num_runs,
            num_epochs_stage1=20,
            num_epochs_stage2=100,
            num_epochs_stage3=50,
            batch_size=128,
            lr=0.001,
            
        )
        if result:
            results.append(result)
            dataset_key = DATASET_KEY_MAP.get(dataset, dataset)
            summary_results[dataset_key] = {
                "test_accuracy": {
                    "mean": result['mean_acc'],
                    "std": result['std_acc']
                },
                "predicted_transition_matrix": {
                    "mean": result['mean_T'].tolist(),
                    "std": result['std_T'].tolist()
                }
            }

    
    # Summary
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY - ALL EXPERIMENTS")
    print(f"{'='*70}")
    for result in results:
        dataset_key = DATASET_KEY_MAP.get(result['dataset'], result['dataset'])
        print(f"\nDataset: {dataset_key}, Algorithm: {result['algorithm']}")
        print(f"  Mean Accuracy: {result['mean_acc']:.2f}% ± {result['std_acc']:.2f}%")
        print(f"  Transition Matrix:\n{np.round(result['mean_T'], 2)}")
    print(f"{'='*70}")
    summary_path = os.path.join(output_path, "summary_bootstrap.json")
    with open(summary_path, 'w') as f:
        json.dump(summary_results, f, indent=4)

if __name__ == "__main__":
    run_bootstrap_experiments(num_runs=2, output_path="results/", datasets=["Fash03", "Fash06","CIFAR"])