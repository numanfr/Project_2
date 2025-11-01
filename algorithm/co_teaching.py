import pandas as pd
import numpy as np
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import json
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler
import os

def load_datasets(base_path='data/'):
    """
    Load multiple datasets from .npz files
    
    Args:
        base_path (str): Base directory path where datasets are stored
    
    Returns:
        dict: Dictionary containing all loaded datasets
    """
    datasets = {}
    
    # Define dataset configurations
    dataset_configs = {
        'FashionMNIST_0.3': 'FashionMNIST0.3.npz',
        'FashionMNIST_0.6': 'FashionMNIST0.6.npz',
        'CIFAR': 'CIFAR.npz'
    }
    file_names = ['FashionMNIST0.3.npz', 'FashionMNIST0.6.npz', 'CIFAR.npz']
    for filename in file_names:
        try:
            # Load dataset
            data = np.load(base_path + filename)
            
            # Extract data
            datasets[filename] = {
                'X_train': data['Xtr'],
                'y_train': data['Str'],
                'X_test': data['Xts'],
                'y_test': data['Yts']
            }
            
            # Print shapes
            print(f"\n{filename} dataset shapes:")
            print(f"X_train: {datasets[filename]['X_train'].shape}")
            print(f"y_train: {datasets[filename]['y_train'].shape}")
            print(f"X_test: {datasets[filename]['X_test'].shape}")
            print(f"y_test: {datasets[filename]['y_test'].shape}")

            # Close the dataset
            data.close()
            
        except FileNotFoundError:
            print(f"Warning: {filename} not found at {base_path}")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    return datasets

def preprocess_dataset(file, all_datasets):
    """
    Preprocess FashionMNIST or CIFAR dataset
    Returns: Xtr_t, ytr_t, Xts_t, yts_t, mean, std, is_cifar
    """
    Xtr = all_datasets[file]['X_train']
    ytr = all_datasets[file]['y_train']
    Xts = all_datasets[file]['X_test']
    yts = all_datasets[file]['y_test']
    
    is_cifar = not file.startswith('FashionMNIST')
    
    if not is_cifar:  # FashionMNIST
        # FashionMNIST: reshape to [N, 28, 28] and normalize
        Xtr = Xtr.reshape(-1, 28, 28).astype(np.float32) / 255.0
        Xts = Xts.reshape(-1, 28, 28).astype(np.float32) / 255.0
        
        # Compute stats
        mean = Xtr.mean()
        std = Xtr.std() + 1e-6
        
        # Convert to tensors with channel dimension [N, 1, 28, 28]
        Xtr_t = torch.from_numpy(Xtr[:, None, :, :])
        Xts_t = torch.from_numpy(Xts[:, None, :, :])
        
    else:  # CIFAR
        # CIFAR: already [N, 32, 32, 3], normalize
        Xtr = Xtr.astype(np.float32) / 255.0
        Xts = Xts.astype(np.float32) / 255.0
        
        # Compute per-channel stats
        mean = Xtr.mean(axis=(0, 1, 2))  # [3,]
        std = Xtr.std(axis=(0, 1, 2)) + 1e-6  # [3,]
        
        # Convert to tensors: [N, H, W, C] -> [N, C, H, W]
        Xtr_t = torch.from_numpy(Xtr).permute(0, 3, 1, 2)  # [N, 3, 32, 32]
        Xts_t = torch.from_numpy(Xts).permute(0, 3, 1, 2)  # [N, 3, 32, 32]
    
    # Convert labels to tensors
    ytr_t = torch.from_numpy(ytr.astype(np.int64))
    yts_t = torch.from_numpy(yts.astype(np.int64))
    
    return Xtr_t, ytr_t, Xts_t, yts_t, mean, std, is_cifar

def stratified_split_indices(y: torch.Tensor, train_frac=0.8, seed=42):
    """Stratified train/val split"""
    g = torch.Generator()
    g.manual_seed(seed)
    y_np = y.cpu().numpy()
    classes = np.unique(y_np)
    train_idx, val_idx = [], []
    for c in classes:
        idx_c = np.flatnonzero(y_np == c)
        idx_c = torch.as_tensor(idx_c)
        # shuffle per class
        perm = torch.randperm(idx_c.numel(), generator=g)
        idx_c = idx_c[perm]
        n_train_c = int(np.floor(train_frac * idx_c.numel()))
        train_idx.append(idx_c[:n_train_c])
        val_idx.append(idx_c[n_train_c:])
    train_idx = torch.cat(train_idx).tolist()
    val_idx = torch.cat(val_idx).tolist()
    return train_idx, val_idx


class ArrayDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

# eval function
def eval_top1(model, loader, device):
    model.eval()
    total, correct = 0, 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total

def predict_transition_matrix_anchor(model, loader, device, num_classes=3):
    model.eval()
    prob_batches, pred_batches = [], []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            probs = F.softmax(model(x), dim=1)
            prob_batches.append(probs.cpu())
            pred_batches.append(probs.argmax(dim=1).cpu())
    if not prob_batches:
        return np.zeros((num_classes, num_classes), dtype=np.float32)
    probs = torch.cat(prob_batches)
    preds = torch.cat(pred_batches)
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.float32)
    for cls in range(num_classes):
        mask = preds == cls
        if mask.sum() == 0:
            continue
        probs_cls = probs[mask]
        anchor_idx = probs_cls[:, cls].argmax()
        matrix[cls] = probs_cls[anchor_idx]
    return matrix.numpy()

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 28->14 for MNIST-like 28x28
        x = self.pool(F.relu(self.conv2(x)))   # 14->7
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits

class SimpleCNN_CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Input: 3x32x32 (RGB)
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)   # -> 32x32x32
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # -> 64x32x32
        self.pool = nn.MaxPool2d(2, 2)                # -> 64x16x16
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1) # -> 128x16x16
        # After another pool: -> 128x8x8

        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32->16
        x = self.pool(F.relu(self.conv2(x)))  # 16->8
        x = F.relu(self.conv3(x))             # Keep 8x8 spatial size
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits

def small_loss_indices(losses, keep_k):
    # losses: [B], keep_k: int
    _, idx = torch.topk(-losses, k=keep_k)  # smallest losses
    return idx

def get_keep_k(batch_size, epoch, max_epochs, noise_rate):
    drop_rate = min(noise_rate * (epoch / max_epochs), noise_rate)
    keep_frac = 1.0 - drop_rate
    keep_k = max(1, int(keep_frac * batch_size))
    return keep_k

def train_epoch_coteaching(model_f, model_g, opt_f, opt_g, loader, epoch, max_epochs, noise_rate, device):
    model_f.train(); model_g.train()
    ce = nn.CrossEntropyLoss(reduction='none')
    for x, y_noisy in loader:
        x = x.to(device); y_noisy = y_noisy.to(device)
        B = x.size(0)
        keep_k = get_keep_k(B, epoch, max_epochs, noise_rate)

        # Forward both models
        logits_f = model_f(x)
        logits_g = model_g(x)

        # Per-sample losses
        loss_f_i = ce(logits_f, y_noisy)     # [B]
        loss_g_i = ce(logits_g, y_noisy)     # [B]

        # Small-loss indices per model
        idx_f = small_loss_indices(loss_f_i.detach(), keep_k)
        idx_g = small_loss_indices(loss_g_i.detach(), keep_k)

        # Each model updates on the peer's selected subset
        loss_f_on_g = ce(logits_f[idx_g], y_noisy[idx_g]).mean()
        loss_g_on_f = ce(logits_g[idx_f], y_noisy[idx_f]).mean()

        opt_f.zero_grad(); loss_f_on_g.backward(); opt_f.step()
        opt_g.zero_grad(); loss_g_on_f.backward(); opt_g.step()

def run_coteaching(train_loader, num_classes=10, epochs=50, noise_rate=0.4, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    f = SimpleCNN(num_classes).to(device)
    g = SimpleCNN(num_classes).to(device)
    opt_f = torch.optim.SGD(f.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    opt_g = torch.optim.SGD(g.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    for ep in range(1, epochs+1):
        train_epoch_coteaching(f, g, opt_f, opt_g, train_loader, ep, epochs, noise_rate, device)

    return f, g



def run_coteaching_cv(dataset, n_splits=5, batch_size=128, num_classes=3,
                      epochs=50, noise_rate=0.4, shuffle=True, seed=42, device='cuda',
                      val_metric_fn=None, model_class=SimpleCNN):
    """
    K-fold cross-validation for Co-teaching.
    - dataset: a torch.utils.data.Dataset (map-style) returning (x, y_noisy)
    - val_metric_fn: callable(model, val_loader, device) -> float; if None, returns None
    Returns:
      results: list of dicts per fold with keys {'fold', 'val_metric_f', 'val_metric_g', 'val_metric_ens'}
      models: list of tuples (f, g) trained on each fold
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)

    # Create a proper dataset from the dataframe if needed
    if hasattr(dataset, 'iloc'):  # It's a DataFrame
        X_data = torch.stack([x for x in dataset['X']])
        y_data = torch.stack([y for y in dataset['y']])
        dataset = ArrayDataset(X_data, y_data)

    results = []
    models = []

    indices = torch.arange(len(dataset))
    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"\nFold {fold + 1}/{n_splits}")
        
        # DataLoaders for this fold
        train_sampler = SubsetRandomSampler(indices[train_idx])
        val_sampler   = SubsetRandomSampler(indices[val_idx])
        train_loader  = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True)
        val_loader    = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

        # Fresh models per fold - use the correct SimpleCNN class
        f = model_class(num_classes).to(device)
        g = model_class(num_classes).to(device)
        opt_f = torch.optim.SGD(f.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        opt_g = torch.optim.SGD(g.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

        # Train
        for ep in range(1, epochs+1):
            train_epoch_coteaching(f, g, opt_f, opt_g, train_loader, ep, epochs, noise_rate, device)
            if ep % max(1, epochs // 5) == 0:
                print(f"  Epoch {ep:02d} completed")

        # Optional validation on this fold
        val_metric_f = eval_top1(f, val_loader, device) if val_metric_fn is not None else None
        val_metric_g = eval_top1(g, val_loader, device) if val_metric_fn is not None else None

        # Simple logit-averaged ensemble for validation
        if val_metric_fn is not None:
            class Ensemble(nn.Module):
                def __init__(self, f, g):
                    super().__init__(); self.f=f.eval(); self.g=g.eval()
                def forward(self, x):
                    return (self.f(x) + self.g(x)) / 2
            ens = Ensemble(f, g).to(device)
            val_metric_ens = eval_top1(ens, val_loader, device)
        else:
            val_metric_ens = None

        print(f"Fold {fold + 1} - Model F: {val_metric_f:.2f}%, Model G: {val_metric_g:.2f}%, Ensemble: {val_metric_ens:.2f}%")

        results.append({
            'fold': fold,
            'val_metric_f': val_metric_f,
            'val_metric_g': val_metric_g,
            'val_metric_ens': val_metric_ens
        })
        models.append((f, g))

    # Print summary
    if val_metric_fn is not None:
        mean_f = np.mean([r['val_metric_f'] for r in results])
        mean_g = np.mean([r['val_metric_g'] for r in results])
        mean_ens = np.mean([r['val_metric_ens'] for r in results])
        print(f"\nCross-validation summary:")
        print(f"Model F: {mean_f:.2f}% ± {np.std([r['val_metric_f'] for r in results]):.2f}%")
        print(f"Model G: {mean_g:.2f}% ± {np.std([r['val_metric_g'] for r in results]):.2f}%")
        print(f"Ensemble: {mean_ens:.2f}% ± {np.std([r['val_metric_ens'] for r in results]):.2f}%")

    return results, models



def run_coteaching_experiments(num_runs=10, predict_transition=True, output_path="results/"):
    """
    Run Co-teaching experiments on multiple datasets with cross-validation
    """

    # Load all datasets
    all_datasets = load_datasets()

    summary_results = {}

    for file in all_datasets.keys():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Step 1: Preprocess dataset
        Xtr_t, ytr_t, Xts_t, yts_t, mean, std, is_cifar = preprocess_dataset(file, all_datasets)

        # Step 2: Create train_df with proper tensor data
        train_df = pd.DataFrame({
            'X': [Xtr_t[i] for i in range(len(Xtr_t))],
            'y': [ytr_t[i] for i in range(len(ytr_t))]
        })

        print(f"Dataset: {file}")
        print(f"Train shape: {Xtr_t.shape}, Test shape: {Xts_t.shape}")
        print(f"Is CIFAR: {is_cifar}")

        # Step 3: Stratified split
        train_idx, val_idx = stratified_split_indices(ytr_t, train_frac=0.8, seed=2025)

        # Step 4: Create datasets
        train_ds = ArrayDataset(Xtr_t[train_idx], ytr_t[train_idx])
        val_ds = ArrayDataset(Xtr_t[val_idx], ytr_t[val_idx])
        test_ds = ArrayDataset(Xts_t, yts_t)

        # Step 5: Create data loaders
        train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0, pin_memory=False)
        val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=False)
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=False)



        results, models = run_coteaching_cv(
            dataset=train_df, n_splits=num_runs, batch_size=128, num_classes=3,
            epochs=10, noise_rate=0.3, shuffle=True, seed=42, device='cuda', val_metric_fn=True,
            model_class=SimpleCNN_CIFAR if is_cifar else SimpleCNN)

        # Evaluate all CV models on test set
        test_accuracies_f = []
        test_accuracies_g = []
        test_accuracies_ensemble = []

        for fold_idx, (model_f, model_g) in enumerate(models):
            # Evaluate individual models
            test_f = eval_top1(model_f, test_loader, device)
            test_g = eval_top1(model_g, test_loader, device)
            
            # Create ensemble for this fold
            class Ensemble(nn.Module):
                def __init__(self, model_f, model_g):
                    super().__init__()
                    self.f = model_f.eval()
                    self.g = model_g.eval()
                
                def forward(self, x):
                    return (self.f(x) + self.g(x)) / 2
            
            ensemble_fold = Ensemble(model_f, model_g).to(device)
            test_ensemble = eval_top1(ensemble_fold, test_loader, device)
            
            test_accuracies_f.append(test_f)
            test_accuracies_g.append(test_g)
            test_accuracies_ensemble.append(test_ensemble)
            
            print(f"Fold {fold_idx + 1} - Model F: {test_f:.2f}%, Model G: {test_g:.2f}%, Ensemble: {test_ensemble:.2f}%")

        # Calculate statistics
        mean_test_f = np.mean(test_accuracies_f)
        std_test_f = np.std(test_accuracies_f)
        mean_test_g = np.mean(test_accuracies_g)
        std_test_g = np.std(test_accuracies_g)
        mean_test_ensemble = np.mean(test_accuracies_ensemble)
        std_test_ensemble = np.std(test_accuracies_ensemble)

        print(f"\nTest Set Results (across {len(models)} folds):")
        print(f"Model F: {mean_test_f:.2f}% ± {std_test_f:.2f}%")
        print(f"Model G: {mean_test_g:.2f}% ± {std_test_g:.2f}%")
        print(f"Ensemble: {mean_test_ensemble:.2f}% ± {std_test_ensemble:.2f}%")
        full_train_loader = DataLoader(ArrayDataset(Xtr_t, ytr_t), batch_size=256, shuffle=False, num_workers=0, pin_memory=False)
        transition_summary = None
        if predict_transition:
            transition_estimates = []
            for model_f, model_g in models:
                transition_estimates.append(predict_transition_matrix_anchor(model_f, full_train_loader, device, num_classes=3))
                transition_estimates.append(predict_transition_matrix_anchor(model_g, full_train_loader, device, num_classes=3))
            if transition_estimates:
                stacked = np.stack(transition_estimates)
                transition_summary = {
                    "mean": stacked.mean(axis=0).tolist(),
                    "std": stacked.std(axis=0).tolist()
                }
            else:
                zeros = np.zeros((3, 3), dtype=np.float32)
                transition_summary = {"mean": zeros.tolist(), "std": zeros.tolist()}
        summary_results[file] = {
            "test_accuracy": {
                "mean": mean_test_ensemble,
                "std": std_test_ensemble
            }
        }
        if predict_transition and transition_summary is not None:
            summary_results[file]["predicted_transition_matrix"] = transition_summary

    with open(os.path.join(output_path, "summary_coteaching.json"), 'w') as f:
        json.dump(summary_results, f, indent=4)

if __name__ == "__main__":
    run_coteaching_experiments(num_runs=2, predict_transition=True, output_path="results/")