
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from utils.scalers import fit_robust_scaler, apply_scaler

def build_dataloaders(dataset, batch_size, test_size, val_size, seed=42):
    idx = list(range(len(dataset)))
    train_val, test = train_test_split(idx, test_size=test_size, random_state=seed)
    train, val = train_test_split(train_val, test_size=val_size, random_state=seed)

    train_set = [dataset[i] for i in train]
    val_set = [dataset[i] for i in val]
    test_set = [dataset[i] for i in test]

    scaler = fit_robust_scaler(torch.stack([g.y for g in train_set]))
    apply_scaler(train_set, scaler)
    apply_scaler(val_set, scaler)
    apply_scaler(test_set, scaler)

    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(val_set, batch_size=batch_size),
        DataLoader(test_set, batch_size=batch_size),
        scaler
    )
