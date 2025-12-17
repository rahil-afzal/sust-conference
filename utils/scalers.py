
import torch
from sklearn.preprocessing import RobustScaler

def fit_robust_scaler(y):
    scaler = RobustScaler()
    scaler.fit(y.numpy())
    return scaler

def apply_scaler(dataset, scaler):
    for g in dataset:
        g.y = torch.tensor(scaler.transform(g.y.numpy()), dtype=torch.float32)
