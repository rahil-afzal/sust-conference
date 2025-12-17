
# Graph Neural Networks for Formation Energy Prediction

This repository implements an edge-conditioned graph neural network for
predicting crystal formation energies **from preprocessed atomic graphs**.

## Scope (Important)
- All graphs are assumed to be **preprocessed** PyTorch Geometric `Data` objects.
- No structure-to-graph, chemistry featurization, or neighbor finding is performed here.
- This repository focuses strictly on **model architecture, training, and evaluation**.

## Graph Format
Each graph must contain:
- `x`: Node features `[num_nodes, node_feature_dim]`
- `edge_index`: Graph connectivity `[2, num_edges]`
- `edge_attr`: Edge features `[num_edges, edge_feature_dim]`
- `y`: Formation energy target `[1]` (eV/atom)

## Training
```bash
python scripts/train.py --config configs/default.yaml
```

## Hyperparameter Optimization
```bash
python scripts/optuna_search.py --config configs/optuna.yaml
```

## Metrics
- MAE (eV/atom)
- RMSE (eV/atom)
- R²
