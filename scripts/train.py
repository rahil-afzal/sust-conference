
import yaml, torch
from utils.seed import set_seed
from training.dataloaders import build_dataloaders
from models.gnn import AdvancedGNNModelWithEdge
from training.train import train

set_seed(42)
config = yaml.safe_load(open("configs/default.yaml"))

dataset = torch.load(config["dataset"]["path"])
train_loader, val_loader, _, _ = build_dataloaders(
    dataset,
    config["training"]["batch_size"],
    config["dataset"]["test_size"],
    config["dataset"]["val_size"]
)

sample = dataset[0]
model = AdvancedGNNModelWithEdge(
    node_fea_len=sample.x.shape[1],
    edge_fea_len=sample.edge_attr.shape[1],
    **config["model"]
)

train(model, train_loader, val_loader, config)
