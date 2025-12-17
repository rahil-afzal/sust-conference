
import torch
import torch.nn as nn

def train(model, train_loader, val_loader, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"]
    )
    loss_fn = nn.L1Loss()

    best, patience = float("inf"), 0
    for epoch in range(config["training"]["epochs"]):
        model.train()
        for b in train_loader:
            b = b.to(device)
            opt.zero_grad()
            loss = loss_fn(model(b.x, b.edge_index, b.edge_attr, b.batch), b.y)
            loss.backward()
            opt.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for b in val_loader:
                b = b.to(device)
                val_loss += loss_fn(model(b.x, b.edge_index, b.edge_attr, b.batch), b.y).item()

        if val_loss < best:
            best, patience = val_loss, 0
        else:
            patience += 1
            if patience >= config["training"]["patience"]:
                break
