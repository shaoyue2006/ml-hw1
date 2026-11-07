"""Fold-local transformations and explicit minibatch neural network training."""

import copy
import math
import random

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from contract import FEATURES, TARGET, make_preprocessor


class RentalMLP(nn.Module):
    def __init__(self, input_dim, hidden):
        super().__init__()
        first, second = hidden
        self.layers = nn.Sequential(
            nn.Linear(input_dim, first),
            nn.ReLU(),
            nn.Linear(first, second),
            nn.ReLU(),
            nn.Linear(second, 1),
        )

    def forward(self, features):
        return self.layers(features).squeeze(-1)


def inner_split(frame, fraction):
    dates = frame.dteday.drop_duplicates()
    n_tail = math.ceil(len(dates) * fraction)
    if not frame.dteday.is_monotonic_increasing or not 0 < n_tail < len(dates):
        raise ValueError("Need an ordered window with nonempty fit and stopping days")
    boundary = dates.iloc[-n_tail]
    return frame[frame.dteday < boundary], frame[frame.dteday >= boundary]


def fit_transforms(frame):
    return (
        make_preprocessor(scale_numeric=True, dense=True).fit(frame[FEATURES]),
        StandardScaler().fit(frame[[TARGET]]),
    )


def tensors(frame, transforms):
    features, target = transforms
    x = torch.as_tensor(features.transform(frame[FEATURES]), dtype=torch.float32)
    y = torch.as_tensor(target.transform(frame[[TARGET]]).ravel(), dtype=torch.float32)
    return x, y


def fit_network(train, config, epochs, monitor=None):
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    train_x, train_y = train
    device = torch.device(config["device"])
    model = RentalMLP(train_x.shape[1], config["hidden"]).to(device)
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=config["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    loss_fn = nn.MSELoss()
    best_loss, best_epoch, best_state = float("inf"), 0, None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            train_loss = loss_fn(model(train_x.to(device)), train_y.to(device)).item()
            row = {"epoch": epoch, "train_mse_scaled": train_loss}
            if not math.isfinite(train_loss):
                raise ValueError("Non-finite training loss")
            if monitor is not None:
                x, y = monitor
                val_loss = loss_fn(model(x.to(device)), y.to(device)).item()
                if not math.isfinite(val_loss):
                    raise ValueError("Non-finite early-stopping loss")
                row["inner_mse_scaled"] = val_loss
                if val_loss < best_loss:
                    best_loss, best_epoch = val_loss, epoch
                    best_state = copy.deepcopy(model.state_dict())
        history.append(row)
        if monitor is not None and epoch - best_epoch >= config["patience"]:
            break
    if monitor is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = epochs
    return model.cpu().eval(), history, best_epoch


@torch.inference_mode()
def predict(model, transforms, frame):
    features, target = transforms
    x = torch.as_tensor(features.transform(frame[FEATURES]), dtype=torch.float32)
    scaled = model(x).numpy().astype(np.float64)
    return target.inverse_transform(scaled.reshape(-1, 1)).ravel()
