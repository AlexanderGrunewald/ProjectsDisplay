import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import argparse
import anndata as ad
import torch.nn as nn
from torch import Tensor
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import torch.nn.functional as F
import torch.optim as optim
import sys
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import KFold
from tqdm import tqdm

X_train_nn, y_train_nn, X_test_nn, y_test_nn = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
number_pcs = sys.argv[5]

X_train_nn, X_test_nn = X_train_nn[:, :number_pcs], X_test_nn[:, :number_pcs]


def reset_parameters(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def accuracy(outputs, targets):
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    ss_res = ((targets - outputs) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot)
    return r2


def rmse_loss(outputs, targets):
    mse_loss = F.mse_loss(outputs, targets)
    rmse_loss_ = torch.sqrt(mse_loss)
    return rmse_loss_


if __name__ == "__main__":
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(X_train_nn.shape[1], 312)
            self.fc2 = nn.Linear(312, 270)
            self.fc3 = nn.Linear(270, 103)
            self.fc4 = nn.Linear(103, y_train_nn.shape[1])

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = self.fc4(x)

            return x


    net2 = MLP()

    device = "cuda:0"
    net = net2.to(device)
    reset_parameters(net)

    optimizer = optim.AdamW(net.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    X_train_nn = X_train_nn.to(device)
    y_train_nn = y_train_nn.to(device)

    epochs = 1000
    batch_size = 32
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True)

    # Early stopping parameters
    patience = 10
    best_val_loss = float('inf')

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_nn)):
        print(f'FOLD {fold + 1}')
        net.train()  # set the network to training mode
        counter = 0  # reset the counter for early stopping
        train_data = torch.utils.data.TensorDataset(X_train_nn[train_idx], y_train_nn[train_idx])
        val_data = torch.utils.data.TensorDataset(X_train_nn[val_idx], y_train_nn[val_idx])
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size)

        for epoch in range(epochs):
            train_loss = 0
            val_loss = 0
            train_acc = 0
            val_acc = 0

            progress_bar = tqdm(train_loader, desc=f'Fold: {fold + 1} Epoch: {epoch + 1}/{epochs}')

            for batch_idx, (inputs, targets) in enumerate(progress_bar):
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = rmse_loss(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_acc += accuracy(outputs, targets)  # replace with your own accuracy function

                # Update the progress bar
                progress_bar.set_postfix({'loss': train_loss / (batch_idx + 1)})

            train_loss /= len(train_loader)
            train_acc /= len(train_loader)

            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                with torch.no_grad():
                    net.eval()
                    outputs = net(inputs)
                    loss = rmse_loss(outputs, targets)
                    val_loss += loss.item()
                    val_acc += accuracy(outputs, targets)  # replace with your own accuracy function

            val_loss /= len(val_loader)
            val_acc /= len(val_loader)

            # print training statistics
            tqdm.write(f'Fold: {fold + 1} Epoch: {epoch + 1}/{epochs} Loss: {train_loss:.6f} Val Loss: {val_loss:.6f}')

            # check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    tqdm.write(f'Early stopping at epoch {epoch + 1}')
                    break

    torch.save(net.state_dict(), f'mnt/home/grunew14/ss-23')
