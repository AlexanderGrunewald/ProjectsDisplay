import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import time
from torchensemble import VotingRegressor
import os

model_log = "/mnt/home/grunew14/Documents/ss-23-classes/cmse381/gene-project/logs/ensambleModel.log"

print("Loading data...")
X_train_nn = torch.load("/mnt/home/grunew14/Documents/ss-23-classes/cmse381/gene-project/notebooks/X_train_nn.pt")
y_train_nn = torch.load("/mnt/home/grunew14/Documents/ss-23-classes/cmse381/gene-project/notebooks/y_train_nn.pt")
X_test_nn = torch.load("/mnt/home/grunew14/Documents/ss-23-classes/cmse381/gene-project/notebooks/X_test_nn.pt")
y_test_nn = torch.load("/mnt/home/grunew14/Documents/ss-23-classes/cmse381/gene-project/notebooks/y_test_nn.pt")
print("Shape of X_train_nn:", X_train_nn.shape)
print("Shape of y_train_nn:", y_train_nn.shape)
print("Shape of X_test_nn:", X_test_nn.shape)
print("Shape of y_test_nn:", y_test_nn.shape)

print("Data loaded.")

print("Making training and testing datasets...")
train_dataset = TensorDataset(X_train_nn, y_train_nn)
test_dataset = TensorDataset(X_test_nn, y_test_nn)
train_loader = DataLoader(train_dataset, batch_size=40, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=40, shuffle=True)
print("Datasets made.")
# make a log directory if it doesn't exist
if not os.path.exists("/mnt/home/grunew14/Documents/ss-23-classes/cmse381/gene-project/logs"):
    os.makedirs("/mnt/home/grunew14/Documents/ss-23-classes/cmse381/gene-project/logs")

# if a trained model folder doesn't exist, make one
if not os.path.exists("/mnt/home/grunew14/Documents/ss-23-classes/cmse381/gene-project/trained_models"):
    os.makedirs("/mnt/home/grunew14/Documents/ss-23-classes/cmse381/gene-project/trained_models")


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(X_train_nn.shape[1], 600)
        self.fc2 = nn.Linear(600, 300)
        self.fc3 = nn.Linear(300, 150)
        self.fc4 = nn.Linear(150, y_train_nn.shape[1])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x


if __name__ == "__main__":
    print("Starting training...")
    n_estimators = 10
    lr = 1e-3
    weight_decay = 5e-4
    epochs = 300
    with open(model_log, "a") as f:
        # year, month, day, hour, minute, second
        f.write(f"*"*50 + "\n")
        f.write(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
        f.write(
            f"Voting Regressor Ensemble Model: n_estimators={n_estimators}, lr={lr}, weight_decay={weight_decay}, epochs={epochs}\n"
        )

    # Create ensemble model
    model = VotingRegressor(
        estimator=MLP, n_estimators=n_estimators, cuda=True
    )

    model.set_optimizer("Adam", lr=lr, weight_decay=weight_decay)

    tic = time.time()
    model.fit(train_loader, epochs=epochs)
    toc = time.time()
    training_time = toc - tic

    tic = time.time()
    testing_mse = np.sqrt(model.evaluate(test_loader))
    toc = time.time()
    evaluating_time = toc - tic

    print(f"Testing MSE: {testing_mse}")
    print(f"Training time: {training_time}")
    print(f"Evaluating time: {evaluating_time}")

    with open(model_log, "a") as f:
        # year, month, day, hour, minute, second
        f.write(f"Training time: {training_time/60}\n")
        f.write(f"Testing MSE: {testing_mse}\n")
        f.write(f"Evaluating time: {evaluating_time}\n\n")
        # end time
        f.write(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n\n")

    torch.save(model, "/mnt/home/grunew14/Documents/ss-23-classes/cmse381/gene-project/trained_models/VR_ensambleModel.pt")
    print("Model saved.")
