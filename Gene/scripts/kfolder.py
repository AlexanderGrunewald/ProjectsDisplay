import numpy as np
import anndata as ad
from sklearn.model_selection import KFold
import torch
import pandas as pd



def Kfold_creator(X, y, n_splits=5):
    # X is a tensor

    myKFold = KFold(n_splits=n_splits)
    for i, (train_index, val_index) in enumerate(myKFold.split(X)):

        X_train, X_test = X[train_index], X[val_index]
        y_train, y_test = y[train_index], y[val_index]

        X_train_nn = torch.Tensor(X_train)
        X_test_nn = torch.Tensor(X_test)
        torch.save(X_train_nn, "/mnt/scratch/grunew14/data/X_train_nn_e{}.pt".format(i))
        torch.save(X_test_nn, "/mnt/scratch/grunew14/data/X_test_nn_e{}.pt".format(i))
        torch.save(torch.Tensor(y_train), "/mnt/scratch/grunew14/data/y_train_nn_e{}.pt".format(i))
        torch.save(torch.Tensor(y_test), "/mnt/scratch/grunew14/data/y_test_nn_e{}.pt".format(i))


if __name__ == "__main__":

    X_train = torch.load("/mnt/scratch/grunew14/data/X_train_nn_2.pt").numpy()
    y_train = ad.read_h5ad('/mnt/scratch/grunew14/data/Gex_processed_training.h5ad').X.toarray()

    Kfold_creator(X_train, y_train)
