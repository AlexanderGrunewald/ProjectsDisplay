import numpy as np
import matplotlib.pyplot as plt
import anndata as ad
import pandas as pd
import scanpy as sc
import argparse
import random
import magic
import pickle
import umap
from sklearn.decomposition import PCA
from scipy.sparse import csc_matrix
from torch import Tensor
import torch


if __name__ == "__main__":

    print("Loading data...")
    X_train = ad.read_h5ad('/mnt/scratch/grunew14/data/Gex_processed_training.h5ad').to_df()
    X_test = ad.read_h5ad('/mnt/scratch/grunew14/data/Gex_processed_testing.h5ad').to_df()
    print("Data loaded.")

    print("scale data...")
    mu_train = X_train.mean(axis=0)
    sd_train = X_train.std(axis=0)
    X_train_scaled = (X_train - mu_train)/sd_train
    X_test_scaled = (X_test - mu_train)/sd_train

    print("Imputing...")
    imputer = magic.MAGIC()
    X_train = imputer.fit_transform(X_train_scaled)
    X_test = imputer.fit_transform(X_test_scaled)


    print("UMAP...")
    uma = umap.UMAP(n_components=300, random_state=42)
    X_train_umap = uma.fit_transform(X_train)
    X_test_umap = uma.transform(X_test)
    print("UMAP done.")
    print("X_train shape: ", X_train_umap.shape)
    print("X_test shape: ", X_test_umap.shape)

    print("saving data...")
    # save numpy array as npy file
    X_train_nn = Tensor(X_train_umap)
    X_test_nn = Tensor(X_test_umap)
    torch.save(X_train_nn, "/mnt/scratch/grunew14/data/X_train_nn_4.pt")
    torch.save(X_test_nn, "/mnt/scratch/grunew14/data/X_test_nn_4.pt")
