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
from sklearn.preprocessing import normalize
from torch import Tensor
import torch


if __name__ == "__main__":

    print("Loading data...")
    X_train = ad.read_h5ad('/mnt/scratch/grunew14/data/Gex_processed_training.h5ad').to_df()
    X_test = ad.read_h5ad('/mnt/scratch/grunew14/data/Gex_processed_testing.h5ad').to_df()
    print("Data loaded.")
    
    X_train_scaled = normalize(X_train.X.toarray())
    X_teest_scaled = normalize(X_test.X.toarray())
    
    print("Imputing...")
    
    imputer = magic.MAGIC()
    X_train = imputer.fit_transform(X_train_scaled)
    X_test = imputer.fit_transform(X_test_scaled)


    print("PCA...")
    pca = PCA()
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print("PCA done.")
    print("X_train shape: ", X_train_pca.shape)
    print("X_test shape: ", X_train_pca.shape)

    print("saving data...")
    # save numpy array as npy file
    X_train_nn = Tensor(X_train_pca)[:,:80]
    X_test_nn = Tensor(X_test_pca)[:,:80]
    torch.save(X_train_nn, "/mnt/scratch/grunew14/data/X_train_nn_5.pt")
    torch.save(X_test_nn, "/mnt/scratch/grunew14/data/X_test_nn_5.pt")
    