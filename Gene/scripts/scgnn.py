import numpy as np
import matplotlib.pyplot as plt
import anndata as ad
import pandas as pd
import scanpy as sc
import argparse
from torch import Tensor
import random
import magic
import pickle
import umap

from dance.modules.multi_modality.predict_modality import BabelWrapper
from dance.modules.single_modality.imputation.scgnn2 import ScGNN2


if __name__ == "__main__":
    
    print("Loading in Data")
    X_train_gex_atd = ad.read_h5ad('/mnt/scratch/grunew14/data/teain_gex_proccessed_debatched.h5ad')
    X_test = ad.read_h5ad('/mnt/scratch/grunew14/data/Gex_processed_testing.h5ad')
    train_adt = ad.read_h5ad("/mnt/scratch/grunew14/data/Adt_processed_training.h5ad")
    y_test = ad.read_h5ad('/mnt/scratch/grunew14/data/Adt_processed_testing.h5ad')
    
    print("Setting up argparse")
    args_imputation = argparse.Namespace()
    args_imputation.total_epoch = 31
    args_imputation.device = 'cuda:0'
    args_imputation.ari_threshold = .95
    args_imputation.alpha = 0.5
    args_imputation.graph_change_threshold = 0.01
    args_imputation.feature_AE_batch_size = 12800
    args_imputation.feature_AE_epoch = [100, 100]
    args_imputation.feature_AE_learning_rate = 1e-3
    args_imputation.feature_AE_regu_strength = .9
    args_imputation.feature_AE_dropout_prob = 0
    args_imputation.feature_AE_concat_prev_embed = None
    args_imputation.graph_AE_use_GAT = False
    args_imputation.graph_AE_GAT_dropout = False
    args_imputation.graph_AE_learning_rate = 1e-2
    args_imputation.graph_AE_embedding_size = 16
    args_imputation.graph_AE_concat_prev_embed = False
    args_imputation.graph_AE_normalize_embed = None
    args_imputation.graph_AE_graph_construction = 'v2'
    args_imputation.graph_AE_neighborhood_factor = 0.05
    args_imputation.graph_AE_retain_weights = False
    args_imputation.gat_multi_heads = 2
    args_imputation.gat_hid_embed = 64
    args_imputation.graph_AE_epoch = 200
    args_imputation.graph_AE_use_GAT = False
    args_imputation.clustering_louvain_only = False
    args_imputation.clustering_use_flexible_k = False
    args_imputation.clustering_embed = "graph"
    args_imputation.clustering_method = "AffinityPropagation"
    args_imputation.cluster_AE_batch_size = 12800
    args_imputation.cluster_AE_learning_rate = 1e-3
    args_imputation.cluster_AE_regu_strength = .9
    args_imputation.cluster_AE_dropout_prob = 0
    args_imputation.cluster_AE_epoch = 200
    args_imputation.deconv_opt1_learning_rate = 1e-3
    args_imputation.deconv_opt1_epoch = 5000
    args_imputation.deconv_opt1_epsilon = 1e-4
    args_imputation.deconv_opt1_regu_strength = 1e-2
    args_imputation.deconv_opt2_learning_rate = 1e-1
    args_imputation.deconv_opt2_epoch = 500
    args_imputation.deconv_opt2_epsilon = 1e-4
    args_imputation.deconv_opt2_regu_strength = 1e-2
    args_imputation.deconv_opt3_learning_rate = 1e-1
    args_imputation.deconv_opt3_epoch = 150
    args_imputation.deconv_opt3_epsilon = 1e-4
    args_imputation.deconv_opt3_regu_strength_1 = 0.8
    args_imputation.deconv_opt3_regu_strength_2 = 1e-2
    args_imputation.deconv_opt3_regu_strength_3 = 1
    args_imputation.deconv_tune_learning_rate = 1e-2
    args_imputation.deconv_tune_epoch = 20
    args_imputation.deconv_tune_epsilon = 1e-4
    args_imputation.seed = 1
    
    scgnn = ScGNN2(args_imputation)
    print("scgnn object initialized...")
    print("Now training...")
    
    X_train_arr = X_train_gex_atd.X.toarray()[:int(X_train_gex_atd.shape[0]),:]
    y_train_tensor = Tensor(train_adt.X.toarray())[:int(train_adt.shape[0]),:]
    
    scgnn.fit(X_train_arr)
    
    print("Saving File")
    with open('../trained_models/scgnnTrain4.pkl', 'wb') as f:
        pickle.dump(scgnn, f)
        f.close()
    print("done")
    