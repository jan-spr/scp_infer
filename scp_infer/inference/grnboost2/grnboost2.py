"""Application of GRNBoost2 Algorithm (using arboreto)"""

import numpy as np
import pandas as pd
from anndata import AnnData
from arboreto.algo import grnboost2

from scp_infer.inference import InferenceMethod

"""
run_GRNBoost = False
if run_GRNBoost:
    from arboreto.algo import grnboost2
    from arboreto.utils import load_tf_names

    # Load the TF names
    tf_names = adata.var_names

    # Compute the GRN
    print("Running GRNBoost2")
    network = grnboost2(expression_data=adata.to_df(),verbose = True)
    print("GRNBoost2 fnished")
    print("network shape: ", network.shape)

if run_GRNBoost:
    network.head()
    GRNBoost_matrix = np.zeros((10,10))
    for i in range(90):
        for ind1, gene_1 in enumerate(adata.var_names):
            for ind2, gene_2 in enumerate(adata.var_names):
                if network['TF'].iloc[i] == gene_1:
                    if network['target'].iloc[i] == gene_2:
                            GRNBoost_matrix[ind1,ind2] = network['importance'].iloc[i]
    fig, ax = plt.subplots()
    fig1 = ax.matshow(GRNBoost_matrix)
    plt.colorbar(fig1)
    plt.title("GRNBoost2: Adjacency matrix")
    plt.plot()
"""

class GRNBoost2_imp(InferenceMethod):
