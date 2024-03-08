"""Application of GRNBoost2 Algorithm (using arboreto)"""

import numpy as np
import pandas as pd
from arboreto.algo import grnboost2
import matplotlib.pyplot as plt

from ..inference_method import InferenceMethod


class GRNBoost2Imp(InferenceMethod):
    """
    GRNBoost2 implementation

    Attributes:
    adata_obj: AnnData
        Annotated expression data object from scanpy
    verbose: bool
    tf_names: pd.DataFrame
        TF/gene names used by GRNBosst2
    expression_data: pd.DataFrame
        expression data used by GRNBoost2

    """

    tf_names: pd.DataFrame
    expression_data: pd.DataFrame

    def convert_data(self):
        """convert adata entries into GRNBoost2 format"""
        # Load the TF names
        self.tf_names = self.adata_obj.var_names
        self.expression_data = self.adata_obj.to_df()

    def infer(
        self,
        plot: bool = False,
        **kwargs
    ) -> np.array:
        if self.verbose:
            print("Running GRNBoost2")

        # run algorithm
        network = grnboost2(expression_data=self.expression_data, verbose=self.verbose)

        if self.verbose:
            print("GRNBoost2 fnished")
            print("network shape: ", network.shape)
            network.head()

        num_genes = len(self.adata_obj.var_names)
        grnboost_matrix = np.zeros((num_genes, num_genes))
        for i in range(network.shape[0]):
            for ind1, gene_1 in enumerate(self.adata_obj.var_names):
                for ind2, gene_2 in enumerate(self.adata_obj.var_names):
                    if network['TF'].iloc[i] == gene_1:
                        if network['target'].iloc[i] == gene_2:
                            grnboost_matrix[ind1, ind2] = network['importance'].iloc[i]

        if plot or self.verbose:
            _, ax = plt.subplots()
            fig1 = ax.matshow(grnboost_matrix)
            plt.colorbar(fig1)
            plt.title("GRNBoost2: Adjacency matrix")
            plt.savefig("GRNBoost2_adjacency_matrix.png")
            plt.plot()
