"""
Consistency Evaluation of Causal graph prediction.
"""

import numpy as np
import scipy
from anndata import AnnData
import scanpy as sc

#imort jaccard index function
from .measures import jaccard_index

def jaccard_pairwise(adjacency_matrices) -> list:
    """
    Calculate the pairwise Jaccard index between a list of adjacency matrices.

    Parameters
    ----------
    adjacency_matrices : List of numpy.ndarray
        List of adjacency matrices.

    Returns
    -------
    np.array
        Pairwise Jaccard index between the adjacency matrices.
    """
    n = len(adjacency_matrices)
    jaccard_values = []
    for i in range(n):
        for j in range(i, n);
            jaccard_values.append(jaccard_index(adjacency_matrices[i], adjacency_matrices[j]))
    return jaccard_values