"""Implementation of measures for causal graph prediction."""
import numpy as np

def jaccard_index(adjacency1, adjacency2, cutoff1=0.5, cutoff2=0.5):
    """
    Calculate the Jaccard index between two adjacency matrices.
    
    Parameters
    ----------
    adjacency1 : numpy.ndarray
        The first adjacency matrix.
    adjacency2 : numpy.ndarray
        The second adjacency matrix.
    cutoff1 : float
        The cutoff value for the first adjacency matrix.
    cutoff2 : float
        The cutoff value for the second adjacency matrix.
    
    Returns
    -------
    float
        The Jaccard index between the two adjacency matrices.

    """
    adjacency1 = (adjacency1 > cutoff1).astype(int)
    adjacency2 = (adjacency2 > cutoff2).astype(int)
    intersection = np.sum(np.logical_and(adjacency1, adjacency2))
    union = np.sum(np.logical_or(adjacency1, adjacency2))
    return intersection / union
