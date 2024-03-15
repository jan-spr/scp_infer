"""
Functionalities for handling graphs.
"""

import networkx as nx
import numpy as np

def get_adjacency_matrix_from_graph(graph: nx.DiGraph) -> np.array:
    """
    Return the adjacency matrix of the graph.

    Parameters
    ----------
    graph : networkx.DiGraph
        The graph.

    Returns
    -------
    numpy.ndarray
        The adjacency matrix of the graph.

    """
    return nx.to_numpy_array(graph)

def get_adjacency_matrix_from_dict(graph_as_dict: dict) -> np.array:
    """
    Return the adjacency matrix of the graph.

    Parameters
    ----------
    graph_as_dict : dict
        The graph as a dictionary.

    Returns
    -------
    numpy.ndarray
        The adjacency matrix of the graph.

    """
    n_nodes = len(graph_as_dict.keys())
    adjacency_matrix = np.zeros((n_nodes, n_nodes))
    for parent in graph_as_dict.keys():
        children = graph_as_dict[parent]
        for child in children:
            adjacency_matrix[parent, child] = 1
    return adjacency_matrix

def get_graph_from_adjacency_matrix(adjacency_matrix: np.array, weights = None) -> nx.DiGraph:
    """
    Return the graph from the adjacency matrix.

    Parameters
    ----------
    adjacency_matrix : numpy.ndarray
        The adjacency matrix (binary) of the graph.

    Returns
    -------
    networkx.DiGraph
        The graph.

    """
    if weights is not None:
        return nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph, edge_attribute=weights)
    else:
        return nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)