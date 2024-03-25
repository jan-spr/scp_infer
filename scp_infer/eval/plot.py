"""Plotting functions for Inference Outputs"""

import os
import matplotlib.pyplot as plt


def plot_adjacency_matrix(
        estimate,
        title="GIES",
        output_folder="../data/data_out"
) -> None:
    """
    Plot the adjacency matrix of the estimated graph.

    Parameters
    ----------
    estimate : numpy.ndarray
        The estimated adjacency matrix.
    title : str
        The title of the plot.

    Returns
    -------
    None
    """
    _, ax = plt.subplots()
    fig1 = ax.matshow(estimate)
    plt.colorbar(fig1)
    plt.title(title + ": Adjacency matrix")
    plt.savefig(os.path.join(output_folder, title, "adjacency_matrix.png"))
    plt.plot()
