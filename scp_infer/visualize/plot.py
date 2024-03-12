import numpy as np
import matplotlib.pyplot as plt


def plot_adjacency_matrix(estimate, title="GIES"):
    _, ax = plt.subplots()
    fig1 = ax.matshow(estimate)
    plt.colorbar(fig1)
    plt.title(title + ": Adjacency matrix")
    plt.savefig(title + "_adjacency_matrix.png")
    plt.plot()