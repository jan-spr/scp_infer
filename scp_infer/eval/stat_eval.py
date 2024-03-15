"""
Statistical Evaluation of Causal graph prediction.

Functions for statistical evaluation of Causal Graph predictions.
Some of these are closely related to the CausalBench approaches, whose code is available at:
https://github.com/causalbench/causalbench

Evaluation module to quantitatively evaluate a network using held-out data.

Args:
    expression_matrix: a numpy matrix of expression data of size [nb_samples, nb_genes]
    interventions: a list of size [nb_samples] that indicates which gene has been perturb. 
            "non-targeting" means no gene has been perturbed (observational data)
    gene_names: name of the genes in the expression matrix
    p_value_threshold: threshold for statistical significance, default 0.05
"""
import numpy as np

def get_observational(self, child: str) -> np.array:
    """
    Return all the samples for gene "child" in cells where there was no perturbations

    Args:
        child: Gene name of child to get samples for

    Returns:
        np.array matrix of corresponding samples
    """
    return self.get_interventional(child, "non-targeting")


def get_interventional(self, child: str, parent: str) -> np.array:
    """
    Return all the samples for gene "child" in cells where "parent" was perturbed

    Args:
        child: Gene name of child to get samples for
        parent: Gene name of gene that must have been perturbed

    Returns:
        np.array matrix of corresponding samples
    """
    return self.expression_matrix[
        self.gene_to_interventions[parent], self.gene_to_index[child]
    ]
