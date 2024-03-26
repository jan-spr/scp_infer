"""
Statistical Evaluation of Causal graph prediction.

Functions for statistical evaluation of Causal Graph predictions.
Some of these are closely related to the CausalBench approaches, whose code is available at:
https://github.com/causalbench/causalbench
"""
import numpy as np
import scipy
from anndata import AnnData
import networkx as nx
import scanpy as sc


def get_observational(adata_obj: AnnData, child: str) -> np.array:
    """
    Return all the samples for gene "child" in cells where there was no perturbations

    Args:
        adata_obj: annotated Anndata object containing the expression matrix and interventions
        child: Gene name of child to get samples for

    Returns:
        np.array 1D-matrix of corresponding samples
    """
    observations = adata_obj[adata_obj.obs["perturbation"] == "non-targeting"]
    gene_index = adata_obj.var_names.get_loc(child)
    observations = observations[:, gene_index].X
    return np.reshape(observations, np.size(observations))


def get_interventional(adata_obj: AnnData, child: str, parent: str) -> np.array:
    """
    Return all the samples for gene "child" in cells where "parent" was perturbed

    Args:
        adata_obj: annotated Anndata object containing the expression matrix and interventions
        child: Gene name of child to get samples for
        parent: Gene name of gene that must have been perturbed

    Returns:
        np.array 1D-matrix of corresponding samples
    """
    observations = adata_obj[adata_obj.obs["perturbation"] == parent]
    gene_index = adata_obj.var_names.get_loc(child)
    observations = observations[:, gene_index].X
    return np.reshape(observations, np.size(observations))


def evaluate_wasserstein(
        adata_obj: AnnData,
        adjacency_matrix: np.array,
        p_value_threshold: float = 0.05
        ):
    """
    Evaluate the network's positive predictions using the observational and interventional data.

    Args:
        adata_obj: annotated Anndata object containing the expression matrix and interventions
        adjacency_matrix: The (binary) adjacency matrix of the network
        p_value_threshold: threshold for statistical significance, default 0.05

    Returns:
        true_positive: number of true positives
        false_positive: number of false positives
        wasserstein_distances: 
            list of wasserstein distances between observational and interventional samples
    """
    gene_names = adata_obj.var_names
    true_positive = 0
    false_positive = 0
    wasserstein_distances = []
    network_graph = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
    for parent in network_graph.nodes():
        #print("parent: ", parent)
        children = network_graph.successors(parent)
        for child in children:
            #print("child: ", child)
            #print("getting obs. samples")
            observational_samples = get_observational(adata_obj, gene_names[child])
            #print("obs. samples: ",np.shape(observational_samples))
            #print(observational_samples[:5])
            #print("getting int. samples")
            interventional_samples = \
                get_interventional(adata_obj, gene_names[child], gene_names[parent])
            #print("int. samples: ",np.shape(interventional_samples))
            #print(interventional_samples[:5])
            #print("ranking and whitney U test")
            if len(observational_samples) == 0 or len(interventional_samples) == 0:
                continue
            ranksum_result = scipy.stats.mannwhitneyu(
                observational_samples, interventional_samples
            )
            #print("getting wassertstein distance")
            wasserstein_distance = scipy.stats.wasserstein_distance(
                observational_samples, interventional_samples,
            )
            wasserstein_distances.append(wasserstein_distance)
            #print("wasserstein distance: ", wasserstein_distance)
            p_value = ranksum_result[1]
            if p_value < p_value_threshold:
                # Mannwhitney test rejects the hypothesis that the two distributions are similar
                # -> parent has an effect on the child
                true_positive += 1
            else:
                false_positive += 1
    return true_positive, false_positive, wasserstein_distances

def evaluate_f_o_r(adata_obj: AnnData, adjacency_matrix: np.array, p_value_threshold: float = 0.05):
    """
    Evaluate the network's positive predictions using the observational and interventional data.

    Args:
        adata_obj: annotated Anndata object containing the expression matrix and interventions
        adjacency_matrix: The (binary) adjacency matrix of the network
        p_value_threshold: threshold for statistical significance, default 0.05

    Returns:
        true_positive: number of true positives
        false_positive: number of false positives
        wasserstein_distances: list of wasserstein distances between 
            observational and interventional samples
    """
    network_graph = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
    tranclo_graph = nx.transitive_closure(network_graph)
    independent_pair_graph = nx.complement(tranclo_graph)
    unrelated_adj_matrix = nx.to_numpy_array(independent_pair_graph)

    #print("unrelated_adj_matrix: ", unrelated_adj_matrix)
    print("Evaluating Wasserstein")

    _, f_p, wasserstein = evaluate_wasserstein(adata_obj, unrelated_adj_matrix, p_value_threshold)

    false_omission_rate = f_p / independent_pair_graph.number_of_edges()
    negative_mean_wasserstein = np.mean(wasserstein)

    return false_omission_rate, negative_mean_wasserstein


def de_graph_hierarchy(
        adata_obj: AnnData,
        adjacency_matrix: np.array,
        verbose = False
        ):
    """
    idendify differentially expressed genes per perturbation and score whether they are
    placed upstream, downstream or unrelated to the perturbation in the network

    Args:
        adata_obj: annotated Anndata object containing the expression matrix and interventions
        adjacency_matrix: The (binary) adjacency matrix of the network

    Returns:
        upstream: number of true positives for upstream genes
        downstream: number of true positives for downstream genes
        unrelated: number of true positives for unrelated genes
    """
    perturbed_genes = adata_obj.var_names[adata_obj.var['gene_perturbed']]
    network_graph = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
    tranclo_graph = nx.transitive_closure(network_graph)

    # 1. compute DE genes for each perturbation with respect to rest (-> non-targeting?)
    # a. Add key to adata_obj.obs for perturbation groupings to be used in DE analysis
    adata_obj = adata_obj.copy()
    adata_obj = adata_obj[adata_obj.obs['gene_perturbation_mask'] | adata_obj.obs['non-targeting']]
    adata_obj.obs['perturbation_group'] = adata_obj.obs['perturbation']
    adata_obj.obs['perturbation_group'] = adata_obj.obs['perturbation_group'].astype('category')
    # perturbation group should only contain the perturbed genes and non-targeting

    # b. perform DE analysis
    key = 'rank_genes_perturbations'
    sc.tl.rank_genes_groups(adata_obj, groupby='perturbation_group', method='t-test', key_added=key) 
    reference = str(adata_obj.uns[key]["params"]["reference"])
    group_names = adata_obj.uns[key]["names"].dtype.names
    if verbose:
        print("perturbed_genes: ", perturbed_genes)
        print('reference:', reference)
        print('group_names:', group_names)

    # 2. compute the number of true positives
    upstream = 0
    downstream = 0
    unrelated = 0
    for perturbed_gene in group_names:
        if perturbed_gene == 'non-targeting':
            continue
        # get the DE genes for the perturbation
        perturbed_gene_index = adata_obj.var_names.get_loc(perturbed_gene)
        gene_names = adata_obj.uns[key]["names"][perturbed_gene]
        # remove the perturbed gene itself from DE genes
        gene_names = [gene for gene in gene_names if gene != perturbed_gene]
        # count where DE genes are located in the network
        for gene in gene_names:
            gene_index = adata_obj.var_names.get_loc(gene)
            if gene_index in tranclo_graph.successors(perturbed_gene_index):
                downstream += 1
            elif gene_index in tranclo_graph.predecessors(perturbed_gene_index):
                upstream += 1
            else:
                unrelated += 1

    return upstream, downstream, unrelated