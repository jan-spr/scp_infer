""" Operations on AnnData objects to annotate perturbations, filter genes and adjust counts """

import numpy as np
import scipy
from anndata import AnnData


def get_perturb_labels(
        adata_obj: AnnData,
        filter_genes: bool = True,
) -> AnnData:
    """
    Get the perturbation labels from the AnnData object
    And filter by multiplet, non-targeting and normal perturbations
    Store results in the AnnData object observation annotation
    adata_obj: AnnData object
    filter_genes: bool, whether to filter the genes that are not in the gene
        list
    Returns:
    adata_obj: AnnData object, with the observation annotation updated
    """
    # Get the perturbation labels
    perturb_labels = adata_obj.obs['perturbation'].astype(str).copy()

    perturb_labels_f = [label.split("_")[0] for label in perturb_labels]
    perturb_labels_simple = []
    for entry in perturb_labels_f:
        if ':' in entry:
            perturb_labels_simple.append('nan')
        else:
            perturb_labels_simple.append(entry)

    adata_obj.obs['perturbation'] = perturb_labels_simple
    adata_obj.obs['non-targeting'] = adata_obj.obs['perturbation'] == "non-targeting"
    adata_obj.obs['multiplet'] = adata_obj.obs['perturbation'] == "multiplet"
    adata_obj.obs['control'] = adata_obj.obs['perturbation'] == "control"
    adata_obj.obs['nan'] = adata_obj.obs['perturbation'] == "nan"
    adata_obj.obs['gene_pert'] = ~adata_obj.obs['non-targeting'] & ~adata_obj.obs['multiplet'] \
        & ~adata_obj.obs['control'] & ~adata_obj.obs['nan']

    adata_obj.obs['perturbation'] = adata_obj.obs['perturbation'].astype(
        'category')

    print("Non-targeting:", adata_obj.obs['non-targeting'].sum())
    print("Multiplet:", adata_obj.obs['multiplet'].sum())
    print("Control:", adata_obj.obs['control'].sum())
    print("Nan:", adata_obj.obs['nan'].sum())
    print("Normal pert.:", adata_obj.obs['gene_pert'].sum())

    if filter_genes:
        gene_in_var = [
            gene in adata_obj.var_names for gene in adata_obj.obs['perturbation']]
        gene_pert_f = gene_in_var & adata_obj.obs['gene_pert']
        print("Filtered", np.sum(adata_obj.obs['gene_pert'])-np.sum(
            gene_pert_f), "un-identifiable perturbations: ",
            np.sum(gene_pert_f), "filtered perturbations")
        adata_obj.obs['gene_pert'] = gene_pert_f

    adata_obj.var['gene_perturbed'] = adata_obj.var_names.isin(
        adata_obj.obs['perturbation'][adata_obj.obs['gene_pert']])

    # 1. Create a mask for the perturbed cases
    mask = np.zeros(adata_obj.shape, dtype=bool)
    for i in range(len(adata_obj.obs_names)):
        if adata_obj.obs['gene_pert'].iloc[i]:
            j = adata_obj.var_names.get_loc(
                adata_obj.obs['perturbation'].iloc[i])
            mask[i, j] = True
    mask = ~mask
    adata_obj.layers['perturbed_elem_mask'] = mask

    return adata_obj


def gene_labels_2_index(
        adata_obj: AnnData,
        gene_labels: list
) -> list:
    """
    Get the indices of the genes in the AnnData object
    adata_obj: AnnData object
    gene_labels: list, gene labels
    Returns:
    gene_indices: list, indices of the genes in the AnnData object
    """
    gene_indices = [adata_obj.var_names.get_loc(
        label) for label in gene_labels]
    return gene_indices


def create_data_matrix_gies(
        adata_obj: AnnData,
) -> tuple[list, np.array]:
    """
    Create the data matrix for the GIES algorithm
    adata_obj: AnnData object
    non_target_obs: list of observation indices of non-targeting interventions
    gene_pert_obs: list of observation indices of good gene perturbations
    Returns:
    data_matrix: np.array, data matrix for the GIES algorithm
    """

    # Step 1: Create Intervention Lis
    intervention_list = [[]]
    intervention_gene_names = ["non-targeting"]
    for i, var_name in enumerate(adata_obj.var_names):
        if adata_obj.var['gene_perturbed'].iloc[i]:
            intervention_list.append([i])
            intervention_gene_names.append(var_name)
    print("Intervention List created: ", len(
        intervention_list), "unique perturbations")

    # Step 2: Create Data Matrix
    # 1st create skeleton
    data_matrix = []
    for i in range(len(intervention_list)):
        data_matrix.append([])

    # 2nd fill in the data in index according to intervention list
    for i in range(len(adata_obj.obs_names)):
        if adata_obj.obs['non-targeting'].iloc[i]:
            data_matrix[0].append(adata_obj.X[i, :])

    for i in range(len(adata_obj.obs_names)):
        if adata_obj.obs['gene_pert'].iloc[i]:
            pert_gene_name = adata_obj.obs['perturbation'].iloc[i]
            intervention_i = intervention_gene_names.index(pert_gene_name)
            data_matrix[intervention_i].append(adata_obj.X[i, :])
    return intervention_list, data_matrix


def create_data_matrix_gies_singularized(adata_obj):
    """
    Create the data matrix for the GIES algorithm
    adata_obj: AnnData object
    non_target_obs: list of observation indices of non-targeting interventions
    gene_pert_obs: list of observation indices of good gene perturbations
    Returns:
    data_matrix: np.array, data matrix for the GIES algorithm
    """

    # Step 1: Create Intervention Lis
    intervention_list = []

    # Step 2: Create Data Matrix

    data_matrix = []

    for i in range(len(adata_obj.obs_names)):
        if adata_obj.obs['non-targeting'].iloc[i]:
            data_matrix.append([adata_obj.X[i, :]])
            intervention_list.append([])
        elif adata_obj.obs['gene_pert'].iloc[i]:
            pert_gene_name = adata_obj.obs['perturbation'].iloc[i]
            intervention_i = adata_obj.var_names.get_loc(pert_gene_name)
            data_matrix.append([adata_obj.X[i, :]])
            intervention_list.append([intervention_i])
    return intervention_list, data_matrix


def scale_counts(adata_obj, copy=False, max_value=10, verbose=False):
    """
    Scale the counts of the AnnData object to zero mean and unit variance per each gene
    + account for perturbed expression of genes:
    perturbed counts will be left out when derivating the scaling parameters, 
        but will be included in the scaling
    adata_obj: AnnData object
    adata_obj.X: sparse matrix scipy.sparse.csr.csr_matrix
    copy: bool, whether to return a copy of the AnnData object or to modify it in place
    max_value: float, maximum value for the scaled data
    Returns:
    adata_obj: AnnData object, with the scaled data if copy is False
    """
    if copy:
        adata_obj = adata_obj.copy()

    # 1. Create a mask for the perturbed cases
    mask = np.zeros(adata_obj.shape, dtype=bool)
    for i in range(len(adata_obj.obs_names)):
        if adata_obj.obs['gene_pert'].iloc[i]:
            j = adata_obj.var_names.get_loc(
                adata_obj.obs['perturbation'].iloc[i])
            mask[i, j] = True
    mask = ~mask
    adata_obj.layers['perturbed_elem_mask'] = mask

    # 2. Calculate the scaling parameters
    if scipy.sparse.issparse(adata_obj.X):
        print(np.shape(adata_obj.X.toarray()))
        mean = np.mean(adata_obj.X.toarray(), axis=1, where=mask)
        std = np.std(adata_obj.X.toarray(), axis=1, ddof=1, where=mask)
    else:
        mean = np.mean(np.array(adata_obj.X), axis=1, where=mask)
        std = np.std(adata_obj.X, axis=1, ddof=1, where=mask)
    # is this a unbiased estimator? check later

    print("mean shape: ", np.shape(mean))
    print("std shape: ", np.shape(std))

    # reshape arrays to be able to broadcast
    if verbose:
        print("Mean:", mean.shape)
        print("Std:", std.shape)
        print("X:", adata_obj.X.shape)
    mean = np.repeat(np.array([mean]).T, adata_obj.shape[1], axis=1)
    std = np.repeat(np.array([std]).T, adata_obj.shape[1], axis=1)

    # 3. Apply the scaling
    adata_obj.X = (adata_obj.X - mean) / std

    # 4. Clip the values
    if max_value is not None:
        adata_obj.X = np.clip(adata_obj.X, -max_value, max_value)

    adata_obj.X = np.asarray(adata_obj.X)

    if copy:
        return adata_obj
    else:
        return None
