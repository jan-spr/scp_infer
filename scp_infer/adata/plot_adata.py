import numpy as np

import matplotlib.pyplot as plt


def plot_perturb_vs_non(
        adata_obj,
        all_non_pert=True,
        non_pert_pert=True,
        pert_pert=True,
        xlim=None,
        ylim=None,
        filename=None
):
    """
    Plot the historgram of pertubed gene counts vs non-perturbed gene counts
    adata_obj: AnnData object
    """

    numpy_counts = adata_obj.X
    # check data type and convert to numpy array if necessary
    if not isinstance(numpy_counts, np.ndarray):
        numpy_counts = numpy_counts.toarray()

    total_entries = numpy_counts.size
    numpy_counts_1d = numpy_counts.flatten()
    # take 1st element if the array is 2d
    if len(numpy_counts_1d) != total_entries:
        numpy_counts_1d = numpy_counts_1d[0]
    print(np.shape(numpy_counts_1d))
    mask_1d = adata_obj.layers['perturbed_elem_mask'].flatten()
    print(np.shape(mask_1d))
    perturbed_counts = numpy_counts_1d[~mask_1d]
    all_non_perturbed_counts = numpy_counts_1d[mask_1d]

    # non perturbed counts for only pertubed genes:
    numpy_counts_pert_genes = numpy_counts[:, adata_obj.var['gene_perturbed']]
    mask_1d_pert_genes = adata_obj.layers['perturbed_elem_mask'][:,
                                                                 adata_obj.var['gene_perturbed']].flatten()
    non_perturbed_counts_pert_genes = numpy_counts_pert_genes.flatten()[
        mask_1d_pert_genes]

    # 2. plot histogram
    if all_non_pert:
        plt.hist(all_non_perturbed_counts, bins=100, alpha=0.5,
                 label='non-perturbed', density=True)
    if non_pert_pert:
        plt.hist(non_perturbed_counts_pert_genes, bins=100, alpha=0.5,
                 label='unperturbed target genes', density=True)
    if pert_pert:
        plt.hist(perturbed_counts, bins=100, alpha=0.5,
                 label='perturbed target genes', density=True)
    plt.legend(loc='upper right')
    plt.title('gene counts by perturbation state')

    if ylim is not None:
        plt.ylim(ylim)
    if xlim is not None:
        plt.xlim(xlim)
    if filename is not None:
        plt.savefig(filename)
    plt.show()


def print_expression_mean_std(adata_obj):
    # Check the expression of downregulated genes
    perturbed_gene_expression = np.array([])
    for observation in range(adata_obj.shape[0]):
        if adata_obj.obs['gene_perturbation_mask'].iloc[observation]:
            perturbed_gene = adata_obj.obs['perturbation'].iloc[observation]
            perturbed_gene_expression = np.append(
                perturbed_gene_expression, adata_obj.X[observation, adata_obj.var_names.get_loc(perturbed_gene)])

    print("")
    print("Perturbed Gene Expression:")
    print("Mean: ", np.mean(perturbed_gene_expression))
    print("Std: ",  np.std(perturbed_gene_expression))
    print("Min: ",  np.min(perturbed_gene_expression))
    print("Max: ",  np.max(perturbed_gene_expression))
    print("95% percentile: ", np.percentile(perturbed_gene_expression, 5),
          " - ", np.percentile(perturbed_gene_expression, 95))

    # Check the expression of non-targeting genes
    non_target_gene_expression = np.array([])
    for observation in range(adata_obj.shape[0]):
        if adata_obj.obs['non-targeting'].iloc[observation]:
            non_target_gene_expression = np.append(
                non_target_gene_expression, adata_obj.X[observation, np.random.randint(adata_obj.shape[1])])

    print("")
    print("Non-Target Gene Expression:")
    print("Mean: ", np.mean(non_target_gene_expression))
    print("Std: ",  np.std(non_target_gene_expression))
    print("Min: ",  np.min(non_target_gene_expression))
    print("Max: ",  np.max(non_target_gene_expression))
    print("95% percentile: ", np.percentile(non_target_gene_expression, 5),
          " - ", np.percentile(non_target_gene_expression, 95))
