"""Splitting data into train, validation, and test sets - bzw. into train and hold-out"""
import numpy as np
from anndata import AnnData


def shuffled_split(
        adata_obj: AnnData,
        #train_frac: float = 0.8,
        # val_frac: float = 0.15,
        test_frac: float = 0.2,
        seed: int = 42,
        verbose=False
) -> None:
    """
    Split the data into train, validation, and test sets.
    adata_obj: AnnData object
    train_frac: float, fraction of the data to be used for training
    val_frac: float, fraction of the data to be used for validation
    test_frac: float, fraction of the data to be used for testing
    seed: int, random seed
    Returns:
    None
    """
    np.random.seed(seed)
    n_obs = adata_obj.n_obs
    n_test = int(n_obs * test_frac)
    n_train = n_obs - n_test

    set_list = ['train'] * n_train + ['test'] * n_test
    np.random.shuffle(set_list)
    adata_obj.obs['set'] = set_list
    adata_obj.obs['set'] = adata_obj.obs['set'].astype('category')
    if verbose:
        print("Train:", n_train)
        # print("Validation:", n_val)
        print("Test:", n_test)
    return None


def gene_holdout(
        adata_obj: AnnData,
        hold_out_gene: str,
) -> None:
    """
    Hold out the data for a specific gene perturbation
    Stores the respective assignment in the observation annotation "set"
    adata_obj: AnnData object
    hold_out_gene: str, gene perturbation to be held out
    Returns:
    None
    """
    adata_obj.obs['set'] = 'train'
    adata_obj.obs.loc[adata_obj.obs['perturbation'] == hold_out_gene, 'set'] = 'hold-out'
    adata_obj.obs['set'] = adata_obj.obs['set'].astype('category')
    return None


def total_intervention_holdout(
        adata_obj: AnnData,
        hold_out_proportion: float,
        seed: int = 42
) -> None:
    """
    Hold out the data for a proportion of total gene interventions
    Stores the respective assignment in the observation annotation "set"
    adata_obj: AnnData object
    hold_out_proportion: float, proportion of gene perturbations to be held out
    seed: int, random seed
    Returns:
    None
    """
    raise NotImplementedError("Not implemented yet")
