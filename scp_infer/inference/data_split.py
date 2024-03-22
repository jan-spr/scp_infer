"""Splitting data into train, validation, and test sets - bzw. into train and hold-out"""
import numpy as np
import scipy
from anndata import AnnData

def random_split_data(
        adata_obj: AnnData,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        seed: int = 42
) -> None:
    """
    Split the data into train, validation, and test sets.
    Stores the respective assignment in the observation annotation "set"
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
    n_train = int(n_obs * train_frac)
    n_val = int(n_obs * val_frac)
    n_test = n_obs - n_train - n_val

    idx = np.random.permutation(n_obs)
    adata_obj.obs['set'] = 'train'
    adata_obj.obs.loc[idx[n_train:n_train + n_val], 'set'] = 'val'
    adata_obj.obs.loc[idx[n_train + n_val:], 'set'] = 'test'

    adata_obj.obs['set'] = adata_obj.obs['set'].astype('category')
    print("Train:", n_train)
    print("Validation:", n_val)
    print("Test:", n_test)
    return None

def gene_perturbation_hold_out(
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

def perturbation_proportion_hold_out(
        adata_obj: AnnData,
        hold_out_proportion: float,
        seed: int = 42
) -> None:
    """
    Hold out the data for a specific proportion of gene perturbations
    Stores the respective assignment in the observation annotation "set"
    adata_obj: AnnData object
    hold_out_proportion: float, proportion of gene perturbations to be held out
    seed: int, random seed
    Returns:
    None
    """
    raise NotImplementedError("Not implemented yet")