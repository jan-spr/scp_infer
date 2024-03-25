"""scripts for handling running inference methods (on multiple datasets)"""

import os
import scanpy as sc
import numpy as np

import scp_infer as scpi
from scp_infer.utils.data_split import shuffled_split, gene_holdout  # , total_intervention_holdout


def save_split(adata, split_version, split_label, output_folder="../data/data_out") -> None:
    """
    Save the data split in the appropriate folder
    saves the annotated adata object in folder hierarchy:
    - dataset
        - split-version 1
            - split_label 1
            - split_label 2
            - ...

    Parameters
    ----------
    adata : AnnData
        Annotated expression data object from scanpy
        should be fully preprocessed and ready for inference
    split_version : str
        Version of the split
    split_label : str
        Label for the split
    output_folder : str
        Folder to save the files in

    Returns
    -------
    None

    """
    dataset_name = adata.uns['dataset_name']
    output_folder = os.path.join(output_folder, dataset_name, split_version, split_label)
    os.makedirs(output_folder, exist_ok=True)
    output_folder = os.path.join(output_folder, f"{split_label}.h5ad")
    sc.write(output_folder, adata)
    return None


def create_train_test_split(adata, split_version="shuffled", test_size=0.2, n_splits=1) -> None:
    """
    Create train test splits for a given adata object
    Save results in Folder Hierachy:

    Parameters
    ----------
    adata : AnnData
        Annotated expression data object from scanpy
        should be fully preprocessed and ready for inference
    split_version : str
        Version of the split
        shuffled: random split
        gene-holdout: holdout perturbation on specific genes
        total-intervention: holdout intervention by proportion on entire dataset
    test_size : float
        Size of the test set
    n_splits : int
        Number of splits to create

    Returns
    -------
    None
    """
    # Create a folder for the dataset
    dataset_name = adata.uns['dataset_name']
    os.makedirs(f"../data/data_out/{dataset_name}", exist_ok=True)

    # Create a folder for the split version
    os.makedirs(f"../data/data_out/{dataset_name}/{split_version}", exist_ok=True)

    # Create train test splits
    if split_version == "shuffled":
        for i in range(n_splits):
            split_label = f"split_{i}"
            shuffled_split(adata, test_frac=test_size)
            save_split(adata, split_version, split_label)
    elif split_version == "gene-holdout":
        for i, gene in enumerate(adata.var_names[adata.var['gene_perturbed']]):
            split_label = f"gene_{gene}"
            gene_holdout(adata, hold_out_gene=gene)
            save_split(adata, split_version, split_label)
    elif split_version == "total-intervention":
        raise NotImplementedError("Not implemented yet")
    else:
        raise ValueError("Invalid split version")

    return None


def run_inference(
        adata,
        split_version="shuffled",
        split_label=None,
        model_name=None,
        output_folder="../data/data_out"
) -> None:
    """
    Run inference on a given dataset split
    Save results in Folder Hierachy:

    Parameters
    ----------
    adata : AnnData
        Annotated expression data object from scanpy
        should be fully preprocessed and ready for inference
    split_version : str
        Version of the split
        shuffled: random split
        gene-holdout: holdout perturbation on specific genes
        total-intervention: holdout intervention by proportion on entire dataset
    split_label : str
        Label for the split
        if None, run inference on all splits
    model_name : str
        Name of the model to run inference
    output_folder : str
        Folder where files are stored

    Returns
    -------
    None
    """
    # Load the data split
    dataset_name = adata.uns['dataset_name']
    split_folder = os.path.join(output_folder, dataset_name, split_version)
    if split_label is not None:
        split_folder = [os.path.join(split_folder, split_label)]
    else:
        split_folder = [f for f in os.listdir(split_folder) \
                        if os.path.isdir(os.path.join(split_folder, f))]

    # Select the model
    if model_name is "GIES":
        model_imp = scpi.inference.GIESImp
    elif model_name is "DCDI":
        model_imp = scpi.inference.DCDIImp
    elif model_name is "GRNBoost2":
        model_imp = scpi.inference.GRNBoost2Imp
    else:
        raise ValueError("Invalid model name")

    for split in split_folder:
        # Create the output folder
        infer_out_folder = os.path.join(output_folder, \
                                        dataset_name, split_version, split, model_name)
        os.makedirs(infer_out_folder, exist_ok=True)
        # Load the anotated data & filter for training data
        adata = sc.read(os.path.join(split_folder, split, f"{split}.h5ad"))
        adata = adata[adata.obs['set'] == 'train']
        # Run inference and save the output
        model = model_imp(adata)
        model.convert_data()
        output_matrix = model.infer()
        scpi.eval.plot_adjacency_matrix(\
            output_matrix, title=model_name, output_folder=infer_out_folder)
        np.save(os.path.join(infer_out_folder, "adjacency_matrix.npy"), output_matrix)

    return None
