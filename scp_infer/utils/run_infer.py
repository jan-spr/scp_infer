"""scripts for handling running inference methods (on multiple datasets)"""

import os
import scanpy as sc
import numpy as np

import scp_infer as scpi
from scp_infer.utils.data_split import shuffled_split, gene_holdout  # , total_intervention_holdout


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
    if model_name == "GIES":
        model_imp = scpi.inference.GIESImp
    elif model_name == "DCDI":
        model_imp = scpi.inference.DCDIImp
    elif model_name == "GRNBoost2":
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
