"""
Data management utilities to accomodate for:
different datasets, holdout strategies, splits and inference methods.
"""

import os
import scanpy as sc
import numpy as np

from scp_infer.utils import shuffled_split, gene_holdout  # , total_intervention_holdout


class ScpiDataManager():
    """
    Data management utilities for a given dataset

    A:
    - create multiple train test splits
    - load and save adata with split information:
        split_version, split_label
        -> for inferene and evaluation

    B:
    - load data for inference
    - store inference results

    C:
    - load data for evaluation
    - store evaluation results
    """
    output_folder = "../data/data_out"
    dataset_name = None

    def __init__(self, adata_obj, dataset_name, output_folder="../data/data_out"):
        """
        Initialize the data manager

        Parameters
        ----------
        adata_obj : AnnData
            Annotated expression data object from scanpy
            should be fully preprocessed and ready for inference
        dataset_name : str
            Name of the dataset - where files wil be stored
        output_folder : str
            Base Location to save the files in
        """
        self.adata_obj = adata_obj
        self.output_folder = output_folder
        self.dataset_name = dataset_name

    def split_ver_folder(self, split_version):
        """
        Get the folder for a given split version for this dataset
        """
        return os.path.join(self.output_folder, self.dataset_name, split_version)

    def save_split(self, adata, split_version, split_label) -> None:
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

        save_folder = os.path.join(self.output_folder, \
                                   self.dataset_name, split_version, split_label)
        os.makedirs(save_folder, exist_ok=True)
        save_file = os.path.join(save_folder, f"{split_label}.h5ad")
        sc.write(save_file, adata)
        return None

    def store_train_test_split(self, split_version="shuffled", test_size=0.2, n_splits=1) -> None:
        """
        Create train test splits for a given split_version
        Save results in Folder Hierachy:
        - dataset_name
            - split-version
                - split_label 1
                - split_label 2
                - ...

        Parameters
        ----------
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
        adata = self.adata_obj
        # Create a folder for the dataset
        dataset_name = self.dataset_name
        os.makedirs(os.path.join(self.output_folder, dataset_name), exist_ok=True)

        # Create a folder for the split version
        os.makedirs(os.path.join(self.output_folder, dataset_name, split_version), exist_ok=True)

        # Create train test splits
        if split_version == "shuffled":
            for i in range(n_splits):
                # label with test ratio and index
                split_label = f"%.f_test_split_{i}" % (test_size*100)
                shuffled_split(adata, test_frac=test_size)
                self.save_split(adata, split_version, split_label)
        elif split_version == "gene-holdout":
            for i, gene in enumerate(adata.var_names[adata.var['gene_perturbed']]):
                split_label = f"gene_{gene}"
                gene_holdout(adata, hold_out_gene=gene)
                self.save_split(adata, split_version, split_label)
        elif split_version == "total-intervention":
            raise NotImplementedError("Not implemented yet")
        else:
            raise ValueError("Invalid split version")

        return None

    def get_train_test_splits(self, split_version="shuffled", split_label=None):
        """
        Load a train test split dataset for a given split_version

        Parameters
        ----------
        split_version : str
            Version of the split
            shuffled: random split
            gene-holdout: holdout perturbation on specific genes
            total-intervention: holdout intervention by proportion on entire dataset
        split_label : str
            Label of the split
            if None - load all splits

        Returns
        -------
        split_labels: List of str
            List of split labels
        split_datasets: List of AnnData objects
            List of train test split datasets
        """

        split_version_folder = self.split_ver_folder(split_version)
        # store split_label entries in array:
        if split_label is None:
            split_label = os.listdir(split_version_folder).sort()
        else:
            split_label = [split_label]

        split_datasets = []
        for label in split_label:
            split_datasets.append(sc.read(os.path.join(split_version_folder, label + ".h5ad")))

        return split_label, split_datasets

    def store_inference_results(
            self,
            split_labels,
            adj_matrices,
            split_version="shuffled",
            model_name=None
    ) -> None:
        """
        Store inference results for a given split_version
        Save results in Folder Hierachy:
        - dataset_name
            - split-version
                - split_label
                    - model_name

        Parameters
        ----------
        split_labels : List of str
            List of split labels
        adj_matrix : np.array
            Adjacency matrix from the model (for each split label)
        split_version : str
            Version of the split
        split_label : str
            Label of the split
        model_name : str
            Name of the model to run inference

        Returns
        -------
        None
        """
        split_version_folder = self.split_ver_folder(split_version)
        if model_name is None:
            raise ValueError("Model name not provided")

        for label, adj_matrix in zip(split_labels, adj_matrices):
            model_output_folder = os.path.join(split_version_folder, label, model_name)
            os.makedirs(model_output_folder, exist_ok=True)
            np.save(os.path.join(model_output_folder, model_name + "_adj_matrix.npy"), adj_matrix)
