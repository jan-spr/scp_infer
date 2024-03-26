"""
Class to use for evaluation of model prediction and management of results

A:
- Evaluate model predictions: each adj-matrix x each metric (+ negative control)
- Save evaluation results: pd.DataFrame?

B:
- Load evaluation results
- Compare & Plot results
"""

import os
import pandas as pd
import numpy as np

from scp_infer.eval.stat_eval import evaluate_wasserstein, evaluate_f_o_r, de_graph_hierarchy


class EvalManager():
    """
    Evaluation utilities for a given dataset

    A:
    - load inference results & test data
    - Evaluate model predictions: each adj-matrix(test dat.) x each metric (+ negative control)
    - Save evaluation results: pd.DataFrame?

    B:
    - Load evaluation results
    - Compare & Plot results
    """
    output_folder = "../data/data_out"
    dataset_name = None
    dataframe = None
    dataframe_cols = ["split-version", "split-label", "model-name", "metric", "value"]

    def __init__(self, DataManager):
        """
        Initialize the manager

        Currently wraps around the DataManager object
        -> maybe make this more elegant
        """
        self.DataManager = DataManager
        self.adata_obj = DataManager.adata_obj
        self.output_folder = DataManager.output_folder
        self.dataset_name = DataManager.dataset_name
        #Load the Dataframe:
        if os.path.exists(os.path.join(self.output_folder, self.dataset_name, "evaluation_results.csv")):
            self.dataframe = self.load_evaluation_results()
        else:
            self.dataframe = pd.DataFrame(columns=self.dataframe_cols)

    def save_evaluation_results(self) -> None:
        """
        Save evaluation results in the appropriate folder
        """
        self.dataframe.to_csv(os.path.join(self.output_folder, self.dataset_name, "evaluation_results.csv"))

    def load_evaluation_results(self) -> pd.DataFrame:
        """
        Load evaluation results
        """
        return pd.read_csv(os.path.join(self.output_folder, self.dataset_name, "evaluation_results.csv"))

    def append_eval_result(self, results: list) -> None:
        """
        Append evaluation results to the dataframe
        """
        if len(results) != len(self.dataframe_cols):
            raise ValueError("Results do not match the dataframe columns")
        df = pd.DataFrame(results, columns=self.dataframe_cols)
        self.dataframe = self.dataframe.append(df)

    def load_inference_results(self, split_version="shuffled", model_name=None, split_label=None) -> tuple:
        """
        Load inference results for a given split_version

        Returns:
        split_labels : List of str
            List of split labels
        adj_matrices : List of np.arrays
            List of adjacency matrices
        """
        return self.DataManager.load_inference_results(split_version, model_name, split_label)
    
    def evaluate_model(self, split_version, model_name, metric, split_labels=None):
        """
        Evaluate model predictions: each adj-matrix x each metric (+ negative control)

        Parameters:
        split_version : str
            Version of the split
        model_name : str
            Name of the model to evaluate
        split_labels : List of str
            List of split labels
            if None - load all found in the folder

        metric: str
            Name of the metric to evaluate
            - e.g. "wasserstein", "false_omission_ratio", "de_graph_hierarchy"
        """
        # 1. Load the test data and inference results:

        split_labels, split_datasets = self.DataManager.get_train_test_splits(split_version, split_labels)
        split_labels, adj_matrices = self.load_inference_results(split_version, model_name, split_labels)

        # 2. Filter the AnnData object for the test data:
        for split_label, split_data in zip(split_labels, split_datasets):
            # 2. Filter for test data:
            split_data = split_data[split_data.obs["set"] == "test"]

        # 3. Evaluate the model:
        for split_label, split_data in zip(split_labels, split_datasets):
            if metric == "wasserstein":
                # Evaluate the wasserstein distance
                TP, FP, wasserstein_distances = evaluate_wasserstein(split_data, adj_matrices, p_value_threshold=0.05)
                mean_wasserstein = np.mean(wasserstein_distances)
                # Save the results in the dataframe
                self.append_eval_result([[split_version, split_label, model_name, metric, mean_wasserstein]])
                self.append_eval_result([[split_version, split_label, model_name, "wasserstein_TP", TP]])
                self.append_eval_result([[split_version, split_label, model_name, "wasserstein_FP", FP]])
                
            elif metric == "false_omission_ratio":
                # Evaluate the false omission ratio
                FOR, neg_mean_wasserstein = evaluate_f_o_r(split_data, adj_matrices, p_value_threshold=0.05)
                # Save the results in the dataframe
                self.append_eval_result([[split_version, split_label, model_name, metric, FOR]])
                self.append_eval_result([[split_version, split_label, model_name, "negative_mean_wasserstein", neg_mean_wasserstein]])
            elif metric == "de_graph_hierarchy":
                # Evaluate the de-graph hierarchy
                n_upstr, n_downstr, n_unrel = de_graph_hierarchy(split_data, adj_matrices)
                # Save the results in the dataframe
                self.append_eval_result([[split_version, split_label, model_name, "DE_n_upstream", n_upstr]])
                self.append_eval_result([[split_version, split_label, model_name, "DE_n_downstream", n_downstr]])
                self.append_eval_result([[split_version, split_label, model_name, "DE_n_unrelated", n_unrel]])
            else:
                raise ValueError("Metric not implemented")
        # Save the results
        self.save_evaluation_results()
