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
    csv_file = None

    def __init__(self, datamanager):
        """
        Initialize the manager

        Currently wraps around the datamanager object
        -> maybe make this more elegant
        """
        self.datamanager = datamanager
        self.adata_obj = datamanager.adata_obj
        self.output_folder = datamanager.output_folder
        self.dataset_name = datamanager.dataset_name
        self.csv_file = os.path.join(
            self.output_folder, self.dataset_name, "evaluation_results.csv")
        # Load the Dataframe:
        if os.path.exists(self.csv_file):
            self.dataframe = self.load_evaluation_results()
        else:
            self.dataframe = pd.DataFrame(columns=self.dataframe_cols)

    def save_evaluation_results(self) -> None:
        """
        Save evaluation results in the appropriate folder
        """
        self.dataframe.to_csv(self.csv_file)

    def load_evaluation_results(self) -> pd.DataFrame:
        """
        Load evaluation results
        """
        return pd.read_csv(self.csv_file, index_col=0)

    def append_eval_result(self, results: list) -> None:
        """
        Append evaluation results to the dataframe
        """
        if np.shape(results)[1] != len(self.dataframe_cols):
            raise ValueError("Results do not match the dataframe columns")
        df = pd.DataFrame(results, columns=self.dataframe_cols)
        self.dataframe = pd.concat([self.dataframe, df], ignore_index=True)

    def load_inference_results(
            self,
            split_version="shuffled",
            model_name=None,
            split_label=None
    ) -> tuple:
        """
        Load inference results for a given split_version

        Returns:
        split_labels : List of str
            List of split labels
        adj_matrices : List of np.arrays
            List of adjacency matrices
        """
        return self.datamanager.load_inference_results(split_version, model_name, split_label)

    def evaluate_model(self, split_version, model_name, metric, split_labels=None, adj_cutoff = None):
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

        split_labels, split_datasets = \
            self.datamanager.get_train_test_splits(split_version, split_labels)
        split_labels, adj_matrices = \
            self.load_inference_results(split_version, model_name, split_labels)
        
        if adj_cutoff is not None:
            adj_matrices = [(adj_matrix > adj_cutoff).astype(int) for adj_matrix in adj_matrices]

        # 2. Filter the AnnData object for the test data:
        for split_label, split_data in zip(split_labels, split_datasets):
            # 2. Filter for test data:
            split_data = split_data[split_data.obs["set"] == "test"]

        # 3. Evaluate the model:
        for split_label, split_data, adj_matrix in zip(split_labels, split_datasets, adj_matrices):
            if metric == "wasserstein":
                # Evaluate the wasserstein distance
                tp, fp, wasserstein_distances = \
                    evaluate_wasserstein(split_data, adj_matrix, p_value_threshold=0.05)
                mean_wasserstein = np.mean(wasserstein_distances)
                # Save the results in the dataframe
                self.append_eval_result([[split_version, split_label, model_name, metric, mean_wasserstein]])
                self.append_eval_result([[split_version, split_label, model_name, "wasserstein_TP", tp]])
                self.append_eval_result([[split_version, split_label, model_name, "wasserstein_FP", fp]])

            elif metric == "false_omission_ratio":
                # Evaluate the false omission ratio
                f_o_r, neg_mean_wasserstein = evaluate_f_o_r(split_data, adj_matrix, p_value_threshold=0.05)
                # Save the results in the dataframe
                self.append_eval_result([[split_version, split_label, model_name, metric, f_o_r]])
                self.append_eval_result([[split_version, split_label, model_name,
                                        "negative_mean_wasserstein", neg_mean_wasserstein]])
            elif metric == "de_graph_hierarchy":
                # Evaluate the de-graph hierarchy
                n_upstr, n_downstr, n_unrel = de_graph_hierarchy(split_data, adj_matrix)
                # Save the results in the dataframe
                self.append_eval_result(\
                    [[split_version, split_label, model_name, "DE_n_upstream", n_upstr]])
                self.append_eval_result(\
                    [[split_version, split_label, model_name, "DE_n_downstream", n_downstr]])
                self.append_eval_result(\
                    [[split_version, split_label, model_name, "DE_n_unrelated", n_unrel]])
            else:
                raise ValueError("Metric not implemented")
        # Save the results
        self.save_evaluation_results()
