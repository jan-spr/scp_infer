"""
Application of GIES Algorithm

GIES implementation from: https://github.com/juangamella/gies
Slightly modified files from the original used
they should be stored in scp-infer/algorithm_implementations/gies_local

"""
import os
import sys


import numpy as np
from ..inference_method import InferenceMethod


# 1. the local gies algorithm has to be loaded

current_dir = os.path.abspath(".")

print("Current dir: ", current_dir)
sys.path.append(os.path.join(current_dir, 'algorithm_implementations'))
print(sys.path)
import gies_local as gies


class GIESImp(InferenceMethod):
    """
    GIES implementation

    Attributes:
    adata_obj: AnnData
        Annotated expression data object from scanpy
    verbose: bool


    """

    data_matrix: np.array
    intervention_list: list

    def __create_data_matrix_gies(self, verbose = False):
        """
        Create the data matrix for the GIES algorithm
        shape: (n_interventions, n_observations/intervention, n_features)
        here: take minimum number of observations/intervention and discard all else
        adata_obj: AnnData object
        Returns:
            intervention_list: list, list of list of indices of the perturbed genes per intervention
            data_matrix: list, data matrix for the GIES algorithm
                each entry is a np.array containing the data for one intervention
        """
        adata_obj = self.adata_obj

        # Step 1: Create Intervention Lis
        intervention_list = [[]]
        intervention_gene_names = ["non-targeting"]
        for i, var_name in enumerate(adata_obj.var_names):
            if adata_obj.var['gene_perturbed'].iloc[i]:
                intervention_list.append([i])
                intervention_gene_names.append(var_name)
        if verbose:
            print("Intervention List created: ", len(
                intervention_list), "unique perturbations")

        # Step 2: Create Data Matrix
        # 1st create skeleton
        data_matrix = []
        for i in range(len(intervention_list)):
            data_matrix.append([])

        # 2nd fill in the data in index according to intervention list
        # 2.1 fill in non-targeting as observatonal data (index 0)
        for i in range(len(adata_obj.obs_names)):
            if adata_obj.obs['non-targeting'].iloc[i]:
                data_matrix[0].append(adata_obj.X[i, :])

        # 2.2 fill in the gene perturbations
        for i in range(len(adata_obj.obs_names)):
            if adata_obj.obs['gene_perturbation_mask'].iloc[i]:
                pert_gene_name = adata_obj.obs['perturbation'].iloc[i]
                intervention_i = intervention_gene_names.index(pert_gene_name)
                data_matrix[intervention_i].append(adata_obj.X[i, :])

        # 2.3 turn each list entry into numpy matrix
        for i, dm_elem in enumerate(data_matrix):
            data_matrix[i] = np.array(dm_elem, dtype=float)

        return intervention_list, data_matrix

    def __create_data_matrix_gies_singularized(self):
        """
        !!!DEPRECATED!!!
        Create the data matrix for the GIES algorithm
        shape: (n_interventions, n_observations/intervention, n_features)
        here: 
            n_observations/intervention == 1
            n_interventions == n_observations
            each observation gets stored under separate intervention entry
        adata_obj: AnnData object
        Returns:
            data_matrix: np.array, data matrix for the GIES algorithm
        """
        adata_obj = self.adata_obj

        # Step 1: Create Intervention Lis
        intervention_list = []

        # Step 2: Create Data Matrix
        data_matrix = []

        for i in range(len(adata_obj.obs_names)):
            if adata_obj.obs['non-targeting'].iloc[i]:
                data_matrix.append([adata_obj.X[i, :]])
                intervention_list.append([])
            elif adata_obj.obs['gene_perturbation_mask'].iloc[i]:
                pert_gene_name = adata_obj.obs['perturbation'].iloc[i]
                intervention_i = adata_obj.var_names.get_loc(pert_gene_name)
                data_matrix.append([adata_obj.X[i, :]])
                intervention_list.append([intervention_i])
        data_matrix = np.array(data_matrix, dtype=float)

        return intervention_list, data_matrix

    def convert_data(self, singularized: bool = False):
        """
        convert adata entries into GIES format

        Singularized: bool
            if True, each observation gets stored under separate intervention entry
            if False, store all observations for each intervention in one entry 
                (default - list of np.arrays)
        """
        # Load the data matrix
        if self.verbose:
            print("Converting data to GIES format")
        if singularized:
            self.intervention_list, self.data_matrix = self.__create_data_matrix_gies_singularized()
        else:
            self.intervention_list, self.data_matrix = self.__create_data_matrix_gies()

        if self.verbose:
            # Look at results
            print(self.adata_obj.obs['gene_perturbation_mask'].sum(), " gene perturbations")
            print(len(self.intervention_list), " interventions")
            print("Intervention list: ", self.intervention_list[:15])

            print("")
            print("Data matrix:")
            print("Length of data matrix: ", len(self.data_matrix))

            length = np.array([])
            for sub_array in self.data_matrix:
                length = np.append(length, len(sub_array))

            print("Minimum length: ", np.min(length))
            print("Maximum length: ", np.max(length))
            print("Average length: ", np.mean(length))
            print("Total Samples: ", np.sum(length))
            print("Total interventional Samples: ", np.sum(length[1:]))

            print("Entries per Intervention: ", length)

            #print("GIES final data shape: ", np.shape(self.data_matrix))

    def infer(
        self,
        plot: bool = False,
        **kwargs
    ) -> np.array:
        """
        perform inference

        Returns:
        estimate: np.array, adjacency matrix estimate
        score: float, BIC score
        """
        if self.verbose:
            print("Running GIES")

        # run algorithm
        estimate, _ = gies.fit_bic(self.data_matrix, self.intervention_list, A0=None)

        if self.verbose:
            print("GIES fnished")
            print("estimate shape: ", estimate)
            print("GIES matrix: ", estimate)

        return estimate
