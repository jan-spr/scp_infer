"""Inherited Data Manager class (from dcdi DataManagerFile) to load data from anndata object"""

import torch
import numpy as np


from dcdi_master.dcdi.data import DataManagerFile



class DataManagerAnndata(DataManagerFile):
    """
    A data loader, it can load data, split in train/test, shuffle, normalize, etc.
    NOTE: the 0-th regime should always be the observational one
    """
    def __init__(self, file_path, adata_obj, train_samples=0.8, test_samples=None, train=True,
                 normalize=False, mean=None, std=None, random_seed=42, intervention=False,
                 intervention_knowledge="known", dcd=False):
        """
        :param anndata adata_obj: Anndata object containing the data
        :param int i_dataset: Exemplar to use (usually in [1,10])
        :param float/int train_samples: default=0.8. If float, specifies the proportion of
            data used for training and the rest is used for testing. If an integer, specifies
            the exact number of examples to use for training.
        :param int test_samples: default=None. Specifies the number of examples to use for testing.
            The default value uses all examples that are not used for training.
        :param int random_seed: Random seed to use for data set shuffling and splitting
        :param boolean intervention: If True, use interventional data with interventional targets
        :param str intervention_knowledge: Determine if the intervention target are known or unknown
        :param boolean dcd: If True, use the baseline DCD that use interventional data, but
            with a loss that doesn't take it into account (intervention should be set to False)
        :param list regimes_to_ignore: Regimes that are ignored during training
        """
        self.random = np.random.RandomState(random_seed)
        self.dcd = dcd
        self.adata_obj = adata_obj
        self.file_path = file_path
        #self.i_dataset = i_dataset
        self.intervention = intervention
        if intervention_knowledge == "known":
            self.interv_known = True
        elif intervention_knowledge == "unknown":
            self.interv_known = False
        else:
            raise ValueError("intervention_knowledge should either be 'known' \
                             or 'unknown'")

        data, masks, regimes = self.load_data()

        # index of all regimes, even if not used in the regimes_to_ignore case
        self.all_regimes = np.unique(regimes)


        # Determine train/test partitioning
        if isinstance(train_samples, float):
            train_samples = int(data.shape[0] * train_samples)
        if test_samples is None:
            test_samples = data.shape[0] - train_samples
        assert train_samples + test_samples <= data.shape[0], "The " + \
            "number of examples to load must be " + \
            "smaller than the total size of the dataset"

        # Shuffle and filter examples
        shuffle_idx = np.arange(data.shape[0])
        self.random.shuffle(shuffle_idx)
        data = data[shuffle_idx[: train_samples + test_samples]]
        if intervention:
            masks = [masks[i] for i in shuffle_idx[: train_samples + test_samples]]
        regimes = regimes[shuffle_idx[: train_samples + test_samples]]

        # Train/test split
        if not train:
            if train_samples == data.shape[0]: # i.e. no test set
                self.dataset = None
                self.masks = None
                self.regimes = None
            else:
                self.dataset = torch.as_tensor(
                    data[train_samples: train_samples + test_samples]
                    ).type(torch.Tensor)
                if intervention:
                    self.masks = masks[train_samples: train_samples + test_samples]
                self.regimes = regimes[train_samples: train_samples + test_samples]
        else:
            self.dataset = torch.as_tensor(data[: train_samples]).type(torch.Tensor)
            if intervention:
                self.masks = masks[: train_samples]
            self.regimes = regimes[: train_samples]

        # Normalize data
        self.mean, self.std = mean, std
        if normalize:
            if self.mean is None or self.std is None:
                self.mean = torch.mean(self.dataset, 0, keepdim=True)
                self.std = torch.std(self.dataset, 0, keepdim=True)
            self.dataset = (self.dataset - self.mean) / self.std

        self.num_regimes = np.unique(self.regimes).shape[0]
        self.num_samples = self.dataset.size(0)
        self.dim = self.dataset.size(1)

        self.initialize_interv_matrix()


    def load_data(self):
        """
        Load the graph, mask, regimes, and data
        """
        # Load the graph
        adjacency = np.zeros((self.adata_obj.shape[1], self.adata_obj.shape[1]))
        self.adjacency = torch.as_tensor(adjacency).type(torch.Tensor)

        # Load data
        #self.data_path = os.path.join(self.file_path, name_data)
        data = self.adata_obj.to_df().to_numpy()

        # Load intervention masks and regimes
        masks = [] # list of index's of intervened genes
        # read masks from anndata object
        for i in range(self.adata_obj.shape[0]):
            masks.append(self.adata_obj.layers['perturbed_elem_mask'][i, :])



        regimes = np.array(self.calc_regimes())

        return data, masks, regimes

    def initialize_interv_matrix(self):
        """
        Generate the intervention matrix I*. It is useful in the unknown case
        to compare learned target to the ground truth
        """
        if self.intervention:
            interv_matrix = self.adata_obj.layers['perturbed_elem_mask']
            interv_matrix = np.array(interv_matrix)
            interv_matrix = np.where(interv_matrix == 0, 1, 0)
            interv_matrix = torch.as_tensor(interv_matrix)

            regimes = np.sort(np.unique(self.regimes))

            self.gt_interv = 1 - interv_matrix
        else:
            self.gt_interv = None

    def sample(self, batch_size):
        """
        Sample without replacement `batch_size` examples from the data and
        return the corresponding masks and regimes
        :param int batch_size: number of samples to sample
        :return: samples, masks, regimes
        """
        sample_idxs = self.random.choice(np.arange(int(self.num_samples)), size=(int(batch_size),), replace=False)
        samples = self.dataset[torch.as_tensor(sample_idxs).long()]
        if self.intervention:
            masks = self.convert_masks(sample_idxs)
            regimes = self.regimes[torch.as_tensor(sample_idxs).long().cpu()]
        else:
            masks = torch.ones_like(samples)
            regimes = None
        return samples, masks, regimes
    
    def calc_regimes(self):
        regimes = []
        interventions_label_list = []
        for i in range(self.adata_obj.shape[0]):
            interv_label = self.adata_obj.obs['perturbation'].iloc[i]
            if interv_label not in interventions_label_list:
                interventions_label_list.append(interv_label)
                regimes.append(len(interventions_label_list))
            regimes.append(interventions_label_list.index(interv_label))
        return regimes
