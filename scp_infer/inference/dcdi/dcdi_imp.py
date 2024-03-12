"""Application of GRNBoost2 Algorithm (using arboreto)"""



import sys
import os
current_dir = os.path.abspath(".")
sys.path.append(os.path.join(current_dir, 'algorithm_implementations'))  # noqa: E402

import torch
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# import the local dcdi algorithm
import dcdi_local as dcdi
from dcdi_local.models.learnables import LearnableModel_NonLinGaussANM
from dcdi_local.models.flows import DeepSigmoidalFlowModel
from dcdi_local.train import train, retrain, compute_loss
from dcdi_local.data import DataManagerFile
from dcdi_local.utils.save import dump

from ..inference_method import InferenceMethod
from .dcdi_load import DataManagerAnndata

def _print_metrics(stage, step, metrics, throttle=None):
    for k, v in metrics.items():
        print("    %s:" % k, v)


def file_exists(prefix, suffix):
    return os.path.exists(os.path.join(prefix, suffix))


class DCDIImp(InferenceMethod):
    """
    DCDI implementation

    Attributes:
    adata_obj: AnnData
        Annotated expression data object from scanpy
    verbose: bool
    """

    tf_names: pd.DataFrame
    expression_data: pd.DataFrame

    def __init__(self, adata_obj, verbose: bool = False):
        """
        Initialize the DCDI implementation

        Set all default Hyperparameters
        """

        self.adata_obj = adata_obj
        self.verbose = verbose

        opt = argparse.Namespace()
        self.opt = opt
        # experiment
        opt.exp_path = './dcdi_implementation/exp_10genes_100k'  # Path to experiments
        opt.train = True            # Run `train` function, get /train folder
        opt.retrain = False         # after to-dag or pruning, retrain model
                                    # from scratch before reporting nll-val
        opt.dag_for_retrain = None  # path to a DAG in .npy format which will be used
                                    # for retrainig. e.g.  /code/stuff/DAG.npy
        opt.random_seed = 42        # Random seed for pytorch and numpy

        # data
        opt.data_path = None        # Path to data files
        opt.i_dataset = None        # dataset index
        opt.num_vars = len(self.adata_obj.var_names)            # Number of variables
        opt.train_samples = 0.8     # Number of samples used for training
                                    # (default is 80% of the total size)
        opt.test_samples = None     # Number of samples used for testing
                                    # (default is whatever is not used for training)
        opt.num_folds = 5           # number of folds for cross-validation
        opt.fold = 0                # fold we should use for testing
        opt.train_batch_size = 64   # number of samples in a minibatch
        opt.num_train_iter = 1000000  # number of meta gradient steps
        opt.normalize_data = False  # (x - mu) / std
        opt.regimes_to_ignore = None  # When loading data, will remove some regimes from data set
        # When using --regimes-to-ignore, we evaluate performance
        # on new regimes never seen during training (use after retraining).
        opt.test_on_new_regimes = False

        # model
        opt.model = 'DCDI-G'        # model class (DCDI-G or DCDI-DSF)
        opt.num_layers = 2          # number of hidden layers
        opt.hid_dim = 16            # number of hidden units per layer
        opt.nonlin = 'leaky-relu'   # leaky-relu | sigmoid
        opt.flow_num_layers = 2     # number of hidden layers of the DSF
        opt.flow_hid_dim = 16       # number of hidden units of the DSF

        # intervention
        opt.intervention = True     # Use data with intervention
        opt.dcd = False             # Use DCD (DCDI with a loss
                                    # not taking into account the intervention)
        opt.intervention_type = "imperfect"  # Type of intervention: perfect or imperfect
        opt.intervention_knowledge = "known"  # If the targets of the
                                    # intervention are known or unknown
        # Coefficient of the regularisation in the unknown interventions case (lambda_R)
        opt.coeff_interv_sparsity = 1e-8

        # optimization
        opt.optimizer = "rmsprop"   # sgd|rmsprop
        opt.lr = 1e-3               # learning rate for optim
        opt.lr_reinit = None        # Learning rate for optim after first subproblem.
                                    # Default mode reuses --lr.
        opt.lr_schedule = None      # Learning rate for optim, change initial lr as a
                                    # function of mu: None|sqrt-mu|log-mu
        opt.stop_crit_win = 100     # window size to compute stopping criterion
        opt.reg_coeff = 0.1         # regularization coefficient (lambda)

        # Augmented Lagrangian options
        opt.omega_gamma = 1e-4      # Precision to declare convergence of subproblems
        opt.omega_mu = 0.9          # After subproblem solved, h should have reduced by this ratio
        opt.mu_init = 1e-8          # initial value of mu
        opt.mu_mult_factor = 2      # Multiply mu by this amount when constraint
                                    # not sufficiently decreasing
        opt.gamma_init = 0.         # initial value of gamma
        opt.h_threshold = 1e-8      # Stop when |h|<X. Zero means stop AL procedure only when h==0

        # misc
        opt.patience = 10           # Early stopping patience in --retrain.
        opt.train_patience = 5      # Early stopping patience in --train after constraint
        opt.train_patience_post = 5  # Early stopping patience in --train after threshold

        # logging
        opt.plot_freq = 100       # plotting frequency
        # do not log weighted adjacency (to save RAM). One plot will be missing (A_\phi plot)
        opt.no_w_adjs_log = False
        opt.plot_density = False    # Plot density (only implemented for 2 vars)

        # device and numerical precision
        opt.gpu = True              # Use GPU
        opt.float = False           # Use Float precision

        self.plotting_callback = None

    def convert_data(self):
        """convert adata entries into GRNBoost2 format"""
        # Load the TF names
        self.tf_names = self.adata_obj.var_names
        self.expression_data = self.adata_obj.to_df()

    def infer(
        self,
        plot: bool = False,
        **kwargs
    ) -> np.array:
        """
        Parameters for the DCDI algorithm

        store parameters as attributes of opt
        """

        if self.verbose:
            print("Running GRNBoost2")

        opt = self.opt

        # Control as much randomness as possible
        torch.manual_seed(opt.random_seed)
        np.random.seed(opt.random_seed)

        if opt.lr_reinit is not None:
            assert opt.lr_schedule is None, "--lr-reinit and --lr-schedule are mutually exclusive"

        # Initialize metric logger if needed
        metrics_callback = _print_metrics

        # adjust some default hparams
        if opt.lr_reinit is None:
            opt.lr_reinit = opt.lr

        # Use GPU
        if opt.gpu:
            if opt.float:
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
            else:
                torch.set_default_tensor_type('torch.cuda.DoubleTensor')
        else:
            if opt.float:
                torch.set_default_tensor_type('torch.FloatTensor')
            else:
                torch.set_default_tensor_type('torch.DoubleTensor')

        # create DataManager for training
        train_data = DataManagerAnndata(opt.data_path, self.adata_obj, opt.train_samples,
                                        opt.test_samples, train=True,
                                        normalize=opt.normalize_data,
                                        random_seed=opt.random_seed,
                                        intervention=opt.intervention,
                                        intervention_knowledge=opt.intervention_knowledge,
                                        dcd=opt.dcd)
        test_data = DataManagerAnndata(opt.data_path, self.adata_obj, opt.train_samples,
                                       opt.test_samples, train=False,
                                       normalize=opt.normalize_data,
                                       mean=train_data.mean, std=train_data.std,
                                       random_seed=opt.random_seed,
                                       intervention=opt.intervention,
                                       intervention_knowledge=opt.intervention_knowledge,
                                       dcd=opt.dcd)

        # create learning model and ground truth model
        if opt.model == "DCDI-G":
            model = LearnableModel_NonLinGaussANM(opt.num_vars,
                                                  opt.num_layers,
                                                  opt.hid_dim,
                                                  nonlin=opt.nonlin,
                                                  intervention=opt.intervention,
                                                  intervention_type=opt.intervention_type,
                                                  intervention_knowledge=opt.intervention_knowledge,
                                                  num_regimes=train_data.num_regimes)
        elif opt.model == "DCDI-DSF":
            model = DeepSigmoidalFlowModel(num_vars=opt.num_vars,
                                           cond_n_layers=opt.num_layers,
                                           cond_hid_dim=opt.hid_dim,
                                           cond_nonlin=opt.nonlin,
                                           flow_n_layers=opt.flow_num_layers,
                                           flow_hid_dim=opt.flow_hid_dim,
                                           intervention=opt.intervention,
                                           intervention_type=opt.intervention_type,
                                           intervention_knowledge=opt.intervention_knowledge,
                                           num_regimes=train_data.num_regimes)
        else:
            raise ValueError("opt.model has to be in {DCDI-G, DCDI-DSF}")

        # print device of samples, masks and regimes
        print("train_data.adjacency.device:", train_data.adjacency.device)
        print("train_data.asmples.device:", train_data.gt_interv.device)
        # print("train_data.regimes.device:", train_data.regimes.device)

        # train until constraint is sufficiently close to being satisfied
        if opt.train:
            train(model, train_data.adjacency.detach().cpu().numpy(),
                  train_data.gt_interv, train_data, test_data, opt, metrics_callback,
                  self.plotting_callback)

        elif opt.retrain:
            initial_dag = np.load(opt.dag_for_retrain)
            model.adjacency[:, :] = torch.as_tensor(initial_dag).type(torch.Tensor)
            best_model = retrain(model, train_data, test_data, "ignored_regimes",
                                 opt, metrics_callback, self.plotting_callback)

        # Evaluate on ignored regimes!
        if opt.test_on_new_regimes:
            all_regimes = train_data.all_regimes

            # take all data, but ignore data on which we trained (want to test on unseen regime)
            regimes_to_ignore = np.setdiff1d(all_regimes, np.array(opt.regimes_to_ignore))
            new_data = DataManagerFile(opt.data_path, opt.i_dataset, 1., None, train=True,
                                       normalize=opt.normalize_data,
                                       random_seed=opt.random_seed,
                                       intervention=opt.intervention,
                                       intervention_knowledge=opt.intervention_knowledge,
                                       dcd=opt.dcd,
                                       regimes_to_ignore=regimes_to_ignore)

            with torch.no_grad():
                weights, biases, extra_params = best_model.get_parameters(mode="wbx")

                # evaluate on train
                x, masks, regimes = train_data.sample(train_data.num_samples)
                loss_train, mean_std_train = compute_loss(x, masks, regimes, best_model,
                                                          weights, biases, extra_params,
                                                          intervention=True,
                                                          intervention_type='structural',
                                                          intervention_knowledge="known",
                                                          mean_std=True)

                # evaluate on valid
                x, masks, regimes = test_data.sample(test_data.num_samples)
                loss_test, mean_std_test = compute_loss(x, masks, regimes, best_model,
                                                        weights, biases, extra_params,
                                                        intervention=True,
                                                        intervention_type='structural',
                                                        intervention_knowledge="known",
                                                        mean_std=True)

                # evaluate on new intervention
                x, masks, regimes = new_data.sample(new_data.num_samples)
                loss_new, mean_std_new = compute_loss(x, masks, regimes, best_model,
                                                      weights, biases, extra_params,
                                                      intervention=True,
                                                      intervention_type='structural',
                                                      intervention_knowledge="known",
                                                      mean_std=True)

                # logging final result
                metrics_callback(stage="test_on_new_regimes", step=0,
                                 metrics={"log_likelihood_train": - loss_train.item(),
                                          "mean_std_train": mean_std_train.item(),
                                          "log_likelihood_test": - loss_test.item(),
                                          "mean_std_test": mean_std_test.item(),
                                          "log_likelihood_new": - loss_new.item(),
                                          "mean_std_new": mean_std_new.item()}, throttle=False)
