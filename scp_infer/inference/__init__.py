"""init inference modules"""

# 0. Algorithm Template
from .inference_method import InferenceMethod

# 1. GRNBoost2
from .grnboost2 import *
# 2. GIES
from .gies import *
# 3. DCDI
from .dcdi import *

# 4. data split
from .data_split import random_split_data, gene_perturbation_hold_out, perturbation_proportion_hold_out