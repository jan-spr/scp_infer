"""init for utils submodule"""

# 1. data split
from .data_split import shuffled_split, gene_holdout, total_intervention_holdout

from .run_infer import save_split, load_split

from .data_manage import ScpiDataManager
