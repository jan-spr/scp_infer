import os
import sys
import json
import numpy as np

import pdb

# 1. import Local GIES installation:
current_dir = os.path.abspath(".")
sys.path.append(os.path.join(current_dir, 'gies_master'))
print("Current dir: ", current_dir)
print(sys.path)

import gies_local as gies


# 2. Load experimental data:
data_m_file ="../../data/gies_data_matrix.npy"
intervention_l_file = "../../data/gies_intervention_list.json"

data_matrix = np.load(data_m_file)
with open(intervention_l_file, 'r') as f:
    intervention_list = json.load(f)

print('intervention list: ', intervention_list[:15])

# 3. Run GIES on data:
if True:
    #breakpoint()
    estimate, score = gies.fit_bic(data_matrix, intervention_list, A0 = None)
    print(estimate)