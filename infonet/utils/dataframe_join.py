import os
from datetime import datetime
import time
import numpy as np
import pandas as pd

# Choose whether to use FDR-corrected results or not
fdr = False

# Choose directories containing DataFrames to be merged
traj_dir_list = [
    os.path.join('../trajectories', 'KSG_on_CLM_10000samples'),
    os.path.join('../trajectories', 'postprocessing_df_concatenate2019_03_19_18h32m03s')
]

# Create output folder for the new analysis
output_dir = os.path.join('../trajectories', 'postprocessing_df_concatenate' + datetime.now().strftime("%Y_%m_%d_%Hh%Mm%Ss"), '')
os.makedirs(output_dir)
time.sleep(5)

# Check that directories exist
for traj_dir in traj_dir_list:
    if os.path.exists(traj_dir):
        print('OK: Directory found: {0}'.format(traj_dir))
    else:
        raise ValueError('ERROR: Directory not found: {0}'.format(traj_dir))

# Load DataFrames
df_list = []
for traj_dir in traj_dir_list:
    if fdr:
        df_list.append(pd.read_pickle(os.path.join(traj_dir, 'postprocessing_fdr.pkl')))
    else:
        df_list.append(pd.read_pickle(os.path.join(traj_dir, 'postprocessing.pkl')))

df_final = pd.DataFrame()
for df in df_list:
    len_before_append = len(df_final)
    # Append partial DataFrame
    df_final = df_final.append(df, ignore_index=True)
    # Preserve column ordering of the partial DataFrame
    df_final = df_final[df.columns.tolist()]
    # Sanity check 1: the keys of the growing fianl DafaFrame should
    # be the same as the partial DataFrame that has been appended 
    assert set(df_final.keys()) == set(df.keys())
    # Sanity check 2: the length of the growing fianl DafaFrame should
    # be the sum of its length before appending the partial
    # DataFrame + the length of the partial DataFrame
    assert len(df_final) == len_before_append + len(df)

# Save DataFrame
if fdr:
    df_final.to_pickle(os.path.join(output_dir, 'postprocessing_fdr.pkl'))
else:
    df_final.to_pickle(os.path.join(output_dir, 'postprocessing.pkl'))
