import sys
import os
import numpy as np
from pypet import Trajectory
from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data
import pickle
import time


# Use pickle module to save dictionaries
def save_obj(obj, file_dir, file_name):
    with open(os.path.join(file_dir, file_name), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


# Read parameters from shell call
traj_dir = sys.argv[1]
traj_filename = sys.argv[2]
file_prefix = sys.argv[3]
target_id = np.int(sys.argv[4])

print('target_id= {0}'.format(target_id))
print('traj_dir= {0}'.format(traj_dir))
print('traj_filename= {0}'.format(traj_filename))
print('file_prefix= {0}'.format(file_prefix))

#try:
# Load the trajectory from the hdf5 file
traj = Trajectory()
traj.f_load(
            filename=os.path.join(traj_dir, traj_filename),
            index=0,
            as_new=False,
            force=True,
            load_parameters=2,
            load_derived_parameters=2,
            load_results=2,
            load_other_data=2
        )

# Load time series
time_series = np.load(os.path.join(traj_dir, '.'.join([file_prefix, 'node_dynamics.time_series.npy'])))

#except:
#    raise

#else:

# Check if data can be normalised per process (assuming the
# first dimension represents processes, as in the rest of the code)
can_be_normalised = np.all(np.std(time_series, axis=1) > 0)
if not can_be_normalised:
    print('WARNING: Time series can not be normalised')

# initialise an empty data object
dat = Data()

# Load time series
dat = Data(time_series, dim_order='ps', normalise=can_be_normalised)

network_inference = traj.parameters.network_inference

algorithm = network_inference.algorithm
if algorithm == 'mTE_greedy':
    # Set analysis options
    network_analysis = MultivariateTE(
        min_lag_sources=network_inference.min_lag_sources,
        max_lag_sources=network_inference.max_lag_sources,
        options={
            'cmi_estimator':  network_inference.cmi_estimator,
            'num_threads': network_inference.jidt_threads_n,
            'permute_in_time': network_inference.permute_in_time,
            'n_perm_max_stat': 2000,
            'n_perm_min_stat': 2000,
            'n_perm_omnibus': 2000,
            'n_perm_max_seq': 2000,
            'fdr_correction': network_inference.fdr_correction,
            'alpha_max_stat': network_inference.p_value,
            'alpha_min_stat': network_inference.p_value,
            'alpha_omnibus': network_inference.p_value,
            'alpha_max_seq': network_inference.p_value,
            'alpha_fdr': network_inference.p_value
        }
    )

    # Run analysis
    res = network_analysis.analyse_single_target(dat, target=target_id)

    # Save results dictionary using pickle
    save_obj(res, traj_dir, '.'.join([file_prefix, 'network_analysis.res', str(target_id), 'pkl']))

#    else:
#        raise ParameterValue(algorithm, msg='Network inference algorithm not yet implemented')  
