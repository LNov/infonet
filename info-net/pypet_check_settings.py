from pypet import Trajectory
import os
import numpy as np
from scipy.linalg import block_diag
from scipy.linalg import solve_discrete_lyapunov
import pandas as pd
import networkx as nx
from mylib_pypet import print_leaves

fdr = False
save_results = True

# Load the trajectory from the hdf5 file
# Only load parameters, results will be loaded at runtime (auto loading)
traj_dir = os.path.join('trajectories', 'GC_on_CLM_100and1000')
if not os.path.isdir(traj_dir):
    traj_dir = os.path.join('..', traj_dir)
traj_filename = 'traj.hdf5'
traj_fullpath = os.path.join(traj_dir, traj_filename)
traj = Trajectory()
traj.f_load(
    filename=traj_fullpath,
    index=0,
    load_parameters=2,
    load_results=0,
    load_derived_parameters=0,
    force=True
)
# Turn on auto loading
traj.v_auto_load = True

# Count number of runs
runs_n = len(traj.f_get_run_names())

# Get list of explored parameters
parameters_explored = [str.split(par, '.').pop() for par in (
    traj.f_get_explored_parameters())]

print_leaves(traj, 'parameters')

# Loop over runs
for run_name in traj.f_get_run_names():

    # Make trajectory behave like a particular single run:
    # all explored parameter’s values will be set to the corresponding
    # values of one particular run.
    traj.f_set_crun(run_name)

    print('\n{0}'.format(run_name))
    print('Exploring current parameter combination:')
    for par in parameters_explored:
        print('{0} = {1}'.format(
            par,
            traj.parameters[par]))

    # Load results object
    res = traj.results[run_name].network_inference.network_inference_result
    # Check if old release
    if type(res) == pd.core.frame.DataFrame:
        release_results_class = False
    else:
        release_results_class = True

    if release_results_class:
        print('Results:')
        print('n_perm_max_stat = {0}'.format(res.settings['n_perm_max_stat']))
        print('n_perm_min_stat = {0}'.format(res.settings['n_perm_min_stat']))
        print('n_perm_max_seq = {0}'.format(res.settings['n_perm_max_seq']))
        print('n_perm_omnibus = {0}'.format(res.settings['n_perm_omnibus']))

        print('alpha_max_stat = {0}'.format(res.settings['alpha_max_stat']))
        print('alpha_min_stat = {0}'.format(res.settings['alpha_min_stat']))
        print('alpha_max_seq = {0}'.format(res.settings['alpha_max_seq']))
        print('alpha_omnibus = {0}'.format(res.settings['alpha_omnibus']))
        print('alpha_fdr = {0}'.format(res.settings['alpha_fdr']))
        print(np.sort([res.get_single_target(0, fdr=False)['selected_sources_pval']]))
        print('\n')
    else:  # old release without results class

        # Covert pandas DataFrame to dictionary
        res = res.to_dict()

        # Reconstruct inferred delay matrices
        if fdr:
            try:
                r = res['fdr_corrected']
            except KeyError:
                raise RuntimeError('No FDR-corrected results found.')
        else:
            r = res.copy()
            try:
                del r['fdr_corrected']
            except KeyError:
                pass
        print('Results:')

        print('n_perm_max_stat = {0}'.format(res[0]['options']['n_perm_max_stat']))
        print('n_perm_min_stat = {0}'.format(res[0]['options']['n_perm_min_stat']))
        print('n_perm_max_seq = {0}'.format(res[0]['options']['n_perm_max_seq']))
        print('n_perm_omnibus = {0}'.format(res[0]['options']['n_perm_omnibus']))

        print('alpha_max_stat = {0}'.format(res[0]['options']['alpha_max_stat']))
        print('alpha_min_stat = {0}'.format(res[0]['options']['alpha_min_stat']))
        print('alpha_max_seq = {0}'.format(res[0]['options']['alpha_max_seq']))
        print('alpha_omnibus = {0}'.format(res[0]['options']['alpha_omnibus']))
        print('alpha_fdr = {0}'.format(res[0]['options']['alpha_fdr']))
        print((np.sort(res[0]['selected_sources_pval'])))
        print('\n')
