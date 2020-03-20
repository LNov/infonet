import sys
import os
import shutil
import pickle
from datetime import datetime
from pypet import Trajectory
from pypet import Environment
from pypet import PickleResult
import numpy as np
from idtxl.stats import network_fdr
from mylib_pypet import print_leaves


# Use pickle module to load dictionaries from disk
def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def assemble(traj):

    run_name = traj.v_crun

    nodes_n = traj.parameters.topology.initial['nodes_n']

    # Gather single target results objects
    res_list = []
    for node_i in range(nodes_n):
        res_single_target_file_name = '.'.join([
            run_name,
            'network_analysis',
            'res',
            str(node_i)
        ])
        path = os.path.join(traj_dir, res_single_target_file_name + '.pkl')
        if os.path.exists(path):
            res_single_target = load_obj(path)
            res_list.append(res_single_target)
        else:
            raise ValueError(
                'WARNING: Results missing for target {0} in {1}'.format(
                    node_i,
                    run_name
                )
            )
    if traj.parameters.network_inference.fdr_correction:
        res = network_fdr(
            {'alpha_fdr': res_list[0].settings['alpha_fdr']},
            *res_list
        )
    else:
        res = res_list[0]
        res.combine_results(*res_list[1:])

    # Add results dictionary to trajectory results
    traj.f_add_result(
        PickleResult,
        '$.network_inference',
        network_inference_result=res,
        comment=''
    )

    # Check if network files exist and add them to the trajectory
    single_run_network_file_names = [
        'topology.initial.adjacency_matrix',
        'delay.initial.delay_matrices',
        'node_coupling.initial.coupling_matrix',
        'node_coupling.initial.coefficient_matrices',
        'node_dynamics.time_series'
    ]
    for filename in single_run_network_file_names:
        path = os.path.join(traj_dir, '.'.join([run_name, filename]) + '.npy')
        if os.path.exists(path):
            obj = np.load(path)
            traj.f_add_result('$.' + filename, obj)
        else:
            raise ValueError('WARNING: file missing: {0}'.format(path))


# ------MAIN---------------------------------------------------------------------------------

# Choose base directory containing trajectory directories
base_dir = '/project/RDS-FEI-InfoDynFuncStruct-RW/Leo/inference/trajectories/'

traj_dir_names = [
    '2019_03_09_21h46m54s_KSG_on_CLM_1000samples_alpha001_run1'
]
traj_filename = 'traj.hdf5'

traj_dir_fullpaths = [
    os.path.join(base_dir, dir_name) for dir_name in traj_dir_names
]

for traj_dir in traj_dir_fullpaths:
    if os.path.exists(traj_dir):
        print('OK: Directory found: {0}'.format(traj_dir))
    else:
        raise ValueError('ERROR: Directory not found: {0}'.format(traj_dir))

for traj_dir in traj_dir_fullpaths:
    # Change current directory to the one containing the trajectory files
    os.chdir(traj_dir)

    # Load the trajectory (only parameters)
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

    # Ensure trajectory was not already assembled
    if not traj.f_is_completed():
        # Save a backup version of the original trajectory
        traj_backup_fullpath = os.path.join(
            traj_dir,
            traj_filename + '.backup' + datetime.now().strftime(
                "%Y_%m_%d_%Hh%Mm%Ss"
            )
        )
        shutil.copy(traj_fullpath, traj_backup_fullpath)

        # Create a pypet Environment object and link it to the trajectory
        env = Environment(trajectory=traj)

        # Run assembly of trajectory
        env.run(assemble)

        print('\nFinished assembling files in folder: {0}'.format(traj_dir))
        print('Parameters:')
        print_leaves(traj, 'parameters')
        print('----------------------------------------------------------\n')

        # Finally disable logging
        env.disable_logging()
    else:
        print('Folder skipped: trajectory already completed: {0}'.format(
            traj_dir)
        )
