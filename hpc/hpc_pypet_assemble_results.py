import sys
import os
import shutil
import pickle
from datetime import datetime
from pypet import Trajectory
from pypet import Environment
import numpy as np
import pandas as pd
from mylib_pypet import print_leaves


# Use pickle module to load dictionaries from disk
def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def assemble(traj):

    run_name = traj.v_crun

    nodes_n = traj.parameters.topology.initial['nodes_n']

    # Initialise empty results dictionary
    res = dict.fromkeys(range(nodes_n))
    for node_i in range(nodes_n):
        res_single_target_file_name = '.'.join([run_name, 'network_analysis', 'res', str(node_i)])
        path = os.path.join(traj_dir, res_single_target_file_name + '.pkl')
        if os.path.exists(path):
            res_single_target = load_obj(path)
            res[node_i] = res_single_target
        else:
            raise ValueError('WARNING: Results dictionary missing for target {0} in {1}'.format(node_i, run_name))
    # Convert to DataFrame
    res_df = pd.DataFrame(res)
    # Add results dictionary to trajectory results
    traj.f_add_result('$.network_inference.network_inference_result', res_df)

    # Check if network files exist and add them to the trajectory
    single_run_network_file_names = [
        'delay.initial.delay_matrix',
        'node_coupling.initial.coupling_matrix',
        'topology.initial.adjacency_matrix'
    ]
    for filename in single_run_network_file_names:
        path = os.path.join(traj_dir, '.'.join([run_name, filename]) + '.csv')
        if os.path.exists(path):
            obj = pd.read_csv(path, index_col=0)#, dtype=np.ndarray)
            # Convert from strings to numpy arrays
            if type(obj.values[0, 0]) == str:
                obj = pd.DataFrame([[np.array(eval(x)) for x in row] for row in obj.values])
            traj.f_add_result('$.' + filename, obj)
        else:
            raise ValueError('WARNING: file missing: {0}'.format(path))

    # Check if time series file exists and add it to the trajectory
    filename = 'node_dynamics.time_series'
    path = os.path.join(traj_dir, '.'.join([run_name, filename]) + '.npy')
    if os.path.exists(path):
        obj = np.load(path)
        traj.f_add_result('$.node_dynamics.time_series', obj)
    else:
        raise ValueError('WARNING: file missing: {0}'.format(path))


#------MAIN---------------------------------------------------------------------------------

#traj_filenames = list(filter(
#    lambda file: file.startswith("100nodes"),
#    os.listdir(traj_dir)
#    ))

# Choose base directory containing trajectory directories
base_dir = '/project/RDS-FEI-InfoDynFuncStruct-RW/Leo/inference/trajectories/'
#base_dir = '/home/leo/Projects/inference/trajectories/'
#base_dir = 'C:\\DATA\\Google Drive\\University\\USyd\\Projects\\Information network inference\\trajectories\\'
traj_dir_names = [
    '2018_04_17_16h03m50s',
    '2018_04_17_17h21m02s',
    '2018_04_18_10h56m50s',
    '2018_04_18_17h02m30s',
    '2018_04_19_00h29m12s'
]
traj_filename = 'traj.hdf5'

traj_dir_fullpaths = [os.path.join(base_dir, dir_name) for dir_name in traj_dir_names]

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
        traj_backup_fullpath = os.path.join(traj_dir, traj_filename + '.backup' + datetime.now().strftime("%Y_%m_%d_%Hh%Mm%Ss"))
        shutil.copy(traj_fullpath, traj_backup_fullpath)

        # Create a pypet Environment object and link it to the trajectory
        env = Environment(
            trajectory=traj
        )

        # Run assembly of trajectory
        env.run(assemble)

        print('\nFinished assembling files in folder: {0}'.format(traj_dir))
        print('Parameters:')
        print_leaves(traj, 'parameters')
        print('----------------------------------------------------------\n')

        # Finally disable logging
        env.disable_logging()
    else:
        print('Folder skipped, trajectory was already completed: {0}'.format(traj_dir))
