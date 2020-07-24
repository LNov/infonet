import network_dynamics
import sys
import os
import time
import numpy as np
from pypet import Trajectory
import networkx as nx
import copyreg
from subprocess import call
import argparse
import pickle


# Use pickle module to save dictionaries
def save_obj(obj, file_dir, file_name):
    with open(os.path.join(file_dir, file_name), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def run_job_array(job_script_path, job_settings, job_args={}):
    settings = ' '.join(['-{0} {1}'.format(key, job_settings[key]) for key in job_settings.keys()])
    args = '-v ' + ','.join(['{0}="{1}"'.format(key, job_args[key]) for key in job_args.keys()])
    # Submit PBS job
    call(
        ('qsub {1} {2} {0}').format(
            job_script_path,
            settings,
            args
            ),
        shell=True,
        timeout=None
    )


def main():
    traj_dir = sys.argv[1]
    traj_filename = sys.argv[2]
    file_prefix = sys.argv[3]
    run_i = 0
    if len(sys.argv) > 4:
        run_i = np.int(sys.argv[4])

    print('run_i= {0}'.format(run_i))
    print('traj_dir= {0}'.format(traj_dir))
    print('traj_filename= {0}'.format(traj_filename))
    print('file_prefix= {0}'.format(file_prefix))

    # Change current directory to the one containing the trajectory files
    os.chdir(traj_dir)
    # Load the trajectory from the hdf5 file
    traj_fullpath = os.path.join(traj_dir, traj_filename)
    traj = Trajectory()
    traj.f_load(
        filename=traj_fullpath,
        index=0,
        as_new=False,
        force=True,
        load_parameters=2,
        load_derived_parameters=2,
        load_results=2,
        load_other_data=2
    )
    # Set current run
    traj.v_idx = run_i
    # Read number of nodes
    nodes_n = traj.par.topology.initial.nodes_n
    # Generate initial network
    G = network_dynamics.generate_network(traj.par.topology.initial)
    # Get adjacency matrix
    adjacency_matrix = np.array(nx.to_numpy_matrix(
        G,
        nodelist=np.array(range(0, traj.par.topology.initial.nodes_n)),
        dtype=int
    ))
    # Add self-loops
    np.fill_diagonal(adjacency_matrix, 1)
    # Generate initial node coupling
    coupling_matrix = network_dynamics.generate_coupling(
        traj.par.node_coupling.initial,
        adjacency_matrix
    )
    # Generate delay
    delay_matrices = network_dynamics.generate_delay(traj.par.delay.initial, adjacency_matrix)
    # Generate coefficient matrices
    coefficient_matrices = np.transpose(delay_matrices * coupling_matrix, (0, 2, 1))
    # Run dynamics
    time_series = network_dynamics.run_dynamics(
        traj.par.node_dynamics,
        coefficient_matrices
    )
    # Create settings dictionary
    print('network_inference.p_value = {0}\n'.format(
        traj.par.network_inference.p_value))
    algorithm = traj.par.network_inference.algorithm
    print('network_inference.algorithm = {0}\n'.format(
        algorithm))
    settings = {
        'min_lag_sources': traj.par.network_inference.min_lag_sources,
        'max_lag_sources': traj.par.network_inference.max_lag_sources,
        'tau_sources': traj.par.network_inference.tau_sources,
        'max_lag_target': traj.par.network_inference.max_lag_target,
        'tau_target': traj.par.network_inference.tau_target,
        'cmi_estimator':  traj.par.network_inference.cmi_estimator,
        'kraskov_k': traj.par.network_inference.kraskov_k,
        'num_threads': traj.par.network_inference.jidt_threads_n,
        'permute_in_time': traj.par.network_inference.permute_in_time,
        'n_perm_max_stat': traj.par.network_inference.n_perm_max_stat,
        'n_perm_min_stat': traj.par.network_inference.n_perm_min_stat,
        'n_perm_omnibus': traj.par.network_inference.n_perm_omnibus,
        'n_perm_max_seq': traj.par.network_inference.n_perm_max_seq,
        'fdr_correction': traj.par.network_inference.fdr_correction,
        'alpha_max_stat': traj.par.network_inference.p_value,
        'alpha_min_stat': traj.par.network_inference.p_value,
        'alpha_omnibus': traj.par.network_inference.p_value,
        'alpha_max_seq': traj.par.network_inference.p_value,
        'alpha_fdr': traj.par.network_inference.p_value,
        'network_inference_algorithm': algorithm,
    }
    # Save objects to disk
    np.save(os.path.join(traj_dir, '.'.join([traj.v_crun, 'topology.initial.adjacency_matrix', 'npy'])), adjacency_matrix)
    np.save(os.path.join(traj_dir, '.'.join([traj.v_crun, 'node_coupling.initial.coupling_matrix', 'npy'])), coupling_matrix)
    np.save(os.path.join(traj_dir, '.'.join([traj.v_crun, 'node_coupling.initial.coefficient_matrices', 'npy'])), coefficient_matrices)
    np.save(os.path.join(traj_dir, '.'.join([traj.v_crun, 'delay.initial.delay_matrices', 'npy'])), delay_matrices)
    np.save(os.path.join(traj_dir, '.'.join([traj.v_crun, 'node_dynamics.time_series', 'npy'])), time_series)
    save_obj(settings, traj_dir, '.'.join([traj.v_crun, 'settings.pkl']))
    # Path to PBS script
    job_script_path = os.path.join(traj_dir, 'run_python_script.pbs')
    # Run job array
    job_walltime_hours = 5#1 + int(np.ceil((nodes_n + 20) / 30))
    job_walltime_minutes = 0
    job_settings = {
        'N': 'run{0}'.format(run_i),
        'J': '{0}-{1}'.format(0, nodes_n - 1),
        'l': 'walltime={0}:{1}:00'.format(job_walltime_hours, job_walltime_minutes),
        'q': 'defaultQ'
    }
    job_args = {
        'python_script_path': '/project/RDS-FEI-InfoDynFuncStruct-RW/Leo/inference/hpc_pypet_single_target.py',
        'traj_dir': traj_dir,
        'traj_filename': traj_filename,
        'file_prefix': traj.v_crun
    }
    run_job_array(job_script_path, job_settings, job_args)


if __name__ == '__main__':
    # This will execute the main function in case the script is called from the one true
    # main process and not from a child processes spawned by your environment.
    # Necessary for multiprocessing under Windows.
    main()
