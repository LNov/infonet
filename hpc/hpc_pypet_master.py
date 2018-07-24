import sys
import os
from datetime import datetime
import numpy as np
from pypet import Trajectory
from pypet.utils.explore import cartesian_product
from subprocess import call


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
    """Main function to protect the *entry point* of the program."""

    # Get current directory
    traj_dir = os.getcwd()
    # Read output path (if provided)
    if len(sys.argv) > 1:
        # Add trailing slash if missing
        dir_provided = os.path.join(sys.argv[1], '')
        # Check if provided directory exists
        if os.path.isdir(dir_provided):
            # Convert to full path
            traj_dir = os.path.abspath(dir_provided)
        else:
            print('WARNING: Output path not found, current directory will be used instead')
    else:
        print('WARNING: Output path not provided, current directory will be used instead')
    # Add time stamp (the final '' is to make sure there is a trailing slash)
    traj_dir = os.path.join(traj_dir, datetime.now().strftime("%Y_%m_%d_%Hh%Mm%Ss"), '')
    # Create directory with time stamp
    os.makedirs(traj_dir)
    # Change current directory to the one containing the trajectory files
    os.chdir(traj_dir)
    print('Trajectory and results will be stored to: {0}'.format(traj_dir))

    # Create new pypet Trajectory object
    traj_filename = 'traj.hdf5'
    traj_fullpath = os.path.join(traj_dir, traj_filename)
    traj = Trajectory(filename=traj_fullpath)

    # -------------------------------------------------------------------
    # Add config parameters (those that DO NOT influence the final result of the experiment)
    traj.f_add_config('parallel_target_analysis', True, comment='Analyse targets in parallel')

    # -------------------------------------------------------------------
    # Add "proper" parameters (those that DO influence the final result of the experiment)

    # -------------------------------------------------------------------
    # Parameters characterizing the network inference algorithm
    traj.f_add_parameter('network_inference.algorithm', 'mTE_greedy')
#    traj.parameters.f_get('network_inference.algorithm').v_comment = network_inference_algorithms['Description'].get(traj.parameters['network_inference.algorithm'])
    traj.f_add_parameter('network_inference.min_lag_sources', 5, comment='')
    traj.f_add_parameter('network_inference.max_lag_sources', 40, comment='')
    traj.f_add_parameter('network_inference.tau_sources', 5, comment='')
    traj.f_add_parameter('network_inference.max_lag_target', 40, comment='')
    traj.f_add_parameter('network_inference.tau_target', 5, comment='')
    # traj.f_add_parameter('network_inference.cmi_estimator', 'JidtGaussianCMI', comment='Conditional Mutual Information estimator')
    traj.f_add_parameter('network_inference.cmi_estimator', 'JidtKraskovCMI', comment='Conditional Mutual Information estimator')
    # traj.f_add_parameter('network_inference.cmi_estimator', 'OpenCLKraskovCMI', comment='Conditional Mutual Information estimator')
    traj.f_add_parameter('network_inference.permute_in_time', False, comment='')
    traj.f_add_parameter('network_inference.jidt_threads_n', 'USE_ALL', comment='Number of threads used by JIDT estimator (default=USE_ALL)')
    traj.f_add_parameter('network_inference.n_perm_max_stat', 200, comment='')
    traj.f_add_parameter('network_inference.n_perm_min_stat', 200, comment='')
    traj.f_add_parameter('network_inference.n_perm_omnibus', 200, comment='')
    traj.f_add_parameter('network_inference.n_perm_max_seq', 200, comment='')
    traj.f_add_parameter('network_inference.fdr_correction', True, comment='')
    traj.f_add_parameter('network_inference.z_standardise', False, comment='')
    traj.f_add_parameter('network_inference.kraskov_k', 4, comment='')
    traj.f_add_parameter('network_inference.p_value', 0.05, comment='critical alpha level for statistical significance testing')

    # traj.f_add_parameter('network_inference.alpha_max_stats', traj.parameters['network_inference.p_value'], comment='')
    # traj.f_add_parameter('network_inference.alpha_min_stats', traj.parameters['network_inference.p_value'], comment='')
    # traj.f_add_parameter('network_inference.alpha_omnibus', traj.parameters['network_inference.p_value'], comment='')
    # traj.f_add_parameter('network_inference.alpha_max_seq', traj.parameters['network_inference.p_value'], comment='')
    # traj.f_add_parameter('network_inference.alpha_fdr', traj.parameters['network_inference.p_value'], comment='')

    # -------------------------------------------------------------------
    # Parameters characterizing the initial topology of the network
#    traj.f_add_parameter('topology.initial.model', 'ER_n_in')
#    traj.parameters.f_get('topology.initial.model').v_comment = topology_models['Description'].get(traj.parameters['topology.initial.model'])
    traj.f_add_parameter('topology.initial.nodes_n', 7, comment='Number of nodes')
#    traj.f_add_parameter('topology.initial.in_degree_expected', 3, comment='Expected in-degree')
#
#    # -------------------------------------------------------------------
#    # Parameters characterizing the evolution of the topology
#    traj.f_add_parameter('topology.evolution.model', 'static')
#    traj.parameters.f_get('topology.evolution.model').v_comment = topology_evolution_models['Description'].get(traj.parameters['topology.evolution.model'])
#
#    # -------------------------------------------------------------------
#    # Parameters characterizing the coupling between the nodes
#    traj.f_add_parameter('node_coupling.initial.model', 'linear', comment='Linear coupling model: the input to each target node is the weighted sum of the outputs of its source nodes')
#    traj.f_add_parameter('node_coupling.initial.weight_distribution', 'deterministic')
#    traj.parameters.f_get('node_coupling.initial.weight_distribution').v_comment = weight_distributions['Description'].get(traj.parameters['node_coupling.initial.weight_distribution'])
#    traj.f_add_parameter('node_coupling.initial.self_coupling', 0.5, comment='The self-coupling is the weight of the self-loop')
#    traj.f_add_parameter('node_coupling.initial.total_cross_coupling', 0.35, comment='The total cross-coupling is the sum of all incoming weights from the sources only')
#
#    # -------------------------------------------------------------------
#    # Parameters characterizing the delay
#    traj.f_add_parameter('delay.initial.distribution', 'uniform')
#    traj.parameters.f_get('delay.initial.distribution').v_comment = delay_distributions['Description'].get(traj.parameters['delay.initial.distribution'])
#    traj.f_add_parameter('delay.initial.delay_links_n_max', 2, comment='Maximum number of delay links')
#    traj.f_add_parameter('delay.initial.delay_min', 1, comment='')
#    traj.f_add_parameter('delay.initial.delay_max', 5, comment='')
#    traj.f_add_parameter('delay.initial.delay_self', 1, comment='')
#
    # -------------------------------------------------------------------
    # Parameters characterizing the dynamics of the nodes
#    traj.f_add_parameter('node_dynamics.model', 'logistic_map')
#    #traj.f_add_parameter('node_dynamics.model', 'AR_gaussian_discrete')
#    traj.parameters.f_get('node_dynamics.model').v_comment = node_dynamics_models['Description'].get(traj.parameters['node_dynamics.model'])
    traj.f_add_parameter('node_dynamics.samples_n', 100, comment='Number of samples (observations) to record')
#    traj.f_add_parameter('node_dynamics.samples_transient_n', 1000 * traj.topology.initial.nodes_n, comment='Number of initial samples (observations) to skip to leave out the transient')
#    traj.f_add_parameter('node_dynamics.replications', 10, comment='Number of replications (trials) to record')
#    traj.f_add_parameter('node_dynamics.noise_std', 0.1, comment='Standard deviation of Gaussian noise')

    # -------------------------------------------------------------------
#    # Parameters characterizing the repetitions of the same run
#    traj.f_add_parameter('repetition_i', 0, comment='Index of the current repetition') # Normally starts from 0

    # Parameters characterizing the repetitions of the same run
    traj.f_add_parameter('subject_label', 'A01', comment='Labels identifying the subjects')

    # -------------------------------------------------------------------
    # Define parameter combinations to explore (a trajectory in
    # the parameter space)
    # The second argument, the tuple, specifies the order of the cartesian product,
    # The variable on the right most side changes fastest and defines the
    # 'inner for-loop' of the cartesian product
#    explore_dict = cartesian_product(
#        {
#            'repetition_i': np.arange(0, 2, 1).tolist(),
#            'topology.initial.nodes_n': np.arange(5, 5+1, 5).tolist(),
#            'node_dynamics.samples_n': (10 ** np.arange(1, 1+0.1, 1)).round().astype(int).tolist(),
#            'network_inference.p_value': np.array([0.05]).tolist()
#        },
#        ('repetition_i', 'topology.initial.nodes_n', 'node_dynamics.samples_n', 'network_inference.p_value')
#    )
    explore_dict = {
        'subject_label': ['A01']
    }
    traj.f_explore(explore_dict)

    # Store trajectory
    traj.f_store()

    # Define PBS script
    bash_lines = '\n'.join([
        '#! /bin/bash',
        '#PBS -P InfoDynFuncStruct',
        '#PBS -l select=1:ncpus=2:mem=4GB',
        '#PBS -M lnov6504@uni.sydney.edu.au',
        '#PBS -m abe',
        'module load java',
        'module load python/3.5.1',
        'source /project/RDS-FEI-InfoDynFuncStruct-RW/Leo/idtxl_env/bin/activate',
        'cd ${traj_dir}',
        'python ${python_script_path} ${traj_dir} ${traj_filename} ${file_prefix} $PBS_ARRAY_INDEX'
        ])

    # Save PBS script file (automatically generated)
    bash_script_name = 'run_python_script.pbs'
    job_script_path = os.path.join(traj_dir, bash_script_name)
    with open(job_script_path, 'w', newline='\n') as bash_file:
        bash_file.writelines(bash_lines)

    # Run job array
    job_walltime_hours = 0
    job_walltime_minutes = 3
    #after_job_array_ends = 1573895
    job_settings = {
        'N': 'run_traj',
        'l': 'walltime={0}:{1}:00'.format(job_walltime_hours, job_walltime_minutes),
        #'W': 'depend=afteranyarray:{0}[]'.format(after_job_array_ends),
        'q': 'defaultQ'
    }
    if len(traj.f_get_run_names()) > 1:
        job_settings['J'] = '{0}-{1}'.format(0, len(traj.f_get_run_names()) - 1)

    job_args = {
        'python_script_path': '/project/RDS-FEI-InfoDynFuncStruct-RW/Leo/inference/hpc_pypet_single_run.py',
        'traj_dir': traj_dir,
        'traj_filename': traj_filename,
        'file_prefix': 'none'
    }
    run_job_array(job_script_path, job_settings, job_args)


if __name__ == '__main__':
    # This will execute the main function in case the script is called from the one true
    # main process and not from a child processes spawned by your environment.
    # Necessary for multiprocessing under Windows.
    main()
