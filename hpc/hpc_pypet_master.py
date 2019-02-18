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
    traj.f_add_config('debug', False, comment='Activate debug mode')
#    #traj.f_add_config('max_mem_frac', 0.7, comment='Fraction of global GPU memory to use')


    # -------------------------------------------------------------------
    # Add "proper" parameters (those that DO influence the final result of the experiment)

    # -------------------------------------------------------------------
    # Parameters characterizing the network inference algorithm
    traj.f_add_parameter('network_inference.algorithm', 'mTE_greedy')
    traj.f_add_parameter('network_inference.min_lag_sources', 1, comment='')
    traj.f_add_parameter('network_inference.max_lag_sources', 5, comment='')
    traj.f_add_parameter('network_inference.tau_sources', 1, comment='')
    traj.f_add_parameter('network_inference.max_lag_target', 5, comment='')
    traj.f_add_parameter('network_inference.tau_target', 1, comment='')
    #traj.f_add_parameter('network_inference.cmi_estimator', 'JidtGaussianCMI', comment='Conditional Mutual Information estimator')
    #traj.f_add_parameter('network_inference.cmi_estimator', 'JidtKraskovCMI', comment='Conditional Mutual Information estimator')
    traj.f_add_parameter('network_inference.cmi_estimator', 'OpenCLKraskovCMI', comment='Conditional Mutual Information estimator')
    traj.f_add_parameter('network_inference.permute_in_time', True, comment='')
    traj.f_add_parameter('network_inference.jidt_threads_n', 1, comment='Number of threads used by JIDT estimator (default=USE_ALL)')
    traj.f_add_parameter('network_inference.n_perm_max_stat', 1000, comment='')
    traj.f_add_parameter('network_inference.n_perm_min_stat', 1000, comment='')
    traj.f_add_parameter('network_inference.n_perm_omnibus', 1000, comment='')
    traj.f_add_parameter('network_inference.n_perm_max_seq', 1000, comment='')
    traj.f_add_parameter('network_inference.fdr_correction', True, comment='')
    traj.f_add_parameter('network_inference.z_standardise', True, comment='')
    traj.f_add_parameter('network_inference.kraskov_k', 4, comment='')
    traj.f_add_parameter('network_inference.p_value', 0.05, comment='critical alpha level for statistical significance testing')

    # traj.f_add_parameter('network_inference.alpha_max_stats', traj.parameters['network_inference.p_value'], comment='')
    # traj.f_add_parameter('network_inference.alpha_min_stats', traj.parameters['network_inference.p_value'], comment='')
    # traj.f_add_parameter('network_inference.alpha_omnibus', traj.parameters['network_inference.p_value'], comment='')
    # traj.f_add_parameter('network_inference.alpha_max_seq', traj.parameters['network_inference.p_value'], comment='')
    # traj.f_add_parameter('network_inference.alpha_fdr', traj.parameters['network_inference.p_value'], comment='')

    # -------------------------------------------------------------------
    # Parameters characterizing the initial topology of the network
    traj.f_add_parameter('topology.initial.model', 'ER_n_in')
    traj.f_add_parameter('topology.initial.nodes_n', 2, comment='Number of nodes')
    traj.f_add_parameter('topology.initial.in_degree_expected', 3, comment='Expected in-degree')

    # -------------------------------------------------------------------
    # Parameters characterizing the evolution of the topology
    traj.f_add_parameter('topology.evolution.model', 'static')

    # -------------------------------------------------------------------
    # Parameters characterizing the coupling between the nodes
    traj.f_add_parameter('node_coupling.initial.model', 'linear', comment='Linear coupling model: the input to each target node is the weighted sum of the outputs of its source nodes')
    traj.f_add_parameter('node_coupling.initial.weight_distribution', 'deterministic')
    traj.f_add_parameter('node_coupling.initial.self_coupling', 0.5, comment='The self-coupling is the weight of the self-loop')
    traj.f_add_parameter('node_coupling.initial.total_cross_coupling', 0.4, comment='The total cross-coupling is the sum of all incoming weights from the sources only')

    # -------------------------------------------------------------------
    # Parameters characterizing the delay
    traj.f_add_parameter('delay.initial.distribution', 'uniform')
    traj.f_add_parameter('delay.initial.delay_links_n_max', 1, comment='Maximum number of delay links')
    traj.f_add_parameter('delay.initial.delay_min', 1, comment='')
    traj.f_add_parameter('delay.initial.delay_max', 5, comment='')
    traj.f_add_parameter('delay.initial.delay_self', 1, comment='')

    # -------------------------------------------------------------------
    # Parameters characterizing the dynamics of the nodes
    traj.f_add_parameter('node_dynamics.model', 'logistic_map')
    #traj.f_add_parameter('node_dynamics.model', 'AR_gaussian_discrete')
    traj.f_add_parameter('node_dynamics.samples_n', 100, comment='Number of samples (observations) to record')
    traj.f_add_parameter('node_dynamics.samples_transient_n', 1000 * traj.topology.initial.nodes_n, comment='Number of initial samples (observations) to skip to leave out the transient')
    traj.f_add_parameter('node_dynamics.replications', 1, comment='Number of replications (trials) to record')
    traj.f_add_parameter('node_dynamics.noise_std', 0.1, comment='Standard deviation of Gaussian noise')

    # -------------------------------------------------------------------
    # Parameters characterizing the repetitions of the same run
    traj.f_add_parameter('repetition_i', 0, comment='Index of the current repetition') # Normally starts from 0

    # Parameters characterizing the repetitions of the same run
#    traj.f_add_parameter('subject_label', 'A01', comment='Labels identifying the subjects')

    # -------------------------------------------------------------------
    # Define parameter combinations to explore (a trajectory in
    # the parameter space)
    # The second argument, the tuple, specifies the order of the cartesian product,
    # The variable on the right most side changes fastest and defines the
    # 'inner for-loop' of the cartesian product
    explore_dict = cartesian_product(
        {
            'repetition_i': np.arange(0, 10, 1).tolist(),
            'topology.initial.nodes_n': np.arange(10, 100+1, 30).tolist(),
            'node_dynamics.samples_n': np.array([300]).tolist(),#(10 ** np.arange(4, 4+0.1, 1)).round().astype(int).tolist(),
            'network_inference.p_value': np.array([0.05]).tolist()
        },
        ('network_inference.p_value',
         'node_dynamics.samples_n',
         'topology.initial.nodes_n',
         'repetition_i'
        )
    )
#    explore_dict = {
#        'subject_label': ['A01']#, 'A02', 'A03', 'PR03', 'PR04', 'PR06']
#    }
    traj.f_explore(explore_dict)

    # Store trajectory
    traj.f_store()

    # Define PBS script
    bash_lines = '\n'.join([
        '#! /bin/bash',
        '#PBS -P InfoDynFuncStruct',
        #'#PBS -l select=1:ncpus=1:mem=16GB',
        '#PBS -l select=1:ncpus=1:ngpus=1:mem=1GB',
        '#PBS -M lnov6504@uni.sydney.edu.au',
        '#PBS -m abe',
        'module load java',
        'module load python/3.5.1',
        'module load cuda/8.0.44',
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
    job_walltime_minutes = 5
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
