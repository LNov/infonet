import sys
import os
import time
import numpy as np
import pandas as pd
from pypet import Trajectory
import networkx as nx
from idtxl.multivariate_te import MultivariateTE
import copyreg
from subprocess import call
import argparse
import scipy.io as spio
import pickle


# Use pickle module to save dictionaries
def save_obj(obj, file_dir, file_name):
    with open(os.path.join(file_dir, file_name), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


# Define parameter options dictionaries
network_inference_algorithms = pd.DataFrame()
network_inference_algorithms['Description'] = pd.Series({
    'mTE_greedy': 'Multivariate Transfer Entropy via greedy algorithm',
    'cross_corr': 'Cross-correlation thresholding algorithm'
})
network_inference_algorithms['Required parameters'] = pd.Series({
    'mTE_greedy': (
        'min_lag_sources',
        'max_lag_sources',
        'tau_sources',
        'max_lag_target',
        'tau_target',
        'cmi_estimator',
        #'kraskov_k',
        'z_standardise',
        'permute_in_time',
        'n_perm_max_stat',
        'n_perm_min_stat',
        'n_perm_omnibus',
        'n_perm_max_seq',
        'fdr_correction',
        'p_value'
        #'alpha_max_stats',
        #'alpha_min_stats',
        #'alpha_omnibus',
        #'alpha_max_seq',
        #'alpha_fdr'
    ),
    'bTE_greedy': (
        'min_lag_sources',
        'max_lag_sources',
        'tau_sources',
        'max_lag_target',
        'tau_target',
        'cmi_estimator',
        'z_standardise',
        'permute_in_time',
        'n_perm_max_stat',
        'n_perm_min_stat',
        'n_perm_omnibus',
        'n_perm_max_seq',
        'fdr_correction',
        'p_value'
        #'alpha_max_stats',
        #'alpha_min_stats',
        #'alpha_omnibus',
        #'alpha_max_seq',
        #'alpha_fdr'
    ),
    'cross_corr': (
        'min_lag_sources',
        'max_lag_sources'
    )
})

topology_models = pd.DataFrame()
topology_models['Description'] = pd.Series({
    'ER_n_p': 'Erdős–Rényi model with a fixed number of nodes and link probability',
    'ER_n_m': 'Erdős–Rényi model with a fixed number of nodes and links',
    'ER_n_in': 'Erdős–Rényi model with a fixed number of nodes and expected in-degree',
    'WS': 'Watts–Strogatz model',
    'BA': 'Barabási–Albert model',
    'ring': 'Ring model',
    'lattice': 'Regular lattice model',
    'star': 'Star model',
    'complete': 'Complete graph model',
    'explicit': 'Explicit topology provided by the user'
})
topology_models['Required parameters'] = pd.Series({
    'ER_n_p': ('nodes_n', 'ER_p'),
    'ER_n_m': ('nodes_n', 'ER_m'),
    'ER_n_in': ('nodes_n', 'in_degree_expected'),
    'WS': ('nodes_n', 'WS_k', 'WS_p')
})

topology_evolution_models = pd.DataFrame()
topology_evolution_models['Description'] = pd.Series({
    'static': 'Static network topology'
})
topology_evolution_models['Required parameters'] = pd.Series({
    'static': ()
})

weight_distributions = pd.DataFrame()
weight_distributions['Description'] = pd.Series({
    'deterministic': 'Deterministic cross-coupling: the incoming links to each node are equally weighted',
    'uniform': 'Uniform cross-coupling: the incoming links to each node are drawn at random from a uniform distribution on the (0, 1] interval'
})
weight_distributions['Required parameters'] = pd.Series({
    'deterministic': ('self_coupling', 'total_cross_coupling'),
    'uniform': ('self_coupling')
})

delay_distributions = pd.DataFrame()
delay_distributions['Description'] = pd.Series({
    'uniform': 'A random number of delay steps are drawn at random from a uniform distribution'
})
delay_distributions['Required parameters'] = pd.Series({
    'uniform': ('delay_links_n_max', 'delay_min', 'delay_max', 'delay_self')
})

node_dynamics_models = pd.DataFrame()
node_dynamics_models['Description'] = pd.Series({
    'AR_gaussian_discrete': 'Discrete autoregressive model with Gaussian noise',
    'logistic_map': 'Coupled logistic map model with Gaussian noise',
    'boolean_XOR': 'Every node computes the XOR of its inputs'
})
node_dynamics_models['Required parameters'] = pd.Series({
    'AR_gaussian_discrete': (
        'samples_n',
        'samples_transient_n',
        'replications',
        'noise_std'
    ),
    'logistic_map': (
        'samples_n',
        'samples_transient_n',
        'replications',
        'noise_std'
    ),
    'boolean_XOR': (
        'samples_n',
        'samples_transient_n',
        'replications'
    )
})


# Define custom error classes
class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class ParameterMissing(Error):
    """Exception raised for missing parameters.

    Attributes:
        par_names -- any sequence containing the names of the missing parameters
        msg  -- explanation of the error
    """

    def __init__(self, par_names, msg='ERROR: one or more parameters missing:'):
        self.par_names = par_names
        self.msg = msg


class ParameterValue(Error):
    """Raised when the provided parameter value is not valid.

    Attributes:
        par_value -- provided value
        msg  -- explanation of the error
    """

    def __init__(self, par_value, msg='ERROR: Invalid parameter values:'):
        self.par_value = par_value
        self.msg = msg


#
def generate_network(topology):
    try:
        # Ensure that a topology model has been specified
        if 'model' not in topology:
            raise ParameterMissing('model')        
        # Ensure that the provided model is implemented
        if topology.model not in topology_models.index:
            raise ParameterValue(topology.model)
        # Ensure that all the parameters required by the model have been provided
        par_required = topology_models['Required parameters'][topology.model]
        for par in par_required:
            if par not in topology:
                raise ParameterMissing(par)

    except ParameterMissing as e:
        print(e.msg, e.par_names)
        raise
    except ParameterValue as e:
        print(e.msg, e.par_value)
        raise

    else:
        model = topology.model
        if model == 'ER_n_p':
            # Read required parameters
            nodes_n = topology.nodes_n
            ER_p = topology.ER_p
            # Generate network
            return nx.gnp_random_graph(
                nodes_n,
                ER_p,
                seed=None,
                directed=True
            )
        elif model == 'ER_n_m':
            # Read required parameters
            nodes_n = topology.nodes_n
            ER_m = topology.ER_m
            # Generate network
            return nx.gnm_random_graph(
                nodes_n,
                ER_m,
                seed=None,
                directed=True
            )
        elif model == 'ER_n_in':
            # Read required parameters
            nodes_n = topology.nodes_n
            in_degree_expected = topology.in_degree_expected
            # Generate network
            ER_p = in_degree_expected / (nodes_n - 1)
            return nx.gnp_random_graph(
                nodes_n,
                ER_p,
                seed=None,
                directed=True
            )
        elif model == 'WS':
            # Read required parameters
            nodes_n = topology.nodes_n
            WS_k = topology.WS_k
            WS_p = topology.WS_p
            # Generate network
            return nx.connected_watts_strogatz_graph(
                nodes_n,
                WS_k,
                WS_p,
                tries=200,
                seed=None
            )
        else:
            raise ParameterValue(
                model,
                msg='Topology model not yet implemented'
            )


def generate_coupling(coupling, adjacency_matrix):
    try:
        # Ensure that a weight distribution has been specified
        if 'weight_distribution' not in coupling:
            raise ParameterMissing('weight_distribution')        
        # Ensure that the provided distribution is implemented
        if coupling.weight_distribution not in weight_distributions.index:
            raise ParameterValue(coupling.weight_distribution)
        # Ensure that all the required distribution parameters have been provided
        par_required = weight_distributions['Required parameters'][coupling.weight_distribution]
        for par in par_required:
            if par not in coupling:
                raise ParameterMissing(par)

    except ParameterMissing as e:
        print(e.msg, e.par_names)
        raise
    except ParameterValue as e:
        print(e.msg, e.par_value)
        raise

    else:
        # Read number of nodes
        nodes_n = len(adjacency_matrix)
        # Initialise coupling matrix as a copy of the adjacency matrix
        coupling_matrix = np.asarray(adjacency_matrix.copy(), dtype=float)
        # Temporarily remove self-loops
        np.fill_diagonal(coupling_matrix, 0)
        # Count links (excluding self-loops)
        links_count = (coupling_matrix > 0).sum()

        # Read distribution
        distribution = coupling.weight_distribution

        if distribution == 'deterministic':
            c = coupling.total_cross_coupling
            # Generate weights and normalise to total cross-coupling
            for node_id in range(0, nodes_n):
                column = coupling_matrix[:, node_id].copy()
                weights_sum_abs = np.abs(column.sum())
                if weights_sum_abs > 0:
                    coupling_matrix[:, node_id] = c * column / weights_sum_abs
            # Set weight of self-loops
            np.fill_diagonal(coupling_matrix, coupling.self_coupling)
            return coupling_matrix

        elif distribution == 'uniform':
            # Generate random coupling strenght matrix by uniformly sampling
            # from the [0,1] interval and normalizing to total_cross_coupling
            coupling_matrix[adjacency_matrix > 0] = np.random.rand(links_count)
            for node_id in range(0, nodes_n):
                column = coupling_matrix[:, node_id]
                weights_sum_abs = np.abs(column.sum())
                if weights_sum_abs > 0:
                    coupling_matrix[:, node_id] = c * column / weights_sum_abs
            # Set weight of self-loops
            np.fill_diagonal(coupling_matrix, coupling.self_coupling)
            return coupling_matrix

        else:
            raise ParameterValue(
                distribution,
                msg='Coupling weight distribution not yet implemented'
            )


def generate_delay(delay, adjacency_matrix):
    try:
        # Ensure that a distribution has been specified
        if 'distribution' not in delay:
            raise ParameterMissing('distribution')        
        # Ensure that the provided distribution is implemented
        if delay.distribution not in delay_distributions.index:
            raise ParameterValue(delay.distribution)
        # Ensure that all the required distribution parameters have been provided
        par_required = delay_distributions['Required parameters'][delay.distribution]
        for par in par_required:
            if par not in delay:
                raise ParameterMissing(par)

    except ParameterMissing as e:
        print(e.msg, e.par_names)
        raise
    except ParameterValue as e:
        print(e.msg, e.par_value)
        raise

    else:
        nodes_n = len(adjacency_matrix)
        distribution = delay.distribution

        if distribution == 'uniform':
            delay_min = delay.delay_min
            delay_max = delay.delay_max
            delay_self = delay.delay_self
            delay_links_n_max = delay.delay_links_n_max
            # Check consistency of parameters
            if delay.delay_min < 1:
                raise ParameterValue(par_value=delay_min, msg='ERROR: Minimum delay must be positive')
            if delay.delay_max < delay.delay_min:
                raise ParameterValue(par_value=delay_max, msg='ERROR: Maximum delay must be larger or equal to the minimum delay')
            if delay.delay_links_n_max > delay.delay_max - delay.delay_min + 1:
                raise ParameterValue(par_value=delay_links_n_max, msg='ERROR: Number of delay links must be smaller or equal to (max delay - min delay + 1)')

            # Generate random delay matrices by uniformly sampling integers from
            # the [delay_min,delay_max] interval
            delay_matrices = np.zeros((delay_max, nodes_n, nodes_n), dtype=int)
            for (x, y) in np.transpose(np.nonzero(adjacency_matrix > 0)):
                if x == y:
                    # Impose specific self-delay
                    delay_matrices[delay_self - 1, x, x] = 1
                else:
                    delay_values = np.random.choice(
                        np.arange(delay_min, delay_max + 1),
                        size=np.random.randint(1, delay_links_n_max + 1),
                        replace=False
                        ).astype(int)
                    delay_matrices[delay_values - 1, x, y] = 1
            return delay_matrices

        else:
            raise ParameterValue(
                par_value=distribution,
                msg='Delay distribution not yet implemented'
            )


def run_dynamics(dynamics, coefficient_matrices):
    try:
        # Ensure that a dynamical model has been specified
        if 'model' not in dynamics:
            raise ParameterMissing('model')
        # Ensure that the provided model is implemented
        if dynamics.model not in node_dynamics_models.index:
            raise ParameterValue(dynamics.model)
        # Ensure that all the parameters required by the model have been provided
        par_required = node_dynamics_models['Required parameters'][dynamics.model]
        for par in par_required:
            if par not in dynamics:
                raise ParameterMissing(par)

    except ParameterMissing as e:
        print(e.msg, e.par_names)
        raise
    except ParameterValue as e:
        print(e.msg, e.par_value)
        raise

    else:
        # Run dynamics

        nodes_n = np.shape(coefficient_matrices)[1]
        delay_max = np.shape(coefficient_matrices)[0]

        model = dynamics.model
        if model == 'AR_gaussian_discrete':

            # Read required parameters
            samples_n = dynamics.samples_n
            samples_transient_n = dynamics.samples_transient_n
            replications = dynamics.replications
            noise_std = dynamics.noise_std

            # Check stability of the VAR process, which is a sufficient condition
            # for stationarity.
            var_reduced_form = np.zeros((
                nodes_n * delay_max,
                nodes_n * delay_max
            ))
            var_reduced_form[0:nodes_n, :] = np.reshape(
                np.transpose(coefficient_matrices, (1, 0, 2)),
                [nodes_n, nodes_n * delay_max]
                )
            var_reduced_form[nodes_n:, 0:nodes_n * (delay_max - 1)] = np.eye(
                nodes_n * (delay_max - 1)
            )
            # Condition for stability: the absolute values of all the eigenvalues
            # of the reduced-form coefficeint matrix are smaller than 1. A stable
            # VAR process is also stationary.
            is_stable = max(np.abs(np.linalg.eigvals(var_reduced_form))) < 1
            if not is_stable:
                RuntimeError('The VAR process is not stable and may be nonstationary.')

            # Initialise time series matrix
            # The 3 dimensions represent (processes, samples, replications)
            x = np.zeros((
                nodes_n,
                delay_max + samples_transient_n + samples_n,
                replications
            ))

            # Generate (different) initial conditions for each replication:
            # Uniformly sample from the [0,1] interval and tile as many
            # times as delay_max along the second dimension
            x[:, 0:delay_max, :] = np.tile(
                np.random.rand(nodes_n, 1, replications),
                (1, delay_max, 1)
            )

            # Generate time series
            for i_repl in range(0, replications):
                for i_sample in range(delay_max, delay_max + samples_transient_n + samples_n):
                    for i_delay in range(1, delay_max + 1):
                        x[:, i_sample, i_repl] += np.dot(
                            coefficient_matrices[i_delay - 1, :, :],
                            x[:, i_sample - i_delay, i_repl]
                        )
                    # Add uncorrelated Gaussian noise vector
                    x[:, i_sample, i_repl] += np.random.normal(
                        0,  # mean
                        noise_std,
                        x[:, i_sample, i_repl].shape
                    )

            # Discard transient effects (only take end of time series)
            time_series = x[:, -(samples_n + 1):-1, :]

        elif model == 'logistic_map':

            # Define activation function
            def f(x):
                return 4 * x * (1 - x)

            # Read required parameters
            samples_n = dynamics.samples_n
            samples_transient_n = dynamics.samples_transient_n
            replications = dynamics.replications
            noise_std = dynamics.noise_std

            # Initialise time series matrix
            # The 3 dimensions represent (processes, samples, replications)
            x = np.zeros((
                nodes_n,
                delay_max + samples_transient_n + samples_n,
                replications
            ))

            # Generate (different) initial conditions for each replication:
            # Uniformly sample from the [0,1] interval and tile as many
            # times as delay_max along the second dimension
            x[:, 0:delay_max, :] = np.tile(
                np.random.rand(nodes_n, 1, replications),
                (1, delay_max, 1)
            )

            # Generate time series
            for i_repl in range(0, replications):
                for i_sample in range(delay_max, delay_max + samples_transient_n + samples_n):
                    for i_delay in range(1, delay_max + 1):
                        x[:, i_sample, i_repl] += np.dot(
                            coefficient_matrices[i_delay - 1, :, :],
                            x[:, i_sample - i_delay, i_repl]
                        )
                    # Compute activation function
                    x[:, i_sample, i_repl] = f(x[:, i_sample, i_repl])
                    # Add uncorrelated Gaussian noise vector
                    x[:, i_sample, i_repl] += np.random.normal(
                        0,  # mean
                        noise_std,
                        x[:, i_sample, i_repl].shape
                    )
                    # ensure values are in the [0, 1] range
                    x[:, i_sample, i_repl] = x[:, i_sample, i_repl] % 1

            # Discard transient effects (only take end of time series)
            time_series = x[:, -(samples_n + 1):-1, :]

        elif model == 'boolean_XOR':

            # Define activation function
            def f(x):
                return x % 2

            # binarize coupling matrices
            coefficient_matrices = np.where(coefficient_matrices > 0, 1, 0)

            # Read required parameters
            samples_n = dynamics.samples_n
            samples_transient_n = dynamics.samples_transient_n
            replications = dynamics.replications

            # Initialise time series matrix
            # The 3 dimensions represent (processes, samples, replications)
            x = np.zeros((
                nodes_n,
                delay_max + samples_transient_n + samples_n,
                replications
            ))

            # Generate (different) initial conditions for each replication:
            # Uniformly sample from the {0,1} set and tile as many
            # times as delay_max along the second dimension
            x[:, 0:delay_max, :] = np.tile(
                np.random.choice(np.array([0, 1]), size=(nodes_n, 1, replications)),
                (1, delay_max, 1)
            )

            # Generate time series
            for i_repl in range(0, replications):
                for i_sample in range(delay_max, delay_max + samples_transient_n + samples_n):
                    for i_delay in range(1, delay_max + 1):
                        x[:, i_sample, i_repl] += np.dot(
                            coefficient_matrices[i_delay - 1, :, :],
                            x[:, i_sample - i_delay, i_repl]
                        )
                    # Compute activation function
                    x[:, i_sample, i_repl] = f(x[:, i_sample, i_repl])

            # Discard transient effects (only take end of time series)
            time_series = x[:, -(samples_n + 1):-1, :]

        else:
            raise ParameterValue(model, msg='Dynamical model not yet implemented')    

        return time_series


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
    G = generate_network(traj.par.topology.initial)
    # Get adjacency matrix
    adjacency_matrix = np.array(nx.to_numpy_matrix(
        G,
        nodelist=np.array(range(0, traj.par.topology.initial.nodes_n)),
        dtype=int
    ))
    # Add self-loops
    np.fill_diagonal(adjacency_matrix, 1)

    # Generate initial node coupling
    coupling_matrix = generate_coupling(
        traj.par.node_coupling.initial,
        adjacency_matrix
    )

    # Generate delay
    delay_matrices = generate_delay(traj.par.delay.initial, adjacency_matrix)

    # Generate coefficient matrices
    coefficient_matrices = np.transpose(delay_matrices * coupling_matrix, (0, 2, 1))

    # Run dynamics
    time_series = run_dynamics(
        traj.par.node_dynamics,
        coefficient_matrices
    )

    # Create settings dictionary
    print('network_inference.p_value = {0}\n'.format(
        traj.par.network_inference.p_value))
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
        'alpha_fdr': traj.par.network_inference.p_value
    }

#    #load data
#    #samples_n = traj.par.node_dynamics.samples_n
#    subject_label = traj.par.subject_label
#    print('analysing subject: {}'.format(subject_label))
#    mat = spio.loadmat(
#        '/project/RDS-FEI-InfoDynFuncStruct-RW/Leo/inference/ASD/patients/' + subject_label + '_eyesclosed_array.mat',
#        squeeze_me=True
#    )
#    time_series = mat["series_array"]#[:, :, 1:101]

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
    job_walltime_hours = 6#1 + int(np.ceil((nodes_n + 20) / 30))
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
