import sys
import os
from datetime import datetime
import time
import numpy as np
import pandas as pd
from pypet import Environment
from pypet import PickleResult
from pypet import pypetconstants
from pypet.utils.explore import cartesian_product
import networkx as nx
from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data
from idtxl.results import Results
from idtxl.stats import network_fdr
from scoop import futures
import itertools
import scipy.io as spio

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
        'z_standardise',
        'p_value'
        #'alpha_max_stats',
        #'alpha_min_stats',
        #'alpha_omnibus',
        #'alpha_max_seq',
        #'alpha_fdr'
    ),
     'bivariateTE': (
        'min_lag_sources',
        'max_lag_sources'
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
    'ER_n_in': ('nodes_n', 'in_degree_expected')
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
            # Generate weights and normalise to total cross-coupling
            for node_id in range(0, nodes_n):
                column = coupling_matrix[:, node_id]
                weights_sum = column.sum()
                if weights_sum > 0:
                    coupling_matrix[:, node_id] = coupling.total_cross_coupling * column / weights_sum
            # Set weight of self-loops
            np.fill_diagonal(coupling_matrix, coupling.self_coupling)
            return coupling_matrix

        elif distribution == 'uniform':
            # Generate random coupling strenght matrix by uniformly sampling
            # from the [0,1] interval and normalizing to total_cross_coupling
            coupling_matrix[adjacency_matrix > 0] = np.random.rand(links_count)
            for node_id in range(0, nodes_n):
                column = coupling_matrix[:, node_id]
                weights_sum = column.sum()
                if weights_sum > 0:
                    coupling_matrix[:, node_id] = coupling.total_cross_coupling * column / weights_sum
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
                RuntimeError('VAR process is not stable and may be nonstationary.')

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


def perform_network_inference(network_inference, time_series, parallel_target_analysis=False):
    try:
        # Ensure that a network inference algorithm has been specified
        if 'algorithm' not in network_inference:
            raise ParameterMissing('algorithm')        
        # Ensure that the provided algorithm is implemented
        if network_inference.algorithm not in network_inference_algorithms.index:
            raise ParameterValue(network_inference.algorithm)
        # Ensure that all the parameters required by the algorithm have been provided
        par_required = network_inference_algorithms['Required parameters'][network_inference.algorithm]
        for par in par_required:
            if par not in network_inference:
                raise ParameterMissing(par)

    except ParameterMissing as e:
        print(e.msg, e.par_names)
        raise
    except ParameterValue as e:
        print(e.msg, e.par_value)
        raise

    else:
        nodes_n = np.shape(time_series)[0]

        can_be_z_standardised = True
        if network_inference.z_standardise:
            # Check if data can be normalised per process (assuming the
            # first dimension represents processes, as in the rest of the code)
            can_be_z_standardised = np.all(np.std(time_series, axis=1) > 0)
            if not can_be_z_standardised:
                print('Time series can not be z-standardised')

        # initialise an empty data object
        dat = Data()

        # Load time series
        dat = Data(time_series, dim_order='psr', normalise=(network_inference.z_standardise & can_be_z_standardised))

        algorithm = network_inference.algorithm
        if algorithm == 'mTE_greedy':
            # Set analysis options
            network_analysis = MultivariateTE()

            settings = {
                'min_lag_sources': network_inference.min_lag_sources,
                'max_lag_sources': network_inference.max_lag_sources,
                'tau_sources': network_inference.tau_sources,
                'max_lag_target': network_inference.max_lag_target,
                'tau_target': network_inference.tau_target,
                'cmi_estimator':  network_inference.cmi_estimator,
                'kraskov_k': network_inference.kraskov_k,
                'num_threads': network_inference.jidt_threads_n,
                'permute_in_time': network_inference.permute_in_time,
                'n_perm_max_stat': network_inference.n_perm_max_stat,
                'n_perm_min_stat': network_inference.n_perm_min_stat,
                'n_perm_omnibus': network_inference.n_perm_omnibus,
                'n_perm_max_seq': network_inference.n_perm_max_seq,
                'fdr_correction': network_inference.fdr_correction,
                'alpha_max_stat': network_inference.p_value,
                'alpha_min_stat': network_inference.p_value,
                'alpha_omnibus': network_inference.p_value,
                'alpha_max_seq': network_inference.p_value,
                'alpha_fdr': network_inference.p_value
            }

            # # Add optional settings
            # optional_settings_keys = {
            #     'config.debug',
            #     'config.max_mem_frac'
            # }

            # for key in optional_settings_keys:
            #     if traj.f_contains(key, shortcuts=True):
            #         key_last = key.rpartition('.')[-1]
            #         settings[key_last] = traj[key]
            #         print('Using optional setting \'{0}\'={1}'.format(
            #             key_last,
            #             traj[key])
            #         )

            if parallel_target_analysis:
                # Use SCOOP to create a generator of map results, each
                # correspinding to one map ieration
                #my_function = network_analysis.analyse_single_target
                res_iterator = futures.map_as_completed(
                    network_analysis.analyse_single_target,
                    itertools.repeat(settings, nodes_n),
                    itertools.repeat(dat, nodes_n),
                    list(range(nodes_n))
                )
                # Run analysis
                res_list = list(res_iterator)
                if settings['fdr_correction']:
                    res = network_fdr(
                        {'alpha_fdr': settings['alpha_fdr']},
                        *res_list
                    )
                else:
                    res = res_list[0]
                    res.combine_results(*res_list[1:])
            else:
                # Run analysis
                res = network_analysis.analyse_network(
                    settings=settings,
                    data=dat
                )
            return res

        else:
            raise ParameterValue(algorithm, msg='Network inference algorithm not yet implemented')


def information_network_inference(traj):
    """Runs Information Network inference

    :param traj:

        Container with all parameters.

    :return:

        Inferred Information Network

    """

    # Start timer
    start_monotonic = time.monotonic()
    start_perf_counter = time.perf_counter()
    start_process_time = time.process_time()

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

#    # Load ASD data
#    samples_n = traj.node_dynamics.samples_n
#    subject_label = traj.subject_label
#    print('analysing subject: {}'.format(subject_label))
#    mat = spio.loadmat(
#        #'C:\\DATA\\Google Drive\\Materiale progetti in corso\\USyd\\Information network inference\\ASD\\patients\\' + subject_label + '_eyesclosed_array.mat',
#        '/home/leo/Projects/inference/ASD/patients/' + subject_label + '_eyesclosed_array.mat',
#        squeeze_me=True
#    )
#    time_series = mat["series_array"]#[:, -samples_n:, :]

    # Perform Information Network Inference
    network_inference_result = perform_network_inference(
        traj.par.network_inference,
        time_series,
        traj.config.parallel_target_analysis
    )

    # Compute elapsed time
    end_monotonic = time.monotonic()
    end_perf_counter = time.perf_counter()
    end_process_time = time.process_time()
    timing_df = pd.DataFrame(
        index=['monotonic', 'perf_counter', 'process_time'],
        columns=['start', 'end', 'resolution']
    )
    timing_df.loc['monotonic'] = [
        start_monotonic, end_monotonic,
        time.get_clock_info('monotonic').resolution
    ]
    timing_df.loc['perf_counter'] = [
        start_perf_counter,
        end_perf_counter,
        time.get_clock_info('perf_counter').resolution
    ]
    timing_df.loc['process_time'] = [
        start_process_time,
        end_process_time,
        time.get_clock_info('process_time').resolution
    ]
    timing_df['elapsed'] = timing_df['end'] - timing_df['start']

    # Add results to the trajectory
    # The wildcard character $ will be replaced by the name of the current run,
    # formatted as `run_XXXXXXXX`
    traj.f_add_result(
        '$.topology.initial',
        adjacency_matrix=adjacency_matrix,
        comment=''
    )
    traj.f_add_result(
        '$.node_coupling.initial',
        coupling_matrix=coupling_matrix,
        coefficient_matrices=coefficient_matrices,
        comment=''
    )
    traj.f_add_result(
        '$.delay.initial',
        delay_matrices=delay_matrices,
        comment=''
    )
    traj.f_add_result(
        '$.node_dynamics',
        time_series=time_series,
        comment=''
    )
    traj.f_add_result(
        PickleResult,
        '$.network_inference',
        network_inference_result=network_inference_result,
        comment=''
    )
    traj.f_add_result('$.timing', timing=timing_df, comment='')

    # return the analysis result
    return network_inference_result


def main():
    """Main function to protect the *entry point* of the program.

    If you want to use multiprocessing with SCOOP you need to wrap your
    main code creating an environment into a function. Otherwise
    the newly started child processes will re-execute the code and throw
    errors (also see http://scoop.readthedocs.org/en/latest/usage.html#pitfalls).

    """

    # Get current directory
    traj_dir = os.getcwd()
    # Read output path (if provided)
    if len(sys.argv) > 1:
        # Only use specified folder if it exists
        if os.path.isdir(sys.argv[1]):
            # Get name of directory
            traj_dir = os.path.dirname(sys.argv[1])
            # Convert to full path
            traj_dir = os.path.abspath(traj_dir)
    # Add time stamp (final '' is to make sure there is a trailing slash)
    traj_dir = os.path.join(traj_dir, datetime.now().strftime("%Y_%m_%d_%Hh%Mm%Ss"), '')
    # Create directory with time stamp
    os.makedirs(traj_dir)
    # Change current directory to the one containing the trajectory files
    os.chdir(traj_dir)
    print('Trajectory and results will be stored to: {0}'.format(traj_dir))

    # # Start timer
    # start_monotonic = time.monotonic()
    # start_perf_counter = time.perf_counter()
    # start_process_time = time.process_time()

    # Create an environment that handles running.
    # Let's enable multiprocessing with scoop:
    env = Environment(
        trajectory='traj',
        comment='Experiment to quantify the discrepancy between '
                'the inferred information network and the original '
                'network structure.',
        add_time=False,
        log_config='DEFAULT',
        log_stdout=True, # log everything thst is printed, will make the log file HUGE
        filename=traj_dir,  #filename or just folder(name will be automatic in this case)
        multiproc=True,

        #use_pool=True,
        #ncores=10,
        
        #freeze_input=True,

        use_scoop=True,
        wrap_mode=pypetconstants.WRAP_MODE_LOCAL,
        memory_cap=10,
        swap_cap=1
        #cpu_cap=30

        #,git_repository='' #path to the root git folder. The commit code will be added in the trajectory
        #,git_fail=True #no automatic commits
        #,sumatra_project='' #path to sumatra root folder,
        #graceful_exit=True
    )

    traj = env.trajectory

    # -------------------------------------------------------------------
    # Add config parameters (those that DO NOT influence the final result of the experiment)
    traj.f_add_config('parallel_target_analysis', True, comment='Analyse targets in parallel')
    #traj.f_add_config('debug', False, comment='Activate debug mode')
    #traj.f_add_config('max_mem_frac', 0.7, comment='Fraction of global GPU memory to use')

    # -------------------------------------------------------------------
    # Add "proper" parameters (those that DO influence the final result of the experiment)

    # -------------------------------------------------------------------
    # Parameters characterizing the network inference algorithm
    traj.f_add_parameter('network_inference.algorithm', 'mTE_greedy')
    traj.parameters.f_get('network_inference.algorithm').v_comment = network_inference_algorithms['Description'].get(traj.parameters['network_inference.algorithm'])
    traj.f_add_parameter('network_inference.min_lag_sources', 1, comment='')
    traj.f_add_parameter('network_inference.max_lag_sources', 5, comment='')
    traj.f_add_parameter('network_inference.tau_sources', 1, comment='')
    traj.f_add_parameter('network_inference.max_lag_target', 5, comment='')
    traj.f_add_parameter('network_inference.tau_target', 1, comment='')
    traj.f_add_parameter('network_inference.cmi_estimator', 'JidtGaussianCMI', comment='Conditional Mutual Information estimator')
    #traj.f_add_parameter('network_inference.cmi_estimator', 'JidtKraskovCMI', comment='Conditional Mutual Information estimator')
    #traj.f_add_parameter('network_inference.cmi_estimator', 'OpenCLKraskovCMI', comment='Conditional Mutual Information estimator')
    traj.f_add_parameter('network_inference.permute_in_time', False, comment='')
    traj.f_add_parameter('network_inference.jidt_threads_n', 1, comment='Number of threads used by JIDT estimator (default=USE_ALL)')
    traj.f_add_parameter('network_inference.n_perm_max_stat', 200, comment='')
    traj.f_add_parameter('network_inference.n_perm_min_stat', 200, comment='')
    traj.f_add_parameter('network_inference.n_perm_omnibus', 500, comment='')
    traj.f_add_parameter('network_inference.n_perm_max_seq', 200, comment='')
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
    traj.parameters.f_get('topology.initial.model').v_comment = topology_models['Description'].get(traj.parameters['topology.initial.model'])
    traj.f_add_parameter('topology.initial.nodes_n', 5, comment='Number of nodes')
    traj.f_add_parameter('topology.initial.in_degree_expected', 3, comment='Expected in-degree')

    # -------------------------------------------------------------------
    # Parameters characterizing the evolution of the topology
    traj.f_add_parameter('topology.evolution.model', 'static')
    traj.parameters.f_get('topology.evolution.model').v_comment = topology_evolution_models['Description'].get(traj.parameters['topology.evolution.model'])

    # -------------------------------------------------------------------
    # Parameters characterizing the coupling between the nodes
    traj.f_add_parameter('node_coupling.initial.model', 'linear', comment='Linear coupling model: the input to each target node is the weighted sum of the outputs of its source nodes')
    traj.f_add_parameter('node_coupling.initial.weight_distribution', 'deterministic')
    traj.parameters.f_get('node_coupling.initial.weight_distribution').v_comment = weight_distributions['Description'].get(traj.parameters['node_coupling.initial.weight_distribution'])
    traj.f_add_parameter('node_coupling.initial.self_coupling', 0.5, comment='The self-coupling is the weight of the self-loop')
    traj.f_add_parameter('node_coupling.initial.total_cross_coupling', 0.4, comment='The total cross-coupling is the sum of all incoming weights from the sources only')

    # -------------------------------------------------------------------
    # Parameters characterizing the delay
    traj.f_add_parameter('delay.initial.distribution', 'uniform')
    traj.parameters.f_get('delay.initial.distribution').v_comment = delay_distributions['Description'].get(traj.parameters['delay.initial.distribution'])
    traj.f_add_parameter('delay.initial.delay_links_n_max', 1, comment='Maximum number of delay links')
    traj.f_add_parameter('delay.initial.delay_min', 1, comment='')
    traj.f_add_parameter('delay.initial.delay_max', 5, comment='')
    traj.f_add_parameter('delay.initial.delay_self', 1, comment='')

    # -------------------------------------------------------------------
    # Parameters characterizing the dynamics of the nodes
    traj.f_add_parameter('node_dynamics.model', 'logistic_map')
    #traj.f_add_parameter('node_dynamics.model', 'AR_gaussian_discrete')
    traj.parameters.f_get('node_dynamics.model').v_comment = node_dynamics_models['Description'].get(traj.parameters['node_dynamics.model'])
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
            'node_dynamics.samples_n': (10 ** np.arange(4, 4+0.1, 1)).round().astype(int).tolist(),
            'network_inference.p_value': np.array([0.05]).tolist(),
            #'node_coupling.initial.self_coupling': np.array([0.2]).tolist(),
            #'node_coupling.initial.total_cross_coupling': np.array([0.85]).tolist()
        },
        ('repetition_i',
         'topology.initial.nodes_n',
         'node_dynamics.samples_n',
         'network_inference.p_value'
         #'node_coupling.initial.self_coupling',
         #'node_coupling.initial.total_cross_coupling'
        )
    )
#    explore_dict={
#        'subject_label': ['A02']
#    }
    print(explore_dict)
    traj.f_explore(explore_dict)

    # -------------------------------------------------------------------
    # Run the experiment
    env.run(information_network_inference)

    # # Compute total elapsed time
    # end_monotonic = time.monotonic()
    # end_perf_counter = time.perf_counter()
    # end_process_time = time.process_time()
    # timing_df = pd.DataFrame(index=['monotonic', 'perf_counter', 'process_time'], columns=['start', 'end', 'resolution'])
    # timing_df.loc['monotonic'] = [start_monotonic, end_monotonic, time.get_clock_info('monotonic').resolution]
    # timing_df.loc['perf_counter'] = [start_perf_counter, end_perf_counter, time.get_clock_info('perf_counter').resolution]
    # timing_df.loc['process_time'] = [start_process_time, end_process_time, time.get_clock_info('process_time').resolution]
    # timing_df['elapsed'] = timing_df['end'] - timing_df['start']
    # # Add total timing to trajectory
    # traj.f_add_result('timing', timing_total=timing_df, comment='')

    # Check that all runs are completed
    assert traj.f_is_completed()

    # Finally disable logging and close all log-files
    env.disable_logging()


if __name__ == '__main__':
    # This will execute the main function in case the script is called from the one true
    # main process and not from a child processes spawned by your environment.
    # Necessary for multiprocessing under Windows.
    main()