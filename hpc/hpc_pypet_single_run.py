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


def pickle_mTE(obj):
    print("pickling a C instance...")
    return MultivariateTE, (obj.max_lag_sources, obj.min_lag_sources, obj.options)


# Make MultivariateTE pickable, reference:
# https://docs.python.org/3/library/copyreg.html
copyreg.pickle(MultivariateTE, pickle_mTE)


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
        'cmi_estimator',
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
    'AR_gaussian_discrete': ('samples_n', 'samples_transient_n', 'noise_amplitude'),
    'logistic_map': ('samples_n', 'samples_transient_n', 'noise_amplitude'),
    'boolean_XOR': ('samples_n', 'samples_transient_n')
})


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
        coupling_matrix = adjacency_matrix.copy()
        # Temporarily remove self-loops
        np.fill_diagonal(coupling_matrix.values, 0)
        # Count links (excluding self-loops)
        links_count = (coupling_matrix > 0).values.sum()
        
        # Read distribution
        distribution = coupling.weight_distribution

        if distribution == 'deterministic':
            # Ensure that the conditions for stationarity are met, see reference:
            # Fatihcan M. Atay, Özkan Karabacak, "Stability of Coupled Map Networks with Delays",
            # SIAM Journal on Applied Dynamical Systems, Vol. 5, No. 3. (2006), pp. 508-527
            c = coupling.total_cross_coupling
            b = coupling.self_coupling + coupling.total_cross_coupling
            # first condition: |b|<1
            if np.abs(b) >= 1:
                raise ValueError('ERROR: absolute value of (self coupling + total cross coupling >= 1')
            # second condition: |b-2c|<1
            if np.abs(b - 2 * c) >= 1:
                raise ValueError('ERROR: absolute value of (self coupling - total cross coupling) >= 1')
            # Generate weights and normalise to total cross-coupling
            for node_id in range(0, nodes_n):
                column = coupling_matrix.iloc[:, node_id]
                weights_sum = column.sum()
                if weights_sum > 0:
                    coupling_matrix.iloc[:, node_id] = c * column / weights_sum
            # Set weight of self-loops
            np.fill_diagonal(coupling_matrix.values, coupling.self_coupling)
            return coupling_matrix

        elif distribution == 'uniform':
            # Generate random coupling strenght matrix by uniformly sampling
            # from the [0,1] interval and normalizing to total_cross_coupling
            coupling_matrix[adjacency_matrix > 0] = np.random.rand(links_count)
            for node_id in range(0, nodes_n):
                column = coupling_matrix.iloc[:, node_id]
                weights_sum = column.sum()
                if weights_sum > 0:
                    coupling_matrix.iloc[:, node_id] = c * column / weights_sum
            # Set weight of self-loops
            np.fill_diagonal(coupling_matrix.values, coupling.self_coupling)
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
            if delay.delay_max <= delay.delay_min:
                raise ParameterValue(par_value=delay_max, msg='ERROR: Maximum delay must be bigger than minimum delay')
            if delay.delay_links_n_max > delay.delay_max - delay.delay_min + 1:
                raise ParameterValue(par_value=delay_links_n_max, msg='ERROR: Number of delay links must be bigger than (max delay - min delay)')

            # Generate random delay matrix by uniformly sampling integers from
            # the [delay_min,delay_max] interval
            delay_matrix = pd.DataFrame([([] for _ in range(nodes_n)) for _ in range(nodes_n)])
            for (x, y) in np.transpose(np.nonzero(adjacency_matrix.values > 0)):
                delay_matrix.iloc[x, y] = np.random.choice(
                    np.arange(delay_min, delay_max),
                    size=np.random.randint(1, delay_links_n_max),
                    replace=False
                    ).astype(int)
            # Impose specific self-delay
            for x in range(nodes_n):
                delay_matrix.iloc[x, x] = np.array([delay_self]).astype(int)
            #np.fill_diagonal(delay_matrix.values, [delay_self])
            # Convert to integer numbers (they will be used as indices and need to be integers)
            #delay_matrix = delay_matrix.astype(int)
            return delay_matrix

        else:
            raise ParameterValue(
                par_value=distribution,
                msg='Delay distribution not yet implemented'
            )


def run_dynamics(dynamics, adjacency_matrix, coupling_matrix, delay_matrix):
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

        nodes_n = len(adjacency_matrix)
        delay_max = max(delay_matrix)

        # Initialise time series matrix to return
        #time_series = np.empty((nodes_n, samples_n))
        #time_series.fill(numpy.nan)
        
        model = dynamics.model
        if model == 'AR_gaussian_discrete':

            # Read required parameters
            samples_n = dynamics.samples_n
            samples_transient_n = dynamics.samples_transient_n
            noise_amplitude = dynamics.noise_amplitude

            # Initialise time series matrix
            # NOTE: By choice, the first dimension will represent processes and
            # the second dimension samples
            x = np.zeros((nodes_n, delay_max + samples_transient_n + samples_n))

            # Generate initial conditions:
            # Uniformly sample from the [0,1] interval
            # and replicate as many times as delay_max
            x[:, 0:delay_max] = np.tile(np.random.rand(nodes_n, 1), delay_max)

            for i_sample in range(delay_max, delay_max + samples_transient_n + samples_n):
                for i_target in range(0, nodes_n):
                    sources = np.flatnonzero(adjacency_matrix.values[:, i_target])
                    past_all_weighted_sum = 0
                    for source in sources:
                        past_single = x[source, i_sample - delay_matrix.iloc[source, i_target]]
                        past_single_weighted_sum = np.sum(past_single * coupling_matrix.values[source, i_target] / len(past_single))
                        past_all_weighted_sum += past_single_weighted_sum
                    x[i_target, i_sample] = past_all_weighted_sum + np.random.normal() * noise_amplitude

            # Skip samples to ensure the removal of transient effects
            # (only take end of time series)
            time_series = x[:, -(samples_n + 1):-1]

            #for i_sample in range(delay_max, delay_max + samples_transient_n + samples_n):
            #    for i_target in range(0, nodes_n):
            #        sources = adjacency_matrix.values[:, i_target] > 0
            #        past = x[sources, i_sample - delay_matrix.values[sources, i_target].astype(int)[0]] #<-----REMOVE [0] AND REPLACE WITH A LOOP OVER SOURCES
            #        x[i_target, i_sample] = np.inner(past, coupling_matrix.values[sources, i_target]) + np.random.normal() * noise_amplitude

        elif model == 'logistic_map':

            # Read required parameters
            samples_n = dynamics.samples_n
            samples_transient_n = dynamics.samples_transient_n
            noise_amplitude = dynamics.noise_amplitude

            # Initialise time series matrix
            # NOTE: By choice, the first dimension will represent processes and
            # the second dimension samples
            x = np.zeros((nodes_n, delay_max + samples_transient_n + samples_n))

            # Generate initial conditions:
            # Uniformly sample from the [0,1] interval
            # and replicate as many times as delay_max
            x[:, 0:delay_max] = np.tile(np.random.rand(nodes_n, 1), delay_max)


            # Define activation function
            def f(x):
                return 4 * x * (1 - x)

            for i_sample in range(delay_max, delay_max + samples_transient_n + samples_n):
                for i_target in range(0, nodes_n):
                    sources = np.flatnonzero(adjacency_matrix.values[:, i_target])
                    past_all_weighted_sum = 0
                    for source in sources:
                        past_single = x[source, i_sample - delay_matrix.iloc[source, i_target]]
                        past_single_weighted_sum = np.sum(past_single * coupling_matrix.values[source, i_target] / len(past_single))
                        past_all_weighted_sum += past_single_weighted_sum
                    x[i_target, i_sample] = (f(past_all_weighted_sum) + np.random.normal() * noise_amplitude) % 1

            # Skip samples to ensure the removal of transient effects
            # (only take end of time series)
            time_series = x[:, -(samples_n + 1):-1]

        elif model == 'boolean_XOR':

            # Read required parameters
            samples_n = dynamics.samples_n
            samples_transient_n = dynamics.samples_transient_n

            # Initialise time series matrix
            # NOTE: By choice, the first dimension will represent processes and
            # the second dimension samples
            x = np.zeros((nodes_n, delay_max + samples_transient_n + samples_n))

            # Generate initial conditions:
            # Uniformly sample from the {0,1} set
            # and replicate as many times as delay_max
            x[:, 0:delay_max] = np.tile(np.random.choice(np.array([0,1]), size=(nodes_n, 1)), delay_max)

            # Define activation function
            def f(x):
                return x % 2

            for i_sample in range(delay_max, delay_max + samples_transient_n + samples_n):
                for i_target in range(0, nodes_n):
                    sources = np.flatnonzero(adjacency_matrix.values[:, i_target])
                    past_all_sum = 0
                    for source in sources:
                        past_single = x[source, i_sample - delay_matrix.iloc[source, i_target]]
                        past_single_sum = np.sum(past_single)
                        past_all_sum += past_single_sum
                    x[i_target, i_sample] = f(past_all_sum)

            # Skip samples to ensure the removal of transient effects
            # (only take end of time series)
            time_series = x[:, -(samples_n + 1):-1]

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
    adjacency_matrix = pd.DataFrame(nx.to_numpy_matrix(G, nodelist=np.array(range(0, nodes_n)), dtype=int))
    # Add self-loops
    np.fill_diagonal(adjacency_matrix.values, 1)

    # Generate initial node coupling
    coupling_matrix = generate_coupling(traj.par.node_coupling.initial, adjacency_matrix)

    # Generate delay
    delay_matrix = generate_delay(traj.par.delay.initial, adjacency_matrix)

    # Run dynamics
    time_series = run_dynamics(traj.par.node_dynamics, adjacency_matrix, coupling_matrix, delay_matrix)

    # Save objects to disk
    adjacency_matrix.to_csv(os.path.join(traj_dir, '.'.join([traj.v_crun, 'topology.initial.adjacency_matrix', 'csv'])))
    coupling_matrix.to_csv(os.path.join(traj_dir, '.'.join([traj.v_crun, 'node_coupling.initial.coupling_matrix', 'csv'])))
    delay_matrix.to_csv(os.path.join(traj_dir, '.'.join([traj.v_crun, 'delay.initial.delay_matrix', 'csv'])))
    np.save(os.path.join(traj_dir, '.'.join([traj.v_crun, 'node_dynamics.time_series', 'npy'])), time_series)

    # Path to PBS script
    job_script_path = os.path.join(traj_dir, 'run_python_script.pbs')

    # Run job array
    job_walltime_hours = 10
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
