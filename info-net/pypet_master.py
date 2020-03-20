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
from idtxl.bivariate_mi import BivariateMI
from idtxl.bivariate_te import BivariateTE
from idtxl.multivariate_mi import MultivariateMI
from idtxl.multivariate_te import MultivariateTE
from idtxl.estimators_jidt import JidtGaussianTE
from idtxl.estimators_jidt import JidtDiscreteTE
from idtxl.data import Data
from idtxl.results import Results
from idtxl.stats import network_fdr
from scoop import futures
import itertools
import scipy.io as spio
import jpype

# Define parameter options dictionaries
network_inference_algorithms = pd.DataFrame()
network_inference_algorithms['Description'] = pd.Series({
    'bMI_greedy': 'Bivariate Mutual Information via greedy algorithm',
    'bTE_greedy': 'Bivariate Transfer Entropy via greedy algorithm',
    'mMI_greedy': 'Multivariate Mutual Information via greedy algorithm',
    'mTE_greedy': 'Multivariate Transfer Entropy via greedy algorithm',
    'cross_corr': 'Cross-correlation thresholding algorithm'
})
network_inference_algorithms['Required parameters'] = pd.Series({
    'bMI_greedy': [
        'min_lag_sources',
        'max_lag_sources',
        'tau_sources',
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
    ],
    'bTE_greedy': [
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
    ],
    'mMI_greedy': [
        'min_lag_sources',
        'max_lag_sources',
        'tau_sources',
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
    ],
    'mTE_greedy': [
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
    ],
    'cross_corr': [
        'min_lag_sources',
        'max_lag_sources'
    ]
})

topology_models = pd.DataFrame()
topology_models['Description'] = pd.Series({
    'ER_n_p': 'Erdős–Rényi model with a fixed number of nodes and link probability',
    'ER_n_m': 'Erdős–Rényi model with a fixed number of nodes and links',
    'ER_n_in': 'Erdős–Rényi model with a fixed number of nodes and expected in-degree',
    'WS': 'Watts–Strogatz model',
    'BA': 'Barabási–Albert model',
    'planted_partition': 'Planted partition model',
    'ring': 'Ring model',
    'lattice': 'Regular lattice model',
    'star': 'Star model',
    'complete': 'Complete graph model',
    'explicit': 'Explicit topology provided by the user'
})
topology_models['Required parameters'] = pd.Series({
    'ER_n_p': ['nodes_n', 'ER_p'],
    'ER_n_m': ['nodes_n', 'ER_m'],
    'ER_n_in': ['nodes_n', 'in_degree_expected'],
    'WS': ['nodes_n', 'WS_k', 'WS_p'],
    'BA': ['nodes_n', 'BA_m'],
    'planted_partition': ['nodes_n', 'partitions_n', 'p_in', 'p_out'],
})

topology_evolution_models = pd.DataFrame()
topology_evolution_models['Description'] = pd.Series({
    'static': 'Static network topology'
})
topology_evolution_models['Required parameters'] = pd.Series({
    'static': []
})

weight_distributions = pd.DataFrame()
weight_distributions['Description'] = pd.Series({
    'deterministic': 'Deterministic cross-coupling: the incoming links to each node are equally weighted',
    'random': 'Uniform cross-coupling: the incoming links to each node are drawn at random from a uniform distribution on the (0, 1] interval',
    'fixed': 'Use the same coupling for all links'
})
weight_distributions['Required parameters'] = pd.Series({
    'deterministic': ['self_coupling', 'total_cross_coupling'],
    'random': ['self_coupling'],
    'fixed': ['fixed_coupling']
})

delay_distributions = pd.DataFrame()
delay_distributions['Description'] = pd.Series({
    'uniform': 'A random number of delay steps are drawn at random from a uniform distribution'
})
delay_distributions['Required parameters'] = pd.Series({
    'uniform': ['delay_links_n_max', 'delay_min', 'delay_max', 'delay_self']
})

node_dynamics_models = pd.DataFrame()
node_dynamics_models['Description'] = pd.Series({
    'AR_gaussian_discrete': 'Discrete autoregressive model with Gaussian noise',
    'logistic_map': 'Coupled logistic map model with Gaussian noise',
    'boolean_mod2': 'A node is active if the number of its active parents is odd',
    'boolean_proportional': 'The activation probability of a node is the percentage of its active parents',
    'boolean_random_fixed_indegree': 'Random Boolean networks',
    'boolean_random': 'Random Boolean networks',
})
node_dynamics_models['Required parameters'] = pd.Series({
    'AR_gaussian_discrete': [
        'samples_n',
        'samples_transient_n',
        'replications',
        'noise_std',
    ],
    'logistic_map': [
        'samples_n',
        'samples_transient_n',
        'replications',
        'noise_std',
    ],
    'boolean_mod2': [
        'samples_n',
        'samples_transient_n',
        'replications',
        'noise_flip_p',
    ],
    'boolean_proportional': [
        'samples_n',
        'samples_transient_n',
        'replications',
        'noise_flip_p',
    ],
    'boolean_random_fixed_indegree': [
        'samples_n',
        'samples_transient_n',
        'replications',
        'RBN_in_degree',
        'RBN_r',
        'noise_flip_p',
    ],
    'boolean_random': [
        'samples_n',
        'samples_transient_n',
        'replications',
        'RBN_r',
        'noise_flip_p',
    ],
})


# Define custom error classes
class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class ParameterMissing(Error):
    """Exception raised for missing parameters.

    Attributes:
        par_names -- any sequence containing the missing parameter names
        msg  -- explanation of the error
    """

    def __init__(self, par_names, msg='ERROR: one or more parameters missing'):
        self.par_names = par_names
        self.msg = msg


class ParameterValue(Error):
    """Raised when the provided parameter value is not valid.

    Attributes:
        par_value -- provided value
        msg  -- explanation of the error
    """

    def __init__(self, par_value, msg='ERROR: Invalid parameter values'):
        self.par_value = par_value
        self.msg = msg


def watts_strogatz_graph(n, k, p):
    """Returns a Watts–Strogatz small-world graph.
    Parameters
    ----------
    n : int
        The number of nodes
    k : int
        Each node is joined with its `k` nearest neighbors in a ring
        topology.
    p : float
        The probability of rewiring each edge
    """
    if k >= n:
        raise nx.NetworkXError("k>=n, choose smaller k or larger n")

    G = nx.DiGraph()
    nodes = list(range(n))  # nodes are labeled 0 to n-1
    # connect each node to k/2 neighbors
    for j in range(1, k // 2 + 1):
        targets = nodes[j:] + nodes[0:j]  # first j nodes are now last in list
        G.add_edges_from(zip(nodes, targets))  # forward links
        G.add_edges_from(zip(targets, nodes))  # backward links
    # rewire edges from each node
    # loop over all nodes in order (label) and neighbors in order (distance)
    # no self loops or multiple edges allowed
    for j in range(1, k // 2 + 1):  # outer loop is over neighbors
        targets = nodes[j:] + nodes[0:j]  # first j nodes are now last in list
        # inner loop in node order
        for s, d in zip(nodes + targets, targets + nodes):  # forward+backward
            if np.random.random() < p:
                new_s = np.random.choice(nodes)
                # Enforce no self-loops or multiple edges
                while new_s == d or G.has_edge(new_s, d):
                    new_s = np.random.choice(nodes)
                    if G.in_degree(s) >= n - 1:
                        break  # skip this rewiring
                else:
                    G.remove_edge(s, d)
                    G.add_edge(new_s, d)
                    # print('old link {0}->{1} replaced by {2}->{1}'.format(
                    #     s, d, new_s))
    return G


def connected_watts_strogatz_graph(n, k, p, tries=100):
    """Returns a connected Watts–Strogatz small-world graph.
    Attempts to generate a connected graph by repeated generation of
    Watts–Strogatz small-world graphs.  An exception is raised if the maximum
    number of tries is exceeded.
    Parameters
    ----------
    n : int
        The number of nodes
    k : int
        Each node is joined with its `k` nearest neighbors in a ring
        topology.
    p : float
        The probability of rewiring each edge
    tries : int
        Number of attempts to generate a connected graph.
    -----
    """
    for i in range(tries):
        G = watts_strogatz_graph(n, k, p)
        if nx.is_strongly_connected(G):
            return G
    # raise nx.NetworkXError('Maximum number of tries exceeded')
    print('Warning: Maximum number of tries exceeded: the network '
          'is NOT strongly connected')
    return G


def directed_barabasi_albert_graph(n, m, seed=None):
    """Returns a random graph according to the Barabási–Albert preferential
    attachment model.
    A graph of $n$ nodes is grown by attaching new nodes each with $m$
    edges that are preferentially attached to existing nodes with high degree.
    Parameters
    ----------
    n : int
        Number of nodes
    m : int
        Number of edges to attach from a new node to existing nodes
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    Returns
    -------
    G : DiGraph
    Raises
    ------
    NetworkXError
        If `m` does not satisfy ``1 <= m < n``.
    References
    ----------
    .. [1] A. L. Barabási and R. Albert "Emergence of scaling in
       random networks", Science 286, pp 509-512, 1999.
    """

    if m < 1 or m >= n:
        raise nx.NetworkXError("Barabási–Albert network must have m >= 1"
                               " and m < n, m = %d, n = %d" % (m, n))

    # Add m initial nodes (m0 in barabasi-speak)
    G = nx.DiGraph()
    G.add_nodes_from([i for i in range(m)])
    # Target nodes for new edges
    targets = list(range(m))
    # List of existing nodes, with nodes repeated once for each adjacent edge
    repeated_nodes = []
    # Start adding the other n-m nodes. The first node is m.
    source = m
    while source < n:
        # Add edges to m nodes from the source.
        G.add_edges_from(zip([source] * m, targets))
        # Add one node to the list for each new edge just created.
        repeated_nodes.extend(targets)
        # And the new node "source" has m edges to add to the list.
        repeated_nodes.extend([source] * m)
        # Now choose m unique nodes from the existing nodes
        # Pick uniformly from repeated_nodes (preferential attachment)
        targets = set()
        while len(targets) < m:
            x = np.random.choice(repeated_nodes)
            targets.add(x)
        source += 1
    return G


#
def generate_network(topology):
    try:
        # Ensure that a topology model has been specified
        if 'model' not in topology:
            raise ParameterMissing('model')
        # Ensure that the provided model is implemented
        if topology.model not in topology_models.index:
            raise ParameterValue(topology.model)
        # Ensure that all the parameters required by the model
        # have been provided
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
            #return nx.connected_watts_strogatz_graph(
            return connected_watts_strogatz_graph(
                nodes_n,
                WS_k,
                WS_p,
                tries=200
            )
        elif model == 'BA':
            # Read required parameters
            nodes_n = topology.nodes_n
            BA_m = topology.BA_m
            # Generate network
            #return directed_barabasi_albert_graph(
            return nx.barabasi_albert_graph(
                nodes_n,
                BA_m
            )
        elif model == 'planted_partition':
            # Read required parameters
            nodes_n = topology.nodes_n
            partitions_n = topology.partitions_n
            p_in = topology.p_in
            p_out = topology.p_out
            # Generate network
            return nx.planted_partition_graph(
                partitions_n,
                int(nodes_n / partitions_n),
                p_in,
                p_out,
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
                weights_sum_abs = np.abs(column.sum())
                if weights_sum_abs > 0:
                    coupling_matrix[:, node_id] = coupling.total_cross_coupling * column / weights_sum_abs
            # Set weight of self-loops
            np.fill_diagonal(coupling_matrix, coupling.self_coupling)
            return coupling_matrix

        elif distribution == 'random':
            # Generate random coupling strenght matrix by uniformly sampling
            # from the [0,1] interval and normalizing to total_cross_coupling
            coupling_matrix[coupling_matrix > 0] = np.random.rand(links_count)
            for node_id in range(0, nodes_n):
                column = coupling_matrix[:, node_id]
                weights_sum_abs = np.abs(column.sum())
                if weights_sum_abs > 0:
                    coupling_matrix[:, node_id] = coupling.total_cross_coupling * column / weights_sum_abs
            # Set weight of self-loops
            np.fill_diagonal(coupling_matrix, coupling.self_coupling)
            return coupling_matrix

        elif distribution == 'fixed':
            # All the links have the same weight
            coupling_matrix = coupling_matrix * coupling.fixed_coupling
            # Set weight of self-loops
            np.fill_diagonal(coupling_matrix, coupling.fixed_coupling)
            return coupling_matrix

        else:
            raise ParameterValue(
                distribution,
                msg='Coupling weight distribution not yet implemented')


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
                raise RuntimeError('VAR process is not stable and may be nonstationary.')

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

        elif model == 'boolean_mod2':

            samples_n = dynamics.samples_n
            samples_transient_n = dynamics.samples_transient_n
            replications = dynamics.replications
            noise_flip_p = dynamics.noise_flip_p

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
                            x[:, i_sample - i_delay, i_repl])
                    # Compute activation function
                    x[:, i_sample, i_repl] = f(x[:, i_sample, i_repl])
                    # Flip stochastically (noise)
                    flip_ids = np.random.rand(nodes_n) < noise_flip_p
                    x[flip_ids, i_sample, i_repl] = 1 - x[flip_ids, i_sample, i_repl]

            # Discard transient effects (only take end of time series)
            time_series = x[:, -(samples_n + 1):-1, :].astype(int)

        elif model == 'boolean_proportional':

            samples_n = dynamics.samples_n
            samples_transient_n = dynamics.samples_transient_n
            replications = dynamics.replications
            noise_flip_p = dynamics.noise_flip_p

            # Define activation function
            def f(x):
                probs = x / in_degrees
                # Make first link harder to find than the following ones
                probs[probs <= (2 / in_degrees)] /= 2
                probs *= 0.75
                y = np.less(np.random.rand(len(probs)), probs)
                return y.astype(int)

            # binarize coupling matrices
            coefficient_matrices = np.where(coefficient_matrices > 0, 1, 0)
            in_degrees = coefficient_matrices.sum(axis=(0, 2))

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
                            x[:, i_sample - i_delay, i_repl])
                    # Compute activation function
                    x[:, i_sample, i_repl] = f(x[:, i_sample, i_repl])
                    # Flip stochastically (noise)
                    flip_ids = np.random.rand(nodes_n) < noise_flip_p
                    x[flip_ids, i_sample, i_repl] = 1 - x[flip_ids, i_sample, i_repl]

            # Discard transient effects (only take end of time series)
            time_series = x[:, -(samples_n + 1):-1, :].astype(int)

        elif model == 'boolean_random_fixed_indegree':

            # Read required parameters
            samples_n = dynamics.samples_n
            samples_transient_n = dynamics.samples_transient_n
            replications = dynamics.replications
            noise_flip_p = dynamics.noise_flip_p
            in_degree = dynamics.RBN_in_degree
            RBN_r = dynamics.RBN_r

            # Number of rows in the Boolean table
            table_rows_n = 2 ** in_degree
            # Create random Boolean outcome for each row and each target
            bool_functions = np.random.rand(table_rows_n, nodes_n) <= RBN_r

            # Define activation function
            def f(input_array, target):
                if not input_array.shape == tuple([in_degree]):
                    raise RuntimeError('Expected indegree {0}, got {1}'.format(
                        tuple([in_degree]), input_array.shape))
                # Convert to decimal to get row index
                row_i = np.sum(
                    [b * (2 ** i) for (i, b) in enumerate(input_array[::-1])])
                return bool_functions[row_i, target]

            # Convert coupling matrices to Boolean values
            coefficient_matrices = coefficient_matrices > 0

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
            )).astype(bool)

            # Generate (different) initial conditions for each replication:
            # Uniformly sample from Boolean values and tile as many
            # times as delay_max along the second dimension
            x[:, 0:delay_max, :] = np.tile(
                np.random.rand(nodes_n, 1, replications) > 0.5,
                (1, delay_max, 1)
            )

            # Generate time series
            for i_repl in range(0, replications):
                for i_sample in range(delay_max, delay_max + samples_transient_n + samples_n):
                    # i_delay = 1  # only use lag 1 for boolean dynamics
                    for t in range(nodes_n):
                        # parents = coefficient_matrices[i_delay - 1, t, :]
                        parents = coefficient_matrices[:, t, :]  # parents for all lags
                        parents = parents.T  # transpose
                        parents = np.fliplr(parents)  # flip left to right
                        if np.max(np.sum(parents, axis=1)) > 1:
                            raise RuntimeError('More than one lag found for one of the parents of node {0}'.format(t))
                        # Get parent values at the corresponding lags
                        # input_array = x[parents, i_sample - i_delay, i_repl]
                        input_array = x[:, i_sample-delay_max:i_sample, i_repl][parents]
                        # Compute activation function
                        x[t, i_sample, i_repl] = f(input_array, target=t)
                        # Flip stochastically (noise)
                        if np.random.rand() < noise_flip_p:
                            x[t, i_sample, i_repl] = not(x[t, i_sample, i_repl])
            # Discard transient effects (only take end of time series)
            time_series = x[:, -(samples_n + 1):-1, :].astype(int)

        elif model == 'boolean_random':

            # Read required parameters
            samples_n = dynamics.samples_n
            samples_transient_n = dynamics.samples_transient_n
            replications = dynamics.replications
            noise_flip_p = dynamics.noise_flip_p
            RBN_r = dynamics.RBN_r

            bool_functions = {}

            # Define activation function
            def f(input_array, target):
                key = hash((tuple(input_array), target))
                if key not in bool_functions:
                    bool_functions[key] = np.random.choice(
                        [True, False],
                        p=[RBN_r, 1 - RBN_r])
                return bool_functions[key]

            # Convert coupling matrices to Boolean values
            coefficient_matrices = coefficient_matrices > 0

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
            )).astype(bool)

            # Generate (different) initial conditions for each replication:
            # Uniformly sample from Boolean values and tile as many
            # times as delay_max along the second dimension
            x[:, 0:delay_max, :] = np.tile(
                np.random.rand(nodes_n, 1, replications) > 0.5,
                (1, delay_max, 1)
            )

            # Generate time series
            for i_repl in range(0, replications):
                for i_sample in range(delay_max, delay_max + samples_transient_n + samples_n):
                    # i_delay = 1  # only use lag 1 for boolean dynamics
                    for t in range(nodes_n):
                        # parents = coefficient_matrices[i_delay - 1, t, :]
                        parents = coefficient_matrices[:, t, :]  # parents for all lags
                        parents = parents.T  # transpose
                        parents = np.fliplr(parents)  # flip left to right
                        if np.max(np.sum(parents, axis=1)) > 1:
                            raise RuntimeError('More than one lag found for one of the parents of node {0}'.format(t))
                        # Get parent values at the corresponding lags
                        # input_array = x[parents, i_sample - i_delay, i_repl]
                        input_array = x[:, i_sample-delay_max:i_sample, i_repl][parents]
                        # Compute activation function
                        x[t, i_sample, i_repl] = f(input_array, target=t)
                        # Flip stochastically (noise)
                        if np.random.rand() < noise_flip_p:
                            x[t, i_sample, i_repl] = not(x[t, i_sample, i_repl])
            # Discard transient effects (only take end of time series)
            time_series = x[:, -(samples_n + 1):-1, :].astype(int)

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

        if len(time_series.shape) == 2:
            dim_order = 'ps'
        else:
            dim_order = 'psr'

        # initialise an empty data object
        dat = Data()

        # Load time series
        dat = Data(
            time_series,
            dim_order=dim_order,
            normalise=(network_inference.z_standardise & can_be_z_standardised))

        algorithm = network_inference.algorithm
        if algorithm in ['bMI_greedy', 'mMI_greedy', 'bTE_greedy', 'mTE_greedy']:
            # Set analysis options
            if algorithm == 'bMI_greedy':
                network_analysis = BivariateMI()
            if algorithm == 'mMI_greedy':
                network_analysis = MultivariateMI()
            if algorithm == 'bTE_greedy':
                network_analysis = BivariateTE()
            if algorithm == 'mTE_greedy':
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
                # correspinding to one map iteration
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
        dtype=int))
    # Add self-loops
    np.fill_diagonal(adjacency_matrix, 1)

    # Generate initial node coupling
    coupling_matrix = generate_coupling(
        traj.par.node_coupling.initial,
        adjacency_matrix)

    # Generate delay
    delay_matrices = generate_delay(traj.par.delay.initial, adjacency_matrix)

    # Generate coefficient matrices
    coefficient_matrices = np.transpose(
        delay_matrices * coupling_matrix,
        (0, 2, 1))

    # Run dynamics
    time_series = run_dynamics(
        traj.par.node_dynamics,
        coefficient_matrices)

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
        traj.config.parallel_target_analysis)

    # Compute elapsed time
    # end_monotonic = time.monotonic()
    # end_perf_counter = time.perf_counter()
    # end_process_time = time.process_time()
    # timing_df = pd.DataFrame(
    #     index=['monotonic', 'perf_counter', 'process_time'],
    #     columns=['start', 'end', 'resolution'])
    # timing_df.loc['monotonic'] = [
    #     start_monotonic, end_monotonic,
    #     time.get_clock_info('monotonic').resolution]
    # timing_df.loc['perf_counter'] = [
    #     start_perf_counter,
    #     end_perf_counter,
    #     time.get_clock_info('perf_counter').resolution]
    # timing_df.loc['process_time'] = [
    #     start_process_time,
    #     end_process_time,
    #     time.get_clock_info('process_time').resolution]
    # timing_df['elapsed'] = timing_df['end'] - timing_df['start']

    # Add results to the trajectory
    # The wildcard character $ will be replaced by the name of the current run,
    # formatted as `run_XXXXXXXX`
    traj.f_add_result(
        '$.topology.initial',
        adjacency_matrix=adjacency_matrix,
        comment='')
    traj.f_add_result(
        '$.node_coupling.initial',
        coupling_matrix=coupling_matrix,
        coefficient_matrices=coefficient_matrices,
        comment='')
    traj.f_add_result(
        '$.delay.initial',
        delay_matrices=delay_matrices,
        comment='')
    traj.f_add_result(
        '$.node_dynamics',
        time_series=time_series,
        comment='')
    traj.f_add_result(
        PickleResult,
        '$.network_inference',
        network_inference_result=network_inference_result,
        comment='')
    # traj.f_add_result('$.timing', timing=timing_df, comment='')

    # Suggest Java garbage collector to run
    jSystem = jpype.JPackage("java.lang").System
    jSystem.gc()

    # return the analysis result
    # return network_inference_result


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
        log_stdout=True,  # log everything that is printed, will make the log file HUGE
        filename=traj_dir,  # filename or just folder (name will be automatic in this case)
        multiproc=False,

        #use_pool=True,
        #ncores=10,
        #freeze_input=True,

        use_scoop=False,
        #wrap_mode=pypetconstants.WRAP_MODE_LOCAL,
        memory_cap=1,
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
    traj.f_add_parameter('network_inference.max_lag_sources', 1, comment='')
    traj.f_add_parameter('network_inference.tau_sources', 1, comment='')
    traj.f_add_parameter('network_inference.max_lag_target', 1, comment='')
    traj.f_add_parameter('network_inference.tau_target', 1, comment='')
    #traj.f_add_parameter('network_inference.cmi_estimator', 'JidtDiscreteCMI', comment='Conditional Mutual Information estimator')
    traj.f_add_parameter('network_inference.cmi_estimator', 'JidtGaussianCMI', comment='Conditional Mutual Information estimator')
    #traj.f_add_parameter('network_inference.cmi_estimator', 'JidtKraskovCMI', comment='Conditional Mutual Information estimator')
    #traj.f_add_parameter('network_inference.cmi_estimator', 'OpenCLKraskovCMI', comment='Conditional Mutual Information estimator')
    traj.f_add_parameter('network_inference.permute_in_time', False, comment='')
    traj.f_add_parameter('network_inference.jidt_threads_n', 1, comment='Number of threads used by JIDT estimator (default=USE_ALL)')
    traj.f_add_parameter('network_inference.n_perm_max_stat', 2000, comment='')
    traj.f_add_parameter('network_inference.n_perm_min_stat', 2000, comment='')
    traj.f_add_parameter('network_inference.n_perm_omnibus', 2000, comment='')
    traj.f_add_parameter('network_inference.n_perm_max_seq', 2000, comment='')
    traj.f_add_parameter('network_inference.fdr_correction', False, comment='')
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
    # traj.f_add_parameter('topology.initial.model', 'ER_n_in')
    #traj.f_add_parameter('topology.initial.model', 'BA')
    #traj.f_add_parameter('topology.initial.model', 'WS')
    traj.f_add_parameter('topology.initial.model', 'planted_partition')
    traj.parameters.f_get('topology.initial.model').v_comment = topology_models['Description'].get(traj.parameters['topology.initial.model'])
    traj.f_add_parameter('topology.initial.nodes_n', 5, comment='Number of nodes')
    # traj.f_add_parameter('topology.initial.in_degree_expected', 3, comment='Expected in-degree')
    #traj.f_add_parameter('topology.initial.WS_k', 4, comment='Number of neighbours (and mean degree) in the Watts-Strogatz model')
    #traj.f_add_parameter('topology.initial.WS_p', 0.0, comment='Rewiring probability in the Watts-Strogatz model')
    #traj.f_add_parameter('topology.initial.BA_m', 1, comment='Number of edges to attach from a new node to existing nodes in the Barabási–Albert model')
    traj.f_add_parameter('topology.initial.partitions_n', 5, comment='Number of partitions in the planted partition model')
    traj.f_add_parameter('topology.initial.p_in', 0.5, comment='Probability of connecting vertices within a group in the planted partition model')
    traj.f_add_parameter('topology.initial.p_out', 0.01, comment='Probability of connecting vertices between groups in the planted partition model')


    # -------------------------------------------------------------------
    # Parameters characterizing the evolution of the topology
    traj.f_add_parameter('topology.evolution.model', 'static')
    traj.parameters.f_get('topology.evolution.model').v_comment = topology_evolution_models['Description'].get(traj.parameters['topology.evolution.model'])

    # -------------------------------------------------------------------
    # Parameters characterizing the coupling between the nodes
    traj.f_add_parameter('node_coupling.initial.model', 'linear', comment='Linear coupling model: the input to each target node is the weighted sum of the outputs of its source nodes')
    traj.f_add_parameter('node_coupling.initial.weight_distribution', 'fixed')
    traj.parameters.f_get('node_coupling.initial.weight_distribution').v_comment = weight_distributions['Description'].get(traj.parameters['node_coupling.initial.weight_distribution'])
    traj.f_add_parameter('node_coupling.initial.fixed_coupling', 0.1)

    # -------------------------------------------------------------------
    # Parameters characterizing the delay
    traj.f_add_parameter('delay.initial.distribution', 'uniform')
    traj.parameters.f_get('delay.initial.distribution').v_comment = delay_distributions['Description'].get(traj.parameters['delay.initial.distribution'])
    traj.f_add_parameter('delay.initial.delay_links_n_max', 1, comment='Maximum number of delay links')
    traj.f_add_parameter('delay.initial.delay_min', 1, comment='')
    traj.f_add_parameter('delay.initial.delay_max', 1, comment='')
    traj.f_add_parameter('delay.initial.delay_self', 1, comment='')

    # -------------------------------------------------------------------
    # Parameters characterizing the dynamics of the nodes
    #traj.f_add_parameter('node_dynamics.model', 'logistic_map')
    traj.f_add_parameter('node_dynamics.model', 'AR_gaussian_discrete')
    #traj.f_add_parameter('node_dynamics.model', 'boolean_random')
    traj.parameters.f_get('node_dynamics.model').v_comment = node_dynamics_models['Description'].get(traj.parameters['node_dynamics.model'])
    traj.f_add_parameter('node_dynamics.samples_n', 100, comment='Number of samples (observations) to record')
    traj.f_add_parameter('node_dynamics.samples_transient_n', 1000 * traj.topology.initial.nodes_n, comment='Number of initial samples (observations) to skip to leave out the transient')
    traj.f_add_parameter('node_dynamics.replications', 1, comment='Number of replications (trials) to record')
    traj.f_add_parameter('node_dynamics.noise_std', 0.1, comment='Standard deviation of Gaussian noise')
    #traj.f_add_parameter('node_dynamics.RBN_in_degree', 4, comment='Indegree for random boolean network dynamics')
    #traj.f_add_parameter('node_dynamics.noise_flip_p', 0.005, comment='Probability of flipping bit in Boolean dynamics')

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
            'network_inference.algorithm': ['bMI_greedy', 'bTE_greedy', 'mTE_greedy'],
            #'node_coupling.initial.weight_distribution': ['fixed'],
            'repetition_i': np.arange(0, 5, 1).tolist(),
            'topology.initial.nodes_n': np.arange(50, 50+1, 300).tolist(),
            'node_dynamics.samples_n': np.array([1000, 10000]).tolist(),
            'network_inference.p_value': np.array([0.001]).tolist(),
            #'node_coupling.initial.self_coupling': np.arange(-0.5, 0.5 + 0.001, 0.1).tolist(),
            #'node_coupling.initial.total_cross_coupling': np.arange(-1., 1 + 0.001, 0.2).tolist(),
            #'topology.initial.WS_p': np.around(np.logspace(-2.2, 0, 10), decimals=4).tolist(),
        },
        (
            'network_inference.algorithm',
            #'node_coupling.initial.weight_distribution',
            'network_inference.p_value',
            'node_dynamics.samples_n',
            'topology.initial.nodes_n',
            #'topology.initial.WS_p',
            #'node_coupling.initial.self_coupling',
            #'node_coupling.initial.total_cross_coupling',
            'repetition_i',
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
