import numpy as np
import pandas as pd
import networkx as nx


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


#
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
    # Define parameter options dictionaries
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
    # Define parameter options dictionaries
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
    # Define parameter options dictionaries
    delay_distributions = pd.DataFrame()
    delay_distributions['Description'] = pd.Series({
        'uniform': 'A random number of delay steps are drawn at random from a uniform distribution'
    })
    delay_distributions['Required parameters'] = pd.Series({
        'uniform': ['delay_links_n_max', 'delay_min', 'delay_max', 'delay_self']
    })
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


def evolve_network():
    # Define parameter options dictionaries
    topology_evolution_models = pd.DataFrame()
    topology_evolution_models['Description'] = pd.Series({
        'static': 'Static network topology'
    })
    topology_evolution_models['Required parameters'] = pd.Series({
        'static': []
    })


def run_dynamics(dynamics, coefficient_matrices):
    # Define parameter options dictionaries
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
