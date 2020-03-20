from pypet import Trajectory
import os
import numpy as np
from scipy.linalg import block_diag
from scipy.linalg import solve_discrete_lyapunov
import pandas as pd
import networkx as nx
import itertools
from collections import Counter
from mylib_pypet import print_leaves

fdr = False
debug_mode = False
save_results = True

# Load the trajectory from the hdf5 file
# Only load parameters, results will be loaded at runtime (auto loading)
traj_dir = os.path.join('trajectories', 'BA_GC_on_VAR_200nodes_1000_and_10000samples_m1_coupling0.1')
if not os.path.isdir(traj_dir):
    traj_dir = os.path.join('..', traj_dir)
traj_filename = 'traj.hdf5'
traj_fullpath = os.path.join(traj_dir, traj_filename)
traj = Trajectory()
traj.f_load(
    filename=traj_fullpath,
    index=0,
    load_parameters=2,
    load_results=0,
    load_derived_parameters=0,
    force=True)
# Turn on auto loading
traj.v_auto_load = True

# Count number of runs
runs_n = len(traj.f_get_run_names())

# Get list of explored parameters
parameters_explored = [str.split(par, '.').pop() for par in (
    traj.f_get_explored_parameters())]

# Initialise analysis summary table
# (it is important that the columns with the explored parameters
# preceed the ones with the results)
df = pd.DataFrame(
    index=traj.f_get_run_names(),
    columns=parameters_explored + [
        'precision',
        'recall',
        'specificity',
        'false_pos_rate',
        'precision_per_target',
        'recall_per_target',
        'specificity_per_target',
        'false_pos_rate_per_target',
        'precision_per_distance',
        'recall_per_distance',
        'specificity_per_distance',
        'false_pos_rate_per_distance',
        'precision_per_motif',
        'recall_per_motif',
        'incorrect_target_rate',
        'density_real',
        'density_inferred',
        'delay_error_mean',
        'delay_error_mean_normalised',
        'TE_omnibus_empirical',
        'TE_complete_empirical',
        # 'TE_apparent_empirical',
        'TE_omnibus_theoretical_inferred_vars',
        'TE_complete_theoretical_inferred_vars',
        'TE_apparent_theoretical_inferred_vars',
        'AIS_theoretical_inferred_vars',
        'TE_omnibus_theoretical_causal_vars',
        'TE_complete_theoretical_causal_vars',
        'TE_apparent_theoretical_causal_vars',
        'AIS_theoretical_causal_vars',
        'spectral_radius',
        'in_degree_real',
        'in_degree_inferred',
        'out_degree_real',
        'out_degree_inferred',
        'clustering_real',
        'clustering_inferred',
        'average_shortest_path_length_real',
        'average_shortest_path_length_inferred',
        'local_efficiency_real',
        'local_efficiency_inferred',
        'average_global_efficiency_real',
        'average_global_efficiency_inferred',
        'rich_club_in_degrees_real',
        'rich_club_in_degrees_inferred',
        'rich_club_out_degrees_real',
        'rich_club_out_degrees_inferred',
        'in_degree_assortativity_real',
        'in_degree_assortativity_inferred',
        'out_degree_assortativity_real',
        'out_degree_assortativity_inferred',
        'reciprocity_real',
        'reciprocity_inferred',
        'overall_reciprocity_real',
        'overall_reciprocity_inferred',
        'time_elapsed_monotonic',
        'time_elapsed_perf_counter',
        'time_elapsed_process_time'
        ],
    dtype=object)
# Add other useful columns to the DataFrame
if 'algorithm' not in parameters_explored:
    df['algorithm'] = np.NaN
if 'weight_distribution' not in parameters_explored:
    df['weight_distribution'] = np.NaN

# Loop over runs
for run_name in traj.f_get_run_names():

    # Make trajectory behave like a particular single run:
    # all explored parameter’s values will be set to the corresponding
    # values of one particular run.
    traj.f_set_crun(run_name)

    print('post-processing of {0} in progress...'.format(run_name))

    # Fill in current explored parameter values
    for par in parameters_explored:
        df.loc[run_name, par] = traj.parameters[par]
        if debug_mode:
            print('{0} = {1}'.format(par, traj.parameters[par]))
    # Fill in additional useful parameter values
    if 'algorithm' not in parameters_explored:
        current_algorithm = traj.parameters['network_inference.algorithm']
        df.loc[run_name, 'algorithm'] = current_algorithm
        if debug_mode:
            print('algorithm added to DataFrame: {0}'.format(
                current_algorithm))
    if 'weight_distribution' not in parameters_explored:
        current_weight_distribution = traj.parameters[
            'node_coupling.initial.weight_distribution']
        df.loc[run_name, 'weight_distribution'] = current_weight_distribution
        if debug_mode:
            print('weight_distribution added to DataFrame: {0}'.format(
                current_weight_distribution))

    # Read elapsed time
    # df.loc[run_name, 'time_elapsed_monotonic'] = (
    #   traj.results[run_name].timing.elapsed['monotonic'])
    # df.loc[run_name, 'time_elapsed_perf_counter'] = (
    #   traj.results[run_name].timing.elapsed['perf_counter'])
    # df.loc[run_name, 'time_elapsed_process_time'] = (
    #   traj.results[run_name].timing.elapsed['process_time'])

    nodes_n = traj.parameters.topology.initial['nodes_n']

    # Load results object
    res = traj.results[run_name].network_inference.network_inference_result
    # Check if old release
    if type(res) == pd.core.frame.DataFrame:
        release_results_class = False
    else:
        release_results_class = True
    if debug_mode:
        print('release_results_class = {0}'.format(release_results_class))

    if release_results_class:
        # Get inferred adjacency matrix
        adjacency_matrix_inferred = res.get_adjacency_matrix(
            weights='binary',
            fdr=fdr)._weight_matrix.astype(float)
        # Get real adjacency matrix
        adjacency_matrix_real = np.asarray(
            traj.results[run_name].topology.initial.adjacency_matrix > 0,
            dtype=float).astype(float)
        # Get real and inferred delay matrices
        delay_matrices_real = (
            traj.results[run_name].delay.initial.delay_matrices
            ).astype(float)
        delay_matrices_inferred = res.get_time_series_graph(
            weights='binary',
            fdr=fdr).astype(float)
        # Get max delays
        delay_max_real = np.shape(delay_matrices_real)[0]
        delay_max_inferred = np.shape(delay_matrices_inferred)[0]
        # Load real coupling matrix
        coupling_matrix_real = (
            traj.results[run_name].node_coupling.initial.coupling_matrix)

    else:  # old release without results class
        # Covert pandas DataFrame to dictionary
        res = res.to_dict()
        # Reconstruct inferred delay matrices
        if fdr:
            try:
                r = res['fdr_corrected']
            except KeyError:
                raise RuntimeError('No FDR-corrected results found.')
        else:
            r = res.copy()
            try:
                del r['fdr_corrected']
            except KeyError:
                pass
        targets = list(r.keys())
        # Find max inferred delay
        delay_max_inferred = max(
            res[0]['max_lag_target'],
            res[0]['max_lag_sources'])
        # Fill in delay_matrices_inferred
        delay_matrices_inferred = np.zeros(
            (delay_max_inferred, nodes_n, nodes_n),
            dtype=float)
        for target in targets:
            all_vars = (
                r[target]['selected_vars_sources']
                + r[target]['selected_vars_target'])
            for (source, lag) in all_vars:
                delay_matrices_inferred[lag - 1, source, target] = 1.
        # Reconstruct inferred adjacency matrix
        adjacency_matrix_inferred = np.any(
            delay_matrices_inferred,
            axis=0).astype(float)
        # Load real delay matrix DataFrame
        delay_matrix_real_df = traj.results[
            run_name].delay.initial.delay_matrix
        # Convert pandas DataFrame to 3D numpy delay matrices
        if type(delay_matrix_real_df.iloc[0, 0]) == np.ndarray:
            # Find max delay
            delay_max_real = max(
                traj.par.delay.initial.delay_max,
                traj.par.network_inference.max_lag_sources)
            # Reconstruct real delay matrices
            delay_matrices_real = np.zeros(
                (delay_max_real, nodes_n, nodes_n),
                dtype=float)
            for x in range(nodes_n):
                for y in range(nodes_n):
                    if len(delay_matrix_real_df.iloc[x, y]) > 0:
                        for d in delay_matrix_real_df.iloc[x, y]:
                            delay_matrices_real[d - 1, x, y] = 1
        else:
            raise RuntimeError(
                'delay_matrix_real_df is not a numpy ndarray')
        # Reconstruct real adjacency matrix
        adjacency_matrix_real = np.any(
            delay_matrices_real,
            axis=0).astype(float)
        # Load real coupling matrix
        coupling_matrix_real = traj.results[
            run_name].node_coupling.initial.coupling_matrix.values

    # Remove self-loops from real and inferred adjacency matrices
    np.fill_diagonal(adjacency_matrix_real, 0)
    np.fill_diagonal(adjacency_matrix_inferred, 0)

    # Get networkx graphs
    G_real = nx.from_numpy_array(
        adjacency_matrix_real,
        parallel_edges=False,
        create_using=nx.DiGraph())
    G_inferred = nx.from_numpy_array(
        adjacency_matrix_inferred,
        parallel_edges=False,
        create_using=nx.DiGraph())

    if debug_mode:
        print('adjacency matrix real:\n {0}'.format(adjacency_matrix_real))
        print('adjacency matrix inf:\n {0}'.format(adjacency_matrix_inferred))
        print('coupling matrix real:\n {0}'.format(coupling_matrix_real))
        print('delay matrices real:\n {0}'.format(delay_matrices_real))
        print('delay matrices inf:\n {0}'.format(delay_matrices_inferred))

    # -------------------------------------------------------------------------
    # region Boolean classification performance measures
    # -------------------------------------------------------------------------

    # Compute density
    df.loc[run_name, 'density_real'] = (
        adjacency_matrix_real.sum() / (nodes_n * (nodes_n - 1)))
    df.loc[run_name, 'density_inferred'] = (
        adjacency_matrix_inferred.sum() / (nodes_n * (nodes_n - 1)))

    # Compute sum and difference between real and inferred adjacency
    # matrices
    adjacency_matrices_sum = (
        adjacency_matrix_real
        + adjacency_matrix_inferred)
    adjacency_matrices_diff = (
        adjacency_matrix_real
        - adjacency_matrix_inferred)
    # Prevent the removed self-loops to be counted as true negatives,
    # by setting the elements on the diagonal to NaN
    np.fill_diagonal(adjacency_matrices_sum, np.NaN)
    # Find true positives
    true_pos_matrix = adjacency_matrices_sum == 2
    true_pos = true_pos_matrix.sum(axis=0)
    true_pos_n = true_pos.sum()
    # Find true negatives
    true_neg_matrix = adjacency_matrices_sum == 0
    true_neg = true_neg_matrix.sum(axis=0)
    true_neg_n = true_neg.sum()
    # Find false positives
    false_pos_matrix = adjacency_matrices_diff == -1
    false_pos = false_pos_matrix.sum(axis=0)
    false_pos_n = false_pos.sum()
    # Find false negatives
    false_neg_matrix = adjacency_matrices_diff == 1
    false_neg = false_neg_matrix.sum(axis=0)
    false_neg_n = false_neg.sum()
    # Sanity check
    assert (true_pos_n + true_neg_n + false_pos_n + false_neg_n
            == nodes_n * (nodes_n - 1))
    # Compute classifier precision
    if true_pos_n + false_pos_n > 0:
        df.loc[run_name, 'precision'] = true_pos_n / (true_pos_n + false_pos_n)
    else:
        print('WARNING: Zero links were inferred: cannot compute precision')
        df.loc[run_name, 'precision'] = np.NaN
    # Compute target-wise precision
    precision_per_target = np.full(nodes_n, np.nan)
    for node in range(nodes_n):
        if true_pos[node] + false_pos[node] > 0:
            precision_per_target[node] = (
                true_pos[node] / (true_pos[node] + false_pos[node]))
    df.loc[run_name, 'precision_per_target'] = precision_per_target
    # Compute classifier recall
    if true_pos_n + false_neg_n > 0:
        df.loc[run_name, 'recall'] = true_pos_n / (true_pos_n + false_neg_n)
    else:
        print('WARNING: The real network is empty: cannot compute recall')
        df.loc[run_name, 'recall'] = np.NaN
    # Compute target-wise recall
    recall_per_target = np.full(nodes_n, np.nan)
    for node in range(nodes_n):
        if true_pos[node] + false_neg[node] > 0:
            recall_per_target[node] = (
                true_pos[node] / (true_pos[node] + false_neg[node]))
    df.loc[run_name, 'recall_per_target'] = recall_per_target
    # Compute classifier specificity and FPR
    if true_neg_n + false_pos_n > 0:
        df.loc[run_name, 'specificity'] = true_neg_n / (
            (true_neg_n + false_pos_n))
        df.loc[run_name, 'false_pos_rate'] = false_pos_n / (
            (true_neg_n + false_pos_n))
    else:
        print('WARNING: The real network is full: cannot compute'
              'specificity and false positive rate')
        df.loc[run_name, 'specificity'] = np.NaN
        df.loc[run_name, 'false_pos_rate'] = np.NaN
    # Compute target-wise specificity and FPR
    specificity_per_target = np.full(nodes_n, np.nan)
    false_pos_rate_per_target = np.full(nodes_n, np.nan)
    for node in range(nodes_n):
        if true_neg[node] + false_pos[node] > 0:
            specificity_per_target[node] = (
                true_neg[node] / (true_neg[node] + false_pos[node]))
            false_pos_rate_per_target[node] = (
                false_pos[node] / (true_neg[node] + false_pos[node]))
    df.loc[run_name, 'specificity_per_target'] = specificity_per_target
    df.loc[run_name, 'false_pos_rate_per_target'] = false_pos_rate_per_target

    # Compute false pos rate at taget level (tFPR)
    df.loc[run_name, 'incorrect_target_rate'] = (false_pos > 0).mean()

#     # Compute performance per distance
#     nn_max = np.floor(nodes_n / 2).astype(int)
#     precision_per_distance = np.full(nn_max, np.NaN)
#     recall_per_distance = np.full(nn_max, np.NaN)
#     specificity_per_distance = np.full(nn_max, np.NaN)
#     false_pos_rate_per_distance = np.full(nn_max, np.NaN)
#     for nn in range(1, nn_max + 1):
#         true_pos_diag = np.concatenate((
#             true_pos_matrix.diagonal(nn),
#             true_pos_matrix.diagonal(-nodes_n + nn),
#             true_pos_matrix.diagonal(nodes_n - nn),
#             true_pos_matrix.diagonal(-nn),
#             ))
#         assert true_pos_diag.shape[0] == nodes_n * 2
#         false_pos_diag = np.concatenate((
#             false_pos_matrix.diagonal(nn),
#             false_pos_matrix.diagonal(-nodes_n + nn),
#             false_pos_matrix.diagonal(nodes_n - nn),
#             false_pos_matrix.diagonal(-nn),
#             ))
#         assert false_pos_diag.shape[0] == nodes_n * 2
#         true_neg_diag = np.concatenate((
#             true_neg_matrix.diagonal(nn),
#             true_neg_matrix.diagonal(-nodes_n + nn),
#             true_neg_matrix.diagonal(nodes_n - nn),
#             true_neg_matrix.diagonal(-nn),
#             ))
#         assert true_neg_diag.shape[0] == nodes_n * 2
#         false_neg_diag = np.concatenate((
#             false_neg_matrix.diagonal(nn),
#             false_neg_matrix.diagonal(-nodes_n + nn),
#             false_neg_matrix.diagonal(nodes_n - nn),
#             false_neg_matrix.diagonal(-nn),
#             ))
#         assert false_neg_diag.shape[0] == nodes_n * 2
#         true_pos_diag_n = true_pos_diag.sum()
#         false_pos_diag_n = false_pos_diag.sum()
#         true_neg_diag_n = true_neg_diag.sum()
#         false_neg_diag_n = false_neg_diag.sum()
#         if true_pos_diag_n + false_pos_diag_n > 0:
#             precision_per_distance[nn - 1] = (
#                 true_pos_diag_n / (true_pos_diag_n + false_pos_diag_n))
#         if true_pos_diag_n + false_neg_diag_n > 0:
#             recall_per_distance[nn - 1] = (
#                 true_pos_diag_n / (true_pos_diag_n + false_neg_diag_n))
#         if true_neg_diag_n + false_pos_diag_n > 0:
#             specificity_per_distance[nn - 1] = (
#                 true_neg_diag_n / (true_neg_diag_n + false_pos_diag_n))
#             false_pos_rate_per_distance[nn - 1] = (
#                 false_pos_diag_n / (true_neg_diag_n + false_pos_diag_n))
#     df.loc[run_name, 'precision_per_distance'] = precision_per_distance
#     df.loc[run_name, 'recall_per_distance'] = recall_per_distance
#     df.loc[run_name, 'specificity_per_distance'] = specificity_per_distance
#     df.loc[run_name, 'false_pos_rate_per_distance'] = (
#         false_pos_rate_per_distance)
# 
#     def tricode(G, v, u, w):
#         """Returns the integer code of the given triad.
# 
#         This is some fancy magic that comes from Batagelj and Mrvar's paper. It
#         treats each edge joining a pair of `v`, `u`, and `w` as a bit in
#         the binary representation of an integer.
# 
#         """
#         combos = ((v, u, 1), (u, v, 2), (v, w, 4), (w, v, 8), (u, w, 16),
#                   (w, u, 32))
#         return sum(x for u, v, x in combos if v in G[u])
# 
#     def generate_motifs():
#         # Returns dictionary mapping triad names to triad graphs
#         def abc_graph():
#             return nx.empty_graph('abc', create_using=nx.DiGraph())
#         motifs = dict((name, abc_graph()) for name in motif_names)
#         motifs['012'].add_edges_from([('a', 'b')])
#         motifs['102'].add_edges_from([('a', 'b'), ('b', 'a')])
#         motifs['102'].add_edges_from([('a', 'b'), ('b', 'a')])
#         motifs['021D'].add_edges_from([('b', 'a'), ('b', 'c')])
#         motifs['021U'].add_edges_from([('a', 'b'), ('c', 'b')])
#         motifs['021C'].add_edges_from([('a', 'b'), ('b', 'c')])
#         motifs['111D'].add_edges_from([('a', 'c'), ('c', 'a'), ('b', 'c')])
#         motifs['111U'].add_edges_from([('a', 'c'), ('c', 'a'), ('c', 'b')])
#         motifs['030T'].add_edges_from([('a', 'b'), ('c', 'b'), ('a', 'c')])
#         motifs['030C'].add_edges_from([('b', 'a'), ('c', 'b'), ('a', 'c')])
#         motifs['201'].add_edges_from(
#             [('a', 'b'), ('b', 'a'), ('a', 'c'), ('c', 'a')])
#         motifs['120D'].add_edges_from(
#             [('b', 'c'), ('b', 'a'), ('a', 'c'), ('c', 'a')])
#         motifs['120C'].add_edges_from(
#             [('a', 'b'), ('b', 'c'), ('a', 'c'), ('c', 'a')])
#         motifs['120U'].add_edges_from(
#             [('a', 'b'), ('c', 'b'), ('a', 'c'), ('c', 'a')])
#         motifs['210'].add_edges_from(
#             [('a', 'b'), ('b', 'c'), ('c', 'b'), ('a', 'c'), ('c', 'a')])
#         motifs['300'].add_edges_from(
#             [('a', 'b'), ('b', 'a'), ('b', 'c'), ('c', 'b'),
#              ('a', 'c'), ('c', 'a')])
#         return motifs
# 
#     def find_motifs_slow(G, motifs):
#         """Find motifs in a directed graph
#         :param G: A ``DiGraph`` object
#         :param motifs: A ``dict`` of motifs to count
#         :returns: A ``dict`` of graphs, with the same keys as ``motifs``
#         This function extracts all 3-grams from the original graph and look
#         for isomorphisms with the motifs contained in the motifs dictionary.
#         """
#         # Ensure that the provided motifs are not isomorphic to each other
#         for pair in itertools.combinations(motifs.keys(), 2):
#             if nx.is_isomorphic(motifs[pair[0]], motifs[pair[1]]):
#                 raise(RuntimeError('motifs {0} and {1} are isomorphic'.format(
#                     pair[0], pair[1])))
#         # Create a dictionary of empty lists with the motif names as keys
#         motif_decomposition = {name: [] for name in motif_names}
#         # For all combinations of three nodes in the original graph,
#         # take the subgraph, and compare it to all of the possible motifs
#         for triplet in itertools.combinations(G.nodes(), 3):
#             sub_gr = G.subgraph(triplet)
#             count = 0
#             for motif_name, motif_graph in motifs.items():
#                 if nx.is_isomorphic(sub_gr, motif_graph):
#                     motif_decomposition[motif_name].append(sub_gr)
#                     count += 1
#             if count == 0:
#                 print('Triplet {0} is not isomorphic to any motifs.'.format(
#                     triplet))
#         return motif_decomposition
# 
#     def find_motifs(G, motif_names):
#         triad_nodes = {name: set([]) for name in motif_names}
#         # Assign a number to all the vertices
#         m = {v: i for i, v in enumerate(G)}
#         for v in G:
#             vnbrs = set(G.pred[v]) | set(G.succ[v])
#             for u in vnbrs:
#                 if m[u] > m[v]:
#                     unbrs = set(G.pred[u]) | set(G.succ[u])
#                     neighbors = (vnbrs | unbrs) - {u, v}
#                     not_neighbors = set(G.nodes()) - neighbors - {u, v}
#                     # Find dyadic triads
#                     for w in not_neighbors:
#                         if v in G[u] and u in G[v]:
#                             triad_nodes['102'].add(tuple(sorted([u, v, w])))
#                         else:
#                             triad_nodes['012'].add(tuple(sorted([u, v, w])))
#                     for w in neighbors:
#                         if m[u] < m[w] or (m[v] < m[w] < m[u] and
#                                            v not in G.pred[w] and
#                                            v not in G.succ[w]):
#                             code = tricode(G, v, u, w)
#                             triad_nodes[tricode_to_name[code]].add(
#                                 tuple(sorted([u, v, w])))
#         return triad_nodes
# 
#     # Define motifs of interest
#     motif_names = ("003", "012", "102", "021D", "021U", "021C", "111D", "111U",
#                    "030T", "030C", "201", "120D", "120U", "120C", "210", "300")
#     #: Triads that are the same up to symmetry have the same code.
#     tricodes = (1, 2, 2, 3, 2, 4, 6, 8, 2, 6, 5, 7, 3, 8, 7, 11, 2, 6, 4, 8,
#                 5, 9, 9, 13, 6, 10, 9, 14, 7, 14, 12, 15, 2, 5, 6, 7, 6, 9,
#                 10, 14, 4, 9, 9, 12, 8, 13, 14, 15, 3, 7, 8, 11, 7, 12, 14,
#                 15, 8, 14, 13, 15, 11, 15, 15, 16)
#     #: A dictionary mapping triad code to triad name.
#     tricode_to_name = dict((i, motif_names[tricodes[i] - 1])
#                            for i in range(len(tricodes)))
#     precision_per_motif = {motif_name: [] for motif_name in motif_names}
#     recall_per_motif = {motif_name: [] for motif_name in motif_names}
#     # Find motifs in the real network
#     # motifs = generate_motifs()
#     motif_decomposition = find_motifs(G_real, motif_names)
#     for motif_name, tuples in motif_decomposition.items():
#         for tpl in tuples:
#             # links_real = list(G.edges())
#             # Get all potential links in the motif
#             links_all = np.asarray(list(itertools.permutations(tpl, 2)))
#             true_pos_motif_n = (
#                 true_pos_matrix[links_all[:, 0], links_all[:, 1]].sum())
#             true_neg_motif_n = (
#                 true_neg_matrix[links_all[:, 0], links_all[:, 1]].sum())
#             false_pos_motif_n = (
#                 false_pos_matrix[links_all[:, 0], links_all[:, 1]].sum())
#             false_neg_motif_n = (
#                 false_neg_matrix[links_all[:, 0], links_all[:, 1]].sum())
#             if true_pos_motif_n + false_pos_motif_n > 0:
#                 precision_per_motif[motif_name].append(
#                     true_pos_motif_n / (true_pos_motif_n + false_pos_motif_n))
#             if true_pos_motif_n + false_neg_motif_n > 0:
#                 recall_per_motif[motif_name].append(
#                     true_pos_motif_n / (true_pos_motif_n + false_neg_motif_n))
#         # Average over occurences of the same motif
#         precision_per_motif[motif_name] = np.nanmean(
#             precision_per_motif[motif_name])
#         recall_per_motif[motif_name] = np.nanmean(
#             recall_per_motif[motif_name])
#     df.loc[run_name, 'precision_per_motif'] = [precision_per_motif]
#     df.loc[run_name, 'recall_per_motif'] = [recall_per_motif]

    # -------------------------------------------------------------------------
    # endregion
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # region Delay error
    # -------------------------------------------------------------------------

#     # Condense delay matrices to a 2D matrix by only taking the mean delay
#     temp = delay_matrices_real.copy()
#     for i in range(delay_max_real):
#         temp[i, :, :] *= (i + 1)
#     temp[temp == 0.] = np.NaN
#     delay_matrix_real = np.nanmean(temp, axis=0)
#     # Remove self delay from real delay matrix (because at the moment
#     # IDTxl always infers them, so they are not included in the results)
#     np.fill_diagonal(delay_matrix_real, np.NaN)
#     delay_matrix_real[np.isnan(delay_matrix_real)] = 0.
#     # Condense delay matrices to a 2D matrix by only taking the mean delay
#     temp = delay_matrices_inferred.copy()
#     for i in range(delay_max_inferred):
#         temp[i, :, :] *= (i + 1)
#     temp[temp == 0.] = np.NaN
#     delay_matrix_inferred = np.nanmean(temp, axis=0)
#     # Remove self delay from inferred delay matrix (because at the moment
#     # IDTxl always infers them, so they are not included in the results)
#     np.fill_diagonal(delay_matrix_inferred, np.NaN)
#     delay_matrix_inferred[np.isnan(delay_matrix_inferred)] = 0.
#     # Compute mean absolute delay error over all inferred links (if it's
#     # a false positive, the inferred lag is counted as error).
#     if (true_pos_n + false_neg_n) > 0:
#         abs_diff = np.abs(delay_matrix_real - delay_matrix_inferred)
#         delay_error_mean = np.mean(abs_diff[delay_matrix_inferred > 0])
#     else:
#         print('WARNING: The real network is empty: cannot compute'
#               'average delay error')
#         delay_error_mean = np.NaN
#     # Write to DataFrame
#     df.loc[run_name, 'delay_error_mean'] = delay_error_mean
#     # Computed normalised error with respect to the expected absolute
#     # error at random
#     abs_diff_random = np.mean([
#         np.abs(l1 - l2)
#         for l1 in range(1, delay_max_real + 1)
#         for l2 in range(1, delay_max_inferred + 1)])
#     # Write to DataFrame
#     df.loc[run_name, 'delay_error_mean_normalised'] = (
#         delay_error_mean / abs_diff_random)

    # -------------------------------------------------------------------------
    # endregion
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # region TE
    # -------------------------------------------------------------------------

    # Initialise vectors
    TE_omnibus_empirical = np.zeros(nodes_n)
    TE_complete_empirical = np.full(
        shape=(delay_max_inferred, nodes_n, nodes_n),
        fill_value=np.NaN,
        dtype=float)
    # TE_complete_apparent = np.full(
    #     shape=(delay_max_inferred, nodes_n, nodes_n),
    #     fill_value=np.NaN,
    #     dtype=float)
    if 'TE' in traj.par.network_inference.algorithm:
        TE_complete_empirical = res.get_time_series_graph(
            weights='TE_complete',
            fdr=fdr)
        # TE_apparent_empirical = res.get_time_series_graph(
        #     weights='TE_apparent',
        #     fdr=fdr)
    TE_omnibus_theoretical_inferred_vars = np.zeros(nodes_n)
    TE_complete_theoretical_inferred_vars = np.zeros(
        (delay_max_inferred, nodes_n, nodes_n))
    TE_apparent_theoretical_inferred_vars = np.zeros(
        (delay_max_inferred, nodes_n, nodes_n))
    AIS_theoretical_inferred_vars = np.zeros(nodes_n)
    TE_omnibus_theoretical_causal_vars = np.zeros(nodes_n)
    TE_complete_theoretical_causal_vars = np.zeros(
        (delay_max_real, nodes_n, nodes_n))
    TE_apparent_theoretical_causal_vars = np.zeros(
        (delay_max_real, nodes_n, nodes_n))
    AIS_theoretical_causal_vars = np.zeros(nodes_n)

    # Compute theoretical info-theoretic measures if VAR process
    if traj.par.node_dynamics.model == 'AR_gaussian_discrete':
        # Recover coefficient matrices
        coefficient_matrices_real = np.transpose(
            delay_matrices_real * coupling_matrix_real,
            (0, 2, 1))
        # Build VAR reduced form
        # See Appendix Faes et al. (PRE, 2015, doi: 10.1103/PhysRevE.91.032904)
        if traj.f_contains('max_lag_target', shortcuts=True):
            max_lag_target = traj.network_inference.max_lag_target
        else:
            max_lag_target = 1
        lags = 1 + max(
            traj.network_inference.max_lag_sources,
            max_lag_target,
            delay_max_real)
        VAR_reduced_form = np.zeros((nodes_n * lags, nodes_n * lags))
        VAR_reduced_form[0:nodes_n, 0:nodes_n*delay_max_real] = np.reshape(
            np.transpose(coefficient_matrices_real, (1, 0, 2)),
            [nodes_n, nodes_n * delay_max_real])
        VAR_reduced_form[nodes_n:, 0:-nodes_n] = np.eye(nodes_n * (lags - 1))
        # Recover process noise covariance matrix
        if traj.f_contains('noise_std', shortcuts=True):
            variance = traj.parameters.node_dynamics.noise_std ** 2
        else:  # old version
            variance = traj.parameters.node_dynamics.noise_amplitude ** 2
        process_noise_cov = variance * np.eye(nodes_n, nodes_n)
        # Pad noise covariance matrix with zeros along both dimensions
        # (to the right and at the bottom)
        VAR_reduced_noise_cov = block_diag(
            process_noise_cov,
            np.zeros((nodes_n * (lags - 1), nodes_n * (lags - 1))))
        # Compute lagged cov matrix by solving discrete Lyapunov equation
        # cov = VAR_reduced_form * cov * VAR_reduced_form.T + noise_cov
        # (See scipy documentation for 'solve_discrete_lyapunov' function)
        VAR_reduced_cov = solve_discrete_lyapunov(
            VAR_reduced_form,
            VAR_reduced_noise_cov,
            method='bilinear')  # use 'bilinear' if 'direct' fails or too slow
        # Check solution
        abs_diff = np.max(np.abs((
            VAR_reduced_cov
            - VAR_reduced_form.dot(VAR_reduced_cov).dot(VAR_reduced_form.T)
            - VAR_reduced_noise_cov)))
        assert abs_diff < 10**-6, "large absolute error = {}".format(abs_diff)

    def partial_variance(variance, cross_cov, cov):
        inverse_cov = np.linalg.inv(cov)
        return variance - np.dot(np.dot(cross_cov, inverse_cov), cross_cov.T)

    def compute_MI(X_IDs, Y_IDs):
        # Compute theoretical MI from covariance matrix of VAR (reduced form)
        if len(X_IDs) > 0 and len(Y_IDs) > 0:
            # Use np._ix to extract submatrix from row and column indices
            Y_cov = VAR_reduced_cov[np.ix_(Y_IDs, Y_IDs)]
            numerator = Y_cov
            Y_X_crosscov = VAR_reduced_cov[np.ix_(Y_IDs, X_IDs)]
            X_cov = VAR_reduced_cov[np.ix_(X_IDs, X_IDs)]
            denominator = partial_variance(Y_cov, Y_X_crosscov, X_cov)
            MI = 0.5 * np.log(numerator / denominator)
        else:
            print('Empty source or target set. Will return MI=NaN')
            MI = np.NaN
        return MI

    def compute_CMI(X_IDs, Y_IDs, Z_IDs):
        # Compute theoretical CMI from covariance matrix of VAR (reduced form)
        if len(X_IDs) > 0 and len(Y_IDs) > 0:
            # Use np._ix to extract submatrix from row and column indices
            Y_cov = VAR_reduced_cov[np.ix_(Y_IDs, Y_IDs)]
            Y_Z_crosscov = VAR_reduced_cov[np.ix_(Y_IDs, Z_IDs)]
            Z_cov = VAR_reduced_cov[np.ix_(Z_IDs, Z_IDs)]
            numerator = partial_variance(Y_cov, Y_Z_crosscov, Z_cov)
            ZX_IDs = Z_IDs + X_IDs
            Y_ZX_crosscov = VAR_reduced_cov[np.ix_(Y_IDs, ZX_IDs)]
            ZX_cov = VAR_reduced_cov[np.ix_(ZX_IDs, ZX_IDs)]
            denominator = partial_variance(Y_cov, Y_ZX_crosscov, ZX_cov)
            CMI = 0.5 * np.log(numerator / denominator)
        else:
            print('Empty source or target set. Will return CMI=NaN')
            CMI = np.NaN
        return CMI

    def compute_TE_omnibus(source_IDs, target_IDs, targetPast_IDs):
        return compute_CMI(source_IDs, target_IDs, targetPast_IDs)

    def compute_TE_apparent(source_IDs, target_IDs, targetPast_IDs):
        # Compute theoretical apparent transfer entropy from covariance
        # matrix of the VAR (reduced form)
        TE_apparent = np.full(shape=(len(source_IDs)), fill_value=np.NaN)
        for i, s in enumerate(source_IDs):
            TE_apparent[i] = compute_CMI(
                [s],
                target_IDs,
                targetPast_IDs)
        return TE_apparent

    def compute_TE_complete(source_IDs, target_IDs, targetPast_IDs):
        # Compute theoretical complete transfer entropy from covariance
        # matrix of the VAR (reduced form)
        TE_complete = np.full(shape=(len(source_IDs)), fill_value=np.NaN)
        for i, s in enumerate(source_IDs):
            # Define conditioning set and exclude source s
            cond_IDs = targetPast_IDs + source_IDs
            cond_IDs.remove(s)
            TE_complete[i] = compute_CMI(
                [s],
                target_IDs,
                cond_IDs)
        return TE_complete

    for t in range(nodes_n):
        if debug_mode:
            print('\nTarget = {0}'.format(t))
        # Read and store empirical TE
        if 'TE' in traj.par.network_inference.algorithm:
            if debug_mode:
                print('Reading and storing empirical TE...')
            if release_results_class:
                TE_omnibus_empirical[t] = (
                    res.get_single_target(t, fdr=fdr)['omnibus_te'])
            else:
                TE_omnibus_empirical[t] = r[t]['omnibus_te']
            if debug_mode:
                print('TE_omnibus_empirical = {0}'.format(
                    TE_omnibus_empirical[t]))
                print('TE_complete_empirical = {0}'.format(
                    TE_complete_empirical[:, :, t]))

        # Compute theoretical TE and AIS if VAR process
        if traj.par.node_dynamics.model == 'AR_gaussian_discrete':
            if debug_mode:
                print('Computing theoretical TE and AIS for VAR process...')
            target_IDs = [t]

            # Compute theoretical TE on inferred variables
            # if release_results_class:
            #     targetPast_IDs = [
            #         s[1] * nodes_n + s[0]
            #         for s in res.get_single_target(t, fdr=fdr)[
            #             'selected_vars_target']]
            #     source_IDs = [
            #         s[1] * nodes_n + s[0]
            #         for s in res.get_single_target(t, fdr=fdr)[
            #             'selected_vars_sources']]
            # else:
            #     targetPast_IDs = [
            #         s[1] * nodes_n + s[0]
            #         for s in r[t]['selected_vars_target']]
            #     source_IDs = [
            #         s[1] * nodes_n + s[0] for s in r[t]['selected_vars_sources']]
            # targetPast_IDs.sort()
            # source_IDs.sort()

            # get all inferred vars as (variable, delay) pairs
            vars_inferred = np.rot90(
                np.array(delay_matrices_inferred[:, :, t].nonzero()),
                axes=(1, 0))  # rotate clockwise
            # Split into source and target variables
            target_self = vars_inferred[:, 0] == t
            others = np.invert(target_self)
            targetPast_inferred = vars_inferred[target_self]
            source_inferred = vars_inferred[others]
            source_IDs_inferred = (
                (source_inferred[:, 1] + 1) * nodes_n
                + source_inferred[:, 0]).tolist()
            targetPast_IDs_inferred = (
                (targetPast_inferred[:, 1] + 1) * nodes_n
                + targetPast_inferred[:, 0]).tolist()
            if debug_mode:
                print('Inferred variables:')
                print('source_IDs_inferred = {0}'.format(source_IDs_inferred))
                print('targetPast_IDs_inferred = {0}'.format(
                    targetPast_IDs_inferred))
            # Compute omnibus TE
            TE_omnibus_theoretical_inferred_vars[t] = compute_TE_omnibus(
                source_IDs_inferred,
                target_IDs,
                targetPast_IDs_inferred)
            # Compute complete TE
            TE_complete_theoretical_inferred_vars[
                source_inferred[:, 1], source_inferred[:, 0], t] = compute_TE_complete(
                source_IDs_inferred,
                target_IDs,
                targetPast_IDs_inferred)
            # Compute apparent TE
            TE_apparent_theoretical_inferred_vars[
                source_inferred[:, 1], source_inferred[:, 0], t] = compute_TE_apparent(
                source_IDs_inferred,
                target_IDs,
                targetPast_IDs_inferred)
            # Compute AIS
            if len(targetPast_IDs_inferred) > 0:
                AIS_theoretical_inferred_vars[t] = compute_CMI(
                    targetPast_IDs_inferred,
                    target_IDs,
                    [])
            else:
                AIS_theoretical_inferred_vars[t] = np.NaN

            # Compute theoretical TE on causal variables
            # get all causal vars as (variable, delay) pairs
            vars_causal = np.rot90(
                np.array(delay_matrices_real[:, :, t].nonzero()),
                axes=(1, 0))  # rotate clockwise
            # Split into source and target variables
            target_self = vars_causal[:, 0] == t
            others = np.invert(target_self)
            targetPast_causal = vars_causal[target_self]
            source_causal = vars_causal[others]
            source_IDs_causal = (
                (source_causal[:, 1] + 1) * nodes_n
                + source_causal[:, 0]).tolist()
            targetPast_IDs_causal = (
                (targetPast_causal[:, 1] + 1) * nodes_n
                + targetPast_causal[:, 0]).tolist()
            if debug_mode:
                print('Causal variables:')
                print('source_IDs_causal = {0}'.format(source_IDs_causal))
                print('targetPast_IDs_causal = {0}'.format(
                    targetPast_IDs_causal))
            # Compute omnibus TE
            TE_omnibus_theoretical_causal_vars[t] = compute_TE_omnibus(
                source_IDs_causal,
                target_IDs,
                targetPast_IDs_causal)
            # Compute complete TE
            TE_complete_theoretical_causal_vars[
                source_causal[:, 1], source_causal[:, 0], t] = compute_TE_complete(
                source_IDs_causal,
                target_IDs,
                targetPast_IDs_causal)
            # Compute apparent TE
            TE_apparent_theoretical_causal_vars[
                source_causal[:, 1], source_causal[:, 0], t] = compute_TE_apparent(
                source_IDs_causal,
                target_IDs,
                targetPast_IDs_causal)
            # Compute AIS
            if len(targetPast_IDs_causal) > 0:
                AIS_theoretical_causal_vars[t] = compute_CMI(
                    target_IDs,
                    targetPast_IDs_causal,
                    [])
            else:
                AIS_theoretical_causal_vars[t] = np.NaN

            if debug_mode:
                print('Using inferred variables:')
                print('TE_omnibus_theoretical_inferred_vars = {0}'.format(
                    TE_omnibus_theoretical_inferred_vars[t]))
                print('TE_complete_theoretical_inferred_vars = {0}'.format(
                    TE_complete_theoretical_inferred_vars[
                        source_inferred[:, 1], source_inferred[:, 0], t]))
                print('TE_apparent_theoretical_inferred_vars = {0}'.format(
                    TE_apparent_theoretical_inferred_vars[
                        source_inferred[:, 1], source_inferred[:, 0], t]))
                print('AIS_theoretical_inferred_vars = {0}'.format(
                    AIS_theoretical_inferred_vars[t]))
                print('Using causal variables:')
                print('TE_omnibus_theoretical_causal_vars = {0}'.format(
                    TE_omnibus_theoretical_causal_vars[t]))
                print('TE_complete_theoretical_causal_vars = {0}'.format(
                    TE_complete_theoretical_causal_vars[
                        source_causal[:, 1], source_causal[:, 0], t]))
                print('TE_apparent_theoretical_causal_vars = {0}'.format(
                    TE_apparent_theoretical_causal_vars[
                        source_causal[:, 1], source_causal[:, 0], t]))
                print('AIS_theoretical_causal_vars = {0}'.format(
                    AIS_theoretical_causal_vars[t]))
        else:
            TE_omnibus_theoretical_inferred_vars = np.NaN
            TE_complete_theoretical_inferred_vars = np.NaN
            TE_apparent_theoretical_inferred_vars = np.NaN
            AIS_theoretical_inferred_vars = np.NaN
            TE_omnibus_theoretical_causal_vars = np.NaN
            TE_complete_theoretical_causal_vars = np.NaN
            TE_apparent_theoretical_causal_vars = np.NaN
            AIS_theoretical_causal_vars = np.NaN
    # Add results to DataFrame
    df.loc[run_name, 'TE_omnibus_empirical'] = TE_omnibus_empirical
    df.loc[run_name, 'TE_complete_empirical'] = TE_complete_empirical
    # df.loc[run_name, 'TE_apparent_empirical'] = TE_apparent_empirical
    df.loc[
        run_name,
        'TE_omnibus_theoretical_inferred_vars'
        ] = TE_omnibus_theoretical_inferred_vars
    df.loc[
        run_name,
        'TE_complete_theoretical_inferred_vars'
        ] = TE_complete_theoretical_inferred_vars
    df.loc[
        run_name,
        'TE_apparent_theoretical_inferred_vars'
        ] = TE_apparent_theoretical_inferred_vars
    df.loc[
        run_name,
        'AIS_theoretical_inferred_vars'
        ] = AIS_theoretical_inferred_vars

    df.loc[
        run_name,
        'TE_omnibus_theoretical_causal_vars'
        ] = TE_omnibus_theoretical_causal_vars
    df.loc[
        run_name,
        'TE_complete_theoretical_causal_vars'
        ] = TE_complete_theoretical_causal_vars
    df.loc[
        run_name,
        'TE_apparent_theoretical_causal_vars'
        ] = TE_apparent_theoretical_causal_vars
    df.loc[
        run_name,
        'AIS_theoretical_causal_vars'
        ] = AIS_theoretical_causal_vars

    # -------------------------------------------------------------------------
    # endregion
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # region Spectral radius
    # -------------------------------------------------------------------------
#     radius = np.NaN
#     if traj.par.node_dynamics.model == 'AR_gaussian_discrete':
#         # Use VAR reduced form defined above to compute the radius
#         radius = max(np.abs(np.linalg.eigvals(VAR_reduced_form)))
#     df.loc[run_name, 'spectral_radius'] = radius

    # -------------------------------------------------------------------------
    # endregion
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # region Check if the system got stuck on attractors
    # -------------------------------------------------------------------------
    # time_series = traj.results[run_name].node_dynamics.time_series
    # if traj.par.node_dynamics.model == 'boolean_random':
    #     initial_states = time_series[:, 0, :]
    #     # Reshape for broadcasting (subtracting from matrix)
    #     initial_states = initial_states[:, np.newaxis]
    #     # Compute Hamming distance over time (wrt initial state)
    #     dist = np.count_nonzero(time_series - initial_states, axis=0)
    #     # Normalise: divide by number of processes
    #     dist = dist / time_series.shape[0]

    # -------------------------------------------------------------------------
    # endregion
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # region Network measures
    # -------------------------------------------------------------------------

    def efficiency(G, u, v):
        """Returns the efficiency of a pair of nodes in a graph.
        The *efficiency* of a pair of nodes is the multiplicative inverse of the
        shortest path distance between the nodes [1]_. Returns 0 if no path
        between nodes.
        Parameters
        ----------
        G : :class:`networkx.Graph`
            A graph for which to compute the average local efficiency.
        u, v : node
            Nodes in the graph ``G``.
        Returns
        -------
        float
            Multiplicative inverse of the shortest path distance between the nodes.
        Notes
        -----
        Edge weights are ignored when computing the shortest path distances.

        References
        ----------
        .. [1] Latora, Vito, and Massimo Marchiori.
            "Efficient behavior of small-world networks."
            *Physical Review Letters* 87.19 (2001): 198701.
            <https://doi.org/10.1103/PhysRevLett.87.198701>
        """
        try:
            eff = 1 / nx.shortest_path_length(G, u, v)
        except nx.NetworkXNoPath:
            eff = 0
        return eff

    def average_global_efficiency(G):
        """Returns the average global efficiency of the graph.
        The *efficiency* of a pair of nodes in a graph is the multiplicative
        inverse of the shortest path distance between the nodes. The *average
        global efficiency* of a graph is the average efficiency of all pairs of
        nodes [1]_.
        Parameters
        ----------
        G : :class:`networkx.Graph`
            A graph for which to compute the average global efficiency.
        Returns
        -------
        float
            The average global efficiency of the graph.
        Notes
        -----
        Edge weights are ignored when computing the shortest path distances.
        See also
        --------
        local_efficiency
        References
        ----------
        .. [1] Latora, Vito, and Massimo Marchiori.
            "Efficient behavior of small-world networks."
            *Physical Review Letters* 87.19 (2001): 198701.
            <https://doi.org/10.1103/PhysRevLett.87.198701>
        """
        n = len(G)
        denom = n * (n - 1)
        if denom != 0:
            g_eff = sum(efficiency(G, u, v) for u, v in itertools.permutations(G, 2)) / denom
        else:
            g_eff = 0
        return g_eff

    def local_efficiency(G):
        """Returns the local efficiency of each node in the graph.
        The *efficiency* of a pair of nodes in a graph is the multiplicative
        inverse of the shortest path distance between the nodes. The *local
        efficiency* of a node in the graph is the average global efficiency of
        the subgraph induced by the neighbors of the node[1]_.
        Parameters
        ----------
        G : :class:`networkx.Graph`
            A graph for which to compute the average local efficiency.
        Returns
        -------
        np.array
            The local efficiency of each node the graph.
        Notes
        -----
        Edge weights are ignored when computing the shortest path distances.

        References
        ----------
        .. [1] Latora, Vito, and Massimo Marchiori.
            "Efficient behavior of small-world networks."
            *Physical Review Letters* 87.19 (2001): 198701.
            <https://doi.org/10.1103/PhysRevLett.87.198701>
        """
        # TODO This summation can be trivially parallelized.
        eff_list = [average_global_efficiency(G.subgraph(G[v])) for v in G]
        return np.array(eff_list)

    def rich_club_in_degrees(G):
        """Returns the rich-club coefficient for each in-degree in the graph
        `G`.
        Returns a dictionary mapping degree to rich-club coefficient for
        that degree.
        """
        # deghist = nx.degree_histogram(G)
        in_degrees = G.in_degree()
        counts = Counter(d for n, d in in_degrees)
        deghist = [counts.get(i, 0) for i in range(max(counts) + 1)]       
        total = sum(deghist)
        # Compute the number of nodes with degree greater than `k`, for each
        # degree `k` (omitting the last entry, which is zero).
        nks = (total - cs for cs in nx.utils.accumulate(deghist) if total - cs > 1)
        # Create a sorted list of pairs of edge endpoint degrees.
        #
        # The list is sorted in reverse order so that we can pop from the
        # right side of the list later, instead of popping from the left
        # side of the list, which would have a linear time cost.
        edge_degrees = sorted(
            (sorted(map(in_degrees, e)) for e in G.in_edges()), reverse=True)
        ek = G.number_of_edges()
        k1, k2 = edge_degrees.pop()
        rc = {}
        for d, nk in enumerate(nks):
            while k1 <= d:
                if len(edge_degrees) == 0:
                    ek = 0
                    break
                k1, k2 = edge_degrees.pop()
                ek -= 1
            rc[d] = ek / (nk * (nk - 1))
        return rc

    def rich_club_out_degrees(G):
        """Returns the rich-club coefficient for each out-degree in the graph
        `G`.
        Returns a dictionary mapping degree to rich-club coefficient for
        that degree.
        """
        # deghist = nx.degree_histogram(G)
        out_degrees = G.out_degree()
        counts = Counter(d for n, d in out_degrees)
        deghist = [counts.get(i, 0) for i in range(max(counts) + 1)]       
        total = sum(deghist)
        # Compute the number of nodes with degree greater than `k`, for each
        # degree `k` (omitting the last entry, which is zero).
        nks = (total - cs for cs in nx.utils.accumulate(deghist) if total - cs > 1)
        # Create a sorted list of pairs of edge endpoint degrees.
        #
        # The list is sorted in reverse order so that we can pop from the
        # right side of the list later, instead of popping from the left
        # side of the list, which would have a linear time cost.
        edge_degrees = sorted(
            (sorted(map(out_degrees, e)) for e in G.out_edges()), reverse=True)
        ek = G.number_of_edges()
        k1, k2 = edge_degrees.pop()
        rc = {}
        for d, nk in enumerate(nks):
            while k1 <= d:
                if len(edge_degrees) == 0:
                    ek = 0
                    break
                k1, k2 = edge_degrees.pop()
                ek -= 1
            rc[d] = ek / (nk * (nk - 1))
        return rc

    # In-degree and out-degree
    in_degree_real = G_real.in_degree
    in_degree_inferred = G_inferred.in_degree
    out_degree_real = G_real.out_degree
    out_degree_inferred = G_inferred.out_degree
    # Clustering coefficient
    clustering_real = nx.clustering(
        G_real,
        weight=None)
    clustering_inferred = nx.clustering(
        G_inferred,
        weight=None)
    # Average shortest path length
    if nx.is_strongly_connected(G_real):
        average_shortest_path_length_real = (
            nx.average_shortest_path_length(
                G_real,
                weight=None))
    else:
        average_shortest_path_length_real = np.NaN
        print('The real graph is not connected')
    if nx.is_strongly_connected(G_inferred):
        average_shortest_path_length_inferred = (
            nx.average_shortest_path_length(
                G_inferred,
                weight=None))
    else:
        average_shortest_path_length_inferred = np.NaN
        print('The inferred graph is not connected')
    # Average local and global efficiency
    average_global_efficiency_real = average_global_efficiency(G_real)
    average_global_efficiency_inferred = average_global_efficiency(G_inferred)
    local_efficiency_real = local_efficiency(G_real)
    local_efficiency_inferred = local_efficiency(G_inferred)
    # Rich-club coefficient (normalised)
    #rich_club_in_degrees_real = nx.rich_club_coefficient(G_real.to_undirected(), normalized=True, Q=100)
    rich_club_in_degrees_real = rich_club_in_degrees(G_real)
        #G_real.to_undirected(),
        #normalized=False,
        #Q=100,
        #seed=None)
    rich_club_in_degrees_inferred = rich_club_in_degrees(G_inferred)
        #G_inferred.to_undirected(),
        #normalized=False,
        #Q=100,
        #seed=None)
    rich_club_out_degrees_real = rich_club_out_degrees(G_real)
    rich_club_out_degrees_inferred = rich_club_out_degrees(G_inferred)
    # Degree assortativity coefficient
    in_degree_assortativity_real = nx.degree_assortativity_coefficient(
        G_real, x='in', y='in')
    in_degree_assortativity_inferred = nx.degree_assortativity_coefficient(
        G_inferred, x='in', y='in')
    out_degree_assortativity_real = nx.degree_assortativity_coefficient(
        G_real, x='out', y='out')
    out_degree_assortativity_inferred = nx.degree_assortativity_coefficient(
        G_inferred, x='out', y='out')
    # Reciprocity
    if not nx.is_empty(G_real):
        reciprocity_real = nx.reciprocity(G_real)
        overall_reciprocity_real = nx.overall_reciprocity(G_real)
    if not nx.is_empty(G_inferred):
        reciprocity_inferred = nx.reciprocity(G_inferred)
        overall_reciprocity_inferred = nx.overall_reciprocity(G_inferred)

    # https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.communicability_alg.communicability.html#networkx.algorithms.communicability_alg.communicability
    df.loc[run_name, 'in_degree_real'] = np.array(in_degree_real)[:, 1]
    df.loc[run_name, 'in_degree_inferred'] = np.array(in_degree_inferred)[:, 1]
    df.loc[run_name, 'out_degree_real'] = np.array(out_degree_real)[:, 1]
    df.loc[run_name, 'out_degree_inferred'] = np.array(
        out_degree_inferred)[:, 1]
    df.loc[run_name, 'clustering_real'] = np.array(list(
        clustering_real.values()))
    df.loc[run_name, 'clustering_inferred'] = np.array(list(
        clustering_inferred.values()))
    df.loc[run_name, 'average_shortest_path_length_real'] = (
        average_shortest_path_length_real)
    df.loc[run_name, 'average_shortest_path_length_inferred'] = (
        average_shortest_path_length_inferred)
    df.loc[run_name, 'average_global_efficiency_real'] = (
        average_global_efficiency_real)
    df.loc[run_name, 'average_global_efficiency_inferred'] = (
        average_global_efficiency_inferred)
    df.loc[run_name, 'local_efficiency_real'] = local_efficiency_real
    df.loc[run_name, 'local_efficiency_inferred'] = local_efficiency_inferred
    df.loc[run_name, 'rich_club_in_degrees_real'] = np.array(list(
        rich_club_in_degrees_real.values()))
    df.loc[run_name, 'rich_club_in_degrees_inferred'] = np.array(list(
        rich_club_in_degrees_inferred.values()))
    df.loc[run_name, 'rich_club_out_degrees_real'] = np.array(list(
        rich_club_out_degrees_real.values()))
    df.loc[run_name, 'rich_club_out_degrees_inferred'] = np.array(list(
        rich_club_out_degrees_inferred.values()))
    df.loc[run_name, 'in_degree_assortativity_real'] = (
        in_degree_assortativity_real)
    df.loc[run_name, 'in_degree_assortativity_inferred'] = (
        in_degree_assortativity_inferred)
    df.loc[run_name, 'out_degree_assortativity_real'] = (
        out_degree_assortativity_real)
    df.loc[run_name, 'out_degree_assortativity_inferred'] = (
        out_degree_assortativity_inferred)
    df.loc[run_name, 'reciprocity_real'] = reciprocity_real
    df.loc[run_name, 'reciprocity_inferred'] = reciprocity_inferred
    df.loc[run_name, 'overall_reciprocity_real'] = overall_reciprocity_real
    df.loc[run_name, 'overall_reciprocity_inferred'] = (
        overall_reciprocity_inferred)
    # -------------------------------------------------------------------------
    # endregion
    # -------------------------------------------------------------------------

# Reset trajectory to the default settings, to release its belief to
# be the last run:
traj.f_restore_default()

if save_results:
    # Save DataFrame
    if fdr:
        df.to_pickle(os.path.join(traj_dir, 'postprocessing_fdr.pkl'))
    else:
        df.to_pickle(os.path.join(traj_dir, 'postprocessing.pkl'))
else:
    print('\nPostprocessing DataFrame NOT saved (debug mode)')
