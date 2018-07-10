from pypet import Trajectory
import os
import numpy as np
import pandas as pd
from idtxl.visualise_graph import _get_adj_matrix
from idtxl.stats import network_fdr
# from mylib_pypet import print_leaves

fdr = False

# Load the trajectory from the hdf5 file
# Only load parameters, results will be loaded at runtime (auto loading)
traj_dir = os.path.join('trajectories', '2018_04_19_15h06m42s')
traj_filename = 'traj.hdf5'
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

# Count number of runs
runs_n = len(traj.f_get_run_names())

# Get list of explored parameters
parameters_explored = [str.split(par, '.').pop() for par in (
    traj.f_get_explored_parameters())]

# Initialise analysis summary table
df = pd.DataFrame(
    index=traj.f_get_run_names(),
    columns=parameters_explored + [
        'precision',
        'recall',
        'specificity',
        'false_pos_rate',
        'incorrect_target_rate',
        'delay_error_mean',
        'time_elapsed_monotonic',
        'time_elapsed_perf_counter',
        'time_elapsed_process_time'
        ],
    dtype=float
)

# Loop over runs
for run_name in traj.f_get_run_names():

    # Make trajectory behave like a particular single run:
    # all explored parameter’s values will be set to the corresponding
    # values of one particular run.
    traj.f_set_crun(run_name)

    print('post-processing of {0} in progress...'.format(run_name))

    for par in parameters_explored:
        df.loc[run_name, par] = traj.parameters[par]

    # Read elapsed time
    # df.loc[run_name, 'time_elapsed_monotonic'] = (
    #   traj.results[run_name].timing.elapsed['monotonic'])
    # df.loc[run_name, 'time_elapsed_perf_counter'] = (
    #   traj.results[run_name].timing.elapsed['perf_counter'])
    # df.loc[run_name, 'time_elapsed_process_time'] = (
    #   traj.results[run_name].timing.elapsed['process_time'])

    nodes_n = traj.parameters.topology.initial['nodes_n']

    # Load original delay matrix
    delay_matrix_original = traj.results[run_name].delay.initial.delay_matrix
    # TODO: Deal with multiple delays, at the moment I only consider the
    # shortest delay <----------------------------
    if type(delay_matrix_original.iloc[0, 0]) == np.ndarray:
        for x in range(nodes_n):
            for y in range(nodes_n):
                if len(delay_matrix_original.iloc[x, y]) == 0:
                    delay_matrix_original.iloc[x, y] = 0
                else:
                    delay_matrix_original.iloc[x, y] = min(
                        delay_matrix_original.iloc[x, y])

    # Remove self delay from original delay matrix (because at the moment IDTxl
    # always infers them, so they are not included in the results)
    np.fill_diagonal(delay_matrix_original.values, 0)

    # Load results dictionary
    res = traj.results[run_name].network_inference.network_inference_result.to_dict()

    #-----temporary
    test = np.ones(nodes_n)
    for ind in res.keys():
        test[ind] = res[ind]['omnibus_pval']
    if np.any(test > 0):
        print(min(test[test > 0]))
        print(max(test[test > 0]))
    #-----

    # Perform FDR (if specified)
    if fdr:
        for t in res.keys():
            res[t]['settings'] = res[t]['options']
        res['fdr_corrected'] = network_fdr({}, res)

    # Get inferred delay matrix
    delay_matrix_inferred = pd.DataFrame(
        _get_adj_matrix(
            res=res,
            n_nodes=nodes_n,
            fdr=fdr,
            find_u='max_te'
        )
    )

    # Get original adjacency matrix (without self-loops)
    adjacency_matrix_original = delay_matrix_original.copy()
    adjacency_matrix_original.values[adjacency_matrix_original.values > 0] = 1
    # Get inferred adjacency matrix (without self-loops)
    adjacency_matrix_inferred = delay_matrix_inferred.copy()
    adjacency_matrix_inferred.values[adjacency_matrix_inferred.values > 0] = 1

    # Compute sum and difference between original and inferred adjacency
    # matrices
    adjacency_matrices_sum = (
        adjacency_matrix_original
        + adjacency_matrix_inferred
    )
    # Prevent the removed self-loops to be counted as true negatives,
    # by setting the elements on the diagonal to NaN
    np.fill_diagonal(adjacency_matrices_sum.values, np.NaN)

    adjacency_matrices_diff = (
        adjacency_matrix_original
        - adjacency_matrix_inferred
    )

    # Find true positives
    true_pos_matrix = adjacency_matrices_sum.values == 2
    true_pos_n = true_pos_matrix.sum()
    # Find true negatives
    true_neg_matrix = adjacency_matrices_sum.values == 0
    true_neg_n = true_neg_matrix.sum()
    # Find false positives
    false_pos_matrix = adjacency_matrices_diff.values == -1
    false_pos_n = false_pos_matrix.sum()
    # Find false negatives
    false_neg_matrix = adjacency_matrices_diff.values == 1
    false_neg_n = false_neg_matrix.sum()

    # Sanity check
    assert (
        true_pos_n + true_neg_n + false_pos_n + false_neg_n
        == nodes_n * (nodes_n - 1)
    )

    # Compute classifier performance tests
    if (true_pos_n + false_pos_n) > 0:
        df.loc[run_name, 'precision'] = true_pos_n / (true_pos_n + false_pos_n)
    else:
        print('WARNING: Zero links were inferred: can not compute precision')
        df.loc[run_name, 'precision'] = np.NaN

    if (true_pos_n + false_neg_n) > 0:
        df.loc[run_name, 'recall'] = true_pos_n / (true_pos_n + false_neg_n)
    else:
        print('WARNING: The original network is empty: can not compute recall')
        df.loc[run_name, 'recall'] = np.NaN

    if (true_neg_n + false_pos_n) > 0:
        df.loc[run_name, 'specificity'] = true_neg_n / (
            true_neg_n + false_pos_n)
        df.loc[run_name, 'false_pos_rate'] = false_pos_n / (
            true_neg_n + false_pos_n)
    else:
        print('WARNING: The original network is full: can not compute'
              'specificity and false positive rate')
        df.loc[run_name, 'specificity'] = np.NaN
        df.loc[run_name, 'false_pos_rate'] = np.NaN

    #df.loc[run_name, 'incorrect_target_rate'] = (
    #    (adjacency_matrices_diff.values == 0).sum(0) < nodes_n).mean()
    df.loc[run_name, 'incorrect_target_rate'] = (
        false_pos_matrix.sum(0) > 0).mean()

    # Show original vs. inferred adjacency matrix and highlisght false
    # positives and false negatives
    # fig, ax = plt.subplots()
    # true_pos_colorvalue = 1
    # true_neg_colorvalue = 0
    # false_pos_colorvalue = 3
    # false_neg_colorvalue = 2
    # colors = ["light grey", "black", "orange", "red"]
    # cmap = ListedColormap(sns.xkcd_palette(colors))
    # sns.heatmap(true_pos * true_pos_colorvalue +
    #             true_neg * true_neg_colorvalue +
    #             false_pos * false_pos_colorvalue +
    #             false_neg * false_neg_colorvalue,
    #             vmin=np.min([
    #               true_pos_colorvalue,
    #               true_neg_colorvalue,
    #               false_pos_colorvalue,
    #               false_neg_colorvalue]),
    #             vmax=np.max([
    #               true_pos_colorvalue,
    #               true_neg_colorvalue,
    #               false_pos_colorvalue,
    #               false_neg_colorvalue]),
    #             cmap=cmap,
    #             cbar=True,
    #             ax=ax,
    #             square=True,
    #             linewidths=1,
    #             xticklabels=range(nodes_n),
    #             yticklabels=range(nodes_n)
    #             )
    # ax.xaxis.tick_top()
    # plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)
    # plt.show()

    # Compute mean relative delay error (mean over true positives,
    # i.e ignoring the delay error on the false positives)
    if (true_pos_n + false_neg_n) > 0:
        abs_diff = np.abs(delay_matrix_original - delay_matrix_inferred)
        df.loc[run_name, 'delay_error_mean'] = np.mean(
            abs_diff.values[delay_matrix_original.values > 0]
            / delay_matrix_original.values[delay_matrix_original.values > 0]
        )
    else:
        print('WARNING: The original network is empty: can not compute average'
              'delay error')
        df.loc[run_name, 'delay_error_mean'] = np.NaN

# Reset trajectory to the default settings, to release its belief to
# be the last run:
traj.f_restore_default()

# Save DataFrame
if fdr:
    df.to_pickle(os.path.join(traj_dir, 'postprocessing_fdr.pkl'))
else:
    df.to_pickle(os.path.join(traj_dir, 'postprocessing.pkl'))
