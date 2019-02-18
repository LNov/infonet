from pypet import Trajectory
import os
import numpy as np
from scipy.linalg import block_diag
from scipy.linalg import solve_discrete_lyapunov
import pandas as pd
from mylib_pypet import print_leaves

fdr = False
save_results = True

# Load the trajectory from the hdf5 file
# Only load parameters, results will be loaded at runtime (auto loading)
traj_dir = os.path.join('trajectories', 'KSG_on_CLM_3000samples')
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
# (it is important that the columns with the explored parameters
# preceed the ones with the results)
df = pd.DataFrame(
    index=traj.f_get_run_names(),
    columns=parameters_explored + [
        'precision',
        'recall',
        'specificity',
        'false_pos_rate',
        'incorrect_target_rate',
        'delay_error_mean',
        'TE_omnibus_empirical',
        'TE_omnibus_theoretical_inferred_vars',
        'TE_omnibus_theoretical_causal_vars',
        'spectral_radius',
        'time_elapsed_monotonic',
        'time_elapsed_perf_counter',
        'time_elapsed_process_time'
        ],
    dtype=object
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

    # Load results object
    res = traj.results[run_name].network_inference.network_inference_result
    # Check if old release
    if type(res) == pd.core.frame.DataFrame:
        release_results_class = False
    else:
        release_results_class = True

    if release_results_class:
        # Get inferred adjacency matrix
        adjacency_matrix_inferred = res.get_adjacency_matrix(
            weights='binary',
            fdr=fdr
        ).astype(float)
        # Get original adjacency matrix
        adjacency_matrix_original = np.asarray(
            traj.results[run_name].topology.initial.adjacency_matrix > 0,
            dtype=float
        ).astype(float)
        # Get original and inferred delay matrices
        delay_matrices_original = (
            traj.results[run_name].delay.initial.delay_matrices
            ).astype(float)
        delay_matrices_inferred = res.get_time_series_graph(
            weights='binary',
            fdr=fdr
            ).astype(float)
        # Get max delays
        delay_max_original = np.shape(delay_matrices_original)[0]
        delay_max_inferred = np.shape(delay_matrices_inferred)[0]

        # Load original coupling matrix
        coupling_matrix_original = (
            traj.results[run_name].node_coupling.initial.coupling_matrix
        )

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
        delay_max_inferred = 0
        for target in targets:
            all_vars_lags = [
                x[1] for x in
                r[target]['selected_vars_sources']
                + r[target]['selected_vars_target']
                ]
            if len(all_vars_lags) > 0:
                if max(all_vars_lags) > delay_max_inferred:
                    delay_max_inferred = max(all_vars_lags)
        # Fill in delay_matrices_inferred
        delay_matrices_inferred = np.zeros(
            (delay_max_inferred, nodes_n, nodes_n),
            dtype=float
            )
        for target in targets:
            all_vars = (
                r[target]['selected_vars_sources']
                + r[target]['selected_vars_target']
                )
            for (source, lag) in all_vars:
                delay_matrices_inferred[lag - 1, source, target] = 1.

        # Reconstruct inferred adjacency matrix
        adjacency_matrix_inferred = np.any(
            delay_matrices_inferred,
            axis=0).astype(float)

        # Load original delay matrix DataFrame
        delay_matrix_original_df = traj.results[
            run_name].delay.initial.delay_matrix
        # Convert pandas DataFrame to 3D numpy delay matrices
        if type(delay_matrix_original_df.iloc[0, 0]) == np.ndarray:
            # Find max delay
            delay_max_original = 0
            for x in range(nodes_n):
                for y in range(nodes_n):
                    if len(delay_matrix_original_df.iloc[x, y]) > 0:
                        d = np.max(delay_matrix_original_df.iloc[x, y])
                        if d > delay_max_original:
                            delay_max_original = d
            # Reconstruct original delay matrices
            delay_matrices_original = np.zeros(
                (delay_max_original, nodes_n, nodes_n),
                dtype=float
                )
            for x in range(nodes_n):
                for y in range(nodes_n):
                    if len(delay_matrix_original_df.iloc[x, y]) > 0:
                        for d in delay_matrix_original_df.iloc[x, y]:
                            delay_matrices_original[d - 1, x, y] = 1
        else:
            raise RuntimeError(
                'delay_matrix_original_df is not a numpy ndarray'
                )

        # Reconstruct original adjacency matrix
        adjacency_matrix_original = np.any(
            delay_matrices_original,
            axis=0
            ).astype(float)

        # Load original coupling matrix
        coupling_matrix_original = (
            traj.results[run_name].node_coupling.initial.coupling_matrix.values
        )

    # -------------------------------------------------------------------------
    # Boolean classification performance measures
    # -------------------------------------------------------------------------

    # Remove self-loops from original and inferred adjacency matrices
    np.fill_diagonal(adjacency_matrix_original, 0)
    np.fill_diagonal(adjacency_matrix_inferred, 0)
    # Compute sum and difference between original and inferred adjacency
    # matrices
    adjacency_matrices_sum = (
        adjacency_matrix_original
        + adjacency_matrix_inferred
    )
    adjacency_matrices_diff = (
        adjacency_matrix_original
        - adjacency_matrix_inferred
    )
    # Prevent the removed self-loops to be counted as true negatives,
    # by setting the elements on the diagonal to NaN
    np.fill_diagonal(adjacency_matrices_sum, np.NaN)
    # Find true positives
    true_pos_matrix = adjacency_matrices_sum == 2
    true_pos_n = true_pos_matrix.sum()
    # Find true negatives
    true_neg_matrix = adjacency_matrices_sum == 0
    true_neg_n = true_neg_matrix.sum()
    # Find false positives
    false_pos_matrix = adjacency_matrices_diff == -1
    false_pos_n = false_pos_matrix.sum()
    # Find false negatives
    false_neg_matrix = adjacency_matrices_diff == 1
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
        print('WARNING: Zero links were inferred: cannot compute precision')
        df.loc[run_name, 'precision'] = np.NaN

    if (true_pos_n + false_neg_n) > 0:
        df.loc[run_name, 'recall'] = true_pos_n / (true_pos_n + false_neg_n)
    else:
        print('WARNING: The original network is empty: cannot compute recall')
        df.loc[run_name, 'recall'] = np.NaN

    if (true_neg_n + false_pos_n) > 0:
        df.loc[run_name, 'specificity'] = true_neg_n / (
            true_neg_n + false_pos_n)
        df.loc[run_name, 'false_pos_rate'] = false_pos_n / (
            true_neg_n + false_pos_n)
    else:
        print('WARNING: The original network is full: cannot compute'
              'specificity and false positive rate')
        df.loc[run_name, 'specificity'] = np.NaN
        df.loc[run_name, 'false_pos_rate'] = np.NaN

    df.loc[run_name, 'incorrect_target_rate'] = (
        false_pos_matrix.sum(0) > 0).mean()

    # -------------------------------------------------------------------------
    # Delay error
    # -------------------------------------------------------------------------

    # Condense delay matrices to a 2D matrix by only taking the min delay
    temp = delay_matrices_original.copy()
    for i in range(delay_max_original):
        temp[i, :, :] *= (i + 1)
    temp[temp == 0.] = np.Infinity
    delay_matrix_original = np.min(temp, axis=0)
    delay_matrix_original[delay_matrix_original == np.Infinity] = 0.

    temp = delay_matrices_inferred.copy()
    for i in range(delay_max_inferred):
        temp[i, :, :] *= (i + 1)
    temp[temp == 0.] = np.Infinity
    delay_matrix_inferred = np.min(temp, axis=0)
    delay_matrix_inferred[delay_matrix_inferred == np.Infinity] = 0.

    # Remove self delay from original delay matrix (because at the moment
    # IDTxl always infers them, so they are not included in the results)
    np.fill_diagonal(delay_matrix_original, 0)
    # Compute mean relative delay error (mean over true positives,
    # i.e ignoring the delay error on the false positives)
    if (true_pos_n + false_neg_n) > 0:
        abs_diff = np.abs(delay_matrix_original - delay_matrix_inferred)
        delay_error_mean = np.mean(abs_diff[delay_matrix_original > 0])
    else:
        print('WARNING: The original network is empty: cannot compute'
              'average delay error')
        delay_error_mean = np.NaN
    df.loc[run_name, 'delay_error_mean'] = delay_error_mean

    # -------------------------------------------------------------------------
    # TE
    # -------------------------------------------------------------------------

    # Initialise vectors
    TE_omnibus_empirical = np.zeros(nodes_n, dtype=float)
    TE_omnibus_theoretical_inferred_vars = np.zeros(nodes_n, dtype=float)
    TE_omnibus_theoretical_causal_vars = np.zeros(nodes_n, dtype=float)

    # Recover coefficient matrices
    coefficient_matrices_original = np.transpose(
        delay_matrices_original * coupling_matrix_original,
        (0, 2, 1)
        )
    # Build VAR reduced form
    # See Appendix of Faes et al. (PRE, 2015, doi: 10.1103/PhysRevE.91.032904)
    if traj.f_contains('max_lag_target', shortcuts=True):
        max_lag_target = traj.network_inference.max_lag_target
    else:
        max_lag_target = 1
    lags = 1 + max(
        traj.network_inference.max_lag_sources,
        max_lag_target,
        delay_max_original
        )
    var_reduced_form = np.zeros((nodes_n * lags, nodes_n * lags))
    var_reduced_form[0:nodes_n, 0:nodes_n*delay_max_original] = np.reshape(
        np.transpose(coefficient_matrices_original, (1, 0, 2)),
        [nodes_n, nodes_n * delay_max_original]
        )
    var_reduced_form[nodes_n:, 0:-nodes_n] = np.eye(nodes_n * (lags - 1))
    # Recover process noise covariance matrix
    if traj.f_contains('noise_std', shortcuts=True):
        variance = traj.parameters.node_dynamics.noise_std ** 2
    else:  # old version
        variance = traj.parameters.node_dynamics.noise_amplitude ** 2
    process_noise_cov = variance * np.eye(nodes_n, nodes_n)
    # Pad noise covariance matrix with zeros along both dimensions
    # (to the right and at the bottom)
    var_reduced_noise_cov = block_diag(
        process_noise_cov,
        np.zeros((nodes_n * (lags - 1), nodes_n * (lags - 1)))
        )
    # Compute lagged covariance matrix by solving discrete Lyapunov equation
    # cov = var_reduced_form * cov * var_reduced_form.T + noise_cov
    # (See scipy documentation for 'solve_discrete_lyapunov' function)
    var_reduced_cov = solve_discrete_lyapunov(
        var_reduced_form,
        var_reduced_noise_cov,
        method='bilinear'  # use 'bilinear' if 'direct' fails or is too slow
        )
    # Check solution
    abs_diff = np.max(np.abs((
        var_reduced_cov
        - var_reduced_form.dot(var_reduced_cov).dot(var_reduced_form.T)
        - var_reduced_noise_cov
        )))
    assert abs_diff < 10**-6, "large absolute error = {}".format(abs_diff)

    def conditional_cov(s11, s12, s22):
        return s11 - np.dot(np.dot(s12, np.linalg.inv(s22)), np.transpose(s12))

    for t in range(nodes_n):
        target_IDs = [t]
        if release_results_class:
            source_IDs = [
                s[1] * nodes_n + s[0]
                for s in (
                    res.get_single_target(t, fdr=fdr)['selected_vars_sources'])
                ]
        else:
            source_IDs = [
                s[1] * nodes_n + s[0] for s in r[t]['selected_vars_sources']
                ]
        source_IDs.sort()
        # Read and store empirical TE
        if release_results_class:
            TE_omnibus_empirical[t] = (
                res.get_single_target(t, fdr=fdr)['omnibus_te'])
        else:
            TE_omnibus_empirical[t] = r[t]['omnibus_te']

        if traj.par.node_dynamics.model == 'AR_gaussian_discrete':
            # Compute omnibus TE (from all sources to target)
            if release_results_class:
                targetPast_IDs = [
                    s[1] * nodes_n + s[0]
                    for s in (
                        res.get_single_target(t, fdr=fdr)['selected_vars_target'])
                    ]
            else:
                targetPast_IDs = [
                    s[1] * nodes_n + s[0]
                    for s in r[t]['selected_vars_target']
                    ]

            target_cov = var_reduced_cov[np.ix_(
                target_IDs,
                target_IDs
                )]
            targetPastSourcesPast_IDs = targetPast_IDs + source_IDs
            target_targetPast_crosscov = var_reduced_cov[np.ix_(
                target_IDs, targetPast_IDs
                )]
            targetPast_cov = var_reduced_cov[np.ix_(
                targetPast_IDs,
                targetPast_IDs
                )]
            target_targetPastSourcesPast_crosscov = var_reduced_cov[np.ix_(
                target_IDs,
                targetPastSourcesPast_IDs
                )]
            targetPastSourcesPast_cov = var_reduced_cov[np.ix_(
                targetPastSourcesPast_IDs,
                targetPastSourcesPast_IDs
                )]
            numerator = conditional_cov(
                target_cov,
                target_targetPast_crosscov,
                targetPast_cov
                )
            denominator = conditional_cov(
                target_cov,
                target_targetPastSourcesPast_crosscov,
                targetPastSourcesPast_cov
                )
            TE_omnibus_theoretical_inferred_vars[t] = 0.5 * np.log(
                numerator / denominator
                )
            # Select causal source variables
            source_temp = np.array(
                delay_matrices_original[:, :, t].nonzero()
                ).transpose()
            # Exclude target variables
            target_temp = source_temp[source_temp[:, 1] == t].tolist()
            source_temp = source_temp[source_temp[:, 1] != t].tolist()
            # Compute theoretical omnibus TE
            source_IDs_causal = [
                (s[0]+1) * nodes_n + s[1] for s in source_temp
                ]
            targetPast_IDs_causal = [
                (s[0]+1) * nodes_n + s[1] for s in target_temp
                ]
            targetPastSourcesPast_IDs = (
                targetPast_IDs_causal
                + source_IDs_causal
                )
            target_targetPast_crosscov = var_reduced_cov[np.ix_(
                target_IDs,
                targetPast_IDs_causal
                )]
            targetPast_cov = var_reduced_cov[np.ix_(
                targetPast_IDs_causal,
                targetPast_IDs_causal
                )]
            target_targetPastSourcesPast_crosscov = var_reduced_cov[np.ix_(
                target_IDs,
                targetPastSourcesPast_IDs
                )]
            targetPastSourcesPast_cov = var_reduced_cov[np.ix_(
                targetPastSourcesPast_IDs,
                targetPastSourcesPast_IDs
                )]
            numerator = conditional_cov(
                target_cov,
                target_targetPast_crosscov,
                targetPast_cov
                )
            denominator = conditional_cov(
                target_cov,
                target_targetPastSourcesPast_crosscov,
                targetPastSourcesPast_cov
                )
            TE_omnibus_theoretical_causal_vars[t] = 0.5 * np.log(
                numerator / denominator
                )
        else:
            TE_omnibus_theoretical_inferred_vars = np.NaN
            TE_omnibus_theoretical_causal_vars = np.NaN

    df.loc[run_name, 'TE_omnibus_empirical'] = TE_omnibus_empirical
    df.loc[
        run_name,
        'TE_omnibus_theoretical_inferred_vars'
        ] = TE_omnibus_theoretical_inferred_vars
    df.loc[
        run_name,
        'TE_omnibus_theoretical_causal_vars'
        ] = TE_omnibus_theoretical_causal_vars

    # -------------------------------------------------------------------------
    # Spectral radius
    # -------------------------------------------------------------------------
    radius = np.NaN
    if traj.par.node_dynamics.model == 'AR_gaussian_discrete':
        # Use VAR reduced form defined above to compute the radius
        radius = max(np.abs(np.linalg.eigvals(var_reduced_form)))
    df.loc[run_name, 'spectral_radius'] = radius

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
    print('WARNING: postprocessing DataFrame NOT saved (debug mode)')
