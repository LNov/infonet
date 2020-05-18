import numpy as np
import pandas as pd
from idtxl.data import Data
from idtxl.bivariate_mi import BivariateMI
from idtxl.bivariate_te import BivariateTE
from idtxl.multivariate_mi import MultivariateMI
from idtxl.multivariate_te import MultivariateTE
from idtxl.estimators_jidt import JidtGaussianTE
from idtxl.estimators_jidt import JidtDiscreteTE
from idtxl.stats import network_fdr
from scoop import futures
import multiprocessing as mp
import itertools


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


def infer_network(network_inference, time_series, parallel_target_analysis=False):
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
            # 'alpha_max_stat',
            # 'alpha_min_stat',
            # 'alpha_omnibus',
            # 'alpha_max_seq',
            # 'alpha_fdr'
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
            # 'alpha_max_stat',
            # 'alpha_min_stat',
            # 'alpha_omnibus',
            # 'alpha_max_seq',
            # 'alpha_fdr'
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
            # 'alpha_max_stat',
            # 'alpha_min_stat',
            # 'alpha_omnibus',
            # 'alpha_max_seq',
            # 'alpha_fdr'
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
            # 'alpha_max_stat',
            # 'alpha_min_stat',
            # 'alpha_omnibus',
            # 'alpha_max_seq',
            # 'alpha_fdr'
        ],
        'cross_corr': [
            'min_lag_sources',
            'max_lag_sources'
        ]
    })
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
            #     'config.max_mem_frac',
            #     'kraskov_k',
            # }
            # for key in optional_settings_keys:
            #     if traj.f_contains(key, shortcuts=True):
            #         key_last = key.rpartition('.')[-1]
            #         settings[key_last] = traj[key]
            #         print('Using optional setting \'{0}\'={1}'.format(
            #             key_last,
            #             traj[key])
            #         )

            if algorithm == 'bMI_greedy':
                network_analysis = BivariateMI()
                settings['min_lag_sources'] = 0
                settings['max_lag_sources'] = 0
            if algorithm == 'mMI_greedy':
                network_analysis = MultivariateMI()
                settings['min_lag_sources'] = 0
                settings['max_lag_sources'] = 0
            if algorithm == 'bTE_greedy':
                network_analysis = BivariateTE()
            if algorithm == 'mTE_greedy':
                network_analysis = MultivariateTE()

            if parallel_target_analysis:
                # # Use SCOOP to create a generator of map results, each
                # # correspinding to one map iteration
                # res_iterator = futures.map_as_completed(
                #     network_analysis.analyse_single_target,
                #     itertools.repeat(settings, nodes_n),
                #     itertools.repeat(dat, nodes_n),
                #     list(range(nodes_n))
                # )
                # # Run analysis
                # res_list = list(res_iterator)

                # use multiprocessing.Pool() to parallelise over targets
                processors_n = mp.cpu_count()
                print('Starting parallel analysis over {0} processors'.format(
                    processors_n))
                pool = mp.Pool(processors_n)
                result_objects = [
                    pool.apply_async(
                        network_analysis.analyse_single_target,
                        args=(settings, dat, target))
                    for target in range(nodes_n)]
                # result_objects is a list of pool.ApplyResult objects
                res_list = [r.get() for r in result_objects]
                pool.close()
                pool.join()
                # combine results (and apply FDR if requested)
                if settings['fdr_correction']:
                    res = network_fdr(
                        {'alpha_fdr': settings['alpha_fdr']},
                        *res_list)
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
            raise ParameterValue(
                    algorithm,
                    msg='Network inference algorithm not yet implemented')


def compute_bTE_on_existing_links(time_series, delay_matrices, node_dynamics_model, history_target, history_source):
    nodes_n = np.shape(delay_matrices)[1]
    delay_flattened = delay_matrices.max(axis=0)
    # initialise an empty data object
    dat = Data()
    # Load time series
    if node_dynamics_model == 'boolean_random':
        normalise = False
    elif node_dynamics_model == 'AR_gaussian_discrete':
        normalise = True
    dat = Data(
        time_series,
        dim_order='psr',
        normalise=normalise)
    data = dat.data
    # Compute empirical bTE between all pairs
    settings = {}
    settings['history_target'] = history_target
    settings['history_source'] = history_source
    bTE_empirical_matrix = np.full((nodes_n, nodes_n), np.NaN)
    for X in range(nodes_n):
        for Y in range(nodes_n):
            if (delay_flattened[X, Y] > 0) and (X != Y):
                settings['source_target_delay'] = int(delay_flattened[X, Y])
                if node_dynamics_model == 'boolean_random':
                    estimator = JidtDiscreteTE(settings)
                elif node_dynamics_model == 'AR_gaussian_discrete':
                    estimator = JidtGaussianTE(settings)
                bTE_empirical_matrix[X, Y] = estimator.estimate(
                    data[X, :, 0], data[Y, :, 0])
    return bTE_empirical_matrix
