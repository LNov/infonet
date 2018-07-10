import sys
import os
from datetime import datetime
import shutil
import time
import numpy as np
import pandas as pd
from pypet import Environment
from pypet import Trajectory
from pypet import pypetconstants
from pypet.utils.explore import cartesian_product
import networkx as nx
from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data
from scoop import futures
import copyreg
import itertools


def pickle_mTE(obj):
    print("pickling a C instance...")
    return MultivariateTE, (obj.max_lag_sources, obj.min_lag_sources, obj.options)


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
        nodes_n = len(time_series)

        # Check if data can be normalised per process (assuming the
        # first dimension represents processes, as in the rest of the code)
        can_be_normalised = np.all(np.std(time_series, axis=1) > 0)
        if not can_be_normalised:
            print('Time series can not be normalised')

        # initialise an empty data object
        dat = Data()

        # Load time series
        dat = Data(time_series, dim_order='ps', normalise=can_be_normalised)

        algorithm = network_inference.algorithm
        if algorithm == 'mTE_greedy':
            new_release = False
            if new_release:
                # Set analysis options
                network_analysis = MultivariateTE()

                settings = {
                    'min_lag_sources': network_inference.min_lag_sources,
                    'max_lag_sources': network_inference.max_lag_sources,
                    'cmi_estimator':  network_inference.cmi_estimator,
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
                    res = {res_partial['target']: res_partial for res_partial in list(res_iterator)}
                else:
                    # Run analysis
                    res = network_analysis.analyse_network(data=dat, settings=settings)

            else:
                # Make MultivariateTE pickable, reference:
                # https://docs.python.org/3/library/copyreg.html
                copyreg.pickle(MultivariateTE, pickle_mTE)

                # Set analysis options
                network_analysis = MultivariateTE(
                    min_lag_sources=network_inference.min_lag_sources,
                    max_lag_sources=network_inference.max_lag_sources,
                    options={
                        'cmi_estimator':  network_inference.cmi_estimator,
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
                )

                if parallel_target_analysis:
                    # Use SCOOP to create a generator of map results, each
                    # correspinding to one map ieration
                    #my_function = network_analysis.analyse_single_target
                    res_iterator = futures.map_as_completed(
                        network_analysis.analyse_single_target,
                        itertools.repeat(dat, nodes_n),
                        list(range(nodes_n))
                    )
                    # Run analysis
                    res = {res_partial['target']: res_partial for res_partial in list(res_iterator)}
                else:
                    # Run analysis
                    res = network_analysis.analyse_network(dat)
            
            return res

        else:
            raise ParameterValue(algorithm, msg='Network inference algorithm not yet implemented')  


def information_network_inference(traj, traj_dir):

    # Change current directory to the one containing the trajectory files
    #os.chdir(traj_dir)

    # Check if network files exist and add them to the trajectory
    single_run_network_file_names = [
        'delay.initial.delay_matrix',
        'node_coupling.initial.coupling_matrix',
        'topology.initial.adjacency_matrix'
    ]
    for filename in single_run_network_file_names:
        path = os.path.join(traj_dir, '.'.join([traj.v_crun, filename]) + '.csv')
        if os.path.exists(path):
            obj = pd.read_csv(path, index_col=0)#, dtype=np.ndarray)
            # Convert from strings to numpy arrays
            if type(obj.values[0, 0]) == str:
                obj = pd.DataFrame([[np.array(eval(x)) for x in row] for row in obj.values])
            traj.f_add_result('$.' + filename, obj)
        else:
            raise ValueError('ERROR: file missing: {0}'.format(path))

    # Check if time series file exists and add it to the trajectory
    filename = 'node_dynamics.time_series'
    path = os.path.join(traj_dir, '.'.join([traj.v_crun, filename]) + '.npy')
    if os.path.exists(path):
        time_series = np.load(path)
        traj.f_add_result('$.node_dynamics.time_series', time_series)

        # Perform Information Network Inference
        network_inference_result = pd.DataFrame(perform_network_inference(traj.par.network_inference, time_series, traj.config.parallel_target_analysis))

        # Save to disk
        network_inference_result.to_csv(os.path.join(traj_dir, '.'.join([traj.v_crun, 'network_inference.network_inference_result', 'csv'])))

        # Add results to the trajectory
        traj.f_add_result('$.network_inference', network_inference_result=network_inference_result, comment='')

        #traj.f_store_item('results.$.network_inference')
        
    else:
        raise ValueError('ERROR: file missing: {0}'.format(path))
    return


def main():

    # Set path of original trajectory
    traj_dir = os.path.abspath(os.path.join('trajectories', 'logistic_100samples'))
    traj_filename = 'traj.hdf5'
    traj_fullpath = os.path.join(traj_dir, traj_filename)

    # Choose output folder for the new analysis and set it as current directory
    output_dir = os.path.join(traj_dir, 'new_analysis' + datetime.now().strftime("%Y_%m_%d_%Hh%Mm%Ss"), '')
    os.makedirs(output_dir)
    os.chdir(output_dir)
    # Copy the original trajectory file to the new output_dir
    traj_fullpath_new = os.path.join(output_dir, traj_filename)
    shutil.copy(traj_fullpath, traj_fullpath_new)

    # Load the trajectory from the hdf5 file
    traj1 = Trajectory()
    traj1.f_load(
        filename=traj_fullpath_new,
        index=0,
        load_parameters=2,
        load_results=2,
        force=True
    )

    # Save single result files to disk
    for run_name in traj1.f_get_run_names():

        traj1.f_set_crun(run_name)

        adjacency_matrix = traj1.results[run_name].topology.initial.adjacency_matrix
        coupling_matrix = traj1.results[run_name].node_coupling.initial.coupling_matrix
        delay_matrix = traj1.results[run_name].delay.initial.delay_matrix
        time_series = traj1.results[run_name].node_dynamics.time_series
        delay_matrix = traj1.results[run_name].delay.initial.delay_matrix

        # Save objects to disk
        adjacency_matrix.to_csv(os.path.join(output_dir, '.'.join([traj1.v_crun, 'topology.initial.adjacency_matrix', 'csv'])))
        coupling_matrix.to_csv(os.path.join(output_dir, '.'.join([traj1.v_crun, 'node_coupling.initial.coupling_matrix', 'csv'])))
        delay_matrix.to_csv(os.path.join(output_dir, '.'.join([traj1.v_crun, 'delay.initial.delay_matrix', 'csv'])))
        np.save(os.path.join(output_dir, '.'.join([traj1.v_crun, 'node_dynamics.time_series', 'npy'])), time_series)

    # Load trajectory again, this time using the option
    # as_new, which does not load the results and allows
    # to re-explore the trajectory
    traj = Trajectory()
    traj.f_load(
        filename=traj_fullpath_new,
        index=0,
        as_new=True,
        force=True
    )

    # Rename traj file to be a backup
    os.rename(traj_fullpath_new, traj_fullpath_new + '.backup')

    # Store the loaded trajectory (only parameters, no results)
    # Since the file was renamed, it will be stored to a new file
    traj.f_store()

    # Create an environment that handles running.
    # Let's enable multiprocessing with scoop:
    env = Environment(
        trajectory=traj,
        #do_single_runs=True,
        comment='Perform Granger causality analysis on logistic '
                'dataset to compare the performance to the mTE analysis. ',
        add_time=False,
        log_config='DEFAULT',
        log_stdout=True,  # log everything thst is printed, will make the log file HUGE
        multiproc=True,

        use_scoop=True,
        wrap_mode=pypetconstants.WRAP_MODE_LOCAL,
        memory_cap=10,
        swap_cap=1
        #cpu_cap=30
    )

    assert not traj.f_is_completed()

    # Unlock parameters
    for param in traj._parameters.values():
        param.f_unlock()

    # -------------------------------------------------------------------
    # Add config parameters (those that DO NOT influence the final result of the experiment)
    traj.f_add_config('parallel_target_analysis', True, comment='Analyse targets in parallel')

    # -------------------------------------------------------------------
    # Parameters characterizing the network inference algorithm
    traj.parameters['network_inference.cmi_estimator'] = 'JidtGaussianCMI'
    #traj.parameters['network_inference.cmi_estimator'] = 'JidtKraskovCMI'
    #traj.parameters['network_inference.cmi_estimator'] = 'OpenCLKraskovCMI'
    traj.parameters['network_inference.permute_in_time'] = True
    traj.parameters['network_inference.jidt_threads_n'] = 1
    traj.parameters['network_inference.n_perm_max_stat'] = 2000
    traj.parameters['network_inference.n_perm_min_stat'] = 2000
    traj.parameters['network_inference.n_perm_omnibus'] = 2000
    traj.parameters['network_inference.n_perm_max_seq'] = 2000
    traj.parameters['network_inference.fdr_correction'] = False

    # -------------------------------------------------------------------
    # Run the experiment again
    env.run(information_network_inference, traj_dir=output_dir)

    # Check that all runs are completed
    assert traj.f_is_completed()

    #
    #traj.f_store()

    # Finally disable logging and close all log-files
    env.disable_logging()


if __name__ == '__main__':
    # This will execute the main function in case the script is called from the one true
    # main process and not from a child processes spawned by your environment.
    # Necessary for multiprocessing under Windows.
    main()
