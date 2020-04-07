import network_dynamics
import information_dynamics
import pypet_utils
import sys
import os
from datetime import datetime
import pickle
import numpy as np
from pypet import Environment
from pypet import PickleResult
from pypet import pypetconstants
from pypet.utils.explore import cartesian_product
import networkx as nx
import jpype


# Use pickle module to load dictionaries from disk
def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def information_network_inference(traj):
    """Runs Information Network inference

    :param traj:

        Container with all parameters.

    :return:

        Inferred Information Network

    """

    # Start timer
    # start_monotonic = time.monotonic()
    # start_perf_counter = time.perf_counter()
    # start_process_time = time.process_time()

    # Generate initial network
    G = network_dynamics.generate_network(traj.par.topology.initial)
    # Get adjacency matrix
    adjacency_matrix = np.array(nx.to_numpy_matrix(
        G,
        nodelist=np.array(range(0, traj.par.topology.initial.nodes_n)),
        dtype=int))
    # Add self-loops
    np.fill_diagonal(adjacency_matrix, 1)

    # Generate initial node coupling
    coupling_matrix = network_dynamics.generate_coupling(
        traj.par.node_coupling.initial,
        adjacency_matrix)

    # Generate delay
    delay_matrices = network_dynamics.generate_delay(traj.par.delay.initial, adjacency_matrix)

    # Generate coefficient matrices
    coefficient_matrices = np.transpose(
        delay_matrices * coupling_matrix,
        (0, 2, 1))

    # Run dynamics
    time_series = network_dynamics.run_dynamics(
        traj.par.node_dynamics,
        coefficient_matrices)

    # Perform Information Network Inference
    network_inference_result = information_dynamics.infer_network(
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


def bTE_on_existing_links(traj):
    nodes_n = traj.par.topology.initial.nodes_n
    # Generate initial network
    G = network_dynamics.generate_network(traj.par.topology.initial)
    # Get adjacency matrix
    adjacency_matrix = np.array(nx.to_numpy_matrix(
        G,
        nodelist=np.array(range(0, nodes_n)),
        dtype=int))
    # Add self-loops
    np.fill_diagonal(adjacency_matrix, 1)
    # Generate initial node coupling
    coupling_matrix = network_dynamics.generate_coupling(
        traj.par.node_coupling.initial,
        adjacency_matrix)
    # Generate delay
    delay_matrices = network_dynamics.generate_delay(
        traj.par.delay.initial, adjacency_matrix)
    # Generate coefficient matrices
    coefficient_matrices = np.transpose(
        delay_matrices * coupling_matrix,
        (0, 2, 1))
    # Run dynamics
    time_series = network_dynamics.run_dynamics(
        traj.par.node_dynamics,
        coefficient_matrices)

    bTE_empirical_matrix = information_dynamics.compute_bTE_on_existing_links(
        time_series,
        delay_matrices,
        traj.par.node_dynamics.model,
        traj.par.estimation.history_target,
        traj.par.estimation.history_source,
    )

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
    # traj.f_add_result(
    #     '$.node_dynamics',
    #     time_series=time_series,
    #     comment='')
    traj.f_add_result(
        PickleResult,
        '$.bTE',
        bTE_empirical_matrix=bTE_empirical_matrix,
        comment='')

    jSystem = jpype.JPackage("java.lang").System
    jSystem.gc()


def main():
    """Main function to protect the *entry point* of the program.

    If you want to use multiprocessing with SCOOP you need to wrap your
    main code creating an environment into a function. Otherwise
    the newly started child processes will re-execute the code and throw
    errors (also see http://scoop.readthedocs.org/en/latest/usage.html#pitfalls).

    """

    # Load settings from file
    settings_file = 'pypet_settings.pkl'
    settings = load_obj(settings_file)
    # Print settings dictionary
    print('\nSettings dictionary:')
    for key, value in settings.items():
        print(key, ' : ', value)
    print('\nParameters to explore:')
    for key, value in settings.items():
            if isinstance(value, list):
                print(key, ' : ', value)

    # Create new folder to store results
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
    print('Trajectory and results will be stored in: {0}'.format(traj_dir))

    # Create an environment that handles running.
    # Let's enable multiprocessing with scoop:
    env = Environment(
        trajectory='traj',
        comment='',
        add_time=False,
        log_config='DEFAULT',
        log_stdout=True,  # log everything that is printed, will make the log file HUGE
        filename=traj_dir,  # filename or just folder (name will be automatic in this case)
        multiproc=True,
        #use_pool=True,
        #ncores=10,
        #freeze_input=True,
        use_scoop=True,
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

    # Add config parameters (those that DO NOT influence the final result of the experiment)
    traj.f_add_config('parallel_target_analysis', True)
    #traj.f_add_config('debug', False)
    #traj.f_add_config('max_mem_frac', 0.7)

    # Set up trajectory parameters
    param_to_explore = {}
    for key, val in settings.items():
        if isinstance(val, list):
            param_to_explore[key] = val
            traj.f_add_parameter(key, val[0])
        else:
            traj.f_add_parameter(key, val)

    # Define parameter combinations to explore (a trajectory in
    # the parameter space). The second argument, the tuple, specifies the order
    #  of the cartesian product.
    # The variable on the right most side changes fastest and defines the
    # 'inner for-loop' of the cartesian product
    explore_dict = cartesian_product(
        param_to_explore,
        tuple(param_to_explore.keys())
    )
    # explore_dict = cartesian_product(
    #     {
    #         'network_inference.algorithm': ['bMI_greedy', 'bTE_greedy', 'mTE_greedy'],
    #         #'node_coupling.initial.weight_distribution': ['fixed'],
    #         'repetition_i': np.arange(0, 5, 1).tolist(),
    #         'topology.initial.nodes_n': np.arange(50, 50+1, 300).tolist(),
    #         'node_dynamics.samples_n': np.array([1000, 10000]).tolist(),
    #         'network_inference.p_value': np.array([0.001]).tolist(),
    #         #'node_coupling.initial.self_coupling': np.arange(-0.5, 0.5 + 0.001, 0.1).tolist(),
    #         #'node_coupling.initial.total_cross_coupling': np.arange(-1., 1 + 0.001, 0.2).tolist(),
    #         #'topology.initial.WS_p': np.around(np.logspace(-2.2, 0, 10), decimals=4).tolist(),
    #     },
    #     (
    #         'network_inference.algorithm',
    #         #'node_coupling.initial.weight_distribution',
    #         'network_inference.p_value',
    #         'node_dynamics.samples_n',
    #         'topology.initial.nodes_n',
    #         #'topology.initial.WS_p',
    #         #'node_coupling.initial.self_coupling',
    #         #'node_coupling.initial.total_cross_coupling',
    #         'repetition_i',
    #     )
    # )
    traj.f_explore(explore_dict)

    # Store trajectory parameters to disk
    pypet_utils.print_traj_leaves(traj, 'parameters', 'traj_parameters.txt')

    # Run the experiment
    env.run(information_network_inference)
    # env.run(bTE_on_existing_links)

    # Check that all runs are completed
    assert traj.f_is_completed()

    # Finally disable logging and close all log-files
    env.disable_logging()


if __name__ == '__main__':
    # This will execute the main function in case the script is called from the one true
    # main process and not from a child processes spawned by your environment.
    # Necessary for multiprocessing under Windows.
    main()
