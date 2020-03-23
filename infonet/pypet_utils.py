from pypet import Trajectory
import os
import numpy as np


def my_print(*args, sep=',', file=None):
    if isinstance(file, str):
        print(args, sep=sep, file=open(file, "a"))
    else:
        print(args, sep=sep)


def print_traj_leaves(traj, node_name, print_to_file=None):
    try:
        node = traj.f_get(node_name)
    except AttributeError as e:
        print(e)
    else:
        if node.v_is_group:
            if node.f_has_leaves():
                for leave in node.f_iter_leaves():
                    if leave.f_has_range():
                        my_print(
                            leave.v_full_name,
                            np.unique(leave.f_get_range()),
                            sep=' : ',
                            file=print_to_file)
                    else:
                        my_print(
                            leave.v_full_name,
                            leave.f_val_to_str(),
                            sep=' : ',
                            file=print_to_file)
            else:
                print('node \'{}\' has no leaves'.format(node_name))
        else:
            if node.f_has_range():
                my_print(
                    node.v_full_name,
                    np.unique(node.f_get_range()),
                    sep=' : ',
                    file=print_to_file)
            else:
                print(
                    node.v_full_name,
                    node.f_val_to_str(),
                    sep=' : ',
                    file=print_to_file)


def print_traj_parameters_explored(traj_dir):
    # Load the trajectory from the hdf5 file
    # Only load parameters, results will be loaded at runtime (auto loading)
    
    #traj_dir = os.path.join('trajectories', '2019_03_21_22h48m29s_HCP_test')
    #if not os.path.isdir(traj_dir):
    #    traj_dir = os.path.join('..', traj_dir)
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
    print('number of runs = {0}'.format(runs_n))
    # Get list of explored parameters
    parameters_explored = [str.split(par, '.').pop() for par in (
        traj.f_get_explored_parameters())]
    print(parameters_explored)
