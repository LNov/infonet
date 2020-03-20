from pypet import Trajectory
import os
import numpy as np
from scipy.linalg import block_diag
from scipy.linalg import solve_discrete_lyapunov
import pandas as pd
import networkx as nx
from mylib_pypet import print_leaves

fdr = False
save_results = True

# Load the trajectory from the hdf5 file
# Only load parameters, results will be loaded at runtime (auto loading)
traj_dir = os.path.join('trajectories', '2019_03_21_22h48m29s_HCP_test')
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

print_leaves(traj, 'parameters')
