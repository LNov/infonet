import os
import shutil
from datetime import datetime
from pypet import Trajectory
from mylib_pypet import print_leaves

# Choose base directory containing trajectory directories
base_dir = '/project/RDS-FEI-InfoDynFuncStruct-RW/Leo/inference/trajectories/'
# base_dir = '/home/leo/Projects/inference/trajectories/'
# base_dir = 'C:\\DATA\\Google Drive\\Materiale progetti in corso\\USyd\\Information network inference\\trajectories\\'
traj_dir_names = [
    '2019_03_06_12h43m37s_GC_on_AR_100samples',
    '2019_03_06_14h29m41s_GC_on_AR_1000samples',
    '2019_03_06_19h04m58s_GC_on_AR_10000samples',
]
traj_filename = 'traj.hdf5'

# Choose output directory to store merged trajectories
output_dir = base_dir

# Assemble full paths
traj_fullpath_list = [
    os.path.join(
        base_dir, dir_name, traj_filename) for dir_name in traj_dir_names]

# Check if trajectory files exist
print('Checking if trajectory files exist...')
for traj_fullpath in traj_fullpath_list:
    if os.path.exists(traj_fullpath):
        print('OK: Trajectory file found: {0}'.format(traj_fullpath))
    else:
        raise ValueError(
            'ERROR: Trajectory file not found: {0}'.format(traj_fullpath))

# Load parameters and results, and check that all trajectories
# to be merged are completed
print('Loading parameters and results...')
trajectories = [Trajectory() for traj_fullpath in traj_fullpath_list]
for traj_i in range(len(traj_fullpath_list)):
    trajectories[traj_i].f_load(
        filename=traj_fullpath_list[traj_i],
        index=0,
        load_parameters=2,
        load_derived_parameters=2,
        load_results=2,
        load_other_data=2,
        force=True)
    # Ensure that the trajectory was explored
    if not trajectories[traj_i].f_is_completed():
        raise ValueError(
            'ERROR: Trajectory not completed or not assembled: {0}'.format(
                traj_fullpath_list[traj_i]))

print('Creating output directory...')
# Create output sub-directory with timestamp
# Add trailing slash if missing
output_dir = os.path.join(output_dir, '')
# Ensure that provided directory exists
if not os.path.isdir(output_dir):
    raise ValueError("ERROR: Output directory not found: {0}".format(
        output_dir))
# Convert to full path
output_dir = os.path.abspath(output_dir)
# Add time stamp (the final '' is to make sure there is a trailing slash)
output_dir = os.path.join(
    output_dir,
    datetime.now().strftime("%Y_%m_%d_%Hh%Mm%Ss") + '_merged',
    '')
# Create directory with time stamp
os.makedirs(output_dir)
# Change current directory to the one containing the trajectory files
os.chdir(output_dir)
print('Merged trajectory will be stored in: {0}'.format(output_dir))

# Set final trajectory path
traj_final_fullpath = os.path.join(output_dir, traj_filename)
# Copy the last trajectory in the list to the output directory
# It will be used as a starting trajectory to merge the others into
shutil.copy(traj_fullpath_list[-1], traj_final_fullpath)
# Then remove last trajectory from list
trajectories.pop()
traj_fullpath_list.pop()
# Load copied trajectory from output directory.
# We'll merge the other trajectories into this one,
# which will then become the final trajectory
traj_final = Trajectory()
traj_final.f_load(
        filename=traj_final_fullpath,
        index=0,
        load_parameters=2,
        load_derived_parameters=2,
        load_results=2,
        load_other_data=2,
        force=True)

# Merge trajectories
print('Merging trajectory files...')
for traj_to_merge in trajectories:
    traj_final.f_merge(
        traj_to_merge,
        remove_duplicates=True,
        backup=False,
        # backup_filename=True,
        delete_other_trajectory=False,
        slow_merge=True,
        ignore_data=()
    )
print('Finished merging')

print('\nFinal trajectory parameters:')
print_leaves(traj_final, 'parameters')
print('----------------------------------------------------------\n')
