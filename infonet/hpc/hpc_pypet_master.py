from infonet import pypet_utils
import sys
import os
from datetime import datetime
import pickle
from pypet import Trajectory
from pypet.utils.explore import cartesian_product
from subprocess import call


# Use pickle module to load dictionaries from disk
def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def run_job_array(job_script_path, job_settings, job_args={}):
    settings = ' '.join(['-{0} {1}'.format(key, job_settings[key]) for key in job_settings.keys()])
    args = '-v ' + ','.join(['{0}="{1}"'.format(key, job_args[key]) for key in job_args.keys()])
    # Submit PBS job
    call(
        ('qsub {1} {2} {0}').format(
            job_script_path,
            settings,
            args
            ),
        shell=True,
        timeout=None
    )


def main():
    """Main function to protect the *entry point* of the program."""

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
        # Add trailing slash if missing
        dir_provided = os.path.join(sys.argv[1], '')
        # Check if provided directory exists
        if os.path.isdir(dir_provided):
            # Convert to full path
            traj_dir = os.path.abspath(dir_provided)
        else:
            print('WARNING: Output path not found, current directory will be used instead')
    else:
        print('WARNING: Output path not provided, current directory will be used instead')
    # Add time stamp (the final '' is to make sure there is a trailing slash)
    traj_dir = os.path.join(traj_dir, datetime.now().strftime("%Y_%m_%d_%Hh%Mm%Ss"), '')
    # Create directory with time stamp
    os.makedirs(traj_dir)
    # Change current directory to the one containing the trajectory files
    os.chdir(traj_dir)
    print('Trajectory and results will be stored in: {0}'.format(traj_dir))

    # Create new pypet Trajectory object
    traj_filename = 'traj.hdf5'
    traj_fullpath = os.path.join(traj_dir, traj_filename)
    traj = Trajectory(filename=traj_fullpath)

    # -------------------------------------------------------------------
    # Add config parameters (those that DO NOT influence the final result of the experiment)
    traj.f_add_config('debug', False, comment='Activate debug mode')
#    #traj.f_add_config('max_mem_frac', 0.7, comment='Fraction of global GPU memory to use')

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
        tuple(param_to_explore.keys()))

    print(explore_dict)
    traj.f_explore(explore_dict)

    # Store trajectory parameters to disk
    pypet_utils.print_traj_leaves(
        traj,
        'parameters',
        file=os.path.join(traj_dir, 'traj_parameters.txt'))

    # Store trajectory
    traj.f_store()

    # Define PBS script
    bash_lines = '\n'.join([
        '#! /bin/bash',
        '#PBS -P InfoDynFuncStruct',
        '#PBS -l select=1:ncpus=1:mem=1GB',
        #'#PBS -l select=1:ncpus=1:ngpus=1:mem=1GB',
        '#PBS -M lnov6504@uni.sydney.edu.au',
        '#PBS -m abe',
        'module load java',
        'module load python/3.5.1',
        'module load cuda/8.0.44',
        'source /project/RDS-FEI-InfoDynFuncStruct-RW/Leo/idtxl_env/bin/activate',
        'cd ${traj_dir}',
        'python ${python_script_path} ${traj_dir} ${traj_filename} ${file_prefix} $PBS_ARRAY_INDEX'
        ])

    # Save PBS script file (automatically generated)
    bash_script_name = 'run_python_script.pbs'
    job_script_path = os.path.join(traj_dir, bash_script_name)
    with open(job_script_path, 'w', newline='\n') as bash_file:
        bash_file.writelines(bash_lines)

    # Run job array
    job_walltime_hours = 0
    job_walltime_minutes = 5
    #after_job_array_ends = 1573895
    job_settings = {
        'N': 'run_traj',
        'l': 'walltime={0}:{1}:00'.format(job_walltime_hours, job_walltime_minutes),
        #'W': 'depend=afteranyarray:{0}[]'.format(after_job_array_ends),
        'q': 'defaultQ'
    }
    if len(traj.f_get_run_names()) > 1:
        job_settings['J'] = '{0}-{1}'.format(0, len(traj.f_get_run_names()) - 1)

    job_args = {
        'python_script_path': '/project/RDS-FEI-InfoDynFuncStruct-RW/Leo/inference/hpc_pypet_single_run.py',
        'traj_dir': traj_dir,
        'traj_filename': traj_filename,
        'file_prefix': 'none'
    }
    run_job_array(job_script_path, job_settings, job_args)


if __name__ == '__main__':
    # This will execute the main function in case the script is called from the one true
    # main process and not from a child processes spawned by your environment.
    # Necessary for multiprocessing under Windows.
    main()
