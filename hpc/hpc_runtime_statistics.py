import os
import fnmatch
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Choose base directory containing trajectory directories
#base_dir = '/project/RDS-FEI-InfoDynFuncStruct-RW/Leo/inference/trajectories/'
base_dir = 'C:\\DATA\\Google Drive\\Materiale progetti in corso\\USyd\\Information network inference\\trajectories\\'

# Choose trajectory folders
traj_dir_names = [
    #'2018_01_23_11h15m34s_logistic_70nodes_10000samples_rep01',
    #'2018_01_29_15h45m48s_logistic_70nodes_10000samples_rep23',
    #'2018_01_31_20h16m54s_logistic_70nodes_10000samples_rep45',
    #'2018_02_04_16h40m03s_logistic_70nodes_10000samples_rep67',
    #'2018_02_09_20h57m49s_logistic_70nodes_10000samples_rep89',
    #'2018_02_22_11h28m59s_logistic_100nodes_10000samples_rep0',
    #'2018_02_23_15h37m43s_logistic_100nodes_10000samples_rep1',
    #'2018_02_26_18h02m06s_logistic_100nodes_10000samples_rep2',
    #'2018_02_28_22h59m48s_logistic_100nodes_10000samples_rep3',
    #'2018_03_02_15h29m15s_logistic_100nodes_10000samples_rep4',
    #'2018_03_05_11h59m06s_logistic_100nodes_10000samples_rep5',
    #'2018_03_06_22h57m58s_logistic_100nodes_10000samples_rep6',
    #'2018_03_09_00h25m26s_logistic_100nodes_10000samples_rep7',
    #'2018_03_12_10h35m32s_logistic_100nodes_10000samples_rep8',
    #'2018_03_21_10h27m54s_logistic_100nodes_10000samples_rep9',
    '2018_11_30_19h00m44s'
    ]
traj_filename = 'traj.hdf5'
traj_dir_fullpaths = [
    os.path.join(base_dir, dir_name) for dir_name in traj_dir_names]
for traj_dir in traj_dir_fullpaths:
    if os.path.exists(traj_dir):
        print('OK: Directory found: {0}'.format(traj_dir))
    else:
        raise ValueError('ERROR: Directory not found: {0}'.format(traj_dir))

regexes_dict = {
    'name': re.compile(r'Job Name:\s+(\w+)'),
    'id': re.compile(r'Job Id:\s+(\d+)'),
    'array_id': re.compile(r'array index:\s+(\d+)'),
    'exit_status': re.compile(r'Exit Status:\s+(\d+)'),
    'ncpus': re.compile(r'ncpus=(\d+)'),
    'ngpus': re.compile(r'ngpus=(\d+)'),
    'mem_requested': re.compile(r'Mem requested:\s+(\d.+)GB'),
    'mem_used': re.compile(r'Mem used:\s+(\d.+)GB'),
    'walltime_requested_h': re.compile(
        r'Walltime requested:\s+(\d+):(?:\d+):(?:\d+)(?: +):'),
    'walltime_requested_m': re.compile(
        r'Walltime requested:\s+(?:\d+):(\d+):(?:\d+)(?: +):'),
    'walltime_requested_s': re.compile(
        r'Walltime requested:\s+(?:\d+):(?:\d+):(\d+)(?: +):'),
    'walltime_used_h': re.compile(r'Walltime used:\s+(\d+):(?:\d+):(?:\d+)'),
    'walltime_used_m': re.compile(r'Walltime used:\s+(?:\d+):(\d+):(?:\d+)'),
    'walltime_used_s': re.compile(r'Walltime used:\s+(?:\d+):(?:\d+):(\d+)')
    }

keys = regexes_dict.keys()


for traj_dir in traj_dir_fullpaths:
    # Change current directory to the one containing the trajectory files
    os.chdir(traj_dir)

    # Create new DataFrame
    df = pd.DataFrame(columns=keys)

    for filename in fnmatch.filter(os.listdir(traj_dir), 'run[!_]*_usage'):
        with open(filename, 'r') as file:
            content = file.read()
            s = pd.Series(index=keys)
            for key in keys:
                search = regexes_dict[key].search(content)
                if search:
                    s[key] = search.group(1)
            s['runtime'] = (
                float(s['walltime_used_s'])
                + 60 * float(s['walltime_used_m'])
                + 3600 * float(s['walltime_used_h'])
                )
            df = df.append(s, ignore_index=True)

    # upgrade Score from float to integer
    df = df.apply(pd.to_numeric, errors='ignore')

    aggregate_repetitions = df.groupby('name').agg(lambda x: x.tolist())

    for index, row in aggregate_repetitions.iterrows():
        print(row.name)
        print('estimated network size: {0}'.format(len(row['array_id'])))
        array = np.array(row['runtime'])
        #print(array.mean())
        #print(array.std())
        print('{0:.3f} hours = {1:.2f} minutes = {2:.2f} seconds\n'.format(
            array.max()/3600,
            array.max()/60,
            array.max()
            ))
