import os
import fnmatch
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Choose base directory containing trajectory directories
base_dir = '/project/RDS-FEI-InfoDynFuncStruct-RW/Leo/inference/trajectories/'
#base_dir = 'C:\\DATA\\Google Drive\\Materiale progetti in corso\\USyd\\Information network inference\\trajectories\\'

# Choose trajectory folders
traj_dir_names = [
    '2019_03_21_08h06m59s'

    #'2017_08_31_23h27m22s_logistic_10nodes_1000samples_rep0123',
    #'2017_09_01_00h33m23s_logistic_10nodes_1000samples_rep456789'

    #'2017_08_26_22h50m40s_logistic_40nodes_1000samples_rep01',
    #'2017_09_01_01h39m47s_logistic_40nodes_1000samples_rep23',
    #'2017_09_01_09h22m51s_logistic_40nodes_1000samples_rep45',
    #'2017_09_01_17h15m32s_logistic_40nodes_1000samples_rep67',
    #'2017_09_01_23h17m23s_logistic_40nodes_1000samples_rep89'

    #'2017_08_25_20h48m36s_logistic_70nodes_1000samples_0001_rep01',
    #'2017_08_30_17h06m39s_logistic_70nodes_1000samples_rep2',
    #'2017_08_30_17h23m32s_logistic_70nodes_1000samples_rep3',
    #'2017_08_30_23h28m10s_logistic_70nodes_1000samples_rep4',
    #'2017_08_30_23h42m41s_logistic_70nodes_1000samples_rep5',
    #'2017_08_31_03h58m11s_logistic_70nodes_1000samples_rep6',
    #'2017_08_31_05h01m54s_logistic_70nodes_1000samples_rep7',
    #'2017_09_02_06h23m41s_logistic_70nodes_1000samples_rep8',
    #'2017_09_02_06h46m09s_logistic_70nodes_1000samples_rep9'

    #'2017_08_25_04h04m55s_logistic_100nodes_1000samples_0001_rep0',
    #'2017_08_25_04h28m46s_logistic_100nodes_1000samples_0001_rep1',
    #'2017_08_27_06h43m56s_logistic_100nodes_1000samples_0001_rep23',
    #'2017_08_28_02h34m29s_logistic_100nodes_1000samples_0001_rep45',
    #'2017_08_28_19h42m27s_logistic_100nodes_1000samples_0001_rep67',
    #'2017_08_29_15h39m09s_logistic_100nodes_1000samples_0001_rep89'

    #'2017_12_23_19h15m28s_logistic_10nodes_10000samples',
    #'2018_01_09_17h09m42s_logistic_40nodes_10000samples_rep01234',
    #'2018_01_20_10h10m16s_logistic_40nodes_10000samples_rep56789',
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
    #'2018_03_21_10h27m54s_logistic_100nodes_10000samples_rep9'
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
    'mem_requested': re.compile(r'Mem requested:\s+(\d+.\d+)GB'),
    'mem_used_GB': re.compile(r'Mem used:\s+(\d.+)GB'),
    'mem_used_MB': re.compile(r'Mem used:\s+(\d.+)MB'),
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

    df['mem_used_GB'].values[np.isnan(df['mem_used_GB'])] = df[
        'mem_used_MB'].values[np.isnan(df['mem_used_GB'])] / 1000

    aggregate_repetitions = df.groupby('name').agg(lambda x: x.tolist())

    for index, row in aggregate_repetitions.iterrows():
        print(row.name)
        print('Estimated network size: {0}'.format(len(row['array_id'])))
        runtime = np.array(row['runtime'])
        #print(runtime.mean())
        #print(runtime.std())
        print('Max runtime: {0:.3f} hours = {1:.2f} minutes = {2:.2f} seconds'.format(
            runtime.max()/3600,
            runtime.max()/60,
            runtime.max()
            ))
        mem_used_GB = np.array(row['mem_used_GB'])
        #print(mem_used_GB.mean())
        #print(mem_used_GB.std())
        print('Max memory: {0} GB \n'.format(mem_used_GB.max()))
