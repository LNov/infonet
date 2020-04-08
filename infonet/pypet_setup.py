import os
import pickle
import numpy as np


# Use pickle module to save dictionaries
def save_obj(obj, file_dir, file_name):
    with open(os.path.join(file_dir, file_name), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


settings = {}

# -------------------------------------------------------------------
# Parameters characterizing the initial topology of the network
# -------------------------------------------------------------------
# settings['topology.initial.model'] = 'ER_n_in'
# settings['topology.initial.model'] = 'BA'
# settings['topology.initial.model'] = 'WS'
# settings['topology.initial.model'] = 'planted_partition'
settings['topology.initial.model'] = 'planted_partition_fixed_links_n'
# traj.parameters.f_get('topology.initial.model').v_comment = topology_models[
#   'Description'].get(traj.parameters['topology.initial.model'])
settings['topology.initial.nodes_n'] = 100  # np.arange(100, 100+1, 300).tolist()
# settings['topology.initial.in_degree_expected'] = 3
# settings['topology.initial.WS_k'] = 4
# settings['topology.initial.WS_p'] = np.around(
# np.logspace(-2.2, 0, 10), decimals=4).tolist()
# settings['topology.initial.BA_m'] = 1
settings['topology.initial.partitions_n'] = 5
# settings['topology.initial.p_in'] = 0.5
# settings['topology.initial.p_out'] = 0.01
settings['topology.initial.links_total'] = 10
settings['topology.initial.links_out'] = np.arange(
    0, settings['topology.initial.links_total'] + 1, 2).tolist()

# -------------------------------------------------------------------
# Parameters characterizing the evolution of the topology
# -------------------------------------------------------------------
settings['topology.evolution.model'] = 'static'

# -------------------------------------------------------------------
# Parameters characterizing the coupling between the nodes
# -------------------------------------------------------------------
settings['node_coupling.initial.model'] = 'linear'
settings['node_coupling.initial.weight_distribution'] = 'fixed'
settings['node_coupling.initial.fixed_coupling'] = 0.05

# -------------------------------------------------------------------
# Parameters characterizing the delay
# -------------------------------------------------------------------
settings['delay.initial.distribution'] = 'uniform'
settings['delay.initial.delay_links_n_max'] = 1
settings['delay.initial.delay_min'] = 1
settings['delay.initial.delay_max'] = 1
settings['delay.initial.delay_self'] = 1

# -------------------------------------------------------------------
# Parameters characterizing the dynamics of the nodes
# -------------------------------------------------------------------
# settings['node_dynamics.model'] = 'logistic_map'
settings['node_dynamics.model'] = 'AR_gaussian_discrete'
# settings['node_dynamics.model'] = 'boolean_random'
settings['node_dynamics.samples_n'] = np.array([10000]).tolist()
settings['node_dynamics.samples_transient_n'] = 100000
settings['node_dynamics.replications'] = 1
settings['node_dynamics.noise_std'] = 0.1
# settings['node_dynamics.RBN_in_degree'] = 4
# settings['node_dynamics.noise_flip_p'] = 0.005

# -------------------------------------------------------------------
# Parameters characterizing the estimator
# -------------------------------------------------------------------
# settings['estimation.history_source'] = 1
# settings['estimation.history_target'] = 14

# -------------------------------------------------------------------
# Parameters characterizing the network inference algorithm
# -------------------------------------------------------------------
settings['network_inference.algorithm'] = [
    'bMI_greedy',
    'bTE_greedy',
    'mTE_greedy']
settings['network_inference.min_lag_sources'] = 1
settings['network_inference.max_lag_sources'] = 1
settings['network_inference.tau_sources'] = 1
settings['network_inference.max_lag_target'] = 1
settings['network_inference.tau_target'] = 1
# settings['network_inference.cmi_estimator'] = 'JidtDiscreteCMI'
settings['network_inference.cmi_estimator'] = 'JidtGaussianCMI'
# settings['network_inference.cmi_estimator'] = 'JidtKraskovCMI'
# settings['network_inference.cmi_estimator'] = 'OpenCLKraskovCMI'
settings['network_inference.permute_in_time'] = False
settings['network_inference.jidt_threads_n'] = 1
settings['network_inference.n_perm_max_stat'] = 10000
settings['network_inference.n_perm_min_stat'] = 10000
settings['network_inference.n_perm_omnibus'] = 10000
settings['network_inference.n_perm_max_seq'] = 10000
settings['network_inference.fdr_correction'] = False
settings['network_inference.z_standardise'] = True
settings['network_inference.kraskov_k'] = 4
settings['network_inference.p_value'] = 0.001
# settings['network_inference.alpha_max_stats'] = 0.05
# settings['network_inference.alpha_min_stats'] = 0.05
# settings['network_inference.alpha_omnibus'] = 0.05
# settings['network_inference.alpha_max_seq'] = 0.05
# settings['network_inference.alpha_fdr'] = 0.05

# -------------------------------------------------------------------
# Parameters characterizing the repetitions of the same run
# -------------------------------------------------------------------
settings['repetition_i'] = (
    np.arange(0, 2, step=1).tolist())  # Normally starts from 0

# ------------------------------------------------------------------

# Print settings dictionary
print('\nSettings dictionary:')
for key, value in settings.items():
    print(key, ' : ', value)
print('\nParameters to explore:')
for key, value in settings.items():
        if isinstance(value, list):
            print(key, ' : ', value)

# Save settings dictionary
save_obj(settings, os.getcwd(), 'pypet_settings.pkl')
print('\nSettings dictionary saved to {0}'.format(os.getcwd()))
