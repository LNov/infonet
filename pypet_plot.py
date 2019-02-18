import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
from cycler import cycler
from mylib_pypet import print_leaves
import collections


def save_figures_to_pdf(figures, path, pdf_metadata={}):
    """
    Create a PDF file with several pages.
    Also add PDF file metadata if provided.
    Some possible metadata keys: ['Title', 'Author', 'Subject', 'Keywords',
        'CreationDate', 'ModDate']
    """

    # Create the PdfPages object to which we will save the pages:
    # The with statement makes sure that the PdfPages object is closed
    # properly at the end of the block, even if an Exception occurs.
    with PdfPages(path) as pdf:
        for fig_i in range(len(fig_list)):

            # Get figure and subplots
            fig = fig_list[fig_i]
            axs = axs_list[fig_i]

            # Set figure as current figure
            plt.figure(fig.number)

            # Set figure height
            ndim = axs.ndim
            if ndim == 1:
                fig.set_figheight(3 * len(axs))
            if ndim > 1:
                fig.set_figheight(3 * len(axs[:, 0]))
                fig.set_figwidth(4 * len(axs[0, :]))

            # Set tight layout to avoid overlap between subplots
            fig.tight_layout()

            # Add a pdf note to attach metadata to a page
            # pdf.attach_note('note...')

            # Save figure to PDF page
            pdf.savefig(fig)

            # Also export as EPS
            eps_path = os.path.join(traj_dir, '{0}.pdf'.format(fig.number))
            plt.savefig(eps_path, format='pdf')

            print('figure {0} saved'.format(fig_i))

        # Set PDF file metadata via the PdfPages object
        d = pdf.infodict()
        for key in pdf_metadata.keys():
            d[key] = pdf_metadata.get(key, '')


# Choose whether to use FDR-corrected results or not
fdr = False
# Select alpha threshold to plot (in those plots where only one value is used)
alpha_interest = 0.05

# Choose folder
traj_dir = os.path.join('trajectories', 'KSG_on_CLM_10000samples_alpha05')
if not os.path.isdir(traj_dir):
    traj_dir = os.path.join('..', traj_dir)

# Set up plot style
# Use "matplotlib.rcdefaults()" to restore the default plot style"
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['lines.markersize'] = 3
mpl.rcParams['errorbar.capsize'] = 3

# Colours
cycler_default = cycler(
    'color',
    ['#1f77b4',
     '#ff7f0e',
     '#2ca02c',
     '#d62728',
     '#9467bd',
     '#8c564b',
     '#e377c2',
     '#7f7f7f',
     '#bcbd22',
     '#17becf'])

# Load DataFrame
if fdr:
    df = pd.read_pickle(os.path.join(traj_dir, 'postprocessing_fdr.pkl'))
else:
    df = pd.read_pickle(os.path.join(traj_dir, 'postprocessing.pkl'))

#df = df.drop('permute_in_time', 1)

# Determine explore parameters (by construction, they are the name
# of the columns that precede 'precision')
# WARNING: this method is prone to errors, when the columns are rearranged,
# however, in such cases the "assertion" sanity checks below will fail, so I
# will know
parameters_explored = df.loc[[], :'precision'].keys().tolist()[:-1]

#
nodes_n_range = np.unique(df['nodes_n']).astype(int)
samples_n_range = np.unique(df['samples_n']).astype(int)
alpha_c_range = np.unique(df['p_value']).astype(float)

#
fig_list = []
axs_list = []

# # Plot performance tests vs. number of samples
# # Subplots: network size
# fig, axs = plt.subplots(len(nodes_n_range), 1, sharey=True)
# #fig.suptitle(r'$Performance\ tests\ (subplots:\ n.\ nodes)$')
# # Select data of interest
# df_interest = df.loc[df['p_value'] == 0.05]
# # Drop data of interest column
# df_interest = df_interest.drop('p_value', 1)
# # Drop non relevant columns
# df_interest = df_interest.drop('TE_omnibus_empirical', 1)
# df_interest = df_interest.drop('TE_omnibus_theoretical_inferred_vars', 1)
# df_interest = df_interest.drop('TE_omnibus_theoretical_causal_vars', 1)
# # Convert remaining DataFrame to float type for averaging
# df_interest = df_interest.astype(float)
# # Choose which of the explored parameters to average over when computing the
# # error bars
# parameters_to_average = {'repetition_i'}
# # If axs is not a list (i.e. it only contains one subplot), turn it into a
# # list with one element
# if not hasattr(axs, "__len__"):
#     axs = np.array([axs])
# axs_i = 0
# for nodes_n in nodes_n_range:
#     # Set subplot title
#     axs[axs_i].set_title(r'$network\ size\ =\ ${}'.format(nodes_n))
#     # Select dataframe entries(i.e runs) with the same number of nodes
#     df_nodes = df_interest.loc[df_interest['nodes_n'] == nodes_n].drop(
#         'nodes_n', 1)
#     # Group by number of samples and then compute mean and std
#     aggregate_repetitions = df_nodes.groupby('samples_n').agg(['mean', 'std'])
#     # Ensure that the desired parameters are averaged over when computing the
#     # error bars
#     df_keys_remaining = aggregate_repetitions.columns.get_level_values(0)
#     assert set.intersection(
#         set(df_keys_remaining),
#         set(parameters_explored)) == parameters_to_average
#     # Plot precision
#     axs[axs_i].errorbar(
#         aggregate_repetitions.index,
#         aggregate_repetitions['precision']['mean'],
#         # Add vertical (symmetric) error bars
#         yerr=aggregate_repetitions['precision']['std'],
#         fmt='-o'
#     )
#     # Plot recall
#     axs[axs_i].errorbar(
#         aggregate_repetitions.index,
#         aggregate_repetitions['recall']['mean'],
#         # Add vertical (symmetric) error bars
#         yerr=aggregate_repetitions['recall']['std'],
#         fmt='-o'
#     )
#     # Plot false positive rate
#     axs[axs_i].errorbar(
#         aggregate_repetitions.index,
#         aggregate_repetitions['specificity']['mean'],
#         # Add vertical (symmetric) error bars
#         yerr=aggregate_repetitions['specificity']['std'],
#         fmt='o-'
#     )
#     # Set axes properties
#     axs[axs_i].set_xlabel(r'$T$', horizontalalignment='right', x=1.0)
#     axs[axs_i].set_ylabel(r'$Performance$')  # , horizontalalignment='right', y=1.0)
#     axs[axs_i].set_xscale('log')
#     axs[axs_i].yaxis.set_ticks_position('both')
#     axs[axs_i].set_xlim([100, 10000])
#     axs[axs_i].set_ylim(bottom=0)
#     # Add legend
# #    axs[axs_i].legend(labels=['Precision', 'Recall', 'Specificity'])
# 
#     axs_i += 1
# fig_list.append(fig)
# axs_list.append(axs)

# # Plot performance tests vs. network size (max-min error bars)
# # Subplots: number of samples
# fig, axs = plt.subplots(len(samples_n_range), 1, sharey=True)
# # fig.suptitle(r'$Performance\ tests\ (subplots:\ n.\ samples)$')
# # Select data of interest
# df_interest = df[parameters_explored + ['precision', 'recall', 'specificity']]
# df_interest = df_interest.loc[df_interest['p_value'] == alpha_interest].drop('p_value', 1)
# # Convert remaining DataFrame to float type for averaging
# df_interest = df_interest.astype(float)
# # Choose which of the explored parameters to average over when computing the
# # error bars
# parameters_to_average = {'repetition_i'}
# # If axs is not a list (i.e. it only contains one subplot), turn it into a
# # list with one element
# if not hasattr(axs, "__len__"):
#     axs = np.array([axs])
# axs_i = 0
# for samples_n in samples_n_range:
#     # Set subplot title
#     #axs[axs_i].set_title(r'$T=${}'.format(samples_n))
#     # Select dataframe entries(i.e runs) with the same number of samples
#     df_samples = df_interest.loc[df_interest['samples_n'] == samples_n].drop(
#         'samples_n', 1)
#     # Drop NaN values
#     df_samples = df_samples.dropna()
#     # Group by number of nodes and then compute mean and extrema
#     aggregate_repetitions = df_samples.groupby('nodes_n').agg(
#         ['mean', 'max', 'min'])
#     # Ensure that the desired parameters are averaged over when computing the
#     # error bars
#     df_keys_remaining = aggregate_repetitions.columns.get_level_values(0)
#     assert set.intersection(
#         set(df_keys_remaining),
#         set(parameters_explored)) == parameters_to_average
#     # Plot precision
#     axs[axs_i].errorbar(
#         aggregate_repetitions.index,
#         aggregate_repetitions['precision']['mean'],
#         # Add vertical (asymmetric) error bars
#         yerr=[
#             (aggregate_repetitions['precision']['mean']
#              - aggregate_repetitions['precision']['min']),
#             (aggregate_repetitions['precision']['max']
#              - aggregate_repetitions['precision']['mean'])
#             ],
#         fmt='-o'
#     )
#     # Plot recall
#     axs[axs_i].errorbar(
#         aggregate_repetitions.index,
#         aggregate_repetitions['recall']['mean'],
#         # Add vertical (symmetric) error bars
#         yerr=[
#             (aggregate_repetitions['recall']['mean']
#              - aggregate_repetitions['recall']['min']),
#             (aggregate_repetitions['recall']['max']
#              - aggregate_repetitions['recall']['mean'])
#             ],
#         fmt='-o'
#     )
#     # Plot false positive rate
#     axs[axs_i].errorbar(
#         aggregate_repetitions.index,
#         aggregate_repetitions['specificity']['mean'],
#         # Add vertical (symmetric) error bars
#         yerr=[
#             (aggregate_repetitions['specificity']['mean']
#              - aggregate_repetitions['specificity']['min']),
#             (aggregate_repetitions['specificity']['max']
#              - aggregate_repetitions['specificity']['mean'])
#             ],
#         fmt='o-'
#     )
#     # Set axes properties
#     axs[axs_i].set_xlabel(r'$Network\ size$', horizontalalignment='right', x=1.0)
#     axs[axs_i].set_ylabel(r'$T={}$'.format(samples_n))#, horizontalalignment='right', y=1.0)
#     #axs[axs_i].set_xscale('log')
#     axs[axs_i].yaxis.set_ticks_position('both')
#     axs[axs_i].set_ylim(bottom=0)
#     # Add legend
#     axs[axs_i].legend(labels=['Precision', 'Recall', 'Specificity'])
# 
#     axs_i += 1
# fig_list.append(fig)
# axs_list.append(axs)

# # Plot performance tests vs. network size (box-whiskers error bars)
# # Subplots: number of samples
# fig, axs = plt.subplots(len(samples_n_range), 1, sharey=True)
# # fig.suptitle(r'$Performance\ tests\ (subplots:\ n.\ samples)$')
# # Select data of interest
# df_interest = df[parameters_explored + ['precision', 'recall', 'specificity']]
# df_interest = df_interest.loc[df_interest['p_value'] == alpha_interest].drop('p_value', 1)
# # Convert remaining DataFrame to float type for averaging
# df_interest = df_interest.astype(float)
# # Choose which of the explored parameters to average over when computing the
# # error bars
# parameters_to_average = {'repetition_i'}
# # If axs is not a list (i.e. it only contains one subplot), turn it into a
# # list with one element
# if not hasattr(axs, "__len__"):
#     axs = np.array([axs])
# axs_i = 0
# for samples_n in samples_n_range:
#     # Set subplot title
#     #axs[axs_i].set_title(r'$T={}$'.format(samples_n))
#     # Select dataframe entries(i.e runs) with the same number of samples
#     df_samples = df_interest.loc[df_interest['samples_n'] == samples_n].drop(
#         'samples_n', 1)
#     # Drop NaN values
#     df_samples = df_samples.dropna()
#     # Group by number of nodes and then compute mean and extrema
#     aggregate_repetitions = df_samples.groupby('nodes_n').agg(
#         lambda x: list(x))
#     # Ensure that the desired parameters are averaged over when computing the
#     # error bars
#     df_keys_remaining = aggregate_repetitions.columns.get_level_values(0)
#     assert set.intersection(
#         set(df_keys_remaining),
#         set(parameters_explored)) == parameters_to_average
#     # Plot precision
#     axs[axs_i].boxplot(
#         [aggregate_repetitions['precision'][nodes_n]
#          for nodes_n in nodes_n_range],
#         positions=aggregate_repetitions.index.astype(int).tolist()
#     )
#     # Plot recall
#     axs[axs_i].boxplot(
#         [aggregate_repetitions['recall'][nodes_n]
#          for nodes_n in nodes_n_range],
#         positions=aggregate_repetitions.index.astype(int).tolist()
#     )
#     # Plot false positive rate
#     axs[axs_i].boxplot(
#         [aggregate_repetitions['specificity'][nodes_n]
#          for nodes_n in nodes_n_range],
#         positions=aggregate_repetitions.index.astype(int).tolist()
#     )
#     # Set axes properties
#     axs[axs_i].set_xlabel(r'$Network\ size$', horizontalalignment='right', x=1.0)
#     axs[axs_i].set_ylabel('$T={}$'.format(samples_n))#, horizontalalignment='right', y=1.0)
#     #axs[axs_i].set_xscale('log')
#     axs[axs_i].yaxis.set_ticks_position('both')
#     axs[axs_i].set_ylim(bottom=0)
#     # Add legend
#     axs[axs_i].legend(labels=['Precision', 'Recall', 'Specificity'])
# 
#     axs_i += 1
# fig_list.append(fig)
# axs_list.append(axs)

# Plot performance tests vs. network size (scatter error)
# Subplots: number of samples
fig, axs = plt.subplots(len(samples_n_range), 1, sharey=True)
# fig.suptitle(r'$Performance\ tests\ (subplots:\ n.\ samples)$')
# Select data of interest
df_interest = df[parameters_explored + ['precision', 'recall', 'specificity']]
df_interest = df_interest.loc[df_interest['p_value'] == alpha_interest].drop('p_value', 1)
# Convert remaining DataFrame to float type for averaging
df_interest = df_interest.astype(float)
# Choose which of the explored parameters to average over when computing the
# error bars
parameters_to_average = {'repetition_i'}
# If axs is not a list (i.e. it only contains one subplot), turn it into a
# list with one element
if not hasattr(axs, "__len__"):
    axs = np.array([axs])
axs_i = 0
for samples_n in samples_n_range:
    # Set subplot title
    #axs[axs_i].set_title(r'$T={}$'.format(samples_n))
    # Select dataframe entries(i.e runs) with the same number of samples
    df_samples = df_interest.loc[df_interest['samples_n'] == samples_n].drop(
        'samples_n', 1)
    # Drop NaN values
    df_samples = df_samples.dropna()
    # Group by number of nodes and then compute mean and extrema
    aggregate_repetitions = df_samples.groupby('nodes_n').agg(
        lambda x: list(x))
    # Ensure that the desired parameters are averaged over when computing the
    # error bars
    df_keys_remaining = aggregate_repetitions.columns.get_level_values(0)
    assert set.intersection(
        set(df_keys_remaining),
        set(parameters_explored)) == parameters_to_average
    # Plot precision
    axs[axs_i].plot(
        nodes_n_range,
        [np.mean(aggregate_repetitions['precision'][nodes_n])
            for nodes_n in nodes_n_range],
        marker='o'
    )
    axs[axs_i].scatter(
        df_samples['nodes_n'].values.astype(int).tolist(),
        df_samples['precision'].values,
        alpha=0.3
    )
    # Plot recall
    axs[axs_i].plot(
        nodes_n_range,
        [np.mean(aggregate_repetitions['recall'][nodes_n])
         for nodes_n in nodes_n_range],
        marker='o'
    )
    axs[axs_i].scatter(
        df_samples['nodes_n'].values.astype(int).tolist(),
        df_samples['recall'].values,
        alpha=0.3
    )
    # Plot false positive rate
    axs[axs_i].plot(
        nodes_n_range,
        [np.mean(aggregate_repetitions['specificity'][nodes_n])
         for nodes_n in nodes_n_range],
        marker='o'
    )
    axs[axs_i].scatter(
        df_samples['nodes_n'].values.astype(int).tolist(),
        df_samples['specificity'].values,
        alpha=0.3
    )
    # Set axes properties
    axs[axs_i].set_xlabel(r'$Network\ size$', horizontalalignment='right', x=1.0)
    axs[axs_i].set_ylabel(r'$T={}$'.format(samples_n))#, horizontalalignment='right', y=1.0)
    #axs[axs_i].set_xscale('log')
    axs[axs_i].yaxis.set_ticks_position('both')
    axs[axs_i].set_ylim(bottom=0)
    # Add legend
    axs[axs_i].legend(labels=['Precision', 'Recall', 'Specificity'], loc=(0.05, 0.16))

    axs_i += 1
fig_list.append(fig)
axs_list.append(axs)

# Plot precision vs. network size (scatter error)
# Subplots: none
# Legend: samples
# Set color pallette and/or line styles
mpl.rcParams['axes.prop_cycle'] = cycler('ls', ['-', '--', ':'])
fig = plt.figure()
axs = plt.gca()
axs = np.array([axs])
# Select data of interest
df_interest = df[parameters_explored + ['precision']]
df_interest = df_interest.loc[df_interest['p_value'] == alpha_interest].drop('p_value', 1)
# Convert remaining DataFrame to float type for averaging
df_interest = df_interest.astype(float)
# Choose which of the explored parameters to average over when computing the
# error bars
parameters_to_average = {'repetition_i'}
for samples_n in samples_n_range:
    # Select dataframe entries(i.e runs) with the same number of samples
    df_samples = df_interest.loc[df_interest['samples_n'] == samples_n].drop(
        'samples_n', 1)
    # Drop NaN values
    df_samples = df_samples.dropna()
    # Group by number of nodes and then compute mean and extrema
    aggregate_repetitions = df_samples.groupby('nodes_n').agg(
        lambda x: list(x))
    # Ensure that the desired parameters are averaged over when computing the
    # error bars
    df_keys_remaining = aggregate_repetitions.columns.get_level_values(0)
    assert set.intersection(
        set(df_keys_remaining),
        set(parameters_explored)) == parameters_to_average
    # Plot precision
    axs[0].plot(
        aggregate_repetitions.index.astype(int).tolist(),
        [np.mean(aggregate_repetitions['precision'][nodes_n])
         for nodes_n in nodes_n_range],
        marker='o',
        color='tab:blue'
    )
    axs[0].scatter(
        df_samples['nodes_n'].values.astype(int).tolist(),
        df_samples['precision'].values,
        alpha=0.3,
        c='tab:blue'
    )
# Set axes properties
axs[0].set_xlabel(r'$Network\ size$', horizontalalignment='right', x=1.0)
axs[0].set_ylabel(r'$Precision$')#, horizontalalignment='right', y=1.0)
#axs[0].set_xscale('log')
axs[0].yaxis.set_ticks_position('both')
axs[0].set_ylim(bottom=0)
# Add legend
legend = [
    r'$T={}$'.format(samples_n) for samples_n in samples_n_range]
axs[0].legend(labels=legend, loc=(0.05, 0.16)) #remove loc for AR
# Reset color pallette and/or line styles to default
mpl.rcParams['axes.prop_cycle'] = cycler_default
fig_list.append(fig)
axs_list.append(axs)

# Plot recall vs. network size (scatter error)
# Subplots: none
# Legend: samples
# Set color pallette and/or line styles
mpl.rcParams['axes.prop_cycle'] = cycler('ls', ['-', '--', ':'])
fig = plt.figure()
axs = plt.gca()
axs = np.array([axs])
# Select data of interest
df_interest = df[parameters_explored + ['recall']]
df_interest = df_interest.loc[df_interest['p_value'] == alpha_interest].drop('p_value', 1)
# Convert remaining DataFrame to float type for averaging
df_interest = df_interest.astype(float)
# Choose which of the explored parameters to average over when computing the
# error bars
parameters_to_average = {'repetition_i'}
for samples_n in samples_n_range:
    # Select dataframe entries(i.e runs) with the same number of samples
    df_samples = df_interest.loc[df_interest['samples_n'] == samples_n].drop(
        'samples_n', 1)
    # Drop NaN values
    df_samples = df_samples.dropna()
    # Group by number of nodes and then compute mean and extrema
    aggregate_repetitions = df_samples.groupby('nodes_n').agg(
        lambda x: list(x))
    # Ensure that the desired parameters are averaged over when computing the
    # error bars
    df_keys_remaining = aggregate_repetitions.columns.get_level_values(0)
    assert set.intersection(
        set(df_keys_remaining),
        set(parameters_explored)) == parameters_to_average
    # Plot recall
    axs[0].plot(
        aggregate_repetitions.index.astype(int).tolist(),
        [np.mean(aggregate_repetitions['recall'][nodes_n])
         for nodes_n in nodes_n_range],
        marker='o',
        color='tab:orange'
    )
    axs[0].scatter(
        df_samples['nodes_n'].values.astype(int).tolist(),
        df_samples['recall'].values,
        alpha=0.3,
        c='tab:orange'
    )
# Set axes properties
axs[0].set_xlabel(r'$Network\ size$', horizontalalignment='right', x=1.0)
axs[0].set_ylabel(r'$Recall$')#, horizontalalignment='right', y=1.0)
#axs[0].set_xscale('log')
axs[0].yaxis.set_ticks_position('both')
axs[0].set_ylim(bottom=0)
# Add legend
legend = [
    r'$T={}$'.format(samples_n) for samples_n in samples_n_range]
axs[0].legend(labels=legend, loc=(0.05, 0.16))# AR: use loc=(0.75, 0.56))
# Reset color pallette and/or line styles to default
mpl.rcParams['axes.prop_cycle'] = cycler_default
fig_list.append(fig)
axs_list.append(axs)

# Plot spectral radius vs. network size (scatter error)
# Subplots: none
# Legend: none
fig = plt.figure()
axs = plt.gca()
axs = np.array([axs])
# Select data of interest
df_interest = df[['nodes_n', 'repetition_i', 'spectral_radius']]
# Convert remaining DataFrame to float type for averaging
df_interest = df_interest.astype(float)
# Choose which of the explored parameters to average over when computing the
# error bars
parameters_to_average = {'repetition_i'}
# Delete duplicate rows
df_interest = df_interest.drop_duplicates(
    ['nodes_n', 'repetition_i'])
# Group by number of nodes and then compute mean and extrema
aggregate_repetitions = df_interest.groupby('nodes_n').agg(
    lambda x: list(x))
# Ensure that the desired parameters are averaged over when computing the
# error bars
df_keys_remaining = aggregate_repetitions.columns.get_level_values(0)
assert set.intersection(
    set(df_keys_remaining),
    set(parameters_explored)) == parameters_to_average
# Plot recall
axs[0].plot(
    aggregate_repetitions.index.astype(int).tolist(),
    [np.mean(aggregate_repetitions['spectral_radius'][nodes_n])
        for nodes_n in nodes_n_range],
    marker='o',
    color='k'
)
axs[0].scatter(
    df_interest['nodes_n'].values.astype(int).tolist(),
    df_interest['spectral_radius'].values,
    alpha=0.3,
    c='k'
)
# Set axes properties
axs[0].set_xlabel(r'$Network\ size$', horizontalalignment='right', x=1.0)
axs[0].set_ylabel(r'$\rho$')#, horizontalalignment='right', y=1.0)
#axs[0].set_xscale('log')
axs[0].yaxis.set_ticks_position('both')
#axs[0].set_ylim(bottom=0)
fig_list.append(fig)
axs_list.append(axs)

# # Plot false positive rate vs. critical alpha
# # Legend: number of samples
# # Subplots: number of nodes
# fig, axs = plt.subplots(len(nodes_n_range), 1, sharey=True)
# # fig.suptitle(r'$False\ positive\ rate\ vs.\ \alpha$')
# # Select data of interest
# df_interest = df
# # Drop non relevant columns
# df_interest = df_interest.drop('TE_omnibus_empirical', 1)
# df_interest = df_interest.drop('TE_omnibus_theoretical_inferred_vars', 1)
# df_interest = df_interest.drop('TE_omnibus_theoretical_causal_vars', 1)
# # Convert remaining DataFrame to float type for averaging
# df_interest = df_interest.astype(float)
# # Choose which of the explored parameters to average over when computing the
# # error bars
# parameters_to_average = {'repetition_i'}
# # If axs is not a list (i.e. it only contains one subplot), turn it into a
# # list with one element
# if not hasattr(axs, "__len__"):
#     axs = np.array([axs])
# axs_i = 0
# for nodes_n in nodes_n_range:
#     # Set subplot title
#     axs[axs_i].set_title(r'$N={}$'.format(nodes_n))
#     # Select dataframe entries(i.e runs) with the same number of nodes
#     df_nodes = df_interest.loc[df_interest['nodes_n'] == nodes_n].drop('nodes_n', 1)
#     #
#     for samples_n in samples_n_range:
#         # Select dataframe entries(i.e runs) with the same number of samples
#         df_samples = df_nodes.loc[df_nodes['samples_n'] == samples_n].drop(
#             'samples_n', 1)
#         # Group by p_value and then compute mean and std
#         aggregate_repetitions = df_samples.groupby('p_value').agg(
#             ['mean', 'std'])
#         # Ensure that the desired parameters are averaged over when computing
#         # the error bars
#         df_keys_remaining = aggregate_repetitions.columns.get_level_values(0)
#         assert set.intersection(
#             set(df_keys_remaining),
#             set(parameters_explored)) == parameters_to_average
#         # Plot false positive rate
#         axs[axs_i].errorbar(
#             aggregate_repetitions.index[
#                 aggregate_repetitions['false_pos_rate']['mean'] > 0],
#             aggregate_repetitions['false_pos_rate']['mean'][
#                 aggregate_repetitions['false_pos_rate']['mean'] > 0],
#             # Add vertical error bars
#             yerr=[
#                 0*aggregate_repetitions['false_pos_rate']['std'][
#                     aggregate_repetitions['false_pos_rate']['mean'] > 0],
#                 aggregate_repetitions['false_pos_rate']['std'][
#                     aggregate_repetitions['false_pos_rate']['mean'] > 0]
#                 ],
#             fmt='o-'
#         )
#     axs[axs_i].plot(
#         aggregate_repetitions.index,
#         aggregate_repetitions.index,
#         '--',
#         color='tab:gray'
#     )
#     # Set axes properties
#     axs[axs_i].set_xlabel(r'$\alpha$', horizontalalignment='right', x=1.0)
#     axs[axs_i].set_ylabel(r'$False\ positive\ rate$')#, horizontalalignment='right', y=1.0)
#     axs[axs_i].set_xscale('log')
#     axs[axs_i].set_yscale('log')
#     axs[axs_i].yaxis.set_ticks_position('both')
#     # axs[axs_i].set_ylim([0, 1])
#     # Add legend
#     legend = ['Identity'] + list(samples_n_range.astype(int))
#     #axs[axs_i].legend(labels=legend)
# 
#     axs_i += 1
# fig_list.append(fig)
# axs_list.append(axs)
# 
# # Plot false positive rate vs. critical alpha
# # Legend: number of nodes
# # Subplots: number of samples
# fig, axs = plt.subplots(len(samples_n_range), 1, sharey=True)
# # fig.suptitle(r'$False\ positive\ rate\ vs.\ \alpha$')
# # Select data of interest
# df_interest = df
# # Drop non relevant columns
# df_interest = df_interest.drop('TE_omnibus_empirical', 1)
# df_interest = df_interest.drop('TE_omnibus_theoretical_inferred_vars', 1)
# df_interest = df_interest.drop('TE_omnibus_theoretical_causal_vars', 1)
# # Convert remaining DataFrame to float type for averaging
# df_interest = df_interest.astype(float)
# # Choose which of the explored parameters to average over when computing the
# # error bars
# parameters_to_average = {'repetition_i'}
# # If axs is not a list (i.e. it only contains one subplot), turn it into a
# # list with one element
# if not hasattr(axs, "__len__"):
#     axs = np.array([axs])
# axs_i = 0
# for samples_n in samples_n_range:
#     # Set subplot title
#     axs[axs_i].set_title(r'$T={}$'.format(samples_n))
#     # Select dataframe entries(i.e runs) with the same number of samples
#     df_samples = df_interest.loc[df_interest['samples_n'] == samples_n].drop('samples_n', 1)
#     #
#     for nodes_n in nodes_n_range:
#         # Select dataframe entries(i.e runs) with the same number of nodes
#         df_nodes = df_samples.loc[df_samples['nodes_n'] == nodes_n].drop(
#             'nodes_n', 1)
#         # Group by p_value and then compute mean and std
#         aggregate_repetitions = df_nodes.groupby('p_value').agg(
#             ['mean', 'std'])
#         # Ensure that the desired parameters are averaged over when computing
#         # the error bars
#         df_keys_remaining = aggregate_repetitions.columns.get_level_values(0)
#         assert set.intersection(
#             set(df_keys_remaining),
#             set(parameters_explored)) == parameters_to_average
#         # Plot false positive rate
#         axs[axs_i].errorbar(
#             aggregate_repetitions.index[
#                 aggregate_repetitions['false_pos_rate']['mean'] > 0],
#             aggregate_repetitions['false_pos_rate']['mean'][
#                 aggregate_repetitions['false_pos_rate']['mean'] > 0],
#             # Add vertical (symmetric) error bars
#             yerr=[
#                 0*aggregate_repetitions['false_pos_rate']['std'][
#                     aggregate_repetitions['false_pos_rate']['mean'] > 0],
#                 aggregate_repetitions['false_pos_rate']['std'][
#                     aggregate_repetitions['false_pos_rate']['mean'] > 0]
#                 ],
#             fmt='o-'
#         )
#     axs[axs_i].plot(
#         aggregate_repetitions.index,
#         aggregate_repetitions.index,
#         '--',
#         color='tab:gray'
#     )
#     # Set axes properties
#     axs[axs_i].set_xlabel(r'$\alpha$', horizontalalignment='right', x=1.0)
#     axs[axs_i].set_ylabel(r'$False\ positive\ rate$')#, horizontalalignment='right', y=1.0)
#     axs[axs_i].set_xscale('log')
#     axs[axs_i].set_yscale('log')
#     axs[axs_i].yaxis.set_ticks_position('both')
#     # axs[axs_i].set_ylim([0, 1])
#     # Add legend
#     legend = ['Identity'] + list(nodes_n_range.astype(int))
#     #axs[axs_i].legend(labels=legend)
# 
#     axs_i += 1
# fig_list.append(fig)
# axs_list.append(axs)

# Plot incorrect targets rate vs. critical alpha
# Legend: number of nodes
# Subplots: number of samples
fig, axs = plt.subplots(1, 1, sharey=True) #len(samples_n_range)
# Select data of interest
df_interest = df[parameters_explored + ['incorrect_target_rate']]
# Convert remaining DataFrame to float type for averaging
df_interest = df_interest.astype(float)
# Choose which of the explored parameters to average over when computing the
# error bars
parameters_to_average = {'repetition_i'}
# If axs is not a list (i.e. it only contains one subplot), turn it into a
# list with one element
if not hasattr(axs, "__len__"):
    axs = np.array([axs])
axs_i = 0
for samples_n in samples_n_range[-1:]:
    # Set subplot title
    #axs[axs_i].set_title('$T={}$'.format(samples_n.astype(int)))
    # Select dataframe entries(i.e runs) with the same number of samples
    df_samples = df_interest.loc[df_interest['samples_n'] == samples_n].drop('samples_n', 1)
    # Drop NaN values
    df_samples = df_samples.dropna()
    for nodes_n in nodes_n_range:
        # Select dataframe entries(i.e runs) with the same number of nodes
        df_nodes = df_samples.loc[df_samples['nodes_n'] == nodes_n].drop(
            'nodes_n', 1)
        # Group by p_value and then compute mean and std
        aggregate_repetitions = df_nodes.groupby('p_value').agg(
            ['mean', 'std', 'max', 'min'])
        # Ensure that the desired parameters are averaged over when computing
        # the error bars
        df_keys_remaining = aggregate_repetitions.columns.get_level_values(0)
        assert set.intersection(
            set(df_keys_remaining),
            set(parameters_explored)) == parameters_to_average
        # Plot false positive rate
        axs[axs_i].errorbar(
            aggregate_repetitions[
                    aggregate_repetitions['incorrect_target_rate']['mean'] > 0].index,
            aggregate_repetitions['incorrect_target_rate']['mean'][
                    aggregate_repetitions['incorrect_target_rate']['mean'] > 0],
            yerr=[
                0*aggregate_repetitions['incorrect_target_rate']['std'][
                    aggregate_repetitions['incorrect_target_rate']['mean'] > 0],
                aggregate_repetitions['incorrect_target_rate']['std'][
                    aggregate_repetitions['incorrect_target_rate']['mean'] > 0]
                ],
            fmt='o-'
        )
    axs[axs_i].plot(
        aggregate_repetitions.index,
        aggregate_repetitions.index,
        '--',
        color='tab:gray'
    )
    # Set axes properties
    axs[axs_i].set_xlabel(r'$\alpha_{max}$', horizontalalignment='right', x=1.0)
    axs[axs_i].set_ylabel(r'$False\ positive\ rate$')#, horizontalalignment='right', y=1.0)
    axs[axs_i].set_xscale('log')
    axs[axs_i].set_yscale('log')
    axs[axs_i].yaxis.set_ticks_position('both')
    axs[axs_i].set_xlim(right=1)
    axs[axs_i].set_ylim(top=1)
    # Add legend
    legend = ['Identity'] + [
        r'$N={}$'.format(nodes_n) for nodes_n in nodes_n_range]
    axs[axs_i].legend(labels=legend, loc='lower right')

    axs_i += 1
fig_list.append(fig)
axs_list.append(axs)

# # Plot incorrect targets rate vs. critical alpha
# # Legend: number of nodes
# # Subplots: number of samples
# fig, axs = plt.subplots(len(samples_n_range), 1, sharey=True)
# # fig.suptitle(r'$False\ positive\ rate\ vs.\ \alpha$')
# # Select data of interest
# df_interest = df[parameters_explored + ['incorrect_target_rate']]
# # Convert remaining DataFrame to float type for averaging
# df_interest = df_interest.astype(float)
# # Choose which of the explored parameters to average over when computing the
# # error bars
# parameters_to_average = {'repetition_i'}
# # If axs is not a list (i.e. it only contains one subplot), turn it into a
# # list with one element
# if not hasattr(axs, "__len__"):
#     axs = np.array([axs])
# axs_i = 0
# for samples_n in samples_n_range:
#     # Set subplot title
#     axs[axs_i].set_title('$T={}$'.format(samples_n))
#     # Select dataframe entries(i.e runs) with the same number of samples
#     df_samples = df_interest.loc[df_interest['samples_n'] == samples_n].drop('samples_n', 1)
#     # Drop NaN values
#     df_samples = df_samples.dropna()
#     for nodes_n in nodes_n_range:
#         # Select dataframe entries(i.e runs) with the same number of nodes
#         df_nodes = df_samples.loc[df_samples['nodes_n'] == nodes_n].drop(
#             'nodes_n', 1)
#         # Group by p_value and then compute mean and std
#         aggregate_repetitions = df_nodes.groupby('p_value').agg(
#             lambda x: list(x))
#         # Ensure that the desired parameters are averaged over when computing
#         # the error bars
#         df_keys_remaining = aggregate_repetitions.columns.get_level_values(0)
#         assert set.intersection(
#             set(df_keys_remaining),
#             set(parameters_explored)) == parameters_to_average
#         # Plot false positive rate
#         axs[axs_i].plot(
#             aggregate_repetitions.index,
#             [np.mean(aggregate_repetitions['incorrect_target_rate'][alpha_c])
#                 for alpha_c in alpha_c_range],
#             marker='o'
#             )
#         axs[axs_i].scatter(
#             df_samples['p_value'].values.tolist(),
#             df_samples['incorrect_target_rate'].values,
#             alpha=0.3
#             )
#     axs[axs_i].plot(
#         aggregate_repetitions.index,
#         aggregate_repetitions.index,
#         '--',
#         color='tab:gray'
#     )
#     # Set axes properties
#     axs[axs_i].set_xlabel(r'$\alpha_{max}$', horizontalalignment='right', x=1.0)
#     axs[axs_i].set_ylabel(r'$False\ positive\ rate$')#, horizontalalignment='right', y=1.0)
#     axs[axs_i].set_xscale('log')
#     axs[axs_i].set_yscale('log')
#     axs[axs_i].yaxis.set_ticks_position('both')
#     axs[axs_i].set_ylim([10**-4,1])
#     # Add legend
#     legend = ['Identity'] + list(nodes_n_range.astype(int))
#     #axs[axs_i].legend(labels=legend)
# 
#     axs_i += 1
# fig_list.append(fig)
# axs_list.append(axs)

# Plot precision and recall vs. alpha
# Legend: number of nodes
# Subplots vertical: number of samples
# Subplots horizontal: precision, recall
fig, axs = plt.subplots(len(samples_n_range), 2, sharex=True, sharey=True)
# ##########fig.suptitle(r'$False\ positive\ rate\ vs.\ alpha$')
# Select data of interest
df_interest = df[parameters_explored + ['precision', 'recall']]
# Convert remaining DataFrame to float type for averaging
df_interest = df_interest.astype(float)
# Choose which of the explored parameters to average over when computing the
# error bars
parameters_to_average = {'repetition_i'}
# If axs is not a list (i.e. it only contains one subplot), turn it into a
# list with one element
if not hasattr(axs[0], "__len__"):
    axs = np.array([axs])
axs_i = 0
for samples_n in samples_n_range:
    # Set subplot title
    axs[axs_i][0].set_title(r'$T={}$'.format(
        samples_n))
    axs[axs_i][1].set_title(r'$T={}$'.format(
        samples_n))
    # Select dataframe entries(i.e runs) with the same number of samples
    df_samples = df_interest.loc[df_interest['samples_n'] == samples_n].drop('samples_n', 1)
    #
    for nodes_n in nodes_n_range:
        # Select dataframe entries(i.e runs) with the same number of nodes
        df_nodes = df_samples.loc[df_samples['nodes_n'] == nodes_n].drop(
            'nodes_n', 1)
        # Group by p_value and then compute mean and std
        aggregate_repetitions = df_nodes.groupby('p_value').agg(
            ['mean', 'std'])
        # Ensure that the desired parameters are averaged over when computing
        # the error bars
        df_keys_remaining = aggregate_repetitions.columns.get_level_values(0)
        assert set.intersection(
            set(df_keys_remaining),
            set(parameters_explored)) == parameters_to_average

        # Plot precision
        axs[axs_i][0].errorbar(
            aggregate_repetitions.index,
            aggregate_repetitions['precision']['mean'],
            # Add vertical (symmetric) error bars
            yerr=aggregate_repetitions['precision']['std'],
            fmt='o-'
        )
        # Plot recall
        axs[axs_i][1].errorbar(
            aggregate_repetitions.index,
            aggregate_repetitions['recall']['mean'],
            # Add vertical (symmetric) error bars
            yerr=aggregate_repetitions['recall']['std'],
            fmt='o-'
        )
    # Set axes properties
    axs[axs_i][0].set_xlabel(
        r'$\alpha_{{max}}$',
        horizontalalignment='right',
        x=1.0)
    axs[axs_i][0].set_xscale('log')
    axs[axs_i][0].set_ylabel(r'$Precision$')#, horizontalalignment='right', y=1.0)
    axs[axs_i][0].yaxis.set_ticks_position('both')
    axs[axs_i][0].set_ylim(bottom=0)

    axs[axs_i][1].set_xlabel(
        r'$\alpha_{{max}}$',
        horizontalalignment='right',
        x=1.0)
    axs[axs_i][1].set_xscale('log')
    axs[axs_i][1].set_ylabel(r'$Recall$')#, horizontalalignment='right', y=1.0)
    axs[axs_i][1].set_ylim(bottom=0)
    # Add legend
    legend = [r'$N={}$'.format(nodes_n) for nodes_n in nodes_n_range]
    axs[axs_i][0].legend(labels=legend)
    axs[axs_i][1].legend(labels=legend)

    axs_i += 1
fig_list.append(fig)
axs_list.append(axs)

# # Plot false positive rate vs. number of nodes
# # Legend: p-value
# # Subplots: number of samples
# fig, axs = plt.subplots(len(samples_n_range), 1, sharey=True)
# # fig.suptitle(r'$False\ positive\ rate\ vs.\ number\ of\ nodes\ (empty\ network)$')
# # Select data of interest
# df_interest = df
# # Drop non relevant columns
# df_interest = df_interest.drop('TE_omnibus_empirical', 1)
# df_interest = df_interest.drop('TE_omnibus_theoretical_inferred_vars', 1)
# df_interest = df_interest.drop('TE_omnibus_theoretical_causal_vars', 1)
# # Convert remaining DataFrame to float type for averaging
# df_interest = df_interest.astype(float)
# # Choose which of the explored parameters to average over when computing the
# # error bars
# parameters_to_average = {'repetition_i'}
# # If axs is not a list (i.e. it only contains one subplot), turn it into a
# # list with one element
# if not hasattr(axs, "__len__"):
#     axs = np.array([axs])
# axs_i = 0
# for samples_n in samples_n_range:
#     # Set subplot title
#     axs[axs_i].set_title(r'$T={}$'.format(samples_n))
#     # Select dataframe entries(i.e runs) with the same number of samples
#     df_samples = df_interest.loc[df_interest['samples_n'] == samples_n].drop('samples_n', 1)
#     #
#     for alpha_c in alpha_c_range:
#         # Select dataframe entries(i.e runs) with the same p_value
#         df_alpha_c = df_samples.loc[df_samples['p_value'] == alpha_c].drop(
#             'p_value', 1)
#         # Group by number of nodes and then compute mean and std
#         aggregate_repetitions = df_alpha_c.groupby('nodes_n').agg(
#             ['mean', 'std'])
#         # Ensure that the desired parameters are averaged over when computing
#         # the error bars
#         df_keys_remaining = aggregate_repetitions.columns.get_level_values(0)
#         assert set.intersection(
#             set(df_keys_remaining),
#             set(parameters_explored)) == parameters_to_average
#         # Plot false positive rate
#         axs[axs_i].errorbar(
#             aggregate_repetitions.index,
#             aggregate_repetitions['false_pos_rate']['mean'],
#             # Add vertical (symmetric) error bars
#             yerr=aggregate_repetitions['false_pos_rate']['std'],
#             fmt='o-'
#         )
#     # Set axes properties
#     axs[axs_i].set_xlabel(r'$Network\ size$', horizontalalignment='right', x=1.0)
#     axs[axs_i].set_ylabel(r'$False\ positive\ rate$')#, horizontalalignment='right', y=1.0)
#     # axs[axs_i].set_xscale('log')
#     axs[axs_i].yaxis.set_ticks_position('both')
#     axs[axs_i].set_ylim(bottom=0)
#     # Add legend
#     #axs[axs_i].legend(labels=alpha_c_range)
# 
#     axs_i += 1
# fig_list.append(fig)
# axs_list.append(axs)

# # Receiver operating characteristic (linear scale)
# fig, axs = plt.subplots(len(nodes_n_range), 1, sharex=True, sharey=True)
# # fig.suptitle(r'$Receiver\ operating\ characteristic\ (linear\ scale)$')
# # Select data of interest
# df_interest = df
# # Drop non relevant columns
# df_interest = df_interest.drop('TE_omnibus_empirical', 1)
# df_interest = df_interest.drop('TE_omnibus_theoretical_inferred_vars', 1)
# df_interest = df_interest.drop('TE_omnibus_theoretical_causal_vars', 1)
# # Convert remaining DataFrame to float type for averaging
# df_interest = df_interest.astype(float)
# # Choose which of the explored parameters to average over when computing the
# # error bars
# parameters_to_average = {'repetition_i'}
# # If axs is not a list (i.e. it only contains one subplot), turn it into a
# # list with one element
# if not hasattr(axs, "__len__"):
#     axs = np.array([axs])
# axs_i = 0
# for nodes_n in nodes_n_range:
#     # Set subplot title
#     axs[axs_i].set_title(r'$N={}$'.format(nodes_n))
#     # Select dataframe entries(i.e runs) with the same number of nodes
#     df_nodes = df_interest.loc[df_interest['nodes_n'] == nodes_n].drop('nodes_n', 1)
#     for samples_n in samples_n_range:
#         # Select dataframe entries(i.e runs) with the same number of samples
#         df_samples = df_nodes.loc[df_nodes['samples_n'] == samples_n].drop(
#             'samples_n', 1)
#         # Group by p-value and then compute mean and std
#         aggregate_repetitions = df_samples.groupby('p_value').agg(
#             ['mean', 'std'])
#         # Ensure that the desired parameters are averaged over when computing
#         # the error bars
#         df_keys_remaining = aggregate_repetitions.columns.get_level_values(0)
#         assert set.intersection(
#             set(df_keys_remaining),
#             set(parameters_explored)) == parameters_to_average
#         # Plot true positive ratio vs False positive ratio
#         #axs[axs_i].scatter(
#         #    aggregate_repetitions['false_pos_rate']['mean'],  #False positive rate=1-specificity
#         #    aggregate_repetitions['recall']['mean']  #Recall=Sensitivity=True positive rate
#         #)
#         axs[axs_i].errorbar(
#             aggregate_repetitions['false_pos_rate']['mean'],
#             aggregate_repetitions['recall']['mean'],
#             # Add horizontal (symmetric) error bars
#             xerr=aggregate_repetitions['false_pos_rate']['std'],
#             # Add vertical (symmetric) error bars
#             yerr=aggregate_repetitions['recall']['std'],
#             fmt='o-'
#         )
#     # Set axes properties
#     axs[axs_i].set_xlabel(r'$False\ positive\ rate$', horizontalalignment='right', x=1.0)
#     axs[axs_i].set_ylabel(r'$True\ positive\ rate$')#, horizontalalignment='right', y=1.0)
#     axs[axs_i].yaxis.set_ticks_position('both')
#     axs[axs_i].set_xlim([0, 1])
#     axs[axs_i].set_ylim([0, 1])
#     # Add legend
# #    axs[axs_i].legend(labels=samples_n_range.astype(int))
# 
#     axs_i += 1
# fig_list.append(fig)
# axs_list.append(axs)

## Receiver operating characteristic (log scale)
#fig, axs = plt.subplots(len(nodes_n_range), 1, sharex=True, sharey=True)
##fig.suptitle(r'$Receiver\ operating\ characteristic\ (log\ scale)$')
## Select data of interest
#df_interest = df
## Drop non relevant columns
#df_interest = df_interest.drop('TE_omnibus_empirical', 1)
#df_interest = df_interest.drop('TE_omnibus_theoretical_inferred_vars', 1)
#df_interest = df_interest.drop('TE_omnibus_theoretical_causal_vars', 1)
## Convert remaining DataFrame to float type for averaging 
#df_interest = df_interest.astype(float)
## Choose which of the explored parameters to average over when computing the error bars
#parameters_to_average = {'repetition_i'}
## If axs is not a list (i.e. it only contains one subplot), turn it into a list with one element
#if not hasattr(axs, "__len__"):
#    axs = np.array([axs])
#axs_i = 0
#for nodes_n in nodes_n_range:
#    # Set subplot title
#    axs[axs_i].set_title(r'$N={}$'.format(nodes_n))
#    # Plot true positive ratio vs False positive ratio
#    # Select dataframe entries(i.e runs) with the same number of nodes
#    df_nodes = df_interest.loc[df_interest['nodes_n'] == nodes_n].drop('nodes_n', 1)
#    for samples_n in samples_n_range:
#        # Select dataframe entries(i.e runs) with the same number of samples
#        df_samples = df_nodes.loc[df_nodes['samples_n'] == samples_n].drop('samples_n', 1)
#        # Group by p_value and then compute mean and std
#        aggregate_repetitions = df_samples.groupby('p_value').agg(['mean', 'std'])
#        # Remove zero values on the x axis because of the log scale
#        aggregate_repetitions = aggregate_repetitions.loc[aggregate_repetitions['false_pos_rate']['mean'] > 0]
#        # Ensure that the desired parameters are averaged over when computing the error bars
#        df_keys_remaining = aggregate_repetitions.columns.get_level_values(0)
#        assert set.intersection(set(df_keys_remaining), set(parameters_explored)) == parameters_to_average
#        # Plot true positive ratio vs False positive ratio
#        #axs[axs_i].scatter(
#        #    aggregate_repetitions['false_pos_rate']['mean'],  #False pos rate=1-specificity
#        #    aggregate_repetitions['recall']['mean']  #Recall=Sensitivity
#        #)
#        axs[axs_i].errorbar(
#            aggregate_repetitions['false_pos_rate']['mean'],
#            aggregate_repetitions['recall']['mean'],
#            # Add horizontal (symmetric) error bars
#            xerr=aggregate_repetitions['false_pos_rate']['std'],
#            # Add vertical (symmetric) error bars
#            yerr=aggregate_repetitions['recall']['std'],
#            fmt='o'
#        )
#    # Set axes properties
#    axs[axs_i].set_xlabel(r'$False\ positive\ rate$', horizontalalignment='right', x=1.0)
#    axs[axs_i].set_ylabel(r'$True\ positive\ rate$')#, horizontalalignment='right', y=1.0)
#    axs[axs_i].set_xscale('log')
#    axs[axs_i].yaxis.set_ticks_position('both')
#    axs[axs_i].set_xlim([0, 1])
#    axs[axs_i].set_ylim([0, 1])
#    # Add legend
#    axs[axs_i].legend(labels=samples_n_range.astype(int))
#
#    axs_i += 1
#fig_list.append(fig)
#axs_list.append(axs)

# # Precision/Recall scatter plot
# # Subplots: network size
# # Probably better than Sensitivity/Specificity in our case, because the number
# # of negatives outweight the number of positives, see reference:
# # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4349800/
# fig, axs = plt.subplots(len(nodes_n_range), 1, sharex=True, sharey=True)
# #fig.suptitle(r'$Precision\ vs.\ Recall\ for\ different\ alpha\ values$')
# # Select data of interest
# df_interest = df[parameters_explored + ['precision', 'recall']]
# # Convert remaining DataFrame to float type for averaging 
# df_interest = df_interest.astype(float)
# # Choose which of the explored parameters to average over when computing the
# # error bars
# parameters_to_average = {'repetition_i'}
# # If axs is not a list (i.e. it only contains one subplot), turn it into a
# # list with one element
# if not hasattr(axs, "__len__"):
#     axs = np.array([axs])
# axs_i = 0
# for nodes_n in nodes_n_range:
#     # Set subplot title
#     axs[axs_i].set_title(r'$N={}$'.format(nodes_n))
#     # Select dataframe entries(i.e runs) with the same number of nodes
#     df_nodes = df_interest.loc[df_interest['nodes_n'] == nodes_n].drop('nodes_n', 1)
#     for samples_n in samples_n_range:
#         # Select dataframe entries(i.e runs) with the same number of samples
#         df_samples = df_nodes.loc[df_nodes['samples_n'] == samples_n].drop(
#             'samples_n', 1)
#         # Group by number of samples and then compute mean and std
#         aggregate_repetitions = df_samples.groupby('p_value').agg(
#             ['mean', 'std'])
#         # Ensure that the desired parameters are averaged over when computing
#         # the error bars
#         df_keys_remaining = aggregate_repetitions.columns.get_level_values(0)
#         assert set.intersection(
#             set(df_keys_remaining),
#             set(parameters_explored)) == parameters_to_average
#         
#         # Plot true positive ratio vs False positive ratio
#         #axs[axs_i].scatter(
#         #    aggregate_repetitions['recall']['mean'],
#         #    aggregate_repetitions['precision']['mean']
#         #)
#         axs[axs_i].errorbar(
#             aggregate_repetitions['recall']['mean'],
#             aggregate_repetitions['precision']['mean'],
#             # Add horizontal (symmetric) error bars
#             xerr=aggregate_repetitions['recall']['std'],
#             # Add vertical (symmetric) error bars
#             yerr=aggregate_repetitions['precision']['std'],
#             fmt='o'
#         )
#     # Set axes properties
#     axs[axs_i].set_xlabel(r'$Recall$', horizontalalignment='right', x=1.0)
#     axs[axs_i].set_ylabel(r'$Precision$')#, horizontalalignment='right', y=1.0)
#     #axs[axs_i].set_xscale('log')
#     axs[axs_i].yaxis.set_ticks_position('both')
#     axs[axs_i].set_xlim([0, 1])
#     axs[axs_i].set_ylim([0, 1])
#     # Add legend
#     axs[axs_i].legend(labels=samples_n_range.astype(int))
# 
#     axs_i += 1
# fig_list.append(fig)
# axs_list.append(axs)

# Precision/Recall scatter plot
# Subplots: sample size
# Probably better than Sensitivity/Specificity in our case, because the number
# of negatives outweight the number of positives, see reference:
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4349800/
# Set color pallette and/or line styles
mpl.rcParams['axes.prop_cycle'] = cycler(color='mybkgrc')
fig, axs = plt.subplots(len(samples_n_range), 1, sharex=True, sharey=True)
#fig.suptitle(r'$Precision\ vs.\ Recall\ for\ different\ p-values$')
# Select data of interest
df_interest = df[parameters_explored + ['precision', 'recall']]
# Convert remaining DataFrame to float type for averaging 
df_interest = df_interest.astype(float)
# Choose which of the explored parameters to average over when computing the
# error bars
parameters_to_average = {'nodes_n', 'repetition_i'}
# If axs is not a list (i.e. it only contains one subplot), turn it into a
# list with one element
if not hasattr(axs, "__len__"):
    axs = np.array([axs])
axs_i = 0
for samples_n in samples_n_range:
    # Set subplot title
    axs[axs_i].set_title('$T={}$'.format(samples_n))
    # Select dataframe entries(i.e runs) with the same number of samples
    df_samples = df_interest.loc[df_interest['samples_n'] == samples_n].drop('samples_n', 1)
    for alpha_c in alpha_c_range:
        # Select dataframe entries(i.e runs) with the same alpha_c
        df_alpha_c = df_samples.loc[df_samples['p_value'] == alpha_c].drop(
            'p_value', 1)
        # Ensure that the desired parameters are averaged over when computing
        # the error bars
        df_keys_remaining = df_alpha_c.columns
        assert set.intersection(
            set(df_keys_remaining),
            set(parameters_explored)) == parameters_to_average
        
        # Plot precision vs. recall
        axs[axs_i].scatter(
            df_alpha_c['recall'],
            df_alpha_c['precision']
        )
        #axs[axs_i].errorbar(
        #    aggregate_repetitions['recall']['mean'],
        #    aggregate_repetitions['precision']['mean'],
        #    # Add horizontal (symmetric) error bars
        #    xerr=aggregate_repetitions['recall']['std'],
        #    # Add vertical (symmetric) error bars
        #    yerr=aggregate_repetitions['precision']['std'],
        #    fmt='o'
        #)
    # Set axes properties
    axs[axs_i].set_xlabel(r'$Recall$', horizontalalignment='right', x=1.0)
    axs[axs_i].set_ylabel(r'$Precision$')#, horizontalalignment='right', y=1.0)
    #axs[axs_i].set_xscale('log')
    #axs[axs_i].set_yscale('log')
    axs[axs_i].yaxis.set_ticks_position('both')
    #axs[axs_i].set_xlim([0, 1])
    #axs[axs_i].set_ylim([0, 1])
    # Add legend
    legend = [
    r'$\alpha_{{max}}={}$'.format(alpha_c) for alpha_c in alpha_c_range]
    axs[axs_i].legend(labels=legend)

    axs_i += 1
# Reset color pallette and/or line styles to default
mpl.rcParams['axes.prop_cycle'] = cycler_default
fig_list.append(fig)
axs_list.append(axs)

# Precision/Recall scatter plot (aggregate, only show mean)
# Error bars: max and min values
# Probably better than Sensitivity/Specificity in our case, because the number
# of negatives outweight the number of positives, see reference:
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4349800/
# Set color pallette and/or line styles
mpl.rcParams['axes.prop_cycle'] = cycler(color='mybkgrc')
arrow_color = 'b'
fig = plt.figure()
axs = plt.gca()
axs = np.array([axs])
# Select data of interest
df_interest = df[parameters_explored + ['precision', 'recall']]
# Convert remaining DataFrame to float type for averaging 
df_interest = df_interest.astype(float)
# Choose which of the explored parameters to average over when computing the
# error bars
parameters_to_average = {'nodes_n', 'repetition_i'}
# Arrow coordinates
arrow_coord = np.zeros(shape=(len(alpha_c_range), len(samples_n_range), 2))
arrow_i = 0
for alpha_c in alpha_c_range:
    # Select dataframe entries(i.e runs) with the same alpha_c
    df_alpha_c = df_interest.loc[df_interest['p_value'] == alpha_c].drop(
        'p_value', 1)
    # Group by number of samples and then compute mean and std
    aggregate_repetitions = df_alpha_c.groupby('samples_n').agg(
        ['mean', 'min', 'max'])
    # Ensure that the desired parameters are averaged over when computing
    # the error bars
    df_keys_remaining = aggregate_repetitions.columns.get_level_values(0)
    assert set.intersection(
        set(df_keys_remaining),
        set(parameters_explored)) == parameters_to_average
    # Plot precision vs. recall
    axs[0].errorbar(
        aggregate_repetitions['recall']['mean'],
        aggregate_repetitions['precision']['mean'],
        # Add horizontal (asymmetric) error bars
        xerr=[
            (aggregate_repetitions['recall']['mean']
             - aggregate_repetitions['recall']['min']),
            (aggregate_repetitions['recall']['max']
             - aggregate_repetitions['recall']['mean'])
            ],
        # Add vertical (asymmetric) error bars
        yerr=[
            (aggregate_repetitions['precision']['mean']
             - aggregate_repetitions['precision']['min']),
            (aggregate_repetitions['precision']['max']
             - aggregate_repetitions['precision']['mean'])
            ],
        fmt='o'
    )
    # arrow_coord[arrow_i, :, 0] = aggregate_repetitions['recall']['mean']
    arrow_coord[arrow_i, 0:len(aggregate_repetitions), 0] = (
        aggregate_repetitions['recall']['mean'])
    # arrow_coord[arrow_i, :, 1] = aggregate_repetitions['recall']['mean']
    arrow_coord[arrow_i, 0:len(aggregate_repetitions), 1] = (
        aggregate_repetitions['precision']['mean'])
    arrow_i += 1
for i in range(len(samples_n_range) - 1):
    arrow_width = 0.01
    arrow_head_length = 0.02
    plt.arrow(
        arrow_coord[0, i, 0],
        arrow_coord[0, i, 1],
        arrow_coord[-1, i, 0] - arrow_coord[0, i, 0],
        arrow_coord[-1, i, 1] - arrow_coord[0, i, 1],
        alpha=0.3,
        width=arrow_width,
        head_length=arrow_head_length,
        length_includes_head=True,
        color=arrow_color
        )
# Add legend
legend = [
    r'$\alpha_{{max}}={}$'.format(alpha_c) for alpha_c in alpha_c_range]
axs[0].legend(labels=legend, loc=(0.5, 0.3))
# Set axes properties
axs[0].set_xlabel(r'$Recall$', horizontalalignment='right', x=1.0)
axs[0].set_ylabel(r'$Precision$')#, horizontalalignment='right', y=1.0)
#axs[0].set_xscale('log')
#axs[0].set_yscale('log')
axs[0].xaxis.set_ticks_position('both')
axs[0].yaxis.set_ticks_position('both')
axs[0].set_xlim([-0.01, 1.01])
#axs[0].set_ylim([0.4, 1])
#axs[0].set_xticks([0.3, 0.5, 0.95])
#axs[0].set_xticks(np.arange(0, 1.1, 0.1), minor=True)
#axs[0].set_yticks([0, 0.5, 1])
#axs[0].set_yticks(np.arange(0,1.1,0.1), minor=True)
#plt.xticks([0.15,0.3,0.5,0.95,0.975], ('100 samples', '0.3', '1000 samples', '0.95', '10000 samples'), rotation=30)
axs[0].axvline(0.3, linestyle='--', color='k')
axs[0].axvline(0.95, linestyle='--', color='k')
# Add text
#plt.text(0.1, 0.33, r'$T=100$', fontsize=10)
#plt.text(0.57, 0.33, r'$T=1000$', fontsize=10)
#plt.text(0.965, 0.4, r'$T=1000$', fontsize=10, rotation=90)
plt.text(0.12, plt.ylim()[0] + 0.08, r'$T=100$', fontsize=10)
plt.text(0.57, plt.ylim()[0] + 0.08, r'$T=1000$', fontsize=10)
plt.text(0.97, plt.ylim()[0] + 0.21, r'$T=10000$', fontsize=10, rotation=90)
# Reset color pallette and/or line styles to default
mpl.rcParams['axes.prop_cycle'] = cycler_default
fig_list.append(fig)
axs_list.append(axs)

# Precision/Recall scatter plot (aggregate, scatter)
# Probably better than Sensitivity/Specificity in our case, because the number
# of negatives outweight the number of positives, see reference:
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4349800/
# Set color pallette and/or line styles
mpl.rcParams['axes.prop_cycle'] = cycler(color='mybkgrc')
arrow_color = 'b'
fig = plt.figure()
axs = plt.gca()
axs = np.array([axs])
# Select data of interest
df_interest = df[parameters_explored + ['precision', 'recall']]
# Convert remaining DataFrame to float type for averaging 
df_interest = df_interest.astype(float)
# Choose which of the explored parameters to average over when computing the
# error bars
parameters_to_average = {'nodes_n', 'samples_n', 'repetition_i'}
for alpha_c in alpha_c_range:
    # Select dataframe entries(i.e runs) with the same alpha_c
    df_alpha_c = df_interest.loc[df_interest['p_value'] == alpha_c].drop(
        'p_value', 1)
    
    # Ensure that the desired parameters are averaged over when computing
    # the error bars
    df_keys_remaining = df_alpha_c.columns
    assert set.intersection(
        set(df_keys_remaining),
        set(parameters_explored)) == parameters_to_average
    
    # Plot precision vs. recall
    axs[0].scatter(
        df_alpha_c['recall'],
        df_alpha_c['precision']
    )
for i in range(len(samples_n_range) - 1):
    arrow_width = 0.01
    arrow_head_length = 0.02
    plt.arrow(
        arrow_coord[0, i, 0],
        arrow_coord[0, i, 1],
        arrow_coord[-1, i, 0] - arrow_coord[0, i, 0],
        arrow_coord[-1, i, 1] - arrow_coord[0, i, 1],
        alpha=0.3,
        width=arrow_width,
        head_length=arrow_head_length,
        length_includes_head=True,
        color=arrow_color
        )
# Add legend
legend = [
    r'$\alpha_{{max}}={}$'.format(alpha_c) for alpha_c in alpha_c_range]
axs[0].legend(labels=legend, loc=(0.5, 0.3))
# Set axes properties
axs[0].set_xlabel(r'$Recall$', horizontalalignment='right', x=1.0)
axs[0].set_ylabel(r'$Precision$')#, horizontalalignment='right', y=1.0)
#axs[0].set_xscale('log')
#axs[0].set_yscale('log')
axs[0].yaxis.set_ticks_position('both')
axs[0].set_xlim([-0.01, 1.01])
#axs[0].set_ylim([0.4, 1])
#axs[0].set_xticks([0.3, 0.5, 0.95])
#axs[0].set_xticks(np.arange(0, 1.1, 0.1), minor=True)
#axs[0].set_yticks([0, 0.5, 1])
#axs[0].set_yticks(np.arange(0,1.1,0.1), minor=True)
#plt.xticks([0.15,0.3,0.5,0.95,0.975], ('100 samples', '0.3', '1000 samples', '0.95', '10000 samples'), rotation=30)
axs[0].axvline(0.3, linestyle='--', color='k') #use 0.45 for CLM and 0.3 for VAR
axs[0].axvline(0.95, linestyle='--', color='k') #use 0.91 for CLM and 0.95 for VAR
# Add text
#plt.text(0.1, 0.33, r'$T=100$', fontsize=10)
#plt.text(0.57, 0.33, r'$T=1000$', fontsize=10)
#plt.text(0.965, 0.4, r'$T=1000$', fontsize=10, rotation=90)
plt.text(0.12, plt.ylim()[0] + 0.08, r'$T=100$', fontsize=10)
plt.text(0.57, plt.ylim()[0] + 0.08, r'$T=1000$', fontsize=10)
plt.text(0.97, plt.ylim()[0] + 0.21, r'$T=10000$', fontsize=10, rotation=90)
# Reset color pallette and/or line styles to default
mpl.rcParams['axes.prop_cycle'] = cycler_default
fig_list.append(fig)
axs_list.append(axs)

# Plot delay error mean vs. samples_number
# Each subplot corresponds to a different nerwork size
fig, axs = plt.subplots(len(alpha_c_range), 1, sharex=True, sharey=True)
# Select data of interest
df_interest = df[parameters_explored + ['delay_error_mean']]
# Convert remaining DataFrame to float type for averaging
df_interest = df_interest.astype(float)
# Choose which of the explored parameters to average over when computing the
# error bars
parameters_to_average = {'repetition_i'}
# If axs is not a list (i.e. it only contains one subplot), turn it into a
# list with one element
if not hasattr(axs, "__len__"):
    axs = np.array([axs])
axs_i = 0
for alpha_c in alpha_c_range:
    # Set subplot title
    # axs[axs_i].set_title(r'$\alpha_{{max}}={}$'.format(alpha_c))
    # Select dataframe entries(i.e runs) with the same number of nodes
    df_alpha_c = df_interest.loc[df_interest['p_value'] == alpha_c].drop('p_value', 1)
    for nodes_n in nodes_n_range:
        # Select dataframe entries(i.e runs) with the same number of nodes
        df_nodes = df_alpha_c.loc[df_alpha_c['nodes_n'] == nodes_n].drop(
            'nodes_n', 1)
        # Group by number of samples and then compute mean and std
        aggregate_repetitions = df_nodes.groupby('samples_n').agg(
            ['mean', 'std'])
        # Ensure that the desired parameters are averaged over when computing
        # the error bars
        df_keys_remaining = aggregate_repetitions.columns.get_level_values(0)
        assert set.intersection(
            set(df_keys_remaining),
            set(parameters_explored)) == parameters_to_average

        # Plot delay error
        axs[axs_i].errorbar(
            aggregate_repetitions.index,
            aggregate_repetitions['delay_error_mean']['mean'],
            # Add vertical (symmetric) error bars
            yerr=aggregate_repetitions['delay_error_mean']['std'],
            fmt='-o'
        )
        # Set axes properties
        axs[axs_i].set_xlabel(r'$T$', horizontalalignment='right', x=1.0)
        axs[axs_i].set_ylabel(r'$Absolute\ error$')#, horizontalalignment='right', y=1.0)
        axs[axs_i].set_xscale('log')
        axs[axs_i].yaxis.set_ticks_position('both')
        axs[axs_i].set_xlim([95, 11000])
        axs[axs_i].set_ylim(bottom=0)
        # Add legend
        legend = [r'$N={}$'.format(nodes_n) for nodes_n in nodes_n_range]
        axs[axs_i].legend(labels=legend)

    axs_i += 1
fig_list.append(fig)
axs_list.append(axs)

# Plot delay error mean vs. samples_number (only alpha_of_interest)
# Each subplot corresponds to a different nerwork size
fig = plt.figure()
axs = plt.gca()
axs = np.array([axs])
# Select data of interest
df_interest = df[parameters_explored + ['delay_error_mean']]
df_interest = df_interest.loc[df_interest['p_value'] == alpha_interest].drop('p_value', 1)
# Convert remaining DataFrame to float type for averaging
df_interest = df_interest.astype(float)
# Choose which of the explored parameters to average over when computing the
# error bars
parameters_to_average = {'repetition_i'}

for nodes_n in nodes_n_range:
    # Select dataframe entries(i.e runs) with the same number of nodes
    df_nodes = df_interest.loc[df_interest['nodes_n'] == nodes_n].drop(
        'nodes_n', 1)
    # Group by number of samples and then compute mean and std
    aggregate_repetitions = df_nodes.groupby('samples_n').agg(
        ['mean', 'std'])
    # Ensure that the desired parameters are averaged over when computing
    # the error bars
    df_keys_remaining = aggregate_repetitions.columns.get_level_values(0)
    assert set.intersection(
        set(df_keys_remaining),
        set(parameters_explored)) == parameters_to_average

    # Plot delay error
    axs[0].errorbar(
        aggregate_repetitions.index,
        aggregate_repetitions['delay_error_mean']['mean'],
        # Add vertical (symmetric) error bars
        yerr=aggregate_repetitions['delay_error_mean']['std'],
        fmt='-o'
    )
    print(nodes_n)
    print(aggregate_repetitions['delay_error_mean']['mean'])
    # Set axes properties
    axs[0].set_xlabel(r'$T$', horizontalalignment='right', x=1.0)
    axs[0].set_ylabel(r'$Absolute\ error$')#, horizontalalignment='right', y=1.0)
    axs[0].set_xscale('log')
    axs[0].yaxis.set_ticks_position('both')
    axs[0].set_xlim([95, 11000])
    axs[0].set_ylim(bottom=0)
    # Add legend
    legend = [r'$N={}$'.format(nodes_n) for nodes_n in nodes_n_range]
    axs[0].legend(labels=legend)

fig_list.append(fig)
axs_list.append(axs)

# Plot TE_omnibus_empirical histogram
# Subplots vertical: number of samples
# Subplots horizontal: number of nodes
fig, axs = plt.subplots(
    len(samples_n_range),
    len(nodes_n_range),
    sharex=True,
    sharey=True
    )
# Select data of interest
df_interest = df[parameters_explored + ['TE_omnibus_empirical']]
df_interest = df_interest.loc[df_interest['p_value'] == alpha_interest].drop('p_value', 1)
# Choose which of the explored parameters to collect data over
parameters_to_average = {'repetition_i'}
# If axs is not a list (i.e. it only contains one subplot), turn it into a
# list with one element
if not hasattr(axs[0], "__len__"):
    axs = np.array([axs])
for (axs_row, samples_n) in enumerate(samples_n_range):
    # Select dataframe entries(i.e runs) with the same number of samples
    df_samples = df_interest.loc[df_interest['samples_n'] == samples_n].drop(
        'samples_n', 1)
    # Group by nodes_n and concatenate TE values lists
    aggregate_repetitions = df_samples.groupby('nodes_n').agg(
        lambda x: x.tolist())
    # Ensure that the desired parameters are averaged over when computing
    # the error bars
    df_keys_remaining = aggregate_repetitions.columns.get_level_values(0)
    assert set.intersection(
        set(df_keys_remaining),
        set(parameters_explored)) == parameters_to_average
    for (axs_col, nodes_n) in enumerate(nodes_n_range):
        # Set subplot title
        axs[axs_row, axs_col].set_title(r'$T={0}$, $N={1}$'.format(
            samples_n,
            nodes_n
            ))
        if not np.all(np.isnan(aggregate_repetitions.loc[nodes_n].TE_omnibus_empirical)):
            TE_omnibus_empirical = np.concatenate(
                aggregate_repetitions.loc[nodes_n].TE_omnibus_empirical)
            # Remove NaN values
            TE_omnibus_empirical = TE_omnibus_empirical[~np.isnan(TE_omnibus_empirical)]
            # Plot omnibus TE histogram
            axs[axs_row, axs_col].hist(
                TE_omnibus_empirical)

        # Set axes properties
        axs[axs_row, axs_col].set_xlabel(r'$Omnibus\ TE\ empirical$', horizontalalignment='right', x=1.0)
        #axs[axs_row, axs_col].set_xscale('log')
        #axs[axs_row, axs_col].set_ylabel(' ')#, horizontalalignment='right', y=1.0)
        axs[axs_row, axs_col].yaxis.set_ticks_position('both')
        axs[axs_row, axs_col].set_ylim(bottom=0)

fig_list.append(fig)
axs_list.append(axs)

# Plot TE_omnibus_empirical histogram
fig = plt.figure()
axs = plt.gca()
axs = np.array([axs])
# Select data of interest
df_interest = df[parameters_explored + ['TE_omnibus_empirical']]
df_interest = df_interest.loc[df_interest['p_value'] == alpha_interest].drop('p_value', 1)
df_interest = df_interest.loc[df_interest['samples_n'] == 10000].drop('samples_n', 1)
# Set subplot title
axs[0].set_title(r'$T=10000$')
if not np.all(np.isnan(df_interest.TE_omnibus_empirical.tolist()[0])):
    TE_omnibus_empirical = np.concatenate(
        df_interest.TE_omnibus_empirical.tolist()
        )
    # Remove NaN values
    TE_omnibus_empirical = TE_omnibus_empirical[~np.isnan(TE_omnibus_empirical)]
    # Plot omnibus TE histogram
    axs[0].hist(TE_omnibus_empirical)
# Set axes properties
axs[0].set_xlabel(r'$Omnibus\ TE\ empirical$', horizontalalignment='right', x=1.0)
axs[0].yaxis.set_ticks_position('both')
axs[0].set_ylim(bottom=0)

fig_list.append(fig)
axs_list.append(axs)
# ----------------------------------------------------------------------------------
# Save figures to PDF file
pdf_metadata = {}
if fdr:
    pdf_path = os.path.join(traj_dir, 'figures_fdr.pdf')
else:
    pdf_path = os.path.join(traj_dir, 'figures.pdf')
save_figures_to_pdf(fig_list, pdf_path)
