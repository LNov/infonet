import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
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
            # pdf.attach_note("plot of sin(x)")  

            # Save current figure to PDF page
            pdf.savefig(fig)

            print('figure {0} saved'.format(fig_i))

        # Set PDF file metadata via the PdfPages object
        d = pdf.infodict()
        for key in pdf_metadata.keys():
            d[key] = pdf_metadata.get(key, '')


# Choose whether to use FDR-corrected results or not
fdr = False

# Choose folder
traj_dir = os.path.join('trajectories', '2018_04_19_15h06m42s')

# Set up plot style
# Use "matplotlib.rcdefaults()" to restore the default plot style"
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['lines.markersize'] = 3
mpl.rcParams['errorbar.capsize'] = 3

# Load DataFrame
if fdr:
    df = pd.read_pickle(os.path.join(traj_dir, 'postprocessing_fdr.pkl'))
else:
    df = pd.read_pickle(os.path.join(traj_dir, 'postprocessing.pkl'))

# Determine explore parameters (by construction, they are the name
# of the columns that precede 'precision')
# WARNING: this method is prone to errors, when the columns are rearranged,
# however, in such cases the "assertion" sanity checks below will fail, so I
# will know
parameters_explored = df.loc[[], :'precision'].keys().tolist()[:-1]

#
nodes_n_range = np.unique(df['nodes_n'])
samples_n_range = np.unique(df['samples_n'])
p_value_range = np.unique(df['p_value'])

#
fig_list = []
axs_list = []

# Plot performance tests vs. number of samples
# Subplots: network size
fig, axs = plt.subplots(len(nodes_n_range), 1, sharey=True)
#fig.suptitle('$Performance\ tests\ (subplots:\ n.\ nodes)$')
# Select data of interest
df_interest = df.loc[df['p_value'] == 0.001].drop('p_value', 1)
# Choose which of the explored parameters to average over when computing the
# error bars
parameters_to_average = {'repetition_i'}
# If axs is not a list (i.e. it only contains one subplot), turn it into a
# list with one element
if not hasattr(axs, "__len__"):
    axs = np.array([axs])
axs_i = 0
for nodes_n in nodes_n_range:
    # Set subplot title
    axs[axs_i].set_title('$network\ size\ =\ ${}'.format(nodes_n.astype(int)))
    # Select dataframe entries(i.e runs) with the same number of nodes
    df_nodes = df_interest.loc[df_interest['nodes_n'] == nodes_n].drop(
        'nodes_n', 1)
    # Group by number of samples and then compute mean and std
    aggregate_repetitions = df_nodes.groupby('samples_n').agg(['mean', 'std'])
    # Ensure that the desired parameters are averaged over when computing the
    # error bars
    df_keys_remaining = aggregate_repetitions.columns.get_level_values(0)
    assert set.intersection(
        set(df_keys_remaining),
        set(parameters_explored)) == parameters_to_average
    # Plot precision
    axs[axs_i].errorbar(
        aggregate_repetitions.index,
        aggregate_repetitions['precision']['mean'],
        # Add vertical (symmetric) error bars
        yerr=aggregate_repetitions['precision']['std'],
        fmt='-o'
    )
    # Plot recall
    axs[axs_i].errorbar(
        aggregate_repetitions.index,
        aggregate_repetitions['recall']['mean'],
        # Add vertical (symmetric) error bars
        yerr=aggregate_repetitions['recall']['std'],
        fmt='-o'
    )
    # Plot false positive rate
    axs[axs_i].errorbar(
        aggregate_repetitions.index,
        aggregate_repetitions['specificity']['mean'],
        # Add vertical (symmetric) error bars
        yerr=aggregate_repetitions['specificity']['std'],
        fmt='o-'
    )
    # Set axes properties
    axs[axs_i].set_xlabel('$Samples$', horizontalalignment='right', x=1.0)
    axs[axs_i].set_ylabel('$Performance$')  # , horizontalalignment='right', y=1.0)
    axs[axs_i].set_xscale('log')
    axs[axs_i].yaxis.set_ticks_position('both')
    axs[axs_i].set_xlim([100, 10000])
    axs[axs_i].set_ylim(bottom=0)
    # Add legend
    axs[axs_i].legend(labels=['Precision', 'Recall', 'Specificity'])

    axs_i += 1
fig_list.append(fig)
axs_list.append(axs)

# Plot performance tests vs. network size
# Subplots: number of samples
fig, axs = plt.subplots(len(samples_n_range), 1, sharey=True)
# fig.suptitle('$Performance\ tests\ (subplots:\ n.\ samples)$')
# Select data of interest
df_interest = df.loc[df['p_value'] == 0.001].drop('p_value', 1)
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
    axs[axs_i].set_title('$n.\ samples\ =\ ${}'.format(samples_n.astype(int)))
    # Select dataframe entries(i.e runs) with the same number of samples
    df_samples = df_interest.loc[df_interest['samples_n'] == samples_n].drop(
        'samples_n', 1)
    # Group by number of nodes and then compute mean and std
    aggregate_repetitions = df_samples.groupby('nodes_n').agg(['mean', 'std'])
    # Ensure that the desired parameters are averaged over when computing the
    # error bars
    df_keys_remaining = aggregate_repetitions.columns.get_level_values(0)
    assert set.intersection(
        set(df_keys_remaining),
        set(parameters_explored)) == parameters_to_average
    # Plot precision
    axs[axs_i].errorbar(
        aggregate_repetitions.index,
        aggregate_repetitions['precision']['mean'],
        # Add vertical (symmetric) error bars
        yerr=aggregate_repetitions['precision']['std'],
        fmt='-o'
    )
    # Plot recall
    axs[axs_i].errorbar(
        aggregate_repetitions.index,
        aggregate_repetitions['recall']['mean'],
        # Add vertical (symmetric) error bars
        yerr=aggregate_repetitions['recall']['std'],
        fmt='-o'
    )
    # Plot false positive rate
    axs[axs_i].errorbar(
        aggregate_repetitions.index,
        aggregate_repetitions['specificity']['mean'],
        # Add vertical (symmetric) error bars
        yerr=aggregate_repetitions['specificity']['std'],
        fmt='o-'
    )
    # Set axes properties
    axs[axs_i].set_xlabel('$Nodes$', horizontalalignment='right', x=1.0)
    axs[axs_i].set_ylabel('$Performance$')#, horizontalalignment='right', y=1.0)
    #axs[axs_i].set_xscale('log')
    axs[axs_i].yaxis.set_ticks_position('both')
    axs[axs_i].set_ylim(bottom=0)
    # Add legend
    axs[axs_i].legend(labels=['Precision', 'Recall', 'Specificity'])

    axs_i += 1
fig_list.append(fig)
axs_list.append(axs)

# Plot false positive rate vs. critical alpha
# Legend: number of samples
# Subplots: number of nodes
fig, axs = plt.subplots(len(nodes_n_range), 1, sharey=True)
# fig.suptitle('$False\ positive\ rate\ vs.\ \\alpha$')
# Choose which of the explored parameters to average over when computing the
# error bars
parameters_to_average = {'repetition_i'}
# If axs is not a list (i.e. it only contains one subplot), turn it into a
# list with one element
if not hasattr(axs, "__len__"):
    axs = np.array([axs])
axs_i = 0
for nodes_n in nodes_n_range:
    # Set subplot title
    axs[axs_i].set_title('$network\ size\ =\ ${}'.format(nodes_n.astype(int)))
    # Select dataframe entries(i.e runs) with the same number of nodes
    df_nodes = df.loc[df['nodes_n'] == nodes_n].drop('nodes_n', 1)
    #
    for samples_n in samples_n_range:
        # Select dataframe entries(i.e runs) with the same number of samples
        df_samples = df_nodes.loc[df_nodes['samples_n'] == samples_n].drop(
            'samples_n', 1)
        # Group by p_value and then compute mean and std
        aggregate_repetitions = df_samples.groupby('p_value').agg(
            ['mean', 'std'])
        # Ensure that the desired parameters are averaged over when computing
        # the error bars
        df_keys_remaining = aggregate_repetitions.columns.get_level_values(0)
        assert set.intersection(
            set(df_keys_remaining),
            set(parameters_explored)) == parameters_to_average
        # Plot false positive rate
        axs[axs_i].errorbar(
            aggregate_repetitions.index[
                aggregate_repetitions['false_pos_rate']['mean'] > 0],
            aggregate_repetitions['false_pos_rate']['mean'][
                aggregate_repetitions['false_pos_rate']['mean'] > 0],
            # Add vertical error bars
            yerr=[
                0*aggregate_repetitions['false_pos_rate']['std'][
                    aggregate_repetitions['false_pos_rate']['mean'] > 0],
                aggregate_repetitions['false_pos_rate']['std'][
                    aggregate_repetitions['false_pos_rate']['mean'] > 0]
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
    axs[axs_i].set_xlabel('$\\alpha$', horizontalalignment='right', x=1.0)
    axs[axs_i].set_ylabel('$False\ positive\ rate$')#, horizontalalignment='right', y=1.0)
    axs[axs_i].set_xscale('log')
    axs[axs_i].set_yscale('log')
    axs[axs_i].yaxis.set_ticks_position('both')
    # axs[axs_i].set_ylim([0, 1])
    # Add legend
    legend = ['Identity'] + list(samples_n_range.astype(int))
    #axs[axs_i].legend(labels=legend)

    axs_i += 1
fig_list.append(fig)
axs_list.append(axs)

# Plot false positive rate vs. critical alpha
# Legend: number of nodes
# Subplots: number of samples
fig, axs = plt.subplots(len(samples_n_range), 1, sharey=True)
# fig.suptitle('$False\ positive\ rate\ vs.\ \\alpha$')
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
    axs[axs_i].set_title('$n.\ samples\ =\ ${}'.format(samples_n.astype(int)))
    # Select dataframe entries(i.e runs) with the same number of samples
    df_samples = df.loc[df['samples_n'] == samples_n].drop('samples_n', 1)
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
        # Plot false positive rate
        axs[axs_i].errorbar(
            aggregate_repetitions.index[
                aggregate_repetitions['false_pos_rate']['mean'] > 0],
            aggregate_repetitions['false_pos_rate']['mean'][
                aggregate_repetitions['false_pos_rate']['mean'] > 0],
            # Add vertical (symmetric) error bars
            yerr=[
                0*aggregate_repetitions['false_pos_rate']['std'][
                    aggregate_repetitions['false_pos_rate']['mean'] > 0],
                aggregate_repetitions['false_pos_rate']['std'][
                    aggregate_repetitions['false_pos_rate']['mean'] > 0]
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
    axs[axs_i].set_xlabel('$\\alpha$', horizontalalignment='right', x=1.0)
    axs[axs_i].set_ylabel('$False\ positive\ rate$')#, horizontalalignment='right', y=1.0)
    axs[axs_i].set_xscale('log')
    axs[axs_i].set_yscale('log')
    axs[axs_i].yaxis.set_ticks_position('both')
    # axs[axs_i].set_ylim([0, 1])
    # Add legend
    legend = ['Identity'] + list(nodes_n_range.astype(int))
    #axs[axs_i].legend(labels=legend)

    axs_i += 1
fig_list.append(fig)
axs_list.append(axs)

# Plot incorrect targets rate vs. critical alpha
# Legend: number of nodes
# Subplots: number of samples
fig, axs = plt.subplots(len(samples_n_range), 1, sharey=True)
# fig.suptitle('$False\ positive\ rate\ vs.\ \\alpha$')
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
    axs[axs_i].set_title('$n.\ samples\ =\ ${}'.format(samples_n.astype(int)))
    # Select dataframe entries(i.e runs) with the same number of samples
    df_samples = df.loc[df['samples_n'] == samples_n].drop('samples_n', 1)
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
        # Plot false positive rate
        axs[axs_i].errorbar(
            aggregate_repetitions.index[
                aggregate_repetitions['incorrect_target_rate']['mean'] > 0],
            aggregate_repetitions['incorrect_target_rate']['mean'][
                aggregate_repetitions['incorrect_target_rate']['mean'] > 0],
            # Add vertical (symmetric) error bars
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
    axs[axs_i].set_xlabel('$\\alpha_{max}$', horizontalalignment='right', x=1.0)
    axs[axs_i].set_ylabel('$False\ pos\ target$')#, horizontalalignment='right', y=1.0)
    axs[axs_i].set_xscale('log')
    axs[axs_i].set_yscale('log')
    axs[axs_i].yaxis.set_ticks_position('both')
    # axs[axs_i].set_ylim([0, 1])
    # Add legend
    legend = ['Identity'] + list(nodes_n_range.astype(int))
    #axs[axs_i].legend(labels=legend)

    axs_i += 1
fig_list.append(fig)
axs_list.append(axs)

# Plot precision, recall, and false positive rate vs. p-value
# Legend: number of nodes
# Subplots vertical: number of samples
# Subplots horizontal: precision, recall, false positive rate
fig, axs = plt.subplots(len(samples_n_range), 2, sharex=True, sharey=True)
# ##########fig.suptitle('$False\ positive\ rate\ vs.\ p-value$')
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
    axs[axs_i][0].set_title('$n.\ samples\ =\ ${}'.format(
        samples_n.astype(int)))
    axs[axs_i][1].set_title('$n.\ samples\ =\ ${}'.format(
        samples_n.astype(int)))
    # Select dataframe entries(i.e runs) with the same number of samples
    df_samples = df.loc[df['samples_n'] == samples_n].drop('samples_n', 1)
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
    axs[axs_i][0].set_xlabel('$p-value$', horizontalalignment='right', x=1.0)
    axs[axs_i][0].set_xscale('log')
    axs[axs_i][0].set_ylabel('$Precision$')#, horizontalalignment='right', y=1.0)
    axs[axs_i][0].yaxis.set_ticks_position('both')
    axs[axs_i][0].set_ylim(bottom=0)

    axs[axs_i][1].set_xlabel('$p-value$', horizontalalignment='right', x=1.0)
    axs[axs_i][1].set_xscale('log')
    axs[axs_i][1].set_ylabel('$Recall$')#, horizontalalignment='right', y=1.0)
    axs[axs_i][1].set_ylim(bottom=0)
    # Add legend
    axs[axs_i][0].legend(labels=nodes_n_range.astype(int))
    axs[axs_i][1].legend(labels=nodes_n_range.astype(int))

    axs_i += 1
fig_list.append(fig)
axs_list.append(axs)

# Plot false positive rate vs. number of nodes
# Legend: p-value
# Subplots: number of samples
fig, axs = plt.subplots(len(samples_n_range), 1, sharey=True)
# fig.suptitle('$False\ positive\ rate\ vs.\ number\ of\ nodes\ (empty\ network)$')
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
    axs[axs_i].set_title('$n.\ samples\ =\ ${}'.format(samples_n.astype(int)))
    # Select dataframe entries(i.e runs) with the same number of samples
    df_samples = df.loc[df['samples_n'] == samples_n].drop('samples_n', 1)
    #
    for p_value in p_value_range:
        # Select dataframe entries(i.e runs) with the same p_value
        df_p_value = df_samples.loc[df_samples['p_value'] == p_value].drop(
            'p_value', 1)
        # Group by number of nodes and then compute mean and std
        aggregate_repetitions = df_p_value.groupby('nodes_n').agg(
            ['mean', 'std'])
        # Ensure that the desired parameters are averaged over when computing
        # the error bars
        df_keys_remaining = aggregate_repetitions.columns.get_level_values(0)
        assert set.intersection(
            set(df_keys_remaining),
            set(parameters_explored)) == parameters_to_average
        # Plot false positive rate
        axs[axs_i].errorbar(
            aggregate_repetitions.index,
            aggregate_repetitions['false_pos_rate']['mean'],
            # Add vertical (symmetric) error bars
            yerr=aggregate_repetitions['false_pos_rate']['std'],
            fmt='o-'
        )
    # Set axes properties
    axs[axs_i].set_xlabel('$Nodes$', horizontalalignment='right', x=1.0)
    axs[axs_i].set_ylabel('$False\ positive\ rate$')#, horizontalalignment='right', y=1.0)
    # axs[axs_i].set_xscale('log')
    axs[axs_i].yaxis.set_ticks_position('both')
    axs[axs_i].set_ylim(bottom=0)
    # Add legend
    #axs[axs_i].legend(labels=p_value_range)

    axs_i += 1
fig_list.append(fig)
axs_list.append(axs)

# Receiver operating characteristic (linear scale)
fig, axs = plt.subplots(len(nodes_n_range), 1, sharex=True, sharey=True)
# fig.suptitle('$Receiver\ operating\ characteristic\ (linear\ scale)$')
# Choose which of the explored parameters to average over when computing the
# error bars
parameters_to_average = {'repetition_i'}
# If axs is not a list (i.e. it only contains one subplot), turn it into a
# list with one element
if not hasattr(axs, "__len__"):
    axs = np.array([axs])
axs_i = 0
for nodes_n in nodes_n_range:
    # Set subplot title
    axs[axs_i].set_title('$network\ size\ =\ ${}'.format(nodes_n.astype(int)))
    # Select dataframe entries(i.e runs) with the same number of nodes
    df_nodes = df.loc[df['nodes_n'] == nodes_n].drop('nodes_n', 1)
    for samples_n in samples_n_range:
        # Select dataframe entries(i.e runs) with the same number of samples
        df_samples = df_nodes.loc[df_nodes['samples_n'] == samples_n].drop(
            'samples_n', 1)
        # Group by p-value and then compute mean and std
        aggregate_repetitions = df_samples.groupby('p_value').agg(
            ['mean', 'std'])
        # Ensure that the desired parameters are averaged over when computing
        # the error bars
        df_keys_remaining = aggregate_repetitions.columns.get_level_values(0)
        assert set.intersection(
            set(df_keys_remaining),
            set(parameters_explored)) == parameters_to_average
        # Plot true positive ratio vs False positive ratio
        #axs[axs_i].scatter(
        #    aggregate_repetitions['false_pos_rate']['mean'],  #False positive rate=1-specificity
        #    aggregate_repetitions['recall']['mean']  #Recall=Sensitivity=True positive rate
        #)
        axs[axs_i].errorbar(
            aggregate_repetitions['false_pos_rate']['mean'],
            aggregate_repetitions['recall']['mean'],
            # Add horizontal (symmetric) error bars
            xerr=aggregate_repetitions['false_pos_rate']['std'],
            # Add vertical (symmetric) error bars
            yerr=aggregate_repetitions['recall']['std'],
            fmt='o-'
        )
    # Set axes properties
    axs[axs_i].set_xlabel('$False\ positive\ rate$', horizontalalignment='right', x=1.0)
    axs[axs_i].set_ylabel('$True\ positive\ rate$')#, horizontalalignment='right', y=1.0)
    axs[axs_i].yaxis.set_ticks_position('both')
    axs[axs_i].set_xlim([0, 1])
    axs[axs_i].set_ylim([0, 1])
    # Add legend
    axs[axs_i].legend(labels=samples_n_range.astype(int))

    axs_i += 1
fig_list.append(fig)
axs_list.append(axs)

## Receiver operating characteristic (log scale)
#fig, axs = plt.subplots(len(nodes_n_range), 1, sharex=True, sharey=True)
##fig.suptitle('$Receiver\ operating\ characteristic\ (log\ scale)$')
## Choose which of the explored parameters to average over when computing the error bars
#parameters_to_average = {'repetition_i'}
## If axs is not a list (i.e. it only contains one subplot), turn it into a list with one element
#if not hasattr(axs, "__len__"):
#    axs = np.array([axs])
#axs_i = 0
#for nodes_n in nodes_n_range:
#    # Set subplot title
#    axs[axs_i].set_title('$network\ size\ =\ ${}'.format(nodes_n.astype(int)))
#    # Plot true positive ratio vs False positive ratio
#    # Select dataframe entries(i.e runs) with the same number of nodes
#    df_nodes = df.loc[df['nodes_n'] == nodes_n].drop('nodes_n', 1)
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
#    axs[axs_i].set_xlabel('$False\ positive\ rate$', horizontalalignment='right', x=1.0)
#    axs[axs_i].set_ylabel('$True\ positive\ rate$')#, horizontalalignment='right', y=1.0)
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

# Precision/Recall scatter plot
# Probably better than Sensitivity/Specificity in our case, because the number
# of negatives outweight the number of positives, see reference:
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4349800/
fig, axs = plt.subplots(len(nodes_n_range), 1, sharex=True, sharey=True)
#fig.suptitle('$Precision\ vs.\ Recall\ for\ different\ p-values$')
# Choose which of the explored parameters to average over when computing the
# error bars
parameters_to_average = {'repetition_i'}
# If axs is not a list (i.e. it only contains one subplot), turn it into a
# list with one element
if not hasattr(axs, "__len__"):
    axs = np.array([axs])
axs_i = 0
for nodes_n in nodes_n_range:
    # Set subplot title
    axs[axs_i].set_title('$network\ size\ =\ ${}'.format(nodes_n.astype(int)))
    # Select dataframe entries(i.e runs) with the same number of nodes
    df_nodes = df.loc[df['nodes_n'] == nodes_n].drop('nodes_n', 1)
    for samples_n in samples_n_range:
        # Select dataframe entries(i.e runs) with the same number of samples
        df_samples = df_nodes.loc[df_nodes['samples_n'] == samples_n].drop(
            'samples_n', 1)
        # Group by number of samples and then compute mean and std
        aggregate_repetitions = df_samples.groupby('p_value').agg(
            ['mean', 'std'])
        # Ensure that the desired parameters are averaged over when computing
        # the error bars
        df_keys_remaining = aggregate_repetitions.columns.get_level_values(0)
        assert set.intersection(
            set(df_keys_remaining),
            set(parameters_explored)) == parameters_to_average
        
        # Plot true positive ratio vs False positive ratio
        #axs[axs_i].scatter(
        #    aggregate_repetitions['recall']['mean'],
        #    aggregate_repetitions['precision']['mean']
        #)
        axs[axs_i].errorbar(
            aggregate_repetitions['recall']['mean'],
            aggregate_repetitions['precision']['mean'],
            # Add horizontal (symmetric) error bars
            xerr=aggregate_repetitions['recall']['std'],
            # Add vertical (symmetric) error bars
            yerr=aggregate_repetitions['precision']['std'],
            fmt='o'
        )
    # Set axes properties
    axs[axs_i].set_xlabel('$Recall$', horizontalalignment='right', x=1.0)
    axs[axs_i].set_ylabel('$Precision$')#, horizontalalignment='right', y=1.0)
    #axs[axs_i].set_xscale('log')
    axs[axs_i].yaxis.set_ticks_position('both')
    axs[axs_i].set_xlim([0, 1])
    axs[axs_i].set_ylim([0, 1])
    # Add legend
    axs[axs_i].legend(labels=samples_n_range.astype(int))

    axs_i += 1
fig_list.append(fig)
axs_list.append(axs)

# Plot delay error mean vs. samples_number
# Each subplot corresponds to a different nerwork size
fig, axs = plt.subplots(len(p_value_range), 1, sharex=True, sharey=True)
#fig.suptitle('$Relative\ delay\ error\ (mean\ over\ true\ positives)\ (subplots:\ n.\ nodes)$')
# Choose which of the explored parameters to average over when computing the
# error bars
parameters_to_average = {'repetition_i'}
# to set common label, add a big axes, hide frame
#fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
#plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
#plt.xlabel('$xlabel$')
#plt.ylabel('$ylabel$')
# If axs is not a list (i.e. it only contains one subplot), turn it into a
# list with one element
if not hasattr(axs, "__len__"):
    axs = np.array([axs])
axs_i = 0
for p_value in p_value_range:
    # Set subplot title
    axs[axs_i].set_title('$p-value\ =\ ${}'.format(p_value))
    # Select dataframe entries(i.e runs) with the same number of nodes
    df_p_value = df.loc[df['p_value'] == p_value].drop('p_value', 1)
    for nodes_n in nodes_n_range:
        # Select dataframe entries(i.e runs) with the same number of nodes
        df_nodes = df_p_value.loc[df_p_value['nodes_n'] == nodes_n].drop(
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
        axs[axs_i].set_xlabel('$Samples$', horizontalalignment='right', x=1.0)
        axs[axs_i].set_ylabel('$Relative\ error$')#, horizontalalignment='right', y=1.0)
        axs[axs_i].set_xscale('log')
        axs[axs_i].yaxis.set_ticks_position('both')
        axs[axs_i].set_xlim([100, 10000])
        axs[axs_i].set_ylim(bottom=0)
        # Add legend
        #axs[count].legend(labels=[''])

    axs_i += 1
fig_list.append(fig)
axs_list.append(axs)


#network_size_list = np.array(traj.f_get('parameters.topology.initial.nodes_n').f_get_range())
#samples_n_list = np.array(traj.f_get('parameters.node_dynamics.samples_n').f_get_range())
#
#
#G_original=nx.DiGraph(traj.results.run_00000000.topology.initial.adjacency_matrix.values)
#delay_matrix_inferred = pd.DataFrame(
#        _get_adj_matrix(
#            traj.results['run_00000000'].network_inference.network_inference_result,
#            False,
#            'max_te'
#        )
#    )
#G_inferred=nx.DiGraph(delay_matrix_inferred.values)
#
#
#fig0, (ax1, ax2) = plt.subplots(1, 2)
#nx.draw_circular(G_original, with_labels=True, node_size=500, alpha=1.0, ax=ax1,
#                     node_color='cadetblue', hold=True, font_size=14,
#                     font_weight='bold')
#nx.draw_circular(G_inferred, with_labels=True, node_size=500, alpha=1.0, ax=ax2,
#                     node_color='cadetblue', hold=True, font_size=14,
#                     font_weight='bold')
#
#fig = plt.figure()
#for axs_id in range(int(nodes_n)):
#    plt.subplot(nodes_n + 1, 1, axs_id + 1)
#    plt.plot(traj.results[traj.f_get_run_names()[-1]].node_dynamics.time_series[axs_id,-100:-1])
#
#
#

# Save figures to PDF file
pdf_metadata = {}
if fdr:
    pdf_path = os.path.join(traj_dir, 'figures_fdr.pdf')
else:
    pdf_path = os.path.join(traj_dir, 'figures.pdf')
save_figures_to_pdf(fig_list, pdf_path)
