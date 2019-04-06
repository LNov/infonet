import os
import numpy as np
from scipy.stats import binom
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid.inset_locator import (
    inset_axes, InsetPosition, mark_inset)
from cycler import cycler
from mylib_pypet import print_leaves
# import collections


def new_fig(function_name):
    fig, axs = globals()[function_name]()
    fig_list.append(fig)
    axs_list.append(axs)


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


def plot_performance_vs_T():
    # Plot performance tests vs. number of samples
    # Subplots: network size

    # Define (sub)plot(s)
    fig, axs = plt.subplots(len(nodes_n_range), 1, sharey=True)
    # Select data of interest
    df_interest = df[
        parameters_explored + ['precision', 'recall', 'specificity']]
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    # Convert remaining DataFrame to float type for averaging
    df_interest = df_interest.astype(float)
    # Choose which of the explored parameters to average over when computing
    # the error bars
    parameters_to_average = {'repetition_i'}
    # If axs is not a list (i.e. it only contains one subplot), turn it into a
    # list with one element
    if not hasattr(axs, "__len__"):
        axs = np.array([axs])
    axs_i = 0
    for nodes_n in nodes_n_range:
        # Set subplot title
        axs[axs_i].set_title(r'$network\ size\ =\ ${}'.format(nodes_n))
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
        axs[axs_i].set_xlabel(r'$T$', horizontalalignment='right', x=1.0)
        axs[axs_i].set_ylabel(r'$Performance$')  # , horizontalalignment='right', y=1.0)
        axs[axs_i].set_xscale('log')
        axs[axs_i].yaxis.set_ticks_position('both')
        axs[axs_i].set_xlim([100, 10000])
        axs[axs_i].set_ylim(bottom=0)
        # Add legend
        axs[axs_i].legend(labels=['Precision', 'Recall', 'Specificity'])

        axs_i += 1
    
    return fig, axs


def plot_performance_vs_N_scatter():
    # Plot performance tests vs. network size (scatter error)
    # Subplots: number of samples

    # Define (sub)plot(s)
    fig, axs = plt.subplots(len(samples_n_range), 1, sharey=True)

    # Select data of interest
    df_interest = df[
        parameters_explored + ['precision', 'recall', 'specificity']]
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    # Convert remaining DataFrame to float type for averaging
    df_interest = df_interest.astype(float)
    # Choose which of the explored parameters to average over when computing
    # the error bars
    parameters_to_average = {'repetition_i'}
    # If axs is not a list (i.e. it only contains one subplot), turn it into a
    # list with one element
    if not hasattr(axs, "__len__"):
        axs = np.array([axs])
    axs_i = 0
    for samples_n in samples_n_range:
        # Set subplot title
        # axs[axs_i].set_title(r'$T={}$'.format(samples_n))
        # Select dataframe entries(i.e runs) with the same number of samples
        df_samples = df_interest.loc[
            df_interest['samples_n'] == samples_n].drop('samples_n', 1)
        # Drop NaN values
        df_samples = df_samples.dropna()
        # Group by number of nodes
        aggregate_repetitions = df_samples.groupby('nodes_n').agg(
            lambda x: list(x))
        # Ensure that the desired parameters are averaged over when computing
        # the error bars
        df_keys_remaining = aggregate_repetitions.columns.get_level_values(0)
        averaging_set = set.intersection(
            set(df_keys_remaining),
            set(parameters_explored))
        assert averaging_set == parameters_to_average, 'attempting to average over {0}, which differs from the specified set {1}'.format(averaging_set, parameters_to_average)
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
        axs[axs_i].set_ylabel(r'$T={}$'.format(samples_n))  #, horizontalalignment='right', y=1.0)
        # axs[axs_i].set_xscale('log')
        axs[axs_i].yaxis.set_ticks_position('both')
        axs[axs_i].set_ylim(bottom=0)
        # Add legend
        axs[axs_i].legend(
            labels=['Precision', 'Recall', 'Specificity'],
            loc=(0.05, 0.16)
            )

        axs_i += 1

    return fig, axs


# region OLD # Plot performance tests vs. network size (max-min error bars)
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
#endregion


# region OLD # Plot performance tests vs. network size (box-whiskers error bars)
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
#endregion


def plot_precision_vs_N_scatter():
    # Plot precision vs. network size (scatter error)
    # Subplots: none
    # Legend: samples

    # Set color pallette and/or line styles
    mpl.rcParams['axes.prop_cycle'] = cycler('ls', ['-', '--', ':'])

    # Define (sub)plot(s)
    fig = plt.figure()
    axs = plt.gca()
    axs = np.array([axs])
    # Select data of interest
    df_interest = df[parameters_explored + ['precision']]
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    # Convert remaining DataFrame to float type for averaging
    df_interest = df_interest.astype(float)
    # Choose which of the explored parameters to average over when computing
    # the error bars
    parameters_to_average = {'repetition_i'}
    for samples_n in samples_n_range:
        # Select dataframe entries(i.e runs) with the same number of samples
        df_samples = df_interest.loc[
            df_interest['samples_n'] == samples_n].drop('samples_n', 1)
        # Drop NaN values
        df_samples = df_samples.dropna()
        # Group by number of nodes and then compute mean and extrema
        aggregate_repetitions = df_samples.groupby('nodes_n').agg(
            lambda x: list(x))
        # Ensure that the desired parameters are averaged over when computing
        # the error bars
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
            color='tab:blue')
        axs[0].scatter(
            df_samples['nodes_n'].values.astype(int).tolist(),
            df_samples['precision'].values,
            alpha=0.3,
            c='tab:blue')
    # Set axes properties
    axs[0].set_xlabel(r'$Network\ size$', horizontalalignment='right', x=1.0)
    axs[0].set_ylabel(r'$Precision$')  # , horizontalalignment='right', y=1.0)
    axs[0].yaxis.set_ticks_position('both')
    #axs[0].set_yscale('log')
    axs[0].set_ylim(bottom=0)
    # Add legend
    legend = [
        r'$T={}$'.format(samples_n) for samples_n in samples_n_range]
    axs[0].legend(labels=legend, loc=(0.05, 0.16))  #remove loc for AR
    # Reset color pallette and/or line styles to default
    mpl.rcParams['axes.prop_cycle'] = cycler_default

    return fig, axs


def plot_precision_vs_N_scatter_inset():
    # Plot precision vs. network size (scatter error)
    # Subplots: none
    # Legend: samples

    # Set color pallette and/or line styles
    # mpl.rcParams['axes.prop_cycle'] = cycler('ls', ['-', '--', ':'])
    mpl.rcParams['axes.prop_cycle'] = cycler(color=[
        'tab:red',
        'tab:purple',
        'tab:brown',
        'tab:gray',
        'tab:olive',
        ])
    # Define (sub)plot(s)
    fig = plt.figure()
    axs = plt.gca()
    axs = np.array([axs])
    # Define inset plot
    # Create a set of inset Axes: these should fill the bounding box allocated
    # to them.
    axs_inset = plt.axes([0, 0, 1, 1])
    # Manually set the position and relative size of the inset axes within ax1
    ip = InsetPosition(axs[0], [0.495, 0.16, 0.4*1.2, 0.3*1.2])
    axs_inset.set_axes_locator(ip)
    # Mark the region corresponding to the inset axes on ax1 and draw lines
    # in grey linking the two axes.
    # mark_inset(axs[0], axs_inset, loc1=3, loc2=4, fc="none", ec='0.3')
    # Select data of interest
    df_interest = df[parameters_explored + ['precision']]
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    # Convert remaining DataFrame to float type for averaging
    df_interest = df_interest.astype(float)
    # Choose which of the explored parameters to average over when computing
    # the error bars
    parameters_to_average = {'repetition_i'}
    for samples_n in samples_n_range:
        # Select dataframe entries(i.e runs) with the same number of samples
        df_samples = df_interest.loc[
            df_interest['samples_n'] == samples_n].drop('samples_n', 1)
        # Drop NaN values
        df_samples = df_samples.dropna()
        # Group by number of nodes and then compute mean and extrema
        aggregate_repetitions = df_samples.groupby('nodes_n').agg(
            lambda x: list(x))
        # Ensure that the desired parameters are averaged over when computing
        # the error bars
        df_keys_remaining = aggregate_repetitions.columns.get_level_values(0)
        assert set.intersection(
            set(df_keys_remaining),
            set(parameters_explored)) == parameters_to_average
        # Plot precision
        axs[0].plot(
            aggregate_repetitions.index.astype(int).tolist(),
            [np.mean(aggregate_repetitions['precision'][nodes_n])
             for nodes_n in nodes_n_range],
            marker='o')
        axs_inset.plot(
            aggregate_repetitions.index.astype(int).tolist(),
            [np.mean(aggregate_repetitions['precision'][nodes_n])
             for nodes_n in nodes_n_range],
            marker='o')
        axs[0].scatter(
            df_samples['nodes_n'].values.astype(int).tolist(),
            df_samples['precision'].values,
            alpha=0.3,
            c='tab:blue')
        axs_inset.scatter(
            df_samples['nodes_n'].values.astype(int).tolist(),
            df_samples['precision'].values,
            alpha=0.3)
    # Set axes properties
    axs[0].set_xlabel(r'$Network\ size$', horizontalalignment='right', x=1.0)
    axs[0].set_ylabel(r'$Precision$')  # , horizontalalignment='right', y=1.0)
    axs[0].yaxis.set_ticks_position('both')
    plt.setp(axs_inset, xticks=nodes_n_range, yticks=[0.9, 0.95, 1.0])
    axs_inset.tick_params(axis='both', which='major', labelsize=10)
    axs_inset.set_ylim(bottom=0.89, top=1.01)
    # axs[0].set_yscale('log')
    axs[0].set_ylim(bottom=0)
    # Add legend
    legend = [
        r'$T={}$'.format(samples_n) for samples_n in samples_n_range]
    axs[0].legend(labels=legend, loc=(0.05, 0.16))
    # Reset color pallette and/or line styles to default
    mpl.rcParams['axes.prop_cycle'] = cycler_default

    return fig, axs


def plot_recall_vs_N_scatter():
    # Plot recall vs. network size (scatter error)
    # Subplots: none
    # Legend: samples

    # Set color pallette and/or line styles
    # mpl.rcParams['axes.prop_cycle'] = cycler('ls', ['-', '--', ':'])
    mpl.rcParams['axes.prop_cycle'] = cycler(color=[
        'tab:red',
        'tab:purple',
        'tab:brown',
        'tab:gray',
        'tab:olive',
        ])

    # Define (sub)plot(s)
    fig = plt.figure()
    axs = plt.gca()
    axs = np.array([axs])
    # Select data of interest
    df_interest = df[parameters_explored + ['recall']]
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    # Convert remaining DataFrame to float type for averaging
    df_interest = df_interest.astype(float)
    # Choose which of the explored parameters to average over when computing
    # the error bars
    parameters_to_average = {'repetition_i'}
    for samples_n in samples_n_range:
        # Select dataframe entries(i.e runs) with the same number of samples
        df_samples = df_interest.loc[
            df_interest['samples_n'] == samples_n].drop('samples_n', 1)
        # Drop NaN values
        df_samples = df_samples.dropna()
        # Group by number of nodes and then compute mean and extrema
        aggregate_repetitions = df_samples.groupby('nodes_n').agg(
            lambda x: list(x))
        # Ensure that the desired parameters are averaged over when computing
        # the error bars
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
            #color='tab:orange'
        )
        axs[0].scatter(
            df_samples['nodes_n'].values.astype(int).tolist(),
            df_samples['recall'].values,
            alpha=0.3,
            #c='tab:orange'
        )
    # Set axes properties
    axs[0].set_xlabel(r'$Network\ size$', horizontalalignment='right', x=1.0)
    axs[0].set_ylabel(r'$Recall$')  # , horizontalalignment='right', y=1.0)
    # axs[0].set_xscale('log')
    axs[0].yaxis.set_ticks_position('both')
    axs[0].set_ylim(bottom=0)
    # Add legend
    legend = [
        r'$T={}$'.format(samples_n) for samples_n in samples_n_range]
    # axs[0].legend(labels=legend, loc=(0.05, 0.16))  # CLM
    # axs[0].legend(labels=legend, loc=(0.75, 0.56))  # VAR
    axs[0].legend(labels=legend, loc=(0.08, 0.21))  # VAR
    # Reset color pallette and/or line styles to default
    mpl.rcParams['axes.prop_cycle'] = cycler_default

    return fig, axs


def plot_spectral_radius_vs_N_scatter():
    # Plot spectral radius vs. network size (scatter error)
    # Subplots: none
    # Legend: none

    # Define (sub)plot(s)
    fig = plt.figure()
    axs = plt.gca()
    axs = np.array([axs])
    # Select data of interest
    df_interest = df[['nodes_n', 'repetition_i', 'spectral_radius']]
    # Convert remaining DataFrame to float type for averaging
    df_interest = df_interest.astype(float)
    # Choose which of the explored parameters to average over when computing
    # the error bars
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
    axs[0].set_ylabel(r'$\rho$')  # , horizontalalignment='right', y=1.0)
    # axs[0].set_xscale('log')
    axs[0].yaxis.set_ticks_position('both')
    # axs[0].set_ylim(bottom=0)

    return fig, axs


def plot_FPR_target_vs_alpha():
    # Plot incorrect targets rate vs. critical alpha
    # Legend: number of nodes
    # Subplots: number of samples

    # Define (sub)plot(s)
    fig, axs = plt.subplots(len(samples_n_range), 1, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + ['incorrect_target_rate']]
    # Convert remaining DataFrame to float type for averaging
    df_interest = df_interest.astype(float)
    # Choose which of the explored parameters to average over when computing
    # the error bars
    parameters_to_average = {'repetition_i'}
    # If axs is not a list (i.e. it only contains one subplot), turn it into a
    # list with one element
    if not hasattr(axs, "__len__"):
        axs = np.array([axs])
    axs_i = 0
    for samples_n in samples_n_range:
        # Set subplot title
        axs[axs_i].set_title('$T={}$'.format(samples_n.astype(int)))
        # Select dataframe entries(i.e runs) with the same number of samples
        df_samples = df_interest.loc[
            df_interest['samples_n'] == samples_n].drop('samples_n', 1)
        # Drop NaN values
        df_samples = df_samples.dropna()
        for nodes_n in nodes_n_range:
            # Select dataframe entries(i.e runs) with the same number of nodes
            df_nodes = df_samples.loc[df_samples['nodes_n'] == nodes_n].drop(
                'nodes_n', 1)
            # Group by p_value and then compute mean and std
            aggregate_repetitions = df_nodes.groupby('p_value').agg(
                ['mean', 'std', 'max', 'min'])
            # Ensure that the desired parameters are averaged over when
            # computing the error bars
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
        axs[axs_i].set_ylabel(r'$False\ positive\ rate$')  # , horizontalalignment='right', y=1.0)
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

    return fig, axs


def plot_FPR_target_vs_alpha_quantile_std():
    # Plot incorrect targets rate vs. critical alpha
    # Legend: number of nodes
    # Subplots: number of samples

    # Set color pallette and/or line styles
    mpl.rcParams['axes.prop_cycle'] = cycler(color=[
        'tab:blue',
        'tab:blue',
        'tab:orange',
        'tab:orange',
        'tab:green',
        'tab:green',
        'tab:red',
        'tab:red'])
    # Define (sub)plot(s)
    fig, axs = plt.subplots(1, 1, sharey=True)  # len(samples_n_range)
    # Select data of interest
    df_interest = df[parameters_explored + ['incorrect_target_rate']]
    # Convert remaining DataFrame to float type for averaging
    df_interest = df_interest.astype(float)
    # Choose which of the explored parameters to average over when computing
    # the error bars
    parameters_to_average = {'repetition_i'}
    # If axs is not a list (i.e. it only contains one subplot), turn it into a
    # list with one element
    if not hasattr(axs, "__len__"):
        axs = np.array([axs])
    axs_i = 0
    for samples_n in samples_n_range[-1:]:
        # Set subplot title
        # axs[axs_i].set_title('$T={}$'.format(samples_n.astype(int)))
        # Select dataframe entries(i.e runs) with the same number of samples
        df_samples = df_interest.loc[
            df_interest['samples_n'] == samples_n].drop('samples_n', 1)
        # Drop NaN values
        df_samples = df_samples.dropna()
        for nodes_n in nodes_n_range:
            # Select dataframe entries(i.e runs) with the same number of nodes
            df_nodes = df_samples.loc[df_samples['nodes_n'] == nodes_n].drop(
                'nodes_n', 1)
            # Group by p_value and then compute mean and std
            aggregate_repetitions = df_nodes.groupby('p_value').agg(
                ['mean', 'std', 'max', 'min', 'size'])
            # Ensure that the desired parameters are averaged over when
            # computing the error bars
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
            s = 100000
            quantile = 0.95
            quantile_int = []
            for alpha in alpha_c_range:
                repetitions = aggregate_repetitions['incorrect_target_rate']['size'][alpha]
                x = np.random.binomial(
                    nodes_n, alpha, size=(repetitions, s))
                x = x / nodes_n
                data = x.std(axis=0)
                data_sorted = np.sort(data)
                quantile_int.append(data_sorted[int(quantile * s)])
            axs[axs_i].plot(
               alpha_c_range,
               np.array(aggregate_repetitions['incorrect_target_rate']['mean']) + quantile_int,
               'x'
            )
        axs[axs_i].plot(
            aggregate_repetitions.index,
            aggregate_repetitions.index,
            '--',
            color='tab:gray'
        )
        # Set axes properties
        axs[axs_i].set_xlabel(r'$\alpha_{max}$', horizontalalignment='right', x=1.0)
        axs[axs_i].set_ylabel(r'$False\ positive\ rate$')  # , horizontalalignment='right', y=1.0)
        axs[axs_i].set_xscale('log')
        axs[axs_i].set_yscale('log')
        axs[axs_i].yaxis.set_ticks_position('both')
        # axs[axs_i].set_xlim(right=1)
        # axs[axs_i].set_ylim(top=1)
        # # Add legend
        # legend = ['Identity'] + [
        #     r'$N={}$'.format(nodes_n) for nodes_n in nodes_n_range]
        # axs[axs_i].legend(labels=legend, loc='lower right')

        axs_i += 1

    # Reset color pallette and/or line styles to default
    mpl.rcParams['axes.prop_cycle'] = cycler_default

    return fig, axs


def plot_FPR_target_vs_alpha_quantile_mean():
    # Plot incorrect targets rate vs. critical alpha
    # Legend: number of nodes
    # Subplots: number of samples

    # Define (sub)plot(s)
    fig, axs = plt.subplots(1, 1, sharey=True)  # len(samples_n_range)
    # Select data of interest
    df_interest = df[parameters_explored + ['incorrect_target_rate']]
    # Convert remaining DataFrame to float type for averaging
    df_interest = df_interest.astype(float)
    # Choose which of the explored parameters to average over when computing
    # the error bars
    parameters_to_average = {'repetition_i'}
    # If axs is not a list (i.e. it only contains one subplot), turn it into a
    # list with one element
    if not hasattr(axs, "__len__"):
        axs = np.array([axs])
    axs_i = 0
    for samples_n in samples_n_range[-1:]:
        # Set subplot title
        # axs[axs_i].set_title('$T={}$'.format(samples_n.astype(int)))
        # Select dataframe entries(i.e runs) with the same number of samples
        df_samples = df_interest.loc[
            df_interest['samples_n'] == samples_n].drop('samples_n', 1)
        # Drop NaN values
        df_samples = df_samples.dropna()
        i_color = 0
        for nodes_n in nodes_n_range:
            # Select dataframe entries(i.e runs) with the same number of nodes
            df_nodes = df_samples.loc[df_samples['nodes_n'] == nodes_n].drop(
                'nodes_n', 1)
            # Group by p_value and then compute mean and std
            aggregate_repetitions = df_nodes.groupby('p_value').agg(
                ['mean', 'std', 'max', 'min', 'size'])
            # Ensure that the desired parameters are averaged over when
            # computing the error bars
            df_keys_remaining = aggregate_repetitions.columns.get_level_values(0)
            assert set.intersection(
                set(df_keys_remaining),
                set(parameters_explored)) == parameters_to_average
            # Plot false positive rate
            axs[axs_i].plot(
                aggregate_repetitions[aggregate_repetitions[
                    'incorrect_target_rate']['mean'] > 0].index,
                aggregate_repetitions['incorrect_target_rate']['mean'][
                        aggregate_repetitions['incorrect_target_rate']['mean'] > 0],
                'o',
                ms=2,
                color=colors_default[i_color],
                label=r'$N={}$'.format(nodes_n))
            ## Add legend
            #legend = ['Identity'] + [
            #    r'$N={}$'.format(nodes_n) for nodes_n in nodes_n_range]
            quantile = 0.95
            quantiles_top = []
            quantiles_bottom = []
            # alpha_more = np.logspace(
            #     np.log10(alpha_c_range.min()),
            #     np.log10(alpha_c_range.max()),
            #     num=10)
            # repetitions = aggregate_repetitions[
            #     'incorrect_target_rate']['size'][alpha_c_range[0]]
            for alpha in alpha_c_range:#alpha_more:
                repetitions = aggregate_repetitions[
                   'incorrect_target_rate']['size'][alpha]
                data_fpr = np.random.binomial(
                    nodes_n, alpha, size=(repetitions, 100000)) / nodes_n
                data_fpr_rep = data_fpr.mean(axis=0)
                quantiles_top.append(
                    np.quantile(data_fpr_rep, quantile))
                quantiles_bottom.append(
                    np.quantile(data_fpr_rep, 1 - quantile))
            quantiles_bottom = np.array(quantiles_bottom)
            quantiles_bottom[quantiles_bottom == 0.] = np.nan
            axs[axs_i].plot(
                alpha_c_range,#alpha_more,
                quantiles_top,
                '_',
                alpha_c_range,#alpha_more,
                quantiles_bottom,
                '_',
                ms=10,
                color=colors_default[i_color],
                label='_nolegend_')
            i_color += 1
        axs[axs_i].plot(
            aggregate_repetitions.index,
            aggregate_repetitions.index,
            '--',
            color='tab:gray',
            label='Identity')
        # Set axes properties
        axs[axs_i].set_xlabel(r'$\alpha_{max}$', horizontalalignment='right', x=1.0)
        axs[axs_i].set_ylabel(r'$False\ positive\ rate$')  # , horizontalalignment='right', y=1.0)
        axs[axs_i].set_xscale('log')
        axs[axs_i].set_yscale('log')
        #axs[axs_i].set_yscale('symlog', linthreshx=0.00001)
        axs[axs_i].yaxis.set_ticks_position('both')
        # Print legend
        axs[axs_i].legend(loc='lower right')

        axs_i += 1

    return fig, axs


def plot_FPR_target_vs_alpha_quantile_N_interest():
    # Plot incorrect targets rate vs. critical alpha
    # Subplots: number of samples

    # Define (sub)plot(s)
    fig, axs = plt.subplots(len(samples_n_range), 1)
    # Select data of interest
    df_interest = df[parameters_explored + ['incorrect_target_rate']]
    df_interest = df_interest.loc[
            df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    # Convert remaining DataFrame to float type for averaging
    df_interest = df_interest.astype(float)
    # Choose which of the explored parameters to average over when computing
    # the error bars
    parameters_to_average = {'repetition_i'}
    # If axs is not a list (i.e. it only contains one subplot), turn it into a
    # list with one element
    if not hasattr(axs, "__len__"):
        axs = np.array([axs])
    axs_i = 0
    for samples_n in samples_n_range:
        # Set subplot title
        axs[axs_i].set_title('$T={}$'.format(samples_n))
        # Select dataframe entries(i.e runs) with the same number of samples
        df_samples = df_interest.loc[
            df_interest['samples_n'] == samples_n].drop('samples_n', 1)
        # Drop NaN values
        df_samples = df_samples.dropna()
        # Group by p_value and then compute mean and std
        aggregate_repetitions = df_samples.groupby('p_value').agg(
            ['mean', 'std', 'max', 'min', 'size'])
        print(aggregate_repetitions)
        # Ensure that the desired parameters are averaged over when
        # computing the error bars
        df_keys_remaining = aggregate_repetitions.columns.get_level_values(0)
        assert set.intersection(
            set(df_keys_remaining),
            set(parameters_explored)) == parameters_to_average
        # Plot false positive rate
        axs[axs_i].scatter(
            aggregate_repetitions[aggregate_repetitions[
                'incorrect_target_rate']['mean'] > 0].index,
            aggregate_repetitions['incorrect_target_rate']['mean'][
                    aggregate_repetitions['incorrect_target_rate']['mean'] > 0]
        )
        quantile = 0.95
        quantiles_top = []
        quantiles_bottom = []
        for alpha in alpha_c_range:
            repetitions = aggregate_repetitions[
                'incorrect_target_rate']['size'][alpha]
            data_fpr = np.random.binomial(
                    N_interest, alpha, size=(repetitions, 100000)) / N_interest
            data_fpr_rep = data_fpr.mean(axis=0)
            quantiles_top.append(
                np.quantile(data_fpr_rep, quantile))
            quantiles_bottom.append(
                np.quantile(data_fpr_rep, 1 - quantile))
        axs[axs_i].plot(
            alpha_c_range,
            quantiles_top,
            alpha_c_range,
            quantiles_bottom,
            '-',
            color='tab:blue')
        axs[axs_i].plot(
            aggregate_repetitions.index,
            aggregate_repetitions.index,
            '--',
            color='tab:gray'
        )
        # Set axes properties
        axs[axs_i].set_xlabel(r'$\alpha_{max}$', horizontalalignment='right', x=1.0)
        axs[axs_i].set_ylabel(r'$False\ positive\ rate$')  # , horizontalalignment='right', y=1.0)
        axs[axs_i].set_xscale('log')
        axs[axs_i].set_yscale('log')
        axs[axs_i].yaxis.set_ticks_position('both')
        # axs[axs_i].set_xlim(right=1)
        # axs[axs_i].set_ylim(top=1)

        axs_i += 1

    return fig, axs


def plot_precision_recall_vs_alpha():
    # Plot precision and recall vs. alpha
    # Subplots vertical: number of samples
    # Subplots horizontal: precision, recall
    # Legend: number of nodes

    # Define (sub)plot(s)
    fig, axs = plt.subplots(len(samples_n_range), 2, sharex=True, sharey=True)

    # Select data of interest
    df_interest = df[parameters_explored + ['precision', 'recall']]
    # Convert remaining DataFrame to float type for averaging
    df_interest = df_interest.astype(float)
    # Choose which of the explored parameters to average over when computing
    # the error bars
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
        df_samples = df_interest.loc[
            df_interest['samples_n'] == samples_n].drop('samples_n', 1)
        #
        for nodes_n in nodes_n_range:
            # Select dataframe entries(i.e runs) with the same number of nodes
            df_nodes = df_samples.loc[df_samples['nodes_n'] == nodes_n].drop(
                'nodes_n', 1)
            # Group by p_value and then compute mean and std
            aggregate_repetitions = df_nodes.groupby('p_value').agg(
                ['mean', 'std'])
            # Ensure that the desired parameters are averaged over when
            # computing the error bars
            df_keys_remaining = (
                aggregate_repetitions.columns.get_level_values(0))
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
        axs[axs_i][0].set_ylabel(r'$Precision$')  # , horizontalalignment='right', y=1.0)
        axs[axs_i][0].yaxis.set_ticks_position('both')
        axs[axs_i][0].set_ylim(bottom=0)

        axs[axs_i][1].set_xlabel(
            r'$\alpha_{{max}}$',
            horizontalalignment='right',
            x=1.0)
        axs[axs_i][1].set_xscale('log')
        axs[axs_i][1].set_ylabel(r'$Recall$')  # , horizontalalignment='right', y=1.0)
        axs[axs_i][1].set_ylim(bottom=0)
        # Add legend
        legend = [r'$N={}$'.format(nodes_n) for nodes_n in nodes_n_range]
        axs[axs_i][0].legend(labels=legend)
        axs[axs_i][1].legend(labels=legend)

        axs_i += 1

    return fig, axs


# region OLD Precision/Recall scatter plot (aggregate, only show mean)
# # Error bars: max and min values
# # Probably better than Sensitivity/Specificity in our case, because the number
# # of negatives outweight the number of positives, see reference:
# # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4349800/
# # Set color pallette and/or line styles
# mpl.rcParams['axes.prop_cycle'] = cycler(color='mybkgrc')
# arrow_color = 'b'
# fig = plt.figure()
# axs = plt.gca()
# axs = np.array([axs])
# # Select data of interest
# df_interest = df[parameters_explored + ['precision', 'recall']]
# # Convert remaining DataFrame to float type for averaging 
# df_interest = df_interest.astype(float)
# # Choose which of the explored parameters to average over when computing the
# # error bars
# parameters_to_average = {'nodes_n', 'repetition_i'}
# # Arrow coordinates
# arrow_coord = np.zeros(shape=(len(alpha_c_range), len(samples_n_range), 2))
# arrow_i = 0
# for alpha_c in alpha_c_range:
#     # Select dataframe entries(i.e runs) with the same alpha_c
#     df_alpha_c = df_interest.loc[df_interest['p_value'] == alpha_c].drop(
#         'p_value', 1)
#     # Group by number of samples and then compute mean and std
#     aggregate_repetitions = df_alpha_c.groupby('samples_n').agg(
#         ['mean', 'min', 'max'])
#     # Ensure that the desired parameters are averaged over when computing
#     # the error bars
#     df_keys_remaining = aggregate_repetitions.columns.get_level_values(0)
#     assert set.intersection(
#         set(df_keys_remaining),
#         set(parameters_explored)) == parameters_to_average
#     # Plot precision vs. recall
#     axs[0].errorbar(
#         aggregate_repetitions['recall']['mean'],
#         aggregate_repetitions['precision']['mean'],
#         # Add horizontal (asymmetric) error bars
#         xerr=[
#             (aggregate_repetitions['recall']['mean']
#              - aggregate_repetitions['recall']['min']),
#             (aggregate_repetitions['recall']['max']
#              - aggregate_repetitions['recall']['mean'])
#             ],
#         # Add vertical (asymmetric) error bars
#         yerr=[
#             (aggregate_repetitions['precision']['mean']
#              - aggregate_repetitions['precision']['min']),
#             (aggregate_repetitions['precision']['max']
#              - aggregate_repetitions['precision']['mean'])
#             ],
#         fmt='o'
#     )
#     # arrow_coord[arrow_i, :, 0] = aggregate_repetitions['recall']['mean']
#     arrow_coord[arrow_i, 0:len(aggregate_repetitions), 0] = (
#         aggregate_repetitions['recall']['mean'])
#     # arrow_coord[arrow_i, :, 1] = aggregate_repetitions['recall']['mean']
#     arrow_coord[arrow_i, 0:len(aggregate_repetitions), 1] = (
#         aggregate_repetitions['precision']['mean'])
#     arrow_i += 1
# for i in range(len(samples_n_range) - 1):
#     arrow_width = 0.01
#     arrow_head_length = 0.02
#     plt.arrow(
#         arrow_coord[0, i, 0],
#         arrow_coord[0, i, 1],
#         arrow_coord[-1, i, 0] - arrow_coord[0, i, 0],
#         arrow_coord[-1, i, 1] - arrow_coord[0, i, 1],
#         alpha=0.3,
#         width=arrow_width,
#         head_length=arrow_head_length,
#         length_includes_head=True,
#         color=arrow_color
#         )
# # Add legend
# legend = [
#     r'$\alpha_{{max}}={}$'.format(alpha_c) for alpha_c in alpha_c_range]
# axs[0].legend(labels=legend, loc=(0.5, 0.3))
# # Set axes properties
# axs[0].set_xlabel(r'$Recall$', horizontalalignment='right', x=1.0)
# axs[0].set_ylabel(r'$Precision$')#, horizontalalignment='right', y=1.0)
# #axs[0].set_xscale('log')
# #axs[0].set_yscale('log')
# axs[0].xaxis.set_ticks_position('both')
# axs[0].yaxis.set_ticks_position('both')
# axs[0].set_xlim([-0.01, 1.01])
# #axs[0].set_ylim([0.4, 1])
# #axs[0].set_xticks([0.3, 0.5, 0.95])
# #axs[0].set_xticks(np.arange(0, 1.1, 0.1), minor=True)
# #axs[0].set_yticks([0, 0.5, 1])
# #axs[0].set_yticks(np.arange(0,1.1,0.1), minor=True)
# #plt.xticks([0.15,0.3,0.5,0.95,0.975], ('100 samples', '0.3', '1000 samples', '0.95', '10000 samples'), rotation=30)
# axs[0].axvline(0.3, linestyle='--', color='k')
# axs[0].axvline(0.95, linestyle='--', color='k')
# # Add text
# #plt.text(0.1, 0.33, r'$T=100$', fontsize=10)
# #plt.text(0.57, 0.33, r'$T=1000$', fontsize=10)
# #plt.text(0.965, 0.4, r'$T=1000$', fontsize=10, rotation=90)
# plt.text(0.12, plt.ylim()[0] + 0.08, r'$T=100$', fontsize=10)
# plt.text(0.57, plt.ylim()[0] + 0.08, r'$T=1000$', fontsize=10)
# plt.text(0.97, plt.ylim()[0] + 0.21, r'$T=10000$', fontsize=10, rotation=90)
# # Reset color pallette and/or line styles to default
# mpl.rcParams['axes.prop_cycle'] = cycler_default
# fig_list.append(fig)
# axs_list.append(axs)
#endregion

# region OLD # Receiver operating characteristic (linear scale)
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
#endregion

# region OLD Receiver operating characteristic (log scale)
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
#endregion


def plot_precision_vs_recall_subplots_N_scatter():
    # Precision/Recall scatter plot
    # Subplots: network size

    # Probably better than Sensitivity/Specificity in our case, because the
    # number of negatives outweight the number of positives, see reference:
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4349800/

    # Set color pallette and/or line styles
    mpl.rcParams['axes.prop_cycle'] = cycler(color='mybkgrc')

    # Define (sub)plot(s)
    fig, axs = plt.subplots(len(nodes_n_range), 1, sharex=True, sharey=True)

    # Select data of interest
    df_interest = df[parameters_explored + ['precision', 'recall']]
    # Convert remaining DataFrame to float type for averaging
    df_interest = df_interest.astype(float)
    # Choose which of the explored parameters to average over when computing
    # the error bars
    parameters_to_average = {'repetition_i'}
    # If axs is not a list (i.e. it only contains one subplot), turn it into a
    # list with one element
    if not hasattr(axs, "__len__"):
        axs = np.array([axs])
    axs_i = 0
    for nodes_n in nodes_n_range:
        # Set subplot title
        axs[axs_i].set_title(r'$N={}$'.format(nodes_n))
        # Select dataframe entries(i.e runs) with the same number of nodes
        df_nodes = df_interest.loc[df_interest['nodes_n'] == nodes_n].drop('nodes_n', 1)
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
        axs[axs_i].set_xlabel(r'$Recall$', horizontalalignment='right', x=1.0)
        axs[axs_i].set_ylabel(r'$Precision$')#, horizontalalignment='right', y=1.0)
        #axs[axs_i].set_xscale('log')
        axs[axs_i].yaxis.set_ticks_position('both')
        axs[axs_i].set_xlim([0, 1])
        axs[axs_i].set_ylim([0, 1])
        # Add legend
        legend = [
         r'$T={}$'.format(samples_n) for samples_n in samples_n_range]
        axs[axs_i].legend(labels=legend)

        axs_i += 1
    # Reset color pallette and/or line styles to default
    mpl.rcParams['axes.prop_cycle'] = cycler_default

    return fig, axs


def plot_precision_vs_recall_subplots_T_scatter():
    # Precision/Recall scatter plot
    # Subplots: sample size

    # Probably better than Sensitivity/Specificity in our case, because the
    # number of negatives outweight the number of positives, see reference:
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4349800/

    # Set color pallette and/or line styles
    mpl.rcParams['axes.prop_cycle'] = cycler(color='mybkgrc')

    # Define (sub)plot(s)
    fig, axs = plt.subplots(len(samples_n_range), 1, sharex=True, sharey=True)

    # Select data of interest
    df_interest = df[parameters_explored + ['precision', 'recall']]
    # Convert remaining DataFrame to float type for averaging
    df_interest = df_interest.astype(float)
    # Choose which of the explored parameters to average over when computing
    # the error bars
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
        df_samples = df_interest.loc[
            df_interest['samples_n'] == samples_n].drop('samples_n', 1)
        for alpha_c in alpha_c_range:
            # Select dataframe entries(i.e runs) with the same alpha_c
            df_alpha_c = df_samples.loc[df_samples['p_value'] == alpha_c].drop(
                'p_value', 1)
            # Ensure that the desired parameters are averaged over when
            # computing the error bars
            df_keys_remaining = df_alpha_c.columns
            assert set.intersection(
                set(df_keys_remaining),
                set(parameters_explored)) == parameters_to_average

            # Plot precision vs. recall
            axs[axs_i].scatter(
                df_alpha_c['recall'],
                df_alpha_c['precision']
            )
            # axs[axs_i].errorbar(
            #     aggregate_repetitions['recall']['mean'],
            #     aggregate_repetitions['precision']['mean'],
            #     # Add horizontal (symmetric) error bars
            #     xerr=aggregate_repetitions['recall']['std'],
            #     # Add vertical (symmetric) error bars
            #     yerr=aggregate_repetitions['precision']['std'],
            #     fmt='o'
            # )
        # Set axes properties
        axs[axs_i].set_xlabel(r'$Recall$', horizontalalignment='right', x=1.0)
        axs[axs_i].set_ylabel(r'$Precision$')  # , horizontalalignment='right', y=1.0)
        # axs[axs_i].set_xscale('log')
        # axs[axs_i].set_yscale('log')
        axs[axs_i].yaxis.set_ticks_position('both')
        # axs[axs_i].set_xlim([0, 1])
        # axs[axs_i].set_ylim([0, 1])
        # Add legend
        legend = [
         r'$\alpha_{{max}}={}$'.format(alpha_c) for alpha_c in alpha_c_range]
        axs[axs_i].legend(labels=legend)

        axs_i += 1
    # Reset color pallette and/or line styles to default
    mpl.rcParams['axes.prop_cycle'] = cycler_default

    return fig, axs


def plot_precision_vs_recall_subplots_T_scatter_aggregate():
    # Precision/Recall scatter plot (aggregate, scatter)
    # Legend: alpha

    # Probably better than Sensitivity/Specificity in our case, because the
    # number of negatives outweight the number of positives, see reference:
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4349800/

    # Set color pallette and/or line styles
    mpl.rcParams['axes.prop_cycle'] = cycler(color='mybkgrc')
    arrow_color = 'b'

    # Define (sub)plot(s)
    fig = plt.figure()
    axs = plt.gca()
    axs = np.array([axs])
    # Select data of interest
    df_interest = df[parameters_explored + ['precision', 'recall']]
    # Convert remaining DataFrame to float type for averaging
    df_interest = df_interest.astype(float)
    # Choose which of the explored parameters to average over when computing
    # the error bars
    parameters_to_average = {'nodes_n', 'samples_n', 'repetition_i'}
    # Initialise arrow coordinates array
    arrow_coord = np.zeros(shape=(len(alpha_c_range), len(samples_n_range), 2))
    arrow_i = 0
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

        # Update arrow coordinates array
        aggregate_repetitions = df_alpha_c.groupby('samples_n').agg(['mean'])
        arrow_coord[arrow_i, 0:len(aggregate_repetitions), 0] = (
            aggregate_repetitions['recall']['mean'])
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
    axs[0].set_ylabel(r'$Precision$')  # , horizontalalignment='right', y=1.0)
    # axs[0].set_xscale('log')
    # axs[0].set_yscale('log')
    axs[0].yaxis.set_ticks_position('both')
    axs[0].set_xlim([-0.01, 1.01])
    # axs[0].set_ylim([0.4, 1])
    # axs[0].set_xticks([0.3, 0.5, 0.95])
    # axs[0].set_xticks(np.arange(0, 1.1, 0.1), minor=True)
    # axs[0].set_yticks([0, 0.5, 1])
    # axs[0].set_yticks(np.arange(0,1.1,0.1), minor=True)
    # plt.xticks([0.15,0.3,0.5,0.95,0.975], ('100 samples', '0.3', '1000 samples', '0.95', '10000 samples'), rotation=30)
    axs[0].axvline(0.3, linestyle='--', color='k')  # use 0.45 for CLM and 0.3 for VAR
    axs[0].axvline(0.95, linestyle='--', color='k')  # use 0.91 for CLM and 0.95 for VAR
    # Add text
    # plt.text(0.1, 0.33, r'$T=100$', fontsize=10)
    # plt.text(0.57, 0.33, r'$T=1000$', fontsize=10)
    # plt.text(0.965, 0.4, r'$T=1000$', fontsize=10, rotation=90)
    plt.text(0.12, plt.ylim()[0] + 0.08, r'$T=100$', fontsize=10)
    plt.text(0.57, plt.ylim()[0] + 0.08, r'$T=1000$', fontsize=10)
    plt.text(0.97, plt.ylim()[0] + 0.21, r'$T=10000$', fontsize=10, rotation=90)
    # Reset color pallette and/or line styles to default
    mpl.rcParams['axes.prop_cycle'] = cycler_default

    return fig, axs


def plot_delay_error_mean_vs_T():
    # Plot delay error mean vs. samples_number
    # Subplots: nerwork size

    fig, axs = plt.subplots(len(alpha_c_range), 1, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + ['delay_error_mean']]
    # Convert remaining DataFrame to float type for averaging
    df_interest = df_interest.astype(float)
    # Choose which of the explored parameters to average over when computing
    # the error bars
    parameters_to_average = {'repetition_i'}
    # If axs is not a list (i.e. it only contains one subplot), turn it into a
    # list with one element
    if not hasattr(axs, "__len__"):
        axs = np.array([axs])
    axs_i = 0
    for alpha_c in alpha_c_range:
        # Set subplot title
        axs[axs_i].set_title(r'$\alpha_{{max}}={}$'.format(alpha_c))
        # Select dataframe entries(i.e runs) with the same number of nodes
        df_alpha_c = df_interest.loc[
            df_interest['p_value'] == alpha_c].drop('p_value', 1)
        for nodes_n in nodes_n_range:
            # Select dataframe entries(i.e runs) with the same number of nodes
            df_nodes = df_alpha_c.loc[df_alpha_c['nodes_n'] == nodes_n].drop(
                'nodes_n', 1)
            # Group by number of samples and then compute mean and std
            aggregate_repetitions = df_nodes.groupby('samples_n').agg(
                ['mean', 'std'])
            # Ensure that the desired parameters are averaged over when
            # computing the error bars
            df_keys_remaining = (
                aggregate_repetitions.columns.get_level_values(0))
            assert set.intersection(
                set(df_keys_remaining),
                set(parameters_explored)) == parameters_to_average

            # Plot delay error
            axs[axs_i].errorbar(
                aggregate_repetitions.index,
                aggregate_repetitions['delay_error_mean']['mean'],
                # Add vertical (symmetric) error bars
                yerr=aggregate_repetitions['delay_error_mean']['std'],
                fmt='-o')
            # Set axes properties
            axs[axs_i].set_xlabel(r'$T$', horizontalalignment='right', x=1.0)
            axs[axs_i].set_ylabel(r'$Absolute\ error$')  # , horizontalalignment='right', y=1.0)
            axs[axs_i].set_xscale('log')
            # axs[axs_i].set_yscale('log')
            axs[axs_i].yaxis.set_ticks_position('both')
            # axs[axs_i].set_xlim([95, 11000])
            # axs[axs_i].set_ylim(bottom=0)
            # Add legend
            legend = [r'$N={}$'.format(nodes_n) for nodes_n in nodes_n_range]
            axs[axs_i].legend(labels=legend)

        axs_i += 1

    return fig, axs


def plot_delay_error_mean_vs_T_relative():
    # Plot delay error mean vs. samples_number
    # Subplots: nerwork size

    fig, axs = plt.subplots(len(alpha_c_range), 1, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + ['delay_error_mean_normalised']]
    # Convert remaining DataFrame to float type for averaging
    df_interest = df_interest.astype(float)
    # Choose which of the explored parameters to average over when computing
    # the error bars
    parameters_to_average = {'repetition_i'}
    # If axs is not a list (i.e. it only contains one subplot), turn it into a
    # list with one element
    if not hasattr(axs, "__len__"):
        axs = np.array([axs])
    axs_i = 0
    for alpha_c in alpha_c_range:
        # Set subplot title
        axs[axs_i].set_title(r'$\alpha_{{max}}={}$'.format(alpha_c))
        # Select dataframe entries(i.e runs) with the same number of nodes
        df_alpha_c = df_interest.loc[
            df_interest['p_value'] == alpha_c].drop('p_value', 1)
        for nodes_n in nodes_n_range:
            # Select dataframe entries(i.e runs) with the same number of nodes
            df_nodes = df_alpha_c.loc[df_alpha_c['nodes_n'] == nodes_n].drop(
                'nodes_n', 1)
            # Group by number of samples and then compute mean and std
            aggregate_repetitions = df_nodes.groupby('samples_n').agg(
                ['mean', 'std'])
            # Ensure that the desired parameters are averaged over when
            # computing the error bars
            df_keys_remaining = (
                aggregate_repetitions.columns.get_level_values(0))
            assert set.intersection(
                set(df_keys_remaining),
                set(parameters_explored)) == parameters_to_average

            # Plot delay error
            axs[axs_i].errorbar(
                aggregate_repetitions.index,
                aggregate_repetitions['delay_error_mean_normalised']['mean'],
                # Add vertical (symmetric) error bars
                yerr=aggregate_repetitions['delay_error_mean_normalised']['std'],
                fmt='-o')
            # Set axes properties
            axs[axs_i].set_xlabel(r'$T$', horizontalalignment='right', x=1.0)
            #axs[axs_i].set_ylabel(r'$Relative\ error$')  # , horizontalalignment='right', y=1.0)
            axs[axs_i].set_xscale('log')
            # axs[axs_i].set_yscale('log')
            axs[axs_i].yaxis.set_ticks_position('both')
            # axs[axs_i].set_xlim([95, 11000])
            # axs[axs_i].set_ylim(bottom=0)
            # Add legend
            legend = [r'$N={}$'.format(nodes_n) for nodes_n in nodes_n_range]
            axs[axs_i].legend(labels=legend)

        axs_i += 1

    return fig, axs


def plot_delay_error_mean_vs_T_alpha_interest():
    # Plot delay error mean vs. samples_number
    # Subplots: nerwork size

    fig, axs = plt.subplots(1, 1)
    # Select data of interest
    df_interest = df[parameters_explored + ['delay_error_mean']]
    # Convert remaining DataFrame to float type for averaging
    df_interest = df_interest.astype(float)
    # Choose which of the explored parameters to average over when computing
    # the error bars
    parameters_to_average = {'repetition_i'}
    # If axs is not a list (i.e. it only contains one subplot), turn it into a
    # list with one element
    if not hasattr(axs, "__len__"):
        axs = np.array([axs])
    axs_i = 0
    for alpha_c in [alpha_interest]:
        # Set subplot title
        # axs[axs_i].set_title(r'$\alpha_{{max}}={}$'.format(alpha_c))
        # Select dataframe entries(i.e runs) with the same number of nodes
        df_alpha_c = df_interest.loc[
            df_interest['p_value'] == alpha_c].drop('p_value', 1)
        for nodes_n in nodes_n_range:
            # Select dataframe entries(i.e runs) with the same number of nodes
            df_nodes = df_alpha_c.loc[df_alpha_c['nodes_n'] == nodes_n].drop(
                'nodes_n', 1)
            # Group by number of samples and then compute mean and std
            aggregate_repetitions = df_nodes.groupby('samples_n').agg(
                ['mean', 'std', 'max', 'min'])
            # Ensure that the desired parameters are averaged over when
            # computing the error bars
            df_keys_remaining = (
                aggregate_repetitions.columns.get_level_values(0))
            assert set.intersection(
                set(df_keys_remaining),
                set(parameters_explored)) == parameters_to_average

            # Plot delay error
            axs[axs_i].errorbar(
                aggregate_repetitions.index,
                aggregate_repetitions['delay_error_mean']['mean'],
                # Add vertical (symmetric) error bars
                yerr=[
                    0*aggregate_repetitions['delay_error_mean']['std'],
                    aggregate_repetitions['delay_error_mean']['std']],
                fmt='-o')
            # Set axes properties
            axs[axs_i].set_xlabel(r'$T$', horizontalalignment='right', x=1.0)
            axs[axs_i].set_ylabel(r'$Absolute\ error$')  # , horizontalalignment='right', y=1.0)
            axs[axs_i].set_xscale('log')
            # axs[axs_i].set_yscale('log')
            axs[axs_i].yaxis.set_ticks_position('both')
            # axs[axs_i].set_xlim([95, 11000])
            # axs[axs_i].set_ylim(bottom=0)
            # Add legend
            legend = [r'$N={}$'.format(nodes_n) for nodes_n in nodes_n_range]
            axs[axs_i].legend(labels=legend)

        axs_i += 1

    return fig, axs


def plot_delay_error_mean_vs_T_relative_alpha_interest():
    # Plot delay error mean vs. samples_number
    # Subplots: nerwork size

    fig, axs = plt.subplots(1, 1)
    # Select data of interest
    df_interest = df[parameters_explored + ['delay_error_mean_normalised']]
    # Convert remaining DataFrame to float type for averaging
    df_interest = df_interest.astype(float)
    # Choose which of the explored parameters to average over when computing
    # the error bars
    parameters_to_average = {'repetition_i'}
    # If axs is not a list (i.e. it only contains one subplot), turn it into a
    # list with one element
    if not hasattr(axs, "__len__"):
        axs = np.array([axs])
    axs_i = 0
    for alpha_c in [alpha_interest]:
        # Set subplot title
        # axs[axs_i].set_title(r'$\alpha_{{max}}={}$'.format(alpha_c))
        # Select dataframe entries(i.e runs) with the same number of nodes
        df_alpha_c = df_interest.loc[
            df_interest['p_value'] == alpha_c].drop('p_value', 1)
        for nodes_n in nodes_n_range:
            # Select dataframe entries(i.e runs) with the same number of nodes
            df_nodes = df_alpha_c.loc[df_alpha_c['nodes_n'] == nodes_n].drop(
                'nodes_n', 1)
            # Group by number of samples and then compute mean and std
            aggregate_repetitions = df_nodes.groupby('samples_n').agg(
                ['mean', 'std', 'max', 'min'])
            # Ensure that the desired parameters are averaged over when
            # computing the error bars
            df_keys_remaining = (
                aggregate_repetitions.columns.get_level_values(0))
            assert set.intersection(
                set(df_keys_remaining),
                set(parameters_explored)) == parameters_to_average

            # Plot delay error
            axs[axs_i].errorbar(
                aggregate_repetitions.index,
                aggregate_repetitions['delay_error_mean_normalised']['mean'],
                # Add vertical (symmetric) error bars
                yerr=[
                    0*aggregate_repetitions['delay_error_mean_normalised']['std'],
                    aggregate_repetitions['delay_error_mean_normalised']['std']],
                fmt='-o')
            # Set axes properties
            axs[axs_i].set_xlabel(r'$T$', horizontalalignment='right', x=1.0)
            axs[axs_i].set_ylabel(r'$Lag\ error$')  # , horizontalalignment='right', y=1.0)
            axs[axs_i].set_xscale('log')
            # axs[axs_i].set_yscale('log')
            axs[axs_i].yaxis.set_ticks_position('both')
            # axs[axs_i].set_xlim([95, 11000])
            # axs[axs_i].set_ylim(bottom=0)
            # axs[axs_i].set_ylim(bottom=-0.015, top=0.64)
            axs[axs_i].set_ylim(top=0.3)
            # Add legend
            legend = [r'$N={}$'.format(nodes_n) for nodes_n in nodes_n_range]
            axs[axs_i].legend(labels=legend)#, loc=(0.72, 0.54))

        axs_i += 1

    return fig, axs


def plot_omnibus_TE_empirical_histogram_alpha_interest():
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
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}
    # If axs is not a list (i.e. it only contains one subplot), turn it into a
    # list with one element
    if not hasattr(axs[0], "__len__"):
        axs = np.array([axs])
    for (axs_row, samples_n) in enumerate(samples_n_range):
        # Select dataframe entries(i.e runs) with the same number of samples
        df_samples = df_interest.loc[
            df_interest['samples_n'] == samples_n].drop('samples_n', 1)
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
            axs[axs_row, axs_col].set_xlabel(
                r'$Omnibus\ TE\ empirical$', horizontalalignment='right', x=1.0)
            # axs[axs_row, axs_col].set_xscale('log')
            # axs[axs_row, axs_col].set_ylabel(' ')#, horizontalalignment='right', y=1.0)
            axs[axs_row, axs_col].yaxis.set_ticks_position('both')
            axs[axs_row, axs_col].set_ylim(bottom=0)
    return fig, axs


def plot_omnibus_TE_empirical_histogram_alpha_interest_T_interest():
    # Plot TE_omnibus_empirical histogram (only T_interest)

    fig = plt.figure()
    axs = plt.gca()
    axs = np.array([axs])
    # Select data of interest
    df_interest = df[parameters_explored + ['TE_omnibus_empirical']]
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Set subplot title
    axs[0].set_title(r'$T={}$'.format(T_interest))
    if not np.all(np.isnan(df_interest.TE_omnibus_empirical.tolist()[0])):
        TE_omnibus_empirical = np.concatenate(
            df_interest.TE_omnibus_empirical.tolist()
            )
        # Remove NaN values
        TE_omnibus_empirical = TE_omnibus_empirical[
            ~np.isnan(TE_omnibus_empirical)]
        # Plot omnibus TE histogram
        axs[0].hist(TE_omnibus_empirical)
    # Set axes properties
    axs[0].set_xlabel(r'$Omnibus\ TE\ empirical$', horizontalalignment='right', x=1.0)
    axs[0].yaxis.set_ticks_position('both')
    axs[0].set_ylim(bottom=0)

    return fig, axs


def plot_omnibus_TE_empirical_vs_theoretical_causal_vars_alpha_interest():
    # Scatter TE_omnibus_empirical vs. TE_omnibus_theoretical_causal_vars
    # Subplots vertical: number of samples
    # Subplots horizontal: number of nodes

    fig, axs = plt.subplots(
        len(samples_n_range),
        len(nodes_n_range),
        sharex=True,
        sharey=True)
    fig.suptitle(r'$Omnibus\ TE\ theoretical\ (causal\ vars)\ vs.\ Omnibus\ TE\ empirical$')    
    # Select data of interest
    df_interest = df[parameters_explored + [
        'TE_omnibus_empirical', 'TE_omnibus_theoretical_causal_vars']]
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}
    # If axs is not a list (i.e. it only contains one subplot), turn it into a
    # list with one element
    if not hasattr(axs[0], "__len__"):
        axs = np.array([axs])
    for (axs_row, samples_n) in enumerate(samples_n_range):
        # Select dataframe entries(i.e runs) with the same number of samples
        df_samples = df_interest.loc[
            df_interest['samples_n'] == samples_n].drop('samples_n', 1)
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
                nodes_n))
            if not np.all(np.isnan(aggregate_repetitions.loc[nodes_n].TE_omnibus_empirical)):
                TE_omnibus_empirical = np.concatenate(
                    aggregate_repetitions.loc[nodes_n].TE_omnibus_empirical)
                TE_omnibus_theoretical_causal_vars = np.concatenate(
                    aggregate_repetitions.loc[nodes_n].TE_omnibus_theoretical_causal_vars)
                # Remove NaN values
                #TE_omnibus_empirical = TE_omnibus_empirical[~np.isnan(TE_omnibus_empirical)]
                # Scatter omnibus TE vs theoretical
                axs[axs_row, axs_col].scatter(
                    TE_omnibus_empirical,
                    TE_omnibus_theoretical_causal_vars)
                # Plot identity line
                axs[axs_row, axs_col].plot(
                    [min(TE_omnibus_theoretical_causal_vars), max(TE_omnibus_theoretical_causal_vars)],
                    [min(TE_omnibus_theoretical_causal_vars), max(TE_omnibus_theoretical_causal_vars)],
                    'g--')
            # Set axes properties
            axs[axs_row, axs_col].set_xlabel(
                r'$TE\ empirical$', horizontalalignment='right', x=1.0)
            axs[axs_row, axs_col].set_ylabel(
                r'$TE\ theoretical$')
            # axs[axs_row, axs_col].set_xscale('log')
            axs[axs_row, axs_col].yaxis.set_ticks_position('both')
            axs[axs_row, axs_col].set_xlim(left=0)
            axs[axs_row, axs_col].set_ylim(bottom=0)
    return fig, axs


def plot_omnibus_TE_empirical_vs_theoretical_inferred_vars_alpha_interest():
    # Scatter TE_omnibus_empirical vs. TE_omnibus_theoretical_inferred_vars
    # Subplots vertical: number of samples
    # Subplots horizontal: number of nodes

    fig, axs = plt.subplots(
        len(samples_n_range),
        len(nodes_n_range),
        sharex=True,
        sharey=True)
    fig.suptitle(r'$Omnibus\ TE\ theoretical\ (inferred\ vars)\ vs.\ Omnibus\ TE\ empirical$')
    # Select data of interest
    df_interest = df[parameters_explored + [
        'TE_omnibus_empirical', 'TE_omnibus_theoretical_inferred_vars']]
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}
    # If axs is not a list (i.e. it only contains one subplot), turn it into a
    # list with one element
    if not hasattr(axs[0], "__len__"):
        axs = np.array([axs])
    for (axs_row, samples_n) in enumerate(samples_n_range):
        # Select dataframe entries(i.e runs) with the same number of samples
        df_samples = df_interest.loc[
            df_interest['samples_n'] == samples_n].drop('samples_n', 1)
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
                TE_omnibus_theoretical_inferred_vars = np.concatenate(
                    aggregate_repetitions.loc[nodes_n].TE_omnibus_theoretical_inferred_vars)
                # Remove NaN values
                #TE_omnibus_empirical = TE_omnibus_empirical[~np.isnan(TE_omnibus_empirical)]
                # Scatter omnibus TE vs theoretical
                axs[axs_row, axs_col].scatter(
                    TE_omnibus_empirical,
                    TE_omnibus_theoretical_inferred_vars)
                # Plot identity line
                axs[axs_row, axs_col].plot(
                    [min(TE_omnibus_theoretical_inferred_vars), max(TE_omnibus_theoretical_inferred_vars)],
                    [min(TE_omnibus_theoretical_inferred_vars), max(TE_omnibus_theoretical_inferred_vars)],
                    'g--')
            # Set axes properties
            axs[axs_row, axs_col].set_xlabel(
                r'$TE\ empirical$', horizontalalignment='right', x=1.0)
            axs[axs_row, axs_col].set_ylabel(
                r'$TE\ theoretical$')
            # axs[axs_row, axs_col].set_xscale('log')
            axs[axs_row, axs_col].yaxis.set_ticks_position('both')
            axs[axs_row, axs_col].set_xlim(left=0)
            axs[axs_row, axs_col].set_ylim(bottom=0)
    return fig, axs


def plot_relative_error_TE_empirical_vs_theoretical_causal_vars_alpha_interest():
    # Plot absolute diff between TE empirical and TE theoretical
    # (relative to TE theoretical) vs. T
    # (aggregate over N)

    fig = plt.figure()
    axs = plt.gca()
    axs = np.array([axs])
    # Select data of interest
    df_interest = df[parameters_explored + [
        'TE_omnibus_empirical', 'TE_omnibus_theoretical_causal_vars']]
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    # Compute absolute relative error
    df_interest['TE_err_abs_relative'] = (
        np.abs(df_interest['TE_omnibus_empirical']
        - df_interest['TE_omnibus_theoretical_causal_vars'])
            / df_interest['TE_omnibus_theoretical_causal_vars'])
    # Replace np.inf with np.nan in all the arrays in the TE_err_abs_relative column
    def replace_inf(arr):
        for i, x in enumerate(arr):
            if x >= 1E308:
                arr[i] = np.nan
        return arr
    df_interest['TE_err_abs_relative'] = df_interest['TE_err_abs_relative'].apply(lambda x: replace_inf(x))
    # Compute mean error excluding nan values
    df_interest['TE_err_abs_relative'] = df_interest['TE_err_abs_relative'].apply(np.nanmean)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i', 'nodes_n'}
    # Group by samples_n and concatenate TE values lists
    aggregate_repetitions = df_interest.groupby('samples_n').agg(
        lambda x: x.tolist())
    # Ensure that the desired parameters are averaged over when computing
    # the error bars
    df_keys_remaining = aggregate_repetitions.columns.get_level_values(0)
    assert set.intersection(
        set(df_keys_remaining),
        set(parameters_explored)) == parameters_to_average
    axs[0].plot(
        aggregate_repetitions.index.astype(int).tolist(),
        [np.nanmean(aggregate_repetitions['TE_err_abs_relative'][samples_n])
            for samples_n in samples_n_range],
        marker='o')
    axs[0].scatter(
        df_interest['samples_n'].values.astype(int).tolist(),
        df_interest['TE_err_abs_relative'].values,
        alpha=0.3)
    # Set axes properties
    axs[0].set_xlabel(r'$T$', horizontalalignment='right', x=1.0)
    axs[0].set_ylabel(r'$Relative\ TE\ error$')
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].yaxis.set_ticks_position('both')

    return fig, axs


def plot_spectral_radius_vs_WS_p_scatter():
    # Plot spectral radius vs. Watts-Strogatz rewiring
    # probability (scatter error)
    # Subplots: number of samples

    # Define (sub)plot(s)
    fig, axs = plt.subplots(len(samples_n_range), 1, sharey=True)

    # Select data of interest
    df_interest = df[
        parameters_explored + ['spectral_radius']]
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    # Convert remaining DataFrame to float type for averaging
    df_interest = df_interest.astype(float)
    # Choose which of the explored parameters to average over when computing
    # the error bars
    parameters_to_average = {'repetition_i'}
    # If axs is not a list (i.e. it only contains one subplot), turn it into a
    # list with one element
    if not hasattr(axs, "__len__"):
        axs = np.array([axs])
    axs_i = 0
    for samples_n in samples_n_range:
        # Set subplot title
        # axs[axs_i].set_title(r'$T={}$'.format(samples_n))
        # Select dataframe entries(i.e runs) with the same number of samples
        df_samples = df_interest.loc[
            df_interest['samples_n'] == samples_n].drop('samples_n', 1)
        # Drop NaN values
        df_samples = df_samples.dropna()
        # Group by WS_p
        aggregate_repetitions = df_samples.groupby('WS_p').agg(
            lambda x: list(x))
        # Ensure that the desired parameters are averaged over when computing
        # the error bars
        df_keys_remaining = aggregate_repetitions.columns.get_level_values(0)
        assert set.intersection(
            set(df_keys_remaining),
            set(parameters_explored)) == parameters_to_average
        # Plot spectral radius
        axs[axs_i].plot(
            WS_p_range,
            [np.mean(aggregate_repetitions['spectral_radius'][WS_p])
                for WS_p in WS_p_range],
            marker='o'
        )
        axs[axs_i].scatter(
            df_samples['WS_p'].values.astype(float).tolist(),
            df_samples['spectral_radius'].values,
            alpha=0.3
        )
        # Set axes properties
        axs[axs_i].set_xlabel(r'$Rewiring\ probability$', horizontalalignment='right', x=1.0)
        axs[axs_i].set_ylabel(r'$\rho$')
        axs[axs_i].set_xscale('log')
        axs[axs_i].yaxis.set_ticks_position('both')
        axs[axs_i].set_ylim(bottom=0)

        axs_i += 1

    return fig, axs


def plot_performance_vs_WS_p_scatter():
    # Plot performance tests vs. Watts-Strogatz rewiring
    # probability (scatter error)
    # Subplots: number of samples

    # Define (sub)plot(s)
    fig, axs = plt.subplots(len(samples_n_range), 1, sharey=True)

    # Select data of interest
    df_interest = df[
        parameters_explored + ['precision', 'recall', 'specificity']]
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    # Convert remaining DataFrame to float type for averaging
    df_interest = df_interest.astype(float)
    # Choose which of the explored parameters to average over when computing
    # the error bars
    parameters_to_average = {'repetition_i'}
    # If axs is not a list (i.e. it only contains one subplot), turn it into a
    # list with one element
    if not hasattr(axs, "__len__"):
        axs = np.array([axs])
    axs_i = 0
    for samples_n in samples_n_range:
        # Set subplot title
        # axs[axs_i].set_title(r'$T={}$'.format(samples_n))
        # Select dataframe entries(i.e runs) with the same number of samples
        df_samples = df_interest.loc[
            df_interest['samples_n'] == samples_n].drop('samples_n', 1)
        # Drop NaN values
        df_samples = df_samples.dropna()
        # Group by WS_p
        aggregate_repetitions = df_samples.groupby('WS_p').agg(
            lambda x: list(x))
        # Ensure that the desired parameters are averaged over when computing
        # the error bars
        df_keys_remaining = aggregate_repetitions.columns.get_level_values(0)
        assert set.intersection(
            set(df_keys_remaining),
            set(parameters_explored)) == parameters_to_average
        # Plot precision
        axs[axs_i].plot(
            WS_p_range,
            [np.mean(aggregate_repetitions['precision'][WS_p])
                for WS_p in WS_p_range],
            marker='o'
        )
        axs[axs_i].scatter(
            df_samples['WS_p'].values.astype(float).tolist(),
            df_samples['precision'].values,
            alpha=0.3
        )
        # Plot recall
        axs[axs_i].plot(
            WS_p_range,
            [np.mean(aggregate_repetitions['recall'][WS_p])
             for WS_p in WS_p_range],
            marker='o'
        )
        axs[axs_i].scatter(
            df_samples['WS_p'].values.astype(float).tolist(),
            df_samples['recall'].values,
            alpha=0.3
        )
        # Plot false positive rate
        axs[axs_i].plot(
            WS_p_range,
            [np.mean(aggregate_repetitions['specificity'][WS_p])
             for WS_p in WS_p_range],
            marker='o'
        )
        axs[axs_i].scatter(
            df_samples['WS_p'].values.astype(float).tolist(),
            df_samples['specificity'].values,
            alpha=0.3
        )
        # Set axes properties
        axs[axs_i].set_xlabel(r'$Rewiring\ probability$', horizontalalignment='right', x=1.0)
        axs[axs_i].set_ylabel(r'$T={}$'.format(samples_n))  #, horizontalalignment='right', y=1.0)
        axs[axs_i].set_xscale('log')
        axs[axs_i].yaxis.set_ticks_position('both')
        axs[axs_i].set_ylim(bottom=0)
        # Add legend
        axs[axs_i].legend(
            labels=['Precision', 'Recall', 'Specificity']#,
            #loc=(0.05, 0.16)
            )

        axs_i += 1

    return fig, axs


def plot_clustering_vs_WS_p_scatter():
    # Plot average clustering vs. Watts-Strogatz rewiring prob (scatter error)
    # Subplots: number of samples

    # Define (sub)plot(s)
    fig, axs = plt.subplots(len(samples_n_range), 1, sharey=True)

    # Select data of interest
    df_interest = df[parameters_explored + [
        'clustering_original',
        'clustering_inferred']]
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    # Convert remaining DataFrame to float type for averaging
    #df_interest = df_interest.astype(float)
    # Choose which of the explored parameters to average over when computing
    # the error bars
    parameters_to_average = {'repetition_i'}
    # If axs is not a list (i.e. it only contains one subplot), turn it into a
    # list with one element
    if not hasattr(axs, "__len__"):
        axs = np.array([axs])
    axs_i = 0
    for samples_n in samples_n_range:
        # Set subplot title
        # axs[axs_i].set_title(r'$T={}$'.format(samples_n))
        # Select dataframe entries(i.e runs) with the same number of samples
        df_samples = df_interest.loc[
            df_interest['samples_n'] == samples_n].drop('samples_n', 1)
        # Drop NaN values
        df_samples = df_samples.dropna()
        # Group by WS_p
        aggregate_repetitions = df_samples.groupby('WS_p').agg(
            lambda x: list(x))
        # Ensure that the desired parameters are averaged over when computing
        # the error bars
        df_keys_remaining = aggregate_repetitions.columns.get_level_values(0)
        assert set.intersection(
            set(df_keys_remaining),
            set(parameters_explored)) == parameters_to_average
        # Plot average clustering coefficient original
        axs[axs_i].plot(
            WS_p_range,
            [np.mean(aggregate_repetitions['clustering_original'][WS_p])
                for WS_p in WS_p_range],
            marker='o'
        )
        axs[axs_i].scatter(
            df_samples['WS_p'].values.astype(float).tolist(),
            [x.mean() for x in df_samples['clustering_original'].values],
            alpha=0.3
        )
        # Plot average clustering coefficient inferred
        axs[axs_i].plot(
            WS_p_range,
            [np.mean(aggregate_repetitions['clustering_inferred'][WS_p])
             for WS_p in WS_p_range],
            marker='o'
        )
        axs[axs_i].scatter(
            df_samples['WS_p'].values.astype(float).tolist(),
            [x.mean() for x in df_samples['clustering_inferred'].values],
            alpha=0.3
        )
        # Set axes properties
        axs[axs_i].set_xlabel(r'$Rewiring\ probability$', horizontalalignment='right', x=1.0)
        axs[axs_i].set_ylabel(r'$T={}$'.format(samples_n))  #, horizontalalignment='right', y=1.0)
        axs[axs_i].set_xscale('log')
        axs[axs_i].yaxis.set_ticks_position('both')
        axs[axs_i].set_ylim(bottom=0)
        # Add legend
        axs[axs_i].legend(
            labels=['mean clustering original', 'mean clustering inferred']#,
            #loc=(0.05, 0.16)
            )

        axs_i += 1

    return fig, axs


def plot_path_lenght_vs_WS_p_scatter():
    # Plot average path lenght vs. Watts-Strogatz rewiring prob (scatter error)
    # Subplots: number of samples

    # Define (sub)plot(s)
    fig, axs = plt.subplots(len(samples_n_range), 1, sharey=True)

    # Select data of interest
    df_interest = df[parameters_explored + [
        'average_shortest_path_length_original',
        'average_shortest_path_length_inferred']]
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    # Convert remaining DataFrame to float type for averaging
    df_interest = df_interest.astype(float)
    # Choose which of the explored parameters to average over when computing
    # the error bars
    parameters_to_average = {'repetition_i'}
    # If axs is not a list (i.e. it only contains one subplot), turn it into a
    # list with one element
    if not hasattr(axs, "__len__"):
        axs = np.array([axs])
    axs_i = 0
    for samples_n in samples_n_range:
        # Set subplot title
        # axs[axs_i].set_title(r'$T={}$'.format(samples_n))
        # Select dataframe entries(i.e runs) with the same number of samples
        df_samples = df_interest.loc[
            df_interest['samples_n'] == samples_n].drop('samples_n', 1)
        # Drop NaN values
        df_samples = df_samples.dropna()
        # Group by WS_p
        aggregate_repetitions = df_samples.groupby('WS_p').agg(
            lambda x: list(x))
        # Ensure that the desired parameters are averaged over when computing
        # the error bars
        df_keys_remaining = aggregate_repetitions.columns.get_level_values(0)
        assert set.intersection(
            set(df_keys_remaining),
            set(parameters_explored)) == parameters_to_average
        # Plot average path lenght original
        axs[axs_i].plot(
            WS_p_range,
            [np.mean(aggregate_repetitions['average_shortest_path_length_original'][WS_p])
                for WS_p in WS_p_range],
            marker='o'
        )
        axs[axs_i].scatter(
            df_samples['WS_p'].values.astype(float).tolist(),
            df_samples['average_shortest_path_length_original'].values,
            alpha=0.3
        )
        # Plot average path lenght inferred
        axs[axs_i].plot(
            WS_p_range,
            [np.mean(aggregate_repetitions['average_shortest_path_length_inferred'][WS_p])
             for WS_p in WS_p_range],
            marker='o'
        )
        axs[axs_i].scatter(
            df_samples['WS_p'].values.astype(float).tolist(),
            df_samples['average_shortest_path_length_inferred'].values,
            alpha=0.3
        )
        # Set axes properties
        axs[axs_i].set_xlabel(r'$Rewiring\ probability$', horizontalalignment='right', x=1.0)
        axs[axs_i].set_ylabel(r'$T={}$'.format(samples_n))  #, horizontalalignment='right', y=1.0)
        axs[axs_i].set_xscale('log')
        axs[axs_i].yaxis.set_ticks_position('both')
        axs[axs_i].set_ylim(bottom=0)
        # Add legend
        axs[axs_i].legend(
            labels=[
                'average shortest path original',
                'average shortest path inferred'
            ]#,
            #loc=(0.05, 0.16)
            )

        axs_i += 1

    return fig, axs


def plot_SW_index_vs_WS_p_scatter(WS_k):
    # Plot average clustering vs. Watts-Strogatz rewiring prob (scatter error)
    # Subplots: number of samples

    # Define (sub)plot(s)
    fig, axs = plt.subplots(len(samples_n_range), 1, sharey=True)

    # Select data of interest
    df_interest = df[parameters_explored + [
        'clustering_original',
        'clustering_inferred',
        'average_shortest_path_length_original',
        'average_shortest_path_length_inferred']]
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    # Choose which of the explored parameters to average over when computing
    # the error bars
    parameters_to_average = {'repetition_i'}
    # If axs is not a list (i.e. it only contains one subplot), turn it into a
    # list with one element
    if not hasattr(axs, "__len__"):
        axs = np.array([axs])
    axs_i = 0
    for samples_n in samples_n_range:
        # Set subplot title
        # axs[axs_i].set_title(r'$T={}$'.format(samples_n))
        # Select dataframe entries(i.e runs) with the same number of samples
        df_samples = df_interest.loc[
            df_interest['samples_n'] == samples_n].drop('samples_n', 1)
        # Drop NaN values
        df_samples = df_samples.dropna()
        # Group by WS_p
        aggregate_repetitions = df_samples.groupby('WS_p').agg(
            lambda x: list(x))
        # Ensure that the desired parameters are averaged over when computing
        # the error bars
        df_keys_remaining = aggregate_repetitions.columns.get_level_values(0)
        assert set.intersection(
            set(df_keys_remaining),
            set(parameters_explored)) == parameters_to_average

        # Compute normalisation coefficients
        normalisation_clustering = (3 * WS_k - 6) / (4 * WS_k - 4)
        normalisation_path_length = N_interest / (2 * WS_k)

        # Plot SW index original original
        axs[axs_i].plot(
            WS_p_range,
            (np.array([np.mean(aggregate_repetitions['clustering_original'][WS_p]) for WS_p in WS_p_range]) / normalisation_clustering) / (np.array([np.mean(aggregate_repetitions['average_shortest_path_length_original'][WS_p]) for WS_p in WS_p_range]) / normalisation_path_length),
            marker='o'
        )
        axs[axs_i].scatter(
            df_samples['WS_p'].values.astype(float).tolist(),
            [x.mean() / normalisation_clustering for x in df_samples['clustering_original'].values] / (df_samples['average_shortest_path_length_original'].to_numpy() / normalisation_path_length),
            alpha=0.3
        )

        # Plot SW index original inferred
        axs[axs_i].plot(
            WS_p_range,
            (np.array([np.mean(aggregate_repetitions['clustering_inferred'][WS_p]) for WS_p in WS_p_range]) / normalisation_clustering) / (np.array([np.mean(aggregate_repetitions['average_shortest_path_length_inferred'][WS_p]) for WS_p in WS_p_range]) / normalisation_path_length),
            marker='o'
        )
        axs[axs_i].scatter(
            df_samples['WS_p'].values.astype(float).tolist(),
            [x.mean() / normalisation_clustering for x in df_samples['clustering_inferred'].values] / (df_samples['average_shortest_path_length_inferred'].to_numpy() / normalisation_path_length),
            alpha=0.3
        )

        # Set axes properties
        axs[axs_i].set_xlabel(r'$Rewiring\ probability$', horizontalalignment='right', x=1.0)
        axs[axs_i].set_ylabel(r'$T={}$'.format(samples_n))  #, horizontalalignment='right', y=1.0)
        axs[axs_i].set_xscale('log')
        axs[axs_i].yaxis.set_ticks_position('both')
        axs[axs_i].set_ylim(bottom=0)
        # Add legend
        axs[axs_i].legend(
            labels=['SW index original', 'SW index inferred']#,
            #loc=(0.05, 0.16)
            )

        axs_i += 1

    return fig, axs


# ----------------------------------------------------------------------------

# Save figures?
save_figures = True

# Choose whether to use FDR-corrected results or not
fdr = False
# Select alpha threshold to plot (in those plots where only one value is used)
alpha_interest = 0.001
N_interest = 10
T_interest = 100

# Choose folder
traj_dir = os.path.join('trajectories', '2019_04_01_11h24m17s_10nodes_100samples_10ksurr')
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
colors_default = [
    '#1f77b4',
    '#ff7f0e',
    '#2ca02c',
    '#d62728',
    '#9467bd',
    '#8c564b',
    '#e377c2',
    '#7f7f7f',
    '#bcbd22',
    '#17becf']
cycler_default = cycler('color', colors_default)

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

# Ignore non-relevant explored parameters
ignore_par = {
    'jidt_threads_n',
    'n_perm_max_stat',
    'n_perm_min_stat',
    'n_perm_max_seq',
    'n_perm_omnibus'
    }
if (len(ignore_par) > 0 and bool(
     set(ignore_par).intersection(parameters_explored))):
    print('\nWARNING: Ignoring the following explored parameters: {0}'.format(
        ignore_par))
    parameters_explored = [
        item for item in parameters_explored if item not in ignore_par]
    print('parameters_explored = : {0}\n'.format(parameters_explored))

# Get parameter ranges
nodes_n_range = np.unique(df['nodes_n']).astype(int)
samples_n_range = np.unique(df['samples_n']).astype(int)
alpha_c_range = np.unique(df['p_value']).astype(float)
#alpha_c_range = np.array([0.001, 0.05])
#WS_p_range = np.unique(df['WS_p']).astype(float)

# Initialise empty figure and axes lists
fig_list = []
axs_list = []

# ----------------------------------------------------------------------------
# Random Erdos-Renyi 
# ----------------------------------------------------------------------------
new_fig('plot_performance_vs_N_scatter')
new_fig('plot_precision_vs_N_scatter')
new_fig('plot_precision_vs_N_scatter_inset')
new_fig('plot_recall_vs_N_scatter')
# new_fig('plot_spectral_radius_vs_N_scatter')
# new_fig('plot_FPR_target_vs_alpha')
# new_fig('plot_FPR_target_vs_alpha_quantile_mean')
# new_fig('plot_FPR_target_vs_alpha_quantile_N_interest')
# new_fig('plot_FPR_target_vs_alpha_quantile_std')
# new_fig('plot_precision_recall_vs_alpha')
new_fig('plot_precision_vs_recall_subplots_T_scatter')
# new_fig('plot_precision_vs_recall_subplots_T_scatter_aggregate')
new_fig('plot_precision_vs_recall_subplots_N_scatter')
# new_fig('plot_delay_error_mean_vs_T')
new_fig('plot_delay_error_mean_vs_T_relative')
# new_fig('plot_delay_error_mean_vs_T_alpha_interest')
new_fig('plot_delay_error_mean_vs_T_relative_alpha_interest')
# new_fig('plot_omnibus_TE_empirical_histogram_alpha_interest')
# new_fig('plot_omnibus_TE_empirical_histogram_alpha_interest_T_interest')
# new_fig('plot_omnibus_TE_empirical_vs_theoretical_causal_vars_alpha_interest')
# new_fig('plot_omnibus_TE_empirical_vs_theoretical_inferred_vars_alpha_interest')
# new_fig('plot_relative_error_TE_empirical_vs_theoretical_causal_vars_alpha_interest')


# ----------------------------------------------------------------------------
# WS
# ----------------------------------------------------------------------------
# new_fig('plot_spectral_radius_vs_WS_p_scatter')
# new_fig('plot_performance_vs_WS_p_scatter')
# new_fig('plot_clustering_vs_WS_p_scatter')
# new_fig('plot_path_lenght_vs_WS_p_scatter')
# new_fig('plot_SW_index_vs_WS_p_scatter')

# ----------------------------------------------------------------------------

# Save figures to PDF file
if save_figures:
    pdf_metadata = {}
    if fdr:
        pdf_path = os.path.join(traj_dir, 'figures_fdr.pdf')
    else:
        pdf_path = os.path.join(traj_dir, 'figures.pdf')
    save_figures_to_pdf(fig_list, pdf_path)
else:
    print('WARNING: figures not saved (debug mode)')
    plt.show()