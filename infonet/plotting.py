import sys
import os
import numpy as np
from scipy.stats import binom
from scipy.interpolate import make_interp_spline, BSpline, spline, interp1d
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid.inset_locator import (
    inset_axes, InsetPosition, mark_inset)
from matplotlib.patches import FancyArrowPatch, Circle, Patch
from matplotlib.ticker import FuncFormatter
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
import matplotlib.gridspec as gridspec
from matplotlib import colors
from cycler import cycler
from itertools import tee
from collections import Counter
# from mylib_pypet import print_leaves
from collections import defaultdict
import networkx as nx
from io import StringIO, BytesIO
from flask import Flask, send_file
from scipy.io import savemat


def get_parameters_explored(first_not_explored, ignore_par):
    # Determine explore parameters
    # WARNING: this method is prone to errors, when the columns are rearranged,
    # however, in such cases the "assertion" sanity checks below will fail, so I
    # will know
    parameters_explored = df.loc[[], :first_not_explored].keys().tolist()[:-1]
    if (len(ignore_par) > 0 and bool(
        set(ignore_par).intersection(parameters_explored))):
        print('\nWARNING: Ignoring the following explored parameters: {0}'.format(
            ignore_par))
        parameters_explored = [
            item for item in parameters_explored if item not in ignore_par]
    print('parameters_explored = : {0}\n'.format(parameters_explored))
    return parameters_explored


def new_fig(function_name):
    fig, axs = globals()[function_name]()
    fig_list.append(fig)
    axs_list.append(axs)


def my_subplots(subplots_v, subplots_h, sharex=True, sharey=True):
    fig, axs = plt.subplots(
        subplots_v,
        subplots_h,
        sharex=sharex,
        sharey=sharey)
    # Avoid indexing issues if only one row or one column
    if subplots_v == 1 and subplots_h == 1:
        axs = np.array([axs])
        print('One vertical and one horizontal subplots only')
    else:
        if subplots_v == 1:
            print('One vertical subplot only')
            axs = np.array([axs])
            if subplots_h == 1:
                print('One horizontal subplot only')
                axs = np.array([axs])
                axs = np.transpose(axs)
    # Set figure size
    fig.set_figheight(3 * subplots_v)
    if subplots_h > 1:
        fig.set_figwidth(4 * subplots_h)

    return fig, axs


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
        for (fig_i, fig) in enumerate(fig_list):

            # Set figure as current figure
            plt.figure(fig.number)

            # # Set figure size
            # ndim = axs.ndim
            # if ndim == 1:
            #     fig.set_figheight(3 * len(axs))
            # if ndim > 1:
            #     fig.set_figheight(3 * len(axs[:, 0]))
            #     fig.set_figwidth(4 * len(axs[0, :]))
            
            # Set tight layout to avoid overlap between subplots
            fig.tight_layout()

            # Add a pdf note to attach metadata to a page
            # pdf.attach_note('note...')

            # Save figure to PDF page
            pdf.savefig(fig)

            # Also export single figures
            # figure_format = 'PNG'
            # export_path = os.path.join(traj_dir, '{0}.{1}'.format(fig.number, figure_format))
            # plt.savefig(export_path, format=figure_format)

            print('figure {0} saved'.format(fig_i + 1))

        # Set PDF file metadata via the PdfPages object
        d = pdf.infodict()
        for key in pdf_metadata.keys():
            d[key] = pdf_metadata.get(key, '')


def check_remaining_dimensions(df_keys_remaining, parameters_to_average):
    # Ensure that only the desired parameters are aggregated or averaged
    averaging_set = set.intersection(
        set(df_keys_remaining),
        set(parameters_explored))
    assert averaging_set == parameters_to_average, (
        "Attempting to average over {0}, which differs from the "
        "specified set {1}".format(averaging_set, parameters_to_average))


def concat_if_not_empty(x):
    # Concatenate list of arrays if list is not empty
    y = np.array([])
    if len(x) > 0:
        y = np.concatenate(x)
    return y


def triad_graphs():
    # Returns dictionary mapping triad names to triad graphs

    def abc_graph():
        g = nx.DiGraph()
        g.add_nodes_from('abc')
        return g
    triad_names = ("003", "012", "102", "021D", "021U", "021C", "111D", "111U",
                   "030T", "030C", "201", "120D", "120U", "120C", "210", "300")
    tg = dict((n, abc_graph()) for n in triad_names)
    tg['012'].add_edges_from([('a', 'b')])
    tg['102'].add_edges_from([('a', 'b'), ('b', 'a')])
    tg['102'].add_edges_from([('a', 'b'), ('b', 'a')])
    tg['021D'].add_edges_from([('b', 'a'), ('b', 'c')])
    tg['021U'].add_edges_from([('a', 'b'), ('c', 'b')])
    tg['021C'].add_edges_from([('a', 'b'), ('b', 'c')])
    tg['111D'].add_edges_from([('a', 'c'), ('c', 'a'), ('b', 'c')])
    tg['111U'].add_edges_from([('a', 'c'), ('c', 'a'), ('c', 'b')])
    tg['030T'].add_edges_from([('a', 'b'), ('c', 'b'), ('a', 'c')])
    tg['030C'].add_edges_from([('b', 'a'), ('c', 'b'), ('a', 'c')])
    tg['201'].add_edges_from([('a', 'b'), ('b', 'a'), ('a', 'c'), ('c', 'a')])
    tg['120D'].add_edges_from([('b', 'c'), ('b', 'a'), ('a', 'c'), ('c', 'a')])
    tg['120C'].add_edges_from([('a', 'b'), ('b', 'c'), ('a', 'c'), ('c', 'a')])
    tg['120U'].add_edges_from([('a', 'b'), ('c', 'b'), ('a', 'c'), ('c', 'a')])
    tg['210'].add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'b'), ('a', 'c'),
                              ('c', 'a')])
    tg['300'].add_edges_from([('a', 'b'), ('b', 'a'), ('b', 'c'), ('c', 'b'),
                              ('a', 'c'), ('c', 'a')])
    return tg

# -------------------------------------------------------------------------
# region Network Neuroscience validation paper
# -------------------------------------------------------------------------


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
    # Choose which of the explored parameters will be aggregated or averaged
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
        df_aggregate = df_nodes.groupby('samples_n').agg(
            ['mean', 'std'])
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        # Plot precision
        axs[axs_i].errorbar(
            df_aggregate.index,
            df_aggregate['precision']['mean'],
            # Add vertical (symmetric) error bars
            yerr=df_aggregate['precision']['std'],
            fmt='-o'
        )
        # Plot recall
        axs[axs_i].errorbar(
            df_aggregate.index,
            df_aggregate['recall']['mean'],
            # Add vertical (symmetric) error bars
            yerr=df_aggregate['recall']['std'],
            fmt='-o'
        )
        # Plot false positive rate
        axs[axs_i].errorbar(
            df_aggregate.index,
            df_aggregate['specificity']['mean'],
            # Add vertical (symmetric) error bars
            yerr=df_aggregate['specificity']['std'],
            fmt='-'
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

    return fig


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
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'repetition_i'}
    # If axs is not a list (i.e. it only contains one subplot), turn it into a
    # list with one element
    if not hasattr(axs, "__len__"):
        axs = np.array([axs])
    for (axs_row, samples_n) in enumerate(samples_n_range):
        # Set subplot title
        # axs[axs_row].set_title(r'$T={}$'.format(samples_n))
        # Select dataframe entries(i.e runs) with the same number of samples
        df_samples = df_interest.loc[
            df_interest['samples_n'] == samples_n].drop('samples_n', 1)
        # Group by number of nodes
        df_aggregate = df_samples.groupby('nodes_n').agg(
            lambda x: list(x))
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        # Plot curves
        curves_to_plot = ['precision', 'recall', 'specificity']
        for (curve_i, curve) in enumerate(curves_to_plot):
            axs[axs_row].plot(
                nodes_n_range,
                [np.nanmean(df_aggregate[curve][nodes_n])
                    for nodes_n in nodes_n_range],
                color=colors_default[curve_i],
                marker=markers_default[curve_i],
                )
            axs[axs_row].scatter(
                df_samples['nodes_n'].values.astype(int).tolist(),
                df_samples[curve].values,
                alpha=0.3,
                color=colors_default[curve_i],
                marker=markers_default[curve_i],
            )
        # Set axes properties
        axs[axs_row].set_xlabel(r'$Network\ size$', horizontalalignment='right', x=1.0)
        axs[axs_row].set_ylabel(r'$T={}$'.format(samples_n))  #, horizontalalignment='right', y=1.0)
        # axs[axs_row].set_xscale('log')
        axs[axs_row].yaxis.set_ticks_position('both')
        axs[axs_row].set_ylim(bottom=0)
        # Add legend
        axs[axs_row].legend(
            labels=['Precision', 'Recall', 'Specificity'],
            loc=(0.05, 0.16)
            )

    return fig


# region OLD # Plot performance tests vs. network size (max-min error bars)
# # Subplots: number of samples
# fig, axs = plt.subplots(len(samples_n_range), 1, sharey=True)
# # fig.suptitle(r'$Performance\ tests\ (subplots:\ n.\ samples)$')
# # Select data of interest
# df_interest = df[parameters_explored + ['precision', 'recall', 'specificity']]
# df_interest = df_interest.loc[df_interest['p_value'] == alpha_interest].drop('p_value', 1)
# # Convert remaining DataFrame to float type for averaging
# df_interest = df_interest.astype(float)
# # Choose which of the explored parameters will be aggregated or averaged
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
#     # Group by number of nodes and then compute mean and extrema
#     df_aggregate = df_samples.groupby('nodes_n').agg(
#         ['mean', 'max', 'min'])
#     # Ensure that only the desired parameters are aggregated or averaged
#     df_keys_remaining = df_aggregate.columns.get_level_values(0)
#     assert set.intersection(
#         set(df_keys_remaining),
#         set(parameters_explored)) == parameters_to_average
#     # Plot precision
#     axs[axs_i].errorbar(
#         df_aggregate.index,
#         df_aggregate['precision']['mean'],
#         # Add vertical (asymmetric) error bars
#         yerr=[
#             (df_aggregate['precision']['mean']
#              - df_aggregate['precision']['min']),
#             (df_aggregate['precision']['max']
#              - df_aggregate['precision']['mean'])
#             ],
#         fmt='-o'
#     )
#     # Plot recall
#     axs[axs_i].errorbar(
#         df_aggregate.index,
#         df_aggregate['recall']['mean'],
#         # Add vertical (symmetric) error bars
#         yerr=[
#             (df_aggregate['recall']['mean']
#              - df_aggregate['recall']['min']),
#             (df_aggregate['recall']['max']
#              - df_aggregate['recall']['mean'])
#             ],
#         fmt='-o'
#     )
#     # Plot false positive rate
#     axs[axs_i].errorbar(
#         df_aggregate.index,
#         df_aggregate['specificity']['mean'],
#         # Add vertical (symmetric) error bars
#         yerr=[
#             (df_aggregate['specificity']['mean']
#              - df_aggregate['specificity']['min']),
#             (df_aggregate['specificity']['max']
#              - df_aggregate['specificity']['mean'])
#             ],
#         fmt='-'
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
# # Choose which of the explored parameters will be aggregated or averaged
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
#     # Group by number of nodes and then compute mean and extrema
#     df_aggregate = df_samples.groupby('nodes_n').agg(
#         lambda x: list(x))
#     # Ensure that only the desired parameters are aggregated or averaged
#     df_keys_remaining = df_aggregate.columns.get_level_values(0)
#     assert set.intersection(
#         set(df_keys_remaining),
#         set(parameters_explored)) == parameters_to_average
#     # Plot precision
#     axs[axs_i].boxplot(
#         [df_aggregate['precision'][nodes_n]
#          for nodes_n in nodes_n_range],
#         positions=df_aggregate.index.astype(int).tolist()
#     )
#     # Plot recall
#     axs[axs_i].boxplot(
#         [df_aggregate['recall'][nodes_n]
#          for nodes_n in nodes_n_range],
#         positions=df_aggregate.index.astype(int).tolist()
#     )
#     # Plot false positive rate
#     axs[axs_i].boxplot(
#         [df_aggregate['specificity'][nodes_n]
#          for nodes_n in nodes_n_range],
#         positions=df_aggregate.index.astype(int).tolist()
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
    fig = plt.figure()
    axs = plt.gca()
    axs = np.array([axs])
    # Select data of interest
    df_interest = df[parameters_explored + ['precision']]
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    # Convert remaining DataFrame to float type for averaging
    df_interest = df_interest.astype(float)
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'repetition_i'}
    for (curve_i, samples_n) in enumerate(samples_n_range):
        # Select dataframe entries(i.e runs) with the same number of samples
        df_samples = df_interest.loc[
            df_interest['samples_n'] == samples_n].drop('samples_n', 1)
        # Group by number of nodes and then compute mean and extrema
        df_aggregate = df_samples.groupby('nodes_n').agg(
            lambda x: list(x))
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        # Plot precision
        axs[0].plot(
            df_aggregate.index.astype(int).tolist(),
            [np.nanmean(df_aggregate['precision'][nodes_n])
                for nodes_n in nodes_n_range],
            color=colors_default[curve_i],
            marker=markers_default[curve_i]
            )
        axs[0].scatter(
            df_samples['nodes_n'].values.astype(int).tolist(),
            df_samples['precision'].values,
            alpha=0.3,
            color=colors_default[curve_i],
            marker=markers_default[curve_i],
            )
    # Set axes properties
    axs[0].set_xlabel(r'$Network\ size$', horizontalalignment='right', x=1.0)
    axs[0].set_ylabel(r'$Precision$')  # , horizontalalignment='right', y=1.0)
    axs[0].yaxis.set_ticks_position('both')
    axs[0].set_ylim(bottom=0)
    # Add legend
    legend = [
        r'$T={}$'.format(samples_n) for samples_n in samples_n_range]
    axs[0].legend(labels=legend, loc=(0.05, 0.16))  #remove loc for AR

    return fig


def plot_precision_vs_N_scatter_inset():
    # Plot precision vs. network size (scatter error)
    # Subplots: none
    # Legend: samples
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
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'repetition_i'}
    for (curve_i, samples_n) in enumerate(samples_n_range):
        # Select dataframe entries(i.e runs) with the same number of samples
        df_samples = df_interest.loc[
            df_interest['samples_n'] == samples_n].drop('samples_n', 1)
        # Group by number of nodes and then compute mean and extrema
        df_aggregate = df_samples.groupby('nodes_n').agg(
            lambda x: list(x))
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        # Plot precision
        axs[0].plot(
            df_aggregate.index.astype(int).tolist(),
            [np.nanmean(df_aggregate['precision'][nodes_n])
             for nodes_n in nodes_n_range],
            color=colors_default[curve_i],
            marker=markers_default[curve_i])
        axs_inset.plot(
            df_aggregate.index.astype(int).tolist(),
            [np.nanmean(df_aggregate['precision'][nodes_n])
             for nodes_n in nodes_n_range],
            color=colors_default[curve_i],
            marker=markers_default[curve_i])
        axs[0].scatter(
            df_samples['nodes_n'].values.astype(int).tolist(),
            df_samples['precision'].values,
            alpha=0.3,
            color=colors_default[curve_i],
            marker=markers_default[curve_i])
        axs_inset.scatter(
            df_samples['nodes_n'].values.astype(int).tolist(),
            df_samples['precision'].values,
            alpha=0.3,
            color=colors_default[curve_i],
            marker=markers_default[curve_i])
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

    return fig


def plot_recall_vs_N_scatter():
    # Plot recall vs. network size (scatter error)
    # Subplots: none
    # Legend: samples

    fig = plt.figure()
    axs = plt.gca()
    axs = np.array([axs])
    # Select data of interest
    df_interest = df[parameters_explored + ['recall']]
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    # Convert remaining DataFrame to float type for averaging
    df_interest = df_interest.astype(float)
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'repetition_i'}
    for (curve_i, samples_n) in enumerate(samples_n_range):
        # Select dataframe entries(i.e runs) with the same number of samples
        df_samples = df_interest.loc[
            df_interest['samples_n'] == samples_n].drop('samples_n', 1)
        # Group by number of nodes and then compute mean and extrema
        df_aggregate = df_samples.groupby('nodes_n').agg(
            lambda x: list(x))
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        # Plot recall
        axs[0].plot(
            df_aggregate.index.astype(int).tolist(),
            [np.nanmean(df_aggregate['recall'][nodes_n])
             for nodes_n in nodes_n_range],
            color=colors_default[curve_i],
            marker=markers_default[curve_i])
        axs[0].scatter(
            df_samples['nodes_n'].values.astype(int).tolist(),
            df_samples['recall'].values,
            alpha=0.3,
            color=colors_default[curve_i],
            marker=markers_default[curve_i])
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

    return fig


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
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'repetition_i'}
    # Delete duplicate rows
    df_interest = df_interest.drop_duplicates(
        ['nodes_n', 'repetition_i'])
    # Group by number of nodes and then compute mean and extrema
    df_aggregate = df_interest.groupby('nodes_n').agg(
        lambda x: list(x))
    # Ensure that only the desired parameters are aggregated or averaged
    df_keys_remaining = df_aggregate.columns.get_level_values(0)
    check_remaining_dimensions(df_keys_remaining, parameters_to_average)
    # Plot recall
    axs[0].plot(
        df_aggregate.index.astype(int).tolist(),
        [np.nanmean(df_aggregate['spectral_radius'][nodes_n])
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

    return fig


def plot_FPR_target_vs_alpha():
    # Plot incorrect targets rate vs. critical alpha
    # Legend: number of nodes
    # Subplots: number of samples

    fig, axs = plt.subplots(len(samples_n_range), 1, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + ['incorrect_target_rate']]
    # Convert remaining DataFrame to float type for averaging
    df_interest = df_interest.astype(float)
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'repetition_i'}
    # If axs is not a list (i.e. it only contains one subplot), turn it into a
    # list with one element
    if not hasattr(axs, "__len__"):
        axs = np.array([axs])
    for (axs_row, samples_n) in enumerate(samples_n_range):
        # Set subplot title
        axs[axs_row].set_title('$T={}$'.format(samples_n.astype(int)))
        # Select dataframe entries(i.e runs) with the same number of samples
        df_samples = df_interest.loc[
            df_interest['samples_n'] == samples_n].drop('samples_n', 1)
        for (curve_i, nodes_n) in enumerate(nodes_n_range):
            # Select dataframe entries(i.e runs) with the same number of nodes
            df_nodes = df_samples.loc[df_samples['nodes_n'] == nodes_n].drop(
                'nodes_n', 1)
            # Group by p_value and then compute mean and std
            df_aggregate = df_nodes.groupby('p_value').agg(
                ['mean', 'std', 'max', 'min'])
            # Ensure that only the desired parameters are aggregated or averaged
            df_keys_remaining = df_aggregate.columns.get_level_values(0)
            check_remaining_dimensions(df_keys_remaining, parameters_to_average)
            # Plot false positive rate
            axs[axs_row].errorbar(
                df_aggregate[
                        df_aggregate['incorrect_target_rate']['mean'] > 0].index,
                df_aggregate['incorrect_target_rate']['mean'][
                        df_aggregate['incorrect_target_rate']['mean'] > 0],
                yerr=[
                    0*df_aggregate['incorrect_target_rate']['std'][
                        df_aggregate['incorrect_target_rate']['mean'] > 0],
                    df_aggregate['incorrect_target_rate']['std'][
                        df_aggregate['incorrect_target_rate']['mean'] > 0]
                    ],
                fmt='-',
                color=colors_default[curve_i],
                marker=markers_default[curve_i]
            )
        axs[axs_row].plot(
            df_aggregate.index,
            df_aggregate.index,
            '--',
            color='tab:gray'
        )
        # Set axes properties
        axs[axs_row].set_xlabel(r'$\alpha_{max}$', horizontalalignment='right', x=1.0)
        axs[axs_row].set_ylabel(r'$False\ positive\ rate$')  # , horizontalalignment='right', y=1.0)
        axs[axs_row].set_xscale('log')
        axs[axs_row].set_yscale('log')
        axs[axs_row].yaxis.set_ticks_position('both')
        axs[axs_row].set_xlim(right=1)
        axs[axs_row].set_ylim(top=1)
        # Add legend
        legend = ['Identity'] + [
            r'$N={}$'.format(nodes_n) for nodes_n in nodes_n_range]
        axs[axs_row].legend(labels=legend, loc='lower right')

    return fig


def plot_FPR_target_vs_alpha_quantile_std():
    # Plot incorrect targets rate vs. critical alpha
    # Legend: number of nodes
    # Subplots: number of samples
    fig, axs = plt.subplots(len(samples_n_range), 1, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + ['incorrect_target_rate']]
    # Convert remaining DataFrame to float type for averaging
    df_interest = df_interest.astype(float)
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'repetition_i'}
    # If axs is not a list (i.e. it only contains one subplot), turn it into a
    # list with one element
    if not hasattr(axs, "__len__"):
        axs = np.array([axs])
    for (axs_row, samples_n) in enumerate(samples_n_range):
        # Set subplot title
        axs[axs_row].set_title('$T={}$'.format(samples_n.astype(int)))
        # Select dataframe entries(i.e runs) with the same number of samples
        df_samples = df_interest.loc[
            df_interest['samples_n'] == samples_n].drop('samples_n', 1)
        for (curve_i, nodes_n) in enumerate(nodes_n_range):
            # Select dataframe entries(i.e runs) with the same number of nodes
            df_nodes = df_samples.loc[df_samples['nodes_n'] == nodes_n].drop(
                'nodes_n', 1)
            # Group by p_value and then compute mean and std
            df_aggregate = df_nodes.groupby('p_value').agg(
                ['mean', 'std', 'max', 'min', 'size'])
            # Ensure that only the desired parameters are aggregated or averaged
            df_keys_remaining = df_aggregate.columns.get_level_values(0)
            check_remaining_dimensions(df_keys_remaining, parameters_to_average)
            # Plot false positive rate
            axs[axs_row].errorbar(
                df_aggregate[
                        df_aggregate['incorrect_target_rate']['mean'] > 0].index,
                df_aggregate['incorrect_target_rate']['mean'][
                        df_aggregate['incorrect_target_rate']['mean'] > 0],
                yerr=[
                    0*df_aggregate['incorrect_target_rate']['std'][
                        df_aggregate['incorrect_target_rate']['mean'] > 0],
                    df_aggregate['incorrect_target_rate']['std'][
                        df_aggregate['incorrect_target_rate']['mean'] > 0]
                    ],
                fmt='',
                color=colors_default[curve_i],
                marker=markers_default[curve_i],
                label=r'$N={}$'.format(nodes_n)
            )
            s = 100000
            quantile = 0.95
            quantile_int = []
            for alpha in alpha_c_range:
                repetitions = df_aggregate['incorrect_target_rate']['size'][alpha]
                x = np.random.binomial(
                    nodes_n, alpha, size=(repetitions, s))
                x = x / nodes_n
                data = x.std(axis=0)
                data_sorted = np.sort(data)
                quantile_int.append(data_sorted[int(quantile * s)])
            axs[axs_row].plot(
                alpha_c_range,
                np.array(df_aggregate['incorrect_target_rate']['mean']) + quantile_int,
                'x',
                color=colors_default[curve_i],
                label='_nolegend_'
            )
        axs[axs_row].plot(
            df_aggregate.index,
            df_aggregate.index,
            '--',
            color='tab:gray',
            label='Identity'
        )
        # Set axes properties
        axs[axs_row].set_xlabel(r'$\alpha_{max}$', horizontalalignment='right', x=1.0)
        axs[axs_row].set_ylabel(r'$False\ positive\ rate$')  # , horizontalalignment='right', y=1.0)
        axs[axs_row].set_xscale('log')
        axs[axs_row].set_yscale('log')
        axs[axs_row].yaxis.set_ticks_position('both')
        # axs[axs_row].set_xlim(right=1)
        # axs[axs_row].set_ylim(top=1)
        # # Add legend
        axs[axs_row].legend(loc='lower right')

    return fig


def plot_FPR_target_vs_alpha_quantile_mean():
    # Plot incorrect targets rate vs. critical alpha
    # Legend: number of nodes
    # Subplots: number of samples

    fig, axs = plt.subplots(1, 1, sharey=True)  # len(samples_n_range)
    # Select data of interest
    df_interest = df[parameters_explored + ['incorrect_target_rate']]
    # Convert remaining DataFrame to float type for averaging
    df_interest = df_interest.astype(float)
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'repetition_i'}
    # If axs is not a list (i.e. it only contains one subplot), turn it into a
    # list with one element
    if not hasattr(axs, "__len__"):
        axs = np.array([axs])
    for (axs_row, samples_n) in enumerate(samples_n_range[-1:]):
        # Set subplot title
        # axs[axs_row].set_title('$T={}$'.format(samples_n.astype(int)))
        # Select dataframe entries(i.e runs) with the same number of samples
        df_samples = df_interest.loc[
            df_interest['samples_n'] == samples_n].drop('samples_n', 1)
        for (curve_i, nodes_n) in enumerate(nodes_n_range):
            # Select dataframe entries(i.e runs) with the same number of nodes
            df_nodes = df_samples.loc[df_samples['nodes_n'] == nodes_n].drop(
                'nodes_n', 1)
            # Group by p_value and then compute mean and std
            df_aggregate = df_nodes.groupby('p_value').agg(
                ['mean', 'std', 'max', 'min', 'size'])
            # Ensure that only the desired parameters are aggregated or averaged
            df_keys_remaining = df_aggregate.columns.get_level_values(0)
            check_remaining_dimensions(df_keys_remaining, parameters_to_average)
            # Plot false positive rate
            axs[axs_row].plot(
                df_aggregate[df_aggregate[
                    'incorrect_target_rate']['mean'] > 0].index,
                df_aggregate['incorrect_target_rate']['mean'][
                        df_aggregate['incorrect_target_rate']['mean'] > 0],
                ms=3,
                linestyle='',
                color=colors_default[curve_i],
                marker=markers_default[curve_i],
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
            # repetitions = df_aggregate[
            #     'incorrect_target_rate']['size'][alpha_c_range[0]]
            for alpha in alpha_c_range:#alpha_more:
                repetitions = df_aggregate[
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
            axs[axs_row].plot(
                alpha_c_range,#alpha_more,
                quantiles_top,
                '_',
                alpha_c_range,#alpha_more,
                quantiles_bottom,
                '_',
                ms=10,
                color=colors_default[curve_i],
                label='_nolegend_')
        axs[axs_row].plot(
            df_aggregate.index,
            df_aggregate.index,
            '--',
            color='tab:gray',
            label='Identity')
        # Set axes properties
        axs[axs_row].set_xlabel(r'$\alpha_{max}$', horizontalalignment='right', x=1.0)
        axs[axs_row].set_ylabel(r'$False\ positive\ rate$')  # , horizontalalignment='right', y=1.0)
        axs[axs_row].set_xscale('log')
        axs[axs_row].set_yscale('log')
        #axs[axs_row].set_yscale('symlog', linthreshx=0.00001)
        axs[axs_row].yaxis.set_ticks_position('both')
        # Print legend
        axs[axs_row].legend(loc='lower right')

    return fig


def plot_FPR_target_vs_alpha_quantile_N_interest():
    # Plot incorrect targets rate vs. critical alpha
    # Subplots: number of samples
    fig, axs = plt.subplots(len(samples_n_range), 1)
    # Select data of interest
    df_interest = df[parameters_explored + ['incorrect_target_rate']]
    df_interest = df_interest.loc[
            df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    # Convert remaining DataFrame to float type for averaging
    df_interest = df_interest.astype(float)
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'repetition_i'}
    # If axs is not a list (i.e. it only contains one subplot), turn it into a
    # list with one element
    if not hasattr(axs, "__len__"):
        axs = np.array([axs])
    curve_i = 0
    for (axs_row, samples_n) in enumerate(samples_n_range):
        # Set subplot title
        axs[axs_row].set_title('$T={}$'.format(samples_n))
        # Select dataframe entries(i.e runs) with the same number of samples
        df_samples = df_interest.loc[
            df_interest['samples_n'] == samples_n].drop('samples_n', 1)
        # Group by p_value and then compute mean and std
        df_aggregate = df_samples.groupby('p_value').agg(
            ['mean', 'std', 'max', 'min', 'size'])
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        # Plot false positive rate
        axs[axs_row].scatter(
            df_aggregate[df_aggregate[
                'incorrect_target_rate']['mean'] > 0].index,
            df_aggregate['incorrect_target_rate']['mean'][
                    df_aggregate['incorrect_target_rate']['mean'] > 0],
            color=colors_default[curve_i],
            label=r'$N={}$'.format(N_interest)
        )
        quantile = 0.95
        quantiles_top = []
        quantiles_bottom = []
        for alpha in alpha_c_range:
            repetitions = df_aggregate[
                'incorrect_target_rate']['size'][alpha]
            data_fpr = np.random.binomial(
                    N_interest, alpha, size=(repetitions, 100000)) / N_interest
            data_fpr_rep = data_fpr.mean(axis=0)
            quantiles_top.append(
                np.quantile(data_fpr_rep, quantile))
            quantiles_bottom.append(
                np.quantile(data_fpr_rep, 1 - quantile))
        axs[axs_row].plot(
            alpha_c_range,
            quantiles_top,
            alpha_c_range,
            quantiles_bottom,
            '-',
            color=colors_default[curve_i],
            label='_nolegend_')
        axs[axs_row].plot(
            df_aggregate.index,
            df_aggregate.index,
            '--',
            color='tab:gray',
            label='Identity'
        )
        # Set axes properties
        axs[axs_row].set_xlabel(r'$\alpha_{max}$', horizontalalignment='right', x=1.0)
        axs[axs_row].set_ylabel(r'$False\ positive\ rate$')  # , horizontalalignment='right', y=1.0)
        axs[axs_row].set_xscale('log')
        axs[axs_row].set_yscale('log')
        axs[axs_row].yaxis.set_ticks_position('both')
        # axs[axs_row].set_xlim(right=1)
        # axs[axs_row].set_ylim(top=1)
        # Show legend
        axs[axs_row].legend()

    return fig


def plot_precision_recall_vs_alpha():
    # Plot precision and recall vs. alpha
    # Subplots vertical: number of samples
    # Subplots horizontal: precision, recall
    # Legend: number of nodes

    fig, axs = plt.subplots(len(samples_n_range), 2, sharex=True, sharey=True)

    # Select data of interest
    df_interest = df[parameters_explored + ['precision', 'recall']]
    # Convert remaining DataFrame to float type for averaging
    df_interest = df_interest.astype(float)
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'repetition_i'}
    # If axs is not a list (i.e. it only contains one subplot), turn it into a
    # list with one element
    if not hasattr(axs[0], "__len__"):
        axs = np.array([axs])
    for (axs_row, samples_n) in enumerate(samples_n_range):
        # Set subplot title
        axs[axs_row][0].set_title(r'$T={}$'.format(
            samples_n))
        axs[axs_row][1].set_title(r'$T={}$'.format(
            samples_n))
        # Select dataframe entries(i.e runs) with the same number of samples
        df_samples = df_interest.loc[
            df_interest['samples_n'] == samples_n].drop('samples_n', 1)
        #
        for (curve_i, nodes_n) in enumerate(nodes_n_range):
            # Select dataframe entries(i.e runs) with the same number of nodes
            df_nodes = df_samples.loc[df_samples['nodes_n'] == nodes_n].drop(
                'nodes_n', 1)
            # Group by p_value and then compute mean and std
            df_aggregate = df_nodes.groupby('p_value').agg(
                ['mean', 'std'])
            # Ensure that only the desired parameters are aggregated or averaged
            df_keys_remaining = (
                df_aggregate.columns.get_level_values(0))
            check_remaining_dimensions(
                df_keys_remaining,
                parameters_to_average)
            # Plot precision
            axs[axs_row][0].errorbar(
                df_aggregate.index,
                df_aggregate['precision']['mean'],
                # Add vertical (symmetric) error bars
                yerr=df_aggregate['precision']['std'],
                fmt='-',
                color=colors_default[curve_i],
                marker=markers_default[curve_i]
            )
            # Plot recall
            axs[axs_row][1].errorbar(
                df_aggregate.index,
                df_aggregate['recall']['mean'],
                # Add vertical (symmetric) error bars
                yerr=df_aggregate['recall']['std'],
                fmt='-',
                color=colors_default[curve_i],
                marker=markers_default[curve_i]
            )
        # Set axes properties
        axs[axs_row][0].set_xlabel(
            r'$\alpha_{{max}}$',
            horizontalalignment='right',
            x=1.0)
        axs[axs_row][0].set_xscale('log')
        axs[axs_row][0].set_ylabel(r'$Precision$')  # , horizontalalignment='right', y=1.0)
        axs[axs_row][0].yaxis.set_ticks_position('both')
        axs[axs_row][0].set_ylim(bottom=0)

        axs[axs_row][1].set_xlabel(
            r'$\alpha_{{max}}$',
            horizontalalignment='right',
            x=1.0)
        axs[axs_row][1].set_xscale('log')
        axs[axs_row][1].set_ylabel(r'$Recall$')  # , horizontalalignment='right', y=1.0)
        axs[axs_row][1].set_ylim(bottom=0)
        # Add legend
        legend = [r'$N={}$'.format(nodes_n) for nodes_n in nodes_n_range]
        axs[axs_row][0].legend(labels=legend)
        axs[axs_row][1].legend(labels=legend)

    return fig


# region OLD Precision/Recall scatter plot (aggregate, only show mean)
# # Error bars: max and min values
# # Probably better than Sensitivity/Specificity in our case, because the number
# # of negatives outweight the number of positives, see reference:
# # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4349800/
# arrow_color = 'b'
# fig = plt.figure()
# axs = plt.gca()
# axs = np.array([axs])
# # Select data of interest
# df_interest = df[parameters_explored + ['precision', 'recall']]
# # Convert remaining DataFrame to float type for averaging 
# df_interest = df_interest.astype(float)
# # Choose which of the explored parameters will be aggregated or averaged
# parameters_to_average = {'nodes_n', 'repetition_i'}
# # Arrow coordinates
# arrow_coord = np.zeros(shape=(len(alpha_c_range), len(samples_n_range), 2))
# arrow_i = 0
# for alpha_c in alpha_c_range:
#     # Select dataframe entries(i.e runs) with the same alpha_c
#     df_alpha_c = df_interest.loc[df_interest['p_value'] == alpha_c].drop(
#         'p_value', 1)
#     # Group by number of samples and then compute mean and std
#     df_aggregate = df_alpha_c.groupby('samples_n').agg(
#         ['mean', 'min', 'max'])
#     # Ensure that only the desired parameters are aggregated or averaged
#     df_keys_remaining = df_aggregate.columns.get_level_values(0)
#     assert set.intersection(
#         set(df_keys_remaining),
#         set(parameters_explored)) == parameters_to_average
#     # Plot precision vs. recall
#     axs[0].errorbar(
#         df_aggregate['recall']['mean'],
#         df_aggregate['precision']['mean'],
#         # Add horizontal (asymmetric) error bars
#         xerr=[
#             (df_aggregate['recall']['mean']
#              - df_aggregate['recall']['min']),
#             (df_aggregate['recall']['max']
#              - df_aggregate['recall']['mean'])
#             ],
#         # Add vertical (asymmetric) error bars
#         yerr=[
#             (df_aggregate['precision']['mean']
#              - df_aggregate['precision']['min']),
#             (df_aggregate['precision']['max']
#              - df_aggregate['precision']['mean'])
#             ],
#         fmt=''
#     )
#     # arrow_coord[arrow_i, :, 0] = df_aggregate['recall']['mean']
#     arrow_coord[arrow_i, 0:len(df_aggregate), 0] = (
#         df_aggregate['recall']['mean'])
#     # arrow_coord[arrow_i, :, 1] = df_aggregate['recall']['mean']
#     arrow_coord[arrow_i, 0:len(df_aggregate), 1] = (
#         df_aggregate['precision']['mean'])
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
# # Choose which of the explored parameters will be aggregated or averaged
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
#         df_aggregate = df_samples.groupby('p_value').agg(
#             ['mean', 'std'])
#         # Ensure that only the desired parameters are aggregated or averaged
#         df_keys_remaining = df_aggregate.columns.get_level_values(0)
#         assert set.intersection(
#             set(df_keys_remaining),
#             set(parameters_explored)) == parameters_to_average
#         # Plot true positive ratio vs False positive ratio
#         #axs[axs_i].scatter(
#         #    df_aggregate['false_pos_rate']['mean'],  #False positive rate=1-specificity
#         #    df_aggregate['recall']['mean']  #Recall=Sensitivity=True positive rate
#         #)
#         axs[axs_i].errorbar(
#             df_aggregate['false_pos_rate']['mean'],
#             df_aggregate['recall']['mean'],
#             # Add horizontal (symmetric) error bars
#             xerr=df_aggregate['false_pos_rate']['std'],
#             # Add vertical (symmetric) error bars
#             yerr=df_aggregate['recall']['std'],
#             fmt='-'
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
## Choose which of the explored parameters will be aggregated or averaged
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
#        df_aggregate = df_samples.groupby('p_value').agg(['mean', 'std'])
#        # Remove zero values on the x axis because of the log scale
#        df_aggregate = df_aggregate.loc[df_aggregate['false_pos_rate']['mean'] > 0]
#        # Ensure that only the desired parameters are aggregated or averaged
#        df_keys_remaining = df_aggregate.columns.get_level_values(0)
#        assert set.intersection(set(df_keys_remaining), set(parameters_explored)) == parameters_to_average
#        # Plot true positive ratio vs False positive ratio
#        #axs[axs_i].scatter(
#        #    df_aggregate['false_pos_rate']['mean'],  #False pos rate=1-specificity
#        #    df_aggregate['recall']['mean']  #Recall=Sensitivity
#        #)
#        axs[axs_i].errorbar(
#            df_aggregate['false_pos_rate']['mean'],
#            df_aggregate['recall']['mean'],
#            # Add horizontal (symmetric) error bars
#            xerr=df_aggregate['false_pos_rate']['std'],
#            # Add vertical (symmetric) error bars
#            yerr=df_aggregate['recall']['std'],
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

    # Define (sub)plot(s)
    fig, axs = plt.subplots(len(nodes_n_range), 1, sharex=True, sharey=True)

    # Select data of interest
    df_interest = df[parameters_explored + ['precision', 'recall']]
    # Convert remaining DataFrame to float type for averaging
    df_interest = df_interest.astype(float)
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'repetition_i'}
    # If axs is not a list (i.e. it only contains one subplot), turn it into a
    # list with one element
    if not hasattr(axs, "__len__"):
        axs = np.array([axs])
    for (axs_row, nodes_n) in enumerate(nodes_n_range):
        # Set subplot title
        axs[axs_row].set_title(r'$N={}$'.format(nodes_n))
        # Select dataframe entries(i.e runs) with the same number of nodes
        df_nodes = df_interest.loc[df_interest['nodes_n'] == nodes_n].drop('nodes_n', 1)
        for (curve_i, samples_n) in enumerate(samples_n_range):
            # Select dataframe entries(i.e runs) with the same number of samples
            df_samples = df_nodes.loc[df_nodes['samples_n'] == samples_n].drop(
                'samples_n', 1)
            # Group by number of samples and then compute mean and std
            df_aggregate = df_samples.groupby('p_value').agg(
                ['mean', 'std'])
            # Ensure that only the desired parameters are aggregated or averaged
            df_keys_remaining = df_aggregate.columns.get_level_values(0)
            check_remaining_dimensions(df_keys_remaining, parameters_to_average)
            # Plot true positive ratio vs False positive ratio
            #axs[axs_row].scatter(
            #    df_aggregate['recall']['mean'],
            #    df_aggregate['precision']['mean']
            #)
            axs[axs_row].errorbar(
                df_aggregate['recall']['mean'],
                df_aggregate['precision']['mean'],
                # Add horizontal (symmetric) error bars
                xerr=df_aggregate['recall']['std'],
                # Add vertical (symmetric) error bars
                yerr=df_aggregate['precision']['std'],
                color=colors_default[curve_i],
                marker=markers_default[curve_i]
            )
        # Set axes properties
        axs[axs_row].set_xlabel(r'$Recall$', horizontalalignment='right', x=1.0)
        axs[axs_row].set_ylabel(r'$Precision$')#, horizontalalignment='right', y=1.0)
        #axs[axs_row].set_xscale('log')
        axs[axs_row].yaxis.set_ticks_position('both')
        axs[axs_row].set_xlim([0, 1])
        axs[axs_row].set_ylim([0, 1])
        # Add legend
        legend = [
         r'$T={}$'.format(samples_n) for samples_n in samples_n_range]
        axs[axs_row].legend(labels=legend)

    return fig


def plot_precision_vs_recall_subplots_T_scatter():
    # Precision/Recall scatter plot
    # Subplots: sample size

    # Probably better than Sensitivity/Specificity in our case, because the
    # number of negatives outweight the number of positives, see reference:
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4349800/

    # Define (sub)plot(s)
    fig, axs = plt.subplots(len(samples_n_range), 1, sharex=True, sharey=True)

    # Select data of interest
    df_interest = df[parameters_explored + ['precision', 'recall']]
    # Convert remaining DataFrame to float type for averaging
    df_interest = df_interest.astype(float)
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'nodes_n', 'repetition_i'}
    # If axs is not a list (i.e. it only contains one subplot), turn it into a
    # list with one element
    if not hasattr(axs, "__len__"):
        axs = np.array([axs])
    for (axs_row, samples_n) in enumerate(samples_n_range):
        # Set subplot title
        axs[axs_row].set_title('$T={}$'.format(samples_n))
        # Select dataframe entries(i.e runs) with the same number of samples
        df_samples = df_interest.loc[
            df_interest['samples_n'] == samples_n].drop('samples_n', 1)
        for (curve_i, alpha_c) in enumerate(alpha_c_range):
            # Select dataframe entries(i.e runs) with the same alpha_c
            df_alpha_c = df_samples.loc[df_samples['p_value'] == alpha_c].drop(
                'p_value', 1)
            # Ensure that only the desired parameters are aggregated or averaged
            df_keys_remaining = df_alpha_c.columns
            check_remaining_dimensions(df_keys_remaining, parameters_to_average)
            # Plot precision vs. recall
            axs[axs_row].scatter(
                df_alpha_c['recall'],
                df_alpha_c['precision'],
                color=colors_default[curve_i],
                marker=markers_default[curve_i]
            )
            # axs[axs_row].errorbar(
            #     df_aggregate['recall']['mean'],
            #     df_aggregate['precision']['mean'],
            #     # Add horizontal (symmetric) error bars
            #     xerr=df_aggregate['recall']['std'],
            #     # Add vertical (symmetric) error bars
            #     yerr=df_aggregate['precision']['std'],
            # )
        # Set axes properties
        axs[axs_row].set_xlabel(r'$Recall$', horizontalalignment='right', x=1.0)
        axs[axs_row].set_ylabel(r'$Precision$')  # , horizontalalignment='right', y=1.0)
        # axs[axs_row].set_xscale('log')
        # axs[axs_row].set_yscale('log')
        axs[axs_row].yaxis.set_ticks_position('both')
        # axs[axs_row].set_xlim([0, 1])
        # axs[axs_row].set_ylim([0, 1])
        # Add legend
        legend = [
         r'$\alpha_{{max}}={}$'.format(alpha_c) for alpha_c in alpha_c_range]
        axs[axs_row].legend(labels=legend)

    return fig


def plot_precision_vs_recall_subplots_T_scatter_aggregate():
    # Precision/Recall scatter plot (aggregate, scatter)
    # Legend: alpha

    # Probably better than Sensitivity/Specificity in our case, because the
    # number of negatives outweight the number of positives, see reference:
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4349800/

    # Set color pallette and/or line styles
    arrow_color = 'b'

    # Define (sub)plot(s)
    fig = plt.figure()
    axs = plt.gca()
    axs = np.array([axs])
    # Select data of interest
    df_interest = df[parameters_explored + ['precision', 'recall']]
    # Convert remaining DataFrame to float type for averaging
    df_interest = df_interest.astype(float)
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'nodes_n', 'samples_n', 'repetition_i'}
    # Initialise arrow coordinates array
    arrow_coord = np.zeros(shape=(len(alpha_c_range), len(samples_n_range), 2))
    arrow_i = 0
    for (curve_i, alpha_c) in enumerate(alpha_c_range):
        # Select dataframe entries(i.e runs) with the same alpha_c
        df_alpha_c = df_interest.loc[df_interest['p_value'] == alpha_c].drop(
            'p_value', 1)
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_alpha_c.columns
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        # Plot precision vs. recall
        axs[0].scatter(
            df_alpha_c['recall'],
            df_alpha_c['precision'],
            color=colors_default[curve_i],
            marker=markers_default[curve_i]
        )
        # Update arrow coordinates array
        df_aggregate = df_alpha_c.groupby('samples_n').agg(['mean'])
        arrow_coord[arrow_i, 0:len(df_aggregate), 0] = (
            df_aggregate['recall']['mean'])
        arrow_coord[arrow_i, 0:len(df_aggregate), 1] = (
            df_aggregate['precision']['mean'])
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
    axs[0].axvline(0.3, linestyle='--', color='k')  # VAR
    # axs[0].axvline(0.45, linestyle='--', color='k')  # CLM
    axs[0].axvline(0.95, linestyle='--', color='k')  # VAR
    # axs[0].axvline(0.91, linestyle='--', color='k')  # CLM
    # Add text
    # plt.text(0.1, 0.33, r'$T=100$', fontsize=10)
    # plt.text(0.57, 0.33, r'$T=1000$', fontsize=10)
    # plt.text(0.965, 0.4, r'$T=1000$', fontsize=10, rotation=90)
    plt.text(0.12, plt.ylim()[0] + 0.08, r'$T=100$', fontsize=10)
    plt.text(0.57, plt.ylim()[0] + 0.08, r'$T=1000$', fontsize=10)
    plt.text(0.97, plt.ylim()[0] + 0.21, r'$T=10000$', fontsize=10, rotation=90)

    return fig


def plot_delay_error_mean_vs_T():
    # Plot delay error mean vs. samples_number
    # Subplots: nerwork size

    fig, axs = plt.subplots(len(alpha_c_range), 1, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + ['delay_error_mean']]
    # Convert remaining DataFrame to float type for averaging
    df_interest = df_interest.astype(float)
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'repetition_i'}
    # If axs is not a list (i.e. it only contains one subplot), turn it into a
    # list with one element
    if not hasattr(axs, "__len__"):
        axs = np.array([axs])
    for (axs_row, alpha_c) in enumerate(alpha_c_range):
        # Set subplot title
        axs[axs_row].set_title(r'$\alpha_{{max}}={}$'.format(alpha_c))
        # Select dataframe entries(i.e runs) with the same number of nodes
        df_alpha_c = df_interest.loc[
            df_interest['p_value'] == alpha_c].drop('p_value', 1)
        for (curve_i, nodes_n) in enumerate(nodes_n_range):
            # Select dataframe entries(i.e runs) with the same number of nodes
            df_nodes = df_alpha_c.loc[df_alpha_c['nodes_n'] == nodes_n].drop(
                'nodes_n', 1)
            # Group by number of samples and then compute mean and std
            df_aggregate = df_nodes.groupby('samples_n').agg(
                ['mean', 'std'])
            # Ensure that only the desired parameters are aggregated or averaged
            df_keys_remaining = (
                df_aggregate.columns.get_level_values(0))
            check_remaining_dimensions(df_keys_remaining, parameters_to_average)
            # Plot delay error
            axs[axs_row].errorbar(
                df_aggregate.index,
                df_aggregate['delay_error_mean']['mean'],
                # Add vertical (symmetric) error bars
                yerr=df_aggregate['delay_error_mean']['std'],
                fmt='-o',
                color=colors_default[curve_i],
                marker=markers_default[curve_i])
            # Set axes properties
            axs[axs_row].set_xlabel(r'$T$', horizontalalignment='right', x=1.0)
            axs[axs_row].set_ylabel(r'$Absolute\ error$')  # , horizontalalignment='right', y=1.0)
            axs[axs_row].set_xscale('log')
            # axs[axs_row].set_yscale('log')
            axs[axs_row].yaxis.set_ticks_position('both')
            # axs[axs_row].set_xlim([95, 11000])
            # axs[axs_row].set_ylim(bottom=0)
            # Add legend
            legend = [r'$N={}$'.format(nodes_n) for nodes_n in nodes_n_range]
            axs[axs_row].legend(labels=legend)

    return fig


def plot_delay_error_mean_vs_T_relative():
    # Plot delay error mean vs. samples_number
    # Subplots: nerwork size

    fig, axs = plt.subplots(len(alpha_c_range), 1, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + ['delay_error_mean_normalised']]
    # Convert remaining DataFrame to float type for averaging
    df_interest = df_interest.astype(float)
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'repetition_i'}
    # If axs is not a list (i.e. it only contains one subplot), turn it into a
    # list with one element
    if not hasattr(axs, "__len__"):
        axs = np.array([axs])
    for (axs_row, alpha_c) in enumerate(alpha_c_range):
        # Set subplot title
        axs[axs_row].set_title(r'$\alpha_{{max}}={}$'.format(alpha_c))
        # Select dataframe entries(i.e runs) with the same number of nodes
        df_alpha_c = df_interest.loc[
            df_interest['p_value'] == alpha_c].drop('p_value', 1)
        for (curve_i, nodes_n) in enumerate(nodes_n_range):
            # Select dataframe entries(i.e runs) with the same number of nodes
            df_nodes = df_alpha_c.loc[df_alpha_c['nodes_n'] == nodes_n].drop(
                'nodes_n', 1)
            # Group by number of samples and then compute mean and std
            df_aggregate = df_nodes.groupby('samples_n').agg(
                ['mean', 'std'])
            # Ensure that only the desired parameters are aggregated or averaged
            df_keys_remaining = (
                df_aggregate.columns.get_level_values(0))
            check_remaining_dimensions(df_keys_remaining, parameters_to_average)
            # Plot delay error
            axs[axs_row].errorbar(
                df_aggregate.index,
                df_aggregate['delay_error_mean_normalised']['mean'],
                # Add vertical (symmetric) error bars
                yerr=df_aggregate['delay_error_mean_normalised']['std'],
                fmt='-o',
                color=colors_default[curve_i],
                marker=markers_default[curve_i])
            # Set axes properties
            axs[axs_row].set_xlabel(r'$T$', horizontalalignment='right', x=1.0)
            #axs[axs_row].set_ylabel(r'$Relative\ error$')  # , horizontalalignment='right', y=1.0)
            axs[axs_row].set_xscale('log')
            # axs[axs_row].set_yscale('log')
            axs[axs_row].yaxis.set_ticks_position('both')
            # axs[axs_row].set_xlim([95, 11000])
            # axs[axs_row].set_ylim(bottom=0)
            # Add legend
            legend = [r'$N={}$'.format(nodes_n) for nodes_n in nodes_n_range]
            axs[axs_row].legend(labels=legend)

    return fig


def plot_delay_error_mean_vs_T_alpha_interest():
    # Plot delay error mean vs. samples_number
    # Subplots: nerwork size

    fig, axs = plt.subplots(1, 1)
    # Select data of interest
    df_interest = df[parameters_explored + ['delay_error_mean']]
    # Convert remaining DataFrame to float type for averaging
    df_interest = df_interest.astype(float)
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'repetition_i'}
    # If axs is not a list (i.e. it only contains one subplot), turn it into a
    # list with one element
    if not hasattr(axs, "__len__"):
        axs = np.array([axs])
    for (axs_row, alpha_c) in enumerate([alpha_interest]):
        # Set subplot title
        # axs[axs_row].set_title(r'$\alpha_{{max}}={}$'.format(alpha_c))
        # Select dataframe entries(i.e runs) with the same number of nodes
        df_alpha_c = df_interest.loc[
            df_interest['p_value'] == alpha_c].drop('p_value', 1)
        for (curve_i, nodes_n) in enumerate(nodes_n_range):
            # Select dataframe entries(i.e runs) with the same number of nodes
            df_nodes = df_alpha_c.loc[df_alpha_c['nodes_n'] == nodes_n].drop(
                'nodes_n', 1)
            # Group by number of samples and then compute mean and std
            df_aggregate = df_nodes.groupby('samples_n').agg(
                ['mean', 'std', 'max', 'min'])
            # Ensure that only the desired parameters are aggregated or averaged
            df_keys_remaining = (
                df_aggregate.columns.get_level_values(0))
            check_remaining_dimensions(df_keys_remaining, parameters_to_average)
            # Plot delay error
            axs[axs_row].errorbar(
                df_aggregate.index,
                df_aggregate['delay_error_mean']['mean'],
                # Add vertical (symmetric) error bars
                yerr=[
                    0*df_aggregate['delay_error_mean']['std'],
                    df_aggregate['delay_error_mean']['std']],
                fmt='-o',
                color=colors_default[curve_i],
                marker=markers_default[curve_i])
            # Set axes properties
            axs[axs_row].set_xlabel(r'$T$', horizontalalignment='right', x=1.0)
            axs[axs_row].set_ylabel(r'$Absolute\ error$')  # , horizontalalignment='right', y=1.0)
            axs[axs_row].set_xscale('log')
            # axs[axs_row].set_yscale('log')
            axs[axs_row].yaxis.set_ticks_position('both')
            # axs[axs_row].set_xlim([95, 11000])
            # axs[axs_row].set_ylim(bottom=0)
            # Add legend
            legend = [r'$N={}$'.format(nodes_n) for nodes_n in nodes_n_range]
            axs[axs_row].legend(labels=legend)

    return fig


def plot_delay_error_mean_vs_T_relative_alpha_interest():
    # Plot delay error mean vs. samples_number
    # Subplots: nerwork size
    fig, axs = plt.subplots(1, 1)
    # Select data of interest
    df_interest = df[parameters_explored + ['delay_error_mean_normalised']]
    # Convert remaining DataFrame to float type for averaging
    df_interest = df_interest.astype(float)
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'repetition_i'}
    # If axs is not a list (i.e. it only contains one subplot), turn it into a
    # list with one element
    if not hasattr(axs, "__len__"):
        axs = np.array([axs])
    for (axs_row, alpha_c) in enumerate([alpha_interest]):
        # Set subplot title
        # axs[axs_row].set_title(r'$\alpha_{{max}}={}$'.format(alpha_c))
        # Select dataframe entries(i.e runs) with the same number of nodes
        df_alpha_c = df_interest.loc[
            df_interest['p_value'] == alpha_c].drop('p_value', 1)
        for (curve_i, nodes_n) in enumerate(nodes_n_range):
            # Select dataframe entries(i.e runs) with the same number of nodes
            df_nodes = df_alpha_c.loc[df_alpha_c['nodes_n'] == nodes_n].drop(
                'nodes_n', 1)
            # Group by number of samples and then compute mean and std
            df_aggregate = df_nodes.groupby('samples_n').agg(
                ['mean', 'std', 'max', 'min'])
            # Ensure that only the desired parameters are aggregated or averaged
            df_keys_remaining = (
                df_aggregate.columns.get_level_values(0))
            check_remaining_dimensions(df_keys_remaining, parameters_to_average)
            # Plot delay error
            axs[axs_row].errorbar(
                df_aggregate.index,
                df_aggregate['delay_error_mean_normalised']['mean'],
                # Add vertical (symmetric) error bars
                yerr=[
                    0*df_aggregate['delay_error_mean_normalised']['std'],
                    df_aggregate['delay_error_mean_normalised']['std']],
                fmt='-o',
                color=colors_default[curve_i],
                marker=markers_default[curve_i])
            # Set axes properties
            axs[axs_row].set_xlabel(r'$T$', horizontalalignment='right', x=1.0)
            axs[axs_row].set_ylabel(r'$Lag\ error$')  # , horizontalalignment='right', y=1.0)
            axs[axs_row].set_xscale('log')
            # axs[axs_row].set_yscale('log')
            axs[axs_row].yaxis.set_ticks_position('both')
            # axs[axs_row].set_xlim([95, 11000])
            # axs[axs_row].set_ylim(bottom=0)
            # axs[axs_row].set_ylim(bottom=-0.015, top=0.64)
            axs[axs_row].set_ylim(top=0.3)
            # Add legend
            legend = [r'$N={}$'.format(nodes_n) for nodes_n in nodes_n_range]
            axs[axs_row].legend(labels=legend)#, loc=(0.72, 0.54))

    return fig


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
        df_aggregate = df_samples.groupby('nodes_n').agg(
            lambda x: x.tolist())
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        for (axs_col, nodes_n) in enumerate(nodes_n_range):
            # Set subplot title
            axs[axs_row, axs_col].set_title(r'$T={0}$, $N={1}$'.format(
                samples_n,
                nodes_n
                ))
            if not np.all(np.isnan(df_aggregate.loc[nodes_n].TE_omnibus_empirical)):
                TE_omnibus_empirical = np.concatenate(
                    df_aggregate.loc[nodes_n].TE_omnibus_empirical)
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
    return fig


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

    return fig


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
        df_aggregate = df_samples.groupby('nodes_n').agg(
            lambda x: x.tolist())
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        for (axs_col, nodes_n) in enumerate(nodes_n_range):
            # Set subplot title
            axs[axs_row, axs_col].set_title(r'$T={0}$, $N={1}$'.format(
                samples_n,
                nodes_n))
            if not np.all(np.isnan(df_aggregate.loc[nodes_n].TE_omnibus_empirical)):
                TE_omnibus_empirical = np.concatenate(
                    df_aggregate.loc[nodes_n].TE_omnibus_empirical)
                TE_omnibus_theoretical_causal_vars = np.concatenate(
                    df_aggregate.loc[nodes_n].TE_omnibus_theoretical_causal_vars)
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
    return fig


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
        df_aggregate = df_samples.groupby('nodes_n').agg(
            lambda x: x.tolist())
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        for (axs_col, nodes_n) in enumerate(nodes_n_range):
            # Set subplot title
            axs[axs_row, axs_col].set_title(r'$T={0}$, $N={1}$'.format(
                samples_n,
                nodes_n
                ))
            if not np.all(np.isnan(df_aggregate.loc[nodes_n].TE_omnibus_empirical)):
                TE_omnibus_empirical = np.concatenate(
                    df_aggregate.loc[nodes_n].TE_omnibus_empirical)
                TE_omnibus_theoretical_inferred_vars = np.concatenate(
                    df_aggregate.loc[nodes_n].TE_omnibus_theoretical_inferred_vars)
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
    return fig


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
    df_aggregate = df_interest.groupby('samples_n').agg(
        lambda x: x.tolist())
    # Ensure that only the desired parameters are aggregated or averaged
    df_keys_remaining = df_aggregate.columns.get_level_values(0)
    check_remaining_dimensions(df_keys_remaining, parameters_to_average)
    axs[0].plot(
        df_aggregate.index.astype(int).tolist(),
        [np.nanmean(df_aggregate['TE_err_abs_relative'][samples_n])
            for samples_n in samples_n_range])
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

    return fig


# -------------------------------------------------------------------------
# endregion
# -------------------------------------------------------------------------



# -------------------------------------------------------------------------
# region TE theoretical VAR1 motifs
# -------------------------------------------------------------------------

def plot_bTE_vs_WS_p():
    # Plot bTE vs. Watts-Strogatz rewiring prob
    # Legend: methods for computing bTE
    bTE_methods = [
        'bTE_empirical_causal_vars',
        'bTE_theoretical_causal_vars',
        'bTE_approx4_causal_vars',
        'bTE_motifs_acde_causal_vars',
        'bTE_motifs_ae_causal_vars',
        #'bTE_approx2_causal_vars',
        ]
    fig, axs = my_subplots(1, 1, sharex=True, sharey=True)
    
    # mpl.rcParams['axes.labelsize'] = 44
    # mpl.rcParams['axes.linewidth'] = 1
    # mpl.rcParams['xtick.labelsize'] = 40
    # mpl.rcParams['ytick.labelsize'] = 20
    # mpl.rcParams['lines.linewidth'] = 1.5
    # mpl.rcParams['lines.markersize'] = 4
    # mpl.rcParams['legend.fontsize'] = 12

    # Select data of interest
    df_interest = df[parameters_explored + bTE_methods]
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    df_interest = df_interest.loc[
        df_interest['weight_distribution'] == weight_interest].drop('weight_distribution', 1)
    # Average TE values in each matrix
    # print(df_interest['bTE_empirical_causal_vars'][0])
    # print(df_interest['bTE_approx2_causal_vars'][0])
    # print(df_interest['bTE_approx4_causal_vars'][0])
    # print(df_interest['bTE_theoretical_causal_vars'][0])
    # 
    # print(np.nansum(df_interest['bTE_empirical_causal_vars'][0] > 0))
    # print(np.nansum(df_interest['bTE_approx2_causal_vars'][0] > 0))
    # print(np.nansum(df_interest['bTE_approx4_causal_vars'][0] > 0))
    # print(np.nansum(df_interest['bTE_theoretical_causal_vars'][0] > 0))
    # savemat('bTE_matrix_python', {'bTE_matrix_python':df_interest['bTE_theoretical_causal_vars'][-1]})
    # df_interest['bTE_empirical_causal_vars'] = df_interest['bTE_empirical_causal_vars'].apply(np.nanmean)
    # df_interest['bTE_approx2_causal_vars'] = df_interest['bTE_approx2_causal_vars'].apply(np.nanmean)
    # df_interest['bTE_approx4_causal_vars'] = df_interest['bTE_approx4_causal_vars'].apply(np.nanmean)
    # df_interest['bTE_theoretical_causal_vars'] = df_interest['bTE_theoretical_causal_vars'].apply(np.nanmean)
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'repetition_i'}
    for (method_i, bTE_method) in enumerate(bTE_methods):
        df_interest[bTE_method] = df_interest[bTE_method].apply(np.nanmean)
        # Group by WS_p
        df_aggregate = df_interest.groupby('WS_p').agg(
            lambda x: list(x))
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        # Plot average path lenght inferred
        axs[0].plot(
            WS_p_range,
            [np.nanmean(df_aggregate[bTE_method][WS_p])
             for WS_p in WS_p_range],
            marker=markers_default[method_i],
            label=bTE_methods_legends[bTE_method])
        axs[0].scatter(
            df_interest['WS_p'].values.astype(float).tolist(),
            df_interest[bTE_method].values,
            alpha=0.3,
            marker=markers_default[method_i])
    # Set axes properties
    axs[0].set_xlabel(r'$\text{Rewiring probability }(\gamma)$')#, horizontalalignment='right', x=1.0)
    axs[0].set_ylabel(r'$\text{Transfer Entropy}$')
    axs[0].set_xscale('log')
    axs[0].yaxis.set_ticks_position('both')
    axs[0].set_xlim(left=0.002)
    # Add legend
    axs[0].legend(loc=(0.03, 0.127))

    fig.set_figheight(3.3)
    fig.set_figwidth(4)

    return fig


def imshow_bTE_vs_indegree_source_target():
    # Plot bTE as a function if indegree of source and target
    # Legend: methods for computing bTE
    bTE_methods = [
        'bTE_empirical_causal_vars',
        'bTE_theoretical_causal_vars',
        #'bTE_motifs_acde_causal_vars',
        #'bTE_approx2_causal_vars',
        'bTE_approx4_causal_vars',
        ]
    fig, axs = my_subplots(1, len(bTE_methods), sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + bTE_methods]
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    df_interest = df_interest.loc[
        df_interest['weight_distribution'] == weight_interest].drop('weight_distribution', 1)
    if 'WS_p' in df_interest.columns:
        df_interest = df_interest.drop('WS_p', 1)
    
    #savemat('bTE_matrix_python', {'bTE_matrix_python':df_interest['bTE_theoretical_causal_vars'][-1]})
    
    
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {"repetition_i"}
    # Ensure that only the desired parameters are aggregated or averaged
    df_keys_remaining = df_interest.columns.get_level_values(0)
    check_remaining_dimensions(df_keys_remaining, parameters_to_average)
    imshow_plots = []
    for i, bTE_method in enumerate(bTE_methods):
        # Set subplot title
        axs[0, i].set_title(bTE_methods_legends[bTE_method])
        max_indegree = 1
        im_matrix_sparse = []
        for bTE_matrix in df_interest[bTE_method]:
            adj_matrix = bTE_matrix > 0
            indegrees = np.sum(adj_matrix, 1)
            max_indegree = max(max_indegree, np.max(indegrees))
            for s in range(N_interest):
                for t in range(N_interest):
                    if adj_matrix[s, t] > 0:
                        im_matrix_sparse.append((
                                indegrees[s],
                                indegrees[t],
                                bTE_matrix[s, t]))
        # go from sparse to dense representation of image_matrix
        image_matrix = np.zeros(shape=(max_indegree + 1, max_indegree + 1))
        n_points = np.zeros(shape=(max_indegree + 1, max_indegree + 1))
        for v in im_matrix_sparse:
            image_matrix[v[0], v[1]] += v[2]
            n_points[v[0], v[1]] += 1
        image_matrix[image_matrix == 0] = np.NaN
        # divide sum by number of points to get the average
        n_points[n_points == 0] = 1
        image_matrix = image_matrix / n_points
        # plot image_matrix
        #im = axs[0, i].imshow(image_matrix, origin='lower')
        #fig.colorbar(im, orientation="vertical")
        imshow_plots.append(axs[0, i].imshow(image_matrix, origin='lower'))
        # Set axes properties
        axs[0, i].set_xlabel(r'$\text{Target in-degree}$', horizontalalignment='right', x=1.0)
        axs[0, i].set_ylabel(r'$\text{Source in-degree}$')
    # Find the min and max of all colors for use in setting the color scale.
    # As shown in https://matplotlib.org/3.1.0/gallery/images_contours_and_fields/multi_image.html
    vmin = min(image.get_array().min() for image in imshow_plots)
    vmax = max(image.get_array().max() for image in imshow_plots)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for im in imshow_plots:
        im.set_norm(norm)
    # Show colorbar
    fig.colorbar(imshow_plots[0], ax = axs[0, -1], orientation='vertical', pad=0.2)
    # Make images respond to changes in the norm of other images (e.g. via the
    # "edit axis, curves and images parameters" GUI on Qt), but be careful not to
    # recurse infinitely!
    def update(changed_image):
        for im in imshow_plots:
            if (changed_image.get_cmap() != im.get_cmap()
                    or changed_image.get_clim() != im.get_clim()):
                im.set_cmap(changed_image.get_cmap())
                im.set_clim(changed_image.get_clim())
    for im in imshow_plots:
        im.callbacksSM.connect('changed', update)

    return fig


def imshow_bTE_empirical_vs_indegree_source_target():
    # Plot bTE as a function if indegree of source and target
    fig, axs = my_subplots(1, 1, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + ['bTE_empirical_causal_vars']]
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    df_interest = df_interest.loc[
        df_interest['weight_distribution'] == weight_interest].drop('weight_distribution', 1)
    if 'WS_p' in df_interest.columns:
        df_interest = df_interest.drop('WS_p', 1)
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {"repetition_i"}
    # Ensure that only the desired parameters are aggregated or averaged
    df_keys_remaining = df_interest.columns.get_level_values(0)
    check_remaining_dimensions(df_keys_remaining, parameters_to_average)
    max_indegree = 1
    im_matrix_sparse = []
    for bTE_matrix in df_interest['bTE_empirical_causal_vars']:
        adj_matrix = bTE_matrix > 0
        indegrees = np.sum(adj_matrix, 1)
        max_indegree = max(max_indegree, np.max(indegrees))
        for s in range(N_interest):
            for t in range(N_interest):
                if adj_matrix[s, t] > 0:
                    im_matrix_sparse.append((
                            indegrees[s],
                            indegrees[t],
                            bTE_matrix[s, t]))
    # go from sparse to dense representation of image_matrix
    image_matrix = np.zeros(shape=(max_indegree + 1, max_indegree + 1))
    n_points = np.zeros(shape=(max_indegree + 1, max_indegree + 1))
    for v in im_matrix_sparse:
        image_matrix[v[0], v[1]] += v[2]
        n_points[v[0], v[1]] += 1
    image_matrix[image_matrix == 0] = np.NaN
    # divide sum by number of points to get the average
    n_points[n_points == 0] = 1
    image_matrix = image_matrix / n_points
    # plot image_matrix
    im = axs[0].imshow(image_matrix, origin='lower')
    fig.colorbar(im, orientation="vertical")
    # Set axes properties
    #axs[0].set_ylim(top=15)
    axs[0].set_xlabel(r'$\text{Target in-degree}$')#, horizontalalignment='right', x=1.0)
    axs[0].set_ylabel(r'$\text{Source in-degree}$')

    return fig


def imshow_bTE_theoretical_vs_indegree_source_target():
    # Plot bTE as a function if indegree of source and target
    fig, axs = my_subplots(1, 1, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + ['bTE_empirical_causal_vars']]
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    df_interest = df_interest.loc[
        df_interest['weight_distribution'] == weight_interest].drop('weight_distribution', 1)
    if 'WS_p' in df_interest.columns:
        df_interest = df_interest.drop('WS_p', 1)
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {"repetition_i"}
    # Ensure that only the desired parameters are aggregated or averaged
    df_keys_remaining = df_interest.columns.get_level_values(0)
    check_remaining_dimensions(df_keys_remaining, parameters_to_average)
    max_indegree = 1
    im_matrix_sparse = []
    for bTE_matrix in df_interest['bTE_empirical_causal_vars']:
        adj_matrix = bTE_matrix > 0
        indegrees = np.sum(adj_matrix, 1)
        max_indegree = max(max_indegree, np.max(indegrees))
        for s in range(N_interest):
            for t in range(N_interest):
                if adj_matrix[s, t] > 0:
                    im_matrix_sparse.append((
                            indegrees[s],
                            indegrees[t],
                            bTE_matrix[s, t]))
    # go from sparse to dense representation of image_matrix
    image_matrix = np.zeros(shape=(max_indegree + 1, max_indegree + 1))
    n_points = np.zeros(shape=(max_indegree + 1, max_indegree + 1))
    for v in im_matrix_sparse:
        image_matrix[v[0], v[1]] += v[2]
        n_points[v[0], v[1]] += 1
    image_matrix[image_matrix == 0] = np.NaN
    # divide sum by number of points to get the average
    n_points[n_points == 0] = 1
    image_matrix = image_matrix / n_points
    # plot image_matrix
    im = axs[0].imshow(image_matrix, origin='lower')
    fig.colorbar(im, orientation="vertical")
    # Set axes properties
    axs[0].set_xlabel(r'$\text{Target in-degree}$')#, horizontalalignment='right', x=1.0)
    #axs[0].set_xlabel(r'$d_\textnormal{in}(Y)$', horizontalalignment='right', x=1.0)
    axs[0].set_ylabel(r'$\text{Source in-degree}$')
    #axs[0].set_ylabel(r'$d_\textnormal{in}(X)$')

    return fig

# -------------------------------------------------------------------------
# endregion
# -------------------------------------------------------------------------



# -------------------------------------------------------------------------
# region Watts-Strogatz
# -------------------------------------------------------------------------


def violin_plot_omnibus_TE_empirical_vs_WS_p():
    # Violin plot of TE_omnibus_empirical vs. WS_p
    # Subplots vertical: algorithms
    # Only select TE algorithms
    TE_algorithms = np.extract(['TE' in alg for alg in algorithms], algorithms)
    subplots_v = len(TE_algorithms)
    subplots_h = 1
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + ['TE_omnibus_empirical']]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}
    for (axs_row, algorithm) in enumerate(TE_algorithms):
        # Set subplot title
        axs[axs_row].set_title('{0}'.format(
            algorithm_names[algorithm]))
        # Select dataframe entries(i.e runs) with the same algorithm
        df_algorithm = df_interest.loc[
            df_interest['algorithm'] == algorithm].drop('algorithm', 1)
        # Group by WS_p and concatenate TE values lists
        df_aggregate = df_algorithm.groupby('WS_p').agg(
            lambda x: x.tolist())
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        TE_omnibus_empirical = [
            np.concatenate(df_aggregate.loc[WS_p].TE_omnibus_empirical).astype(float)
                for WS_p in WS_p_range]
        # Violin plot of omnibus TE vs WS_p
        violin = axs[axs_row].violinplot(
            TE_omnibus_empirical,
            positions=WS_p_range,
            widths=WS_p_range/2,
            showmeans=False,
            showextrema=False,
            showmedians=False,
            #points=100,
            #bw_method=
            )
        # Only plot right-hand half of the violins
        for b in violin['bodies']:
            if len(b.get_paths()) > 0:
                m = np.nanmean(b.get_paths()[0].vertices[:, 0])
                b.get_paths()[0].vertices[:, 0] = np.clip(
                    b.get_paths()[0].vertices[:, 0], m, np.inf)
                b.set_color('tab:blue')
        # Join mean values
        mean_vals = [np.nanmean(np.concatenate(df_aggregate.loc[WS_p].TE_omnibus_empirical)) for WS_p in WS_p_range]
        axs[axs_row].plot(
            WS_p_range,
            mean_vals,
            '-o',
            color='tab:blue')
        # Set axes properties
        axs[axs_row].set_xlabel(
            r'$Rewiring\ probability$', horizontalalignment='right', x=1.0)
        axs[axs_row].set_ylabel(
            r'$Omnibus\ TE$')
        axs[axs_row].set_xscale('log')
        axs[axs_row].yaxis.set_ticks_position('both')
        axs[axs_row].set_ylim(bottom=0)
    return fig


def violin_TE_complete_theoretical_causal_vars_vs_WS_p():
    # Violin plot of TE_complete_theoretical_causal_vars vs. WS_p
    # Subplots vertical: algorithms
    # Only select TE algorithms
    TE_algorithms = np.extract(['TE' in alg for alg in algorithms], algorithms)
    subplots_v = len(TE_algorithms)
    subplots_h = 1
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + ['TE_complete_theoretical_causal_vars']]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Get a numpy array from a list of arrays
    df_interest['TE_complete_theoretical_causal_vars'] = df_interest['TE_complete_theoretical_causal_vars'].apply(concat_if_not_empty)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}
    for (axs_row, algorithm) in enumerate(TE_algorithms):
        # Set subplot title
        axs[axs_row].set_title('{0}'.format(
            algorithm_names[algorithm]))
        # Select dataframe entries(i.e runs) with the same algorithm
        df_algorithm = df_interest.loc[
            df_interest['algorithm'] == algorithm].drop('algorithm', 1)
        # Group by WS_p and concatenate TE values lists
        df_aggregate = df_algorithm.groupby('WS_p').agg(
            lambda x: x.tolist())
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        TE_complete_theoretical_causal_vars = [
            np.concatenate(df_aggregate.loc[WS_p].TE_complete_theoretical_causal_vars).astype(float)
                for WS_p in WS_p_range]
        # Violin plot of TE vs WS_p
        violin = axs[axs_row].violinplot(
            TE_complete_theoretical_causal_vars,
            positions=WS_p_range,
            widths=WS_p_range/2,
            showmeans=False,
            showextrema=False,
            showmedians=False,
            #points=100,
            #bw_method=
            )
        # Only plot right-hand half of the violins
        for b in violin['bodies']:
            if len(b.get_paths()) > 0:
                m = np.nanmean(b.get_paths()[0].vertices[:, 0])
                b.get_paths()[0].vertices[:, 0] = np.clip(
                    b.get_paths()[0].vertices[:, 0], m, np.inf)
                b.set_color('tab:blue')
        # Join mean values
        mean_vals = [np.nanmean(np.concatenate(df_aggregate.loc[WS_p].TE_complete_theoretical_causal_vars)) for WS_p in WS_p_range]
        axs[axs_row].plot(
            WS_p_range,
            mean_vals,
            '-o',
            color='tab:blue')
        # Set axes properties
        axs[axs_row].set_xlabel(
            r'$Rewiring\ probability$', horizontalalignment='right', x=1.0)
        axs[axs_row].set_ylabel(
            r'$Complete\ TE$')
        axs[axs_row].set_xscale('log')
        axs[axs_row].yaxis.set_ticks_position('both')
        #axs[axs_row].set_ylim(bottom=0)
    return fig


def violin_TE_empirical_vs_WS_p():
    # Violin plot of TE_complete_empirical and TE_omnibus_empirical vs. WS_p
    # Subplots vertical: algorithms
    # Only select TE algorithms
    TE_algorithms = np.extract(['TE' in alg for alg in algorithms], algorithms)
    subplots_v = len(TE_algorithms)
    subplots_h = 1
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'TE_complete_empirical',
        'TE_omnibus_empirical',
        ]]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # # Get a numpy array from a list of arrays
    # df_interest['TE_complete_empirical'] = df_interest['TE_complete_empirical'].apply(lambda x: [np.array([]) if v is None else v for v in x])
    # df_interest['TE_complete_empirical'] = df_interest['TE_complete_empirical'].apply(concat_if_not_empty)
    # Flatten TE matrices and remove NaNs
    df_interest['TE_complete_empirical'] = df_interest['TE_complete_empirical'].apply(np.ndarray.flatten)
    df_interest['TE_complete_empirical'] = df_interest['TE_complete_empirical'].apply(lambda x: x[~np.isnan(x)])
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}
    for (axs_row, algorithm) in enumerate(TE_algorithms):
        # Set subplot title
        axs[axs_row].set_title('{0}'.format(
            algorithm_names[algorithm]))
        # Select dataframe entries(i.e runs) with the same algorithm
        df_algorithm = df_interest.loc[
            df_interest['algorithm'] == algorithm].drop('algorithm', 1)
        # Group by WS_p and concatenate TE values lists
        df_aggregate = df_algorithm.groupby('WS_p').agg(
            lambda x: x.tolist())
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        TE_complete_empirical = [
            np.concatenate(df_aggregate.loc[WS_p].TE_complete_empirical).astype(float)
                for WS_p in WS_p_range]
        TE_omnibus_empirical = [
            np.concatenate(df_aggregate.loc[WS_p].TE_omnibus_empirical).astype(float)
                for WS_p in WS_p_range]
        # Violin plot of omnibus TE vs WS_p
        violin_omnibus = axs[axs_row].violinplot(
            TE_omnibus_empirical,
            positions=WS_p_range,
            widths=WS_p_range/2,
            showmeans=False,
            showextrema=False,
            showmedians=False,
            #points=100,
            #bw_method=,
            )
        # Only plot right-hand half of the violins
        for b in violin_omnibus['bodies']:
            if len(b.get_paths()) > 0:
                m = np.nanmean(b.get_paths()[0].vertices[:, 0])
                b.get_paths()[0].vertices[:, 0] = np.clip(
                    b.get_paths()[0].vertices[:, 0], m, np.inf)
                b.set_color('tab:blue')
        # Join mean values
        mean_vals = [np.nanmean(np.concatenate(df_aggregate.loc[WS_p].TE_omnibus_empirical)) for WS_p in WS_p_range]
        axs[axs_row].plot(
            WS_p_range,
            mean_vals,
            '-o',
            color='tab:blue'
        )
        # Violin plot of complete TE vs WS_p
        violin_complete = axs[axs_row].violinplot(
            TE_complete_empirical,
            positions=WS_p_range,
            widths=WS_p_range/2,
            showmeans=False,
            showextrema=False,
            showmedians=False,
            #points=100,
            #bw_method=,
            )
        # Only plot right-hand half of the violins
        for b in violin_complete['bodies']:
            if len(b.get_paths()) > 0:
                m = np.nanmean(b.get_paths()[0].vertices[:, 0])
                b.get_paths()[0].vertices[:, 0] = np.clip(
                    b.get_paths()[0].vertices[:, 0], m, np.inf)
                b.set_color('tab:orange')
        # Join mean values
        mean_vals = [np.nanmean(np.concatenate(df_aggregate.loc[WS_p].TE_complete_empirical)) for WS_p in WS_p_range]
        axs[axs_row].plot(
            WS_p_range,
            mean_vals,
            '-o',
            color='tab:orange'
        )
        # Set axes properties
        axs[axs_row].set_xlabel(
            r'$Rewiring\ probability$', horizontalalignment='right', x=1.0)
        axs[axs_row].set_ylabel(
            r'$TE\ (empirical)$')
        axs[axs_row].set_xscale('log')
        axs[axs_row].yaxis.set_ticks_position('both')
        if algorithm == 'bTE_greedy':
            axs[axs_row].legend(['Omnibus', 'Apparent'])
        if algorithm == 'mTE_greedy':
            axs[axs_row].legend(['Omnibus', 'Complete'])
        #axs[axs_row].set_ylim(bottom=0)
    return fig


def violin_TE_theoretical_vs_WS_p():
    # Violin plot of TE theoretical on causal vars and TE theoretical on inferred vars vs. WS_p
    # Subplots vertical: algorithms
    # Only select TE algorithms
    TE_algorithms = np.extract(['TE' in alg for alg in algorithms], algorithms)
    subplots_v = len(TE_algorithms)
    subplots_h = 1
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=False)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'TE_complete_theoretical_causal_vars',
        'TE_complete_theoretical_inferred_vars',
        'TE_apparent_theoretical_causal_vars',
        'TE_apparent_theoretical_inferred_vars',
        ]]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Flatten TE matrices and ignore zeros
    df_interest['TE_complete_theoretical_causal_vars'] = df_interest['TE_complete_theoretical_causal_vars'].apply(np.ndarray.flatten)
    df_interest['TE_complete_theoretical_causal_vars'] = df_interest['TE_complete_theoretical_causal_vars'].apply(lambda x: x[x != 0])
    df_interest['TE_complete_theoretical_inferred_vars'] = df_interest['TE_complete_theoretical_inferred_vars'].apply(np.ndarray.flatten)
    df_interest['TE_complete_theoretical_inferred_vars'] = df_interest['TE_complete_theoretical_inferred_vars'].apply(lambda x: x[x != 0])
    df_interest['TE_apparent_theoretical_causal_vars'] = df_interest['TE_apparent_theoretical_causal_vars'].apply(np.ndarray.flatten)
    df_interest['TE_apparent_theoretical_causal_vars'] = df_interest['TE_apparent_theoretical_causal_vars'].apply(lambda x: x[x != 0])
    df_interest['TE_apparent_theoretical_inferred_vars'] = df_interest['TE_apparent_theoretical_inferred_vars'].apply(np.ndarray.flatten)
    df_interest['TE_apparent_theoretical_inferred_vars'] = df_interest['TE_apparent_theoretical_inferred_vars'].apply(lambda x: x[x != 0])
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}
    for (axs_row, algorithm) in enumerate(TE_algorithms):
        # Set subplot title
        axs[axs_row].set_title('{0}'.format(
            algorithm_names[algorithm]))
        # Select dataframe entries(i.e runs) with the same algorithm
        df_algorithm = df_interest.loc[
            df_interest['algorithm'] == algorithm].drop('algorithm', 1)
        # Group by WS_p and concatenate TE values lists
        df_aggregate = df_algorithm.groupby('WS_p').agg(
            lambda x: x.tolist())
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        if algorithm == 'bTE_greedy':
            TE_theoretical_causal_vars = [
                np.concatenate(df_aggregate.loc[WS_p].TE_apparent_theoretical_causal_vars).astype(float)
                    for WS_p in WS_p_range]
            TE_theoretical_inferred_vars = [
                np.concatenate(df_aggregate.loc[WS_p].TE_apparent_theoretical_inferred_vars).astype(float)
                    for WS_p in WS_p_range]
            mean_vals_causal_vars = [np.nanmean(np.concatenate(df_aggregate.loc[WS_p].TE_apparent_theoretical_causal_vars)) for WS_p in WS_p_range]
            mean_vals_inferred_vars = [np.nanmean(np.concatenate(df_aggregate.loc[WS_p].TE_apparent_theoretical_inferred_vars)) for WS_p in WS_p_range]
        if algorithm == 'mTE_greedy':
            TE_theoretical_causal_vars = [
                np.concatenate(df_aggregate.loc[WS_p].TE_complete_theoretical_causal_vars).astype(float)
                    for WS_p in WS_p_range]
            TE_theoretical_inferred_vars = [
                np.concatenate(df_aggregate.loc[WS_p].TE_complete_theoretical_inferred_vars).astype(float)
                    for WS_p in WS_p_range]
            mean_vals_causal_vars = [np.nanmean(np.concatenate(df_aggregate.loc[WS_p].TE_complete_theoretical_causal_vars)) for WS_p in WS_p_range]
            mean_vals_inferred_vars = [np.nanmean(np.concatenate(df_aggregate.loc[WS_p].TE_complete_theoretical_inferred_vars)) for WS_p in WS_p_range]
        # Violin plot of TE theoretical on causal vars vs WS_p
        violin_causal_vars = axs[axs_row].violinplot(
            TE_theoretical_causal_vars,
            positions=WS_p_range,
            widths=WS_p_range/2,
            showmeans=False,
            showextrema=False,
            showmedians=False,
            #points=100,
            #bw_method=,
            )
        # Only plot right-hand half of the violins
        for b in violin_causal_vars['bodies']:
            if len(b.get_paths()) > 0:
                m = np.nanmean(b.get_paths()[0].vertices[:, 0])
                b.get_paths()[0].vertices[:, 0] = np.clip(
                    b.get_paths()[0].vertices[:, 0], m, np.inf)
                b.set_color('tab:orange')
        # Join mean values
        axs[axs_row].plot(
            WS_p_range,
            mean_vals_causal_vars,
            '-o',
            color='tab:orange',
            label='Causal variables'
        )
        # Violin plot of TE theoretical on inferred vars vs WS_p
        violin_inferred_vars = axs[axs_row].violinplot(
            TE_theoretical_inferred_vars,
            positions=WS_p_range,
            widths=WS_p_range/2,
            showmeans=False,
            showextrema=False,
            showmedians=False,
            #points=100,
            #bw_method=,
            )
        # Only plot right-hand half of the violins
        for b in violin_inferred_vars['bodies']:
            if len(b.get_paths()) > 0:
                m = np.nanmean(b.get_paths()[0].vertices[:, 0])
                b.get_paths()[0].vertices[:, 0] = np.clip(
                    b.get_paths()[0].vertices[:, 0], m, np.inf)
                b.set_color('tab:blue')
        # Join mean values
        axs[axs_row].plot(
            WS_p_range,
            mean_vals_inferred_vars,
            '-o',
            color='tab:blue',
            label='Inferred variables'
        )
        # Set axes properties
        axs[axs_row].set_xlabel(
            r'$Rewiring\ probability$', horizontalalignment='right', x=1.0)
        axs[axs_row].set_xscale('log')
        axs[axs_row].yaxis.set_ticks_position('both')
        if algorithm == 'bTE_greedy':
            axs[axs_row].set_ylabel(r'$Apparent\ TE\ (theoretical)$')
        if algorithm == 'mTE_greedy':
            axs[axs_row].set_ylabel(r'$Complete\ TE\ (theoretical)$')
        axs[axs_row].legend()
        #axs[axs_row].set_ylim(bottom=0)
    return fig


def violin_AIS_theoretical_vs_WS_p():
    # Violin plot of AIS theoretical on causal vars and AIS theoretical on inferred vars vs. WS_p
    # Subplots vertical: algorithms
    subplots_v = len(algorithms)
    subplots_h = 1
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'AIS_theoretical_causal_vars',
        'AIS_theoretical_inferred_vars',
        ]]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}
    for (axs_row, algorithm) in enumerate(algorithms):
        # Set subplot title
        axs[axs_row].set_title('{0}'.format(
            algorithm_names[algorithm]))
        # Select dataframe entries(i.e runs) with the same algorithm
        df_algorithm = df_interest.loc[
            df_interest['algorithm'] == algorithm].drop('algorithm', 1)
        # Group by WS_p and concatenate TE values lists
        df_aggregate = df_algorithm.groupby('WS_p').agg(
            lambda x: x.tolist())
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        AIS_theoretical_causal_vars = [
            np.concatenate(df_aggregate.loc[WS_p].AIS_theoretical_causal_vars).astype(float)
                for WS_p in WS_p_range]
        AIS_theoretical_inferred_vars = [
            np.concatenate(df_aggregate.loc[WS_p].AIS_theoretical_inferred_vars).astype(float)
                for WS_p in WS_p_range]
        # Violin plot of AIS on causal vars vs WS_p
        violin_causal_vars = axs[axs_row].violinplot(
            AIS_theoretical_causal_vars,
            positions=WS_p_range,
            widths=WS_p_range/2,
            showmeans=False,
            showextrema=False,
            showmedians=False,
            #points=100,
            #bw_method=,
            )
        # Only plot right-hand half of the violins
        for b in violin_causal_vars['bodies']:
            if len(b.get_paths()) > 0:
                m = np.nanmean(b.get_paths()[0].vertices[:, 0])
                b.get_paths()[0].vertices[:, 0] = np.clip(
                    b.get_paths()[0].vertices[:, 0], m, np.inf)
                b.set_color('tab:blue')
        # Join mean values
        mean_vals = [np.nanmean(np.concatenate(df_aggregate.loc[WS_p].AIS_theoretical_causal_vars)) for WS_p in WS_p_range]
        axs[axs_row].plot(
            WS_p_range,
            mean_vals,
            '-o',
            color='tab:blue',
            label='Causal variables'
        )
        # Violin plot of AIS on inferred vars vs WS_p
        violin_inferred_vars = axs[axs_row].violinplot(
            AIS_theoretical_inferred_vars,
            positions=WS_p_range,
            widths=WS_p_range/2,
            showmeans=False,
            showextrema=False,
            showmedians=False,
            #points=100,
            #bw_method=,
            )
        # Only plot right-hand half of the violins
        for b in violin_inferred_vars['bodies']:
            if len(b.get_paths()) > 0:
                m = np.nanmean(b.get_paths()[0].vertices[:, 0])
                b.get_paths()[0].vertices[:, 0] = np.clip(
                    b.get_paths()[0].vertices[:, 0], m, np.inf)
                b.set_color('tab:orange')
        # Join mean values
        mean_vals = [np.nanmean(np.concatenate(df_aggregate.loc[WS_p].AIS_theoretical_inferred_vars)) for WS_p in WS_p_range]
        axs[axs_row].plot(
            WS_p_range,
            mean_vals,
            '-o',
            color='tab:orange',
            label='Inferred variables'
        )
        # Set axes properties
        axs[axs_row].set_xlabel(
            r'$Rewiring\ probability$', horizontalalignment='right', x=1.0)
        axs[axs_row].set_ylabel(
            r'$AIS\ (theoretical)$')
        axs[axs_row].set_xscale('log')
        axs[axs_row].yaxis.set_ticks_position('both')
        #axs[axs_row].set_ylim(bottom=0)
        axs[axs_row].legend()
    return fig


def violin_TE_apparent_and_conditional_pairs():
    # Violin plot of TE_apparent_theoretical_causal_vars and TE_conditional_pairs_theoretical vs. WS_p
    # Subplots vertical: total_cross_coupling
    subplots_v = len(self_couplings)
    subplots_h = len(cross_couplings)
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'TE_apparent_theoretical_causal_vars',
        'TE_conditional_pairs_theoretical',
        ]]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['algorithm'] == algorithm_interest].drop('algorithm', 1)
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Get a numpy array from a list of arrays
    df_interest['TE_apparent_theoretical_causal_vars'] = df_interest['TE_apparent_theoretical_causal_vars'].apply(lambda x: np.array([[np.NaN]]) if type(x) == float else x)
    df_interest['TE_apparent_theoretical_causal_vars'] = df_interest['TE_apparent_theoretical_causal_vars'].apply(lambda x: [np.array([]) if v is None else v for v in x])
    df_interest['TE_apparent_theoretical_causal_vars'] = df_interest['TE_apparent_theoretical_causal_vars'].apply(concat_if_not_empty)
    df_interest['TE_conditional_pairs_theoretical'] = df_interest['TE_conditional_pairs_theoretical'].apply(lambda x: np.array([[np.NaN]]) if type(x) == float else x)
    df_interest['TE_conditional_pairs_theoretical'] = df_interest['TE_conditional_pairs_theoretical'].apply(lambda x: [np.array([]) if v is None else v for v in x])
    df_interest['TE_conditional_pairs_theoretical'] = df_interest['TE_conditional_pairs_theoretical'].apply(concat_if_not_empty)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}
    for (axs_row, self_coupling) in enumerate(self_couplings):
        # Select dataframe entries(i.e runs) with the same total_cross_coupling
        df_self_coupling = df_interest.loc[
            df_interest['self_coupling'] == self_coupling].drop('self_coupling', 1)
        for (axs_col, total_cross_coupling) in enumerate(cross_couplings):
            # Set subplot title
            axs[axs_row, axs_col].set_title('self = {0}, cross = {1}'.format(
                np.around(self_coupling, decimals=2),
                np.around(total_cross_coupling, decimals=2)))
            # Select dataframe entries(i.e runs) with the same total_cross_coupling
            df_cross_coupling = df_self_coupling.loc[
                df_self_coupling['total_cross_coupling'] == total_cross_coupling].drop('total_cross_coupling', 1)
            # Group by WS_p and concatenate TE values lists
            df_aggregate = df_cross_coupling.groupby('WS_p').agg(
                lambda x: x.tolist())
            # Ensure that only the desired parameters are aggregated or averaged
            df_keys_remaining = df_aggregate.columns.get_level_values(0)
            check_remaining_dimensions(df_keys_remaining, parameters_to_average)
            TE_apparent_theoretical_causal_vars = [
                np.concatenate(df_aggregate.loc[WS_p].TE_apparent_theoretical_causal_vars).astype(float)
                    for WS_p in WS_p_range]
            TE_conditional_pairs_theoretical = [
                np.concatenate(df_aggregate.loc[WS_p].TE_conditional_pairs_theoretical).astype(float)
                    for WS_p in WS_p_range]
            # Violin plot of apparent TE vs WS_p
            violin_apparent = axs[axs_row, axs_col].violinplot(
                TE_apparent_theoretical_causal_vars,
                positions=WS_p_range,
                widths=WS_p_range/2,
                showmeans=False,
                showextrema=False,
                showmedians=False,
                #points=100,
                #bw_method=,
                )
            # Only plot right-hand half of the violins
            for b in violin_apparent['bodies']:
                if len(b.get_paths()) > 0:
                    m = np.nanmean(b.get_paths()[0].vertices[:, 0])
                    b.get_paths()[0].vertices[:, 0] = np.clip(
                        b.get_paths()[0].vertices[:, 0], m, np.inf)
                    b.set_color('tab:blue')
            # Join mean values
            mean_vals = [np.nanmean(np.concatenate(df_aggregate.loc[WS_p].TE_apparent_theoretical_causal_vars)) for WS_p in WS_p_range]
            axs[axs_row, axs_col].plot(
                WS_p_range,
                mean_vals,
                '-o',
                color='tab:blue',
                label='Apparent'
            )
            # Violin plot of conditional_pairs TE vs WS_p
            violin_conditional_pairs = axs[axs_row, axs_col].violinplot(
                TE_conditional_pairs_theoretical,
                positions=WS_p_range,
                widths=WS_p_range/2,
                showmeans=False,
                showextrema=False,
                showmedians=False,
                #points=100,
                #bw_method=,
                )
            # Only plot right-hand half of the violins
            for b in violin_conditional_pairs['bodies']:
                if len(b.get_paths()) > 0:
                    m = np.nanmean(b.get_paths()[0].vertices[:, 0])
                    b.get_paths()[0].vertices[:, 0] = np.clip(
                        b.get_paths()[0].vertices[:, 0], m, np.inf)
                    b.set_color('tab:orange')
            # Join mean values
            mean_vals = [np.nanmean(np.concatenate(df_aggregate.loc[WS_p].TE_conditional_pairs_theoretical)) for WS_p in WS_p_range]
            axs[axs_row, axs_col].plot(
                WS_p_range,
                mean_vals,
                '-o',
                color='tab:orange',
                label='Conditional (ordered pairs)'
            )
            # Set axes properties
            axs[axs_row, axs_col].set_xlabel(
                r'$Rewiring\ probability$', horizontalalignment='right', x=1.0)
            axs[axs_row, axs_col].set_ylabel(
                r'$TE\ (theoretical)$')
            axs[axs_row, axs_col].set_xscale('log')
            axs[axs_row, axs_col].yaxis.set_ticks_position('both')
            axs[axs_row, axs_col].legend()
            axs[axs_row, axs_col].set_ylim(bottom=0, top=0.08)
        
    return fig


def plot_omnibus_TE_empirical_vs_theoretical_causal_vars():
    # Scatter TE_omnibus_empirical vs. TE_omnibus_theoretical_causal_vars
    # Subplots vertical: WS_p
    # Subplots horizontal: algorithms
    # Only select TE algorithms
    TE_algorithms = np.extract(['TE' in alg for alg in algorithms], algorithms)
    subplots_v = len(WS_p_range)
    subplots_h = len(TE_algorithms)
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'TE_omnibus_empirical', 'TE_omnibus_theoretical_causal_vars']]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}
    for (axs_row, WS_p) in enumerate(WS_p_range):
        # Select dataframe entries(i.e runs) with the same WS_p
        df_WS_p = df_interest.loc[
            df_interest['WS_p'] == WS_p].drop('WS_p', 1)
        # Group by algorithm and concatenate TE values lists
        df_aggregate = df_WS_p.groupby('algorithm').agg(
            lambda x: x.tolist())
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        for (axs_col, algorithm) in enumerate(TE_algorithms):
            # Set subplot title
            axs[axs_row, axs_col].set_title('p={0}$, ${1}'.format(
                WS_p,
                algorithm))
            if not np.all(np.isnan(df_aggregate.loc[algorithm].TE_omnibus_empirical)):
                TE_omnibus_empirical = np.concatenate(
                    df_aggregate.loc[algorithm].TE_omnibus_empirical)
                TE_omnibus_theoretical_causal_vars = np.concatenate(
                    df_aggregate.loc[algorithm].TE_omnibus_theoretical_causal_vars)
                # Remove NaN values
                #TE_omnibus_empirical = TE_omnibus_empirical[~np.isnan(TE_omnibus_empirical)]
                # Scatter omnibus TE vs theoretical
                axs[axs_row, axs_col].scatter(
                    TE_omnibus_theoretical_causal_vars,
                    TE_omnibus_empirical)
                # Plot identity line
                axs[axs_row, axs_col].plot(
                    [min(TE_omnibus_theoretical_causal_vars), max(TE_omnibus_theoretical_causal_vars)],
                    [min(TE_omnibus_theoretical_causal_vars), max(TE_omnibus_theoretical_causal_vars)],
                    'g--')
            # Set axes properties
            axs[axs_row, axs_col].set_xlabel(
                r'$TE\ theoretical$', horizontalalignment='right', x=1.0)
            axs[axs_row, axs_col].set_ylabel(
                r'$TE\ empirical$')
            # axs[axs_row, axs_col].set_xscale('log')
            axs[axs_row, axs_col].yaxis.set_ticks_position('both')
            axs[axs_row, axs_col].set_xlim(left=0)
            axs[axs_row, axs_col].set_ylim(bottom=0)
    return fig


def plot_relative_error_TE_empirical_vs_theoretical_causal_vars_vs_WS_p():
    # Plot absolute diff between TE empirical and TE theoretical
    # (relative to TE theoretical) vs. WS_p
    # (aggregate over N)
    fig = plt.figure()
    axs = plt.gca()
    axs = np.array([axs])
    # Select data of interest
    df_interest = df[parameters_explored + [
        'TE_omnibus_empirical', 'TE_omnibus_theoretical_causal_vars']]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
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
    # Only select TE algorithms
    TE_algorithms = np.extract(['TE' in alg for alg in algorithms], algorithms)
    for algorithm in TE_algorithms:
        # Select dataframe entries(i.e runs) with the same number of samples
        df_algorithm = df_interest.loc[
            df_interest['algorithm'] == algorithm].drop('algorithm', 1)
        # Group by WS_p
        df_aggregate = df_algorithm.groupby('WS_p').agg(
            lambda x: list(x))
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        axs[0].plot(
            df_aggregate.index.tolist(),
            [np.nanmean(df_aggregate['TE_err_abs_relative'][WS_p])
                for WS_p in WS_p_range],
            label=algorithm_names[algorithm])
        axs[0].scatter(
            df_algorithm['WS_p'].values.tolist(),
            df_algorithm['TE_err_abs_relative'].values,
            alpha=0.3)
    # Set axes properties
    axs[0].set_xlabel(r'$Rewiring\ probability$', horizontalalignment='right', x=1.0)
    axs[0].set_ylabel(r'$Relative\ TE\ error$')
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].yaxis.set_ticks_position('both')
    axs[0].legend()

    return fig


def plot_spectral_radius_vs_WS_p_alpha_interest():
    # Plot spectral radius vs. Watts-Strogatz rewiring probability
    fig, axs = my_subplots(1, 1, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[
        parameters_explored + ['spectral_radius']]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    df_interest = df_interest.loc[
        df_interest['algorithm'] == 'mTE_greedy'].drop('algorithm', 1)
    # Convert remaining DataFrame to float type for averaging
    df_interest = df_interest.astype(float)
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'repetition_i'}
    # Group by WS_p
    df_aggregate = df_interest.groupby('WS_p').agg(
        lambda x: list(x))
    # Ensure that only the desired parameters are aggregated or averaged
    df_keys_remaining = df_aggregate.columns.get_level_values(0)
    check_remaining_dimensions(df_keys_remaining, parameters_to_average)
    # Plot spectral radius
    axs[0].plot(
        WS_p_range,
        [np.nanmean(df_aggregate['spectral_radius'][WS_p])
            for WS_p in WS_p_range],
    )
    axs[0].scatter(
        df_interest['WS_p'].values.astype(float).tolist(),
        df_interest['spectral_radius'].values,
        alpha=0.3
    )
    # Set axes properties
    axs[0].set_xlabel(r'$Rewiring\ probability$', horizontalalignment='right', x=1.0)
    axs[0].set_ylabel(r'$\rho$')
    axs[0].set_xscale('log')
    axs[0].yaxis.set_ticks_position('both')

    return fig


def plot_performance_vs_WS_p():
    # Plot performance tests vs. Watts-Strogatz rewiring
    # probability (scatter error)
    # Subplots: algorithms
    subplots_v = len(algorithms)
    subplots_h = 1
    fig, axs = my_subplots(subplots_v, subplots_h)
    # Select data of interest
    df_interest = df[
        parameters_explored + ['precision', 'recall', 'specificity']]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'repetition_i'}
    for (axs_row, algorithm) in enumerate(algorithms):
        # Set subplot title
        axs[axs_row].set_title('{0}'.format(algorithm_names[algorithm]))
        # Select dataframe entries(i.e runs) with the same algorithm
        df_algorithm = df_interest.loc[
            df_interest['algorithm'] == algorithm].drop('algorithm', 1)
         # Convert remaining DataFrame to float type for averaging
        # df_interest = df_interest.astype(float)
        # Group by WS_p
        df_aggregate = df_algorithm.groupby('WS_p').agg(
            lambda x: list(x))
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        # Plot precision
        axs[axs_row].plot(
            WS_p_range,
            [np.nanmean(df_aggregate['precision'][WS_p])
                for WS_p in WS_p_range],
        )
        axs[axs_row].scatter(
            df_algorithm['WS_p'].values.astype(float).tolist(),
            df_algorithm['precision'].values,
            alpha=0.3
        )
        # Plot recall
        axs[axs_row].plot(
            WS_p_range,
            [np.nanmean(df_aggregate['recall'][WS_p])
             for WS_p in WS_p_range],
        )
        axs[axs_row].scatter(
            df_algorithm['WS_p'].values.astype(float).tolist(),
            df_algorithm['recall'].values,
            alpha=0.3
        )
        # Plot false positive rate
        axs[axs_row].plot(
            WS_p_range,
            [np.nanmean(df_aggregate['specificity'][WS_p])
             for WS_p in WS_p_range],
        )
        axs[axs_row].scatter(
            df_algorithm['WS_p'].values.astype(float).tolist(),
            df_algorithm['specificity'].values,
            alpha=0.3
        )
        # Set axes properties
        axs[axs_row].set_xlabel(r'$Rewiring\ probability$', horizontalalignment='right', x=1.0)
        axs[axs_row].set_ylabel(r'$Performance$')
        axs[axs_row].set_xscale('log')
        axs[axs_row].yaxis.set_ticks_position('both')
        axs[axs_row].set_ylim(bottom=0)
        # Add legend
        axs[axs_row].legend(
            labels=['Precision', 'Recall', 'Specificity']#,
            #loc=(0.05, 0.16)
            )

    return fig


def plot_density_vs_WS_p():
    # Plot density vs. Watts-Strogatz rewiring prob
    # Legend: algorithms
    fig, axs = my_subplots(1, 1, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'density_real',
        'density_inferred']]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'repetition_i'}
    # Plot average path lenght real
    df_algorithm = df_interest.loc[df_interest['algorithm'] == algorithms[-1]]
    df_aggregate = df_algorithm.groupby('WS_p').agg(lambda x: list(x))
    axs[0].plot(
        WS_p_range,
        [np.nanmean(df_aggregate['density_real'][WS_p])
            for WS_p in WS_p_range],
        label='Real')
    axs[0].scatter(
        df_algorithm['WS_p'].values.astype(float).tolist(),
        df_algorithm['density_real'].values,
        alpha=0.3)
    for algorithm in algorithms:
        # Select dataframe entries(i.e runs) with the same number of samples
        df_algorithm = df_interest.loc[
            df_interest['algorithm'] == algorithm].drop('algorithm', 1)
        # Group by WS_p
        df_aggregate = df_algorithm.groupby('WS_p').agg(
            lambda x: list(x))
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        # Plot average path lenght inferred
        axs[0].plot(
            WS_p_range,
            [np.nanmean(df_aggregate['density_inferred'][WS_p])
             for WS_p in WS_p_range],
            label=algorithm_names[algorithm])
        axs[0].scatter(
            df_algorithm['WS_p'].values.astype(float).tolist(),
            df_algorithm['density_inferred'].values,
            alpha=0.3)
    # Set axes properties
    axs[0].set_xlabel(r'$Rewiring\ probability$', horizontalalignment='right', x=1.0)
    axs[0].set_ylabel(r'$Density$')
    axs[0].set_xscale('log')
    axs[0].yaxis.set_ticks_position('both')
    axs[0].set_ylim(bottom=0)
    # Add legend
    axs[0].legend()

    return fig


def scatter_clustering_inferred_vs_clustering_real():
    # Scatter clustering inferred vs. real
    # Subplots vertical: WS_p
    # Subplots horizontal: algorithms
    subplots_v = len(WS_p_range)
    subplots_h = len(algorithms)
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'clustering_inferred',
        'clustering_real',
        ]]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}
    for (axs_row, WS_p) in enumerate(WS_p_range):
        # Select dataframe entries(i.e runs) with the same WS_p
        df_WS_p = df_interest.loc[
            df_interest['WS_p'] == WS_p].drop('WS_p', 1)
        # Group by algorithm and concatenate TE values lists
        df_aggregate = df_WS_p.groupby('algorithm').agg(
            lambda x: x.tolist())
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        for (axs_col, algorithm) in enumerate(algorithms):
            # Set subplot title
            axs[axs_row, axs_col].set_title('p={0}$, ${1}'.format(
                WS_p,
                algorithm))
            clustering_inferred = np.concatenate(
                df_aggregate.loc[algorithm].clustering_inferred)
            clustering_real = np.concatenate(
                df_aggregate.loc[algorithm].clustering_real)
            # Scatter
            axs[axs_row, axs_col].scatter(
                clustering_real,
                clustering_inferred,
                )
            # Plot identity line
            axs[axs_row, axs_col].plot(
                [min(clustering_real), max(clustering_real)],
                [min(clustering_real), max(clustering_real)],
                'g--')
            # Set axes properties
            axs[axs_row, axs_col].set_xlabel(
                r'$Clustering\ real$', horizontalalignment='right', x=1.0)
            axs[axs_row, axs_col].set_ylabel(
                r'$Clustering\ inferred$')
            # axs[axs_row, axs_col].set_xscale('log')
            axs[axs_row, axs_col].yaxis.set_ticks_position('both')
            #axs[axs_row, axs_col].set_xlim(left=0)
            #axs[axs_row, axs_col].set_ylim(bottom=0)
    return fig


def plot_clustering_CDF():
    # Plot CDF of clustering coefficient for different algorithms
    # Subplots vertical: WS_p
    subplots_v = len(WS_p_range)
    subplots_h = 1
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'clustering_real',
        'clustering_inferred']]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}
    for (axs_row, WS_p) in enumerate(WS_p_range):
        # Set subplot title
        axs[axs_row].set_title(r'$p={0}$'.format(WS_p))
        # Select dataframe entries(i.e runs) with the same WS_p
        df_WS_p = df_interest.loc[
            df_interest['WS_p'] == WS_p].drop('WS_p', 1)
        # Group by algorithm and concatenate TE values lists
        df_aggregate = df_WS_p.groupby('algorithm').agg(
            lambda x: x.tolist())
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        # Plot CDF
        df_algorithm = df_WS_p.loc[df_WS_p['algorithm'] == algorithms[0]]
        clustering_real = np.concatenate(df_algorithm.clustering_real)
        clustering_real_sorted = np.sort(clustering_real)
        # y = np.arange(len(x)) / float(len(x))
        # y = (np.arange(len(x)) + 0.5) / len(x)
        cdf = np.arange(1 , len(clustering_real) + 1) / float(len(clustering_real))
        axs[axs_row].plot(
            clustering_real_sorted,
            cdf,
            label='Real'
            )
        # # Interpolate
        # x_unique = [clustering_real_sorted[0]]
        # y_unique = [cdf[0]]
        # for (i, x) in enumerate(clustering_real_sorted):
        #     if x > x_unique[-1]:
        #         x_unique.append(x)
        #         y_unique.append(cdf[i])
        # xnew = np.linspace(clustering_real.min(), clustering_real.max(), 500)
        # interp_function = interp1d(x_unique, y_unique, kind='slinear')
        # axs[axs_row].plot(xnew, interp_function(xnew))

        for algorithm in algorithms:
            df_algorithm = df_WS_p.loc[df_WS_p['algorithm'] == algorithm]
            clustering_inferred = np.concatenate(df_algorithm.clustering_inferred)
            # Plot CDF
            clustering_inferred_sorted = np.sort(clustering_inferred)
            # y = np.arange(len(x)) / float(len(x))
            # y = (np.arange(len(x)) + 0.5) / len(x)
            cdf = np.arange(1 , len(clustering_inferred) + 1) / float(len(clustering_inferred))
            axs[axs_row].plot(
                clustering_inferred_sorted,
                cdf,
                label=algorithm_names[algorithm]
                )
            # # Interpolate
            # x_unique = [clustering_inferred_sorted[0]]
            # y_unique = [cdf[0]]
            # for (i, x) in enumerate(clustering_inferred_sorted):
            #     if x > x_unique[-1]:
            #         x_unique.append(x)
            #         y_unique.append(cdf[i])
            # xnew = np.linspace(clustering_inferred.min(), clustering_inferred.max(), 500)
            # interp_function = interp1d(x_unique, y_unique, kind='slinear')
            # axs[axs_row].plot(xnew, interp_function(xnew))
        # Set axes properties
        axs[axs_row].set_xlabel(
            r'$Clustering\ coefficient$', horizontalalignment='right', x=1.0)
        axs[axs_row].set_ylabel(
            r'$CDF$')
        # axs[axs_row, axs_col].set_xscale('log')
        axs[axs_row].yaxis.set_ticks_position('both')
        #axs[axs_row].set_xlim(left=0)
        #axs[axs_row].set_ylim(bottom=0)
        axs[axs_row].legend()
    return fig


def plot_clustering_vs_WS_p_subplot_algorithm():
    # Plot average clustering vs. Watts-Strogatz rewiring prob (scatter error)
    # Subplots: algorithms
    subplots_v = len(algorithms)
    subplots_h = 1
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'clustering_real',
        'clustering_inferred']]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'repetition_i'}
    for (axs_row, algorithm) in enumerate(algorithms):
        # Set subplot title
        axs[axs_row].set_title('{0}'.format(algorithm_names[algorithm]))
        # Select dataframe entries(i.e runs) with the same number of samples
        df_algorithm = df_interest.loc[
            df_interest['algorithm'] == algorithm].drop('algorithm', 1)
        # Group by WS_p
        df_aggregate = df_algorithm.groupby('WS_p').agg(
            lambda x: list(x))
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        # Plot average clustering coefficient real
        axs[axs_row].plot(
            WS_p_range,
            [np.nanmean(df_aggregate['clustering_real'][WS_p])
                for WS_p in WS_p_range],
        )
        axs[axs_row].scatter(
            df_algorithm['WS_p'].values.astype(float).tolist(),
            [np.nanmean(x) for x in df_algorithm['clustering_real'].values],
            alpha=0.3
        )
        # Plot average clustering coefficient inferred
        axs[axs_row].plot(
            WS_p_range,
            [np.nanmean(df_aggregate['clustering_inferred'][WS_p])
             for WS_p in WS_p_range],
        )
        axs[axs_row].scatter(
            df_algorithm['WS_p'].values.astype(float).tolist(),
            [np.nanmean(x) for x in df_algorithm['clustering_inferred'].values],
            alpha=0.3
        )
        # Set axes properties
        axs[axs_row].set_xlabel(r'$Rewiring\ probability$', horizontalalignment='right', x=1.0)
        axs[axs_row].set_ylabel(r'$Clustering\ coefficient$')
        axs[axs_row].set_xscale('log')
        axs[axs_row].yaxis.set_ticks_position('both')
        axs[axs_row].set_ylim(bottom=0)
        # Add legend
        axs[axs_row].legend(
            labels=['Real', 'Inferred'],
            #loc=(0.05, 0.16)
            )

    return fig


def plot_clustering_vs_WS_p():
    # Plot average clustering vs. Watts-Strogatz rewiring prob (scatter error)
    fig, axs = my_subplots(1, 1, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'clustering_real',
        'clustering_inferred']]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'repetition_i'}
    # Plot average clustering coefficient real
    df_algorithm = df_interest.loc[df_interest['algorithm'] == algorithms[-1]]
    df_aggregate = df_algorithm.groupby('WS_p').agg(lambda x: list(x))
    axs[0].plot(
        WS_p_range,
        [np.nanmean(df_aggregate['clustering_real'][WS_p])
            for WS_p in WS_p_range],
        label='Real')
    axs[0].scatter(
        df_algorithm['WS_p'].values.astype(float).tolist(),
        [np.nanmean(x) for x in df_algorithm['clustering_real'].values],
        alpha=0.3)
    for algorithm in algorithms:
        # Select dataframe entries(i.e runs) with the same number of samples
        df_algorithm = df_interest.loc[
            df_interest['algorithm'] == algorithm].drop('algorithm', 1)
        # Group by WS_p
        df_aggregate = df_algorithm.groupby('WS_p').agg(
            lambda x: list(x))
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        # Plot average clustering coefficient inferred
        axs[0].plot(
            WS_p_range,
            [np.nanmean(df_aggregate['clustering_inferred'][WS_p])
             for WS_p in WS_p_range],
            label=algorithm_names[algorithm])
        axs[0].scatter(
            df_algorithm['WS_p'].values.astype(float).tolist(),
            [np.nanmean(x) for x in df_algorithm['clustering_inferred'].values],
            alpha=0.3)
        # Set axes properties
        axs[0].set_xlabel(r'$Rewiring\ probability$', horizontalalignment='right', x=1.0)
        axs[0].set_ylabel(r'$Clustering\ coefficient$')
        axs[0].set_xscale('log')
        axs[0].yaxis.set_ticks_position('both')
        axs[0].set_ylim(bottom=0, top=0.7)
        # Add legend
        axs[0].legend()

    return fig


def plot_clustering_correlation_vs_WS_p():
    # Plot correlation of real and real clustering coefficient
    # vs. Watts-Strogatz rewiring prob (scatter error)
    fig, axs = my_subplots(1, 1, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'clustering_real',
        'clustering_inferred']]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'repetition_i'}
    # Compute correlation between real and inferred clustering
    for row in df_interest.itertuples():
        df_interest.loc[row.Index, 'correlation'] = (
            np.corrcoef(row.clustering_real, row.clustering_inferred)[0, 1])
    for algorithm in algorithms:
        # Select dataframe entries(i.e runs) with the same number of samples
        df_algorithm = df_interest.loc[
            df_interest['algorithm'] == algorithm].drop('algorithm', 1)
        # Group by WS_p
        df_aggregate = df_algorithm.groupby('WS_p').agg(
            lambda x: list(x))
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        # Plot average correlation
        axs[0].plot(
            WS_p_range,
            [np.nanmean(df_aggregate['correlation'][WS_p]) for WS_p in WS_p_range],
            label=algorithm_names[algorithm])
        # Scatter values over repetitions
        axs[0].scatter(
            df_algorithm['WS_p'].values.astype(float).tolist(),
            df_algorithm['correlation'].values,
            alpha=0.3)
        # Set axes properties
        axs[0].set_xlabel(r'$Rewiring\ probability$', horizontalalignment='right', x=1.0)
        axs[0].set_ylabel(r'$Correlation$')
        axs[0].set_xscale('log')
        axs[0].yaxis.set_ticks_position('both')
        # Add legend
        axs[0].legend()

    return fig


def scatter_path_lenght_inferred_vs_path_lenght_real():
    # Scatter average shortest path lenght inferred vs. real
    # Subplots vertical: WS_p
    # Subplots horizontal: algorithms
    subplots_v = len(WS_p_range)
    subplots_h = len(algorithms)
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'average_shortest_path_length_inferred',
        'average_shortest_path_length_real',
        ]]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}
    for (axs_row, WS_p) in enumerate(WS_p_range):
        # Select dataframe entries(i.e runs) with the same WS_p
        df_WS_p = df_interest.loc[
            df_interest['WS_p'] == WS_p].drop('WS_p', 1)
        # Group by algorithm and concatenate TE values lists
        df_aggregate = df_WS_p.groupby('algorithm').agg(
            lambda x: x.tolist())
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        for (axs_col, algorithm) in enumerate(algorithms):
            # Set subplot title
            axs[axs_row, axs_col].set_title('p={0}$, ${1}'.format(
                WS_p,
                algorithm))
            path_lenght_inferred = df_aggregate.loc[algorithm].average_shortest_path_length_inferred
            path_lenght_real = df_aggregate.loc[algorithm].average_shortest_path_length_real
            # Scatter
            axs[axs_row, axs_col].scatter(
                path_lenght_real,
                path_lenght_inferred,
                )
            # Plot identity line
            axs[axs_row, axs_col].plot(
                [min(path_lenght_real), max(path_lenght_real)],
                [min(path_lenght_real), max(path_lenght_real)],
                'g--')
            # Set axes properties
            axs[axs_row, axs_col].set_xlabel(
                r'$Shortest\ path\ real$', horizontalalignment='right', x=1.0)
            axs[axs_row, axs_col].set_ylabel(
                r'$Shortest\ path\ inferred$')
            # axs[axs_row, axs_col].set_xscale('log')
            axs[axs_row, axs_col].yaxis.set_ticks_position('both')
            #axs[axs_row, axs_col].set_xlim(left=0)
            #axs[axs_row, axs_col].set_ylim(bottom=0)
    return fig


def plot_path_lenght_vs_WS_p_subplot_algorithm():
    # Plot average path lenght vs. Watts-Strogatz rewiring prob (scatter error)
    # Subplots: algorithm
    subplots_v = len(algorithms)
    subplots_h = 1
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Avoid indexing issues if only one row or one column
    if subplots_v == 1:
        print('One vertical subplot only')
        axs = np.array([axs])
    # Select data of interest
    df_interest = df[parameters_explored + [
        'average_shortest_path_length_real',
        'average_shortest_path_length_inferred']]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Convert remaining DataFrame to float type for averaging
    # df_interest = df_interest.astype(float)
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'repetition_i'}
    for (axs_row, algorithm) in enumerate(algorithms):
        # Set subplot title
        axs[axs_row].set_title('{0}'.format(algorithm_names[algorithm]))
        # Select dataframe entries(i.e runs) with the same number of samples
        df_algorithm = df_interest.loc[
            df_interest['algorithm'] == algorithm].drop('algorithm', 1)
        # Group by WS_p
        df_aggregate = df_algorithm.groupby('WS_p').agg(
            lambda x: list(x))
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        # Plot average path lenght real
        axs[axs_row].plot(
            WS_p_range,
            [np.nanmean(df_aggregate['average_shortest_path_length_real'][WS_p])
                for WS_p in WS_p_range],
        )
        axs[axs_row].scatter(
            df_algorithm['WS_p'].values.astype(float).tolist(),
            df_algorithm['average_shortest_path_length_real'].values,
            alpha=0.3
        )
        # Plot average path lenght inferred
        axs[axs_row].plot(
            WS_p_range,
            [np.nanmean(df_aggregate['average_shortest_path_length_inferred'][WS_p])
             for WS_p in WS_p_range],
        )
        axs[axs_row].scatter(
            df_algorithm['WS_p'].values.astype(float).tolist(),
            df_algorithm['average_shortest_path_length_inferred'].values,
            alpha=0.3
        )
        # Set axes properties
        axs[axs_row].set_xlabel(r'$Rewiring\ probability$', horizontalalignment='right', x=1.0)
        axs[axs_row].set_ylabel(r'$Shortest\ path\ length$')
        axs[axs_row].set_xscale('log')
        axs[axs_row].yaxis.set_ticks_position('both')
        axs[axs_row].set_ylim(bottom=0)
        # Add legend
        axs[axs_row].legend(
            labels=[
                'Real',
                'Inferred'
            ]#,
            #loc=(0.05, 0.16)
            )

    return fig


def plot_path_lenght_vs_WS_p():
    # Plot average path lenght vs. Watts-Strogatz rewiring prob (scatter error)
    fig, axs = my_subplots(1, 1, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'average_shortest_path_length_real',
        'average_shortest_path_length_inferred']]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'repetition_i'}
    # Plot average path lenght real
    df_algorithm = df_interest.loc[df_interest['algorithm'] == algorithms[-1]]
    df_aggregate = df_algorithm.groupby('WS_p').agg(lambda x: list(x))
    axs[0].plot(
        WS_p_range,
        [np.nanmean(df_aggregate['average_shortest_path_length_real'][WS_p])
            for WS_p in WS_p_range],
        label='Real')
    axs[0].scatter(
        df_algorithm['WS_p'].values.astype(float).tolist(),
        df_algorithm['average_shortest_path_length_real'].values,
        alpha=0.3)
    for algorithm in algorithms:
        # Select dataframe entries(i.e runs) with the same number of samples
        df_algorithm = df_interest.loc[
            df_interest['algorithm'] == algorithm].drop('algorithm', 1)
        # Group by WS_p
        df_aggregate = df_algorithm.groupby('WS_p').agg(
            lambda x: list(x))
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        # Plot average path lenght inferred
        axs[0].plot(
            WS_p_range,
            [np.nanmean(df_aggregate['average_shortest_path_length_inferred'][WS_p])
             for WS_p in WS_p_range],
            label=algorithm_names[algorithm]
        )
        axs[0].scatter(
            df_algorithm['WS_p'].values.astype(float).tolist(),
            df_algorithm['average_shortest_path_length_inferred'].values,
            alpha=0.3
        )
        # Set axes properties
        axs[0].set_xlabel(r'$Rewiring\ probability$', horizontalalignment='right', x=1.0)
        axs[0].set_ylabel(r'$Shortest\ path\ length$')
        axs[0].set_xscale('log')
        axs[0].yaxis.set_ticks_position('both')
        axs[0].set_ylim(bottom=0)
        # Add legend
        axs[0].legend()

    return fig


def plot_average_global_efficiency_vs_WS_p():
    # Plot global_efficiency vs. Watts-Strogatz rewiring prob (scatter error)
    fig, axs = my_subplots(1, 1, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'average_global_efficiency_real',
        'average_global_efficiency_inferred']]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'repetition_i'}
    # Plot global_efficiency real
    df_algorithm = df_interest.loc[df_interest['algorithm'] == algorithms[-1]]
    df_aggregate = df_algorithm.groupby('WS_p').agg(lambda x: list(x))
    axs[0].plot(
        WS_p_range,
        [np.nanmean(df_aggregate['average_global_efficiency_real'][WS_p])
            for WS_p in WS_p_range],
        label='Real')
    axs[0].scatter(
        df_algorithm['WS_p'].values.astype(float).tolist(),
        df_algorithm['average_global_efficiency_real'].values,
        alpha=0.3)
    for algorithm in algorithms:
        # Select dataframe entries(i.e runs) with the same number of samples
        df_algorithm = df_interest.loc[
            df_interest['algorithm'] == algorithm].drop('algorithm', 1)
        # Group by WS_p
        df_aggregate = df_algorithm.groupby('WS_p').agg(
            lambda x: list(x))
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        # Plot global_efficiency inferred
        axs[0].plot(
            WS_p_range,
            [np.nanmean(df_aggregate['average_global_efficiency_inferred'][WS_p])
             for WS_p in WS_p_range],
            label=algorithm_names[algorithm])
        axs[0].scatter(
            df_algorithm['WS_p'].values.astype(float).tolist(),
            df_algorithm['average_global_efficiency_inferred'].values,
            alpha=0.3)
        # Set axes properties
        axs[0].set_xlabel(r'$Rewiring\ probability$', horizontalalignment='right', x=1.0)
        axs[0].set_ylabel(r'$Global\ efficiency$')
        axs[0].set_xscale('log')
        axs[0].yaxis.set_ticks_position('both')
        axs[0].set_ylim(bottom=0)
        # Add legend
        axs[0].legend()

    return fig


def plot_local_efficiency_vs_WS_p():
    # Plot local efficiency vs. Watts-Strogatz rewiring prob (scatter error)
    fig, axs = my_subplots(1, 1, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'local_efficiency_real',
        'local_efficiency_inferred']]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'repetition_i'}
    # Real
    df_algorithm = df_interest.loc[df_interest['algorithm'] == algorithms[-1]]
    df_aggregate = df_algorithm.groupby('WS_p').agg(lambda x: list(x))
    axs[0].plot(
        WS_p_range,
        [np.nanmean(df_aggregate['local_efficiency_real'][WS_p])
            for WS_p in WS_p_range],
        label='Real')
    axs[0].scatter(
        df_algorithm['WS_p'].values.astype(float).tolist(),
        [np.nanmean(x) for x in df_algorithm['local_efficiency_real'].values],
        alpha=0.3)
    for algorithm in algorithms:
        # Select dataframe entries(i.e runs) with the same number of samples
        df_algorithm = df_interest.loc[
            df_interest['algorithm'] == algorithm].drop('algorithm', 1)
        # Group by WS_p
        df_aggregate = df_algorithm.groupby('WS_p').agg(
            lambda x: list(x))
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        # Inferred
        axs[0].plot(
            WS_p_range,
            [np.nanmean(df_aggregate['local_efficiency_inferred'][WS_p])
             for WS_p in WS_p_range],
            label=algorithm_names[algorithm])
        axs[0].scatter(
            df_algorithm['WS_p'].values.astype(float).tolist(),
            [np.nanmean(x) for x in df_algorithm['local_efficiency_inferred'].values],
            alpha=0.3)
        # Set axes properties
        axs[0].set_xlabel(r'$Rewiring\ probability$', horizontalalignment='right', x=1.0)
        axs[0].set_ylabel(r'$Local\ efficiency$')
        axs[0].set_xscale('log')
        axs[0].yaxis.set_ticks_position('both')
        axs[0].set_ylim(bottom=0)
        # Add legend
        axs[0].legend()

    return fig


def plot_local_efficiency_correlation_vs_WS_p():
    # Plot correlation of real and inferred local efficiency
    # vs. Watts-Strogatz rewiring prob (scatter error)
    fig, axs = my_subplots(1, 1, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'local_efficiency_real',
        'local_efficiency_inferred']]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'repetition_i'}
    # Compute correlation between real and inferred local efficiency
    for row in df_interest.itertuples():
        df_interest.loc[row.Index, 'correlation'] = (
            np.corrcoef(row.local_efficiency_real, row.local_efficiency_inferred)[0, 1])
    for algorithm in algorithms:
        # Select dataframe entries(i.e runs) with the same number of samples
        df_algorithm = df_interest.loc[
            df_interest['algorithm'] == algorithm].drop('algorithm', 1)
        # Group by WS_p
        df_aggregate = df_algorithm.groupby('WS_p').agg(
            lambda x: list(x))
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        # Plot average correlation
        axs[0].plot(
            WS_p_range,
            [np.nanmean(df_aggregate['correlation'][WS_p]) for WS_p in WS_p_range],
            label=algorithm_names[algorithm])
        # Scatter values over repetitions
        axs[0].scatter(
            df_algorithm['WS_p'].values.astype(float).tolist(),
            df_algorithm['correlation'].values,
            alpha=0.3)
        # Set axes properties
        axs[0].set_xlabel(r'$Rewiring\ probability$', horizontalalignment='right', x=1.0)
        axs[0].set_ylabel(r'$Correlation$')
        axs[0].set_xscale('log')
        axs[0].yaxis.set_ticks_position('both')
        # Add legend
        axs[0].legend()

    return fig


def plot_SW_index_vs_WS_p(WS_k=4):
    # Plot average clustering vs. Watts-Strogatz rewiring prob (scatter error)
    fig, axs = my_subplots(1, 1, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'clustering_real',
        'clustering_inferred',
        'average_shortest_path_length_real',
        'average_shortest_path_length_inferred']]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'repetition_i'}
    # Compute normalisation coefficients
    normalisation_clustering = (3 * WS_k - 6) / (4 * WS_k - 4)
    normalisation_path_length = N_interest / (2 * WS_k)
    # Plot SW index real
    df_algorithm = df_interest.loc[df_interest['algorithm'] == algorithms[-1]]
    df_aggregate = df_algorithm.groupby('WS_p').agg(lambda x: list(x))
    axs[0].plot(
        WS_p_range,
        (np.array([np.nanmean(df_aggregate['clustering_real'][WS_p]) for WS_p in WS_p_range]) / normalisation_clustering) / (np.array([np.nanmean(df_aggregate['average_shortest_path_length_real'][WS_p]) for WS_p in WS_p_range]) / normalisation_path_length),
        label='Real')
    axs[0].scatter(
        df_algorithm['WS_p'].values.astype(float).tolist(),
        [np.nanmean(x) / normalisation_clustering for x in df_algorithm['clustering_real'].values] / (df_algorithm['average_shortest_path_length_real'].to_numpy() / normalisation_path_length),
        alpha=0.3)
    for algorithm in algorithms:
        # Select dataframe entries(i.e runs) with the same algorithm
        df_algorithm = df_interest.loc[
            df_interest['algorithm'] == algorithm].drop('algorithm', 1)
        # Group by WS_p
        df_aggregate = df_algorithm.groupby('WS_p').agg(
            lambda x: list(x))
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        # Plot SW index inferred
        axs[0].plot(
            WS_p_range,
            (np.array([np.nanmean(df_aggregate['clustering_inferred'][WS_p]) for WS_p in WS_p_range]) / normalisation_clustering) / (np.array([np.nanmean(df_aggregate['average_shortest_path_length_inferred'][WS_p]) for WS_p in WS_p_range]) / normalisation_path_length),
            label=algorithm_names[algorithm])
        axs[0].scatter(
            df_algorithm['WS_p'].values.astype(float).tolist(),
            [np.nanmean(x) / normalisation_clustering for x in df_algorithm['clustering_inferred'].values] / (df_algorithm['average_shortest_path_length_inferred'].to_numpy() / normalisation_path_length),
            alpha=0.3)
    # Set axes properties
    axs[0].set_xlabel(r'$Rewiring\ probability$', horizontalalignment='right', x=1.0)
    axs[0].set_ylabel(r'$Small-world\ index$')
    axs[0].set_xscale('log')
    axs[0].yaxis.set_ticks_position('both')
    axs[0].set_ylim(bottom=0)
    # Add legend
    axs[0].legend()

    return fig


def scatter_out_degree_inferred_vs_out_degree_real():
    # Scatter out-degree inferred vs. real
    # Subplots vertical: WS_p
    # Subplots horizontal: algorithms
    subplots_v = len(WS_p_range)
    subplots_h = len(algorithms)
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'out_degree_inferred',
        'out_degree_real',
        ]]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}
    for (axs_row, WS_p) in enumerate(WS_p_range):
        # Select dataframe entries(i.e runs) with the same WS_p
        df_WS_p = df_interest.loc[
            df_interest['WS_p'] == WS_p].drop('WS_p', 1)
        # Group by algorithm and concatenate TE values lists
        df_aggregate = df_WS_p.groupby('algorithm').agg(
            lambda x: x.tolist())
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        for (axs_col, algorithm) in enumerate(algorithms):
            # Set subplot title
            axs[axs_row, axs_col].set_title('p={0}$, ${1}'.format(
                WS_p,
                algorithm))
            out_degree_inferred = np.concatenate(
                df_aggregate.loc[algorithm].out_degree_inferred)
            out_degree_real = np.concatenate(
                df_aggregate.loc[algorithm].out_degree_real)
            # Scatter
            axs[axs_row, axs_col].scatter(
                out_degree_real,
                out_degree_inferred,
                )
            # Plot identity line
            axs[axs_row, axs_col].plot(
                [min(out_degree_real), max(out_degree_real)],
                [min(out_degree_real), max(out_degree_real)],
                'g--')
            # Set axes properties
            axs[axs_row, axs_col].set_xlabel(
                r'$Out-degree\ real$', horizontalalignment='right', x=1.0)
            axs[axs_row, axs_col].set_ylabel(
                r'$Out-degree\ inferred$')
            # axs[axs_row, axs_col].set_xscale('log')
            axs[axs_row, axs_col].yaxis.set_ticks_position('both')
            #axs[axs_row, axs_col].set_xlim(left=0)
            #axs[axs_row, axs_col].set_ylim(bottom=0)
    return fig


def plot_out_degree_CDF():
    # Plot CDF of out-degree for different algorithms
    # Subplots vertical: WS_p
    subplots_v = len(WS_p_range)
    subplots_h = 1
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'out_degree_real',
        'out_degree_inferred']]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}
    for (axs_row, WS_p) in enumerate(WS_p_range):
        # Set subplot title
        axs[axs_row].set_title(r'$p={0}$'.format(WS_p))
        # Select dataframe entries(i.e runs) with the same WS_p
        df_WS_p = df_interest.loc[
            df_interest['WS_p'] == WS_p].drop('WS_p', 1)
        # Group by algorithm and concatenate TE values lists
        df_aggregate = df_WS_p.groupby('algorithm').agg(
            lambda x: x.tolist())
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        # Plot CDF
        df_algorithm = df_WS_p.loc[df_WS_p['algorithm'] == algorithms[0]]
        out_degree_real = np.concatenate(df_algorithm.out_degree_real)
        out_degree_real_sorted = np.sort(out_degree_real)
        # y = np.arange(len(x)) / float(len(x))
        # y = (np.arange(len(x)) + 0.5) / len(x)
        cdf = np.arange(1 , len(out_degree_real) + 1) / float(len(out_degree_real))
        axs[axs_row].plot(
            out_degree_real_sorted,
            cdf,
            label='Real'
            )
        # # Interpolate
        # x_unique = [out_degree_real_sorted[0]]
        # y_unique = [cdf[0]]
        # for (i, x) in enumerate(out_degree_real_sorted):
        #     if x > x_unique[-1]:
        #         x_unique.append(x)
        #         y_unique.append(cdf[i])
        # xnew = np.linspace(out_degree_real.min(), out_degree_real.max(), 500)
        # interp_function = interp1d(x_unique, y_unique, kind='slinear')
        # axs[axs_row].plot(xnew, interp_function(xnew))

        for algorithm in algorithms:
            df_algorithm = df_WS_p.loc[df_WS_p['algorithm'] == algorithm]
            out_degree_inferred = np.concatenate(df_algorithm.out_degree_inferred)
            # Plot CDF
            out_degree_inferred_sorted = np.sort(out_degree_inferred)
            # y = np.arange(len(x)) / float(len(x))
            # y = (np.arange(len(x)) + 0.5) / len(x)
            cdf = np.arange(1 , len(out_degree_inferred) + 1) / float(len(out_degree_inferred))
            axs[axs_row].plot(
                out_degree_inferred_sorted,
                cdf,
                label=algorithm_names[algorithm]
                )
            # # Interpolate
            # x_unique = [out_degree_inferred_sorted[0]]
            # y_unique = [cdf[0]]
            # for (i, x) in enumerate(out_degree_inferred_sorted):
            #     if x > x_unique[-1]:
            #         x_unique.append(x)
            #         y_unique.append(cdf[i])
            # xnew = np.linspace(out_degree_inferred.min(), out_degree_inferred.max(), 500)
            # interp_function = interp1d(x_unique, y_unique, kind='slinear')
            # axs[axs_row].plot(xnew, interp_function(xnew))
        # Set axes properties
        axs[axs_row].set_xlabel(
            r'$Out-degree$', horizontalalignment='right', x=1.0)
        axs[axs_row].set_ylabel(
            r'$CDF$')
        # axs[axs_row, axs_col].set_xscale('log')
        axs[axs_row].yaxis.set_ticks_position('both')
        #axs[axs_row].set_xlim(left=0)
        #axs[axs_row].set_ylim(bottom=0)
        axs[axs_row].legend()
    return fig


def plot_out_degree_vs_WS_p():
    # Plot out-degree vs. Watts-Strogatz rewiring prob (scatter error)
    fig, axs = my_subplots(1, 1, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'out_degree_real',
        'out_degree_inferred']]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'repetition_i'}
    # Plot real out-degree
    df_algorithm = df_interest.loc[df_interest['algorithm'] == algorithms[-1]]
    df_aggregate = df_algorithm.groupby('WS_p').agg(lambda x: list(x))
    axs[0].plot(
        WS_p_range,
        [np.nanmean(df_aggregate['out_degree_real'][WS_p])
            for WS_p in WS_p_range],
        label='Real'
    )
    axs[0].scatter(
        df_algorithm['WS_p'].values.astype(float).tolist(),
        [np.nanmean(x) for x in df_algorithm['out_degree_real'].values],
        alpha=0.3
    )
    for algorithm in algorithms:
        # Select dataframe entries(i.e runs) with the same number of samples
        df_algorithm = df_interest.loc[
            df_interest['algorithm'] == algorithm].drop('algorithm', 1)
        # Group by WS_p
        df_aggregate = df_algorithm.groupby('WS_p').agg(
            lambda x: list(x))
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        # Plot average clustering coefficient inferred
        axs[0].plot(
            WS_p_range,
            [np.nanmean(df_aggregate['out_degree_inferred'][WS_p])
             for WS_p in WS_p_range],
            label=algorithm_names[algorithm])
        axs[0].scatter(
            df_algorithm['WS_p'].values.astype(float).tolist(),
            [np.nanmean(x) for x in df_algorithm['out_degree_inferred'].values],
            alpha=0.3)
        # Set axes properties
        axs[0].set_xlabel(r'$Rewiring\ probability$', horizontalalignment='right', x=1.0)
        axs[0].set_ylabel(r'$Average\ out-degree$')
        axs[0].set_xscale('log')
        axs[0].yaxis.set_ticks_position('both')
        # Add legend
        axs[0].legend()

    return fig


def scatter_omnibus_TE_vs_out_degree_real():
    # Scatter plot performance vs real out-degree
    # Subplots vertical: WS_p
    # Subplots horizontal: algorithms
    # Only select TE algorithms
    TE_algorithms = np.extract(['TE' in alg for alg in algorithms], algorithms)
    subplots_v = len(WS_p_range)
    subplots_h = len(TE_algorithms)
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'out_degree_real',
        'TE_omnibus_theoretical_causal_vars',
        'TE_omnibus_empirical'
        ]]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}
    for (axs_row, WS_p) in enumerate(WS_p_range):
        # Select dataframe entries(i.e runs) with the same WS_p
        df_WS_p = df_interest.loc[
            df_interest['WS_p'] == WS_p].drop('WS_p', 1)
        # Group by algorithm and concatenate TE values lists
        df_aggregate = df_WS_p.groupby('algorithm').agg(
            lambda x: x.tolist())
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        for (axs_col, algorithm) in enumerate(TE_algorithms):
            # Set subplot title
            axs[axs_row, axs_col].set_title('p={0}$, ${1}'.format(
                WS_p,
                algorithm))
            out_degree_real = np.concatenate(
                df_aggregate.loc[algorithm].out_degree_real)
            TE_omnibus_theoretical_causal_vars = np.concatenate(
                df_aggregate.loc[algorithm].TE_omnibus_theoretical_causal_vars)
            TE_omnibus_empirical = np.concatenate(
                df_aggregate.loc[algorithm].TE_omnibus_empirical)
            # Scatter TE_omnibus_empirical vs out_degree
            axs[axs_row, axs_col].scatter(
                out_degree_real,
                TE_omnibus_empirical,
                label='Empirical')
            # Scatter TE_omnibus_theoretical_causal_vars vs out_degree
            axs[axs_row, axs_col].scatter(
                out_degree_real,
                TE_omnibus_theoretical_causal_vars,
                label='Theoretical')
            # Set axes properties
            axs[axs_row, axs_col].set_xlabel(
                r'$Real\ out-degree$', horizontalalignment='right', x=1.0)
            axs[axs_row, axs_col].set_ylabel(
                r'$TE$')
            # axs[axs_row, axs_col].set_xscale('log')
            axs[axs_row, axs_col].yaxis.set_ticks_position('both')
            #axs[axs_row, axs_col].set_xlim(left=0)
            axs[axs_row, axs_col].set_ylim(bottom=0)
            axs[axs_row, axs_col].legend()
    return fig


def scatter_performance_vs_out_degree_real():
    # Scatter plot precision vs real out-degree
    # Subplots vertical: WS_p
    # Subplots horizontal: algorithms
    subplots_v = len(WS_p_range)
    subplots_h = len(algorithms)
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'out_degree_real',
        'precision_per_target',
        'recall_per_target',
        ]]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}
    for (axs_row, WS_p) in enumerate(WS_p_range):
        # Select dataframe entries(i.e runs) with the same WS_p
        df_WS_p = df_interest.loc[
            df_interest['WS_p'] == WS_p].drop('WS_p', 1)
        # Group by algorithm and concatenate values into lists
        df_aggregate = df_WS_p.groupby('algorithm').agg(
            lambda x: x.tolist())
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        for (axs_col, algorithm) in enumerate(algorithms):
            # Set subplot title
            axs[axs_row, axs_col].set_title('p={0}$, ${1}'.format(
                WS_p,
                algorithm))
            out_degree_real = np.concatenate(
                df_aggregate.loc[algorithm].out_degree_real)
            precision_per_target = np.concatenate(
                df_aggregate.loc[algorithm].precision_per_target)
            recall_per_target = np.concatenate(
                df_aggregate.loc[algorithm].recall_per_target)
            # Scatter precision vs out_degree
            axs[axs_row, axs_col].scatter(
                out_degree_real,
                precision_per_target,
                label='Precision')
            # Scatter recall vs out_degree
            axs[axs_row, axs_col].scatter(
                out_degree_real,
                recall_per_target,
                label='Recall')
            # Set axes properties
            axs[axs_row, axs_col].set_xlabel(
                r'$Real\ out-degree$', horizontalalignment='right', x=1.0)
            axs[axs_row, axs_col].set_ylabel(
                r'$Performance$')
            # axs[axs_row, axs_col].set_xscale('log')
            axs[axs_row, axs_col].yaxis.set_ticks_position('both')
            #axs[axs_row, axs_col].set_xlim(left=0)
            #axs[axs_row, axs_col].set_ylim(bottom=0)
            axs[axs_row, axs_col].legend()
    return fig


def scatter_performance_vs_clustering_real():
    # Scatter plot precision vs real clustering coefficient
    # Subplots vertical: WS_p
    # Subplots horizontal: algorithms
    subplots_v = len(WS_p_range)
    subplots_h = len(algorithms)
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'clustering_real',
        'precision_per_target',
        'recall_per_target',
        ]]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}
    for (axs_row, WS_p) in enumerate(WS_p_range):
        # Select dataframe entries(i.e runs) with the same WS_p
        df_WS_p = df_interest.loc[
            df_interest['WS_p'] == WS_p].drop('WS_p', 1)
        # Group by algorithm and concatenate values into lists
        df_aggregate = df_WS_p.groupby('algorithm').agg(
            lambda x: x.tolist())
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        for (axs_col, algorithm) in enumerate(algorithms):
            # Set subplot title
            axs[axs_row, axs_col].set_title('p={0}$, ${1}'.format(
                WS_p,
                algorithm))
            clustering_real = np.concatenate(
                df_aggregate.loc[algorithm].clustering_real)
            precision_per_target = np.concatenate(
                df_aggregate.loc[algorithm].precision_per_target)
            recall_per_target = np.concatenate(
                df_aggregate.loc[algorithm].recall_per_target)
            # Scatter precision
            axs[axs_row, axs_col].scatter(
                clustering_real,
                precision_per_target,
                label='Precision')
            # Scatter recall
            axs[axs_row, axs_col].scatter(
                clustering_real,
                recall_per_target,
                label='Recall')
            # Set axes properties
            axs[axs_row, axs_col].set_xlabel(
                r'$Real\ clustering\ coefficient$', horizontalalignment='right', x=1.0)
            axs[axs_row, axs_col].set_ylabel(
                r'$Performance$')
            # axs[axs_row, axs_col].set_xscale('log')
            axs[axs_row, axs_col].yaxis.set_ticks_position('both')
            #axs[axs_row, axs_col].set_xlim(left=0)
            #axs[axs_row, axs_col].set_ylim(bottom=0)
            axs[axs_row, axs_col].legend()
    return fig


def scatter_performance_vs_omnibus_TE():
    # Scatter plot performance vs omnibus TE theoretical
    # Subplots vertical: WS_p
    # Subplots horizontal: algorithms
    # Only select TE algorithms
    TE_algorithms = np.extract(['TE' in alg for alg in algorithms], algorithms)
    subplots_v = len(WS_p_range)
    subplots_h = len(TE_algorithms)
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'precision_per_target',
        'recall_per_target',
        'TE_omnibus_theoretical_causal_vars',
        ]]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}
    for (axs_row, WS_p) in enumerate(WS_p_range):
        # Select dataframe entries(i.e runs) with the same WS_p
        df_WS_p = df_interest.loc[
            df_interest['WS_p'] == WS_p].drop('WS_p', 1)
        # Group by algorithm and concatenate TE values lists
        df_aggregate = df_WS_p.groupby('algorithm').agg(
            lambda x: x.tolist())
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        for (axs_col, algorithm) in enumerate(TE_algorithms):
            # Set subplot title
            axs[axs_row, axs_col].set_title('p={0}$, ${1}'.format(
                WS_p,
                algorithm))
            precision_per_target = np.concatenate(
                df_aggregate.loc[algorithm].precision_per_target)
            recall_per_target = np.concatenate(
                df_aggregate.loc[algorithm].recall_per_target)
            TE_omnibus_theoretical_causal_vars = np.concatenate(
                df_aggregate.loc[algorithm].TE_omnibus_theoretical_causal_vars)
            # Scatter precision vs TE_omnibus_theoretical_causal_vars
            axs[axs_row, axs_col].scatter(
                TE_omnibus_theoretical_causal_vars,
                precision_per_target,
                label='Precision')
            # Scatter recall vs TE_omnibus_theoretical_causal_vars
            axs[axs_row, axs_col].scatter(
                TE_omnibus_theoretical_causal_vars,
                recall_per_target,
                label='Recall')
            # Set axes properties
            axs[axs_row, axs_col].set_xlabel(
                r'$TE$', horizontalalignment='right', x=1.0)
            axs[axs_row, axs_col].set_ylabel(
                r'$Performance$')
            # axs[axs_row, axs_col].set_xscale('log')
            axs[axs_row, axs_col].yaxis.set_ticks_position('both')
            #axs[axs_row, axs_col].set_xlim(left=0)
            #axs[axs_row, axs_col].set_ylim(bottom=0)
            axs[axs_row, axs_col].legend(loc='lower left')
    return fig


def scatter_performance_vs_lattice_distance():
    # Scatter plot performance vs distance on the original lattice
    # Subplots vertical: WS_p
    # Subplots horizontal: algorithms
    subplots_v = len(WS_p_range)
    subplots_h = len(algorithms)
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'precision_per_distance',
        'recall_per_distance',
        ]]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}
    for (axs_row, WS_p) in enumerate(WS_p_range):
        # Select dataframe entries(i.e runs) with the same WS_p
        df_WS_p = df_interest.loc[
            df_interest['WS_p'] == WS_p].drop('WS_p', 1)
        # Group by algorithm and concatenate TE values lists
        df_aggregate = df_WS_p.groupby('algorithm').agg(
            lambda x: x.tolist())
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        for (axs_col, algorithm) in enumerate(algorithms):
            # Set subplot title
            axs[axs_row, axs_col].set_title('p={0}$, ${1}'.format(
                WS_p,
                algorithm))
            precision_per_distance = np.concatenate(
                df_aggregate.loc[algorithm].precision_per_distance)
            recall_per_distance = np.concatenate(
                df_aggregate.loc[algorithm].recall_per_distance)
            nn_max = np.floor(N_interest / 2)
            repetitions = int(precision_per_distance.shape[0] / nn_max)
            distance = np.tile(np.arange(1, nn_max + 1), (repetitions))
            # Scatter precision vs out_degree
            axs[axs_row, axs_col].scatter(
                distance,
                precision_per_distance,
                label='Precision')
            # Scatter
            axs[axs_row, axs_col].scatter(
                distance,
                recall_per_distance,
                label='Recall')
            # Set axes properties
            axs[axs_row, axs_col].set_xlabel(
                r'$Distance$', horizontalalignment='right', x=1.0)
            axs[axs_row, axs_col].set_ylabel(
                r'$Performance$')
            # axs[axs_row, axs_col].set_xscale('log')
            axs[axs_row, axs_col].yaxis.set_ticks_position('both')
            #axs[axs_row, axs_col].set_xlim(left=0)
            #axs[axs_row, axs_col].set_ylim(bottom=0)
            axs[axs_row, axs_col].legend()
    return fig


def plot_performance_vs_WS_p_group_distance():
    # Plot performance vs. Watts-Strogatz rewiring probability
    # Subplots: algorithms
    # Legend: distance (group distance <=WS_k)
    subplots_v = len(algorithms)
    subplots_h = 1
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[
        parameters_explored + [
            'precision_per_distance',
            'recall_per_distance']]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'repetition_i'}
    for (axs_row, algorithm) in enumerate(algorithms):
        # Set subplot title
        axs[axs_row].set_title('{0}'.format(algorithm_names[algorithm]))
        # Select dataframe entries(i.e runs) with the same algorithm
        df_algorithm = df_interest.loc[
            df_interest['algorithm'] == algorithm].drop('algorithm', 1)
        # Group by WS_p
        df_aggregate = df_algorithm.groupby('WS_p').agg(
            lambda x: list(x))
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        nn_max = int(np.floor(N_interest / 2))
        # Get a np array from a list of lists
        df_aggregate['precision_per_distance'] = df_aggregate['precision_per_distance'].apply(concat_if_not_empty)
        # reshape to obtain (repetitions, performance_per_distance)
        df_aggregate['precision_per_distance'] = df_aggregate['precision_per_distance'].apply(lambda x: np.reshape(x, (-1, nn_max)))
        # Average precision over different groups: dist<=WS_k, dist<=2*WS_k, dist>2*WS_k
        df_aggregate['precision_group_1'] = df_aggregate['precision_per_distance'].apply(lambda x: np.nanmean(x[:, :2], axis=1))
        df_aggregate['precision_group_2'] = df_aggregate['precision_per_distance'].apply(lambda x: np.nanmean(x[:, 2:4], axis=1))
        df_aggregate['precision_group_3'] = df_aggregate['precision_per_distance'].apply(lambda x: np.nanmean(x[:, 4:], axis=1))
        # Same for recall
        df_aggregate['recall_per_distance'] = df_aggregate['recall_per_distance'].apply(concat_if_not_empty)
        df_aggregate['recall_per_distance'] = df_aggregate['recall_per_distance'].apply(lambda x: np.reshape(x, (-1, nn_max)))
        df_aggregate['recall_group_1'] = df_aggregate['recall_per_distance'].apply(lambda x: np.nanmean(x[:, :2], axis=1))
        df_aggregate['recall_group_2'] = df_aggregate['recall_per_distance'].apply(lambda x: np.nanmean(x[:, 2:4], axis=1))
        df_aggregate['recall_group_3'] = df_aggregate['recall_per_distance'].apply(lambda x: np.nanmean(x[:, 4:], axis=1))
        # Plot precision
        axs[axs_row].plot(
            WS_p_range,
            [np.nanmean(df_aggregate['precision_group_1'][WS_p])
                for WS_p in WS_p_range],
            label='precision_1'
        )
        axs[axs_row].scatter(
            df_algorithm['WS_p'].values.astype(float).tolist(),
            np.concatenate([
                df_aggregate['precision_group_1'][WS_p]
                for WS_p in WS_p_range]),
            alpha=0.3
        )
        axs[axs_row].plot(
            WS_p_range,
            [np.nanmean(df_aggregate['precision_group_2'][WS_p])
                for WS_p in WS_p_range],
            label='precision_2'
        )
        axs[axs_row].scatter(
            df_algorithm['WS_p'].values.astype(float).tolist(),
            np.concatenate([
                df_aggregate['precision_group_2'][WS_p]
                for WS_p in WS_p_range]),
            alpha=0.3
        )
        axs[axs_row].plot(
            WS_p_range,
            [np.nanmean(df_aggregate['precision_group_3'][WS_p])
                for WS_p in WS_p_range],
            label='precision_3'
        )
        axs[axs_row].scatter(
            df_algorithm['WS_p'].values.astype(float).tolist(),
            np.concatenate([
                df_aggregate['precision_group_3'][WS_p]
                for WS_p in WS_p_range]),
            alpha=0.3
        )
        # Plot recall
        axs[axs_row].plot(
            WS_p_range,
            [np.nanmean(df_aggregate['recall_group_1'][WS_p])
                for WS_p in WS_p_range],
            label='recall_1'
        )
        axs[axs_row].scatter(
            df_algorithm['WS_p'].values.astype(float).tolist(),
            np.concatenate([
                df_aggregate['recall_group_1'][WS_p]
                for WS_p in WS_p_range]),
            alpha=0.3
        )
        axs[axs_row].plot(
            WS_p_range,
            [np.nanmean(df_aggregate['recall_group_2'][WS_p])
                for WS_p in WS_p_range],
            label='recall_2'
        )
        axs[axs_row].scatter(
            df_algorithm['WS_p'].values.astype(float).tolist(),
            np.concatenate([
                df_aggregate['recall_group_2'][WS_p]
                for WS_p in WS_p_range]),
            alpha=0.3
        )
        axs[axs_row].plot(
            WS_p_range,
            [np.nanmean(df_aggregate['recall_group_3'][WS_p])
                for WS_p in WS_p_range],
            label='recall_3'
        )
        axs[axs_row].scatter(
            df_algorithm['WS_p'].values.astype(float).tolist(),
            np.concatenate([
                df_aggregate['recall_group_3'][WS_p]
                for WS_p in WS_p_range]),
            alpha=0.3
        )
        # Set axes properties
        axs[axs_row].set_xlabel(r'$Rewiring\ probability$', horizontalalignment='right', x=1.0)
        axs[axs_row].set_ylabel(r'$Performance$')
        axs[axs_row].yaxis.set_ticks_position('both')
        #axs[axs_row].set_ylim(bottom=0)
        # Add legend
        axs[axs_row].legend()

    return fig


def barchart_precision_per_motif():
    # Bar chart of precision per motif
    # Legend: algorithms
    fig = plt.figure()
    # Select data of interest
    df_interest = df[parameters_explored + ['precision_per_motif']]
    # Get number of motifs
    motif_n = len(df_interest['precision_per_motif'].iloc[0][0])
    # Define grid for subplots
    rows = 6
    cols = motif_n
    gs = gridspec.GridSpec(rows, cols)
    # Set grid margins
    gs.update(wspace=0.3)
    # Get main plot axis
    ax_main = plt.subplot(gs[:-1, :])
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i', 'WS_p'}
    # Group by algorithm and aggregate values into lists
    df_aggregate = df_interest.groupby('algorithm').agg(
        lambda x: x.tolist())
    # Ensure that only the desired parameters are aggregated or averaged
    df_keys_remaining = df_aggregate.columns.get_level_values(0)
    check_remaining_dimensions(df_keys_remaining, parameters_to_average)
    bar_n = algorithms.shape[0]
    width = .8 / bar_n  # the width of the bars
    for (alg_i, algorithm) in enumerate(algorithms):
        precision_list = df_aggregate.loc[algorithm].precision_per_motif
        precision_per_motif = defaultdict(list)
        for d in precision_list:
            for key, value in d[0].items():
                precision_per_motif[key].append(value)
        x_loc = np.arange(motif_n)  # the starting x locations for the bars
        precision_per_motif_mean = {}
        precision_per_motif_std = {}
        for motif_name, values in precision_per_motif.items():
            precision_per_motif_mean[motif_name] = np.nanmean(values)
            precision_per_motif_std[motif_name] = np.nanstd(values)
        ax_main.bar(
            x_loc + alg_i * width,
            precision_per_motif_mean.values(),
            width,
            align='edge',
            yerr=precision_per_motif_std.values(),
            error_kw=dict(lw=0.5, capsize=1, capthick=0.5),
            #color='r',
            label=algorithm_names[algorithm])

        # Set properties of main axis
        ax_main.set_xlim(left=0, right=motif_n-1+bar_n*width)
        #xticks_locations = x_loc + bar_n * width / 2
        #ax_main.set_xticks(xticks_locations)
        ax_main.set_xticks([])
        ax_main.tick_params(labelbottom=False)

        # Add motif subplots
        pos = {'a': (0, 0), 'b': (0.5, 1), 'c': (1, 0)}
        radius = 0.2
        t = triad_graphs()
        i = 0
        for name, G in sorted(t.items()):
            # Annotate plot with motif images
            ax = plt.subplot(gs[-1, i])
            for n in G:
                c = Circle(pos[n], radius=radius, alpha=0.7)
                ax.add_patch(c)
                G.node[n]['patch'] = c
            for u, v in G.edges():
                n1 = G.node[u]['patch']
                n2 = G.node[v]['patch']
                e = FancyArrowPatch(n1.center, n2.center,# patchA=n1, patchB=n2,
                                    arrowstyle='-|>',
                                    connectionstyle='arc3',
                                    mutation_scale=3.0,
                                    lw=0.6, color='k')
                ax.add_patch(e)
            # ax.text(
            #     0.5, 0.0,
            #     name,
            #     transform=ax.transAxes,
            #     horizontalalignment='center')
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            plt.axis('equal')
            plt.axis('off')
            i += 1
        # Set plot labels
        # axs[axs_row, 0].set_xlabel(
        #     r'$Motif$', horizontalalignment='right', x=1.0)
        ax_main.set_ylabel(r'$Precision$')
        # Add legend
        ax_main.legend(loc='lower right')
        # Set figure ratio
        # fig.set_figheight(3)
        # fig.set_figwidth(4)
        # Set tight layout to avoid overlap between subplots
        fig.tight_layout()

    return fig


def barchart_recall_per_motif():
    # Bar chart of recall per motif
    # Legend: algorithms
    fig = plt.figure()
    # Select data of interest
    df_interest = df[parameters_explored + ['recall_per_motif']]
    # Get number of motifs
    motif_n = len(df_interest['recall_per_motif'].iloc[0][0])
    # Define grid for subplots
    rows = 6
    cols = motif_n
    gs = gridspec.GridSpec(rows, cols)
    # Set grid margins
    gs.update(wspace=0.3)
    # Get main plot axis
    ax_main = plt.subplot(gs[:-1, :])
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i', 'WS_p'}
    # Group by algorithm and aggregate values into lists
    df_aggregate = df_interest.groupby('algorithm').agg(
        lambda x: x.tolist())
    # Ensure that only the desired parameters are aggregated or averaged
    df_keys_remaining = df_aggregate.columns.get_level_values(0)
    check_remaining_dimensions(df_keys_remaining, parameters_to_average)
    bar_n = algorithms.shape[0]
    width = .8 / bar_n  # the width of the bars
    for (alg_i, algorithm) in enumerate(algorithms):
        recall_list = df_aggregate.loc[algorithm].recall_per_motif
        recall_per_motif = defaultdict(list)
        for d in recall_list:
            for key, value in d[0].items():
                recall_per_motif[key].append(value)
        x_loc = np.arange(motif_n)  # the starting x locations for the bars
        recall_per_motif_mean = {}
        recall_per_motif_std = {}
        for motif_name, values in recall_per_motif.items():
            recall_per_motif_mean[motif_name] = np.nanmean(values)
            recall_per_motif_std[motif_name] = np.nanstd(values)
        ax_main.bar(
            x_loc + alg_i * width,
            recall_per_motif_mean.values(),
            width,
            align='edge',
            yerr=recall_per_motif_std.values(),
            error_kw=dict(lw=0.5, capsize=1, capthick=0.5),
            #color='r',
            label=algorithm_names[algorithm])

        # Set properties of main axis
        ax_main.set_xlim(left=0, right=motif_n-1+bar_n*width)
        #xticks_locations = x_loc + bar_n * width / 2
        #ax_main.set_xticks(xticks_locations)
        ax_main.set_xticks([])
        ax_main.tick_params(labelbottom=False)

        # Add motif subplots
        pos = {'a': (0, 0), 'b': (0.5, 1), 'c': (1, 0)}
        radius = 0.2
        t = triad_graphs()
        i = 0
        for name, G in sorted(t.items()):
            # Annotate plot with motif images
            ax = plt.subplot(gs[-1, i])
            for n in G:
                c = Circle(pos[n], radius=radius, alpha=0.7)
                ax.add_patch(c)
                G.node[n]['patch'] = c
            for u, v in G.edges():
                n1 = G.node[u]['patch']
                n2 = G.node[v]['patch']
                e = FancyArrowPatch(n1.center, n2.center,# patchA=n1, patchB=n2,
                                    arrowstyle='-|>',
                                    connectionstyle='arc3',
                                    mutation_scale=3.0,
                                    lw=0.6, color='k')
                ax.add_patch(e)
            # ax.text(
            #     0.5, 0.0,
            #     name,
            #     transform=ax.transAxes,
            #     horizontalalignment='center')
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            plt.axis('equal')
            plt.axis('off')
            i += 1
        # Set plot labels
        ax_main.set_ylabel(r'$Recall$')
        # Add legend
        ax_main.legend(loc='lower right')
        # Set figure ratio
        # fig.set_figheight(3)
        # fig.set_figwidth(4)
        # Set tight layout to avoid overlap between subplots
        fig.tight_layout()

    return fig


def imshow_TE_complete_theoretical_causal_vars():
    # only mTE_greedy
    # imshow of TE_complete_theoretical_causal_vars for each source-target pair
    # Subplots vertical: causal and inferred
    # Subplots horizontal: WS_p
    # Only select mTE algorithm
    subplots_v = 2
    subplots_h = len(WS_p_range)
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'TE_complete_theoretical_inferred_vars',
        'TE_complete_theoretical_causal_vars']]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Average TE values over the lags to obtain a NxN matrix (exclude zeros)
    df_interest['TE_complete_theoretical_causal_vars'] = df_interest['TE_complete_theoretical_causal_vars'].apply(lambda x: np.ma.masked_where(x == 0, x))
    df_interest['TE_complete_theoretical_causal_vars'] = df_interest['TE_complete_theoretical_causal_vars'].apply(lambda x: np.nanmean(x, axis=0))
    df_interest['TE_complete_theoretical_inferred_vars'] = df_interest['TE_complete_theoretical_inferred_vars'].apply(lambda x: np.ma.masked_where(x == 0, x))
    df_interest['TE_complete_theoretical_inferred_vars'] = df_interest['TE_complete_theoretical_inferred_vars'].apply(lambda x: np.nanmean(x, axis=0))
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}
    TE_matrices = []
    # Select dataframe entries(i.e runs) with the mTE_greedy algorithm
    df_algorithm = df_interest.loc[
        df_interest['algorithm'] == 'mTE_greedy'].drop('algorithm', 1)
    # Group by WS_p and concatenate TE values lists
    df_aggregate = df_algorithm.groupby('WS_p').agg(
        lambda x: x.tolist())
    # Ensure that only the desired parameters are aggregated or averaged
    df_keys_remaining = df_aggregate.columns.get_level_values(0)
    check_remaining_dimensions(df_keys_remaining, parameters_to_average)
    for (axs_col, WS_p) in enumerate(WS_p_range):
        # Set subplot title
        axs[0, axs_col].set_title(r'$\gamma={0}$'.format(WS_p))
        axs[1, axs_col].set_title(r'$\gamma={0}$'.format(WS_p))
        mean_vals_causal_vars = np.nanmean(df_aggregate.loc[WS_p].TE_complete_theoretical_causal_vars, axis=0)
        mean_vals_inferred_vars = np.nanmean(df_aggregate.loc[WS_p].TE_complete_theoretical_inferred_vars, axis=0)
        TE_matrices.append(
            axs[0, axs_col].imshow(
                mean_vals_causal_vars,
                origin='lower'))
        TE_matrices.append(
            axs[1, axs_col].imshow(
                mean_vals_inferred_vars,
                origin='lower'))
        axs[0, axs_col].label_outer()
        axs[1, axs_col].label_outer()
    # Find the min and max of all colors for use in setting the color scale.
    # As shown in https://matplotlib.org/3.1.0/gallery/images_contours_and_fields/multi_image.html
    vmin = min(image.get_array().min() for image in TE_matrices)
    vmax = max(image.get_array().max() for image in TE_matrices)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for im in TE_matrices:
        im.set_norm(norm)
    # Show colorbar
    fig.colorbar(TE_matrices[0], ax=axs, orientation='horizontal', fraction=.1)
    # Make images respond to changes in the norm of other images (e.g. via the
    # "edit axis, curves and images parameters" GUI on Qt), but be careful not to
    # recurse infinitely!
    def update(changed_image):
        for im in TE_matrices:
            if (changed_image.get_cmap() != im.get_cmap()
                    or changed_image.get_clim() != im.get_clim()):
                im.set_cmap(changed_image.get_cmap())
                im.set_clim(changed_image.get_clim())
    for im in TE_matrices:
        im.callbacksSM.connect('changed', update)
    return fig


def imshow_TE_apparent_theoretical_causal_vars():
    # only bTE_greedy
    # imshow of TE_apparent_theoretical_causal_vars for each source-target pair
    # Subplots vertical: causal and inferred
    # Subplots horizontal: WS_p
    # Only select mTE algorithm
    subplots_v = 2
    subplots_h = len(WS_p_range)
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'TE_apparent_theoretical_inferred_vars',
        'TE_apparent_theoretical_causal_vars']]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Average TE values over the lags to obtain a NxN matrix (exclude zeros)
    df_interest['TE_apparent_theoretical_causal_vars'] = df_interest['TE_apparent_theoretical_causal_vars'].apply(lambda x: np.ma.masked_where(x == 0, x))
    df_interest['TE_apparent_theoretical_causal_vars'] = df_interest['TE_apparent_theoretical_causal_vars'].apply(lambda x: np.nanmean(x, axis=0))
    df_interest['TE_apparent_theoretical_inferred_vars'] = df_interest['TE_apparent_theoretical_inferred_vars'].apply(lambda x: np.ma.masked_where(x == 0, x))
    df_interest['TE_apparent_theoretical_inferred_vars'] = df_interest['TE_apparent_theoretical_inferred_vars'].apply(lambda x: np.nanmean(x, axis=0))
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}
    # Select dataframe entries(i.e runs) with the bTE_greedy algorithm
    df_algorithm = df_interest.loc[df_interest['algorithm'] == 'bTE_greedy'].drop('algorithm', 1)
    # Group by WS_p and concatenate TE values lists
    df_aggregate = df_algorithm.groupby('WS_p').agg(
        lambda x: x.tolist())
    # Ensure that only the desired parameters are aggregated or averaged
    df_keys_remaining = df_aggregate.columns.get_level_values(0)
    check_remaining_dimensions(df_keys_remaining, parameters_to_average)
    TE_matrices = []
    for (axs_col, WS_p) in enumerate(WS_p_range):
        # Set subplot title
        axs[0, axs_col].set_title(r'$\gamma={0}$'.format(WS_p))
        axs[1, axs_col].set_title(r'$\gamma={0}$'.format(WS_p))
        mean_vals_causal_vars = np.nanmean(df_aggregate.loc[WS_p].TE_apparent_theoretical_causal_vars, axis=0)
        mean_vals_inferred_vars = np.nanmean(df_aggregate.loc[WS_p].TE_apparent_theoretical_inferred_vars, axis=0)
        TE_matrices.append(
            axs[0, axs_col].imshow(
                mean_vals_causal_vars,
                origin='lower'))
        TE_matrices.append(
            axs[1, axs_col].imshow(
                mean_vals_inferred_vars,
                origin='lower'))
        axs[0, axs_col].label_outer()
        axs[1, axs_col].label_outer()
    # Find the min and max of all colors for use in setting the color scale.
    # As shown in https://matplotlib.org/3.1.0/gallery/images_contours_and_fields/multi_image.html
    vmin = min(image.get_array().min() for image in TE_matrices)
    vmax = max(image.get_array().max() for image in TE_matrices)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for im in TE_matrices:
        im.set_norm(norm)
    # Show colorbar
    fig.colorbar(TE_matrices[0], ax=axs, orientation='horizontal', fraction=.1)
    # Make images respond to changes in the norm of other images (e.g. via the
    # "edit axis, curves and images parameters" GUI on Qt), but be careful not to
    # recurse infinitely!
    def update(changed_image):
        for im in TE_matrices:
            if (changed_image.get_cmap() != im.get_cmap()
                    or changed_image.get_clim() != im.get_clim()):
                im.set_cmap(changed_image.get_cmap())
                im.set_clim(changed_image.get_clim())
    for im in TE_matrices:
        im.callbacksSM.connect('changed', update)
    return fig


# -------------------------------------------------------------------------
# endregion
# -------------------------------------------------------------------------



# -------------------------------------------------------------------------
# region Barabasi-Albert
# -------------------------------------------------------------------------
def scatter_performance_vs_in_degree_real():
    # Scatter plot precision vs real in-degree
    # Subplots vertical: algorithms
    subplots_v = len(algorithms)
    subplots_h = 1
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'in_degree_real',
        'precision_per_target',
        'recall_per_target',
        ]]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}
    for (axs_col, algorithm) in enumerate(algorithms):
        # Group by algorithm and concatenate values into lists
        df_aggregate = df_interest.groupby('algorithm').agg(
            lambda x: x.tolist())
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        # Set subplot title
        axs[axs_col].set_title('{0}'.format(algorithm_names[algorithm]))
        in_degree_real = np.concatenate(
            df_aggregate.loc[algorithm].in_degree_real)
        precision_per_target = np.concatenate(
            df_aggregate.loc[algorithm].precision_per_target)
        recall_per_target = np.concatenate(
            df_aggregate.loc[algorithm].recall_per_target)
        # Scatter precision vs in_degree
        axs[axs_col].scatter(
            in_degree_real,
            precision_per_target,
            label='Precision')
        # Scatter
        axs[axs_col].scatter(
            in_degree_real,
            recall_per_target,
            label='Recall')
        # Set axes properties
        axs[axs_col].set_xlabel(
            r'$\text{Real in-degree}$', horizontalalignment='right', x=1.0)
        axs[axs_col].set_ylabel(
            r'$\text{Performance}$')
        # axs[axs_col].set_xscale('log')
        axs[axs_col].yaxis.set_ticks_position('both')
        #axs[axs_col].set_xlim(left=0)
        #axs[axs_col].set_ylim(bottom=0)
        axs[axs_col].legend()
    return fig


def scatter_FP_vs_in_degree_real():
    # Scatter plot FP vs real in-degree
    # Subplots vertical: algorithms
    subplots_v = len(algorithms)
    subplots_h = 1
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'in_degree_real',
        'in_degree_inferred',
        'precision_per_target',
        'recall_per_target',
        ]]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}
    for (axs_col, algorithm) in enumerate(algorithms):
        # Group by algorithm and concatenate values into lists
        df_aggregate = df_interest.groupby('algorithm').agg(
            lambda x: x.tolist())
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        # Set subplot title
        axs[axs_col].set_title('{0}'.format(algorithm_names[algorithm]))
        in_degree_real = np.concatenate(
            df_aggregate.loc[algorithm].in_degree_real)
        in_degree_inferred = np.concatenate(
            df_aggregate.loc[algorithm].in_degree_inferred)
        precision_per_target = np.concatenate(
            df_aggregate.loc[algorithm].precision_per_target)
        
        precision_per_target[np.isnan(precision_per_target)] = 1.

        #recall_per_target = np.concatenate(
        #    df_aggregate.loc[algorithm].recall_per_target)
        
        FP = in_degree_inferred * (1 - precision_per_target)
        
        # Scatter FP vs in_degree
        axs[axs_col].scatter(
            in_degree_real,
            FP,
            label='FP')
        # # Scatter
        # axs[axs_col].scatter(
        #     in_degree_real,
        #     recall_per_target,
        #     label='Recall')
        # Set axes properties
        axs[axs_col].set_xlabel(
            r'$\text{Real in-degree}$', horizontalalignment='right', x=1.0)
        axs[axs_col].set_ylabel(
            r'$\text{FP}$')
        # axs[axs_col].set_xscale('log')
        axs[axs_col].yaxis.set_ticks_position('both')
        #axs[axs_col].set_xlim(left=0)
        #axs[axs_col].legend()
    return fig


def scatter_performance_vs_out_degree_real():
    # Scatter plot precision vs real out-degree
    # Subplots vertical: WS_p
    # Subplots horizontal: algorithms
    subplots_v = len(algorithms)
    subplots_h = 1
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'out_degree_real',
        'precision_per_target',
        'recall_per_target',
        ]]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}
    for (axs_col, algorithm) in enumerate(algorithms):
        # Group by algorithm and concatenate values into lists
        df_aggregate = df_interest.groupby('algorithm').agg(
            lambda x: x.tolist())
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        # Set subplot title
        axs[axs_col].set_title('{0}'.format(algorithm_names[algorithm]))
        out_degree_real = np.concatenate(
            df_aggregate.loc[algorithm].out_degree_real)
        precision_per_target = np.concatenate(
            df_aggregate.loc[algorithm].precision_per_target)
        recall_per_target = np.concatenate(
            df_aggregate.loc[algorithm].recall_per_target)
        # Scatter precision vs out_degree
        axs[axs_col].scatter(
            out_degree_real,
            precision_per_target,
            label='Precision')
        # Scatter
        axs[axs_col].scatter(
            out_degree_real,
            recall_per_target,
            label='Recall')
        # Set axes properties
        axs[axs_col].set_xlabel(
            r'$\text{Real out-degree}$', horizontalalignment='right', x=1.0)
        axs[axs_col].set_ylabel(
            r'$\text{Performance}$')
        # axs[axs_col].set_xscale('log')
        axs[axs_col].yaxis.set_ticks_position('both')
        #axs[axs_col].set_xlim(left=0)
        #axs[axs_col].set_ylim(bottom=0)
        axs[axs_col].legend()
    return fig


def scatter_in_degree_inferred_vs_real():
    # Scatter in-degree inferred vs. real
    # Subplots vertical: algorithms
    subplots_v = len(algorithms)
    subplots_h = 1
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'in_degree_inferred',
        'in_degree_real',
        ]]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}
    for (axs_col, algorithm) in enumerate(algorithms):
        # Group by algorithm and concatenate values into lists
        df_aggregate = df_interest.groupby('algorithm').agg(
            lambda x: x.tolist())
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        in_degree_inferred = np.concatenate(
            df_aggregate.loc[algorithm].in_degree_inferred)
        in_degree_real = np.concatenate(
            df_aggregate.loc[algorithm].in_degree_real)
        # Scatter
        axs[axs_col].scatter(
            in_degree_real,
            in_degree_inferred,
            )
        # Plot identity line
        axs[axs_col].plot(
            [min(in_degree_real), max(in_degree_real)],
            [min(in_degree_real), max(in_degree_real)],
            'g--')
        # Compute mean squared error and Person corr
        mse = np.square(np.subtract(in_degree_inferred, in_degree_real)).mean()
        rho=np.corrcoef(np.array([in_degree_real,in_degree_inferred]))[0,1]
        # Set subplot title
        axs[axs_col].set_title('{0}, MSE= {1}'.format(algorithm_names[algorithm], mse))
        # Set axes properties
        axs[axs_col].set_xlabel(
            r'$\text{Real in-degree}$', horizontalalignment='right', x=1.0)
        axs[axs_col].set_ylabel(
            r'$\text{Inferred in-degree}$')
        # axs[axs_row, axs_col].set_xscale('log')
        axs[axs_col].yaxis.set_ticks_position('both')
        #axs[axs_row, axs_col].set_xlim(left=0)
        #axs[axs_row, axs_col].set_ylim(bottom=0)
    return fig


def scatter_out_degree_inferred_vs_real():
    # Scatter out-degree inferred vs. real
    # Subplots vertical: algorithms
    subplots_v = len(algorithms)
    subplots_h = 1
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'out_degree_inferred',
        'out_degree_real',
        ]]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}

    for (axs_col, algorithm) in enumerate(algorithms):
        
        # Group by algorithm and concatenate values into lists
        df_aggregate = df_interest.groupby('algorithm').agg(
            lambda x: x.tolist())
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        
        out_degree_inferred = np.concatenate(
            df_aggregate.loc[algorithm].out_degree_inferred)
        out_degree_real = np.concatenate(
            df_aggregate.loc[algorithm].out_degree_real)
        # Scatter
        axs[axs_col].scatter(
            out_degree_real,
            out_degree_inferred,
            )
        # Plot identity line
        axs[axs_col].plot(
            [min(out_degree_real), max(out_degree_real)],
            [min(out_degree_real), max(out_degree_real)],
            'g--')
        # Compute MSE and Person corr
        rho=np.corrcoef(np.array([out_degree_real,out_degree_inferred]))[0,1]
        mse = np.square(np.subtract(out_degree_inferred, out_degree_real)).mean()
        # Set subplot title
        axs[axs_col].set_title('{0}, MSE= {1}'.format(algorithm_names[algorithm], mse))
        # Set axes properties
        axs[axs_col].set_xlabel(
            r'$\text{Real out-degree}$', horizontalalignment='right', x=1.0)
        axs[axs_col].set_ylabel(
            r'$\text{Inferred out-degree}$')
        # axs[axs_row, axs_col].set_xscale('log')
        axs[axs_col].yaxis.set_ticks_position('both')
        #axs[axs_row, axs_col].set_xlim(left=0)
        #axs[axs_row, axs_col].set_ylim(bottom=0)
    return fig


def histogram_in_degree_inferred_vs_real():
    # Plot in_degree_real and in_degree_inferred histograms (only T_interest)
    # Subplots vertical: algorithms
    subplots_v = len(algorithms)
    subplots_h = 1
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
    'in_degree_inferred',
    'in_degree_real',
    ]]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}
    for (axs_col, algorithm) in enumerate(algorithms):
        # Group by algorithm and concatenate values into lists
        df_aggregate = df_interest.groupby('algorithm').agg(
            lambda x: x.tolist())
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        in_degree_inferred = np.concatenate(
            df_aggregate.loc[algorithm].in_degree_inferred)
        in_degree_real = np.concatenate(
            df_aggregate.loc[algorithm].in_degree_real)
        # Real histogram
        axs[axs_col].hist(
            in_degree_real,
            density=True,
            label='Real')
        # Inferred histogram
        axs[axs_col].hist(
            in_degree_inferred,
            density=True,
            label='Inferred')
        # Set subplot title
        axs[axs_col].set_title('{0}'.format(algorithm_names[algorithm]))
        # Set axes properties
        axs[axs_col].set_xlabel(r'$\text{In-degree}$', horizontalalignment='right', x=1.0)
        axs[axs_col].yaxis.set_ticks_position('both')
        #axs[axs_col].set_ylim(bottom=0)
        # Add legend
        axs[axs_col].legend()

    return fig


def histogram_out_degree_inferred_vs_real():
    # Plot out_degree_real and out_degree_inferred histograms (only T_interest)
    # Subplots vertical: algorithms
    subplots_v = len(algorithms)
    subplots_h = 1
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
    'out_degree_inferred',
    'out_degree_real',
    ]]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}

    for (axs_col, algorithm) in enumerate(algorithms):
        # Group by algorithm and concatenate values into lists
        df_aggregate = df_interest.groupby('algorithm').agg(
            lambda x: x.tolist())
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        out_degree_inferred = np.concatenate(
            df_aggregate.loc[algorithm].out_degree_inferred)
        out_degree_real = np.concatenate(
            df_aggregate.loc[algorithm].out_degree_real)
        # Real histogram
        axs[axs_col].hist(
            out_degree_real,
            density=True,
            label='Real')
        # Inferred histogram
        axs[axs_col].hist(
            out_degree_inferred,
            density=True,
            label='Inferred')
        # Set subplot title
        axs[axs_col].set_title('{0}'.format(algorithm_names[algorithm]))
        # Set axes properties
        axs[axs_col].set_xlabel(r'$\text{Out-degree}$', horizontalalignment='right', x=1.0)
        axs[axs_col].yaxis.set_ticks_position('both')
        #axs[axs_col].set_ylim(bottom=0)
        # Add legend
        axs[axs_col].legend()

    return fig


def loglog_distributions_in_degree_inferred_vs_real():
    # Plot in_degree_real and in_degree_inferred distributions (only T_interest)
    # Subplots vertical: algorithms
    subplots_v = len(algorithms)
    subplots_h = 1
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
    'in_degree_inferred',
    'in_degree_real',
    ]]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}
    for (axs_col, algorithm) in enumerate(algorithms):
        # Group by algorithm and concatenate values into lists
        df_aggregate = df_interest.groupby('algorithm').agg(
            lambda x: x.tolist())
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        in_degree_inferred = np.concatenate(
            df_aggregate.loc[algorithm].in_degree_inferred)
        in_degree_real = np.concatenate(
            df_aggregate.loc[algorithm].in_degree_real)
        
        # degree_counts_inferred = Counter(in_degree_inferred)
        # x_inferred, y_inferred = zip(*degree_counts_inferred.items())
        # degree_counts_real = Counter(in_degree_real)
        # x_real, y_real = zip(*degree_counts_real.items())
        
        bins_inferred = np.arange(np.min(in_degree_inferred), np.max(in_degree_inferred) + 2, 1) - 0.5
        x_inferred = bins_inferred[:-1] + 0.5
        y_inferred, bins = np.histogram(in_degree_inferred, bins=bins_inferred, density=False)
        y_inferred = y_inferred / len(in_degree_inferred)

        bins_real = np.arange(np.min(in_degree_real), np.max(in_degree_real) + 2, 1) - 0.5
        x_real = bins_real[:-1] + 0.5
        y_real, bins = np.histogram(in_degree_real, bins=bins_real, density=False)
        y_real = y_real / len(in_degree_real)

        def power_law(x, coeff, exponent):
            return coeff * x ** exponent

        popt_inferred, pcov_inferred = curve_fit(power_law, x_inferred[1:-1], y_inferred[1:-1])
        popt_real, pcov_real = curve_fit(power_law, x_real[1:-1], y_real[1:-1])

        x_min = min(x_real.min(), x_inferred.min())
        x_max = max(x_real.max(), x_inferred.max())
        x = np.linspace(x_min, x_max)

        # Real
        axs[axs_col].scatter(x_real, y_real, marker='.', label='Real')
        axs[axs_col].plot(
            x,
            power_law(x, *popt_real),
            '--',
            color='tab:blue',
            label='{0} k{1}'.format(*popt_real),
            )
        # Inferred
        axs[axs_col].scatter(x_inferred, y_inferred, marker='.', label='Inferred')
        axs[axs_col].plot(
            x,
            power_law(x, *popt_inferred),
            '--',
            color='tab:orange',
            label='{0} k{1}'.format(*popt_inferred),
            )
        
        # # k^-3
        # axs[axs_col].plot(
        #     x,
        #     x ** (-3),
        #     '--',
        #     color='tab:gray',
        #     label=r'$k^{-3}$',
        #     )

        # Set subplot title
        axs[axs_col].set_title('{0}'.format(algorithm_names[algorithm]))
        # Set axes properties
        axs[axs_col].set_xlabel(r'$\text{In-degree}$', horizontalalignment='right', x=1.0)
        axs[axs_col].set_ylabel(r'$\text{Frequency}$')
        axs[axs_col].yaxis.set_ticks_position('both')
        #axs[axs_col].set_ylim(bottom=0)
        axs[axs_col].set_xscale('log')
        axs[axs_col].set_yscale('log')
        # Add legend
        axs[axs_col].legend()

    return fig


def loglog_distributions_out_degree_inferred_vs_real():
    # Plot out_degree_real and out_degree_inferred distributions (only T_interest)
    # Subplots vertical: algorithms
    subplots_v = len(algorithms)
    subplots_h = 1
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
    'out_degree_inferred',
    'out_degree_real',
    ]]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}
    for (axs_col, algorithm) in enumerate(algorithms):
        # Group by algorithm and concatenate values into lists
        df_aggregate = df_interest.groupby('algorithm').agg(
            lambda x: x.tolist())
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        out_degree_inferred = np.concatenate(
            df_aggregate.loc[algorithm].out_degree_inferred)
        out_degree_real = np.concatenate(
            df_aggregate.loc[algorithm].out_degree_real)
        
        # degree_counts_inferred = Counter(out_degree_inferred)                                                                                                 
        # x_inferred, y_inferred = zip(*degree_counts_inferred.items())   
        # degree_counts_real = Counter(out_degree_real)                                                                                                 
        # x_real, y_real = zip(*degree_counts_real.items())                                                      
        
        bins_inferred = np.arange(np.min(out_degree_inferred), np.max(out_degree_inferred) + 2, 1) - 0.5
        x_inferred = bins_inferred[:-1] + 0.5
        y_inferred, bins = np.histogram(out_degree_inferred, bins=bins_inferred, density=False)
        y_inferred = y_inferred / len(out_degree_inferred)

        bins_real = np.arange(np.min(out_degree_real), np.max(out_degree_real) + 2, 1) - 0.5
        x_real = bins_real[:-1] + 0.5
        y_real, bins = np.histogram(out_degree_real, bins=bins_real, density=False)
        y_real = y_real / len(out_degree_real)

        def power_law(x, coeff, exponent):
            return coeff * x ** exponent

        popt_inferred, pcov_inferred = curve_fit(power_law, x_inferred[1:-1], y_inferred[1:-1])
        popt_real, pcov_real = curve_fit(power_law, x_real[1:-1], y_real[1:-1])

        x_min = min(x_real.min(), x_inferred.min())
        x_max = max(x_real.max(), x_inferred.max())
        x = np.linspace(x_min, x_max)

        # Real
        axs[axs_col].scatter(x_real, y_real, marker='.', label='Real')
        axs[axs_col].plot(
            x,
            power_law(x, *popt_real),
            '--',
            color='tab:blue',
            label='{0} k{1}'.format(*popt_real),
            )
        # Inferred
        axs[axs_col].scatter(x_inferred, y_inferred, marker='.', label='Inferred')
        axs[axs_col].plot(
            x,
            power_law(x, *popt_inferred),
            '--',
            color='tab:orange',
            label='{0} k{1}'.format(*popt_inferred),
            )
        
        
        # Set subplot title
        axs[axs_col].set_title('{0}'.format(algorithm_names[algorithm]))
        # Set axes properties
        axs[axs_col].set_xlabel(r'$\text{Out-degree}$', horizontalalignment='right', x=1.0)
        axs[axs_col].set_ylabel(r'$\text{Frequency}$')
        axs[axs_col].yaxis.set_ticks_position('both')
        #axs[axs_col].set_ylim(bottom=0)
        axs[axs_col].set_xscale('log')
        axs[axs_col].set_yscale('log')
        # Add legend
        axs[axs_col].legend()

    return fig


def scatter_in_degree_assortativity_inferred_vs_real():
    # Scatter in_degree_assortativity_real vs. in_degree_assortativity_inferred
    # Subplots vertical: algorithms
    subplots_v = len(algorithms)
    subplots_h = 1
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'in_degree_assortativity_inferred',
        'in_degree_assortativity_real',
        ]]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}
    for (axs_col, algorithm) in enumerate(algorithms):
        # Group by algorithm and concatenate values into lists
        df_aggregate = df_interest.groupby('algorithm').agg(
            lambda x: x.tolist())
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        
        in_degree_assortativity_inferred = (
            df_aggregate.loc[algorithm].in_degree_assortativity_inferred)
        in_degree_assortativity_real = (
            df_aggregate.loc[algorithm].in_degree_assortativity_real)
        # Scatter
        axs[axs_col].scatter(
            in_degree_assortativity_real,
            in_degree_assortativity_inferred,
            )
        # Plot identity line
        axs[axs_col].plot(
            [min(in_degree_assortativity_real), max(in_degree_assortativity_real)],
            [min(in_degree_assortativity_real), max(in_degree_assortativity_real)],
            'g--')
        # Compute MSE and Person corr
        rho=np.corrcoef(
            np.array([in_degree_assortativity_real, in_degree_assortativity_inferred]))[0,1]
        mse = np.square(
            np.subtract(in_degree_assortativity_inferred, in_degree_assortativity_real)).mean()
        # Set subplot title
        axs[axs_col].set_title('{0}, MSE= {1}'.format(algorithm_names[algorithm], mse))
        # Set axes properties
        axs[axs_col].set_xlabel(
            r'$\text{Real in-degree assortativity}$', horizontalalignment='right', x=1.0)
        axs[axs_col].set_ylabel(
            r'$\text{Inferred in-degree assortativity}$')
        # axs[axs_row, axs_col].set_xscale('log')
        axs[axs_col].yaxis.set_ticks_position('both')
        #axs[axs_row, axs_col].set_xlim(left=0)
        #axs[axs_row, axs_col].set_ylim(bottom=0)
    return fig


def scatter_out_degree_assortativity_inferred_vs_real():
    # Scatter out_degree_assortativity_real vs. out_degree_assortativity_inferred
    # Subplots vertical: algorithms
    subplots_v = len(algorithms)
    subplots_h = 1
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'out_degree_assortativity_inferred',
        'out_degree_assortativity_real',
        ]]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}
    for (axs_col, algorithm) in enumerate(algorithms):
        # Group by algorithm and concatenate values into lists
        df_aggregate = df_interest.groupby('algorithm').agg(
            lambda x: x.tolist())
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        
        out_degree_assortativity_inferred = (
            df_aggregate.loc[algorithm].out_degree_assortativity_inferred)
        out_degree_assortativity_real = (
            df_aggregate.loc[algorithm].out_degree_assortativity_real)
        # Scatter
        axs[axs_col].scatter(
            out_degree_assortativity_real,
            out_degree_assortativity_inferred,
            )
        # Plot identity line
        axs[axs_col].plot(
            [min(out_degree_assortativity_real), max(out_degree_assortativity_real)],
            [min(out_degree_assortativity_real), max(out_degree_assortativity_real)],
            'g--')
        # Compute MSE and Person corr
        rho=np.corrcoef(
            np.array([out_degree_assortativity_real, out_degree_assortativity_inferred]))[0,1]
        mse = np.square(
            np.subtract(out_degree_assortativity_inferred, out_degree_assortativity_real)).mean()
        # Set subplot title
        axs[axs_col].set_title('{0}, MSE= {1}'.format(algorithm_names[algorithm], mse))
        # Set axes properties
        axs[axs_col].set_xlabel(
            r'$\text{Real out-degree assortativity}$', horizontalalignment='right', x=1.0)
        axs[axs_col].set_ylabel(
            r'$\text{Inferred out-degree assortativity}$')
        # axs[axs_row, axs_col].set_xscale('log')
        axs[axs_col].yaxis.set_ticks_position('both')
        #axs[axs_row, axs_col].set_xlim(left=0)
        #axs[axs_row, axs_col].set_ylim(bottom=0)
    return fig


def histogram_reciprocity_inferred_vs_real():
    # Plot reciprocity_real and reciprocity_inferred histograms (only T_interest)
    # Subplots vertical: algorithms
    subplots_v = len(algorithms)
    subplots_h = 1
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
    'reciprocity_inferred',
    'reciprocity_real',
    ]]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}

    for (axs_col, algorithm) in enumerate(algorithms):
        # Group by algorithm and concatenate values into lists
        df_aggregate = df_interest.groupby('algorithm').agg(
            lambda x: x.tolist())
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        reciprocity_inferred = (
            df_aggregate.loc[algorithm].reciprocity_inferred)
        reciprocity_real = (
            df_aggregate.loc[algorithm].reciprocity_real)
        # Real histogram
        axs[axs_col].hist(
            reciprocity_real,
            density=True,
            label='Real')
        # Inferred histogram
        axs[axs_col].hist(
            reciprocity_inferred,
            density=True,
            label='Inferred')
        # Set subplot title
        axs[axs_col].set_title('{0}'.format(algorithm_names[algorithm]))
        # Set axes properties
        axs[axs_col].set_xlabel(r'$\text{Reciprocity}$', horizontalalignment='right', x=1.0)
        axs[axs_col].yaxis.set_ticks_position('both')
        #axs[axs_col].set_ylim(bottom=0)
        # Add legend
        axs[axs_col].legend()

    return fig


def histogram_overall_reciprocity_inferred_vs_real():
    # Plot overall_reciprocity_real and overall_reciprocity_inferred histograms (only T_interest)
    # Subplots vertical: algorithms
    subplots_v = len(algorithms)
    subplots_h = 1
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
    'overall_reciprocity_inferred',
    'overall_reciprocity_real',
    ]]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}

    for (axs_col, algorithm) in enumerate(algorithms):
        # Group by algorithm and concatenate values into lists
        df_aggregate = df_interest.groupby('algorithm').agg(
            lambda x: x.tolist())
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        overall_reciprocity_inferred = (
            df_aggregate.loc[algorithm].overall_reciprocity_inferred)
        overall_reciprocity_real = (
            df_aggregate.loc[algorithm].overall_reciprocity_real)
        # Real histogram
        axs[axs_col].hist(
            overall_reciprocity_real,
            density=True,
            label='Real')
        # Inferred histogram
        axs[axs_col].hist(
            overall_reciprocity_inferred,
            density=True,
            label='Inferred')
        # Set subplot title
        axs[axs_col].set_title('{0}'.format(algorithm_names[algorithm]))
        # Set axes properties
        axs[axs_col].set_xlabel(r'$\text{Overall reciprocity}$', horizontalalignment='right', x=1.0)
        axs[axs_col].yaxis.set_ticks_position('both')
        #axs[axs_col].set_ylim(bottom=0)
        # Add legend
        axs[axs_col].legend()

    return fig


def plot_rich_club_in_degrees_inferred_vs_real():
    # Plot rich_club_in_degrees_inferred vs rich_club_in_degrees_real
    # Subplots vertical: algorithms
    subplots_v = len(algorithms)
    subplots_h = 1
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
    'rich_club_in_degrees_inferred',
    'rich_club_in_degrees_real',
    ]]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}

    # Pad with NaN vaues up to length 500
    def pad_with_nan(arr):
        return np.pad(arr.astype(float), (0,500-len(arr)), 'constant', constant_values=(np.nan,))
    df_interest['rich_club_in_degrees_real'] = df_interest['rich_club_in_degrees_real'].apply(lambda x: pad_with_nan(x))
    df_interest['rich_club_in_degrees_inferred'] = df_interest['rich_club_in_degrees_inferred'].apply(lambda x: pad_with_nan(x))

    for (axs_col, algorithm) in enumerate(algorithms):
        # Group by algorithm and concatenate values into lists
        df_aggregate = df_interest.groupby('algorithm').agg(
            lambda x: x.tolist())
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        
        rich_club_in_degrees_inferred = np.nanmean(
            np.array(df_aggregate.loc[algorithm].rich_club_in_degrees_inferred),
            axis=0)
        rich_club_in_degrees_real = np.nanmean(
            np.array(df_aggregate.loc[algorithm].rich_club_in_degrees_real),
            axis=0)
        max_degree = 30
        # Real
        axs[axs_col].plot(
            range(max_degree),
            rich_club_in_degrees_real[:max_degree],
            label='Real')
        # Inferred
        axs[axs_col].plot(
            range(max_degree),
            rich_club_in_degrees_inferred[:max_degree],
            label='Inferred')
        # Set subplot title
        axs[axs_col].set_title('{0}'.format(algorithm_names[algorithm]))
        # Set axes properties
        axs[axs_col].set_xlabel(r'$\text{In-degree}$', horizontalalignment='right', x=1.0)
        axs[axs_col].set_ylabel(r'$\text{Rich-club coefficient}$')
        axs[axs_col].yaxis.set_ticks_position('both')
        #axs[axs_col].set_ylim(bottom=0)
        # Add legend
        axs[axs_col].legend()

    return fig


def plot_rich_club_out_degrees_inferred_vs_real():
    # Plot rich_club_out_degrees_inferred vs rich_club_out_degrees_real
    # Subplots vertical: algorithms
    subplots_v = len(algorithms)
    subplots_h = 1
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
    'rich_club_out_degrees_inferred',
    'rich_club_out_degrees_real',
    ]]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}

    # Pad with NaN vaues up to length 500
    def pad_with_nan(arr):
        return np.pad(arr.astype(float), (0,500-len(arr)), 'constant', constant_values=(np.nan,))
    df_interest['rich_club_out_degrees_real'] = df_interest['rich_club_out_degrees_real'].apply(lambda x: pad_with_nan(x))
    df_interest['rich_club_out_degrees_inferred'] = df_interest['rich_club_out_degrees_inferred'].apply(lambda x: pad_with_nan(x))

    for (axs_col, algorithm) in enumerate(algorithms):
        # Group by algorithm and concatenate values into lists
        df_aggregate = df_interest.groupby('algorithm').agg(
            lambda x: x.tolist())
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        
        rich_club_out_degrees_inferred = np.nanmean(
            np.array(df_aggregate.loc[algorithm].rich_club_out_degrees_inferred),
            axis=0)
        rich_club_out_degrees_real = np.nanmean(
            np.array(df_aggregate.loc[algorithm].rich_club_out_degrees_real),
            axis=0)
        max_degree = 30
        # Real
        axs[axs_col].plot(
            range(max_degree),
            rich_club_out_degrees_real[:max_degree],
            label='Real')
        # Inferred
        axs[axs_col].plot(
            range(max_degree),
            rich_club_out_degrees_inferred[:max_degree],
            label='Inferred')
        # Set subplot title
        axs[axs_col].set_title('{0}'.format(algorithm_names[algorithm]))
        # Set axes properties
        axs[axs_col].set_xlabel(r'$\text{Out-degree}$', horizontalalignment='right', x=1.0)
        axs[axs_col].set_ylabel(r'$\text{Rich-club coefficient}$')
        axs[axs_col].yaxis.set_ticks_position('both')
        #axs[axs_col].set_ylim(bottom=0)
        # Add legend
        axs[axs_col].legend()

    return fig

# -------------------------------------------------------------------------
# endregion
# -------------------------------------------------------------------------



# -------------------------------------------------------------------------
# region Stochastic Block Model
# -------------------------------------------------------------------------

def violin_plot_SBM_precision_vs_nodes_n_TODO():
    # Violin plot of precision within and between groups vs. nodes_n
    # Subplots vertical: algorithms
    subplots_v = len(algorithms)
    subplots_h = 1
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'precision_within_groups',
        'precision_between_groups',
        ]]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    #df_interest = df_interest.loc[
    #    df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}
    for (axs_row, algorithm) in enumerate(algorithms):
        # Set subplot title
        axs[axs_row].set_title('{0}'.format(
            algorithm_names[algorithm]))
        # Select dataframe entries(i.e runs) with the same algorithm
        df_algorithm = df_interest.loc[
            df_interest['algorithm'] == algorithm].drop('algorithm', 1)
        # Group and concatenate lists
        df_aggregate = df_algorithm.groupby('nodes_n').agg(
            lambda x: x.tolist())
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        precision_within_groups = [
            df_aggregate.loc[nodes_n].precision_within_groups
            for nodes_n in nodes_n_range]
        #precision_within_groups = precision_within_groups[0]
        print(precision_within_groups)
        print(nodes_n_range)
        # Violin plot of omnibus TE vs nodes_n
        violin_within = axs[axs_row].violinplot(
            precision_within_groups,
            positions=nodes_n_range,
            widths=1,#WS_p_range/2,
            showmeans=True,
            showextrema=True,
            showmedians=False,
            #points=100,
            #bw_method=
            )
        # # Only plot right-hand half of the violins
        # for b in violin['bodies']:
        #     if len(b.get_paths()) > 0:
        #         m = np.nanmean(b.get_paths()[0].vertices[:, 0])
        #         b.get_paths()[0].vertices[:, 0] = np.clip(
        #             b.get_paths()[0].vertices[:, 0], m, np.inf)
        #         b.set_color('tab:blue')
        # # Join mean values
        # mean_vals = [np.nanmean(np.concatenate(df_aggregate.loc[WS_p].TE_omnibus_empirical)) for WS_p in WS_p_range]
        # axs[axs_row].plot(
        #     WS_p_range,
        #     mean_vals,
        #     '-o',
        #     color='tab:blue')
        # Set axes properties
        axs[axs_row].set_xlabel(
            r'$\text{N}$', horizontalalignment='right', x=1.0)
        axs[axs_row].set_ylabel(
            r'$\text{Precision}$')
        axs[axs_row].yaxis.set_ticks_position('both')
        axs[axs_row].set_ylim(bottom=0, top=1.1)
    return fig


def violin_plot_SBM_precision_OLD():
    # Violin plot of precision within and between groups vs. nodes_n
    subplots_v = 1
    subplots_h = 1
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'precision_within_groups',
        'precision_between_groups',
        ]]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}
    axs_row = 0
    #for (alg_i, algorithm) in enumerate(algorithms):
        # Select dataframe entries(i.e runs) with the same algorithm
        # df_algorithm = df_interest.loc[
        #     df_interest['algorithm'] == algorithm].drop('algorithm', 1)
        # Group and concatenate lists
        # df_aggregate = df_algorithm.groupby('nodes_n').agg(
        #     lambda x: x.tolist())
        # Ensure that only the desired parameters are aggregated or averaged
    df_algorithm = df_interest
    # df_keys_remaining = df_algorithm.columns
    # check_remaining_dimensions(df_keys_remaining, parameters_to_average)
    precision_within_groups = [df_algorithm.loc[df_algorithm['algorithm'] == algorithm].precision_within_groups.values.tolist() for algorithm in algorithms]
    precision_between_groups = [df_algorithm.loc[df_algorithm['algorithm'] == algorithm].precision_between_groups.values.tolist() for algorithm in algorithms]
    # Labels for legend
    labels = []
    def add_label(violin, label):
        color = violin["bodies"][0].get_facecolor().flatten()
        labels.append((Patch(color=color), label))
    # Violin plot of precision_within_groups
    positions = np.arange(1, len(algorithms) + 1)
    print(positions)
    violin_within = axs[axs_row].violinplot(
        precision_within_groups,
        positions=positions,
        widths=0.5,
        showmeans=True,
        showextrema=False,
        showmedians=False,
        #points=100,
        #bw_method=
        )
    add_label(violin_within, 'Within groups')    
    # Violin plot of precision_between_groups
    violin_between = axs[axs_row].violinplot(
        precision_between_groups,
        positions=positions+len(algorithms),
        widths=0.5,
        showmeans=True,
        showextrema=False,
        showmedians=False,
        #points=100,
        #bw_method=
        )
    add_label(violin_between, 'Between groups')    
    # Set axes properties
    # axs[axs_row].set_xlabel(
    #     r'$\text{N}$', horizontalalignment='right', x=1.0)
    axs[axs_row].set_ylabel(
        r'$\text{Precision}$')
    axs[axs_row].yaxis.set_ticks_position('both')
    axs[axs_row].set_xticklabels([algorithm_names[algorithm] for algorithm in algorithms]*2, rotation=90)
    axs[axs_row].set_ylim(bottom=0, top=1.1)
    axs[axs_row].legend(*zip(*labels), loc=2)
    return fig


def violin_plot_SBM_precision():
    # Violin plot of precision within and between groups vs. nodes_n
    subplots_v = 1
    subplots_h = 1
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'precision_within_groups',
        'precision_between_groups',
        ]]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}
    axs_row = 0
    # Labels for legend
    labels = []
    violins = list(range(len(algorithms)))
    def add_label(violin, label):
        color = violin["bodies"][0].get_facecolor().flatten()
        labels.append((Patch(color=color), label))
    for (alg_i, algorithm) in enumerate(algorithms):
        # Select dataframe entries(i.e runs) with the same algorithm
        df_algorithm = df_interest.loc[
            df_interest['algorithm'] == algorithm].drop('algorithm', 1)
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_algorithm.columns
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        precision_within_groups = df_algorithm.precision_within_groups.values.tolist()
        precision_between_groups = df_algorithm.precision_between_groups.values.tolist()
        # Violin plots
        positions = np.array([alg_i, alg_i + len(algorithms) + 1])
        violin_width = 0.8
        violins[alg_i] = axs[axs_row].violinplot(
            [precision_within_groups, precision_between_groups],
            positions=positions,
            widths=violin_width,
            showmeans=True,
            showextrema=False,
            showmedians=False,
            #points=100,
            #bw_method=
            )
        add_label(violins[alg_i], algorithm_names[algorithm])    
    # Set axes properties
    # axs[axs_row].set_xlabel(
    #     r'$\text{N}$', horizontalalignment='right', x=1.0)
    axs[axs_row].set_ylabel(r'$\text{Precision}$')
    axs[axs_row].yaxis.set_ticks_position('both')
    axs[axs_row].set_xticklabels(['', '', 'Within groups', '', '', '', 'Between groups', ''])
    #axs[axs_row].set_ylim(bottom=0, top=1.1)
    axs[axs_row].legend(*zip(*labels), loc='lower left')
    return fig


def violin_plot_SBM_recall():
    # Violin plot of recall within and between groups vs. nodes_n
    subplots_v = 1
    subplots_h = 1
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'recall_within_groups',
        'recall_between_groups',
        ]]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['p_value'] == alpha_interest].drop('p_value', 1)
    df_interest = df_interest.loc[
        df_interest['nodes_n'] == N_interest].drop('nodes_n', 1)
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}
    axs_row = 0
    # Labels for legend
    labels = []
    violins = list(range(len(algorithms)))
    def add_label(violin, label):
        color = violin["bodies"][0].get_facecolor().flatten()
        labels.append((Patch(color=color), label))
    for (alg_i, algorithm) in enumerate(algorithms):
        # Select dataframe entries(i.e runs) with the same algorithm
        df_algorithm = df_interest.loc[
            df_interest['algorithm'] == algorithm].drop('algorithm', 1)
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_algorithm.columns
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        recall_within_groups = df_algorithm.recall_within_groups.values.tolist()
        recall_between_groups = df_algorithm.recall_between_groups.values.tolist()
        # Violin plots
        positions = np.array([alg_i, alg_i + len(algorithms) + 1])
        violin_width = 0.8
        violins[alg_i] = axs[axs_row].violinplot(
            [recall_within_groups, recall_between_groups],
            positions=positions,
            widths=violin_width,
            showmeans=True,
            showextrema=False,
            showmedians=False,
            #points=100,
            #bw_method=
            )
        add_label(violins[alg_i], algorithm_names[algorithm])    
    # Set axes properties
    # axs[axs_row].set_xlabel(
    #     r'$\text{N}$', horizontalalignment='right', x=1.0)
    axs[axs_row].set_ylabel(r'$\text{Recall}$')
    axs[axs_row].yaxis.set_ticks_position('both')
    axs[axs_row].set_xticklabels(['', '', 'Within groups', '', '', '', 'Between groups', ''])
    #axs[axs_row].set_ylim(bottom=0, top=1.1)
    axs[axs_row].legend(*zip(*labels), loc='lower left')
    return fig


# -------------------------------------------------------------------------
# endregion
# -------------------------------------------------------------------------


# ------------------------------------------------------------------------
# region MAIN
# ------------------------------------------------------------------------

# Read shell inputs (if provided)
fdr = False  # FDR-corrected results
debug_mode = False
save_figures = True
argv = sys.argv
if len(argv) == 1:
    print('\nERROR: no path provided.\n')
if len(argv) >= 2:
    traj_dir = str(argv[1])
    if not os.path.isdir(traj_dir):
        traj_dir = os.path.join('../..', traj_dir)
if len(argv) >= 3:
    fdr = int(argv[2])
if len(argv) >= 4:
    debug_mode = int(argv[3])
if len(argv) >= 5:
    save_figures = int(argv[4])
if len(argv) >= 6:
    print('\nWARNING: too many parameters provided.\n')

# Load DataFrame
if fdr:
    df = pd.read_pickle(os.path.join(traj_dir, 'postprocessing_fdr.pkl'))
else:
    df = pd.read_pickle(os.path.join(traj_dir, 'postprocessing.pkl'))
# Initialise empty figure and axes lists
fig_list = []

# Set up plot style
# use latex based font rendering
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command
# Use "matplotlib.rcdefaults()" to restore the default plot style"
mpl.rcParams['axes.titlesize'] = 18
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['lines.markersize'] = 4
mpl.rcParams['errorbar.capsize'] = 3
mpl.rcParams['legend.fontsize'] = 8
mpl.rcParams['legend.labelspacing'] = 0.5  #0.5 default
mpl.rcParams['legend.handletextpad'] = 0.8  #0.8 default
#### SAVING FIGURES
## the default savefig params can be different from the display params
## e.g., you may want a higher resolution, or to make the figure
## background white
#savefig.dpi         : figure   ## figure dots per inch or 'figure'
#savefig.facecolor   : white    ## figure facecolor when saving
#savefig.edgecolor   : white    ## figure edgecolor when saving
#savefig.format      : png      ## png, ps, pdf, svg
#savefig.bbox        : standard ## 'tight' or 'standard'.
                                ## 'tight' is incompatible with pipe-based animation
                                ## backends but will workd with temporary file based ones:
                                ## e.g. setting animation.writer to ffmpeg will not work,
                                ## use ffmpeg_file instead
#savefig.pad_inches  : 0.1      ## Padding to be used when bbox is set to 'tight'
#savefig.jpeg_quality: 95       ## when a jpeg is saved, the default quality parameter.
#savefig.directory   : ~        ## default directory in savefig dialog box,
                                ## leave empty to always use current working directory
#savefig.transparent : False    ## setting that controls whether figures are saved with a
                                ## transparent background by default
#savefig.frameon : True			## enable frame of figure when saving
#savefig.orientation : portrait	## Orientation of saved figure
# Colours
colors_tab = [
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
# Tableau colorblind 10 palette
colors_tableau_colorblind_10 = [
    '#006BA4',
    '#FF800E',
    '#ABABAB',
    '#595959',
    '#5F9ED1',
    '#C85200',
    '#898989',
    '#A2C8EC',
    '#FFBC79',
    '#CFCFCF']
# Other colorblind palette proposed in https://github.com/matplotlib/matplotlib/issues/9460
colors_petroff = [
    '#7B85D4',
    '#f37738',
    '#83c995',
    '#d7369e',
    '#c4c9d8',
    '#859795',
    '#e9d043',
    '#ad5b50']
colors_default = colors_petroff
# Markers
markers_default = ['x', 'o', '^', 's', 'D', 'v', 'h', '*']
cycler_default = cycler(
    color=colors_petroff,
    marker=markers_default)
#    scatter.marker=markers_default)


# -------------------------------------------------------------------------
# region Network Neuroscience validation paper (Random Erdos-Renyi)
# -------------------------------------------------------------------------

# Select value of interest (for those plots where only one value is used)
# alpha_interest = 0.001
# N_interest = 100
# T_interest = 10000
# first_not_explored = 'precision'
# Ignore non-relevant explored parameters
# ignore_par = {
#     'jidt_threads_n',
#     'n_perm_max_stat',
#     'n_perm_min_stat',
#     'n_perm_max_seq',
#     'n_perm_omnibus',
# }
# parameters_explored = get_parameters_explored(first_not_explored, ignore_par)
# Get parameter ranges
# nodes_n_range = np.unique(df['nodes_n']).astype(int)
# samples_n_range = np.unique(df['samples_n']).astype(int)
# alpha_c_range = np.unique(df['p_value']).astype(float)
# new_fig('plot_performance_vs_N_scatter')
# new_fig('plot_precision_vs_N_scatter')
# new_fig('plot_precision_vs_N_scatter_inset')
# new_fig('plot_recall_vs_N_scatter')
# new_fig('plot_spectral_radius_vs_N_scatter')
# new_fig('plot_FPR_target_vs_alpha_quantile_mean')
# new_fig('plot_FPR_target_vs_alpha_quantile_N_interest')
# new_fig('plot_precision_recall_vs_alpha')
# new_fig('plot_precision_vs_recall_subplots_T_scatter')
# new_fig('plot_precision_vs_recall_subplots_T_scatter_aggregate')
# new_fig('plot_precision_vs_recall_subplots_N_scatter')
# new_fig('plot_delay_error_mean_vs_T_relative')
# new_fig('plot_delay_error_mean_vs_T_relative_alpha_interest')
# new_fig('plot_omnibus_TE_empirical_histogram_alpha_interest')
# new_fig('plot_omnibus_TE_empirical_histogram_alpha_interest_T_interest')
# new_fig('plot_omnibus_TE_empirical_vs_theoretical_causal_vars_alpha_interest')
# new_fig('plot_omnibus_TE_empirical_vs_theoretical_inferred_vars_alpha_interest')
# new_fig('plot_relative_error_TE_empirical_vs_theoretical_causal_vars_alpha_interest')

# -------------------------------------------------------------------------
# endregion
# -------------------------------------------------------------------------



# -------------------------------------------------------------------------
# region TE theoretical VAR1
# -------------------------------------------------------------------------

# Select value of interest (for those plots where only one value is used)
# N_interest = 100
# T_interest = 100000
# weight_interest = 'fixed'
# first_not_explored = 'bTE_empirical_causal_vars'
# ignore_par = {
#     'jidt_threads_n',
#     'n_perm_max_stat',
#     'n_perm_min_stat',
#     'n_perm_max_seq',
#     'n_perm_omnibus',
#     }
# parameters_explored = get_parameters_explored(first_not_explored, ignore_par)
# # Get parameter ranges
# nodes_n_range = np.unique(df['nodes_n']).astype(int)
# samples_n_range = np.unique(df['samples_n']).astype(int)
# weight_distributions = np.unique(df['weight_distribution'])
# 
# bTE_methods_legends = {
#     'bTE_empirical_causal_vars' : 'Empirical',
#     'bTE_theoretical_causal_vars' : 'Theoretical',
#     'bTE_approx2_causal_vars' : 'Order 2',
#     'bTE_approx4_causal_vars' : r'All motifs up to $\mathcal{O}(\|C\|^4)$',
#     'bTE_motifs_acde_causal_vars' : 'Motifs a + c + d + e',
#     'bTE_motifs_ae_causal_vars' : 'Motifs a + e',
# }
# 
# WS_p_range = np.unique(df['WS_p']).astype(float)
# fig_list.append(plot_bTE_vs_WS_p())
# #fig_list.append(imshow_bTE_vs_indegree_source_target())
# #fig_list.append(imshow_bTE_empirical_vs_indegree_source_target())
# #fig_list.append(imshow_bTE_theoretical_vs_indegree_source_target())

# -------------------------------------------------------------------------
# endregion
# -------------------------------------------------------------------------



# -------------------------------------------------------------------------
# region Watts-Strogatz
# -------------------------------------------------------------------------

# # Select value of interest (for those plots where only one value is used)
# alpha_interest = 0.001
# N_interest = 100
# T_interest = 10000
# weight_interest = 'deterministic'
# algorithm_interest = 'mTE_greedy'
# first_not_explored = 'precision'
# # Ignore non-relevant explored parameters
# ignore_par = {
#     'jidt_threads_n',
#     'n_perm_max_stat',
#     'n_perm_min_stat',
#     'n_perm_max_seq',
#     'n_perm_omnibus',
#     'weight_distribution'
#     }
# parameters_explored = get_parameters_explored(first_not_explored, ignore_par)
# # Get parameter ranges
# nodes_n_range = np.unique(df['nodes_n']).astype(int)
# samples_n_range = np.unique(df['samples_n']).astype(int)
# alpha_c_range = np.unique(df['p_value']).astype(float)
# WS_p_range = np.unique(df['WS_p']).astype(float)
# algorithms = np.unique(df['algorithm'])
# weight_distributions = np.unique(df['weight_distribution'])
# # Define dictionary of algorithm abbreviations
# algorithm_names = {
#     'bMI': 'Bivariate Mutual Information',
#     'bMI_greedy': 'Bivariate Mutual Information',
#     'bTE_greedy': 'Bivariate Transfer Entropy',
#     'mTE_greedy': 'Multivariate Transfer Entropy',
# }
# fig_list.append(plot_performance_vs_WS_p())
# fig_list.append(plot_density_vs_WS_p())
# fig_list.append(plot_out_degree_vs_WS_p())
# fig_list.append(plot_path_lenght_vs_WS_p())
# fig_list.append(plot_average_global_efficiency_vs_WS_p())
# fig_list.append(plot_local_efficiency_vs_WS_p())
# fig_list.append(plot_local_efficiency_correlation_vs_WS_p())
# fig_list.append(plot_clustering_vs_WS_p())
# # fig_list.append(scatter_clustering_inferred_vs_clustering_real())
# fig_list.append(plot_clustering_correlation_vs_WS_p())
# fig_list.append(plot_SW_index_vs_WS_p())
# fig_list.append(violin_TE_empirical_vs_WS_p())
# fig_list.append(violin_TE_theoretical_vs_WS_p())
# fig_list.append(imshow_TE_apparent_theoretical_causal_vars())
# fig_list.append(imshow_TE_complete_theoretical_causal_vars())
# fig_list.append(violin_AIS_theoretical_vs_WS_p())
# # fig_list.append(plot_omnibus_TE_empirical_vs_theoretical_causal_vars())
# # # fig_list.append(plot_omnibus_TE_empirical_vs_theoretical_inferred_vars_alpha_interest())
# # fig_list.append(plot_relative_error_TE_empirical_vs_theoretical_causal_vars_vs_WS_p())
# # fig_list.append(scatter_performance_vs_omnibus_TE())
# # # fig_list.append(scatter_performance_vs_out_degree_real())
# # # fig_list.append(scatter_omnibus_TE_vs_out_degree_real())
# # fig_list.append(scatter_performance_vs_clustering_real())
# # # fig_list.append(scatter_performance_vs_lattice_distance())
# fig_list.append(plot_performance_vs_WS_p_group_distance())
# fig_list.append(plot_spectral_radius_vs_WS_p_alpha_interest())
# fig_list.append(barchart_precision_per_motif())
# fig_list.append(barchart_recall_per_motif())



#parameters_explored = df.loc[[], :'TE_omnibus_theoretical_causal_vars'].keys().tolist()[:-1]
# self_couplings = np.unique(df['self_coupling'])
# cross_couplings = np.unique(df['total_cross_coupling'])
# fig_list.append(violin_TE_apparent_and_conditional_pairs())

# -------------------------------------------------------------------------
# endregion
# -------------------------------------------------------------------------



# -------------------------------------------------------------------------
# region Barabasi-Albert
# -------------------------------------------------------------------------
# # Select value of interest (for those plots where only one value is used)
# alpha_interest = 0.001
# N_interest = 200
# T_interest = 10000
# weight_interest = 'fixed'
# algorithm_interest = 'mTE_greedy'
# first_not_explored = 'precision'
# # Ignore non-relevant explored parameters
# ignore_par = {
#     'jidt_threads_n',
#     'n_perm_max_stat',
#     'n_perm_min_stat',
#     'n_perm_max_seq',
#     'n_perm_omnibus',
#     'weight_distribution'
#     }
# parameters_explored = get_parameters_explored(first_not_explored, ignore_par)
# # Get parameter ranges
# nodes_n_range = np.unique(df['nodes_n']).astype(int)
# samples_n_range = np.unique(df['samples_n']).astype(int)
# alpha_c_range = np.unique(df['p_value']).astype(float)
# algorithms = np.unique(df['algorithm'])
# weight_distributions = np.unique(df['weight_distribution'])
# # Define dictionary of algorithm abbreviations
# algorithm_names = {
#     'bMI': 'Bivariate Mutual Information',
#     'bMI_greedy': 'Bivariate Mutual Information',
#     'bTE_greedy': 'Bivariate Transfer Entropy',
#     'mTE_greedy': 'Multivariate Transfer Entropy',
# }
# 
# fig_list.append(scatter_performance_vs_in_degree_real())
# fig_list.append(scatter_performance_vs_out_degree_real())
# fig_list.append(scatter_FP_vs_in_degree_real())
# #fig_list.append(plot_performance_vs_N_scatter())
# #fig_list.append(scatter_performance_vs_omnibus_TE())
# fig_list.append(scatter_in_degree_inferred_vs_real())
# fig_list.append(scatter_out_degree_inferred_vs_real())
# fig_list.append(histogram_in_degree_inferred_vs_real())
# fig_list.append(histogram_out_degree_inferred_vs_real())
# fig_list.append(loglog_distributions_in_degree_inferred_vs_real())
# fig_list.append(loglog_distributions_out_degree_inferred_vs_real())
# fig_list.append(scatter_in_degree_assortativity_inferred_vs_real())
# fig_list.append(scatter_out_degree_assortativity_inferred_vs_real())
# fig_list.append(plot_rich_club_in_degrees_inferred_vs_real())
# fig_list.append(plot_rich_club_out_degrees_inferred_vs_real())
# fig_list.append(histogram_reciprocity_inferred_vs_real())
# fig_list.append(histogram_overall_reciprocity_inferred_vs_real())

# -------------------------------------------------------------------------
# endregion
# -------------------------------------------------------------------------



# -------------------------------------------------------------------------
# region Stochastic Block Model
# -------------------------------------------------------------------------

# Select value of interest (for those plots where only one value is used)
alpha_interest = 0.001
N_interest = 50
T_interest = 10000
algorithm_interest = 'mTE_greedy'
first_not_explored = 'precision'
# Ignore non-relevant explored parameters
ignore_par = {
    'jidt_threads_n',
    'n_perm_max_stat',
    'n_perm_min_stat',
    'n_perm_max_seq',
    'n_perm_omnibus',
    'weight_distribution'
    }
parameters_explored = get_parameters_explored(first_not_explored, ignore_par)
# Get parameter ranges
nodes_n_range = np.unique(df['nodes_n']).astype(int)
samples_n_range = np.unique(df['samples_n']).astype(int)
alpha_c_range = np.unique(df['p_value']).astype(float)
algorithms = np.unique(df['algorithm'])
# Define dictionary of algorithm abbreviations
algorithm_names = {
    'bMI': 'Bivariate Mutual Information',
    'bMI_greedy': 'Bivariate Mutual Information',
    'bTE_greedy': 'Bivariate Transfer Entropy',
    'mTE_greedy': 'Multivariate Transfer Entropy',
}
fig_list.append(violin_plot_SBM_precision())
fig_list.append(violin_plot_SBM_recall())

# -------------------------------------------------------------------------
# endregion
# -------------------------------------------------------------------------





# ----------------------------------------------------------------------------
# Test area
# ----------------------------------------------------------------------------










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