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
import powerlaw


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
        axs[axs_i].set_title(r'$\text{network size = }${}'.format(nodes_n))
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
        axs[axs_i].set_xlabel(r'$\text{T}$', horizontalalignment='right', x=1.0)
        axs[axs_i].set_ylabel(r'$\text{Performance}$')  # , horizontalalignment='right', y=1.0)
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
        axs[axs_row].set_xlabel(r'$\text{Network size}$', horizontalalignment='right', x=1.0)
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
    axs[0].set_xlabel(r'$\text{Network size}$', horizontalalignment='right', x=1.0)
    axs[0].set_ylabel(r'$\text{Precision}$')  # , horizontalalignment='right', y=1.0)
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
    axs[0].set_xlabel(r'$\text{Network size}$', horizontalalignment='right', x=1.0)
    axs[0].set_ylabel(r'$\text{Precision}$')  # , horizontalalignment='right', y=1.0)
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
    axs[0].set_xlabel(r'$\text{Network size}$', horizontalalignment='right', x=1.0)
    axs[0].set_ylabel(r'$\text{Recall}$')  # , horizontalalignment='right', y=1.0)
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
    axs[0].set_xlabel(r'$\text{Network size}$', horizontalalignment='right', x=1.0)
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
        axs[axs_row].set_ylabel(r'$\text{False positive rate}$')  # , horizontalalignment='right', y=1.0)
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
        axs[axs_row].set_ylabel(r'$\text{False positive rate}$')  # , horizontalalignment='right', y=1.0)
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
    # Subplots: T
    # Legend: number of nodes
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
        axs[axs_row].set_ylabel(r'$\text{False positive rate}$')  # , horizontalalignment='right', y=1.0)
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
        axs[axs_row].set_ylabel(r'$\text{False positive rate}$')  # , horizontalalignment='right', y=1.0)
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
        axs[axs_row][0].set_ylabel(r'$\text{Precision}$')  # , horizontalalignment='right', y=1.0)
        axs[axs_row][0].yaxis.set_ticks_position('both')
        axs[axs_row][0].set_ylim(bottom=0)

        axs[axs_row][1].set_xlabel(
            r'$\alpha_{{max}}$',
            horizontalalignment='right',
            x=1.0)
        axs[axs_row][1].set_xscale('log')
        axs[axs_row][1].set_ylabel(r'$\text{Recall}$')  # , horizontalalignment='right', y=1.0)
        axs[axs_row][1].set_ylim(bottom=0)
        # Add legend
        legend = [r'$N={}$'.format(nodes_n) for nodes_n in nodes_n_range]
        axs[axs_row][0].legend(labels=legend)
        axs[axs_row][1].legend(labels=legend)

    return fig


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
        axs[axs_row].set_xlabel(r'$\text{Recall}$', horizontalalignment='right', x=1.0)
        axs[axs_row].set_ylabel(r'$\text{Precision}$')#, horizontalalignment='right', y=1.0)
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
        axs[axs_row].set_xlabel(r'$\text{Recall}$', horizontalalignment='right', x=1.0)
        axs[axs_row].set_ylabel(r'$\text{Precision}$')  # , horizontalalignment='right', y=1.0)
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
    axs[0].set_xlabel(r'$\text{Recall}$', horizontalalignment='right', x=1.0)
    axs[0].set_ylabel(r'$\text{Precision}$')  # , horizontalalignment='right', y=1.0)
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
            axs[axs_row].set_ylabel(r'$\text{Absolute error}$')  # , horizontalalignment='right', y=1.0)
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
            #axs[axs_row].set_ylabel(r'$\text{Relative error}$')  # , horizontalalignment='right', y=1.0)
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
            axs[axs_row].set_ylabel(r'$\text{Absolute error}$')  # , horizontalalignment='right', y=1.0)
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
            axs[axs_row].set_ylabel(r'$\text{Lag error}$')  # , horizontalalignment='right', y=1.0)
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
                r'$\text{Omnibus TE empirical}$', horizontalalignment='right', x=1.0)
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
    axs[0].set_xlabel(r'$\text{Omnibus TE empirical}$', horizontalalignment='right', x=1.0)
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
    fig.suptitle(r'$\text{Omnibus TE theoretical (causal vars) vs. Omnibus TE empirical}$')    
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
                r'$\text{TE empirical}$', horizontalalignment='right', x=1.0)
            axs[axs_row, axs_col].set_ylabel(
                r'$\text{TE theoretical}$')
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
    fig.suptitle(r'$\text{Omnibus TE theoretical (inferred vars) vs. Omnibus TE empirical}$')
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
                r'$\text{TE empirical}$', horizontalalignment='right', x=1.0)
            axs[axs_row, axs_col].set_ylabel(
                r'$\text{TE theoretical}$')
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
    axs[0].set_ylabel(r'$\text{Relative TE error}$')
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].yaxis.set_ticks_position('both')

    return fig


def plot_path_lenght_vs_alpha_subplot_algorithm():
    # Plot average path lenght vs. alpha (scatter error)
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
        # Group by p_value
        df_aggregate = df_algorithm.groupby('p_value').agg(
            lambda x: list(x))
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        # Plot average path lenght real
        axs[axs_row].plot(
            alpha_c_range,
            [np.nanmean(df_aggregate['average_shortest_path_length_real'][p_value])
                for p_value in alpha_c_range],
        )
        axs[axs_row].scatter(
            df_algorithm['p_value'].values.astype(float).tolist(),
            df_algorithm['average_shortest_path_length_real'].values,
            alpha=0.3
        )
        # Plot average path lenght inferred
        axs[axs_row].plot(
            alpha_c_range,
            [np.nanmean(df_aggregate['average_shortest_path_length_inferred'][p_value])
             for p_value in alpha_c_range],
        )
        axs[axs_row].scatter(
            df_algorithm['p_value'].values.astype(float).tolist(),
            df_algorithm['average_shortest_path_length_inferred'].values,
            alpha=0.3
        )
        # Set axes properties
        axs[axs_row].set_xlabel(r'$\alpha_c$', horizontalalignment='right', x=1.0)
        axs[axs_row].set_ylabel(r'$\text{Characteristic path length}$')
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
            r'$\text{Rewiring probability}$', horizontalalignment='right', x=1.0)
        axs[axs_row].set_ylabel(
            r'$\text{Omnibus TE}$')
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
            r'$\text{Rewiring probability}$', horizontalalignment='right', x=1.0)
        axs[axs_row].set_ylabel(
            r'$\text{Complete TE}$')
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
            r'$\text{Rewiring probability}$', horizontalalignment='right', x=1.0)
        axs[axs_row].set_ylabel(
            r'$\text{TE (empirical)}$')
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
            r'$\text{Rewiring probability}$', horizontalalignment='right', x=1.0)
        axs[axs_row].set_xscale('log')
        axs[axs_row].yaxis.set_ticks_position('both')
        if algorithm == 'bTE_greedy':
            axs[axs_row].set_ylabel(r'$\text{Apparent TE (theoretical)}$')
        if algorithm == 'mTE_greedy':
            axs[axs_row].set_ylabel(r'$\text{Complete TE (theoretical)}$')
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
            r'$\text{Rewiring probability}$', horizontalalignment='right', x=1.0)
        axs[axs_row].set_ylabel(
            r'$\text{AIS (theoretical)}$')
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
                r'$\text{Rewiring probability}$', horizontalalignment='right', x=1.0)
            axs[axs_row, axs_col].set_ylabel(
                r'$\text{TE (theoretical)}$')
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
                r'$\text{TE theoretical}$', horizontalalignment='right', x=1.0)
            axs[axs_row, axs_col].set_ylabel(
                r'$\text{TE empirical}$')
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
    axs[0].set_xlabel(r'$\text{Rewiring probability}$', horizontalalignment='right', x=1.0)
    axs[0].set_ylabel(r'$\text{Relative TE error}$')
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
    axs[0].set_xlabel(r'$\text{Rewiring probability}$', horizontalalignment='right', x=1.0)
    axs[0].set_ylabel(r'$\rho$')
    axs[0].set_xscale('log')
    axs[0].yaxis.set_ticks_position('both')

    return fig


def plot_performance_vs_WS_p_T_interest():
    # Plot performance tests vs. Watts-Strogatz rewiring
    # probability (scatter error)
    # Subplots: algorithms
    subplots_v = len(algorithms)
    subplots_h = 1
    fig, axs = my_subplots(subplots_v, subplots_h)
    # Select data of interest
    df_interest = df[
        parameters_explored + ['precision', 'recall', 'specificity']]
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
        axs[axs_row].set_xlabel(r'$\text{Rewiring probability}$', horizontalalignment='right', x=1.0)
        axs[axs_row].set_ylabel(r'$\text{Performance}$')
        axs[axs_row].set_xscale('log')
        axs[axs_row].yaxis.set_ticks_position('both')
        axs[axs_row].set_ylim(bottom=0)
        # Add legend
        axs[axs_row].legend(
            labels=['Precision', 'Recall', 'Specificity']#,
            #loc=(0.05, 0.16)
            )

    return fig


def plot_performance_vs_WS_p():
    # Plot performance tests vs. Watts-Strogatz rewiring
    # probability (scatter error)
    # Subplots: algorithms
    subplots_v = len(algorithms)
    subplots_h = len(samples_n_range)
    fig, axs = my_subplots(subplots_v, subplots_h)
    # Select data of interest
    df_interest = df[
        parameters_explored + ['precision', 'recall', 'specificity']]
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'repetition_i'}
    for (axs_col, samples_n) in enumerate(samples_n_range):
        # Select dataframe entries(i.e runs) with the desired property
        df_samples_n = df_interest.loc[
            df_interest['samples_n'] == samples_n].drop('samples_n', 1)
        for (axs_row, algorithm) in enumerate(algorithms):
            # Set subplot title
            axs[axs_row, axs_col].set_title('{0} (T={1})'.format(
                algorithm_names[algorithm],
                samples_n))
            # Select dataframe entries(i.e runs) with the same algorithm
            df_algorithm = df_samples_n.loc[
                df_interest['algorithm'] == algorithm].drop('algorithm', 1)
            # Group by WS_p
            df_aggregate = df_algorithm.groupby('WS_p').agg(
                lambda x: list(x))
            # Ensure that only the desired parameters are aggregated or averaged
            df_keys_remaining = df_aggregate.columns.get_level_values(0)
            check_remaining_dimensions(df_keys_remaining, parameters_to_average)
            # Plot precision
            axs[axs_row, axs_col].plot(
                WS_p_range,
                [np.nanmean(df_aggregate['precision'][WS_p])
                    for WS_p in WS_p_range],)
            axs[axs_row, axs_col].scatter(
                df_algorithm['WS_p'].values.astype(float).tolist(),
                df_algorithm['precision'].values,
                alpha=0.3)
            # Plot recall
            axs[axs_row, axs_col].plot(
                WS_p_range,
                [np.nanmean(df_aggregate['recall'][WS_p])
                for WS_p in WS_p_range],)
            axs[axs_row, axs_col].scatter(
                df_algorithm['WS_p'].values.astype(float).tolist(),
                df_algorithm['recall'].values,
                alpha=0.3)
            # Plot false positive rate
            axs[axs_row, axs_col].plot(
                WS_p_range,
                [np.nanmean(df_aggregate['specificity'][WS_p])
                for WS_p in WS_p_range])
            axs[axs_row, axs_col].scatter(
                df_algorithm['WS_p'].values.astype(float).tolist(),
                df_algorithm['specificity'].values,
                alpha=0.3)
            # Set axes properties
            axs[axs_row, 0].set_ylabel(r'$\text{Performance}$')
            axs[axs_row, axs_col].set_xscale('log')
            axs[axs_row, axs_col].yaxis.set_ticks_position('both')
            axs[axs_row, axs_col].set_xlim(left=0.003, right=2.)
            axs[axs_row, axs_col].set_ylim(bottom=0)
            # Add legend
            axs[axs_row, axs_col].legend(
                labels=['Precision', 'Recall', 'Specificity'],
                loc='lower left')
        axs[axs_row, axs_col].set_xlabel(r'$\text{Rewiring probability}$') #, horizontalalignment='right', x=1.0)

    return fig


def plot_density_vs_WS_p():
    # Plot density vs. Watts-Strogatz rewiring prob
    # Legend: algorithms
    fig, axs = my_subplots(1, 1, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'density_real',
        'density_inferred']]
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'repetition_i'}
    # Plot real values
    df_aggregate = df_interest.groupby('WS_p').agg(lambda x: list(x))
    axs[0].plot(
        WS_p_range,
        [np.nanmean(df_aggregate['density_real'][WS_p])
            for WS_p in WS_p_range],
        label='Real',
        linestyle = 'None',
        color='k',
        marker='x',
        fillstyle='none',
        zorder=99,  # bring to foreground
        )
    for algorithm_i, algorithm in enumerate(algorithms):
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
            label=algorithm_names[algorithm],
            color=colors_default[algorithm_i + 1],
            marker=markers_default[algorithm_i + 1],)
        axs[0].scatter(
            df_algorithm['WS_p'].values.astype(float).tolist(),
            df_algorithm['density_inferred'].values,
            alpha=0.3,
            color=colors_default[algorithm_i + 1],
            marker=markers_default[algorithm_i + 1],
            )
    # Set axes properties
    axs[0].set_xlabel(r'$\text{Rewiring probability}$', horizontalalignment='right', x=1.0)
    axs[0].set_ylabel(r'$\text{Density}$')
    axs[0].set_xscale('log')
    axs[0].yaxis.set_ticks_position('both')
    # Add legend
    axs[0].legend(loc='upper left')

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
                r'$\text{Clustering real}$', horizontalalignment='right', x=1.0)
            axs[axs_row, axs_col].set_ylabel(
                r'$\text{Clustering inferred}$')
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
            r'$\text{Clustering coefficient}$', horizontalalignment='right', x=1.0)
        axs[axs_row].set_ylabel(
            r'$\text{CDF}$')
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
        axs[axs_row].set_xlabel(r'$\text{Rewiring probability}$', horizontalalignment='right', x=1.0)
        axs[axs_row].set_ylabel(r'$\text{Clustering coefficient}$')
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
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'repetition_i'}
    # Plot real values
    df_aggregate = df_interest.groupby('WS_p').agg(lambda x: list(x))
    axs[0].plot(
        WS_p_range,
        [np.nanmean(df_aggregate['clustering_real'][WS_p])
            for WS_p in WS_p_range],
        label='Real',
        linestyle = 'None',
        color='k',
        marker='x',
        fillstyle='none',
        zorder=99,  # bring to foreground
        )
    # axs[0].scatter(
    #     df_interest['WS_p'].values.astype(float).tolist(),
    #     [np.nanmean(x) for x in df_interest['clustering_real'].values],
    #     alpha=0.1,
    #     color='k',
    #     marker='x',
    #     zorder=99,  # bring to foreground
    #     )
    for algorithm_i, algorithm in enumerate(algorithms):
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
            label=algorithm_names[algorithm],
            color=colors_default[algorithm_i + 1],
            marker=markers_default[algorithm_i + 1],
            )
        axs[0].scatter(
            df_algorithm['WS_p'].values.astype(float).tolist(),
            [np.nanmean(x) for x in df_algorithm['clustering_inferred'].values],
            alpha=0.3,
            color=colors_default[algorithm_i + 1],
            marker=markers_default[algorithm_i + 1],
            )
        # Set axes properties
        axs[0].set_xlabel(r'$\text{Rewiring probability}$', horizontalalignment='right', x=1.0)
        axs[0].set_ylabel(r'$\text{Clustering coefficient}$')
        axs[0].set_xscale('log')
        axs[0].yaxis.set_ticks_position('both')
        axs[0].set_xlim(left=0.003, right=2.)
        # Add legend
        axs[0].legend(loc='upper right')

    return fig


def plot_clustering_correlation_vs_WS_p():
    # Plot correlation of real and real clustering coefficient
    # vs. Watts-Strogatz rewiring prob (scatter error)
    fig, axs = my_subplots(1, 1, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'clustering_real',
        'clustering_inferred']]
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
        axs[0].set_xlabel(r'$\text{Rewiring probability}$', horizontalalignment='right', x=1.0)
        axs[0].set_ylabel(r'$\text{Correlation}$')
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
                r'$\text{Characteristic path length (real)}$', horizontalalignment='right', x=1.0)
            axs[axs_row, axs_col].set_ylabel(
                r'$\text{Characteristic path length (inferred)}$')
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
        axs[axs_row].set_xlabel(r'$\text{Rewiring probability}$', horizontalalignment='right', x=1.0)
        axs[axs_row].set_ylabel(r'$\text{Characteristic path length}$')
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
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'repetition_i'}
    # Plot average path lenght real
    df_aggregate = df_interest.groupby('WS_p').agg(lambda x: list(x))
    axs[0].plot(
        WS_p_range,
        [np.nanmean(df_aggregate['average_shortest_path_length_real'][WS_p])
            for WS_p in WS_p_range],
        label='Real',
        linestyle = 'None',
        color='k',
        marker='x',
        fillstyle='none',
        zorder=99,  # bring to foreground
        )
    # axs[0].scatter(
    #     df_interest['WS_p'].values.astype(float).tolist(),
    #     df_interest['average_shortest_path_length_real'].values,
    #     marker='x',
    #     color='k',
    #     alpha=0.1)
    for algorithm_i, algorithm in enumerate(algorithms):
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
            label=algorithm_names[algorithm],
            color=colors_default[algorithm_i + 1],
            marker=markers_default[algorithm_i + 1],
            )
        axs[0].scatter(
            df_algorithm['WS_p'].values.astype(float).tolist(),
            df_algorithm['average_shortest_path_length_inferred'].values,
            alpha=0.3,
            color=colors_default[algorithm_i + 1],
            marker=markers_default[algorithm_i + 1],
            )
        # Set axes properties
        axs[0].set_xlabel(r'$\text{Rewiring probability}$', horizontalalignment='right', x=1.0)
        axs[0].set_ylabel(r'$\text{Characteristic path length}$')
        axs[0].set_xscale('log')
        axs[0].yaxis.set_ticks_position('both')
        axs[0].set_xlim(left=0.003, right=2.)
        # Add legend
        axs[0].legend(loc='upper right')

    return fig


def plot_average_global_efficiency_vs_WS_p():
    # Plot global_efficiency vs. Watts-Strogatz rewiring prob (scatter error)
    fig, axs = my_subplots(1, 1, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'average_global_efficiency_real',
        'average_global_efficiency_inferred']]
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'repetition_i'}
    # Plot global_efficiency real
    df_aggregate = df_interest.groupby('WS_p').agg(lambda x: list(x))
    axs[0].plot(
        WS_p_range,
        [np.nanmean(df_aggregate['average_global_efficiency_real'][WS_p])
            for WS_p in WS_p_range],
        label='Real',
        linestyle = 'None',
        color='k',
        marker='x',
        fillstyle='none',
        zorder=99,  # bring to foreground
        )
    # axs[0].scatter(
    #     df_algorithm['WS_p'].values.astype(float).tolist(),
    #     df_algorithm['average_global_efficiency_real'].values,
    #     alpha=0.3)
    for algorithm_i, algorithm in enumerate(algorithms):
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
            label=algorithm_names[algorithm],
            color=colors_default[algorithm_i + 1],
            marker=markers_default[algorithm_i + 1],
            )
        axs[0].scatter(
            df_algorithm['WS_p'].values.astype(float).tolist(),
            df_algorithm['average_global_efficiency_inferred'].values,
            alpha=0.3,
            color=colors_default[algorithm_i + 1],
            marker=markers_default[algorithm_i + 1],
            )
        # Set axes properties
        axs[0].set_xlabel(r'$\text{Rewiring probability}$', horizontalalignment='right', x=1.0)
        axs[0].set_ylabel(r'$\text{Global efficiency}$')
        axs[0].set_xscale('log')
        axs[0].yaxis.set_ticks_position('both')
        axs[0].set_xlim(left=0.003, right=2.)
        # Add legend
        axs[0].legend(loc='lower right')

    return fig


def plot_local_efficiency_vs_WS_p():
    # Plot local efficiency vs. Watts-Strogatz rewiring prob (scatter error)
    fig, axs = my_subplots(1, 1, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'local_efficiency_real',
        'local_efficiency_inferred']]
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'repetition_i'}
    # Real
    df_aggregate = df_interest.groupby('WS_p').agg(lambda x: list(x))
    axs[0].plot(
        WS_p_range,
        [np.nanmean(df_aggregate['local_efficiency_real'][WS_p])
            for WS_p in WS_p_range],
        label='Real',
        linestyle = 'None',
        color='k',
        marker='x',
        fillstyle='none',
        zorder=99,  # bring to foreground
        )
    # axs[0].scatter(
    #     df_algorithm['WS_p'].values.astype(float).tolist(),
    #     [np.nanmean(x) for x in df_algorithm['local_efficiency_real'].values],
    #     alpha=0.3,
    #     linestyle = 'None',
    #     color='k',
    #     marker='x',
    #     zorder=99,  # bring to foreground
    #     )
    for algorithm_i, algorithm in enumerate(algorithms):
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
            label=algorithm_names[algorithm],
            color=colors_default[algorithm_i + 1],
            marker=markers_default[algorithm_i + 1],
            )
        axs[0].scatter(
            df_algorithm['WS_p'].values.astype(float).tolist(),
            [np.nanmean(x) for x in df_algorithm['local_efficiency_inferred'].values],
            alpha=0.3,
            color=colors_default[algorithm_i + 1],
            marker=markers_default[algorithm_i + 1],
            )
        # Set axes properties
        axs[0].set_xlabel(r'$\text{Rewiring probability}$', horizontalalignment='right', x=1.0)
        axs[0].set_ylabel(r'$\text{Local efficiency}$')
        axs[0].set_xscale('log')
        axs[0].yaxis.set_ticks_position('both')
        axs[0].set_xlim(left=0.003, right=2.)
        # Add legend
        axs[0].legend(loc='lower left')

    return fig


def plot_local_efficiency_correlation_vs_WS_p():
    # Plot correlation of real and inferred local efficiency
    # vs. Watts-Strogatz rewiring prob (scatter error)
    fig, axs = my_subplots(1, 1, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'local_efficiency_real',
        'local_efficiency_inferred']]
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
        axs[0].set_xlabel(r'$\text{Rewiring probability}$', horizontalalignment='right', x=1.0)
        axs[0].set_ylabel(r'$\text{Correlation}$')
        axs[0].set_xscale('log')
        axs[0].yaxis.set_ticks_position('both')
        # Add legend
        axs[0].legend()

    return fig


def plot_SW_coeff_vs_WS_p(WS_k=4, normalised=True):
    # Plot average clustering vs. Watts-Strogatz rewiring prob (scatter error)
    fig, axs = my_subplots(1, 1, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'clustering_real',
        'clustering_inferred',
        'average_shortest_path_length_real',
        'average_shortest_path_length_inferred']]
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'repetition_i'}
    # Compute normalisation coefficients
    if normalised:
        C_lattice = (3 * WS_k - 6) / (4 * WS_k - 4)
        L_lattice = N_interest / (2 * WS_k)
    else:
        C_lattice = 1
        L_lattice = 1
    # Plot SW coefficient real
    df_aggregate = df_interest.groupby('WS_p').agg(lambda x: list(x))
    axs[0].plot(
        WS_p_range,
        (np.array([np.nanmean(df_aggregate['clustering_real'][WS_p]) for WS_p in WS_p_range]) / C_lattice) / (np.array([np.nanmean(df_aggregate['average_shortest_path_length_real'][WS_p]) for WS_p in WS_p_range]) / L_lattice),
        label='Real',
        linestyle = 'None',
        color='k',
        marker='x',
        fillstyle='none',
        zorder=99,  # bring to foreground
        )
    # axs[0].scatter(
    #     df_interest['WS_p'].values.astype(float).tolist(),
    #     [np.nanmean(x) / C_lattice for x in df_interest['clustering_real'].values] / (df_interest['average_shortest_path_length_real'].to_numpy() / L_lattice),
    #     alpha=0.1,
    #     linestyle = 'None',
    #     color='k',
    #     marker='x',
    #     zorder=99,  # bring to foreground
    #     )
    for algorithm_i, algorithm in enumerate(algorithms):
        # Select dataframe entries(i.e runs) with the same algorithm
        df_algorithm = df_interest.loc[
            df_interest['algorithm'] == algorithm].drop('algorithm', 1)
        # Group by WS_p
        df_aggregate = df_algorithm.groupby('WS_p').agg(
            lambda x: list(x))
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        # Plot SW coefficient inferred
        axs[0].plot(
            WS_p_range,
            (np.array([np.nanmean(df_aggregate['clustering_inferred'][WS_p]) for WS_p in WS_p_range]) / C_lattice) / (np.array([np.nanmean(df_aggregate['average_shortest_path_length_inferred'][WS_p]) for WS_p in WS_p_range]) / L_lattice),
            label=algorithm_names[algorithm],
            color=colors_default[algorithm_i + 1],
            marker=markers_default[algorithm_i + 1],
            )
        axs[0].scatter(
            df_algorithm['WS_p'].values.astype(float).tolist(),
            [np.nanmean(x) / C_lattice for x in df_algorithm['clustering_inferred'].values] / (df_algorithm['average_shortest_path_length_inferred'].to_numpy() / L_lattice),
            alpha=0.3,
            color=colors_default[algorithm_i + 1],
            marker=markers_default[algorithm_i + 1],
            )
    # Set axes properties
    axs[0].set_xlabel(r'$\text{Rewiring probability}$', horizontalalignment='right', x=1.0)
    axs[0].set_ylabel(r'$\text{Small-world coefficient}$')
    axs[0].set_xscale('log')
    axs[0].yaxis.set_ticks_position('both')
    axs[0].set_xlim(left=0.003, right=2.)
    # Add legend
    axs[0].legend(loc='upper right')

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
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'repetition_i'}
    # Compute normalisation coefficients (see Neal 2017, DOI: 10.1017/nws.2017.5)
    C_lattice = (3 * WS_k - 6) / (4 * WS_k - 4)
    L_lattice = N_interest / (2 * WS_k)
    C_random = WS_k / N_interest
    L_random = np.log(N_interest) / np.log(WS_k)
    # Plot SW index real
    df_aggregate = df_interest.groupby('WS_p').agg(lambda x: list(x))
    C_real = np.array([np.nanmean(df_aggregate['clustering_real'][WS_p]) for WS_p in WS_p_range])
    L_real = np.array([np.nanmean(df_aggregate['average_shortest_path_length_real'][WS_p]) for WS_p in WS_p_range])
    axs[0].plot(
        WS_p_range,
        ((L_real - L_lattice) / (L_random - L_lattice)) * ((C_real - C_random) / (C_lattice - C_random)),
        label='Real',
        linestyle = 'None',
        color='k',
        marker='x',
        fillstyle='none',
        zorder=99,  # bring to foreground
        )
    # axs[0].scatter(
    #     df_algorithm['WS_p'].values.astype(float).tolist(),
    #     ((df_algorithm['average_shortest_path_length_real'].to_numpy() - L_lattice) / (L_random - L_lattice)) * (np.array([np.nanmean(x) - C_random for x in df_algorithm['clustering_real'].values]) / (C_lattice - C_random)),
    #     alpha=0.3,
    #     linestyle = 'None',
    #     color='k',
    #     marker='x',
    #     zorder=99,  # bring to foreground
    #     )
    for algorithm_i, algorithm in enumerate(algorithms):
        # Select dataframe entries(i.e runs) with the same algorithm
        df_algorithm = df_interest.loc[
            df_interest['algorithm'] == algorithm].drop('algorithm', 1)
        # Group by WS_p
        df_aggregate = df_algorithm.groupby('WS_p').agg(
            lambda x: list(x))
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        L_inferred = np.array([np.nanmean(df_aggregate['average_shortest_path_length_inferred'][WS_p]) for WS_p in WS_p_range])
        C_inferred = np.array([np.nanmean(df_aggregate['clustering_inferred'][WS_p]) for WS_p in WS_p_range])
        # Plot SW index inferred
        axs[0].plot(
            WS_p_range,
            ((L_inferred - L_lattice) / (L_random - L_lattice)) * ((C_inferred - C_random) / (C_lattice - C_random)),
            label=algorithm_names[algorithm],
            color=colors_default[algorithm_i + 1],
            marker=markers_default[algorithm_i + 1],
            )
        axs[0].scatter(
            df_algorithm['WS_p'].values.astype(float).tolist(),
            ((df_algorithm['average_shortest_path_length_inferred'].to_numpy() - L_lattice) / (L_random - L_lattice)) * (np.array([np.nanmean(x) - C_random for x in df_algorithm['clustering_inferred'].values]) / (C_lattice - C_random)),
            alpha=0.3,
            color=colors_default[algorithm_i + 1],
            marker=markers_default[algorithm_i + 1],
            )
    # Set axes properties
    axs[0].set_xlabel(r'$\text{Rewiring probability}$', horizontalalignment='right', x=1.0)
    axs[0].set_ylabel(r'$\text{Small-world index}$')
    axs[0].set_xscale('log')
    axs[0].yaxis.set_ticks_position('both')
    axs[0].set_xlim(left=0.003, right=2.)
    # Add legend
    axs[0].legend(loc='upper right')

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
                r'$\text{Out-degree real}$', horizontalalignment='right', x=1.0)
            axs[axs_row, axs_col].set_ylabel(
                r'$\text{Out-degree inferred}$')
            # axs[axs_row, axs_col].set_xscale('log')
            axs[axs_row, axs_col].yaxis.set_ticks_position('both')
            #axs[axs_row, axs_col].set_xlim(left=0)
            #axs[axs_row, axs_col].set_ylim(bottom=0)
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
                r'$\text{Real out-degree}$', horizontalalignment='right', x=1.0)
            axs[axs_row, axs_col].set_ylabel(
                r'$\text{TE}$')
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
                r'$\text{Real out-degree}$', horizontalalignment='right', x=1.0)
            axs[axs_row, axs_col].set_ylabel(
                r'$\text{Performance}$')
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
                r'$\text{Real clustering coefficient}$', horizontalalignment='right', x=1.0)
            axs[axs_row, axs_col].set_ylabel(
                r'$\text{Performance}$')
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
                r'$\text{TE}$', horizontalalignment='right', x=1.0)
            axs[axs_row, axs_col].set_ylabel(
                r'$\text{Performance}$')
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
                r'$\text{Distance}$', horizontalalignment='right', x=1.0)
            axs[axs_row, axs_col].set_ylabel(
                r'$\text{Performance}$')
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
        axs[axs_row].set_xlabel(r'$\text{Rewiring probability}$', horizontalalignment='right', x=1.0)
        axs[axs_row].set_ylabel(r'$\text{Performance}$')
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
        #     r'$\text{Motif}$', horizontalalignment='right', x=1.0)
        ax_main.set_ylabel(r'$\text{Precision}$')
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
        ax_main.set_ylabel(r'$\text{Recall}$')
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
        # Plot mean values
        mean_values = []
        in_degree_real_unique = np.unique(in_degree_real)
        for i in in_degree_real_unique:
            mean_values.append(precision_per_target[in_degree_real == i].mean())
        axs[axs_col].plot(
            in_degree_real_unique,
            mean_values,
            )
        # Scatter recall
        axs[axs_col].scatter(
            in_degree_real,
            recall_per_target,
            label='Recall')
        # Plot mean values
        mean_values = []
        in_degree_real_unique = np.unique(in_degree_real)
        for i in in_degree_real_unique:
            mean_values.append(recall_per_target[in_degree_real == i].mean())
        axs[axs_col].plot(
            in_degree_real_unique,
            mean_values,
            )
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
        # Plot mean values
        mean_values = []
        out_degree_real_unique = np.unique(out_degree_real)
        for i in out_degree_real_unique:
            mean_values.append(precision_per_target[out_degree_real == i].mean())
        axs[axs_col].plot(
            out_degree_real_unique,
            mean_values,
            )
        # Scatter
        axs[axs_col].scatter(
            out_degree_real,
            recall_per_target,
            label='Recall')
        # Plot mean values
        mean_values = []
        out_degree_real_unique = np.unique(out_degree_real)
        for i in out_degree_real_unique:
            mean_values.append(recall_per_target[out_degree_real == i].mean())
        axs[axs_col].plot(
            out_degree_real_unique,
            mean_values,
            )
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


def boxplot_BA_performance():
    # Boxplot of performance
    # # Legend: algorithms
    subplots_v = 2
    subplots_h = len(samples_n_range)
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'precision',
        'recall',
        ]]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i', 'samples_n'}
    
    # Precision
    axs_row = 0
    for axs_col, samples_n in enumerate(samples_n_range):
        # Set subplot title
        axs[axs_row, axs_col].set_title(r'$T={0}$'.format(samples_n))
        for (alg_i, algorithm) in enumerate(algorithms):
            # Select dataframe entries(i.e runs) with the same algorithm
            df_algorithm = df_interest.loc[
                df_interest['algorithm'] == algorithm].drop('algorithm', 1)
            # Ensure that only the desired parameters are aggregated or averaged
            df_keys_remaining = df_algorithm.columns
            check_remaining_dimensions(df_keys_remaining, parameters_to_average)
            precision = df_algorithm.loc[
                df_algorithm['samples_n'] == samples_n]['precision'].values.tolist()
            # Boxplots
            positions = np.array([1 + alg_i])
            boxes_width = 0.7
            axs[axs_row, axs_col].boxplot(
                precision,
                positions=positions,
                widths=boxes_width,
                #showmeans=True,
                showfliers=False,
                #notch=True,
                #patch_artist=True,
                #boxprops=dict(color=colors_default[alg_i]),
                medianprops=dict(color=colors_default[0])
                )
        # Set axes properties
        axs[axs_row, 0].set_ylabel(r'$\text{Precision}$')
        axs[axs_row, axs_col].yaxis.set_ticks_position('both')
        axs[axs_row, axs_col].set_xlim(left=0.5)
        #axs[axs_row, axs_col].set_ylim(bottom=0, top=1.1)
        #axs[axs_row, axs_col].axvline(x=3.5, color="black", linestyle="--")
    
    # Recall
    axs_row = 1
    for axs_col, samples_n in enumerate(samples_n_range):
        # Set subplot title
        axs[axs_row, axs_col].set_title(r'$T={0}$'.format(samples_n))
        for (alg_i, algorithm) in enumerate(algorithms):
            # Select dataframe entries(i.e runs) with the same algorithm
            df_algorithm = df_interest.loc[
                df_interest['algorithm'] == algorithm].drop('algorithm', 1)
            # Ensure that only the desired parameters are aggregated or averaged
            df_keys_remaining = df_algorithm.columns
            check_remaining_dimensions(df_keys_remaining, parameters_to_average)
            recall = df_algorithm.loc[
                df_algorithm['samples_n'] == samples_n]['recall'].values.tolist()
            # Boxplots
            positions = np.array([1 + alg_i])
            axs[axs_row, axs_col].boxplot(
                recall,
                positions=positions,
                widths=boxes_width,
                #showmeans=True,
                showfliers=False,
                #patch_artist=True,
                #boxprops=dict(facecolor=colors_default[alg_i]),
                medianprops=dict(color=colors_default[1])
                )
            # labels.append((Patch(color=colors_default[alg_i]), algorithm_names[algorithm]))
        # Set axes properties
        axs[axs_row, 0].set_ylabel(r'$\text{Recall}$')
        axs[axs_row, axs_col].yaxis.set_ticks_position('both')
        axs[axs_row, axs_col].set_xticks(list(range(1, 1 + len(algorithms))))
        axs[axs_row, axs_col].set_xticklabels([algorithm_names[algorithm] for algorithm in algorithms])
        axs[axs_row, axs_col].set_xlim(left=0.5)
        #axs[axs_row, axs_col].set_ylim(bottom=0, top=1.1)
        #axs[axs_row, axs_col].axvline(x=3.5, color="black", linestyle="--")
    return fig


def scatter_FP_per_target_vs_in_degree_real():
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
        
        FP = in_degree_inferred * (1 - precision_per_target)
        
        curve_i = 0

        # Scatter FP vs in_degree
        axs[axs_col].scatter(
            in_degree_real,
            FP
            )
        # Plot mean values
        mean_values = []
        in_degree_real_unique = np.unique(in_degree_real)
        for i in in_degree_real_unique:
            mean_values.append(FP[in_degree_real == i].mean())
        axs[axs_col].plot(
            in_degree_real_unique,
            mean_values,
            label='Mean',
            )
        # Set axes properties
        axs[axs_col].set_xlabel(
            r'$\text{Real in-degree of target}$', horizontalalignment='right', x=1.0)
        axs[axs_col].set_ylabel(
            r'$\text{FP}$')
        # axs[axs_col].set_xscale('log')
        axs[axs_col].yaxis.set_ticks_position('both')
        #axs[axs_col].set_xlim(left=0)
        axs[axs_col].legend()
    return fig


def scatter_FP_per_source_vs_in_degree_real():
    # Scatter plot FP vs real in-degree
    # Subplots vertical: algorithms
    subplots_v = len(algorithms)
    subplots_h = 1
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'in_degree_real',
        'in_degree_inferred',
        'precision_per_source',
        'recall_per_source',
        ]]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
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
        precision_per_source = np.concatenate(
            df_aggregate.loc[algorithm].precision_per_source)
        
        precision_per_source[np.isnan(precision_per_source)] = 1.

        #recall_per_source = np.concatenate(
        #    df_aggregate.loc[algorithm].recall_per_source)
        
        FP = in_degree_inferred * (1 - precision_per_source)
        
        # Scatter FP vs in_degree
        axs[axs_col].scatter(
            in_degree_real,
            FP,
            label='FP')
        # Plot mean values
        mean_values = []
        in_degree_real_unique = np.unique(in_degree_real)
        for i in in_degree_real_unique:
            mean_values.append(FP[in_degree_real == i].mean())
        axs[axs_col].plot(
            in_degree_real_unique,
            mean_values,
            label='Mean',
            )
        # Set axes properties
        axs[axs_col].set_xlabel(
            r'$\text{Real in-degree of source}$', horizontalalignment='right', x=1.0)
        axs[axs_col].set_ylabel(
            r'$\text{FP}$')
        # axs[axs_col].set_xscale('log')
        axs[axs_col].yaxis.set_ticks_position('both')
        #axs[axs_col].set_xlim(left=0)
        #axs[axs_col].legend()
    return fig


def boxplot_BA_average_global_efficiency():
    # Boxplot of average_global_efficiency
    # # Legend: algorithms
    subplots_v = 1
    subplots_h = 1
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'average_global_efficiency_real',
        'average_global_efficiency_inferred',
        ]]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}
    # Labels for legend
    labels = []
    axs_row = 0
    for (alg_i, algorithm) in enumerate(algorithms):
        # Select dataframe entries(i.e runs) with the same algorithm
        df_algorithm = df_interest.loc[
            df_interest['algorithm'] == algorithm].drop('algorithm', 1)
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_algorithm.columns
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        average_global_efficiency_inferred = df_algorithm.average_global_efficiency_inferred.values.tolist()
        # Boxplots
        positions = [1 + alg_i]
        boxes_width = 0.7
        axs[axs_row].boxplot(
            average_global_efficiency_inferred,
            positions=positions,
            widths=boxes_width,
            showfliers=False,
            #notch=True,
            #patch_artist=True,
            #boxprops=dict(color=colors_default[alg_i + 1]),
            medianprops=dict(color=colors_default[alg_i + 1])
            )
        # labels.append((Patch(color=colors_default[alg_i]), algorithm_names[algorithm]))
    average_global_efficiency_real = df_interest.average_global_efficiency_real.values.tolist()
    axs[axs_row].boxplot(
        average_global_efficiency_real,
        positions=[1 + len(algorithms)],
        widths=boxes_width,
        showfliers=False,
        #notch=True,
        #patch_artist=True,
        #boxprops=dict(color=colors_default[alg_i]),
        medianprops=dict(color='k')
        )
    # labels.append((Patch(color='k'), 'Real'))
    axs[axs_row].set_ylabel(r'$\text{Global efficiency}$')
    axs[axs_row].yaxis.set_ticks_position('both')
    axs[axs_row].set_xticks(list(range(1, 2 + len(algorithms))))
    axs[axs_row].set_xticklabels([algorithm_names[algorithm] for algorithm in algorithms] + ['Real'])
    axs[axs_row].set_xlim(left=0.5, right=1.5+len(algorithms))
    axs[axs_row].set_ylim(bottom=0)
    # axs[axs_row].legend(*zip(*labels), loc='upper right')
    return fig


def boxplot_BA_local_efficiency():
    # Boxplot of local_efficiency
    # # Legend: algorithms
    subplots_v = 1
    subplots_h = 1
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'local_efficiency_real',
        'local_efficiency_inferred',
        ]]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}
    # Labels for legend
    labels = []
    axs_row = 0
    for (alg_i, algorithm) in enumerate(algorithms):
        # Select dataframe entries(i.e runs) with the same algorithm
        df_algorithm = df_interest.loc[
            df_interest['algorithm'] == algorithm].drop('algorithm', 1)
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_algorithm.columns
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)

        df_algorithm.local_efficiency_inferred = df_algorithm.local_efficiency_inferred.apply(np.nanmean)
        local_efficiency_inferred = df_algorithm.local_efficiency_inferred.values.tolist()
        # Boxplots
        positions = [1 + alg_i]
        boxes_width = 0.7
        axs[axs_row].boxplot(
            local_efficiency_inferred,
            positions=positions,
            widths=boxes_width,
            showfliers=False,
            #notch=True,
            #patch_artist=True,
            #boxprops=dict(color=colors_default[alg_i = 1]),
            medianprops=dict(color=colors_default[alg_i + 1])
            )
        # labels.append((Patch(color=colors_default[alg_i]), algorithm_names[algorithm]))
    df_real = pd.DataFrame()
    df_real.local_efficiency_real = df_interest.local_efficiency_real.apply(np.nanmean)
    local_efficiency_real = df_real.local_efficiency_real.values.tolist()
    axs[axs_row].boxplot(
        local_efficiency_real,
        positions=[1 + len(algorithms)],
        widths=boxes_width,
        showfliers=False,
        #notch=True,
        #patch_artist=True,
        #boxprops=dict(color=colors_default[alg_i]),
        medianprops=dict(color='k')
        )
    # labels.append((Patch(color='k'), 'Real'))
    axs[axs_row].set_ylabel(r'$\text{Local efficiency}$')
    axs[axs_row].yaxis.set_ticks_position('both')
    axs[axs_row].set_xticks(list(range(1, 2 + len(algorithms))))
    axs[axs_row].set_xticklabels([algorithm_names[algorithm] for algorithm in algorithms] + ['Real'])
    axs[axs_row].set_xlim(left=0.5, right=1.5+len(algorithms))
    axs[axs_row].set_ylim(bottom=0)
    # axs[axs_row].legend(*zip(*labels), loc='upper right')
    return fig


def scatter_clustering_inferred_vs_real():
    # Scatter clustering inferred vs. real
    # Subplots vertical: algorithms
    subplots_v = 1
    subplots_h = 1
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'clustering_inferred',
        'clustering_real',
        ]]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}
    for (alg_i, algorithm) in enumerate(algorithms):
        # Group by algorithm and concatenate values into lists
        df_aggregate = df_interest.groupby('algorithm').agg(
            lambda x: x.tolist())
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        clustering_inferred = np.concatenate(
            df_aggregate.loc[algorithm].clustering_inferred)
        clustering_real = np.concatenate(
            df_aggregate.loc[algorithm].clustering_real)
        # Scatter
        axs[0].scatter(
            clustering_real,
            clustering_inferred,
            color=colors_default[alg_i + 1],
            marker=markers_default[alg_i + 1],
            label=algorithm_names[algorithm],
            )
        # Plot identity line
        axs[0].plot(
            [min(clustering_real), max(clustering_real)],
            [min(clustering_real), max(clustering_real)],
            color='k',
            linestyle='--',
            )
    # Set axes properties
    axs[0].set_xlabel(
        r'$\text{Real clustering}$', horizontalalignment='right', x=1.0)
    axs[0].set_ylabel(
        r'$\text{Inferred clustering}$')
    axs[0].yaxis.set_ticks_position('both')
    axs[0].legend(loc='lower right')
    return fig


def scatter_clustering_vs_in_degree():
    # Subplots vertical: algorithms
    subplots_v = 1
    subplots_h = 1
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'clustering_inferred',
        'clustering_real',
        'in_degree_real',
        ]]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}
    for (alg_i, algorithm) in enumerate(algorithms):
        # Group by algorithm and concatenate values into lists
        df_aggregate = df_interest.groupby('algorithm').agg(
            lambda x: x.tolist())
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        clustering_inferred = np.concatenate(
            df_aggregate.loc[algorithm].clustering_inferred)
        clustering_real = np.concatenate(
            df_aggregate.loc[algorithm].clustering_real)
        in_degree_real = np.concatenate(
            df_aggregate.loc[algorithm].in_degree_real)
        # Scatter
        axs[0].scatter(
            in_degree_real,
            clustering_inferred,
            color=colors_default[alg_i + 1],
            marker=markers_default[alg_i + 1],
            #s=10,
            label=algorithm_names[algorithm],
            alpha=0.8
            )
        axs[0].scatter(
            in_degree_real,
            clustering_real,
            color='k',
            marker='x',
            #s=10,
            #label='Real',
            alpha=0.8
            )
    axs[0].scatter(
        in_degree_real[:1],
        clustering_real[:1],
        color='k',
        marker='x',
        #s=10,
        label='Real',
        )
    # Set axes properties
    axs[0].set_xlabel(
        r'$\text{Real in-degree}$', horizontalalignment='right', x=1.0)
    axs[0].set_ylabel(
        r'$\text{Clustering}$')
    axs[0].yaxis.set_ticks_position('both')
    axs[0].legend(loc='upper right')
    return fig


def boxplot_BA_clustering():
    # Boxplot of clustering
    # # Legend: algorithms
    subplots_v = 1
    subplots_h = 1
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'clustering_real',
        'clustering_inferred',
        ]]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}
    # Labels for legend
    labels = []
    axs_row = 0
    for (alg_i, algorithm) in enumerate(algorithms):
        # Select dataframe entries(i.e runs) with the same algorithm
        df_algorithm = df_interest.loc[
            df_interest['algorithm'] == algorithm].drop('algorithm', 1)
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_algorithm.columns
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)

        df_algorithm.clustering_inferred = df_algorithm.clustering_inferred.apply(np.nanmean)
        clustering_inferred = df_algorithm.clustering_inferred.values.tolist()
        # Boxplots
        positions = [1 + alg_i]
        boxes_width = 0.7
        axs[axs_row].boxplot(
            clustering_inferred,
            positions=positions,
            widths=boxes_width,
            showfliers=False,
            #notch=True,
            #patch_artist=True,
            #boxprops=dict(color=colors_default[alg_i + 1]),
            medianprops=dict(color=colors_default[alg_i + 1])
            )
        #labels.append((Patch(color=colors_default[alg_i]), algorithm_names[algorithm]))
    df_real = pd.DataFrame()
    df_real.clustering_real = df_interest.clustering_real.apply(np.nanmean)
    clustering_real = df_real.clustering_real.values.tolist()
    axs[axs_row].boxplot(
        clustering_real,
        positions=[1 + len(algorithms)],
        widths=boxes_width,
        showfliers=False,
        #notch=True,
        #patch_artist=True,
        #boxprops=dict(color=colors_default[alg_i]),
        medianprops=dict(color='k')
        )
    #labels.append((Patch(color='k'), 'Real'))
    axs[axs_row].set_ylabel(r'$\text{Clustering coefficient}$')
    axs[axs_row].yaxis.set_ticks_position('both')
    axs[axs_row].set_xticks(list(range(1, 2 + len(algorithms))))
    axs[axs_row].set_xticklabels([algorithm_names[algorithm] for algorithm in algorithms] + ['Real'])
    axs[axs_row].set_xlim(left=0.5, right=1.5+len(algorithms))
    axs[axs_row].set_ylim(bottom=0)
    # axs[axs_row].legend(*zip(*labels), loc='upper right')
    return fig


def scatter_betweenness_centrality_inferred_vs_real():
    # Scatter betweenness_centrality inferred vs. real
    # Subplots vertical: algorithms
    subplots_v = 1
    subplots_h = 1
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'betweenness_centrality_inferred',
        'betweenness_centrality_real',
        ]]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}
    for (alg_i, algorithm) in enumerate(algorithms):
        # Group by algorithm and concatenate values into lists
        df_aggregate = df_interest.groupby('algorithm').agg(
            lambda x: x.tolist())
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        betweenness_centrality_inferred = np.concatenate(
            df_aggregate.loc[algorithm].betweenness_centrality_inferred)
        betweenness_centrality_real = np.concatenate(
            df_aggregate.loc[algorithm].betweenness_centrality_real)
        # Scatter
        axs[0].scatter(
            betweenness_centrality_real,
            betweenness_centrality_inferred,
            color=colors_default[alg_i + 1],
            marker=markers_default[alg_i + 1],
            label=algorithm_names[algorithm],
            )
        # Plot identity line
        axs[0].plot(
            [min(betweenness_centrality_real), max(betweenness_centrality_real)],
            [min(betweenness_centrality_real), max(betweenness_centrality_real)],
            color='k',
            linestyle='--',
            )
    # Set axes properties
    axs[0].set_xlabel(
        r'$\text{Real centrality}$', horizontalalignment='right', x=1.0)
    axs[0].set_ylabel(
        r'$\text{Inferred centrality}$')
    axs[0].yaxis.set_ticks_position('both')
    axs[0].legend(loc='upper left')
    return fig


def scatter_eigenvector_centrality_inferred_vs_real():
    # Scatter eigenvector_centrality inferred vs. real
    # Subplots vertical: algorithms
    subplots_v = len(algorithms)
    subplots_h = 1
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'eigenvector_centrality_inferred',
        'eigenvector_centrality_real',
        ]]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
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
        eigenvector_centrality_inferred = np.concatenate(
            df_aggregate.loc[algorithm].eigenvector_centrality_inferred)
        eigenvector_centrality_real = np.concatenate(
            df_aggregate.loc[algorithm].eigenvector_centrality_real)
        # Scatter
        axs[axs_col].scatter(
            eigenvector_centrality_real,
            eigenvector_centrality_inferred,
            color=colors_default[axs_col + 1],
            #marker=markers_default[axs_col + 1],
            )
        # Plot identity line
        axs[axs_col].plot(
            [min(eigenvector_centrality_real), max(eigenvector_centrality_real)],
            [min(eigenvector_centrality_real), max(eigenvector_centrality_real)],
            color='k',
            linestyle='--',
            )
        # Compute mean squared error and Person corr
        mse = np.square(np.subtract(eigenvector_centrality_inferred, eigenvector_centrality_real)).mean()
        rho=np.corrcoef(np.array([eigenvector_centrality_real,eigenvector_centrality_inferred]))[0,1]
        # Set subplot title
        # axs[axs_col].set_title('{0}, MSE= {1:.3f}'.format(algorithm_names[algorithm], mse))
        axs[axs_col].set_title('{0}'.format(algorithm_names[algorithm], mse))
        # Set axes properties
        axs[axs_col].set_xlabel(
            r'$\text{Real centrality}$', horizontalalignment='right', x=1.0)
        axs[axs_col].set_ylabel(
            r'$\text{Inferred centrality}$')
        # axs[axs_row, axs_col].set_xscale('log')
        axs[axs_col].yaxis.set_ticks_position('both')
        #axs[axs_row, axs_col].set_xlim(left=0)
        #axs[axs_row, axs_col].set_ylim(bottom=0)
    return fig


def scatter_in_degree_inferred_vs_real():
    # Scatter in-degree inferred vs. real
    # Subplots vertical: algorithms
    subplots_v = 1
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
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}
    for (alg_i, algorithm) in enumerate(algorithms):
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
        axs[0].scatter(
            in_degree_real,
            in_degree_inferred,
            color=colors_default[alg_i + 1],
            marker=markers_default[alg_i + 1],
            label=algorithm_names[algorithm],
            )
        # Plot identity line
        axs[0].plot(
            [min(in_degree_real), max(in_degree_real)],
            [min(in_degree_real), max(in_degree_real)],
            color='k',
            linestyle='--',
            )
    # Set axes properties
    axs[0].set_xlabel(
        r'$\text{Real in-degree}$', horizontalalignment='right', x=1.0)
    axs[0].set_ylabel(
        r'$\text{Inferred in-degree}$')
    axs[0].yaxis.set_ticks_position('both')
    axs[0].legend(loc='upper right')
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


def loglog_distributions_in_degree_inferred_vs_real():
    # Plot in_degree_real and in_degree_inferred distributions (only T_interest)
    # Subplots vertical: algorithms
    subplots_v = 1
    subplots_h = 1
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'in_degree_real',
        'in_degree_inferred',
    ]]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}
    for (alg_i, algorithm) in enumerate(algorithms):
        # Group by algorithm and concatenate values into lists
        df_aggregate = df_interest.groupby('algorithm').agg(
            lambda x: x.tolist())
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        in_degree_inferred = np.concatenate(
            df_aggregate.loc[algorithm].in_degree_inferred)
        
        def power_law(x, coeff, exponent):
            return coeff * x ** exponent

        
        xmin=4

        # Inferred
        results_inferred = powerlaw.Fit(in_degree_inferred , xmin=xmin)
        R, p = results_inferred.distribution_compare('power_law', 'lognormal')
        print(R, p)

        results_inferred.plot_pdf(
            color=colors_default[alg_i + 1],
            marker=markers_default[alg_i + 1],
            linestyle='',
            label=algorithm_names[algorithm] + r' ($\beta$={0:.1f})'.format(results_inferred.power_law.alpha),
            ax=axs[0])
        results_inferred.power_law.plot_pdf(
            color=colors_default[alg_i + 1],
            linestyle='--',
            ax=axs[0]
            )

    # Real
    in_degree_real = np.concatenate(df_interest.in_degree_real)
    results_real = powerlaw.Fit(in_degree_real , xmin=xmin)
    R, p = results_real.distribution_compare('power_law', 'lognormal')
    print(R, p)

    results_real.plot_pdf(
        color='k',
        marker='x',
        linestyle='',
        label=r'Real ($\beta$={0:.1f})'.format(results_real.power_law.alpha),
        ax=axs[0])
    results_real.power_law.plot_pdf(
        color='k',
        linestyle='--',
        ax=axs[0]
        )
        
    # Theoretical
    # x = np.linspace(xmin, max(in_degree_inferred))
    # y = power_law(x, 32, -3)
    # axs[0].plot(x, y, linestyle='--', color='k', label='Theoretical')

    # Set axes properties
    axs[0].set_xlabel(r'$\text{In-degree}$', horizontalalignment='right', x=1.0)
    axs[0].set_ylabel(r'$\text{Frequency}$')
    axs[0].yaxis.set_ticks_position('both')
    #axs[0].set_ylim(bottom=0)
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    # Add legend
    axs[0].legend()

    return fig


def scatter_in_degree_assortativity_inferred_vs_real():
    # Scatter in_degree_assortativity_real vs. in_degree_assortativity_inferred
    # Subplots vertical: algorithms
    subplots_v = 1
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
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}
    for (alg_i, algorithm) in enumerate(algorithms):
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
        axs[0].scatter(
            in_degree_assortativity_real,
            in_degree_assortativity_inferred,
            color=colors_default[alg_i + 1],
            marker=markers_default[alg_i + 1],
            label=algorithm_names[algorithm],
            )
    # Plot identity line
    axs[0].plot(
        [min(df_interest.in_degree_assortativity_real), max(df_interest.in_degree_assortativity_real)],
        [min(df_interest.in_degree_assortativity_real), max(df_interest.in_degree_assortativity_real)],
        color='k',
        linestyle='--',
        )
    # Set axes properties
    axs[0].set_xlabel(
        r'$\text{Real in-degree assortativity}$', horizontalalignment='right', x=1.0)
    axs[0].set_ylabel(
        r'$\text{Inferred assortativity}$')
    axs[0].yaxis.set_ticks_position('both')
    axs[0].legend(loc='lower right')
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


def plot_rich_club_in_degrees_inferred_vs_real(max_degree=25):
    # Plot rich_club_in_degrees_inferred vs rich_club_in_degrees_real
    # Subplots vertical: algorithms
    subplots_v = 1
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

        # Inferred
        axs[0].plot(
            range(max_degree),
            rich_club_in_degrees_inferred[:max_degree],
            color=colors_default[axs_col + 1],
            marker=markers_default[axs_col + 1],
            label=algorithm_names[algorithm])
    # Real
    rich_club_in_degrees_real = np.nanmean(
        np.row_stack(df_interest.rich_club_in_degrees_real),
        axis=0)
    axs[0].plot(
        range(max_degree),
        rich_club_in_degrees_real[:max_degree],
        color='k',
        marker='x',
        linestyle='',
        label='Real')
    # Set subplot title
    #axs[0].set_title('{0}'.format(algorithm_names[algorithm]))
    # Set axes properties
    axs[0].set_xlabel(r'$\text{In-degree}$', horizontalalignment='right', x=1.0)
    axs[0].set_ylabel(r'$\text{Rich-club coefficient}$')
    axs[0].yaxis.set_ticks_position('both')
    #axs[0].set_xlim(right=max_degree+1)
    # Add legend
    axs[0].legend()

    return fig


# -------------------------------------------------------------------------
# endregion
# -------------------------------------------------------------------------



# -------------------------------------------------------------------------
# region Stochastic Block Model
# -------------------------------------------------------------------------

def plot_SBM_precision_vs_links_out():
    # Precision within and between groups vs. links_out
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
        df_aggregate = df_algorithm.groupby('links_out').agg(
            lambda x: x.tolist())
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        precision_within_groups = [
            df_aggregate.loc[links_out].precision_within_groups
            for links_out in links_out_range]
        precision_between_groups = [
            df_aggregate.loc[links_out].precision_between_groups
            for links_out in links_out_range]
        mean_vals_within = [np.nanmean(df_aggregate.loc[links_out].precision_within_groups) for links_out in links_out_range]
        mean_vals_between = [np.nanmean(df_aggregate.loc[links_out].precision_between_groups) for links_out in links_out_range]
        # mean vals
        axs[axs_row].plot(
            links_out_range,
            mean_vals_within,
            color='tab:blue',
            label='Within groups')
        axs[axs_row].plot(
            links_out_range,
            mean_vals_between,
            color='tab:orange',
            label='Between groups')
        # scatter
        axs[axs_row].scatter(
            df_algorithm['links_out'].values.astype(float).tolist(),
            df_algorithm['precision_within_groups'].values,
            color='tab:blue',
            alpha=0.3)
        axs[axs_row].scatter(
            df_algorithm['links_out'].values.astype(float).tolist(),
            df_algorithm['precision_between_groups'].values,
            color='tab:orange',
            alpha=0.3)
        # # Violin plot of omnibus TE vs links_out
        # violin_within = axs[axs_row].violinplot(
        #     precision_within_groups,
        #     positions=links_out_range,
        #     widths=0.4,#WS_p_range/2,
        #     showmeans=False,
        #     showextrema=False,
        #     showmedians=False,
        #     #points=100,
        #     #bw_method=
        #     )
        # violin_between = axs[axs_row].violinplot(
        #     precision_between_groups,
        #     positions=links_out_range,
        #     widths=0.4,#WS_p_range/2,
        #     showmeans=False,
        #     showextrema=False,
        #     showmedians=False,
        #     #points=100,
        #     #bw_method=
        #     )
        # # Join mean values
        # mean_vals_within = [np.nanmean(df_aggregate.loc[links_out].precision_within_groups) for links_out in links_out_range]
        # axs[axs_row].plot(
        #     links_out_range,
        #     mean_vals_within,
        #     '-o',
        #     color='tab:blue',
        #     label='Within groups')
        # mean_vals_between = [np.nanmean(df_aggregate.loc[links_out].precision_between_groups) for links_out in links_out_range]
        # axs[axs_row].plot(
        #     links_out_range,
        #     mean_vals_between,
        #     '-o',
        #     color='tab:orange',
        #     label='Between groups')
        # Set axes properties
        axs[axs_row].set_xlabel(
            r'$\text{Links between groups}$', horizontalalignment='right', x=1.0)
        axs[axs_row].set_ylabel(
            r'$\text{Precision}$')
        axs[axs_row].yaxis.set_ticks_position('both')
        axs[axs_row].set_ylim(bottom=0, top=1.1)
        axs[axs_row].legend()
    return fig


def plot_SBM_recall_vs_links_out():
    # Recall within and between groups vs. links_out
    # Subplots vertical: algorithms
    subplots_v = len(algorithms)
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
        df_aggregate = df_algorithm.groupby('links_out').agg(
            lambda x: x.tolist())
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        recall_within_groups = [
            df_aggregate.loc[links_out].recall_within_groups
            for links_out in links_out_range]
        recall_between_groups = [
            df_aggregate.loc[links_out].recall_between_groups
            for links_out in links_out_range]
        mean_vals_within = [np.nanmean(df_aggregate.loc[links_out].recall_within_groups) for links_out in links_out_range]
        mean_vals_between = [np.nanmean(df_aggregate.loc[links_out].recall_between_groups) for links_out in links_out_range]
        # mean vals
        axs[axs_row].plot(
            links_out_range,
            mean_vals_within,
            color='tab:blue',
            label='Within groups')
        axs[axs_row].plot(
            links_out_range,
            mean_vals_between,
            color='tab:orange',
            label='Between groups')
        # scatter
        axs[axs_row].scatter(
            df_algorithm['links_out'].values.astype(float).tolist(),
            df_algorithm['recall_within_groups'].values,
            color='tab:blue',
            alpha=0.3)
        axs[axs_row].scatter(
            df_algorithm['links_out'].values.astype(float).tolist(),
            df_algorithm['recall_between_groups'].values,
            color='tab:orange',
            alpha=0.3)
        # # Violin plot of omnibus TE vs links_out
        # violin_within = axs[axs_row].violinplot(
        #     recall_within_groups,
        #     positions=links_out_range,
        #     widths=0.4,#WS_p_range/2,
        #     showmeans=False,
        #     showextrema=False,
        #     showmedians=False,
        #     #points=100,
        #     #bw_method=
        #     )
        # violin_between = axs[axs_row].violinplot(
        #     recall_between_groups,
        #     positions=links_out_range,
        #     widths=0.4,#WS_p_range/2,
        #     showmeans=False,
        #     showextrema=False,
        #     showmedians=False,
        #     #points=100,
        #     #bw_method=
        #     )
        # # Join mean values
        # mean_vals_within = [np.nanmean(df_aggregate.loc[links_out].recall_within_groups) for links_out in links_out_range]
        # axs[axs_row].plot(
        #     links_out_range,
        #     mean_vals_within,
        #     '-o',
        #     color='tab:blue',
        #     label='Within groups')
        # mean_vals_between = [np.nanmean(df_aggregate.loc[links_out].recall_between_groups) for links_out in links_out_range]
        # axs[axs_row].plot(
        #     links_out_range,
        #     mean_vals_between,
        #     '-o',
        #     color='tab:orange',
        #     label='Between groups')
        # Set axes properties
        axs[axs_row].set_xlabel(
            r'$\text{Links between groups}$', horizontalalignment='right', x=1.0)
        axs[axs_row].set_ylabel(
            r'$\text{Recall}$')
        axs[axs_row].yaxis.set_ticks_position('both')
        axs[axs_row].set_ylim(bottom=0, top=1.1)
        axs[axs_row].legend()
    return fig


def plot_SBM_FP_vs_links_out():
    # FP vs. links_out
    # Subplots vertical: algorithms
    subplots_v = len(algorithms)
    subplots_h = 1
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'FP_within_groups',
        'FP_between_groups',
        ]]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
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
        df_aggregate = df_algorithm.groupby('links_out').agg(
            lambda x: x.tolist())
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        FP_within_groups = [
            df_aggregate.loc[links_out].FP_within_groups
            for links_out in links_out_range]
        FP_between_groups = [
            df_aggregate.loc[links_out].FP_between_groups
            for links_out in links_out_range]
        mean_vals_real = [np.nanmean(df_aggregate.loc[links_out].FP_within_groups) for links_out in links_out_range]
        mean_vals_inferred = [np.nanmean(df_aggregate.loc[links_out].FP_between_groups) for links_out in links_out_range]
        # mean vals
        axs[axs_row].plot(
            links_out_range,
            mean_vals_real,
            color='tab:blue',
            label='Within groups')
        axs[axs_row].plot(
            links_out_range,
            mean_vals_inferred,
            color='tab:orange',
            label='Between groups')
        # scatter
        axs[axs_row].scatter(
            df_algorithm['links_out'].values.astype(float).tolist(),
            df_algorithm['FP_within_groups'].values,
            color='tab:blue',
            alpha=0.3)
        axs[axs_row].scatter(
            df_algorithm['links_out'].values.astype(float).tolist(),
            df_algorithm['FP_between_groups'].values,
            color='tab:orange',
            alpha=0.3)
        # Set axes properties
        axs[axs_row].set_xlabel(
            r'$\text{Links between groups}$', horizontalalignment='right', x=1.0)
        axs[axs_row].set_ylabel(
            r'$\text{False positives}$')
        axs[axs_row].yaxis.set_ticks_position('both')
        axs[axs_row].set_ylim(bottom=0)#, top=1.1)
        axs[axs_row].legend()
    return fig


def plot_SBM_FP_rate_vs_links_out():
    # FP vs. links_out
    # Subplots vertical: algorithms
    subplots_v = len(algorithms)
    subplots_h = 1
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'FP_rate_within_groups',
        'FP_rate_between_groups',
        ]]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
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
        df_aggregate = df_algorithm.groupby('links_out').agg(
            lambda x: x.tolist())
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        FP_rate_within_groups = [
            df_aggregate.loc[links_out].FP_rate_within_groups
            for links_out in links_out_range]
        FP_rate_between_groups = [
            df_aggregate.loc[links_out].FP_rate_between_groups
            for links_out in links_out_range]
        mean_vals_real = [np.nanmean(df_aggregate.loc[links_out].FP_rate_within_groups) for links_out in links_out_range]
        mean_vals_inferred = [np.nanmean(df_aggregate.loc[links_out].FP_rate_between_groups) for links_out in links_out_range]
        # mean vals
        axs[axs_row].plot(
            links_out_range,
            mean_vals_real,
            color='tab:blue',
            label='Within groups')
        axs[axs_row].plot(
            links_out_range,
            mean_vals_inferred,
            color='tab:orange',
            label='Between groups')
        # scatter
        axs[axs_row].scatter(
            df_algorithm['links_out'].values.astype(float).tolist(),
            df_algorithm['FP_rate_within_groups'].values,
            color='tab:blue',
            alpha=0.3)
        axs[axs_row].scatter(
            df_algorithm['links_out'].values.astype(float).tolist(),
            df_algorithm['FP_rate_between_groups'].values,
            color='tab:orange',
            alpha=0.3)
        # Set axes properties
        axs[axs_row].set_xlabel(
            r'$\text{Links between groups}$', horizontalalignment='right', x=1.0)
        axs[axs_row].set_ylabel(
            r'$\text{False-positive rate}$')
        axs[axs_row].yaxis.set_ticks_position('both')
        axs[axs_row].set_ylim(bottom=0)#, top=1.1)
        axs[axs_row].legend()
    return fig


def plot_SBM_FP_and_FP_rate_vs_links_out(nodes_n):
    # FP vs. links_out (left column) and FP_rate vs links out (right column)
    # Subplots vertical: algorithms
    subplots_v = len(algorithms)
    subplots_h = 2
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey='col')
    # FP
    axs_col = 0
    # Select data of interest
    df_interest = df[parameters_explored + [
        'FP_within_groups',
        'FP_between_groups',
        ]]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}
    for (axs_row, algorithm) in enumerate(algorithms):
        # Set subplot title
        axs[axs_row, axs_col].set_title('{0}'.format(
            algorithm_names[algorithm]))
        # Select dataframe entries(i.e runs) with the same algorithm
        df_algorithm = df_interest.loc[
            df_interest['algorithm'] == algorithm].drop('algorithm', 1)
        # Group and concatenate lists
        df_aggregate = df_algorithm.groupby('links_out').agg(
            lambda x: x.tolist())
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        mean_vals_real = np.array([np.nanmean(df_aggregate.loc[links_out].FP_within_groups) for links_out in links_out_range]) / nodes_n
        mean_vals_inferred = np.array([np.nanmean(df_aggregate.loc[links_out].FP_between_groups) for links_out in links_out_range]) / nodes_n
        # mean vals
        axs[axs_row, axs_col].plot(
            links_out_range,
            mean_vals_real,
            color='tab:blue',
            label='Within groups')
        axs[axs_row, axs_col].plot(
            links_out_range,
            mean_vals_inferred,
            color='tab:orange',
            label='Between groups')
        # scatter
        axs[axs_row, axs_col].scatter(
            df_algorithm['links_out'].values.astype(int).tolist(),
            df_algorithm['FP_within_groups'].values / nodes_n,
            color='tab:blue',
            alpha=0.3)
        axs[axs_row, axs_col].scatter(
            df_algorithm['links_out'].values.astype(int).tolist(),
            df_algorithm['FP_between_groups'].values / nodes_n,
            color='tab:orange',
            alpha=0.3)
        # Set axes properties
        axs[axs_row, axs_col].set_ylabel(
            r'$\text{False positives (per node)}$')
        axs[axs_row, axs_col].yaxis.set_ticks_position('both')
        axs[axs_row, axs_col].set_ylim(bottom=0)#, top=1.1)
        axs[axs_row, axs_col].legend(loc='upper left')
    axs[axs_row, axs_col].set_xlabel(
        r'$\text{Links between groups (per node)}$')
    # FP rate
    axs_col = 1
    df_interest = df[parameters_explored + [
        'FP_rate_within_groups',
        'FP_rate_between_groups',
        ]]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}
    for (axs_row, algorithm) in enumerate(algorithms):
        # Set subplot title
        axs[axs_row, axs_col].set_title('{0}'.format(
            algorithm_names[algorithm]))
        # Select dataframe entries(i.e runs) with the same algorithm
        df_algorithm = df_interest.loc[
            df_interest['algorithm'] == algorithm].drop('algorithm', 1)
        # Group and concatenate lists
        df_aggregate = df_algorithm.groupby('links_out').agg(
            lambda x: x.tolist())
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        mean_vals_real = [np.nanmean(df_aggregate.loc[links_out].FP_rate_within_groups) for links_out in links_out_range]
        mean_vals_inferred = [np.nanmean(df_aggregate.loc[links_out].FP_rate_between_groups) for links_out in links_out_range]
        # mean vals
        axs[axs_row, axs_col].plot(
            links_out_range,
            mean_vals_real,
            color='tab:blue',
            label='Within groups')
        axs[axs_row, axs_col].plot(
            links_out_range,
            mean_vals_inferred,
            color='tab:orange',
            label='Between groups')
        # scatter
        axs[axs_row, axs_col].scatter(
            df_algorithm['links_out'].values.astype(int).tolist(),
            df_algorithm['FP_rate_within_groups'].values,
            color='tab:blue',
            alpha=0.3)
        axs[axs_row, axs_col].scatter(
            df_algorithm['links_out'].values.astype(int).tolist(),
            df_algorithm['FP_rate_between_groups'].values,
            color='tab:orange',
            alpha=0.3)
        # Set axes properties
        axs[axs_row, axs_col].set_ylabel(
            r'$\text{False positive rate}$')
        axs[axs_row, axs_col].yaxis.set_ticks_position('both')
        axs[axs_row, axs_col].set_ylim(bottom=0)#, top=1.1)
        axs[axs_row, axs_col].legend(loc='upper right')
    axs[axs_row, axs_col].set_xlabel(
        r'$\text{Links between groups (per node)}$')
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


def plot_SBM_partition_modularity_vs_links_out():
    # Partition perfomance vs. links_out
    subplots_v = len(samples_n_range)
    subplots_h = 1
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'block_partition_modularity_real',
        'block_partition_modularity_inferred',
        ]]
    if 'algorithm' not in parameters_explored:
        df_interest['algorithm'] = df['algorithm']
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}
    for axs_row, samples_n in enumerate(samples_n_range):
        # Set subplot title
        axs[axs_row].set_title(r'$T={0}$'.format(samples_n))
        df_samples_n = df_interest.loc[
            df_interest['samples_n'] == samples_n].drop('samples_n', 1)
        # Plot real values
        df_algorithm = df_samples_n.loc[df_samples_n['algorithm'] == algorithms[-1]]
        df_aggregate = df_algorithm.groupby('links_out').agg(lambda x: list(x))
        block_partition_modularity_real = [
            df_aggregate.loc[links_out].block_partition_modularity_real
            for links_out in links_out_range]
        mean_vals_real = [np.nanmean(df_aggregate.loc[links_out].block_partition_modularity_real) for links_out in links_out_range]
        axs[axs_row].plot(
            links_out_range,
            mean_vals_real,
            label='Real',
            linestyle = 'None',
            color='k',
            marker='x',
            fillstyle='none',
            zorder=99,  # bring to foreground
            )
        for (alg_i, algorithm) in enumerate(algorithms):
            # Select dataframe entries(i.e runs) with the same algorithm
            df_algorithm = df_samples_n.loc[
                df_samples_n['algorithm'] == algorithm].drop('algorithm', 1)
            # Group and concatenate lists
            df_aggregate = df_algorithm.groupby('links_out').agg(
                lambda x: x.tolist())
            # Ensure that only the desired parameters are aggregated or averaged
            df_keys_remaining = df_aggregate.columns.get_level_values(0)
            check_remaining_dimensions(df_keys_remaining, parameters_to_average)

            block_partition_modularity_inferred = [
                df_aggregate.loc[links_out].block_partition_modularity_inferred
                for links_out in links_out_range]
            mean_vals_inferred = [np.nanmean(df_aggregate.loc[links_out].block_partition_modularity_inferred) for links_out in links_out_range]
            # mean vals
            axs[axs_row].plot(
                links_out_range,
                mean_vals_inferred,
                color=colors_default[alg_i + 1],
                marker=markers_default[alg_i + 1],
                label=algorithm_names[algorithm])
            # scatter
            axs[axs_row].scatter(
                df_algorithm['links_out'].values.astype(float).tolist(),
                df_algorithm['block_partition_modularity_inferred'].values,
                color=colors_default[alg_i + 1],
                marker=markers_default[alg_i + 1],
                alpha=0.3)
        # Set axes properties
        axs[axs_row].set_ylabel(
            r'$\text{Modularity}$')
        axs[axs_row].yaxis.set_ticks_position('both')
        #axs[axs_row].set_ylim(bottom=0)
        axs[axs_row].legend()
        axs[axs_row].set_xlabel(
            r'$\text{Links between groups}$', horizontalalignment='right', x=1.0)
    return fig


# -------------------------------------------------------------------------
# endregion
# -------------------------------------------------------------------------



# -------------------------------------------------------------------------
# region Macaque
# -------------------------------------------------------------------------

def plot_performance_vs_fixed_coupling():
    subplots_v = len(algorithms)
    subplots_h = 2#len(samples_n_range)
    fig, axs = my_subplots(subplots_v, subplots_h)
    # Select data of interest
    df_interest = df[
        parameters_explored + ['precision', 'recall', 'specificity']]
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'repetition_i'}
    for (axs_col, samples_n) in enumerate(samples_n_range):
        # Select dataframe entries(i.e runs) with the desired property
        df_samples_n = df_interest.loc[
            df_interest['samples_n'] == samples_n].drop('samples_n', 1)
        for (axs_row, algorithm) in enumerate(algorithms):
            # Set subplot title
            axs[axs_row, axs_col].set_title('{0} (T={1})'.format(
                algorithm_names[algorithm],
                samples_n))
            # Select dataframe entries(i.e runs) with the same algorithm
            df_algorithm = df_samples_n.loc[
                df_interest['algorithm'] == algorithm].drop('algorithm', 1)
            # Group by fixed_coupling
            df_aggregate = df_algorithm.groupby('fixed_coupling').agg(
                lambda x: list(x))
            # Ensure that only the desired parameters are aggregated or averaged
            df_keys_remaining = df_aggregate.columns.get_level_values(0)
            check_remaining_dimensions(df_keys_remaining, parameters_to_average)
            # Plot precision
            axs[axs_row, axs_col].plot(
                fixed_coupling_range,
                [np.nanmean(df_aggregate['precision'][fixed_coupling])
                    for fixed_coupling in fixed_coupling_range],)
            axs[axs_row, axs_col].scatter(
                df_algorithm['fixed_coupling'].values.astype(float).tolist(),
                df_algorithm['precision'].values,
                alpha=0.3)
            # Plot recall
            axs[axs_row, axs_col].plot(
                fixed_coupling_range,
                [np.nanmean(df_aggregate['recall'][fixed_coupling])
                for fixed_coupling in fixed_coupling_range],)
            axs[axs_row, axs_col].scatter(
                df_algorithm['fixed_coupling'].values.astype(float).tolist(),
                df_algorithm['recall'].values,
                alpha=0.3)
            # Plot false positive rate
            axs[axs_row, axs_col].plot(
                fixed_coupling_range,
                [np.nanmean(df_aggregate['specificity'][fixed_coupling])
                for fixed_coupling in fixed_coupling_range])
            axs[axs_row, axs_col].scatter(
                df_algorithm['fixed_coupling'].values.astype(float).tolist(),
                df_algorithm['specificity'].values,
                alpha=0.3)
            # Set axes properties
            axs[axs_row, 0].set_ylabel(r'$\text{Performance}$')
            axs[axs_row, axs_col].yaxis.set_ticks_position('both')
            #axs[axs_row, axs_col].set_xlim(left=0.003, right=2.)
            axs[axs_row, axs_col].set_ylim(bottom=0)
            # Add legend
            axs[axs_row, axs_col].legend(
                labels=['Precision', 'Recall', 'Specificity'],
                loc='lower left')
        axs[axs_row, axs_col].set_xlabel(r'$\text{Weight}$') #, horizontalalignment='right', x=1.0)

    return fig


def plot_clustering_vs_fixed_coupling():
    fig, axs = my_subplots(1, 1, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'clustering_real',
        'clustering_inferred']]
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'repetition_i'}
    # Plot real values
    df_aggregate = df_interest.groupby('fixed_coupling').agg(lambda x: list(x))
    axs[0].plot(
        fixed_coupling_range,
        [np.nanmean(df_aggregate['clustering_real'][fixed_coupling])
            for fixed_coupling in fixed_coupling_range],
        label='Real',
        linestyle = 'None',
        color='k',
        marker='x',
        fillstyle='none',
        zorder=99,  # bring to foreground
        )
    # axs[0].scatter(
    #     df_algorithm['fixed_coupling'].values.astype(float).tolist(),
    #     [np.nanmean(x) for x in df_algorithm['clustering_real'].values],
    #     alpha=0.3,
    #     color='k',
    #     marker='x',
    #     zorder=99,  # bring to foreground
    #     )
    for algorithm_i, algorithm in enumerate(algorithms):
        # Select dataframe entries(i.e runs) with the same number of samples
        df_algorithm = df_interest.loc[
            df_interest['algorithm'] == algorithm].drop('algorithm', 1)
        # Group by fixed_coupling
        df_aggregate = df_algorithm.groupby('fixed_coupling').agg(
            lambda x: list(x))
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        # Plot average clustering coefficient inferred
        axs[0].plot(
            fixed_coupling_range,
            [np.nanmean(df_aggregate['clustering_inferred'][fixed_coupling])
             for fixed_coupling in fixed_coupling_range],
            label=algorithm_names[algorithm],
            color=colors_default[algorithm_i + 1],
            marker=markers_default[algorithm_i + 1],
            )
        axs[0].scatter(
            df_algorithm['fixed_coupling'].values.astype(float).tolist(),
            [np.nanmean(x) for x in df_algorithm['clustering_inferred'].values],
            alpha=0.3,
            color=colors_default[algorithm_i + 1],
            marker=markers_default[algorithm_i + 1],
            )
        # Set axes properties
        axs[0].set_xlabel(r'$\text{Weight}$', horizontalalignment='right', x=1.0)
        axs[0].set_ylabel(r'$\text{Clustering coefficient}$')
        axs[0].yaxis.set_ticks_position('both')
        #axs[0].set_xlim(left=0.003, right=2.)
        # Add legend
        axs[0].legend(loc='upper right')

    return fig


def plot_path_lenght_vs_fixed_coupling():
    fig, axs = my_subplots(1, 1, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'average_shortest_path_length_real',
        'average_shortest_path_length_inferred']]
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'repetition_i'}
    # Plot average path lenght real
    df_aggregate = df_interest.groupby('fixed_coupling').agg(lambda x: list(x))
    axs[0].plot(
        fixed_coupling_range,
        [np.nanmean(df_aggregate['average_shortest_path_length_real'][fixed_coupling])
            for fixed_coupling in fixed_coupling_range],
        label='Real',
        linestyle = 'None',
        color='k',
        marker='x',
        fillstyle='none',
        zorder=99,  # bring to foreground
        )
    # axs[0].scatter(
    #     df_algorithm['fixed_coupling'].values.astype(float).tolist(),
    #     df_algorithm['average_shortest_path_length_real'].values,
    #     alpha=0.3)
    for algorithm_i, algorithm in enumerate(algorithms):
        # Select dataframe entries(i.e runs) with the same number of samples
        df_algorithm = df_interest.loc[
            df_interest['algorithm'] == algorithm].drop('algorithm', 1)
        # Group by fixed_coupling
        df_aggregate = df_algorithm.groupby('fixed_coupling').agg(
            lambda x: list(x))
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        # Plot average path lenght inferred
        axs[0].plot(
            fixed_coupling_range,
            [np.nanmean(df_aggregate['average_shortest_path_length_inferred'][fixed_coupling])
             for fixed_coupling in fixed_coupling_range],
            label=algorithm_names[algorithm],
            color=colors_default[algorithm_i + 1],
            marker=markers_default[algorithm_i + 1],
            )
        axs[0].scatter(
            df_algorithm['fixed_coupling'].values.astype(float).tolist(),
            df_algorithm['average_shortest_path_length_inferred'].values,
            alpha=0.3,
            color=colors_default[algorithm_i + 1],
            marker=markers_default[algorithm_i + 1],
            )
        # Set axes properties
        axs[0].set_xlabel(r'$\text{Weight}$', horizontalalignment='right', x=1.0)
        axs[0].set_ylabel(r'$\text{Characteristic path length}$')
        axs[0].yaxis.set_ticks_position('both')
        # axs[0].set_xlim(left=0.003, right=2.)
        # Add legend
        axs[0].legend(loc='upper right')

    return fig


def plot_average_global_efficiency_vs_fixed_coupling():
    # Plot global_efficiency vs. Watts-Strogatz rewiring prob (scatter error)
    fig, axs = my_subplots(1, 1, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'average_global_efficiency_real',
        'average_global_efficiency_inferred']]
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'repetition_i'}
    # Plot global_efficiency real
    df_aggregate = df_interest.groupby('fixed_coupling').agg(lambda x: list(x))
    axs[0].plot(
        fixed_coupling_range,
        [np.nanmean(df_aggregate['average_global_efficiency_real'][fixed_coupling])
            for fixed_coupling in fixed_coupling_range],
        label='Real',
        linestyle = 'None',
        color='k',
        marker='x',
        fillstyle='none',
        zorder=99,  # bring to foreground
        )
    # axs[0].scatter(
    #     df_algorithm['fixed_coupling'].values.astype(float).tolist(),
    #     df_algorithm['average_global_efficiency_real'].values,
    #     alpha=0.3)
    for algorithm_i, algorithm in enumerate(algorithms):
        # Select dataframe entries(i.e runs) with the same number of samples
        df_algorithm = df_interest.loc[
            df_interest['algorithm'] == algorithm].drop('algorithm', 1)
        # Group by fixed_coupling
        df_aggregate = df_algorithm.groupby('fixed_coupling').agg(
            lambda x: list(x))
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        # Plot global_efficiency inferred
        axs[0].plot(
            fixed_coupling_range,
            [np.nanmean(df_aggregate['average_global_efficiency_inferred'][fixed_coupling])
             for fixed_coupling in fixed_coupling_range],
            label=algorithm_names[algorithm],
            color=colors_default[algorithm_i + 1],
            marker=markers_default[algorithm_i + 1],
            )
        axs[0].scatter(
            df_algorithm['fixed_coupling'].values.astype(float).tolist(),
            df_algorithm['average_global_efficiency_inferred'].values,
            alpha=0.3,
            color=colors_default[algorithm_i + 1],
            marker=markers_default[algorithm_i + 1],
            )
        # Set axes properties
        axs[0].set_xlabel(r'$\text{Weight}$', horizontalalignment='right', x=1.0)
        axs[0].set_ylabel(r'$\text{Global efficiency}$')
        axs[0].yaxis.set_ticks_position('both')
        # axs[0].set_xlim(left=0.003, right=2.)
        # Add legend
        axs[0].legend(loc='lower right')

    return fig


def plot_local_efficiency_vs_fixed_coupling():
    # Plot local efficiency vs. Watts-Strogatz rewiring prob (scatter error)
    fig, axs = my_subplots(1, 1, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'local_efficiency_real',
        'local_efficiency_inferred']]
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'repetition_i'}
    # Real
    df_aggregate = df_interest.groupby('fixed_coupling').agg(lambda x: list(x))
    axs[0].plot(
        fixed_coupling_range,
        [np.nanmean(df_aggregate['local_efficiency_real'][fixed_coupling])
            for fixed_coupling in fixed_coupling_range],
        label='Real',
        linestyle = 'None',
        color='k',
        marker='x',
        fillstyle='none',
        zorder=99,  # bring to foreground
        )
    # axs[0].scatter(
    #     df_algorithm['fixed_coupling'].values.astype(float).tolist(),
    #     [np.nanmean(x) for x in df_algorithm['local_efficiency_real'].values],
    #     alpha=0.3,
    #     linestyle = 'None',
    #     color='k',
    #     marker='x',
    #     zorder=99,  # bring to foreground
    #     )
    for algorithm_i, algorithm in enumerate(algorithms):
        # Select dataframe entries(i.e runs) with the same number of samples
        df_algorithm = df_interest.loc[
            df_interest['algorithm'] == algorithm].drop('algorithm', 1)
        # Group by fixed_coupling
        df_aggregate = df_algorithm.groupby('fixed_coupling').agg(
            lambda x: list(x))
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        # Inferred
        axs[0].plot(
            fixed_coupling_range,
            [np.nanmean(df_aggregate['local_efficiency_inferred'][fixed_coupling])
             for fixed_coupling in fixed_coupling_range],
            label=algorithm_names[algorithm],
            color=colors_default[algorithm_i + 1],
            marker=markers_default[algorithm_i + 1],
            )
        axs[0].scatter(
            df_algorithm['fixed_coupling'].values.astype(float).tolist(),
            [np.nanmean(x) for x in df_algorithm['local_efficiency_inferred'].values],
            alpha=0.3,
            color=colors_default[algorithm_i + 1],
            marker=markers_default[algorithm_i + 1],
            )
        # Set axes properties
        axs[0].set_xlabel(r'$\text{Weight}$', horizontalalignment='right', x=1.0)
        axs[0].set_ylabel(r'$\text{Local efficiency}$')
        axs[0].yaxis.set_ticks_position('both')
        # axs[0].set_xlim(left=0.003, right=2.)
        # Add legend
        axs[0].legend(loc='lower left')

    return fig


def scatter_out_degree_inferred_vs_out_degree_real_macaque_fixed_coupling():
    # Scatter out-degree inferred vs. real
    # Subplots horizontal: algorithms
    subplots_v = len(algorithms)
    subplots_h = len(fixed_coupling_range)
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'out_degree_inferred',
        'out_degree_real',
        ]]
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}
    for (axs_row, algorithm) in enumerate(algorithms):
        # Select dataframe entries(i.e runs) with the same algorithm
        df_algorithm = df_interest.loc[
            df_interest['algorithm'] == algorithm].drop('algorithm', 1)
        # Group by fixed_coupling and concatenate TE values lists
        df_aggregate = df_algorithm.groupby('fixed_coupling').agg(
            lambda x: x.tolist())
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        for (axs_col, fixed_coupling) in enumerate(fixed_coupling_range):
            # Set subplot title
            axs[axs_row, axs_col].set_title('{1} (weight={0})'.format(
                fixed_coupling,
                algorithm_names[algorithm]))
            out_degree_inferred = np.concatenate(
                df_aggregate.loc[fixed_coupling].out_degree_inferred)
            out_degree_real = np.concatenate(
                df_aggregate.loc[fixed_coupling].out_degree_real)
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
                r'$\text{Out-degree real}$')  # , horizontalalignment='right', x=1.0)
            axs[axs_row, 0].set_ylabel(
                r'$\text{Out-degree inferred}$')
            axs[axs_row, axs_col].yaxis.set_ticks_position('both')
    return fig


def plot_betweenness_centrality_vs_fixed_coupling():
    fig, axs = my_subplots(1, 1, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'betweenness_centrality_real',
        'betweenness_centrality_inferred']]
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'repetition_i'}
    # Plot real values
    df_aggregate = df_interest.groupby('fixed_coupling').agg(lambda x: list(x))
    axs[0].plot(
        fixed_coupling_range,
        [np.nanmean(df_aggregate['betweenness_centrality_real'][fixed_coupling])
            for fixed_coupling in fixed_coupling_range],
        label='Real',
        linestyle = 'None',
        color='k',
        marker='x',
        fillstyle='none',
        zorder=99,  # bring to foreground
        )
    # axs[0].scatter(
    #     df_algorithm['fixed_coupling'].values.astype(float).tolist(),
    #     [np.nanmean(x) for x in df_algorithm['betweenness_centrality_real'].values],
    #     alpha=0.3,
    #     color='k',
    #     marker='x',
    #     zorder=99,  # bring to foreground
    #     )
    for algorithm_i, algorithm in enumerate(algorithms):
        # Select dataframe entries(i.e runs) with the same number of samples
        df_algorithm = df_interest.loc[
            df_interest['algorithm'] == algorithm].drop('algorithm', 1)
        # Group by fixed_coupling
        df_aggregate = df_algorithm.groupby('fixed_coupling').agg(
            lambda x: list(x))
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        # Plot average betweenness_centrality coefficient inferred
        axs[0].plot(
            fixed_coupling_range,
            [np.nanmean(df_aggregate['betweenness_centrality_inferred'][fixed_coupling])
             for fixed_coupling in fixed_coupling_range],
            label=algorithm_names[algorithm],
            color=colors_default[algorithm_i + 1],
            marker=markers_default[algorithm_i + 1],
            )
        axs[0].scatter(
            df_algorithm['fixed_coupling'].values.astype(float).tolist(),
            [np.nanmean(x) for x in df_algorithm['betweenness_centrality_inferred'].values],
            alpha=0.3,
            color=colors_default[algorithm_i + 1],
            marker=markers_default[algorithm_i + 1],
            )
        # Set axes properties
        axs[0].set_xlabel(r'$\text{Weight}$', horizontalalignment='right', x=1.0)
        axs[0].set_ylabel(r'$\text{Centrality}$')
        axs[0].yaxis.set_ticks_position('both')
        #axs[0].set_xlim(left=0.003, right=2.)
        # Add legend
        axs[0].legend(loc='upper right')

    return fig


# normalised incoming weights


def plot_performance_vs_total_cross_coupling():
    subplots_v = len(algorithms)
    subplots_h = len(samples_n_range)
    fig, axs = my_subplots(subplots_v, subplots_h)
    # Select data of interest
    df_interest = df[
        parameters_explored + ['precision', 'recall', 'specificity']]
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'repetition_i'}
    for (axs_col, samples_n) in enumerate(samples_n_range):
        # Select dataframe entries(i.e runs) with the desired property
        df_samples_n = df_interest.loc[
            df_interest['samples_n'] == samples_n].drop('samples_n', 1)
        for (axs_row, algorithm) in enumerate(algorithms):
            # Set subplot title
            axs[axs_row, axs_col].set_title('{0} (T={1})'.format(
                algorithm_names[algorithm],
                samples_n))
            # Select dataframe entries(i.e runs) with the same algorithm
            df_algorithm = df_samples_n.loc[
                df_interest['algorithm'] == algorithm].drop('algorithm', 1)
            # Group by total_cross_coupling
            df_aggregate = df_algorithm.groupby('total_cross_coupling').agg(
                lambda x: list(x))
            # Ensure that only the desired parameters are aggregated or averaged
            df_keys_remaining = df_aggregate.columns.get_level_values(0)
            check_remaining_dimensions(df_keys_remaining, parameters_to_average)
            # Plot precision
            axs[axs_row, axs_col].plot(
                total_cross_coupling_range,
                [np.nanmean(df_aggregate['precision'][total_cross_coupling])
                    for total_cross_coupling in total_cross_coupling_range],
                '-o'
                ),
            axs[axs_row, axs_col].scatter(
                df_algorithm['total_cross_coupling'].values.astype(float).tolist(),
                df_algorithm['precision'].values,
                alpha=0.3)
            # Plot recall
            axs[axs_row, axs_col].plot(
                total_cross_coupling_range,
                [np.nanmean(df_aggregate['recall'][total_cross_coupling])
                for total_cross_coupling in total_cross_coupling_range],
                '-o'
                )
            axs[axs_row, axs_col].scatter(
                df_algorithm['total_cross_coupling'].values.astype(float).tolist(),
                df_algorithm['recall'].values,
                alpha=0.3)
            # Plot false positive rate
            axs[axs_row, axs_col].plot(
                total_cross_coupling_range,
                [np.nanmean(df_aggregate['specificity'][total_cross_coupling])
                for total_cross_coupling in total_cross_coupling_range],
                '-o'
                )
            axs[axs_row, axs_col].scatter(
                df_algorithm['total_cross_coupling'].values.astype(float).tolist(),
                df_algorithm['specificity'].values,
                alpha=0.3)
            # Set axes properties
            axs[axs_row, 0].set_ylabel(r'$\text{Performance}$')
            axs[axs_row, axs_col].yaxis.set_ticks_position('both')
            #axs[axs_row, axs_col].set_xlim(left=0.003, right=2.)
            axs[axs_row, axs_col].set_ylim(bottom=0)
            # Add legend
            axs[axs_row, axs_col].legend(
                labels=['Precision', 'Recall', 'Specificity'],
                loc='center left')
        axs[axs_row, axs_col].set_xlabel(r'$\text{Cross-coupling}$') #, horizontalalignment='right', x=1.0)

    return fig


def plot_performance_vs_samples_n_macaque():
    subplots_v = len(algorithms)
    subplots_h = len(total_cross_coupling_range)
    fig, axs = my_subplots(subplots_v, subplots_h)
    # Select data of interest
    df_interest = df[
        parameters_explored + ['precision', 'recall', 'specificity']]
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'repetition_i'}
    for (axs_col, total_cross_coupling) in enumerate(total_cross_coupling_range):
        # Select dataframe entries(i.e runs) with the desired property
        df_total_cross_coupling = df_interest.loc[
            df_interest['total_cross_coupling'] == total_cross_coupling].drop('total_cross_coupling', 1)
        for (axs_row, algorithm) in enumerate(algorithms):
            # Set subplot title
            axs[axs_row, axs_col].set_title('{0} (Cross-coupling={1})'.format(
                algorithm_names[algorithm],
                total_cross_coupling))
            # Select dataframe entries(i.e runs) with the same algorithm
            df_algorithm = df_total_cross_coupling.loc[
                df_interest['algorithm'] == algorithm].drop('algorithm', 1)
            # Group by samples_n
            df_aggregate = df_algorithm.groupby('samples_n').agg(
                lambda x: list(x))
            # Ensure that only the desired parameters are aggregated or averaged
            df_keys_remaining = df_aggregate.columns.get_level_values(0)
            check_remaining_dimensions(df_keys_remaining, parameters_to_average)
            # Plot precision
            axs[axs_row, axs_col].plot(
                samples_n_range,
                [np.nanmean(df_aggregate['precision'][samples_n])
                    for samples_n in samples_n_range],)
            axs[axs_row, axs_col].scatter(
                df_algorithm['samples_n'].values.astype(float).tolist(),
                df_algorithm['precision'].values,
                alpha=0.3)
            # Plot recall
            axs[axs_row, axs_col].plot(
                samples_n_range,
                [np.nanmean(df_aggregate['recall'][samples_n])
                for samples_n in samples_n_range],)
            axs[axs_row, axs_col].scatter(
                df_algorithm['samples_n'].values.astype(float).tolist(),
                df_algorithm['recall'].values,
                alpha=0.3)
            # Plot false positive rate
            axs[axs_row, axs_col].plot(
                samples_n_range,
                [np.nanmean(df_aggregate['specificity'][samples_n])
                for samples_n in samples_n_range])
            axs[axs_row, axs_col].scatter(
                df_algorithm['samples_n'].values.astype(float).tolist(),
                df_algorithm['specificity'].values,
                alpha=0.3)
            # Set axes properties
            axs[axs_row, 0].set_ylabel(r'$\text{Performance}$')
            axs[axs_row, axs_col].set_xscale('log')
            axs[axs_row, axs_col].yaxis.set_ticks_position('both')
            axs[axs_row, axs_col].set_xlim(left=0.5*min(samples_n_range), right=2*max(samples_n_range))
            axs[axs_row, axs_col].set_ylim(bottom=0)
            # Add legend
            axs[0, axs_col].legend(
                labels=['Precision', 'Recall', 'Specificity'],
                loc='lower left')
        axs[axs_row, axs_col].set_xlabel(r'$\text{T}$') #, horizontalalignment='right', x=1.0)

    return fig


def plot_clustering_vs_total_cross_coupling():
    fig, axs = my_subplots(1, 1, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'clustering_real',
        'clustering_inferred']]
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'repetition_i'}
    # Plot real values
    df_aggregate = df_interest.groupby('total_cross_coupling').agg(lambda x: list(x))
    axs[0].plot(
        total_cross_coupling_range,
        [np.nanmean(df_aggregate['clustering_real'][total_cross_coupling])
            for total_cross_coupling in total_cross_coupling_range],
        label='Real',
        linestyle = 'None',
        color='k',
        marker='x',
        fillstyle='none',
        zorder=99,  # bring to foreground
        )
    # axs[0].scatter(
    #     df_algorithm['total_cross_coupling'].values.astype(float).tolist(),
    #     [np.nanmean(x) for x in df_algorithm['clustering_real'].values],
    #     alpha=0.3,
    #     color='k',
    #     marker='x',
    #     zorder=99,  # bring to foreground
    #     )
    for algorithm_i, algorithm in enumerate(algorithms):
        # Select dataframe entries(i.e runs) with the same number of samples
        df_algorithm = df_interest.loc[
            df_interest['algorithm'] == algorithm].drop('algorithm', 1)
        # Group by total_cross_coupling
        df_aggregate = df_algorithm.groupby('total_cross_coupling').agg(
            lambda x: list(x))
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        # Plot average clustering coefficient inferred
        axs[0].plot(
            total_cross_coupling_range,
            [np.nanmean(df_aggregate['clustering_inferred'][total_cross_coupling])
             for total_cross_coupling in total_cross_coupling_range],
            label=algorithm_names[algorithm],
            color=colors_default[algorithm_i + 1],
            marker=markers_default[algorithm_i + 1],
            )
        axs[0].scatter(
            df_algorithm['total_cross_coupling'].values.astype(float).tolist(),
            [np.nanmean(x) for x in df_algorithm['clustering_inferred'].values],
            alpha=0.3,
            color=colors_default[algorithm_i + 1],
            marker=markers_default[algorithm_i + 1],
            )
        # Set axes properties
        axs[0].set_xlabel(r'$\text{Weight}$', horizontalalignment='right', x=1.0)
        axs[0].set_ylabel(r'$\text{Clustering coefficient}$')
        axs[0].yaxis.set_ticks_position('both')
        #axs[0].set_xlim(left=0.003, right=2.)
        # Add legend
        axs[0].legend(loc='upper left')

    return fig


def plot_path_lenght_vs_total_cross_coupling():
    fig, axs = my_subplots(1, 1, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'average_shortest_path_length_real',
        'average_shortest_path_length_inferred']]
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'repetition_i'}
    # Plot average path lenght real
    df_aggregate = df_interest.groupby('total_cross_coupling').agg(lambda x: list(x))
    axs[0].plot(
        total_cross_coupling_range,
        [np.nanmean(df_aggregate['average_shortest_path_length_real'][total_cross_coupling])
            for total_cross_coupling in total_cross_coupling_range],
        label='Real',
        linestyle = 'None',
        color='k',
        marker='x',
        fillstyle='none',
        zorder=99,  # bring to foreground
        )
    # axs[0].scatter(
    #     df_algorithm['total_cross_coupling'].values.astype(float).tolist(),
    #     df_algorithm['average_shortest_path_length_real'].values,
    #     alpha=0.3)
    for algorithm_i, algorithm in enumerate(algorithms):
        # Select dataframe entries(i.e runs) with the same number of samples
        df_algorithm = df_interest.loc[
            df_interest['algorithm'] == algorithm].drop('algorithm', 1)
        # Group by total_cross_coupling
        df_aggregate = df_algorithm.groupby('total_cross_coupling').agg(
            lambda x: list(x))
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        # Plot average path lenght inferred
        axs[0].plot(
            total_cross_coupling_range,
            [np.nanmean(df_aggregate['average_shortest_path_length_inferred'][total_cross_coupling])
             for total_cross_coupling in total_cross_coupling_range],
            label=algorithm_names[algorithm],
            color=colors_default[algorithm_i + 1],
            marker=markers_default[algorithm_i + 1],
            )
        axs[0].scatter(
            df_algorithm['total_cross_coupling'].values.astype(float).tolist(),
            df_algorithm['average_shortest_path_length_inferred'].values,
            alpha=0.3,
            color=colors_default[algorithm_i + 1],
            marker=markers_default[algorithm_i + 1],
            )
        # Set axes properties
        axs[0].set_xlabel(r'$\text{Weight}$', horizontalalignment='right', x=1.0)
        axs[0].set_ylabel(r'$\text{Characteristic path length}$')
        axs[0].yaxis.set_ticks_position('both')
        # axs[0].set_xlim(left=0.003, right=2.)
        # Add legend
        axs[0].legend(loc='upper right')

    return fig


def plot_average_global_efficiency_vs_total_cross_coupling():
    # Plot global_efficiency vs. Watts-Strogatz rewiring prob (scatter error)
    fig, axs = my_subplots(1, 1, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'average_global_efficiency_real',
        'average_global_efficiency_inferred']]
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'repetition_i'}
    # Plot global_efficiency real
    df_aggregate = df_interest.groupby('total_cross_coupling').agg(lambda x: list(x))
    axs[0].plot(
        total_cross_coupling_range,
        [np.nanmean(df_aggregate['average_global_efficiency_real'][total_cross_coupling])
            for total_cross_coupling in total_cross_coupling_range],
        label='Real',
        linestyle = 'None',
        color='k',
        marker='x',
        fillstyle='none',
        zorder=99,  # bring to foreground
        )
    # axs[0].scatter(
    #     df_algorithm['total_cross_coupling'].values.astype(float).tolist(),
    #     df_algorithm['average_global_efficiency_real'].values,
    #     alpha=0.3)
    for algorithm_i, algorithm in enumerate(algorithms):
        # Select dataframe entries(i.e runs) with the same number of samples
        df_algorithm = df_interest.loc[
            df_interest['algorithm'] == algorithm].drop('algorithm', 1)
        # Group by total_cross_coupling
        df_aggregate = df_algorithm.groupby('total_cross_coupling').agg(
            lambda x: list(x))
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        # Plot global_efficiency inferred
        axs[0].plot(
            total_cross_coupling_range,
            [np.nanmean(df_aggregate['average_global_efficiency_inferred'][total_cross_coupling])
             for total_cross_coupling in total_cross_coupling_range],
            label=algorithm_names[algorithm],
            color=colors_default[algorithm_i + 1],
            marker=markers_default[algorithm_i + 1],
            )
        axs[0].scatter(
            df_algorithm['total_cross_coupling'].values.astype(float).tolist(),
            df_algorithm['average_global_efficiency_inferred'].values,
            alpha=0.3,
            color=colors_default[algorithm_i + 1],
            marker=markers_default[algorithm_i + 1],
            )
        # Set axes properties
        axs[0].set_xlabel(r'$\text{Cross-coupling}$', horizontalalignment='right', x=1.0)
        axs[0].set_ylabel(r'$\text{Global efficiency}$')
        axs[0].yaxis.set_ticks_position('both')
        # axs[0].set_xlim(left=0.003, right=2.)
        # Add legend
        axs[0].legend(loc='lower right')

    return fig


def plot_local_efficiency_vs_total_cross_coupling():
    # Plot local efficiency vs. Watts-Strogatz rewiring prob (scatter error)
    fig, axs = my_subplots(1, 1, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'local_efficiency_real',
        'local_efficiency_inferred']]
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'repetition_i'}
    # Real
    df_aggregate = df_interest.groupby('total_cross_coupling').agg(lambda x: list(x))
    axs[0].plot(
        total_cross_coupling_range,
        [np.nanmean(df_aggregate['local_efficiency_real'][total_cross_coupling])
            for total_cross_coupling in total_cross_coupling_range],
        label='Real',
        linestyle = 'None',
        color='k',
        marker='x',
        fillstyle='none',
        zorder=99,  # bring to foreground
        )
    # axs[0].scatter(
    #     df_algorithm['total_cross_coupling'].values.astype(float).tolist(),
    #     [np.nanmean(x) for x in df_algorithm['local_efficiency_real'].values],
    #     alpha=0.3,
    #     linestyle = 'None',
    #     color='k',
    #     marker='x',
    #     zorder=99,  # bring to foreground
    #     )
    for algorithm_i, algorithm in enumerate(algorithms):
        # Select dataframe entries(i.e runs) with the same number of samples
        df_algorithm = df_interest.loc[
            df_interest['algorithm'] == algorithm].drop('algorithm', 1)
        # Group by total_cross_coupling
        df_aggregate = df_algorithm.groupby('total_cross_coupling').agg(
            lambda x: list(x))
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        # Inferred
        axs[0].plot(
            total_cross_coupling_range,
            [np.nanmean(df_aggregate['local_efficiency_inferred'][total_cross_coupling])
             for total_cross_coupling in total_cross_coupling_range],
            label=algorithm_names[algorithm],
            color=colors_default[algorithm_i + 1],
            marker=markers_default[algorithm_i + 1],
            )
        axs[0].scatter(
            df_algorithm['total_cross_coupling'].values.astype(float).tolist(),
            [np.nanmean(x) for x in df_algorithm['local_efficiency_inferred'].values],
            alpha=0.3,
            color=colors_default[algorithm_i + 1],
            marker=markers_default[algorithm_i + 1],
            )
        # Set axes properties
        axs[0].set_xlabel(r'$\text{Cross-coupling}$', horizontalalignment='right', x=1.0)
        axs[0].set_ylabel(r'$\text{Local efficiency}$')
        axs[0].yaxis.set_ticks_position('both')
        # axs[0].set_xlim(left=0.003, right=2.)
        # Add legend
        axs[0].legend(loc='lower right')

    return fig


def plot_average_global_efficiency_vs_samples_n():
    # Plot global_efficiency vs. Watts-Strogatz rewiring prob (scatter error)
    fig, axs = my_subplots(1, 1, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'average_global_efficiency_real',
        'average_global_efficiency_inferred']]
    df_interest = df_interest.loc[
        df_interest['total_cross_coupling'] == total_cross_coupling_interest].drop('total_cross_coupling', 1)
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'repetition_i'}
    # Plot global_efficiency real
    df_aggregate = df_interest.groupby('samples_n').agg(lambda x: list(x))
    axs[0].plot(
        samples_n_range,
        [np.nanmean(df_aggregate['average_global_efficiency_real'][samples_n])
            for samples_n in samples_n_range],
        label='Real',
        linestyle = 'None',
        color='k',
        marker='x',
        fillstyle='none',
        zorder=99,  # bring to foreground
        )
    for algorithm_i, algorithm in enumerate(algorithms):
        # Select dataframe entries(i.e runs) with the same number of samples
        df_algorithm = df_interest.loc[
            df_interest['algorithm'] == algorithm].drop('algorithm', 1)
        # Group by samples_n
        df_aggregate = df_algorithm.groupby('samples_n').agg(
            lambda x: list(x))
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        # Plot global_efficiency inferred
        axs[0].plot(
            samples_n_range,
            [np.nanmean(df_aggregate['average_global_efficiency_inferred'][samples_n])
             for samples_n in samples_n_range],
            label=algorithm_names[algorithm],
            color=colors_default[algorithm_i + 1],
            marker=markers_default[algorithm_i + 1],
            )
        axs[0].scatter(
            df_algorithm['samples_n'].values.astype(float).tolist(),
            df_algorithm['average_global_efficiency_inferred'].values,
            alpha=0.3,
            color=colors_default[algorithm_i + 1],
            marker=markers_default[algorithm_i + 1],
            )
        # Set axes properties
        axs[0].set_xlabel(r'$\text{T}$', horizontalalignment='right', x=1.0)
        axs[0].set_ylabel(r'$\text{Global efficiency}$')
        axs[0].yaxis.set_ticks_position('both')
        axs[0].set_xscale('log')
        axs[0].set_xlim(left=0.5*min(samples_n_range), right=2*max(samples_n_range))
        # Add legend
        axs[0].legend(loc='lower right')

    return fig


def plot_local_efficiency_vs_samples_n():
    # Plot local efficiency vs. Watts-Strogatz rewiring prob (scatter error)
    fig, axs = my_subplots(1, 1, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'local_efficiency_real',
        'local_efficiency_inferred']]
    df_interest = df_interest.loc[
        df_interest['total_cross_coupling'] == total_cross_coupling_interest].drop('total_cross_coupling', 1)
    # Choose which of the explored parameters will be aggregated or averaged
    parameters_to_average = {'repetition_i'}
    # Real
    df_aggregate = df_interest.groupby('samples_n').agg(lambda x: list(x))
    axs[0].plot(
        samples_n_range,
        [np.nanmean(df_aggregate['local_efficiency_real'][samples_n])
            for samples_n in samples_n_range],
        label='Real',
        linestyle = 'None',
        color='k',
        marker='x',
        fillstyle='none',
        zorder=99,  # bring to foreground
        )
    for algorithm_i, algorithm in enumerate(algorithms):
        # Select dataframe entries(i.e runs) with the same number of samples
        df_algorithm = df_interest.loc[
            df_interest['algorithm'] == algorithm].drop('algorithm', 1)
        # Group by samples_n
        df_aggregate = df_algorithm.groupby('samples_n').agg(
            lambda x: list(x))
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        # Inferred
        axs[0].plot(
            samples_n_range,
            [np.nanmean(df_aggregate['local_efficiency_inferred'][samples_n])
             for samples_n in samples_n_range],
            label=algorithm_names[algorithm],
            color=colors_default[algorithm_i + 1],
            marker=markers_default[algorithm_i + 1],
            )
        axs[0].scatter(
            df_algorithm['samples_n'].values.astype(float).tolist(),
            [np.nanmean(x) for x in df_algorithm['local_efficiency_inferred'].values],
            alpha=0.3,
            color=colors_default[algorithm_i + 1],
            marker=markers_default[algorithm_i + 1],
            )
        # Set axes properties
        axs[0].set_xlabel(r'$\text{T}$', horizontalalignment='right', x=1.0)
        axs[0].set_ylabel(r'$\text{Local efficiency}$')
        axs[0].yaxis.set_ticks_position('both')
        axs[0].set_xscale('log')
        axs[0].set_xlim(left=0.5*min(samples_n_range), right=2*max(samples_n_range))
        # Add legend
        axs[0].legend(loc='lower right')

    return fig


def scatter_out_degree_inferred_vs_out_degree_real_macaque_total_cross_coupling():
    # Scatter out-degree inferred vs. real
    # Subplots horizontal: algorithms
    subplots_v = len(algorithms)
    subplots_h = len(total_cross_coupling_range)
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'out_degree_inferred',
        'out_degree_real',
        ]]
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}
    for (axs_row, algorithm) in enumerate(algorithms):
        # Select dataframe entries(i.e runs) with the same algorithm
        df_algorithm = df_interest.loc[
            df_interest['algorithm'] == algorithm].drop('algorithm', 1)
        # Group by total_cross_coupling and concatenate TE values lists
        df_aggregate = df_algorithm.groupby('total_cross_coupling').agg(
            lambda x: x.tolist())
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        for (axs_col, total_cross_coupling) in enumerate(total_cross_coupling_range):
            # Set subplot title
            axs[axs_row, axs_col].set_title('{1} (weight={0})'.format(
                total_cross_coupling,
                algorithm_names[algorithm]))
            out_degree_inferred = np.concatenate(
                df_aggregate.loc[total_cross_coupling].out_degree_inferred)
            out_degree_real = np.concatenate(
                df_aggregate.loc[total_cross_coupling].out_degree_real)
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
                r'$\text{Out-degree real}$')  # , horizontalalignment='right', x=1.0)
            axs[axs_row, 0].set_ylabel(
                r'$\text{Out-degree inferred}$')
            axs[axs_row, axs_col].yaxis.set_ticks_position('both')
    return fig


def scatter_betweenness_centrality_inferred_vs_real_macaque_total_cross_coupling():
    # Scatter out-degree inferred vs. real
    # Subplots horizontal: algorithms
    subplots_v = len(algorithms)
    subplots_h = len(total_cross_coupling_range)
    fig, axs = my_subplots(subplots_v, subplots_h, sharex=True, sharey=True)
    # Select data of interest
    df_interest = df[parameters_explored + [
        'betweenness_centrality_inferred',
        'betweenness_centrality_real',
        ]]
    df_interest = df_interest.loc[
        df_interest['samples_n'] == T_interest].drop('samples_n', 1)
    # Choose which of the explored parameters to collect data over
    parameters_to_average = {'repetition_i'}
    for (axs_row, algorithm) in enumerate(algorithms):
        # Select dataframe entries(i.e runs) with the same algorithm
        df_algorithm = df_interest.loc[
            df_interest['algorithm'] == algorithm].drop('algorithm', 1)
        # Group by total_cross_coupling and concatenate TE values lists
        df_aggregate = df_algorithm.groupby('total_cross_coupling').agg(
            lambda x: x.tolist())
        # Ensure that only the desired parameters are aggregated or averaged
        df_keys_remaining = df_aggregate.columns.get_level_values(0)
        check_remaining_dimensions(df_keys_remaining, parameters_to_average)
        for (axs_col, total_cross_coupling) in enumerate(total_cross_coupling_range):
            # Set subplot title
            axs[axs_row, axs_col].set_title('{1} (weight={0})'.format(
                total_cross_coupling,
                algorithm_names[algorithm]))
            betweenness_centrality_inferred = np.concatenate(
                df_aggregate.loc[total_cross_coupling].betweenness_centrality_inferred)
            betweenness_centrality_real = np.concatenate(
                df_aggregate.loc[total_cross_coupling].betweenness_centrality_real)
            # Scatter
            axs[axs_row, axs_col].scatter(
                betweenness_centrality_real,
                betweenness_centrality_inferred,
                )
            # Plot identity line
            axs[axs_row, axs_col].plot(
                [min(betweenness_centrality_real), max(betweenness_centrality_real)],
                [min(betweenness_centrality_real), max(betweenness_centrality_real)],
                'g--')
            # Set axes properties
            axs[axs_row, axs_col].set_xlabel(
                r'$\text{Betweenness real}$')  # , horizontalalignment='right', x=1.0)
            axs[axs_row, 0].set_ylabel(
                r'$\text{Betweenness inferred}$')
            axs[axs_row, axs_col].yaxis.set_ticks_position('both')
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
mpl.rcParams['axes.titlesize'] = 15
mpl.rcParams['axes.labelsize'] = 15
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['lines.markersize'] = 5
mpl.rcParams['errorbar.capsize'] = 3
mpl.rcParams['legend.fontsize'] = 10
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
    '#7B85D4',  # similar to blue
    '#d7369e',  # similar to magenta
    '#83c995',  # similar to green
    '#f37738',  # similar to orange
    '#c4c9d8',
    '#859795',
    '#e9d043',
    '#ad5b50']
colors_default = colors_petroff
# Markers
markers_default = [
    'x',
    'o',
    '^',
    's',
    'D',
    'v',
    'h',
    '*']
cycler_default = cycler(
    color=colors_petroff,
    marker=markers_default)
#    scatter.marker=markers_default)


# -------------------------------------------------------------------------
# region Network Neuroscience validation paper (Random Erdos-Renyi)
# -------------------------------------------------------------------------

# # Select value of interest (for those plots where only one value is used)
# # alpha_interest = 0.001
# N_interest = 40
# T_interest = 10000
# first_not_explored = 'precision'
# # Ignore non-relevant explored parameters
# ignore_par = {
#     'jidt_threads_n',
#     'n_perm_max_stat',
#     'n_perm_min_stat',
#     'n_perm_max_seq',
#     'n_perm_omnibus',
# }
# parameters_explored = get_parameters_explored(first_not_explored, ignore_par)
# # Get parameter ranges
# #   df['nodes_n'] = 40
# #   print(df.head())
# nodes_n_range = np.unique(df['nodes_n']).astype(int)
# samples_n_range = np.unique(df['samples_n']).astype(int)
# alpha_c_range = np.unique(df['p_value']).astype(float)

# algorithms = np.unique(df['algorithm'])
# # Define dictionary of algorithm abbreviations
# algorithm_names = {
#     'bMI': 'Bivariate Mutual Information',
#     'bMI_greedy': 'Bivariate Mutual Information',
#     'bTE_greedy': 'Bivariate Transfer Entropy',
#     'mTE_greedy': 'Multivariate Transfer Entropy',
# }

# # new_fig('plot_performance_vs_N_scatter')
# # new_fig('plot_precision_vs_N_scatter')
# # new_fig('plot_precision_vs_N_scatter_inset')
# # new_fig('plot_recall_vs_N_scatter')
# # new_fig('plot_spectral_radius_vs_N_scatter')
# fig_list.append(plot_FPR_target_vs_alpha_quantile_mean())
# # fig_list.append(plot_FPR_target_vs_alpha_quantile_N_interest())
# # new_fig('plot_precision_recall_vs_alpha')
# # new_fig('plot_precision_vs_recall_subplots_T_scatter')
# # new_fig('plot_precision_vs_recall_subplots_T_scatter_aggregate')
# # new_fig('plot_precision_vs_recall_subplots_N_scatter')
# # new_fig('plot_delay_error_mean_vs_T_relative')
# # new_fig('plot_delay_error_mean_vs_T_relative_alpha_interest')
# # new_fig('plot_omnibus_TE_empirical_histogram_alpha_interest')
# # new_fig('plot_omnibus_TE_empirical_histogram_alpha_interest_T_interest')
# # new_fig('plot_omnibus_TE_empirical_vs_theoretical_causal_vars_alpha_interest')
# # new_fig('plot_omnibus_TE_empirical_vs_theoretical_inferred_vars_alpha_interest')
# # new_fig('plot_relative_error_TE_empirical_vs_theoretical_causal_vars_alpha_interest')
# fig_list.append(plot_path_lenght_vs_alpha_subplot_algorithm())


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

# Select value of interest (for those plots where only one value is used)
#alpha_interest = 0.001
N_interest = 100  # needed for SW coeff even if it's not an explored parameter
T_interest = 10000
#weight_interest = 'deterministic'
algorithm_interest = 'mTE_greedy'
first_not_explored = 'precision'
# Ignore non-relevant explored parameters
ignore_par = {
    'jidt_threads_n',
    'n_perm_max_stat',
    'n_perm_min_stat',
    'n_perm_max_seq',
    'n_perm_omnibus',
    'weight_distribution',
    }
parameters_explored = get_parameters_explored(first_not_explored, ignore_par)
# Get parameter ranges
# nodes_n_range = np.unique(df['nodes_n']).astype(int)
samples_n_range = np.unique(df['samples_n']).astype(int)
# alpha_c_range = np.unique(df['p_value']).astype(float)
WS_p_range = np.unique(df['WS_p']).astype(float)
algorithms = np.unique(df['algorithm'])
#weight_distributions = np.unique(df['weight_distribution'])
# Define dictionary of algorithm abbreviations
algorithm_names = {
    'bMI': 'Bivariate MI',
    'bMI_greedy': 'Bivariate MI',
    'bTE_greedy': 'Bivariate TE',
    'mTE_greedy': 'Multivariate TE',
}
fig_list.append(plot_performance_vs_WS_p())
fig_list.append(plot_density_vs_WS_p())
fig_list.append(plot_path_lenght_vs_WS_p())
fig_list.append(plot_clustering_vs_WS_p())
fig_list.append(plot_average_global_efficiency_vs_WS_p())
fig_list.append(plot_local_efficiency_vs_WS_p())
fig_list.append(plot_SW_coeff_vs_WS_p())  # default parameter WS_k=4
fig_list.append(plot_SW_index_vs_WS_p())  # default parameter WS_k=4

# fig_list.append(violin_TE_empirical_vs_WS_p())
# fig_list.append(violin_TE_theoretical_vs_WS_p())
# fig_list.append(imshow_TE_apparent_theoretical_causal_vars())
# fig_list.append(imshow_TE_complete_theoretical_causal_vars())
# fig_list.append(violin_AIS_theoretical_vs_WS_p())
# fig_list.append(plot_omnibus_TE_empirical_vs_theoretical_causal_vars())
# # fig_list.append(plot_omnibus_TE_empirical_vs_theoretical_inferred_vars_alpha_interest())
# fig_list.append(plot_relative_error_TE_empirical_vs_theoretical_causal_vars_vs_WS_p())
# fig_list.append(scatter_performance_vs_omnibus_TE())
# # fig_list.append(scatter_performance_vs_out_degree_real())
# # fig_list.append(scatter_omnibus_TE_vs_out_degree_real())
# fig_list.append(scatter_performance_vs_clustering_real())
# # fig_list.append(scatter_performance_vs_lattice_distance())
# fig_list.append(plot_performance_vs_WS_p_group_distance())
# fig_list.append(plot_spectral_radius_vs_WS_p_alpha_interest())
# fig_list.append(barchart_precision_per_motif())
# fig_list.append(barchart_recall_per_motif())


# -------------------------------------------------------------------------
# endregion
# -------------------------------------------------------------------------



# -------------------------------------------------------------------------
# region Barabasi-Albert
# -------------------------------------------------------------------------

# # Select value of interest (for those plots where only one value is used)
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
#     }
# parameters_explored = get_parameters_explored(first_not_explored, ignore_par)
# # Get parameter ranges
# # nodes_n_range = np.unique(df['nodes_n']).astype(int)
# samples_n_range = np.unique(df['samples_n']).astype(int)
# # alpha_c_range = np.unique(df['p_value']).astype(float)
# algorithms = np.unique(df['algorithm'])
# # weight_distributions = np.unique(df['weight_distribution'])
# # Define dictionary of algorithm abbreviations
# algorithm_names = {
#     'bMI': 'Bivariate MI',
#     'bMI_greedy': 'Bivariate MI',
#     'bTE_greedy': 'Bivariate TE',
#     'mTE_greedy': 'Multivariate TE',
# }

# fig_list.append(boxplot_BA_performance())
# fig_list.append(boxplot_BA_clustering())
# fig_list.append(boxplot_BA_average_global_efficiency())
# fig_list.append(boxplot_BA_local_efficiency())
# # fig_list.append(scatter_performance_vs_in_degree_real())
# # fig_list.append(scatter_performance_vs_out_degree_real())
# # fig_list.append(scatter_FP_per_target_vs_in_degree_real())
# # fig_list.append(scatter_FP_per_source_vs_in_degree_real())
# fig_list.append(scatter_clustering_inferred_vs_real())
# fig_list.append(scatter_clustering_vs_in_degree())
# fig_list.append(scatter_betweenness_centrality_inferred_vs_real())
# # #fig_list.append(scatter_eigenvector_centrality_inferred_vs_real())

# fig_list.append(scatter_in_degree_inferred_vs_real())
# # fig_list.append(scatter_out_degree_inferred_vs_real())
# # fig_list.append(loglog_distributions_in_degree_inferred_vs_real())
# # fig_list.append(loglog_distributions_out_degree_inferred_vs_real())
# fig_list.append(loglog_distributions_in_degree_inferred_vs_real())

# fig_list.append(scatter_in_degree_assortativity_inferred_vs_real())
# fig_list.append(plot_rich_club_in_degrees_inferred_vs_real())
# # fig_list.append(histogram_reciprocity_inferred_vs_real())

# -------------------------------------------------------------------------
# endregion
# -------------------------------------------------------------------------



# -------------------------------------------------------------------------
# region Stochastic Block Model
# -------------------------------------------------------------------------

# # Select value of interest (for those plots where only one value is used)
# T_interest = 10000
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
# links_out_range = np.unique(df['links_out']).astype(int)
# samples_n_range = np.unique(df['samples_n']).astype(int)
# algorithms = np.unique(df['algorithm'])
# # Define dictionary of algorithm abbreviations
# algorithm_names = {
#     'bMI': 'Bivariate MI',
#     'bMI_greedy': 'Bivariate MI',
#     'bTE_greedy': 'Bivariate TE',
#     'mTE_greedy': 'Multivariate TE',
# }
# fig_list.append(plot_SBM_precision_vs_links_out())
# fig_list.append(plot_SBM_recall_vs_links_out())
# # fig_list.append(plot_SBM_FP_vs_links_out())
# # fig_list.append(plot_SBM_FP_rate_vs_links_out())
# fig_list.append(plot_SBM_FP_and_FP_rate_vs_links_out(nodes_n=100))
# fig_list.append(plot_SBM_partition_modularity_vs_links_out())


# -------------------------------------------------------------------------
# endregion
# -------------------------------------------------------------------------



# -------------------------------------------------------------------------
# region Macaque
# -------------------------------------------------------------------------

# # Select value of interest (for those plots where only one value is used)
# N_interest = 71  # needed for SW coeff even if it's not an explored parameter
# T_interest = 10000
# total_cross_coupling_interest = 0.7
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
# samples_n_range = np.unique(df['samples_n']).astype(int)
# algorithms = np.unique(df['algorithm'])
# # fixed_coupling_range = np.unique(df['fixed_coupling'])
# print(df.head())
# total_cross_coupling_range = np.unique(df['total_cross_coupling'])
# # Define dictionary of algorithm abbreviations
# algorithm_names = {
#     'bMI': 'Bivariate MI',
#     'bMI_greedy': 'Bivariate MI',
#     'bTE_greedy': 'Bivariate TE',
#     'mTE_greedy': 'Multivariate TE',
# }
# fig_list.append(plot_performance_vs_samples_n_macaque())
# fig_list.append(plot_performance_vs_total_cross_coupling())
# fig_list.append(plot_average_global_efficiency_vs_samples_n())
# fig_list.append(plot_local_efficiency_vs_samples_n())
# fig_list.append(plot_average_global_efficiency_vs_total_cross_coupling())
# fig_list.append(plot_local_efficiency_vs_total_cross_coupling())
# fig_list.append(scatter_betweenness_centrality_inferred_vs_real_macaque_total_cross_coupling())
# fig_list.append(scatter_out_degree_inferred_vs_out_degree_real_macaque_total_cross_coupling())



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