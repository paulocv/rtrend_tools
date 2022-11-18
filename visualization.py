"""Data visualization helper methods"""

import matplotlib as mpl
import matplotlib.cm
import matplotlib.colors
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_forecast_states import CDC_QUANTILES_SEQ, NUM_QUANTILES
from toolbox.plot_tools import make_axes_seq


# ----------------------------------------------------------------------------------------------------------------------
# PLOT FUNCTIONS: For post processed data
# ----------------------------------------------------------------------------------------------------------------------

def get_forecast_cmap(base_cmap="turbo", min_alpha=0.05):
    """A colormap suited for plotting forecast as density."""
    basecmap = mpl.cm.get_cmap(base_cmap)
    norm = None  # mpl.colors.Normalize(vmin=0, vmax=0.1)

    # # Create my own: DARK GREEN MONOTONE
    # cdict = {
    #     "red": [
    #         (0.0, 1.0, 1.0),  # Each row here are the (x, y_left, y_right) coordinates of a piecewise linear
    #         (1.0, 0.0, 0.0),  #   function with this color.
    #     ], "green": [
    #         (0.0, 1.0, 1.0),
    #         (1.0, 0.4, 0.4),
    #     ], "blue": [
    #         (0.0, 1.0, 1.0),
    #         (1.0, 0.0, 0.0),
    #     ], "alpha": [
    #         (0.0, 0.0, 0.0),
    #         (min_alpha, 1.0, 1.0),
    #         (1.0, 1.0, 1.0),
    #     ]
    # }
    # cmap = mpl.colors.LinearSegmentedColormap("ForeGreen", cdict)

    # Create myu own: BASED ON EXISTING CMAP
    def f(val):
        return basecmap(val)

    grid = np.linspace(0.0, 1.0, 10)  # Uniform values from 0 to 1

    cdict = {
        "red":   [(x, f(x)[0], f(x)[0]) for x in grid],
        "green": [(x, f(x)[1], f(x)[1]) for x in grid],
        "blue":  [(x, f(x)[2], f(x)[2]) for x in grid],
        "alpha": [
            (0.0, 0.0, 0.0),
            (min_alpha, 0.1, 1.0),
            (1.0, 1.0, 1.0),
        ]
    }
    cmap = mpl.colors.LinearSegmentedColormap("Fore" + base_cmap, cdict)

    return cmap, norm


def plot_forecasts_as_density(ct_fore2d: np.ndarray, ax: mpl.axes.Axes, i_t_pres=0):

    # Preamble
    # --------

    num_samples, num_days_fore = ct_fore2d.shape
    max_ct = ct_fore2d.max()  # Maximum value in th y-axis

    dens_matrix = np.empty((num_days_fore, max_ct + 1), dtype=float)  # Signature | a[i_t][sample]

    # Calculation
    # -----------

    # Density histogram (unsing bincount) for each time step
    # for i_t, day in enumerate(range(i_t_pres, i_t_pres + num_days_fore)):
    for i_t in range(num_days_fore):
        rt_ensemble = ct_fore2d[:, i_t]

        bc = np.bincount(rt_ensemble, minlength=max_ct + 1)
        dens_matrix[i_t, :] = bc / num_samples   # -()- NORMALIZES AT EACH TIME STEP

    # Plot
    # ----
    # ax.imshow(dens_matrix.T, interpolation="bilinear")
    t_array = np.arange(i_t_pres, i_t_pres + num_days_fore)
    y_array = np.arange(0, max_ct + 1)

    cmap, norm = get_forecast_cmap()
    dens = ax.pcolormesh(t_array, y_array, dens_matrix.T, shading="gouraud", cmap=cmap, norm=norm, edgecolor="none")
    # shading: gouraud, auto (or nearest)

    return dens, cmap


# Paired quantiles (EXCLUDES THE MEDIAN q = 0.5)
QUANTILE_TUPLES = [(CDC_QUANTILES_SEQ[i], CDC_QUANTILES_SEQ[-i - 1]) for i in range(int(NUM_QUANTILES / 2))]


def plot_forecasts_median(ct_fore2d: np.ndarray, ax: mpl.axes.Axes, i_t_pres=0,
                          color="green", alpha=1.0):

    num_samples, num_days_fore = ct_fore2d.shape
    t_array = np.arange(i_t_pres, i_t_pres + num_days_fore)

    return ax.plot(t_array, np.median(ct_fore2d, axis=0), ls="--", color=color, alpha=alpha)


def plot_forecasts_as_quantile_layers(ct_fore2d: np.ndarray, ax: mpl.axes.Axes, i_t_pres=0,
                                      color="green", base_alpha=0.08, quant_tuples=None):
    # Preamble
    # --------
    num_samples, num_days_fore = ct_fore2d.shape

    if quant_tuples is None:
        quant_tuples = QUANTILE_TUPLES

    layers = list()
    t_array = np.arange(i_t_pres, i_t_pres + num_days_fore)

    # Calculation and plot
    # --------------------

    # Quantiles
    for i_layer, q_pair in enumerate(quant_tuples):
        low_quant = np.quantile(ct_fore2d, q_pair[0], axis=0)
        hig_quant = np.quantile(ct_fore2d, q_pair[1], axis=0)

        layers.append(ax.fill_between(t_array, low_quant, hig_quant, color=color, alpha=base_alpha, lw=1))

    return layers


# ----------------------------------------------------------------------------------------------------------------------
# PLOT FUNCTIONS: For post processed data
# ----------------------------------------------------------------------------------------------------------------------


# Generic plot function for a whole single page
def plot_page_generic(chunk_seq, plot_func, page_width=9., ncols=3, ax_height=3., plot_kwargs=None):
    """
    Signature of the callable:
        plot_func(ax, item_from_chunk, i_ax, **plot_kwargs)
    """
    n_plots = len(chunk_seq)
    plot_kwargs = dict() if plot_kwargs is None else plot_kwargs

    fig, axes = make_axes_seq(n_plots, ncols, total_width=page_width, ax_height=ax_height)
    plot_outs = list()

    for i_ax, (ax, item) in enumerate(zip(axes, chunk_seq)):
        plot_outs.append(plot_func(ax, item, i_ax, **plot_kwargs))

    return fig, axes, plot_outs

def plot_precalc_quantiles_as_layers(ax, quant_lines: np.ndarray, x_array, alpha=0.1, color="green"):
    """
    Plots a sequence of lines as transparent filled layers.
    Lines are paired from the edges of quant_lines to its center ((0, -1), (1, -2), etc...).
    If the number of lines is odd, the central one is not included. ((n-1)/2)


    """
    num_layers = quant_lines.shape[0] // 2
    layers = list()

    for i_q in range(num_layers):
        layers.append(ax.fill_between(x_array, quant_lines[i_q], quant_lines[-(i_q + 1)],
                      alpha=alpha, color=color))

    return layers


# ----------------------------------------------------------------------------------------------------------------------
# HIGH LEVEL PLOT SCRIPTS
# ----------------------------------------------------------------------------------------------------------------------

def plot_ct_past_and_fore(ax, fore_time_labels: pd.DatetimeIndex, weekly_quantiles, factual_ct: pd.Series,
                          state_name, i_ax=None, synth_name=None,
                          num_quantiles=None, ct_color="C0", insert_point=None):
    """Plot all data and configure ax for C(t) data of a single state."""

    if not isinstance(fore_time_labels, pd.DatetimeIndex):  # Tries to convert into a pandas Index if not yet
        fore_time_labels = pd.DatetimeIndex(fore_time_labels)

    if num_quantiles is None:
        num_quantiles = weekly_quantiles.shape[0]

    # Optionally inserts the "today" point for better visualization of the forecast
    fore_x_data = fore_time_labels
    fore_y_data2d = weekly_quantiles
    if insert_point is not None:
        fore_x_data = fore_x_data.insert(0, insert_point[0])  # Includes present date
        fore_y_data2d = np.insert(fore_y_data2d, 0, insert_point[1], axis=1)

    # Plot C(t) forecast quantiles and median.
    q_layers = plot_precalc_quantiles_as_layers(ax, fore_y_data2d, fore_x_data)
    if num_quantiles % 2 == 1:
        median = ax.plot(fore_x_data, fore_y_data2d[num_quantiles // 2], "g--")
    else:
        median = None

    # Factual C(t) time series (past and, if available, forecast)
    ax.plot(factual_ct, color=ct_color)

    # Write state and synth name
    # ax.text(0.05, 0.9, state_name, transform=ax.transAxes)
    text = state_name if i_ax is None else f"{i_ax+1}) {state_name}"
    ax.text(0.05, 0.9, text, transform=ax.transAxes)

    # Write name of the synth method
    if synth_name is not None:
        ax.text(0.05, 0.8, synth_name, transform=ax.transAxes)

    rotate_ax_labels(ax)

    return q_layers, median


# ----------------------------------------------------------------------------------------------------------------------
# AUX METHODS
# ----------------------------------------------------------------------------------------------------------------------


def rotate_ax_labels(ax, angle=60, xy="x", which="major"):
    """This function could be included in my plot_tools."""
    labels = ax.get_xticklabels(which=which) if xy == "x" else ax.get_yticklabels(which=which)
    for label in labels:
        label.set(rotation=angle, horizontalalignment='right')
