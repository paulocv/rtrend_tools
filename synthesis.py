"""
Hub module for methods to synthesise future data from previous one.
"""
import time
from collections import defaultdict

import numba as nb
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
import warnings

from rtrend_tools.cdc_params import CDC_QUANTILES_SEQ
from rtrend_tools.rt_estimation import McmcRtEnsemble


# ----------------------------------------------------------------------------------------------------------------------
# FREQUENT AUXILIARY METHODS
# ----------------------------------------------------------------------------------------------------------------------

def make_tg_gamma_vector(shape, rate, tmax):
    """
    Prepare a vector with tg(s), where s is the integer time (in days) since infection.
    The vector is normalized but the FIRST ENTRY IS NOT REGARDED for the normalization.
    """

    def tg_dist_unnorm(t):
        return scipy.stats.gamma.pdf(t, a=shape, scale=1. / rate)

    st_array = np.arange(tmax, dtype=int)  # Retrograde time vector
    tg_array = tg_dist_unnorm(st_array)  #
    tg_array /= tg_array[1:].sum()  # Normalize EXCLUDING THE FIRST ELEMENT (which remains in there)

    return tg_array


def _get_arg_or_kwarg(args, kwargs, i_arg, key):
    """
    Use this function to get a required argument that can also be passed as keyword argument
    (using *args and **kwargs)
    """

    try:
        val = args[i_arg]
    except IndexError:
        try:
            val = kwargs[key]
        except KeyError:
            raise TypeError(f"Hey, argument '{key}' is required.")

    return val


def apply_ramp_to(arr, k_start, k_end, ndays_fore):
    # --- Apply the slope
    ramp = np.linspace(k_start, k_end, ndays_fore)
    return np.apply_along_axis(lambda x: x * ramp, axis=1, arr=arr)


def _k_start_f_default(x):
    return 0.93


def _k_end_f_default(x):
    # return -0.156 * x + 0.9375  # 1.6 to 1.2  |  1.1 to 0.9
    return -0.520 * x + 1.46  # 1.6 to 1.2  |  1.0 to 1.0


def _k_linear_func(x, r1, r2):
    return (r2/2 - r1) * (x - 1) + r1


def apply_linear_dynamic_ramp_to(arr, ndays_fore, r1_start, r1_end, r2_start, r2_end, x_array=None, i_saturate=-1,
                                 **kwargs):
    """
    Dynamic ramp, where k_start and k_end depend on an array of arguments.
    This array can be informed as x_array, expected to have the same number of samples as arr. If not informed, the
    first time point of each sample in arr (i.e., arr[:,0]) is used.

    --- For each of the positions, start and end, it defines a k-coeffient function:
    k(R) = a * (R - 1) + b, where
        a = r2 / 2 - r1
        b = r1
    And thus:
        R'(1) = r1
        R'(2) = r2


    """
    # --- Argument (x-array) array parsing
    if x_array is None:
        x_array = arr[:, 0]

    if x_array.shape[0] != arr.shape[0]:
        raise ValueError("Hey, x_array must have the same size as the first dimension of arr.")

    # --- Apply ramp
    ramp = np.linspace(_k_linear_func(x_array, r1_start, r2_start),
                       _k_linear_func(x_array, r1_end, r2_end),
                       ndays_fore).T

    # --- Apply saturation
    if i_saturate != -1:
        val_array = ramp[:, i_saturate]
        tgt_array = ramp[:, i_saturate + 1:]
        sat_array = np.repeat(np.reshape(val_array, (val_array.shape[0], 1)), tgt_array.shape[1], axis=1)

        tgt_array[:] = sat_array[:]

    return ramp * arr


# ----------------------------------------------------------------------------------------------------------------------
# R(t) SYNTHESIS: API METHODS
# ----------------------------------------------------------------------------------------------------------------------

def flat_mean(rtm: McmcRtEnsemble, ct_past, ndays_fore, **kwargs):
    """
    Synthesizes R(t) by taking the mean R(t) of the last ndays.
    """

    raise NotImplementedError()


def random_normal_synth(rtm: McmcRtEnsemble, ct_past, ndays_fore, *args, **kwargs):
    """
    Sample R(t) from a normal distribution.
    """
    center = kwargs.get("center", 1.0)
    sigma = kwargs.get("sigma", 0.05)
    seed = kwargs.get("seed", None)
    num_samples = kwargs.get("num_samples", None)
    out = kwargs.get("out", None)

    if num_samples is None:
        num_samples = rtm.get_array().shape[0]

    if out is None:
        out = np.empty((num_samples, ndays_fore), dtype=float)

    # Sample normal values
    rng = np.random.default_rng(seed)
    vals = rng.normal(center, sigma, num_samples)

    # Repeat values over time
    for i in range(ndays_fore):
        out[:, i] = vals

    return out


def sorted_flat_mean_ensemble(rtm: McmcRtEnsemble, ct_past, ndays_fore, *args, **kwargs):
    """
    Synth a sample of flat R(t) curves from sorted ensembles of R(t) in the past (sorted over iterations in each t).
    """
    ndays_past = _get_arg_or_kwarg(args, kwargs, 0, "ndays_past")

    # --- Keyword arguments parsing
    q_low = kwargs.get("q_low", 0.25)
    q_hig = kwargs.get("q_hig", 0.75)
    r_max = kwargs.get("r_max", None)
    out = kwargs.get("out", None)

    # --- Filter by quantiles
    i_low, i_hig = [round(rtm.iters * q) for q in (q_low, q_hig)]  # Indexes of the lower and upper quantiles
    filt = rtm.get_sortd()[i_low:i_hig, -ndays_past:]

    mean_vals = filt.mean(axis=1)

    # -()- Filter by maximum R value.
    if r_max is not None:
        # # -()- Generic mask
        # mean_vals = mean_vals[mean_vals < r_max]
        # -()- Assuming sorted
        mean_vals = mean_vals[:np.searchsorted(mean_vals, r_max, side="right")]

        if mean_vals.shape[0] == 0:
            warnings.warn(f"\nHey, all mean values were above r_max = {r_max}..\n")

    # --- Produce the R(t) ensemble from the means of the past series.
    if out is None:
        out = np.empty((mean_vals.shape[0], ndays_fore), dtype=float)

    for i in range(ndays_fore):
        out[:, i] = mean_vals

    return out


def sorted_bent_mean_ensemble(rtm: McmcRtEnsemble, ct_past, ndays_fore, *args, **kwargs):

    k_start = kwargs.get("k_start", 1.0)
    k_end = kwargs.get("k_end", 0.8)

    # --- Get the flat version
    out = sorted_flat_mean_ensemble(rtm, ct_past, ndays_fore, *args, **kwargs)

    if out.shape[0] != 0:  # Ensemble can't be empty
        # --- Apply the slope
        out = apply_ramp_to(out, k_start, k_end, ndays_fore)

    return out


def sorted_bent_ensemble_valbased(rtm: McmcRtEnsemble, ct_past, ndays_fore, *args, **kwargs):
    """
    Similar to sorted_bent_mean_ensemble, but take the quantiles based on maximum and/or minimum values instead.
    """
    # First define the quantiles to get the whole sample
    kwargs["q_low"] = 0.0
    kwargs["q_hig"] = 1.0
    k_start = kwargs.get("k_start", 1.0)
    k_end = kwargs.get("k_end", 0.8)
    rmean_top = kwargs.get("rmean_top", 1.2)  # Maximum value of R (from mean ensemble) to base the range on
    max_width = kwargs.get("max_width", 0.2)  # Maximmum width in the range, below rmean_top

    # Take the full flat ensemble
    out = sorted_flat_mean_ensemble(rtm, ct_past, ndays_fore, *args, **kwargs)

    # Select based on rmean_top and max_width
    i_top = np.searchsorted(out[:, 0], rmean_top)
    int_width = min(round(rtm.iters * max_width), i_top)  # Width in integer indexes. Capped by the sample size.
    out = out[i_top - int_width:i_top]  # Slices the array

    # --- Apply the slope
    if out.shape[0] != 0:  # Ensemble can't be empty
        out = apply_ramp_to(out, k_start, k_end, ndays_fore)

    return out


def sorted_dynamic_ramp(rtm: McmcRtEnsemble, ct_past, ndays_fore, *args, **kwargs):
    """
    Applies an R(t) ramp whose slope depends on the average value.
    """
    # k_start = kwargs.get("k_start", 1.0)
    # k_end = kwargs.get("k_end", 0.8)

    # --- Get the flat version
    out = sorted_flat_mean_ensemble(rtm, ct_past, ndays_fore, *args, **kwargs)

    if out.shape[0] != 0:  # Ensemble can't be empty
        # --- Apply the slope
        # out = apply_ramp_to(out, k_start, k_end, ndays_fore)
        out = apply_linear_dynamic_ramp_to(out, ndays_fore=ndays_fore, **kwargs)

    return out

# ----------------------------------------------------------------------------------------------------------------------
# C(t) RECONSTRUCTION
# ----------------------------------------------------------------------------------------------------------------------


def reconstruct_ct(ct_past, rt_fore, tg_array, tg_max=None, ct_fore=None, seed=None, **kwargs):
    """
    This function will later be the wrapper for numbaed functions. It should be able to handle both single R(t) series
    and ensembles of these as well.


    (draft, under construction)
    NOTICE: the calculation is recursive (C(t) depends on C(t - s)), thus cannot be vectorized.
    ------
    ct_past :
        Vector of number of cases in the past.
    rt_fore : Union(np.ndarray, pd.Series, pd.DataFrame)
        A time series with R(t) in future steps. Its size is assumed as steps_fut.
        For now, I expect a pandas Series. I can review this as needed though. E.g. accept 2d arrays.
        The size of the forecast is assumed to be the size of this vector.
    tg_array :
        Vector with Tg(s), where s is time since infection.
    ct_fore :
        c_futu. Contains the forecast.
    tg_max :
        Truncation value (one past) for generation time.
    """
    num_steps_fore = rt_fore.shape[-1]

    # Input types interpretation
    # --------------------------

    # --- R(t) input
    run_mult = False  # Whether to run the reconstruction over multiple R(t) series.
    if isinstance(rt_fore, np.ndarray):  # Numpy array
        if rt_fore.ndim == 2:
            run_mult = True
        elif rt_fore.ndim > 2:
            raise ValueError(f"Hey, input R(t) array rt_fore must either be 1D or 2D (ndim = {rt_fore.ndim}).")

    elif isinstance(rt_fore, pd.Series):  # Pandas 1D series
        rt_fore = rt_fore.to_numpy()

    elif isinstance(rt_fore, pd.DataFrame):  # Pandas 2D data frame
        rt_fore = rt_fore.to_numpy()
        run_mult = True

    else:
        raise TypeError(f"Hey, invalid type for input R(t) rt_fore: {type(rt_fore)}")

    # --- C(t) past array
    if isinstance(ct_past, pd.Series):
        ct_past = ct_past.to_numpy()

    # --- Tg(s) array
    if isinstance(tg_array, pd.Series):
        tg_array = tg_array.to_numpy()

    # Optional arguments handling
    # ---------------------------

    if tg_max is None:
        tg_max = tg_array.shape[0]

    # Output array
    if ct_fore is None:
        ct_fore = np.empty_like(rt_fore, dtype=int)  # Agnostic to 1D or 2D

    # Time-based seed
    if seed is None:
        seed = round(1000 * time.time())

    # Dispatch and run the reconstruction
    # -----------------------------------
    if run_mult:
        return _reconstruct_ct_multiple(ct_past, rt_fore, tg_array, tg_max, ct_fore,
                                        num_steps_fore, seed)
    else:
        return _reconstruct_ct_single(ct_past, rt_fore, tg_array, tg_max, ct_fore,
                                      num_steps_fore, seed)


@nb.njit
def _reconstruct_ct_single(ct_past, rt_fore, tg_dist, tg_max, ct_fore, num_steps_fore, seed):

    np.random.seed(seed)  # Seeds the numba or numpy generator

    # Main loop over future steps
    for i_t_fut in range(num_steps_fore):  # Loop over future steps
        lamb = 0.  # Sum of generated cases from past cases
        # r_curr = rt_fore.iloc[i_t_fut]  # Pandas
        r_curr = rt_fore[i_t_fut]

        # Future series chunk
        for st in range(1, min(i_t_fut + 1, tg_max)):
            lamb += r_curr * tg_dist[st] * ct_fore[i_t_fut - st]

        # Past series chunk
        for st in range(i_t_fut + 1, tg_max):
            lamb += r_curr * tg_dist[st] * ct_past[-(st - i_t_fut)]

        # Poisson number
        ct_fore[i_t_fut] = np.random.poisson(lamb)

    return ct_fore


@nb.njit
def _reconstruct_ct_multiple(ct_past, rt_fore_2d, tg_dist, tg_max, ct_fore_2d, num_steps_fore, seed):

    np.random.seed(seed)  # Seeds the numba or numpy generator

    # Main loop over R(t) samples
    for rt_fore, ct_fore in zip(rt_fore_2d, ct_fore_2d):

        # Loop over future steps
        for i_t_fut in range(num_steps_fore):  # Loop over future steps
            lamb = 0.  # Sum of generated cases from past cases
            # r_curr = rt_fore.iloc[i_t_fut]  # Pandas
            r_curr = rt_fore[i_t_fut]

            # Future series chunk
            for st in range(1, min(i_t_fut + 1, tg_max)):
                lamb += r_curr * tg_dist[st] * ct_fore[i_t_fut - st]

            # Past series chunk
            for st in range(i_t_fut + 1, tg_max):
                lamb += r_curr * tg_dist[st] * ct_past[-(st - i_t_fut)]

            # Poisson number
            ct_fore[i_t_fut] = np.random.poisson(lamb)

    return ct_fore_2d


# @nb.njit  # Not worth the compilation time at each parallel session.
def sample_series_with_replacement(ct_fore_2d: np.ndarray, nsamples_us: int, seed):
    """Randomly sample the ensemble of forecasted values, with the aim of feeding the US time series."""

    nsamples_state, nsteps = ct_fore_2d.shape
    np.random.seed(seed)  # Seeds numpy or numba generator.

    return ct_fore_2d[np.random.randint(0, nsamples_state, size=nsamples_us)]


def calc_tseries_ensemble_quantiles(ct_fore2d, quantiles_seq=None):
    """
    Calculate a sequence of quantiles for each point of a time series ensemble.
    Input signature: ct_fore2d[i_sample, i_t]

    If a custom sequence of quantiles is not informed, uses the default from CDC.
    """

    if quantiles_seq is None:
        quantiles_seq = CDC_QUANTILES_SEQ

    return np.quantile(ct_fore2d, quantiles_seq, axis=0)


def sort_ensembles_by_average(data2d_list: list[np.ndarray], axis=1):
    """
    Sort a list of 2D arrays by the average over a given axis.
    Employed to construct the US ensemble of time series, aiming to produce a more realistic
    set of quantiles.
    """
    result_list = []

    for data2d in data2d_list:

        avg_array = data2d.mean(axis=axis)
        sort_idx = np.argsort(avg_array)

        result_list.append(data2d[sort_idx, :])

    return result_list
