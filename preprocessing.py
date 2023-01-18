"""
Module with utilities for preprocessing the data (after ROI cut, before interpolation).
"""
import warnings

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from rtrend_tools.forecast_structs import ForecastExecutionData, AbstractNoise, CovHospForecastOutput


# ----------------------------------------------------------------------------------------------------------------------
# AUX METHODS
# ----------------------------------------------------------------------------------------------------------------------
def apply_lowpass_filter(signal_array, cutoff=0.4, order=2):
    """Return a Butterworth lowpass filtered time series, with same dimension as input data."""
    if len(signal_array.shape) > 1:
        warnings.warn(f"Hey, a {len(signal_array.shape):d}D array/df was passed to 'apply_lowpass_filter', but 1D is "
                      f"expected. The filtered data may be silently wrong.")
    # noinspection PyTupleAssignmentBalance
    butter_b, butter_a = butter(order, cutoff, btype='lowpass', analog=False)
    return filtfilt(butter_b, butter_a, signal_array, method="gust")


def calc_weekday_noise_ensembles(data: pd.Series, denoised: pd.Series, relative=True):
    """
    Produce an ensemble of noise values grouped by weekday.
    Output signature: a[i_weekday, i_week]
    """
    # Group by weekdays
    data_gp = data.groupby(lambda d: d.weekday())
    deno_gp = denoised.groupby(lambda d: d.weekday())

    # Create weekdau vs week 2D arrays (weekays) containing data grouped per weekday. Signature: a[i_weekday, i_week]
    data_weekay = np.array([g[1].values for g in data_gp])  # WARNING: MISSING DATA or INCOMPLETE WEEK WILL RAISE ERROR
    deno_weekay = np.array([g[1].values for g in deno_gp])

    # For regular data, a 2D array should work well.
    result = data_weekay - deno_weekay
    if relative:
        result /= deno_weekay  # May produce NAN for zero points

    # #If data is irregular (not same amount of samples in each weekday), this should work. Results is a list, not array
    # result = [data - deno for data, deno in zip(data_weekay, deno_weekay)]
    # if relative:
    #     result = [dif / deno for dif, deno in zip(result, deno_weekay)]

    return result


def generate_weekday_noise(noise_call, weekday_params: np.ndarray, time_labels: pd.DatetimeIndex, num_samples):
    """
    Produce weekday sensitive noise for a generic distribution (given by noise_call callable).
    Weekday_params is either a 1D or 2D array with parameters of the distribution:
    * If it's 1D, noise_call must accept only one argument, and weekday_params must have size 7 (one for each weekday).
    * If 2D, then expected signature is a[i_weekday, i_param], where the number of parameters of noise_call is given by
      the second dimension size of weekday_params.

    The signature of noise_call must be:
    noise_call(*params, size=num_samples)

    Where the size of params is the same as the number of columns in weekday_params.

    Output signature: noise[i_sample, i_t]
    """
    weeklen = 7

    # Weekday params check
    w_shape = weekday_params.shape
    if w_shape[0] != weeklen:
        raise ValueError(f"Hey, weekday_params must have size {weeklen} but has shape {w_shape}.")

    if len(w_shape) == 1:  # Convert 1D array
        weekday_params = np.array(weekday_params).T  # Make it a column matrix

    # -----------
    noise = np.ones(shape=(num_samples, time_labels.shape[0]))  # a[i_sample, i_t]

    for i_date, date in enumerate(time_labels):
        wkday = date.weekday()
        noise[:, i_date] = noise_call(*weekday_params[wkday], size=num_samples)

    return noise


def choose_roi_for_denoise(exd: ForecastExecutionData):
    if exd.preproc_params.get("use_pre_roi", "False"):
        return exd.preproc_series
    else:
        return exd.state_series


# ----------------------------------------------------------------------------------------------------------------------
# DENOISING METHODS (COVID-HOSP FOCUSED) (remember to update DENOISE CALLABLE dict)
# ----------------------------------------------------------------------------------------------------------------------
# noinspection PyUnusedLocal
def denoise_lowpass(exd: ForecastExecutionData, fc: CovHospForecastOutput, filt_order=2, cutoff=0.4, **kwargs):
    roi = choose_roi_for_denoise(exd)
    return pd.Series(apply_lowpass_filter(roi.values, cutoff, filt_order),
                     index=roi.index)


# noinspection PyUnusedLocal
def denoise_polyfit(exd: ForecastExecutionData, fc: CovHospForecastOutput, poly_degree=3, **kwargs):
    # Selects proper roi and its integer index
    roi = choose_roi_for_denoise(exd)
    t_daily = np.arange(roi.shape[0])

    poly_coef, poly_resid = np.polyfit(t_daily, roi.values, deg=poly_degree, full=True)[0:2]
    poly_f = np.poly1d(poly_coef)  # Polynomial callable class.

    return pd.Series(poly_f(t_daily), index=roi.index)


# noinspection PyUnusedLocal
def denoise_rolling_average(exd: ForecastExecutionData, fc: CovHospForecastOutput, rollav_window=4, **kwargs):
    roi = choose_roi_for_denoise(exd)
    denoised = roi.rolling(rollav_window).mean()  # Rolling average
    denoised[:rollav_window-1] = roi[:rollav_window-1]  # Fill NAN values with original ones
    # fc.denoised_weekly[:] *= roi[-1] / data_weekly[-1] if data_weekly[-1] else 1
    # #                       ^  Rescale to match last day

    return denoised


# Dummy method, only redirects to the original time series.
# noinspection PyUnusedLocal
def denoise_none(exd: ForecastExecutionData, fc: CovHospForecastOutput, **kwargs):
    return exd.state_series


# Dictionary of denoising methods.
DENOISE_CALLABLE = {
    "polyfit": denoise_polyfit,
    "lowpass": denoise_lowpass,
    "rollav": denoise_rolling_average,
    "none": denoise_none,
}


# ----------------------------------------------------------------------------------------------------------------------
# NOISE FIT AND SYNTH (GENERATE) METHODS
# ----------------------------------------------------------------------------------------------------------------------


class NormalMultNoise(AbstractNoise):
    """Normal (Gaussian) noise, equal for positive and negative deviations."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mean = None
        self.std = None
        self.coef = kwargs.get("noise_coef", 1.0)  # Multiply noise by this coefficient
        self.seed = kwargs.get("noise_seed", None)  # Seed
        self._rng = np.random.default_rng(self.seed)

    def fit(self, data: pd.Series, denoised: pd.Series):
        reldev = calc_relative_dev_sample(data, denoised)
        self.mean = reldev.mean()
        # self.std = reldev.std() / 2.  # -()- Use appropriate standard deviation
        self.std = reldev.std()  # Use doubled standard deviation

    def generate(self, new_denoised: np.ndarray, time_labels):
        """new_denoised: a[i_sample, i_t]"""
        noise = np.maximum(self.coef * self._rng.normal(self.mean, self.std, size=new_denoised.shape), -1.)  # Clamped above -1
        return new_denoised * (1. + noise)


class NormalWeekdayNoise(AbstractNoise):
    """Weekday-sensitive Gaussian noise."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mean_array = None
        self.std_array = None
        self.coef = kwargs.get("noise_coef", 1.0)  # Multiply noise by this coefficient
        self.seed = kwargs.get("noise_seed", None)  # Seed
        self._rng = np.random.default_rng(self.seed)

    def fit(self, data: pd.Series, denoised: pd.Series):
        """Calculate an array of means and standard deviations, one for each weekday."""
        dif_weekay = calc_weekday_noise_ensembles(data, denoised, relative=True)  # Sample of differences

        # One set of parameters for each day of the week, sorted by pd_weekday
        self.mean_array = np.fromiter((x.mean() for x in dif_weekay), dtype=float)
        self.std_array = np.fromiter((x.std() for x in dif_weekay), dtype=float)

    def _noise_call(self, mean, std, size):
        """Callable to generate the normal noise."""
        return self._rng.normal(mean, std, size=size)

    def generate(self, new_denoised: np.ndarray, time_labels):
        """new_denoised: a[i_sample, i_t]"""
        params = np.array([self.mean_array, self.std_array]).T
        num_samples = new_denoised.shape[0]
        noise = generate_weekday_noise(self._noise_call, params, time_labels, num_samples)
        noise = np.maximum(self.coef * noise, -1.)  # Clamped above -1. Applies coefficient

        return new_denoised * (1. + noise)

    # def generate(self, new_denoised, time_labels):
    #     return new_denoised


class NoneNoise(AbstractNoise):
    """Produces zero noise."""
    def generate(self, new_denoised, time_labels):
        return new_denoised


NOISE_CLASS = {
    "normal": NormalMultNoise,
    "weekday_normal": NormalWeekdayNoise,
    "none": NoneNoise,
}


# Common methods for multiple classes
# ------------------------------------
def calc_relative_dev_sample(data: pd.Series, denoised: pd.Series):
    """
    Returns
    -------
    pd.Series
    """
    return (data - denoised) / denoised
