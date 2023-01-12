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


# ----------------------------------------------------------------------------------------------------------------------
# DENOISING METHODS (COVID-HOSP FOCUSED) (remember to update DENOISE CALLABLE dict)
# ----------------------------------------------------------------------------------------------------------------------
# noinspection PyUnusedLocal
def denoise_lowpass(exd: ForecastExecutionData, fc: CovHospForecastOutput, filt_order=2, cutoff=0.4, **kwargs):
    return pd.Series(apply_lowpass_filter(exd.state_series.values, cutoff, filt_order),
                     index=exd.state_series.index)


# noinspection PyUnusedLocal
def denoise_polyfit(exd: ForecastExecutionData, fc: CovHospForecastOutput, poly_degree=3, **kwargs):
    poly_coef, poly_resid = np.polyfit(fc.t_daily, exd.state_series.values, deg=poly_degree, full=True)[0:2]
    poly_f = np.poly1d(poly_coef)  # Polynomial callable class.
    return pd.Series(poly_f(fc.t_daily), index=exd.state_series.index)


# noinspection PyUnusedLocal
def denoise_rolling_average(exd: ForecastExecutionData, fc: CovHospForecastOutput, rollav_window=4, **kwargs):
    denoised = exd.state_series.rolling(rollav_window).mean()  # Rolling average
    denoised[:rollav_window-1] = exd.state_series[:rollav_window-1]  # Fill NAN values with original ones
    # fc.denoised_weekly[:] *= exd.state_series[-1] / data_weekly[-1] if data_weekly[-1] else 1
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

    def fit(self, data: pd.Series, denoised: pd.Series):
        reldev = calc_relative_dev_sample(data, denoised)
        self.mean = reldev.mean()
        # self.std = reldev.std() / 2.  # -()- Use appropriate standard deviation
        self.std = reldev.std()  # Use doubled standard deviation

    def generate(self, new_denoised: np.ndarray):
        noise = np.maximum(np.random.normal(self.mean, self.std, size=new_denoised.shape), -1.)  # Clamped above -1
        return new_denoised * (1. + noise)


class NoneNoise(AbstractNoise):
    """Produces zero noise."""
    def generate(self, new_denoised):
        return new_denoised


NOISE_CLASS = {
    "normal": NormalMultNoise,
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
