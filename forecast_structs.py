"""
Struct-like classes used in the forecast pipeline and surroundings.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline


PD_WEEKDAY = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]  # Weekday code labels as defined by Pandas. 0 = Mon, etc


class CDCDataBunch:

    def __init__(self):
        self.df: pd.DataFrame = None  # DataFrame with CDC data as read from the file.

        self.num_locs = None  # Number of unique locations
        self.loc_names = None  # List of unique location names
        self.loc_ids = None  # List of location ids (as strings)
        self.to_loc_id = None  # Conversion dictionary from location name to location id
        self.to_loc_name = None  # Conversion dictionary from location id to location name

        self.data_time_labels = None  # Array of dates present in the whole dataset.

        # Preprocessing data (not set during loading)
        self.day_roi_start: pd.Timestamp = None
        self.day_pres: pd.Timestamp = None

    def xs_state(self, state_name):
        return self.df.xs(self.to_loc_id[state_name], level="location")


class ForecastExecutionData:
    """
    Data for the handling of the forecast pipeline for a single state.
    Handled inside the function forecast_state().
    """
    def __init__(self):
        # Input parameters and data
        self.state_name = None
        self.state_series: pd.Series = None
        self.preproc_series: pd.Seires = None
        self.nweeks_fore = None
        self.except_params: dict = None
        self.tg_params: dict = None
        self.interp_params: dict = None
        self.preproc_params: dict = None
        self.mcmc_params: dict = None
        self.synth_params: dict = None
        self.recons_params: dict = None
        self.ct_cap = None

        # Pipeline handling
        self.stage = None   # Determines the forecast stage to be run in the next pipeline step. Written at debriefing.
        self.method = None  # Determines the method to use in current stage. Written at briefing.
        self.notes = None   # Debriefing notes to the briefing of the next stage. Written at debriefing.
        self.stage_log = list()

    def log_current_pipe_step(self):
        """Writes current stage and method into the stage_log"""
        self.stage_log.append((self.stage, self.method))


class ForecastOutput:
    """Contains data from forecasting process in one state: from interpolation to weekly reaggregation."""

    def __init__(self):
        self.day_0: pd.Timestamp = None  # First day to have report (7 days before the first stamp in ROI)
        self.day_pres: pd.Timestamp = None  # Day of the current report.
        self.ndays_roi = None
        self.ndays_fore = None

        # MCMC (and etc)
        self.tg_dist: np.ndarray = None

        # Interpolation results
        self.t_daily: np.ndarray = None
        self.past_daily_tlabels: pd.DatetimeIndex = None  # Daily time stamps for the ROI
        # self.fore_daily_tlabels: pd.DatetimeIndex = None  # Daily time stamps for the forecast region
        self.ct_past: np.ndarray = None
        self.float_data_daily: np.ndarray = None  # Float data from the spline of the cumulative sum
        self.daily_spline: UnivariateSpline = None

        # MCMC
        import rtrend_tools.rt_estimation as mcmcrt
        self.rtm: mcmcrt.McmcRtEnsemble = None

        # Synthesis
        self.synth_name = None
        self.rt_fore2d: np.ndarray = None  # Synthesized R(t) ensemble. a[i_sample, i_day]
        self.num_ct_samples = None

        # Reconstruction
        self.ct_fore2d: np.ndarray = None  # Synthesized cases time series ensemble. a[i_sample, i_day]
        self.ct_fore2d_weekly: np.ndarray = None  # a[i_sample, i_week]
        # self.weekly_quantiles: np.ndarray = None


class CovHospForecastOutput:
    """(For COVID Hospitallization incidence)
    Contains data from forecasting process in one state: from smoothening to weekly reaggregation.
    """

    def __init__(self):
        self.day_0: pd.Timestamp = None  # First day to have report (7 days before the first stamp in ROI)
        self.day_pres: pd.Timestamp = None  # Day of the current report.
        self.ndays_roi = None
        self.ndays_fore = None

        # MCMC (and etc)
        self.tg_dist: np.ndarray = None

        # Preprocesing results
        self.t_daily: np.ndarray = None
        self.past_daily_tlabels: pd.DatetimeIndex = None  # Daily time stamps for the ROI
        self.fore_daily_tlabels: pd.DatetimeIndex = None  # Daily time stamps for the forecast region
        self.ct_past: np.ndarray = None  # Preprocessed daily data that goes down the forecast pipeline
        self.float_data_daily: np.ndarray = None  # Float data after denoising process
        self.noise_obj: AbstractNoise = None

        # MCMC
        import rtrend_tools.rt_estimation as mcmcrt
        self.rtm: mcmcrt.McmcRtEnsemble = None

        # Synthesis
        self.synth_name = None
        self.rt_fore2d: np.ndarray = None  # Synthesized R(t) ensemble. a[i_sample, i_day]
        self.num_ct_samples = None

        # Reconstruction
        self.ct_fore2d: np.ndarray = None  # a[i_sample, i_day]
        # self.ct_fore2d_weekly: np.ndarray = None
        # self.weekly_quantiles: np.ndarray = None


class ForecastPost:
    """
    Contains post-processed data from the forecasting process in one state: the quantiles and other lightweight
    arrays that are sent back to the main scope.
    """
    def __init__(self):
        self.state_name = None
        self.synth_name = None

        # Forecasted data
        self.quantile_seq = None  # Just CDC default sequence of quantiles
        self.num_quantiles = None
        self.daily_quantiles: np.ndarray = None   # Forecasted quantiles | a[i_q, i_day]
        self.weekly_quantiles: np.ndarray = None  # Forecasted quantiles | a[i_q, i_week]
        self.samples_to_us: np.ndarray = None  # ct_fore_weekly samples to sum up the US series. | a[i_sample, i_week]

        # Interpolation
        self.t_daily: np.ndarray = None
        self.ct_past: np.ndarray = None
        self.float_data_daily: np.ndarray = None
        self.daily_spline: UnivariateSpline = None

        # MCMC past R(t) stats
        self.rt_past_mean: np.ndarray = None
        self.rt_past_median: np.ndarray = None
        self.rt_past_loquant: np.ndarray = None
        self.rt_past_hiquant: np.ndarray = None

        # Synthesis forecast R(t) stats
        self.rt_fore_median: np.ndarray = None
        self.rt_fore_loquant: np.ndarray = None
        self.rt_fore_hiquant: np.ndarray = None

        # Time labels
        self.day_0: pd.Timestamp = None     # First day of ROI.
        self.day_pres: pd.Timestamp = None  # Day time stamp of present
        self.day_fore: pd.Timestamp = None  # Last day forecasted
        self.past_daily_tlabels: pd.DatetimeIndex = None  # Daily time stamps for the ROI
        self.fore_daily_tlabels: pd.DatetimeIndex = None  # Daily time stamps for the forecast region
        self.fore_time_labels: pd.DatetimeIndex = None  # Weekly time stamps for forecast region

        # Misc
        self.xt = None  # Just some counter for execution time


class USForecastPost:
    """Special post process object for the US time series."""

    def __init__(self):
        self.weekly_quantiles: np.ndarray = None  # Forecasted quantiles | a[i_q, i_week]
        self.daily_quantiles: np.ndarray = None   # Forecasted quantiles | a[i_q, i_day]
        self.num_quantiles = None

        self.day_0: pd.Timestamp = None     # First day of ROI.
        self.day_pres: pd.Timestamp = None  # Day time stamp of present
        self.day_fore: pd.Timestamp = None  # Last day forecasted
        self.fore_daily_tlabels: pd.DatetimeIndex = None  # Daily dates
        self.fore_time_labels: pd.DatetimeIndex = None  # Weekly dates for the forecasted series.


# Polymorphic class for noise fit'n synth
class AbstractNoise:

    def __init__(self, **kwargs):
        pass

    def fit(self, data: pd.Series, denoised: pd.Series):
        """Abstract method. Extracts parameters from the (noisy) data and denoised series. Feeds inner variables."""

    def generate(self, new_denoised: np.ndarray, time_labels: pd.Index):
        """Abstract method. Must return a new data with calibrated noise incorporated.
        """
        raise NotImplementedError
