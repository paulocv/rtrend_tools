"""
This module holds variables from the CDC requirements for the calculation and formatting
of the data.
"""
import numpy as np
from datetime import datetime, timedelta

# --- Parameters common to the Flu and COVID forecasts
CDC_QUANTILES_SEQ = np.round(np.append(np.append([0.01, 0.025], np.arange(0.05, 0.95 + 0.05, 0.050)), [0.975, 0.99]), 6)
NUM_QUANTILES = CDC_QUANTILES_SEQ.shape[0]
NUM_WEEKS_FORE = 4

# --- Flu parameters
FLU_TRUTH_URL = "https://raw.githubusercontent.com/cdcepi/Flusight-forecast-data/master/data-truth/" \
      "truth-Incident%20Hospitalizations.csv"
FLU_FORECAST_FILE_FMT = "{y:04d}-{m:02d}-{d:02d}-CEPH-Rtrend_fluH.csv"
NUM_STATES = 53  # Number of locations: states and jurisdictions, excluding the "US" entry.
WEEKDAY_TGT = 5  # 5 = Saturday
WEEKDAY_FC_DAY = 0  # 0 = Monday. The weekday expected as forecast_day entry (file name and column)
NUM_OUTP_LINES = NUM_WEEKS_FORE * (NUM_STATES + 1) * NUM_QUANTILES  # Expected output file number of lines
NUM_OUTP_LINES_WPOINTS = NUM_WEEKS_FORE * (NUM_STATES + 1) * (NUM_QUANTILES + 1)
REF_DEADLINE = datetime(2022, 11, 15, 23)  # A reference for submission deadlines (a given Monday 11pm)

# --- COVID parameters
COV_TRUTH_URL = "https://media.githubusercontent.com/media/reichlab/covid19-forecast-hub/master/data-truth/" \
                "truth-Incident%20Hospitalizations.csv"
COVH_FORECAST_FILE_FMT = "{y:04d}-{m:02d}-{d:02d}-CEPH-Rtrend_covid.csv"
COV_NUM_STATES = 54  # COVID data includes American Samoa
COV_REF_DEADLINE = datetime(2022, 11, 14, 15)  # A reference for submission deadlines (a given Monday 3pm)


def _get_next_deadline(ref_deadline, now=None):
    if now is None:
        now = datetime.today()

    delta = (now - ref_deadline) % timedelta(weeks=1)  # Time past the last deadline
    return now - delta + timedelta(weeks=1)


def get_next_flu_deadline_from(now=None):
    return _get_next_deadline(REF_DEADLINE, now)


def get_next_cov_deadline_from(now=None):
    return _get_next_deadline(COV_REF_DEADLINE, now)
