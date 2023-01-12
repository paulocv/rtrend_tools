"""
Import and export files fot the CDC Flu Forecast Challenge
"""

import datetime
import numpy as np
import os
import pandas as pd
import warnings

from collections import OrderedDict

from rtrend_tools.cdc_params import CDC_QUANTILES_SEQ, NUM_QUANTILES, NUM_STATES, WEEKDAY_TGT, \
    NUM_OUTP_LINES, NUM_OUTP_LINES_WPOINTS, COV_NUM_STATES
from rtrend_tools.forecast_structs import CDCDataBunch, ForecastPost
from rtrend_tools.interpolate import WEEKLEN


def load_cdc_truth_data(fname):
    """Load a generic truth data file from CDC."""
    cdc = CDCDataBunch()

    # Import
    cdc.df = pd.read_csv(fname, index_col=(0, 1), parse_dates=["date"])

    # Extract unique names and their ids
    cdc.loc_names = np.sort(cdc.df["location_name"].unique())  # Alphabetic order
    cdc.num_locs = cdc.loc_names.shape[0]
    cdc.loc_ids = cdc.df.index.levels[1].unique()

    # Make location id / name conversion
    cdc.to_loc_name = dict()  # Converts location id into name
    cdc.to_loc_id = dict()  # Converts location name into id
    for l_id in cdc.loc_ids:
        name = cdc.df.xs(l_id, level=1).iloc[0]["location_name"]  # Get the name from the first id occurrence
        cdc.to_loc_name[l_id] = name
        cdc.to_loc_id[name] = l_id

    cdc.df.sort_index(level="date", inplace=True)  # Sort by dates.

    cdc.data_time_labels = cdc.df.index.levels[0].unique().sort_values()

    return cdc


def export_forecast_cdc(fname, post_list, us, cdc, nweeks_fore, use_as_point=None,
                        add_week_to_labels=False):
    """
    Data is not assumed to strictly follow the CDC guidelines (all locations and quantiles),
    but warnings are thrown for irregular data.
    """

    # PPREAMBLE
    # ------------------------------------------------------------------------------------------------------------------
    today = datetime.datetime.today().date()
    today_str = today.isoformat()
    target_fmt = "{:d} wk ahead inc flu hosp"

    valid_post_list = [post for post in post_list if post is not None]
    num_valid = len(valid_post_list)

    # Check output size
    if num_valid != NUM_STATES:
        warnings.warn(f"\n[CDC-WARN] Hey, the number of calculated states/jurisdictions ({num_valid} "
                      f"+ US) does not match that required by CDC ({NUM_STATES} + US).\n")

    if use_as_point is not None:
        warnings.warn("HEY, use_as_point is not implemented in export_forecast_cdc.")

    # --- Build list of data arrays to be concatenated in the end
    # forecast_date_list =  # Date of the forecast. Same for entire doc, created later.
    location_list = list()  # Location (state) id
    target_list = list()  # String describing the week of forecast
    target_end_date_list = list()  # Date of the forecast report
    type_list = list()  # Type of output: "point" or "quantile"
    quantile_list = list()  # q-value of the quantile
    value_list = list()  # Value of the forecast data.
    actual_num_outp_lines = 0

    # DATA UNPACKING AND COLLECTION
    # ------------------------------------------------------------------------------------------------------------------

    # Definition of the processing routine
    # ------------------------------------
    def process_location(weekly_quantiles, num_q, quantile_seq, fore_time_labels, state_name):
        num_lines = nweeks_fore * (num_q + int(use_as_point is not None))  # Add point line, if requested

        # CDC format compliance check
        for date in fore_time_labels:
            if date.weekday() != WEEKDAY_TGT:
                warnings.warn(f"\n[CDC-WARN] Hey, wrong weekday ({date.weekday()}) was found in "
                              f"forecast data labels!)\n")

        # Allocate arrays
        # forecast_date_array = not needed here
        location_array = np.repeat(cdc.to_loc_id[state_name], num_lines)
        target_array = np.empty(num_lines, dtype=object)
        target_end_date_array = np.empty(num_lines, dtype=object)
        type_array = np.empty(num_lines, dtype=object)
        quantile_array = np.empty(num_lines, dtype=float)
        value_array = np.empty(num_lines, dtype=float)

        # ---
        i_line = 0
        for i_week in range(nweeks_fore):  # Loop over forecast weeks
            week = fore_time_labels[i_week] + int(add_week_to_labels) * datetime.timedelta(7)

            # Write data into arrays
            target_array[i_line:i_line + num_q] = target_fmt.format(i_week + 1)
            target_end_date_array[i_line: i_line + num_q] = week.date().isoformat()
            type_array[i_line: i_line + num_q] = "quantile"
            quantile_array[i_line: i_line + num_q] = quantile_seq[:]
            value_array[i_line: i_line + num_q] = weekly_quantiles[:, i_week]

            i_line += num_q

            # Point
            if use_as_point is not None:
                pass  # NOT IMPLEMENTED
                i_line += 1

        # Store arrays
        # forecast_date_list  # Not needed
        location_list.append(location_array)
        target_list.append(target_array)
        target_end_date_list.append(target_end_date_array)
        type_list.append(type_array)
        quantile_list.append(quantile_array)
        value_list.append(value_array)

        return num_lines

    # Application for all states and the US
    # -------------------------------------
    for i_post, post in enumerate(valid_post_list):   # Loop over each location
        actual_num_outp_lines += process_location(post.weekly_quantiles, post.num_quantiles,
                                                  post.quantile_seq, post.fore_time_labels, post.state_name)

    # --- APPLY SEPARATELY FOR US
    # post_samp = valid_post_list[0]  # First as example, they better be the same!
    actual_num_outp_lines += process_location(us.weekly_quantiles, NUM_QUANTILES, CDC_QUANTILES_SEQ,
                                              us.fore_time_labels, "US")

    # Concatenate all arrays in desired order
    forecast_date = np.repeat(today_str, actual_num_outp_lines)
    location = np.concatenate(location_list)
    target = np.concatenate(target_list)
    target_end_date = np.concatenate(target_end_date_list)
    type_ = np.concatenate(type_list)
    quantile = np.concatenate(quantile_list)
    value = np.concatenate(value_list)

    # Final check on the number of lines in output
    expec_num_outp_lines = NUM_OUTP_LINES if use_as_point is None else NUM_OUTP_LINES_WPOINTS
    if actual_num_outp_lines != expec_num_outp_lines:
        warnings.warn("\n[CDC-WARN] Hey, total number of lines in file doesn't match the expected "
                      "value.\n"
                      f"Total = {actual_num_outp_lines}   |   Expected = {NUM_OUTP_LINES}")

    # DF CONSTRUCTION AND EXPORT
    # ------------------------------------------------------------------------------------------------------------------

    out_df = pd.DataFrame(OrderedDict(
        forecast_date=forecast_date,
        location=location,
        target=target,
        target_end_date=target_end_date,
        type=type_,
        quantile=quantile,
        value=value,
    ))

    print(out_df)

    os.makedirs(os.path.dirname(fname), exist_ok=True)
    out_df.to_csv(fname, index=False)


def export_forecast_cov_hosp(fname, post_list, us, cdc, nweeks_fore, use_as_point=None, add_days=0, export_range=None):
    """
    Data is not assumed to strictly follow the CDC guidelines (all locations and quantiles),
    but warnings are thrown for irregular data.
    """

    # PREAMBLE
    # ------------------------------------------------------------------------------------------------------------------
    today = datetime.datetime.today().date()
    today_str = today.isoformat()
    # target_fmt = "{:d} wk ahead inc flu hosp"
    target_fmt = "{:d} day ahead inc hosp"

    if export_range is None:
        ndays_fore = WEEKLEN * nweeks_fore
        export_range = np.arange(ndays_fore)  # Standard range, given nweeks_fore
    else:
        ndays_fore = len(export_range)  # Overrides the number of forecast days by export_range

    valid_post_list = [post for post in post_list if post is not None]
    num_valid = len(valid_post_list)

    # Check output size
    if num_valid != COV_NUM_STATES:
        warnings.warn(f"\n[CDC-WARN] Hey, the number of calculated states/jurisdictions ({num_valid} "
                      f"+ US) does not match that required by CDC ({COV_NUM_STATES} + US).\n")

    if use_as_point is not None:
        warnings.warn("HEY, use_as_point is not implemented in export_forecast_cdc.")

    # --- Build list of data arrays to be concatenated in the end
    # forecast_date_list =  # Date of the forecast. Same for entire doc, created later.
    location_list = list()  # Location (state) id
    target_list = list()  # String describing the week of forecast
    target_end_date_list = list()  # Date of the forecast report
    type_list = list()  # Type of output: "point" or "quantile"
    quantile_list = list()  # q-value of the quantile
    value_list = list()  # Value of the forecast data.
    actual_num_outp_lines = 0

    # DATA UNPACKING AND COLLECTION
    # ------------------------------------------------------------------------------------------------------------------

    # Definition of the processing routine
    # ------------------------------------
    def process_location(daily_quantiles, num_q, quantile_seq, fore_time_labels, state_name):
        num_lines = ndays_fore * (num_q + int(use_as_point is not None))  # Add point line, if requested

        # Allocate arrays
        # forecast_date_array = not needed here
        location_array = np.repeat(cdc.to_loc_id[state_name], num_lines)
        target_array = np.empty(num_lines, dtype=object)
        target_end_date_array = np.empty(num_lines, dtype=object)
        type_array = np.empty(num_lines, dtype=object)
        quantile_array = np.empty(num_lines, dtype=float)
        value_array = np.empty(num_lines, dtype=float)

        # ---
        i_line = 0
        for i_day in export_range:  # Loop over forecast days
            day = fore_time_labels[i_day] + pd.Timedelta(add_days, "d")

            # Write data into arrays
            target_array[i_line:i_line + num_q] = target_fmt.format(i_day)
            target_end_date_array[i_line: i_line + num_q] = day.date().isoformat()
            type_array[i_line: i_line + num_q] = "quantile"
            quantile_array[i_line: i_line + num_q] = quantile_seq[:]
            value_array[i_line: i_line + num_q] = daily_quantiles[:, i_day]

            i_line += num_q

            # Point
            if use_as_point is not None:
                pass  # NOT IMPLEMENTED
                i_line += 1

        # Store arrays
        # forecast_date_list  # Not needed
        location_list.append(location_array)
        target_list.append(target_array)
        target_end_date_list.append(target_end_date_array)
        type_list.append(type_array)
        quantile_list.append(quantile_array)
        value_list.append(value_array)

        return num_lines

    # Application for all states and the US
    # -------------------------------------
    for i_post, post in enumerate(valid_post_list):   # Loop over each location
        post: ForecastPost
        actual_num_outp_lines += process_location(post.daily_quantiles, post.num_quantiles,
                                                  post.quantile_seq, post.fore_daily_tlabels, post.state_name)

    # --- APPLY SEPARATELY FOR US
    # post_samp = valid_post_list[0]  # First as example, they better be the same!
    actual_num_outp_lines += process_location(us.daily_quantiles, NUM_QUANTILES, CDC_QUANTILES_SEQ,
                                              us.fore_daily_tlabels, "United States")

    # Concatenate all arrays in desired order
    forecast_date = np.repeat(today_str, actual_num_outp_lines)
    location = np.concatenate(location_list)
    target = np.concatenate(target_list)
    target_end_date = np.concatenate(target_end_date_list)
    type_ = np.concatenate(type_list)
    quantile = np.concatenate(quantile_list)
    value = np.concatenate(value_list)

    # # Final check on the number of lines in output
    # expec_num_outp_lines = NUM_OUTP_LINES if use_as_point is None else NUM_OUTP_LINES_WPOINTS
    # if actual_num_outp_lines != expec_num_outp_lines:
    #     warnings.warn("\n[CDC-WARN] Hey, total number of lines in file doesn't match the expected "
    #                   "value.\n"
    #                   f"Total = {actual_num_outp_lines}   |   Expected = {NUM_OUTP_LINES}")

    # DF CONSTRUCTION AND EXPORT
    # ------------------------------------------------------------------------------------------------------------------

    out_df = pd.DataFrame(OrderedDict(
        forecast_date=forecast_date,
        location=location,
        target=target,
        target_end_date=target_end_date,
        type=type_,
        quantile=quantile,
        value=value,
    ))

    print(out_df)

    os.makedirs(os.path.dirname(fname), exist_ok=True)
    out_df.to_csv(fname, index=False)


def load_forecast_cdc(fname):
    """Import forecast file in the CDC format, possibly created with 'export_forecast_cdc()'. """

    df = pd.read_csv(fname, header=0, parse_dates=["forecast_date", "target_end_date"])

    return df


def make_state_arrays(weekly_quantiles, state_id, num_quantiles, nweeks_fore):
    pass
