"""
General tools and parameters for automation.
"""

import json
import os
import requests
import subprocess
import sys

from colorama import Fore, Style

NEW_SECTION = "\n" + 20 * "-"
ENV_NAME = "rtrend_forecast"
ENV_FLEX_PATH = os.path.join("rtrend_tools", "envs", "latest.yml")
MCMC_BIN = "main_mcmc_rt"
MCMC_COMPILE_SCRIPT = os.path.join("rtrend_tools", "rt_mcmc", "compile_mcmc_rt.sh")
FLU_TRUTH_DATA_FILE = os.path.join('hosp_data', 'truth-Incident Hospitalizations.csv')  # Warning: contains a space
FLU_TRUTH_DATA_BKPDIR = os.path.join("hosp_data", "past_BKP")


def prompt_yn(msg):
    answer = input(msg + " (y/n)")
    while True:
        if answer.lower() == "y":
            return True
        elif answer.lower() == "n":
            return False
        else:
            answer = input("Please answer \"y\" or \"n\"")


def conda_env_exists(env_name):
    proc = subprocess.run("conda env list --json".split(), capture_output=True)
    env_list = json.loads(proc.stdout)["envs"]
    return env_name in [os.path.basename(env) for env in env_list]


def fetch_file_online_raise(url, timeout=None):
    try:
        # FILE REQUEST COMMAND
        r: requests.Response = requests.get(url, timeout=timeout)

    except requests.exceptions.ConnectionError as e:
        sys.stderr.write(Fore.RED + "\n\n#############################\nThere was a connection error"
                         "\n#############################\n\n" + Style.RESET_ALL)
        print(e.args[0])
        print(Fore.RED + "\nThe file was NOT downloaded." + Style.RESET_ALL)
        return None

    return r
