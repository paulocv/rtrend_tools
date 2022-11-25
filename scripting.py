"""
General tools and parameters for automation.
"""

import json
import os
import subprocess

NEW_SECTION = "\n" + 20 * "-"
ENV_NAME = "rtrend_forecast"
ENV_FLEX_PATH = os.path.join("rtrend_tools", "envs", "latest.yml")
MCMC_BIN = "main_mcmc_rt"
MCMC_COMPILE_SCRIPT = os.path.join("rtrend_tools", "rt_mcmc", "compile_mcmc_rt.sh")


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
