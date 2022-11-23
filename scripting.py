"""
General tools and parameters for automation.
"""

NEW_SECTION = "\n" + 20 * "-"
ENV_NAME = "rtrend_forecast"
ENV_FLEX_PATH = "rtrend_tools/envs/latest.yml"
MCMC_BIN = "main_mcmc_rt"


def prompt_yn(msg):
    answer = input(msg + " (y/n)")
    while True:
        if answer.lower() == "y":
            return True
        elif answer.lower() == "n":
            return False
        else:
            answer = input("Please answer \"y\" or \"n\"")
