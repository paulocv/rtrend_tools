"""General utilities. Maybe some could be promoted to my general toolbox?"""
import os
import pandas as pd

from pathos.multiprocessing import ProcessPool


def map_parallel_or_sequential(task, contents, ncpus=1, pool=None):
    """
    Map a function (task) into an iterable of inputs (contents), using either a sequential loop (if ncpus=1) or a
    pool of concurrent processes.
    Alternatively, for running in a pool of processes, a previously created pool can be informed, in which case ncpus is
    ignored.

    Parameters
    ----------
    task : callable
        A single-argument function to map on inputs.
    contents : iterable
        An iterable or sequence of inputs, each one will be passed as the only argument of 'task'.
    ncpus : int
        Number of concurrent processes. If ncpus=1 (default), a simple for loop is used. If greater than 1, a process
        pool is created. If parameter 'pool' is informed, ncpus is ignored.
    pool : Any
        A pathos process pool previously created. Overrides parameter 'ncpus' if informed.
    """
    if ncpus == 1 and pool is None:
        # Run sequentially
        results = list()
        for item in contents:
            results.append(task(item))

        return results

    if pool is None:
        pool = ProcessPool(ncpus=ncpus)

    # Run parallel pool
    return pool.map(task, contents)


def make_dir_from(pathname):
    """
    Creates directories recursively for a given path (possibly to a file).
    No message or error if some directories already exist.
    """
    dn = os.path.dirname(pathname)
    if dn:
        os.makedirs(dn, exist_ok=True)  # OS independent


def format_pd_timestamp_date(fmt: str, ts: pd.Timestamp):
    """
    Formats the date of a Pandas Timestamp object (ts) into a format string (fmt).
    The fmt string must have keyword fields with 'd' for day, 'm' for month and 'y' for year.
    """
    return fmt.format(y=ts.year, m=ts.month, d=ts.day)


def change_jupyter_py_logo(
        img_file, width=64, height=64, href=None):
    """Make a string to be called with IPython.core.display.HTML() to
    change the default python logo into another icon.
    """
    href_line = f"logoParent.innerHTML = " \
        f"'<a href=\"{href}\">' + logoParent.innerHTML + '</a>';\n" if href is not None else ""

    return f'''
    <script>
    var logoParent = document.getElementById("kernel_logo_widget")
    var logo = document.getElementById("kernel_logo_widget").getElementsByClassName("current_kernel_logo")[0];
    logo.src = "{img_file}";
    logo.style = "display: inline; width:{width}px; height:{height}px";
    ''' + href_line + "</script>"

