import functools
import logging
import platform
import subprocess
import warnings
from argparse import Namespace
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, List

import structlog
import torch
from tqdm import tqdm

"""
Misc. functions used by a variety of parts of the library.
"""

log = structlog.get_logger()


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter("always", DeprecationWarning)  # turn off filter
        warnings.warn(
            "Call to deprecated function {}.".format(func.__name__),
            category=DeprecationWarning,
            stacklevel=2,
        )
        warnings.simplefilter("default", DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func


def check_torch_device() -> str:
    """Check which device is available. Returns one of {cpu, cuda, mps}"""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_dataset(path: str) -> List[str]:
    data = []
    with open(path) as f:
        for line in f:
            data.append(line.strip())
    return data


def get_len(filepath):
    """
    Reads number of lines in the mose corpus without using python
    to deal with it. This is some order of magnitude faster!
    """
    if isinstance(filepath, Path):
        filepath = str(filepath)

    filepath = filepath.replace(" ", r"\ ").replace("(", r"\(").replace(")", r"\)")

    command = "wc -l " + filepath
    print(command)

    process = subprocess.run(["wc", "-l", filepath], stdout=subprocess.PIPE)
    _plt = platform.system()
    if _plt == "Linux":
        value = process.stdout.decode("utf-8").strip().split().pop(0)
    elif _plt == "Darwin":
        value = process.stdout.decode("utf-8").strip().split().pop(0)
    return int(value)


def batch_compute(func, args, n_processes=cpu_count() - 1):
    """
    Takes a function and some arguments for those functions,
    and returns a list of outputs generated from the function.
    Pools with multicores.
    """
    p = Pool(n_processes)
    res_list = []
    with tqdm(total=len(args)) as pbar:
        for i, res in enumerate(p.imap_unordered(func, args)):
            pbar.update()
            res_list.append(res)
    pbar.close()
    p.close()
    p.join()
    return res_list


def namespace_to_dict(ns: Namespace) -> Dict[str, Any]:
    # d = {}
    # for key in ns._get_kwargs():
    #     try:
    #         d[key] = getattr(ns, key)
    #     except TypeError:
    #         log.error(f"key {key} is not a string in namespace")
    # return d
    return {k: v for k, v in ns._get_kwargs()}
