import logging
import subprocess
from multiprocessing import Pool, cpu_count

import structlog
from tqdm import tqdm

"""
Misc. functions used by a variety of parts of the library.
"""

log = structlog.get_logger()


def get_len(filepath):
    """
    Reads number of lines in the mose corpus without using python
    to deal with it. This is some order of magnitude faster!
    """
    command = "wc -l " + filepath
    process = subprocess.run(command.split(" "), stdout=subprocess.PIPE)
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
