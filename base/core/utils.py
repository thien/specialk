import subprocess
import tqdm
from multiprocessing import Pool, cpu_count

"""
Misc. functions used by a variety of parts of the library.
"""

def get_len(filepath):
    """
    Reads number of lines in the mose corpus without using python
    to deal with it. This is some order of magnitude faster!
    """
    command = "wc -l " + filepath
    process = subprocess.run(command.split(" "), stdout=subprocess.PIPE)
    return int(process.stdout.decode("utf-8").split(" ")[0])


def batch_compute(func, args, n_processes=cpu_count()-1):
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

