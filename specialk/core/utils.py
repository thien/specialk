import functools
import hashlib
import platform
import subprocess
import warnings
from argparse import Namespace
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

import torch
import yaml
from tqdm import tqdm

from specialk.core.logging import log

"""
Misc. functions used by a variety of parts of the library.
"""


def batch_texts(texts: Iterable[str], batch_size: int) -> Iterable[List[str]]:
    """
    Takes an iterable of texts and yields batches of texts.

    Args:
    texts (Iterable[str]): An iterable of text strings.
    batch_size (int): The number of texts in each batch.

    Yields:
    List[str]: A batch of texts.
    """
    current_batch = []

    for text in texts:
        current_batch.append(text)
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []

    # Yield any remaining texts in the last batch
    if current_batch:
        yield current_batch


def save_dict_to_yaml(data: dict, file_path: str) -> None:
    """
    Save a dictionary to a YAML file.

    Args:
    data (dict): The dictionary to be saved
    file_path (str): The path where the YAML file will be saved

    Returns:
    None
    """
    with open(file_path, "w") as file:
        yaml.dump(data, file, default_flow_style=False)


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
    len_file = get_len(path)
    with open(path, mode="r", encoding="utf-8") as f:
        for line in tqdm(f, total=len_file):
            data.append(line.strip())
    return data


def get_len(filepath: Union[str, Path]) -> int:
    """
    Reads number of lines in the mose corpus without using python
    to deal with it. This is some order of magnitude faster!
    """

    # temporarily convert a string into a Path object
    # so we can check whether the file exists.
    if not isinstance(filepath, Path):
        filepath = Path(filepath)
    if not filepath.exists():
        log.error(f"'{str(filepath)}' does not exist.")
        raise FileNotFoundError

    filepath = str(filepath)

    filepath = filepath.replace(" ", r"\ ").replace("(", r"\(").replace(")", r"\)")

    if platform.system() in {"Linux", "Darwin"}:
        process = subprocess.run(["wc", "-l", filepath], stdout=subprocess.PIPE)
        value = process.stdout.decode("utf-8").strip().split().pop(0)
    else:
        raise NotImplementedError

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


def hash(src: List[str], tgt: List[str]) -> str:
    """Generate hash of dataset.

    Args:
        src (List[str]): src dataset.
        tgt (List[str]): tgt dataset.

    Returns:
        str: md5 checksum of the dataset.
    """
    m = hashlib.md5()
    for x, y in zip(src, tgt):
        m.update(x.encode())
        m.update(y.encode())
    return str(m.hexdigest())
