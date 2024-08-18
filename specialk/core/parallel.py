from typing import Callable, List, Any

import multiprocessing as mp
import pandas as pd
from tqdm import tqdm

class ParallelProcessor:
    @staticmethod
    def batch_generator(series_iterator: pd.Series, batch_size: int):
        """
        Generate batches of data from the pd.Series.

        This is done because I don't want to convert
        the whole series into a list.
        """
        batch = []
        for text in series_iterator:  # still fast.
            batch.append(text)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    @staticmethod
    def process(
        func: Callable,
        data: List[Any],
        chunk_size: int = 100000,
        init_func: Callable = None,
        init_args: tuple = None,
    ) -> List[Any]:
        """Generic parallel operation."""
        results = []
        with mp.Pool(initializer=init_func, initargs=init_args or ()) as pool:
            with tqdm(total=len(data)) as pbar:
                for chunk in (
                    batch
                    for batch in ParallelProcessor.batch_generator(data, chunk_size)
                ):
                    for batch in pool.map(func, chunk):
                        results.append(batch)
                        pbar.update(1)
        return results