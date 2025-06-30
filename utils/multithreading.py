import multiprocessing as mp
import os
from collections.abc import Callable, Iterable, Iterator
from multiprocessing.pool import ThreadPool
from typing import Any


__all__ = ["multithread_exec"]


ENV_VARS_TRUE_VALUES = {"1", "TRUE", "YES", "ON"}  # Add this line

def multithread_exec(func: Callable[[Any], Any], seq: Iterable[Any], threads: int | None = None) -> Iterator[Any]:
    """Execute a given function in parallel for each element of a given sequence

    >>> from doctane.utils.multithreading import multithread_exec
    >>> entries = [1, 4, 8]
    >>> results = multithread_exec(lambda x: x ** 2, entries)

    Args:
        func: function to be executed on each element of the iterable
        seq: iterable
        threads: number of workers to be used for multiprocessing

    Returns:
        iterator of the function's results using the iterable as inputs
    """
    threads = threads if isinstance(threads, int) else min(16, mp.cpu_count())
    # Single-thread
    if threads < 2 or os.environ.get("RECEIPT_CR_MULTIPROCESSING_DISABLE", "").upper() in ENV_VARS_TRUE_VALUES:
        results = map(func, seq)
    # Multi-threading
    else:
        with ThreadPool(threads) as tp:
            # ThreadPool's map function returns a list, but seq could be of a different type
            # That's why wrapping result in map to return iterator
            results = map(lambda x: x, tp.map(func, seq))  # noqa: C417
    return results
