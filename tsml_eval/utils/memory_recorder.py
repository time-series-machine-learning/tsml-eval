"""Utility for recording the maximum memory usage of a function."""

import time
from threading import Thread

import psutil


def record_max_memory(
    function, args=None, kwargs=None, interval=0.1, return_func_time=False
):
    """
    Record the maximum memory usage of a function.

    Parameters
    ----------
    function : function
        The function to run.
    args : list, default=None
        The arguments to pass to the function.
    kwargs : dict, default=None
        The keyword arguments to pass to the function.
    interval : float, default=0.1
        The interval (in seconds) to check the memory usage.
    return_func_time : bool, default=False
        Whether to return the function's runtime.

    Returns
    -------
    max_memory : int
        The maximum memory usage (in bytes).
    runtime : int, optional
        The function's runtime (in milliseconds).

    Examples
    --------
    >>> def f(n):
    ...     return [i for i in range(n)]
    >>> max_mem = record_max_memory(f, args=[10000])
    """
    process = psutil.Process()
    start_memory = process.memory_info().rss

    thread = _FunctionThread(function, args, kwargs)
    thread.start()

    max_memory = process.memory_info().rss

    while True:
        time.sleep(interval)

        mem = process.memory_info().rss
        if mem > max_memory:
            max_memory = mem

        if not thread.is_alive():
            if thread.exception is not None:
                raise thread.exception

            if return_func_time:
                return max_memory - start_memory, thread.function_time
            else:
                return max_memory - start_memory


class _FunctionThread(Thread):
    """Thread that runs a function with arguments."""

    def __init__(self, function, args=None, kwargs=None):
        self.function = function
        self.args = args if args is not None else []
        self.kwargs = kwargs if kwargs is not None else {}

        self.function_time = -1
        self.exception = None

        super().__init__(daemon=True)

    def run(self):
        """Overloads the threading.Thread.run."""
        start = int(round(time.time() * 1000))
        try:
            self.function(*self.args, **self.kwargs)
        except Exception as e:
            self.exception = e
        end = int(round(time.time() * 1000))
        self.function_time = end - start
