import time
from threading import Thread

import psutil


def record_max_memory(
    function, args=None, kwargs=None, interval=0.1, return_func_time=False
):
    process = psutil.Process()
    start_memory = process.memory_info().rss

    thread = FunctionThread(function, args, kwargs)
    thread.start()

    max_memory = process.memory_info().rss

    while True:
        time.sleep(interval)

        mem = process.memory_info().rss
        if mem > max_memory:
            max_memory = mem

        if thread.has_shutdown:
            if return_func_time:
                return max_memory - start_memory, thread.function_time
            else:
                return max_memory - start_memory


class FunctionThread(Thread):
    def __init__(self, function, args=None, kwargs=None):
        self.function = function
        self.args = args if args is not None else []
        self.kwargs = kwargs if kwargs is not None else {}

        self.function_time = -1
        self.has_shutdown = False

        super(FunctionThread, self).__init__(daemon=True)

    def run(self):
        """Overloads the threading.Thread.run."""
        start = int(round(time.time() * 1000))
        self.function(*self.args, **self.kwargs)
        end = int(round(time.time() * 1000)) - start
        self.function_time = end - start
        self.has_shutdown = True
