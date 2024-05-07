import time


class control_timer:

    def __init__(self):
        self._start_time = 0
        self._end_time = 0

    _start_time = 0
    _end_time = 0

    def start(self):
        self._start_time = time.perf_counter_ns()

    def end(self):
        self._end_time = time.perf_counter_ns()

    def get_time(self):
        return (self._end_time - self._start_time) / 10**9

    def reset_time(self):
        self._start_time = 0
        self._end_time = 0
