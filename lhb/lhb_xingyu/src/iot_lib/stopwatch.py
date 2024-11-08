#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import time
from enum import Enum
from typing import List

th_lock = threading.Lock()


class _State(Enum):
    STOP = 0
    RUN = 1
    PAUSE = 2


class StopWatch:
    """
    stopWatch.start()
    stopWatch.pause()
    stopWatch.resume()
    stopWatch.stop()
    stopWatch.get_elapsed_time()
    """

    def __init__(self):
        self.start_time: float = 0.0
        self.stop_time: float = 0.0
        self.moments: List[float] = []
        self._state = _State.STOP

    @property
    def elapsed_time(self):
        return self.get_elapsed_time()

    def start(self):
        if self._state == _State.STOP:
            with th_lock:
                self._state = _State.RUN
                self.start_time = time.time()
                self.moments = []
                self.moments.append(self.start_time)

    def stop(self):
        if self._state == _State.RUN or self._state == _State.PAUSE:
            with th_lock:
                self._state = _State.STOP
                self.stop_time = time.time()
                self.moments.append(self.stop_time)

    def pause(self):
        if self._state == _State.RUN:
            with th_lock:
                self._state = _State.PAUSE
                self.moments.append(time.time())

    def resume(self):
        if self._state == _State.PAUSE:
            with th_lock:
                self._state = _State.RUN
                self.moments.append(time.time())

    def get_elapsed_time(self):
        elapsed = 0.0
        for i in range(len(self.moments)):
            if i % 2 == 1:
                elapsed += self.moments[i] - self.moments[i - 1]
        if self._state == _State.RUN and len(self.moments) % 2 == 1:
            elapsed += time.time() - self.moments[-1]
        return elapsed
