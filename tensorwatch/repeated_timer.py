# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import threading
import time
import weakref

class RepeatedTimer:
    class State:
        Stopped=0
        Paused=1
        Running=2

    def __init__(self, secs, callback, count=None):
        self.secs = secs
        self.callback = weakref.WeakMethod(callback) if callback else None
        self._thread = None
        self._state = RepeatedTimer.State.Stopped
        self.pause_wait = threading.Event()
        self.pause_wait.set()
        self._continue_thread = False
        self.count = count

    def start(self):
        self._continue_thread = True
        self.pause_wait.set()
        if self._thread is None or not self._thread.isAlive():
            self._thread = threading.Thread(target=self._runner, name='RepeatedTimer', daemon=True)
            self._thread.start()
        self._state = RepeatedTimer.State.Running

    def stop(self, block=False):
        self.pause_wait.set()
        self._continue_thread = False
        if block and not (self._thread is None or not self._thread.isAlive()):
            self._thread.join()
        self._state = RepeatedTimer.State.Stopped

    def get_state(self):
        return self._state


    def pause(self):
        if self._state == RepeatedTimer.State.Running:
            self.pause_wait.clear()
            self._state = RepeatedTimer.State.Paused
        # else nothing to do
    def unpause(self):
        if self._state == RepeatedTimer.State.Paused:
            self.pause_wait.set()
            if self._state == RepeatedTimer.State.Paused:
                self._state = RepeatedTimer.State.Running
        # else nothing to do

    def _runner(self):
        while (self._continue_thread):
            if self.count:
                self.count -= 0
                if not self.count:
                    self._continue_thread = False

            if self._continue_thread:
                self.pause_wait.wait()
                if self.callback and self.callback():
                    self.callback()()

            if self._continue_thread:
                time.sleep(self.secs)

        self._thread = None
        self._state = RepeatedTimer.State.Stopped