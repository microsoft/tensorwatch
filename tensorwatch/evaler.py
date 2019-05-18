# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import threading, sys, logging
from collections.abc import Iterator
from .lv_types import EventData

# pylint: disable=unused-wildcard-import
# pylint: disable=wildcard-import
# pylint: disable=unused-import
from functools import *
from itertools import *
from statistics import *
import numpy as np
from .evaler_utils import *

class Evaler:
    class EvalReturn:
        def __init__(self, result=None, is_valid=False, exception=None):
            self.result, self.exception, self.is_valid = \
                result, exception, is_valid
        def reset(self):
            self.result, self.exception, self.is_valid = \
                None, None, False

    class PostableIterator:
        def __init__(self, eval_wait):
            self.eval_wait = eval_wait
            self.post_wait = threading.Event()
            self.event_data, self.ended = None, None # define attributes in init
            self.reset()

        def reset(self):
            self.event_data, self.ended = None, False
            self.post_wait.clear()

        def abort(self):
            self.ended = True
            self.post_wait.set()

        def post(self, event_data:EventData=None, ended=False):
            self.event_data, self.ended = event_data, ended
            self.post_wait.set()

        def get_vals(self):
            while True:
                self.post_wait.wait()
                self.post_wait.clear()
                if self.ended:
                    break
                else:
                    yield self.event_data
                    # below will cause result=None, is_valid=False when
                    # expression has reduce
                    self.eval_wait.set()

    def __init__(self, expr):
        self.eval_wait = threading.Event()
        self.reset_wait = threading.Event()
        self.g = Evaler.PostableIterator(self.eval_wait)
        self.expr = expr
        self.eval_return, self.continue_thread = None, None # define in __init__
        self.reset()

        self.th = threading.Thread(target=self._runner, daemon=True, name='evaler')
        self.th.start()
        self.running = True

    def reset(self):
        self.g.reset()
        self.eval_wait.clear()
        self.reset_wait.clear()
        self.eval_return = Evaler.EvalReturn()
        self.continue_thread = True
        
    def _runner(self):
        while True:
            # this var will be used by eval 
            l = self.g.get_vals() # pylint: disable=unused-variable
            try:
                result = eval(self.expr) # pylint: disable=eval-used
                if isinstance(result, Iterator):
                    for item in result:
                        self.eval_return = Evaler.EvalReturn(item, True)
                else:
                    self.eval_return = Evaler.EvalReturn(result, True)
            except Exception as ex: # pylint: disable=broad-except
                logging.exception('Exception occured while evaluating expression: ' + self.expr)
                self.eval_return = Evaler.EvalReturn(None, True, ex)
            self.eval_wait.set()
            self.reset_wait.wait()
            if not self.continue_thread:
                break
            self.reset()
        self.running = False
        utils.debug_log('eval runner ended!')

    def abort(self):
        utils.debug_log('Evaler Aborted')
        self.continue_thread = False
        self.g.abort()
        self.eval_wait.set()
        self.reset_wait.set()

    def post(self, event_data:EventData=None, ended=False, continue_thread=True):
        if not self.running:
            utils.debug_log('post was called when Evaler is not running')
            return None, False
        self.eval_return.reset()
        self.g.post(event_data, ended)
        self.eval_wait.wait()
        self.eval_wait.clear()
        # save result before it would get reset
        eval_return = self.eval_return
        self.reset_wait.set()
        self.continue_thread = continue_thread
        if isinstance(eval_return.result, Iterator):
            eval_return.result = list(eval_return.result)
        return eval_return

    def join(self):
        self.th.join()
