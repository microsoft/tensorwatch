# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import weakref, uuid
from typing import Any
from . import utils

class Stream:
    def __init__(self, stream_name:str=None, console_debug:bool=False):
        self._subscribers = weakref.WeakSet()
        self._subscribed_to = weakref.WeakSet()
        self.held_refs = set() # on some rare occasion we might want stream to hold references of other streams
        self.closed = False
        self.console_debug = console_debug
        self.stream_name = stream_name or str(uuid.uuid4()) # useful to use as key and avoid circular references

    def subscribe(self, stream:'Stream'): # notify other stream
        utils.debug_log('{} added {} as subscription'.format(self.stream_name, stream.stream_name))
        stream._subscribers.add(self)
        self._subscribed_to.add(stream)

    def unsubscribe(self, stream:'Stream'):
        utils.debug_log('{} removed {} as subscription'.format(self.stream_name, stream.stream_name))
        stream._subscribers.discard(self)
        self._subscribed_to.discard(stream)
        self.held_refs.discard(stream)
        #stream.held_refs.discard(self) # not needed as only subscriber should hold ref

    def write(self, val:Any, from_stream:'Stream'=None):
        if self.console_debug:
            print(self.stream_name, val)

        for subscriber in self._subscribers:
            subscriber.write(val, from_stream=self)

    def read_all(self, from_stream:'Stream'=None):
        for subscribed_to in self._subscribed_to:
            for item in subscribed_to.read_all(from_stream=self):
                yield item

    def load(self, from_stream:'Stream'=None):
        for subscribed_to in self._subscribed_to:
            subscribed_to.load(from_stream=self)

    def save(self, from_stream:'Stream'=None):
        for subscriber in self._subscribers:
            subscriber.save(from_stream=self)

    def close(self):
        if not self.closed:
            for subscribed_to in self._subscribed_to:
                subscribed_to._subscribers.discard(self)
            self._subscribed_to.clear()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

