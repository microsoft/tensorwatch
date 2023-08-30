# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import weakref, uuid
from typing import Any
from . import utils
from .lv_types import StreamItem

class Stream:
    """
    Stream allows you to write values into it. One stream can subscribe to many streams. When a value is
    written in the stream, all subscribers also gets that value written into them (the from_steam parameter
    is set to the source stream).

    You can read values from a stream by calling read_all method. This will yield values as someone or subscribed streams are
    writing into the stream. You can also load all values from the stream. The default stream won't load anything but
    derived streams like file may load all values from the file and write into themselves.

    You can think of stream as a pipe that can be chained to other pipees. As value is put in pipe, it travels through
    connected pipes. The read_all method is like a tap that you can use to read values from the pipe. The load method is
    for specialized streams that may generate values in that pipe.

    Stream class supports full multi-threading.
    """
    def __init__(self, stream_name:str=None, console_debug:bool=False):
        self._subscribers = weakref.WeakSet()
        self._subscribed_to = weakref.WeakSet()
        self.held_refs = set() # on some rare occasion we might want stream to hold references of other streams
        self.closed = False
        self.console_debug = console_debug
        self.stream_name = stream_name or str(uuid.uuid4()) # useful to use as key and avoid circular references
        self.items_written = 0

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

    def to_stream_item(self, val:Any):
        stream_item = val if isinstance(val, StreamItem) else \
            StreamItem(value=val, stream_name=self.stream_name)
        if stream_item.stream_name is None:
            stream_item.stream_name = self.stream_name
        if stream_item.item_index is None:
            stream_item.item_index = self.items_written
        return stream_item

    def write(self, val:Any, from_stream:'Stream'=None):
        # if you override write method, first you must call self.to_stream_item
        # so it can stamp the stamp the stream name
        stream_item = self.to_stream_item(val)

        if self.console_debug:
            print(self.stream_name, stream_item)

        for subscriber in self._subscribers:
            subscriber.write(stream_item, from_stream=self)
        self.items_written += 1

    def read_all(self, from_stream:'Stream'=None):
        for subscribed_to in self._subscribed_to:
            for stream_item in subscribed_to.read_all(from_stream=self):
                yield stream_item

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

