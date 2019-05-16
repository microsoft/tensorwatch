# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any
from .zmq_wrapper import ZmqWrapper
from .stream import Stream
from .lv_types import DefaultPorts, PublisherTopics
from . import utils

# on writes send data on ZMQ transport
class ZmqStream(Stream):
    def __init__(self, for_write:bool, port:int=0, topic=PublisherTopics.StreamItem, block_until_connected=True, 
                 stream_name:str=None, console_debug:bool=False):
        super(ZmqStream, self).__init__(stream_name=stream_name, console_debug=console_debug)

        self.for_write = for_write
        self._zmq = None

        self.topic = topic
        self._open(for_write, port, block_until_connected)
        utils.debug_log('ZmqStream started', verbosity=1)

    def _open(self, for_write:bool, port:int, block_until_connected:bool):
        if for_write:
            self._zmq = ZmqWrapper.Publication(port=DefaultPorts.PubSub+port,
                block_until_connected=block_until_connected)
        else:
            self._zmq = ZmqWrapper.Subscription(port=DefaultPorts.PubSub+port, 
                topic=self.topic, callback=self._on_subscription_item)

    def close(self):
        if not self.closed:
            self._zmq.close()
            self._zmq = None
            utils.debug_log('ZmqStream is closed', verbosity=1)
        super(ZmqStream, self).close()

    def _on_subscription_item(self, val:Any):
        utils.debug_log('Received subscription item', verbosity=5)
        self.write(val)

    def write(self, val:Any, from_stream:'Stream'=None, topic=None):
        stream_item = self.to_stream_item(val)

        if self.for_write:
            topic = topic or self.topic
            utils.debug_log('Sent subscription item', verbosity=5)
            self._zmq.send_obj(stream_item, topic)
        # else if this was opened for read then we have subscription and 
        # we shouldn't be calling send_obj
        super(ZmqStream, self).write(stream_item)
