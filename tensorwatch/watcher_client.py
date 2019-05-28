# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Dict, Sequence, List
from .zmq_wrapper import ZmqWrapper
from .lv_types import CliSrvReqTypes, ClientServerRequest, DefaultPorts
from .lv_types import VisArgs, PublisherTopics, ServerMgmtMsg, StreamCreateRequest
from .stream import Stream
from .zmq_mgmt_stream import ZmqMgmtStream
from . import utils
from .watcher_base import WatcherBase

class WatcherClient(WatcherBase):
    r"""Extends watcher to add methods so calls for create and delete stream can be sent to server.
    """
    def __init__(self, filename:str=None, port:int=0):
        super(WatcherClient, self).__init__()
        self.port = port
        self.filename = filename

        # define vars in __init__
        self._clisrv = None # client-server sockets allows to send create/del stream requests
        self._zmq_srvmgmt_sub = None
        self._file = None

        self._open()

    def _reset(self):
        self._clisrv = None
        self._zmq_srvmgmt_sub = None
        self._file = None
        utils.debug_log("WatcherClient reset", verbosity=1)
        super(WatcherClient, self)._reset()

    def _open(self):
        if self.port is not None:
            self._clisrv = ZmqWrapper.ClientServer(port=DefaultPorts.CliSrv+self.port, 
                is_server=False)
            # create subscription where we will receive server management events
            self._zmq_srvmgmt_sub = ZmqMgmtStream(clisrv=self._clisrv, for_write=False, port=self.port,
                stream_name='zmq_srvmgmt_sub:'+str(self.port)+':False')
    
    def close(self):
        if not self.closed:
            self._zmq_srvmgmt_sub.close()
            self._clisrv.close()
            utils.debug_log("WatcherClient is closed", verbosity=1)
        super(WatcherClient, self).close()

    def devices_or_default(self, devices:Sequence[str])->Sequence[str]: # overridden
        # TODO: this method is duplicated in Watcher and WatcherClient

        # make sure TCP port is attached to tcp device
        if devices is not None:
            return ['tcp:' + str(self.port) if device=='tcp' else device for device in devices]

        # if no devices specified then use our filename and tcp:port as default devices
        devices = []
        # first open file device because it may have older data 
        if self.filename is not None:
            devices.append('file:' + self.filename)
        if self.port is not None:
            devices.append('tcp:' + str(self.port))
        return devices

    # override to send request to server, instead of underlying WatcherBase base class
    def create_stream(self, name:str=None, devices:Sequence[str]=None, event_name:str='',
        expr=None, throttle:float=1, vis_args:VisArgs=None)->Stream: # overriden

        stream_req = StreamCreateRequest(stream_name=name, devices=self.devices_or_default(devices),
            event_name=event_name, expr=expr, throttle=throttle, vis_args=vis_args)

        self._zmq_srvmgmt_sub.add_stream_req(stream_req)

        if stream_req.devices is not None:
            stream = self.open_stream(name=stream_req.stream_name, devices=stream_req.devices)
        else: # we cannot return remote streams that are not backed by a device
            stream = None
        return stream

    # override to set devices default to tcp
    def open_stream(self, name:str=None, devices:Sequence[str]=None)->Stream: # overriden
        return super(WatcherClient, self).open_stream(name=name, devices=devices)


    # override to send request to server
    def del_stream(self, name:str) -> None:
        self._zmq_srvmgmt_sub.del_stream(name)

