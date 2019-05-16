# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, Sequence, List, Any
from .zmq_stream import ZmqStream
from .file_stream import FileStream
from .stream import Stream
from .stream_union import StreamUnion
import uuid
from . import utils

class StreamFactory:
    r"""Allows to create shared stream such as file and ZMQ streams
    """

    def __init__(self)->None:
        self.closed = None
        self._streams:Dict[str, Stream] = None
        self._reset()

    def _reset(self):
        self._streams:Dict[str, Stream] = {}
        self.closed = False

    def close(self):
        if not self.closed:
            for stream in self._streams.values():
                stream.close()
            self._reset()
            self.closed = True

    def __enter__(self):
        return self
    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    def get_streams(self, stream_types:Sequence[str], for_write:bool=None)->List[Stream]:
        streams = [self._create_stream_by_string(stream_type, for_write) for stream_type in stream_types]
        return streams

    def get_combined_stream(self, stream_types:Sequence[str], for_write:bool=None)->Stream:
        streams = [self._create_stream_by_string(stream_type, for_write) for stream_type in stream_types]
        if len(streams) == 1:
            return self._streams[0]
        else:
            # we create new union of child but this is not necessory
            return StreamUnion(streams, for_write=for_write)

    def _get_stream_name(stream_type:str, stream_args:Any, for_write:bool)->str:
        return '{}:{}:{}'.format(stream_type, stream_args, for_write)

    def _create_stream_by_string(self, stream_spec:str, for_write:bool)->Stream:
        parts = stream_spec.split(':', 1) if stream_spec is not None else ['']
        stream_type = parts[0]
        stream_args = parts[1] if len(parts) > 1 else None

        utils.debug_log("Creating stream", (stream_spec, for_write))

        if stream_type == 'tcp':
            port = int(stream_args or 0)
            stream_name = StreamFactory._get_stream_name(stream_type, port, for_write)
            if stream_name not in self._streams:
                self._streams[stream_name] = ZmqStream(for_write=for_write, 
                    port=port, stream_name=stream_name, block_until_connected=False)
            # else we already have this stream
            return self._streams[stream_name]


        if stream_args is None: # file name specified without 'file:' prefix
            stream_args = stream_type
            stream_type = 'file'
        if len(stream_type) == 1: # windows drive letter
            stream_type = 'file'
            stream_args = stream_spec

        if stream_type == 'file':
            if stream_args is None:
                raise ValueError('File name must be specified for stream type "file"')
            stream_name = StreamFactory._get_stream_name(stream_type, stream_args, for_write)
            # each read only file stream should be separate stream or otheriwse sharing will
            # change seek positions
            if not for_write: 
                stream_name += ':' + str(uuid.uuid4())

                # if write file exist then flush it before read stream would read it
                write_stream_name = StreamFactory._get_stream_name(stream_type, stream_args, True)
                write_file_stream = self._streams.get(write_stream_name, None)
                if write_file_stream:
                    write_file_stream.save()
            if stream_name not in self._streams:
                self._streams[stream_name] = FileStream(for_write=for_write, 
                    file_name=stream_args, stream_name=stream_name)
            # else we already have this stream
            return self._streams[stream_name]

        if stream_type == '':
            return Stream()

        raise ValueError('stream_type "{}" has unknown type'.format(stream_type))


