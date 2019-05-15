# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .stream import Stream
from typing import Iterator

class StreamUnion(Stream):
    def __init__(self, child_streams:Iterator[Stream], for_write:bool, stream_name:str=None, console_debug:bool=False) -> None:
        super(StreamUnion, self).__init__(stream_name=stream_name, console_debug=console_debug)

        # save references, child streams does away only if parent goes away
        self.child_streams = child_streams

        # when someone does write to us, we write to all our listeners
        if for_write:
            for child_stream in child_streams:
                child_stream.subscribe(self)
        else:
            # union of all child streams
            for child_stream in child_streams:
                self.subscribe(child_stream)