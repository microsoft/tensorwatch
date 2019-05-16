# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .stream import Stream
from typing import Callable, Any

class FilteredStream(Stream):
    def __init__(self, source_stream:Stream, filter_expr:Callable, stream_name:str=None, 
                 console_debug:bool=False)->None:

        stream_name = stream_name or '{}|{}'.format(source_stream.stream_name, str(filter_expr))
        super(FilteredStream, self).__init__(stream_name=stream_name, console_debug=console_debug)
        self.subscribe(source_stream)
        self.filter_expr = filter_expr

    def _filter(self, stream_item):
        return self.filter_expr(stream_item) \
            if self.filter_expr is not None \
            else (stream_item, True)

    def write(self, val:Any, from_stream:'Stream'=None):
        stream_item = self.to_stream_item(val)

        result, is_valid = self._filter(stream_item)
        if is_valid:
            return super(FilteredStream, self).write(result)
        # else ignore this call

    def read_all(self, from_stream:'Stream'=None): #override->replacement
        for subscribed_to in self._subscribed_to:
            for stream_item in subscribed_to.read_all(from_stream=self):
                result, is_valid = self._filter(stream_item)
                if is_valid:
                    yield stream_item
            
