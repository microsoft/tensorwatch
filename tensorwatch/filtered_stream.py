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

    def write(self, val:Any, from_stream:'Stream'=None):
        result, is_valid = self.filter_expr(val) \
            if self.filter_expr is not None \
            else (val, True)

        if is_valid:
            return super(FilteredStream, self).write(result)
        # else ignore this call
            
