# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .stream import Stream
import pickle, os
from typing import Any
from . import utils
import time

class FileStream(Stream):
    def __init__(self, for_write:bool, file_name:str, stream_name:str=None, console_debug:bool=False):
        super(FileStream, self).__init__(stream_name=stream_name or file_name, console_debug=console_debug)

        self._file = open(file_name, 'wb' if for_write else 'rb')
        self.file_name = file_name
        self.for_write = for_write
        utils.debug_log('FileStream started', os.path.realpath(self._file.name), verbosity=0)

    def close(self):
        if not self._file.closed:
            self._file.close()
            self._file = None
            utils.debug_log('FileStream is closed', os.path.realpath(self._file.name), verbosity=0)
        super(FileStream, self).close()

    def write(self, val:Any, from_stream:'Stream'=None):
        stream_item = self.to_stream_item(val)

        if self.for_write:
            pickle.dump(stream_item, self._file)
            self._file.flush()
            #os.fsync()
        super(FileStream, self).write(stream_item)

    def read_all(self, from_stream:'Stream'=None):
        if self.for_write:
            raise IOError('Cannot use read() call because FileSteam is opened with for_write=True')
        if self._file is not None:
            self._file.seek(0, 0) # we may filter this stream multiple times
            while not utils.is_eof(self._file):
                yield pickle.load(self._file)
        for item in super(FileStream, self).read_all():
            yield item

    def load(self, from_stream:'Stream'=None):
        if self.for_write:
            raise IOError('Cannot use load() call because FileSteam is opened with for_write=True')
        if self._file is not None:
            self._file.seek(0, 0) # we may filter this stream multiple times
            while not utils.is_eof(self._file):
                stream_item = pickle.load(self._file)
                self.write(stream_item)
        super(FileStream, self).load()

    def save(self, from_stream:'Stream'=None):
        if not self._file.closed:
            self._file.flush()
        super(FileStream, self).save(val)

