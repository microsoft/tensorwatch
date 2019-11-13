import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from nbformat import v3, v4
import codecs
from os import linesep, path
import uuid
from . import utils
import re
from typing import List
from .lv_types import VisArgs

class NotebookMaker:
    def __init__(self, watcher, filename:str=None)->None:
        self.filename = filename or \
            (path.splitext(watcher.filename)[0] + '.ipynb' if watcher.filename else \
            'tensorwatch.ipynb')

        self.cells = []
        self._default_vis_args = VisArgs()

        watcher_args_str = NotebookMaker._get_vis_args(watcher)

        # create initial cell
        self.cells.append(new_code_cell(source=linesep.join( 
            ['%matplotlib notebook', 
             'import tensorwatch as tw',
             'client = tw.WatcherClient({})'.format(NotebookMaker._get_vis_args(watcher))])))

    def _get_vis_args(watcher)->str:
        args_strs = []
        for param, default_v in [('port', 0), ('filename', None)]:
            if hasattr(watcher, param):
                v = getattr(watcher, param)
                if v==default_v or (v is None and default_v is None):
                    continue
                args_strs.append("{}={}".format(param, NotebookMaker._val2str(v)))
        return ', '.join(args_strs)

    def _get_stream_identifier(prefix, event_name, stream_name, stream_index)->str:
        if not stream_name or utils.is_uuid4(stream_name):
            if event_name is not None and event_name != '':
                return '{}_{}_{}'.format(prefix, event_name, stream_index)
            else:
                return prefix + str(stream_index)
        else:
            return '{}{}_{}'.format(prefix, stream_index, utils.str2identifier(stream_name)[:8])

    def _val2str(v)->str:
        # TODO: shall we raise error if non str, bool, number (or its container) parameters?
        return str(v) if not isinstance(v, str) else "'{}'".format(v)

    def _add_vis_args_str(self, stream_info, param_strs:List[str])->None:
        default_args = self._default_vis_args.__dict__
        if not stream_info.req.vis_args:
            return
        for k, v in stream_info.req.vis_args.__dict__.items():
            if k in default_args:
                default_v = default_args[k]
                if (v is None and default_v is None) or (v==default_v):
                    continue # skip param if its value is not changed from default
                param_strs.append("{}={}".format(k, NotebookMaker._val2str(v)))

    def _get_stream_code(self, event_name, stream_name, stream_index, stream_info)->List[str]:
        lines = []

        stream_identifier = 's'+str(stream_index)
        lines.append("{} = client.open_stream(name='{}')".format(stream_identifier, stream_name))

        vis_identifier = 'v'+str(stream_index)
        vis_args_strs = ['stream={}'.format(stream_identifier)]
        self._add_vis_args_str(stream_info, vis_args_strs)
        lines.append("{} = tw.Visualizer({})".format(vis_identifier, ', '.join(vis_args_strs)))
        lines.append("{}.show()".format(vis_identifier))
        return lines

    def add_streams(self, event_stream_infos)->None:
        stream_index = 0
        for event_name, stream_infos in event_stream_infos.items(): # per event
            for stream_name, stream_info in stream_infos.items():
                lines = self._get_stream_code(event_name, stream_name, stream_index, stream_info)
                self.cells.append(new_code_cell(source=linesep.join(lines)))
                stream_index += 1

    def write(self):
        nb = new_notebook(cells=self.cells, metadata={'language': 'python',})
        with codecs.open(self.filename, encoding='utf-8', mode='w') as f:
            utils.debug_log('Notebook created', path.realpath(f.name), verbosity=0)
            nbformat.write(nb, f, 4)




