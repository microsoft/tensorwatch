# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys, time, threading, queue, functools
from typing import Any
from types import MethodType
from abc import ABCMeta, abstractmethod

from .lv_types import StreamVisInfo, StreamItem
from . import utils
from .stream import Stream


class VisBase(Stream, metaclass=ABCMeta):
    # these are expensive import so we attach to base class so derived class can use them
    from IPython import get_ipython, display
    import ipywidgets as widgets

    def __init__(self, widget, cell:widgets.Box, title:str, show_legend:bool, stream_name:str=None, console_debug:bool=False, **vis_args):
        super(VisBase, self).__init__(stream_name=stream_name, console_debug=console_debug)

        self.lock = threading.Lock()
        self._use_hbox = True
        #utils.set_default(vis_args, 'cell_width', '100%')

        self.widget = widget

        self.cell = cell or VisBase.widgets.HBox(layout=VisBase.widgets.Layout(\
            width=vis_args.get('cell_width', None))) if self._use_hbox else None
        if self._use_hbox:
            self.cell.children += (self.widget,)
        self._stream_vises = {}
        self.is_shown = cell is not None
        self.title = title
        self.last_ex = None
        self.layout_dirty = False
        self.q_last_processed = 0

    def subscribe(self, stream:Stream, title=None, clear_after_end=False, clear_after_each=False, 
            show:bool=False, history_len=1, dim_history=True, opacity=None, **stream_vis_args):
        # in this ovedrride we don't call base class method
        with self.lock:
            self.layout_dirty = True
        
            stream_vis = StreamVisInfo(stream, title, clear_after_end, 
                clear_after_each, history_len, dim_history, opacity,
                len(self._stream_vises), stream_vis_args, 0)
            stream_vis._clear_pending = False
            stream_vis._pending_items = queue.Queue()
            self._stream_vises[stream.stream_name] = stream_vis

            self._post_add_subscription(stream_vis, **stream_vis_args)

            super(VisBase, self).subscribe(stream)

            if show or (show is None and not self.is_shown):
                return self.show()

    def show(self, blocking:bool=False):
        self.is_shown = True
        if VisBase.get_ipython():
            if self._use_hbox:
                VisBase.display.display(self.cell) # this method doesn't need returns
                #return self.cell
            else:
                return self._show_widget_notebook()
        else:
            return self._show_widget_native(blocking)

    def write(self, val:Any, from_stream:'Stream'=None):
        stream_item = self.to_stream_item(val)

        stream_vis:StreamVisInfo = None
        if from_stream:
            stream_vis = self._stream_vises.get(from_stream.stream_name, None)

        if not stream_vis: # select the first one we have
            stream_vis = next(iter(self._stream_vises.values()))

        VisBase.write_stream_plot(self, stream_vis, stream_item)

        super(VisBase, self).write(stream_item)


    @staticmethod
    def write_stream_plot(vis, stream_vis:StreamVisInfo, stream_item:StreamItem):
        with vis.lock: # this could be from separate thread!
            #if stream_vis is None:
            #    utils.debug_log('stream_vis not specified in VisBase.write')
            #    stream_vis = next(iter(vis._stream_vises.values())) # use first as default
            utils.debug_log("Stream received: {}".format(stream_item.stream_name), verbosity=5)
            stream_vis._pending_items.put(stream_item)

        # if we accumulated enough of pending items then let's process them
        if vis._can_update_stream_plots():
            vis._update_stream_plots()

    def _extract_results(self, stream_vis):
        stream_items, clear_current, clear_history = [], False, False
        while not stream_vis._pending_items.empty():
            stream_item = stream_vis._pending_items.get()
            if stream_item.stream_reset:
                utils.debug_log("Stream reset", stream_item.stream_name)
                stream_items.clear() # no need to process these events
                clear_current, clear_history = True, True
            else:
                # check if there was an exception
                if stream_item.exception is not None:
                    #TODO: need better handling here?
                    print(stream_item.exception, file=sys.stderr)
                    raise stream_item.exception

                # state management for _clear_pending
                # if we need to clear plot before putting in data, do so
                if stream_vis._clear_pending:
                    stream_items.clear()
                    clear_current = True
                    stream_vis._clear_pending = False
                if stream_vis.clear_after_each or (stream_item.ended and stream_vis.clear_after_end):
                    stream_vis._clear_pending = True
                        
                stream_items.append(stream_item)

        return stream_items, clear_current, clear_history

    def _extract_vals(self, stream_items):
        vals = []
        for stream_item in stream_items:
            if stream_item.ended or stream_item.value is None:
                pass # no values to add
            else:
                if utils.is_array_like(stream_item.value, tuple_is_array=False):
                    vals.extend(stream_item.value)
                else:
                    vals.append(stream_item.value)
        return vals

    @abstractmethod
    def clear_plot(self, stream_vis, clear_history):
        """(for derived class) Clears the data in specified plot before new data is redrawn"""
        pass
    @abstractmethod
    def _show_stream_items(self, stream_vis, stream_items):
        """Paint the given stream_items in to visualizer. If visualizer is dirty then return False else True.
        """

        pass
    @abstractmethod
    def _post_add_subscription(self, stream_vis, **stream_vis_args):
        pass

    # typically we want to batch up items for performance
    def _can_update_stream_plots(self):
        return True

    @abstractmethod
    def _post_update_stream_plot(self, stream_vis):
        pass

    def _update_stream_plots(self):
        with self.lock:
            self.q_last_processed = time.time()
            for stream_vis in self._stream_vises.values():
                stream_items, clear_current, clear_history = self._extract_results(stream_vis)

                if clear_current:
                    self.clear_plot(stream_vis, clear_history)

                # if we have something to render
                dirty = not self._show_stream_items(stream_vis, stream_items)
                if dirty:
                    self._post_update_stream_plot(stream_vis)
                    stream_vis.last_update = time.time()

    @abstractmethod
    def _show_widget_native(self, blocking:bool):
        pass
    @abstractmethod
    def _show_widget_notebook(self):
        pass