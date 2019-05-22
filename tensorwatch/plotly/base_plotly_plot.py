# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from ..vis_base import VisBase
import time
from abc import abstractmethod
from .. import utils


class BasePlotlyPlot(VisBase):
    def __init__(self, cell:VisBase.widgets.Box=None, title=None, show_legend:bool=None, is_3d:bool=False, 
                 stream_name:str=None, console_debug:bool=False, **vis_args):
        import plotly.graph_objs as go # function-level import as this takes long time
        super(BasePlotlyPlot, self).__init__(go.FigureWidget(), cell, title, show_legend, 
                                             stream_name=stream_name, console_debug=console_debug, **vis_args)

        self.is_3d = is_3d
        self.widget.layout.title = title
        self.widget.layout.showlegend = show_legend if show_legend is not None else True
      
    def _add_trace(self, stream_vis):
        stream_vis.trace_index = len(self.widget.data)
        trace = self._create_trace(stream_vis)
        if stream_vis.opacity is not None:
            trace.opacity = stream_vis.opacity
        self.widget.add_trace(trace)

    def _add_trace_with_history(self, stream_vis):
        # if history buffer isn't full
        if stream_vis.history_len > len(stream_vis.trace_history):
            self._add_trace(stream_vis)
            stream_vis.trace_history.append(len(self.widget.data)-1)
            stream_vis.cur_history_index = len(stream_vis.trace_history)-1
            #if stream_vis.cur_history_index:
            #    self.widget.data[trace_index].showlegend = False
        else:
            # rotate trace
            stream_vis.cur_history_index = (stream_vis.cur_history_index + 1) % stream_vis.history_len
            stream_vis.trace_index = stream_vis.trace_history[stream_vis.cur_history_index]
            self.clear_plot(stream_vis, False)
            self.widget.data[stream_vis.trace_index].opacity = stream_vis.opacity or 1

        cur_history_len = len(stream_vis.trace_history)
        if stream_vis.dim_history and cur_history_len > 1:
            max_opacity = stream_vis.opacity or 1
            min_alpha, max_alpha, dimmed_len = max_opacity*0.05, max_opacity*0.8, cur_history_len-1
            alphas = list(utils.frange(max_alpha, min_alpha, steps=dimmed_len))
            for i, thi in enumerate(range(stream_vis.cur_history_index+1, 
                                          stream_vis.cur_history_index+cur_history_len)):
                trace_index = stream_vis.trace_history[thi % cur_history_len]
                self.widget.data[trace_index].opacity = alphas[i]

    @staticmethod
    def get_pallet_color(i:int):
        import plotly # function-level import as this takes long time
        return plotly.colors.DEFAULT_PLOTLY_COLORS[i % len(plotly.colors.DEFAULT_PLOTLY_COLORS)]

    @staticmethod
    def _get_axis_common_props(title:str, axis_range:tuple):
        props = {'showline':True, 'showgrid': True, 
                       'showticklabels': True, 'ticks':'inside'}
        if title:
            props['title'] = title
        if axis_range:
            props['range'] = list(axis_range)
        return props

    def _can_update_stream_plots(self):
        return time.time() - self.q_last_processed > 0.5 # make configurable

    def _post_add_subscription(self, stream_vis, **stream_vis_args):
        stream_vis.trace_history, stream_vis.cur_history_index = [], None
        self._add_trace_with_history(stream_vis)
        self._setup_layout(stream_vis)

        if not self.widget.layout.title:
            self.widget.layout.title = stream_vis.title
        # TODO: better way for below?
        if stream_vis.history_len > 1:
            self.widget.layout.showlegend = False
                
    def _show_widget_native(self, blocking:bool):
        pass
        #TODO: save image, spawn browser?

    def _show_widget_notebook(self):
        #plotly.offline.iplot(self.widget)
        return None

    def _post_update_stream_plot(self, stream_vis):
        # not needed for plotly as FigureWidget stays upto date
        pass

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
    def _setup_layout(self, stream_vis):
        pass
    @abstractmethod
    def _create_trace(self, stream_vis):
        pass
