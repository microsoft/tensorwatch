# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .base_mpl_plot import BaseMplPlot
from .. import utils
from .. import image_utils
import numpy as np
from itertools import groupby
import operator

class BarPlot(BaseMplPlot):
    def init_stream_plot(self, stream_vis, 
            xtitle='', ytitle='', ztitle='', colormap=None, color=None, 
            edge_color=None, linewidth=None, align=None, bar_width=None,
            opacity=None, **stream_vis_args):

        # add main subplot
        stream_vis.align, stream_vis.linewidth = align, (linewidth or 2)
        stream_vis.bar_width = bar_width or 1
        stream_vis.ax = self.get_main_axis()
        stream_vis.series = {}
        stream_vis.bars_artists = [] # stores previously drawn bars

        stream_vis.cmap =  image_utils.get_cmap(colormap or 'Set3')
        if color is None:
            if not self.is_3d:
                stream_vis.cmap((len(self._stream_vises)%stream_vis.cmap.N)/stream_vis.cmap.N) # pylint: disable=no-member
        stream_vis.color = color
        stream_vis.edge_color = 'black'
        stream_vis.opacity = opacity
        stream_vis.ax.set_xlabel(xtitle)
        stream_vis.ax.xaxis.label.set_style('italic')
        stream_vis.ax.set_ylabel(ytitle)
        stream_vis.ax.yaxis.label.set_color(color)
        stream_vis.ax.yaxis.label.set_style('italic')
        if self.is_3d:
            stream_vis.ax.set_zlabel(ztitle)
            stream_vis.ax.zaxis.label.set_style('italic')

    def is_show_grid(self): #override
        return False

    def clear_artists(self, stream_vis):
        for bar in stream_vis.bars_artists:
            bar.remove()
        stream_vis.bars_artists.clear()

    def clear_plot(self, stream_vis, clear_history):
        stream_vis.series.clear()
        self.clear_artists(stream_vis)

    def _val2tuple(val, x)->tuple:
        """Accept scaler val, (y,), (x, y), (label, y), (x, y, label), return (x, y, z, color, label)
        """
        if utils.is_array_like(val):
            unpacker = lambda a0=None,a1=None,a2=None,*_:(a0,a1,a2)
            t = unpacker(*val)
            if len(t) == 1: # (y,)
                t = (x, t[0], 0, None, None)
            elif len(t) == 2:
                t = (x, t[1], 0, None, t[0]) if isinstance(t[0], str) else t + (None,)
            elif len(t) == 3: # either (x, y, z) or (x, y, label)
                t = (t[0], t[1], None, t[2]) if isinstance(t[2], str) else t + (None, None)
            elif len(t) == 4: # we assume (x, y, z, color)
                t += (None,) 
            # else leave it alone
        else: # scaler
            t = (x, val, 0, None, None)

        return t


    def _show_stream_items(self, stream_vis, stream_items):
        """Paint the given stream_items in to visualizer. If visualizer is dirty then return False else True.
        """

        vals = self._extract_vals(stream_items)
        if not len(vals):
            return True

        if not self.is_3d:
            existing_len = len(stream_vis.series)
            for i,val in enumerate(vals):
                t = BarPlot._val2tuple(val, i+existing_len)
                stream_vis.series[t[0]] = t # merge x with previous items
            x, y, labels = [t[0] for t in stream_vis.series.values()], \
                           [t[1] for t in stream_vis.series.values()], \
                           [t[4] for t in stream_vis.series.values()]

            self.clear_artists(stream_vis) # remove previous bars
            bar_container = stream_vis.ax.bar(x, y, 
                               width=stream_vis.bar_width, 
                               tick_label = labels if any(l is not None for l in labels) else None,
                               color=stream_vis.color, edgecolor=stream_vis.edge_color, 
                               alpha=stream_vis.opacity, linewidth=stream_vis.linewidth)
        else:
            for i,val in enumerate(vals):
                t = BarPlot._val2tuple(val, None) # we should not use i paameter as 3d expects x,y,z
                z = t[2]
                if z not in stream_vis.series:
                    stream_vis.series[z] = []
                stream_vis.series[t[2]] += [t] # merge z with previous items
    
            # sort by z so we have consistent colors
            ts = sorted(stream_vis.series.items(), key=lambda g: g[0])

            for zi, (z, tg) in enumerate(ts):
                x, y, labels = [t[0] for t in tg], \
                               [t[1] for t in tg], \
                               [t[4] for t in tg]
                colors = stream_vis.color or stream_vis.cmap.colors
                color = colors[zi % len(colors)]

                self.clear_artists(stream_vis) # remove previous bars
                bar_container = stream_vis.ax.bar(x, y, zs=z, zdir='y',
                                   width=stream_vis.bar_width, 
                                   tick_label = labels if any(l is not None for l in labels) else None,
                                   color=color, 
                                   edgecolor=stream_vis.edge_color,
                                   alpha=stream_vis.opacity or 0.8, linewidth=stream_vis.linewidth)

        stream_vis.bars_artists = bar_container.patches

        #stream_vis.ax.relim()
        #stream_vis.ax.autoscale_view()

        return False

