# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .base_mpl_plot import BaseMplPlot
from .. import image_utils
from .. import utils
import numpy as np

class Histogram(BaseMplPlot):
    def init_stream_plot(self, stream_vis, 
            xtitle='', ytitle='', ztitle='', colormap=None, color=None, 
            bins=None, normed=None, histtype='bar', edge_color=None, linewidth=None,
            opacity=None, **stream_vis_args):

        # add main subplot
        stream_vis.bins, stream_vis.normed, stream_vis.linewidth = bins, normed, (linewidth or 2)
        stream_vis.ax = self.get_main_axis()
        stream_vis.series = []
        stream_vis.bars_artists = [] # stores previously drawn bars

        stream_vis.cmap = image_utils.get_cmap(name=colormap or 'Dark2')
        if color is None:
            if not self.is_3d:
                color = stream_vis.cmap((len(self._stream_vises)%stream_vis.cmap.N)/stream_vis.cmap.N) # pylint: disable=no-member
        stream_vis.color = color
        stream_vis.edge_color = 'black'
        stream_vis.histtype = histtype
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

    def _show_stream_items(self, stream_vis, stream_items):
        """Paint the given stream_items in to visualizer. If visualizer is dirty then return False else True.
        """

        vals = self._extract_vals(stream_items)
        if not len(vals):
            return True

        stream_vis.series += vals
        self.clear_artists(stream_vis)
        n, bins, stream_vis.bars_artists = stream_vis.ax.hist(stream_vis.series, bins=stream_vis.bins,
                           normed=stream_vis.normed, color=stream_vis.color, edgecolor=stream_vis.edge_color, 
                           histtype=stream_vis.histtype, alpha=stream_vis.opacity, 
                           linewidth=stream_vis.linewidth)

        stream_vis.ax.set_xticks(bins)

        #stream_vis.ax.relim()
        #stream_vis.ax.autoscale_view()

        return False
