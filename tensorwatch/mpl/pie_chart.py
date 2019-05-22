# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .base_mpl_plot import BaseMplPlot
from .. import image_utils
from .. import utils
import numpy as np

class PieChart(BaseMplPlot):
    def init_stream_plot(self, stream_vis, autopct=None, colormap=None, color=None, 
            shadow=None, **stream_vis_args):

        # add main subplot
        stream_vis.autopct, stream_vis.shadow = autopct, True if shadow is None else shadow
        stream_vis.ax = self.get_main_axis()
        stream_vis.series = []
        stream_vis.wedge_artists = [] # stores previously drawn bars

        stream_vis.cmap = image_utils.get_cmap(name=colormap or 'Dark2')
        if color is None:
            if not self.is_3d:
                color = stream_vis.cmap((len(self._stream_vises)%stream_vis.cmap.N)/stream_vis.cmap.N) # pylint: disable=no-member
        stream_vis.color = color

    def clear_artists(self, stream_vis):
        for artist in stream_vis.wedge_artists:
            artist.remove()
        stream_vis.wedge_artists.clear()

    def clear_plot(self, stream_vis, clear_history):
        stream_vis.series.clear()
        self.clear_artists(stream_vis)

    def _show_stream_items(self, stream_vis, stream_items):
        """Paint the given stream_items in to visualizer. If visualizer is dirty then return False else True.
        """

        vals = self._extract_vals(stream_items)
        if not len(vals):
            return True

        # make sure tuple has 4 elements
        unpacker = lambda a0=None,a1=None,a2=None,a3=None,*_:(a0,a1,a2,a3)
        stream_vis.series.extend([unpacker(*val) for val in vals])
        self.clear_artists(stream_vis)

        labels, sizes, colors, explode = \
            [t[0] for t in stream_vis.series], \
            [t[1] for t in stream_vis.series], \
            [(t[2] or stream_vis.cmap.colors[i % len(stream_vis.cmap.colors)]) \
                for i, t in enumerate(stream_vis.series)], \
            [t[3] or 0 for t in stream_vis.series],


        stream_vis.wedge_artists, *_ = stream_vis.ax.pie( \
                           sizes, explode=explode, labels=labels, colors=colors, 
                           autopct=stream_vis.autopct, shadow=stream_vis.shadow)

        return False

