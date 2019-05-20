# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .base_mpl_plot import BaseMplPlot
import matplotlib
import matplotlib.pyplot as plt
from .. import utils
import numpy as np
from ..lv_types import PointData
import ipywidgets as widgets

class LinePlot(BaseMplPlot):
    def init_stream_plot(self, stream_vis, 
            xtitle='', ytitle='', ztitle='', colormap=None, color=None, xrange=None, yrange=None, **stream_vis_args):
        stream_vis.xylabel_refs = [] # annotation references

        # add main subplot
        if len(self._stream_vises) == 0:
            stream_vis.ax = self.get_main_axis()
        else:
            stream_vis.ax = self.get_main_axis().twinx()

        stream_vis.cmap = plt.cm.get_cmap(name=colormap or 'Dark2')
        if color is None:
            if not self.is_3d:
                stream_vis.cmap((len(self._stream_vises)%stream_vis.cmap.N)/stream_vis.cmap.N) # pylint: disable=no-member

        # add default line in subplot
        stream_vis.color = color
        stream_vis.line = matplotlib.lines.Line2D([], [], 
            label=stream_vis.title or ytitle or str(stream_vis.index), color=color) #, linewidth=3
        if stream_vis.opacity is not None:
            stream_vis.line.set_alpha(stream_vis.opacity)
        stream_vis.ax.add_line(stream_vis.line)
        stream_vis.fill_between_col = None
        # if more than 2 y-axis then place additional outside
        if len(self._stream_vises) > 1:
            pos = (len(self._stream_vises)) * 30
            stream_vis.ax.spines['right'].set_position(('outward', pos))

        stream_vis.ax.set_xlabel(xtitle)
        stream_vis.ax.xaxis.label.set_style('italic')
        stream_vis.ax.set_ylabel(ytitle)
        stream_vis.ax.yaxis.label.set_color(color)
        stream_vis.ax.yaxis.label.set_style('italic')
        if self.is_3d:
            stream_vis.ax.set_zlabel(ztitle)
            stream_vis.ax.zaxis.label.set_style('italic')

        if xrange is not None:
            stream_vis.ax.set_xlim(*xrange)
        if yrange is not None:
            stream_vis.ax.set_ylim(*yrange)

    def clear_plot(self, stream_vis, clear_history):
        lines = stream_vis.ax.get_lines() 
        # if we need to keep history
        if stream_vis.history_len > 1:
            # make sure we have history len - 1 lines
            lines_keep = 0 if clear_history else stream_vis.history_len-1
            while len(lines) > lines_keep:
                lines.pop(0).remove()
            # dim old lines
            if stream_vis.dim_history and len(lines) > 0:
                alphas = np.linspace(0.05, 1, len(lines))
                for line, opacity in zip(lines, alphas):
                    line.set_alpha(opacity)
                    line.set_linewidth(1)
            # add new line
            line = matplotlib.lines.Line2D([], [], linewidth=3)
            stream_vis.ax.add_line(line)
        else: #clear current line
            lines[-1].set_data([], [])
        if stream_vis.fill_between_col is not None:
            stream_vis.fill_between_col.remove()
        # remove annotations
        for label_info in stream_vis.xylabel_refs:
            label_info.set_visible(False)
            label_info.remove()
        stream_vis.xylabel_refs.clear()

    def _show_stream_items(self, stream_vis, stream_items):
        """Paint the given stream_items in to visualizer. If visualizer is dirty then return False else True.
        """

        vals = self._extract_vals(stream_items)
        if not len(vals):
            return True # not dirty

        line = stream_vis.ax.get_lines()[-1]
        xdata, ydata = line.get_data()
        zdata, anndata, txtdata, clrdata = [], [], [], []
        lows, highs = [], [] # confidence interval

        unpacker = lambda a0=None,a1=None,a2=None,a3=None,a4=None,a5=None,a6=None,a7=None,*_:\
            (a0,a1,a2,a3,a4,a5,a6,a7)

        # add each value in trace data
        # each value is of the form:
        # 2D graphs:
        #   y
        #   x [, y [,low, [, high [,annotation [, text [, color]]]]]]
        #   y
        #   x [, y [, z, [,low, [, high [annotation [, text [, color]]]]]
        for val in vals:
            # set defaults
            x, y, z =  len(xdata), None, None
            low, high = None, None
            ann, txt, clr = None, None, None

            # if val turns out to be array-like, extract x,y
            val_l = utils.is_scaler_array(val)
            if val_l >= 0:
                if self.is_3d:
                    x, y, z, low, high, ann, txt, clr = unpacker(*val)
                else:
                    x, y, low, high, ann, txt, clr, _ = unpacker(*val)
            elif isinstance(val, PointData):
                x, y, z, low, high, ann, txt, clr = val.x, val.y, val.z, \
                    val.low, val.high, val.annotation, val.text, val.color
            else:
                y = val

            if ann is not None:
                ann = str(ann)
            if txt is not None:
                txt = str(txt)

            xdata.append(x)
            ydata.append(y)
            zdata.append(z)
            if low is not None:
                lows.append(low)
            if high is not None:
                highs.append(high)
            if (txt):
                txtdata.append(txt)
            if clr:
                clrdata.append(clr)
            if ann: #TODO: yref should be y2 for different y axis
                anndata.append(dict(x=x, y=y, xref='x', yref='y', text=ann, showarrow=False))

        line.set_data(xdata, ydata)
        if stream_vis.fill_between_col is not None:
            stream_vis.fill_between_col.remove()
        if len(lows) > 0 and len(highs) > 0:
            stream_vis.ax.fill_between(xdata, highs, lows, color=stream_vis.color, alpha=0.2)
        for ann in anndata:
            stream_vis.xylabel_refs.append(stream_vis.ax.text( \
                ann['x'], ann['y'], ann['text']))

        stream_vis.ax.relim()
        stream_vis.ax.autoscale_view()

        return False # dirty



   
