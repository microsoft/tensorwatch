# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .stream import Stream
from .vis_base import VisBase
from . import mpl
from . import plotly

class Visualizer:
    """Constructs visualizer for specified vis_type.

       NOTE: If you modify arguments here then also sync VisArgs contructor.
    """
    def __init__(self, stream:Stream, vis_type:str=None, host:'Visualizer'=None, 
            cell:'Visualizer'=None, title:str=None, 
            clear_after_end=False, clear_after_each=False, history_len=1, dim_history=True, opacity=None,

            rows=2, cols=5, img_width=None, img_height=None, img_channels=None,
            colormap=None, viz_img_scale=None,

            # these image params are for hover on point for t-sne
            hover_images=None, hover_image_reshape=None, cell_width:str=None, cell_height:str=None, 

            only_summary=False, separate_yaxis=True, xtitle=None, ytitle=None, ztitle=None, color=None,
            xrange=None, yrange=None, zrange=None, draw_line=True, draw_marker=False,

            # histogram
            bins=None, normed=None, histtype='bar', edge_color=None, linewidth=None, bar_width=None,

            # pie chart
            autopct=None, shadow=None, 

            vis_args={}, stream_vis_args={})->None:

        cell = cell._host_base.cell if cell is not None else None

        if host:
            self._host_base = host._host_base
        else:
            self._host_base = self._get_vis_base(vis_type, cell, title, hover_images=hover_images, hover_image_reshape=hover_image_reshape, 
                                   cell_width=cell_width, cell_height=cell_height,
                                   **vis_args)

        self._host_base.subscribe(stream, show=False, clear_after_end=clear_after_end, clear_after_each=clear_after_each,
            history_len=history_len, dim_history=dim_history, opacity=opacity, 
            only_summary=only_summary if vis_type is None or 'summary' != vis_type else True,
            separate_yaxis=separate_yaxis, xtitle=xtitle, ytitle=ytitle, ztitle=ztitle, color=color,
            xrange=xrange, yrange=yrange, zrange=zrange, 
            draw_line=draw_line if vis_type is not None and 'scatter' in vis_type else True, 
            draw_marker=draw_marker, 
            rows=rows, cols=cols, img_width=img_width, img_height=img_height, img_channels=img_channels,
            colormap=colormap, viz_img_scale=viz_img_scale,
            bins=bins, normed=normed, histtype=histtype, edge_color=edge_color, linewidth=linewidth, bar_width = bar_width,
            autopct=autopct, shadow=shadow,
            **stream_vis_args)

        stream.load()

    def show(self):
        return self._host_base.show()

    def _get_vis_base(self, vis_type, cell:VisBase.widgets.Box, title, hover_images=None, hover_image_reshape=None, cell_width=None, cell_height=None, **vis_args)->VisBase:
        if vis_type is None or vis_type in ['line', 
                        'mpl-line', 'mpl-line3d', 'mpl-scatter3d', 'mpl-scatter']:
            return mpl.line_plot.LinePlot(cell=cell, title=title, cell_width=cell_width, cell_height=cell_height, 
                                is_3d=vis_type is not None and vis_type.endswith('3d'), **vis_args)
        if vis_type in ['image', 'mpl-image']:
            return mpl.image_plot.ImagePlot(cell=cell, title=title, cell_width=cell_width, cell_height=cell_height, **vis_args)
        if vis_type in ['bar', 'bar3d']:
            return mpl.bar_plot.BarPlot(cell=cell, title=title, cell_width=cell_width, cell_height=cell_height, 
                               is_3d=vis_type.endswith('3d'), **vis_args)
        if vis_type in ['histogram']:
            return mpl.histogram.Histogram(cell=cell, title=title, cell_width=cell_width, cell_height=cell_height, **vis_args)
        if vis_type in ['pie']:
            return mpl.pie_chart.PieChart(cell=cell, title=title, cell_width=cell_width, cell_height=cell_height, **vis_args)
        if vis_type in ['text', 'summary']:
            from .text_vis import TextVis
            return TextVis(cell=cell, title=title, cell_width=cell_width, cell_height=cell_height, **vis_args)
        if vis_type in ['line3d', 'scatter', 'scatter3d',
                        'plotly-line', 'plotly-line3d', 'plotly-scatter', 'plotly-scatter3d', 'mesh3d']:
            return plotly.line_plot.LinePlot(cell=cell, title=title, cell_width=cell_width, cell_height=cell_height, 
                                   is_3d=vis_type.endswith('3d'), **vis_args)
        if vis_type in ['tsne', 'embeddings', 'tsne2d', 'embeddings2d']:
            return plotly.embeddings_plot.EmbeddingsPlot(cell=cell, title=title, cell_width=cell_width, cell_height=cell_height, 
                                         is_3d='2d' not in vis_type, 
                                         hover_images=hover_images, hover_image_reshape=hover_image_reshape, **vis_args)
        else:
            raise ValueError('Render vis_type parameter has invalid value: "{}"'.format(vis_type))
