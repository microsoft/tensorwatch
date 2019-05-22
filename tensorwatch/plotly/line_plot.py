# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .base_plotly_plot import BasePlotlyPlot
from ..lv_types import EventData
from .. import utils

class LinePlot(BasePlotlyPlot):
    def _setup_layout(self, stream_vis):
        # handle multiple y axis
        yaxis = 'yaxis' + (str(stream_vis.index + 1) if stream_vis.separate_yaxis else '')

        xaxis = 'xaxis' + str(stream_vis.index+1)
        axis_props = BasePlotlyPlot._get_axis_common_props(stream_vis.xtitle, stream_vis.xrange)
        #axis_props['rangeslider'] = dict(visible = True)
        self.widget.layout[xaxis] = axis_props

        # handle multiple Y-Axis plots
        color = self.widget.data[stream_vis.trace_index].line.color
        yaxis = 'yaxis' + (str(stream_vis.index + 1) if stream_vis.separate_yaxis else '')
        axis_props = BasePlotlyPlot._get_axis_common_props(stream_vis.ytitle, stream_vis.yrange)
        axis_props['linecolor'] = color
        axis_props['tickfont']=axis_props['titlefont'] = dict(color=color)
        if stream_vis.index > 0 and stream_vis.separate_yaxis:
            axis_props['overlaying'] = 'y'
            axis_props['side'] = 'right'
            if stream_vis.index > 1:
                self.widget.layout.xaxis = dict(domain=[0, 1 - 0.085*(stream_vis.index-1)])
                axis_props['anchor'] = 'free'
                axis_props['position'] = 1 - 0.085*(stream_vis.index-2)
        self.widget.layout[yaxis] = axis_props

        if self.is_3d:
            zaxis = 'zaxis' #+ str(stream_vis.index+1)
            axis_props = BasePlotlyPlot._get_axis_common_props(stream_vis.ztitle, stream_vis.zrange)
            self.widget.layout.scene[zaxis] = axis_props
            self.widget.layout.margin = dict(l=0, r=0, b=0, t=0)
            self.widget.layout.hoverdistance = 1

    def _create_2d_trace(self, stream_vis, mode, hoverinfo, marker, line):
        import plotly.graph_objs as go # function-level import as this takes long time

        yaxis = 'y' + (str(stream_vis.index + 1) if stream_vis.separate_yaxis else '')

        trace = go.Scatter(x=[], y=[], mode=mode, name=stream_vis.title or stream_vis.ytitle, yaxis=yaxis, hoverinfo=hoverinfo,
                           line=line, marker=marker)
        return trace

    def _create_3d_trace(self, stream_vis, mode, hoverinfo, marker, line):
        import plotly.graph_objs as go # function-level import as this takes long time

        trace = go.Scatter3d(x=[], y=[], z=[], mode=mode, name=stream_vis.title or stream_vis.ytitle, hoverinfo=hoverinfo,
                           line=line, marker=marker)
        return trace


    def _create_trace(self, stream_vis):
        stream_vis.separate_yaxis = stream_vis.stream_vis_args.get('separate_yaxis', True)
        stream_vis.xtitle = stream_vis.stream_vis_args.get('xtitle',None)
        stream_vis.ytitle = stream_vis.stream_vis_args.get('ytitle',None)
        stream_vis.ztitle = stream_vis.stream_vis_args.get('ztitle',None)
        stream_vis.color = stream_vis.stream_vis_args.get('color',None)
        stream_vis.xrange = stream_vis.stream_vis_args.get('xrange',None)
        stream_vis.yrange = stream_vis.stream_vis_args.get('yrange',None)
        stream_vis.zrange = stream_vis.stream_vis_args.get('zrange',None)
        draw_line = stream_vis.stream_vis_args.get('draw_line',True)
        draw_marker = stream_vis.stream_vis_args.get('draw_marker',True)
        draw_marker_text = stream_vis.stream_vis_args.get('draw_marker_text',False)
        hoverinfo = stream_vis.stream_vis_args.get('hoverinfo',None)
        marker = stream_vis.stream_vis_args.get('marker',{})
        line = stream_vis.stream_vis_args.get('line',{})
        utils.set_default(line, 'color', stream_vis.color or BasePlotlyPlot.get_pallet_color(stream_vis.index))
        
        mode = 'lines' if draw_line else ''
        if draw_marker:
            mode = ('' if mode=='' else mode+'+') + 'markers'
        if draw_marker_text:
            mode = ('' if mode=='' else mode+'+') + 'text'

        if self.is_3d:
            return self._create_3d_trace(stream_vis, mode, hoverinfo, marker, line)  
        else:
            return self._create_2d_trace(stream_vis, mode, hoverinfo, marker, line)  

    def _show_stream_items(self, stream_vis, stream_items):
        """Paint the given stream_items in to visualizer. If visualizer is dirty then return False else True.
        """

        vals = self._extract_vals(stream_items)
        if not len(vals):
            return True # not dirty

        # get trace data
        trace = self.widget.data[stream_vis.trace_index]
        xdata, ydata, zdata, anndata, txtdata, clrdata = list(trace.x), list(trace.y), [], [], [], []
        lows, highs = [], [] # confidence interval

        if self.is_3d:
            zdata = list(trace.z)

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
            if txt is not None:
                txtdata.append(txt)
            if clr is not None:
                clrdata.append(clr)
            if ann: #TODO: yref should be y2 for different y axis
                anndata.append(dict(x=x, y=y, xref='x', yref='y', text=ann, showarrow=False))

        self.widget.data[stream_vis.trace_index].x = xdata
        self.widget.data[stream_vis.trace_index].y = ydata   
        if self.is_3d:
            self.widget.data[stream_vis.trace_index].z = zdata

        # add text
        if len(txtdata):
            exisitng = self.widget.data[stream_vis.trace_index].text
            exisitng = list(exisitng) if utils.is_array_like(exisitng) else []
            exisitng += txtdata
            self.widget.data[stream_vis.trace_index].text = exisitng

        # add annotation
        if len(anndata):
            existing = list(self.widget.layout.annotations)
            existing += anndata
            self.widget.layout.annotations = existing

        # add color
        if len(clrdata):
            exisitng = self.widget.data[stream_vis.trace_index].marker.color
            exisitng = list(exisitng) if utils.is_array_like(exisitng) else []
            exisitng += clrdata
            self.widget.data[stream_vis.trace_index].marker.color = exisitng

        return False # dirty

    def clear_plot(self, stream_vis, clear_history):
        traces = range(len(stream_vis.trace_history)) if clear_history else (stream_vis.trace_index,)
        for i in traces:
            stream_vis.trace_index = i

            self.widget.data[stream_vis.trace_index].x = []
            self.widget.data[stream_vis.trace_index].y = []   
            if self.is_3d:
                self.widget.data[stream_vis.trace_index].z = []
            self.widget.data[stream_vis.trace_index].text = ""
            # TODO: avoid removing annotations for other streams
            self.widget.layout.annotations = []
