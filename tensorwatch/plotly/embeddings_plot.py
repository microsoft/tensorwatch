# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .. import image_utils
import numpy as np
from .line_plot import LinePlot
import time
from .. import utils

class EmbeddingsPlot(LinePlot):
    def __init__(self, cell:LinePlot.widgets.Box=None, title=None, show_legend:bool=False, stream_name:str=None, console_debug:bool=False,
                  is_3d:bool=True, hover_images=None, hover_image_reshape=None, **vis_args):
        utils.set_default(vis_args, 'height', '8in')
        super(EmbeddingsPlot, self).__init__(cell, title, show_legend, 
                                             stream_name=stream_name, console_debug=console_debug, is_3d=is_3d, **vis_args)

        import matplotlib.pyplot as plt # delayed import due to matplotlib threading issue
        if hover_images is not None:
            plt.ioff()
            self.image_output = LinePlot.widgets.Output()
            self.image_figure = plt.figure(figsize=(2,2))
            self.image_ax = self.image_figure.add_subplot(111)
            self.cell.children += (self.image_output,)
            plt.ion()
        self.hover_images, self.hover_image_reshape = hover_images, hover_image_reshape
        self.last_ind, self.last_ind_time = -1, 0

    def hover_fn(self, trace, points, state): # pylint: disable=unused-argument
        if not points:
            return
        ind = points.point_inds[0]
        if ind == self.last_ind or ind > len(self.hover_images) or ind < 0:
            return

        if self.last_ind == -1:
            self.last_ind, self.last_ind_time = ind, time.time()
        else:
            elapsed = time.time() - self.last_ind_time
            if elapsed  < 0.3:
                self.last_ind, self.last_ind_time = ind, time.time()
                if elapsed  < 1:
                    return
                # else too much time since update
            # else we have stable ind
        
        import matplotlib.pyplot as plt # delayed import due to matplotlib threading issue
        with self.image_output:
            plt.ioff()

            if self.hover_image_reshape:
                img = np.reshape(self.hover_images[ind], self.hover_image_reshape)
            else:
                img = self.hover_images[ind]
            if img is not None:
                LinePlot.display.clear_output(wait=True)    
                self.image_ax.imshow(img)
            LinePlot.display.display(self.image_figure)
            plt.ion()

        return None

    def _create_trace(self, stream_vis):
        stream_vis.stream_vis_args.clear() #TODO remove this
        utils.set_default(stream_vis.stream_vis_args, 'draw_line', False)
        utils.set_default(stream_vis.stream_vis_args, 'draw_marker', True)
        utils.set_default(stream_vis.stream_vis_args, 'draw_marker_text', True)
        utils.set_default(stream_vis.stream_vis_args, 'hoverinfo', 'text')
        utils.set_default(stream_vis.stream_vis_args, 'marker', {})

        marker = stream_vis.stream_vis_args['marker']
        utils.set_default(marker, 'size', 6)
        utils.set_default(marker, 'colorscale', 'Jet')
        utils.set_default(marker, 'showscale', False)
        utils.set_default(marker, 'opacity', 0.8)

        return super(EmbeddingsPlot, self)._create_trace(stream_vis)

    def subscribe(self, stream, **stream_vis_args):
        super(EmbeddingsPlot, self).subscribe(stream)
        stream_vis = self._stream_vises[stream.stream_name]
        if stream_vis.index == 0 and self.hover_images is not None:
            self.widget.data[stream_vis.trace_index].on_hover(self.hover_fn)