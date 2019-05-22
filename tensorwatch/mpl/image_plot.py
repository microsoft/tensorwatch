# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .base_mpl_plot import BaseMplPlot
from .. import utils, image_utils
import numpy as np

#from IPython import get_ipython

class ImagePlot(BaseMplPlot):
    def init_stream_plot(self, stream_vis, 
            rows=2, cols=5, img_width=None, img_height=None, img_channels=None,
            colormap=None, viz_img_scale=None, **stream_vis_args):
        stream_vis.rows, stream_vis.cols = rows, cols
        stream_vis.img_channels, stream_vis.colormap = img_channels, colormap
        stream_vis.img_width, stream_vis.img_height = img_width, img_height
        stream_vis.viz_img_scale = viz_img_scale
        # subplots holding each image
        stream_vis.axs = [[None for _ in range(cols)] for _ in range(rows)] 
        # axis image
        stream_vis.ax_imgs = [[None for _ in range(cols)] for _ in range(rows)] 

    def clear_plot(self, stream_vis, clear_history):
        for row in range(stream_vis.rows):
            for col in range(stream_vis.cols):
                img = stream_vis.ax_imgs[row][col]
                if img:
                    x, y = img.get_size()
                    img.set_data(np.zeros((x, y)))

    def _show_stream_items(self, stream_vis, stream_items):
        """Paint the given stream_items in to visualizer. If visualizer is dirty then return False else True.
        """

        # as we repaint each image plot, select last if multiple events were pending
        stream_item = None
        for er in reversed(stream_items):
            if not(er.ended or er.value is None):
                stream_item = er
                break
        if stream_item is None:
            return True

        row, col, i = 0, 0, 0
        dirty = False
        # stream_item.value is expected to be ImagePlotItems
        for image_list in stream_item.value:
            # convert to imshow compatible, stitch images
            images = [image_utils.to_imshow_array(img, stream_vis.img_width, stream_vis.img_height) \
                for img in image_list.images if img is not None]
            img_viz = image_utils.stitch_horizontal(images, width_dim=1)

            # resize if requested
            if stream_vis.viz_img_scale is not None:
                import skimage.transform # expensive import done on demand

                if isinstance(img_viz, np.ndarray) and np.issubdtype(img_viz.dtype, np.floating):
                    img_viz = img_viz.clip(-1, 1) # some MNIST images have out of range values causing exception in sklearn
                img_viz = skimage.transform.rescale(img_viz, 
                    (stream_vis.viz_img_scale, stream_vis.viz_img_scale), mode='reflect', preserve_range=False)

            # create subplot if it doesn't exist
            ax = stream_vis.axs[row][col]
            if ax is None:
                ax = stream_vis.axs[row][col] = \
                    self.figure.add_subplot(stream_vis.rows, stream_vis.cols, i+1)
                ax.set_xticks([])
                ax.set_yticks([])  

            cmap = image_list.cmap or ('Greys' if stream_vis.colormap is None and \
                len(img_viz.shape) == 2 else stream_vis.colormap)

            stream_vis.ax_imgs[row][col] = ax.imshow(img_viz, interpolation="none", cmap=cmap, alpha=image_list.alpha)
            dirty = True

            # set title
            title = image_list.title
            if len(title) > 12: #wordwrap if too long
                title = utils.wrap_string(title) if len(title) > 24 else title
                fontsize = 8
            else:
                fontsize = 12
            ax.set_title(title, fontsize=fontsize) #'fontweight': 'light'

            #ax.autoscale_view() # not needed
            col = col + 1
            if col >= stream_vis.cols:
                col = 0
                row = row + 1
                if row >= stream_vis.rows:
                    break
            i += 1

        return not dirty

    
    def has_legend(self):
        return self.show_legend or False