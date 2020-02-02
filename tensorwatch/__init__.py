# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Iterable, Sequence, Union

from .watcher_client import WatcherClient
from .watcher import Watcher
from .watcher_base import WatcherBase

from .text_vis import TextVis
from .plotly.embeddings_plot import EmbeddingsPlot
from .mpl.line_plot import LinePlot
from .mpl.image_plot import ImagePlot
from .mpl.histogram import Histogram
from .mpl.bar_plot import BarPlot
from .mpl.pie_chart import PieChart
from .visualizer import Visualizer

from .stream import Stream
from .array_stream import ArrayStream
from .lv_types import PointData, ImageData, VisArgs, StreamItem, PredictionResult
from . import utils

###### Import methods for tw namespace #########
#from .receptive_field.rf_utils import plot_receptive_field, plot_grads_at
from .embeddings.tsne_utils import get_tsne_components
from .model_graph.torchstat_utils import model_stats, ModelStats
from .image_utils import show_image, open_image, img2pyt, linear_to_2d, plt_loop
from .data_utils import pyt_ds2list, sample_by_class, col2array, search_similar



def draw_model(model, input_shape=None, orientation='TB', png_filename=None): #orientation = 'LR' for landscpe
    from .model_graph.hiddenlayer import pytorch_draw_model
    g = pytorch_draw_model.draw_graph(model, input_shape)
    return g


