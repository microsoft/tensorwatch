# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# import matplotlib before anything else
# because of VS debugger issue for multiprocessing
# https://github.com/Microsoft/ptvsd/issues/1041
from .line_plot import LinePlot
from .image_plot import ImagePlot
from .histogram import Histogram
from .bar_plot import BarPlot

