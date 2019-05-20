* Fix cell size issue
* Refactor _plot* interface to accept all values, for ImagePlot only use last value
* Refactor ImagePlot for arbitrary number of images with alpha, cmap
* Change tw.open -> tw.create_viz
* Make sure streams have names as key, each data point has index
* Add tw.open_viz(stream_name, from_index)_
* Add persist=device_name option for streams
* Ability to use streams in standalone mode
* tw.create_viz on server side
* tw.log for server side
* experiment with IPC channel
* confusion matrix as in https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
* Speed up import
* Do linting
* live perf data
* NaN tracing
* PCA
* Remove error if MNIST notebook is on and we run fruits
* Remove 2nd image from fruits
* clear exisitng streams when starting client
* ImageData should accept numpy array or pillow or torch tensor 
* image plot getting refreshed at 12hz instead of 2 hz in MNIST
* image plot doesn't title
* Animated mesh/surface graph demo
* Move to h5 storage?
* Error envelop
* histogram
* new graph on end
* TF support
* generic visualizer -> Given obj and Box, paint in box
* visualize image and text with attention
* add confidence interval for plotly: https://plot.ly/python/continuous-error-bars/