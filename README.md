# Welcome to TensorWatch

TensorWatch is a debugging and visualization tool designed for deep learning and reinforcement learning. It fully leverages Jupyter Notebook to show real time visualizations and offers unique capabilities to query the live training process without having to sprinkle logging statements all over. You can also use TensorWatch to build your own UIs and dashboards. In addition, TensorWatch leverages several excellent libraries for visualizing model graph, review model statistics, explain prediction and so on.

TensorWatch is under heavy development with a goal of providing a research platform for debugging machine learning in one easy to use, extensible and hackable package.

<img src="docs/images/teaser.gif" alt="TensorWatch in Jupyter Notebook" width="400"/>

## How to Get It

```
pip install tensorwatch
```

TensorWatch supports Python 3.x and is tested with PyTorch 0.4-1.x. Most features should also work with TensorFlow eager tensors.

## How to Use It

### Quick Start

Here's simple code that logs some metric every second:
```
import tensorwatch as tw
import time

w = tw.Watcher(filename='test.log')
s = w.create_stream(name='my_metric')
w.make_notebook()

for i in range(1000):
    s.write((i, i*i)) 
    time.sleep(1)
```
When you run this code you will notice a Jupyter Notebook file `test.ipynb` gets created in your script folder. From command prompt type `jupyter notebook` and click on `test.ipynb`. Click on Run button to see live line graph as values gets written in your script.

Here's the output in Jupyter Notebook:

<img src="docs/images/quick_start.gif" alt="TensorWatch in Jupyter Notebook" width="250"/>

Please [Tutorials](#tutorials) and [notebooks](https://github.com/microsoft/tensorwatch/tree/master/notebooks) for more information.

## Visualizations
In above example, the line graph is used as default visualization. However TensorWatch supports many other visualizations including histogram, pie charts, scatter charts, bar charts and 3D versions of many of these plots. You can simply log your data, specify the chart type you had like to visualize and let TensorWatch take care of the rest. You can also easily create custom visualizations specific to your data.

## Training within Jupyter Notebook
Many times you might prefer to do data analysis, ML training and testing from within Jupyter Notebook instead of from a separate script. TensorWatch can help you do sophisticated visualizations effortlessly for your code running within Jupyter Notebook.

## Querying the Process (Lazy Log Mode)
One of the unique capability TensorWatch offers is to be able to query the live running training process, retrieve the result of this query as stream and direct this stream to your preferred visualization - all of these without explicitly logging any data before hand! We call this new way of debugging *lazy logging mode*.

For example, below we show input and output image pairs randomly sampled during the training of an autoencoder on fruits dataset. These images were not logged, instead stream of these sampled images were returned on the fly as a response to the user query:

<img src="docs/images/fruits.gif" alt="TensorWatch in Jupyter Notebook" width="200"/>

### One Stop Shop for Debugging and Visualization

TensorWatch builds on several excellent libraries including [hiddenlayer](https://github.com/waleedka/hiddenlayer), [torchstat](https://github.com/Swall0w/torchstat), [Visual Attribution](https://github.com/yulongwang12/visual-attribution) to allow performing debugging and analysis activities in one consistent package and interface.

For example, you can view the model graph with tensor shapes with one liner:

<img src="docs/images/draw_model.png" alt="Model graph for Alexnet" width="400"/>

You can view statistics for different layers such as flops, number of parameters etc:

<img src="docs/images/model_stats.png" alt="Model statistics for Alexnet" width="600"/>

You can view dataset in lower dimensional space using techniques such as t-sne:

<img src="docs/images/tsne.gif" alt="t-sne visualization for MNIST" width="400"/>

### Prediction Explanations
We have a goal to provide various tools for explaining predictions over time to help debugging models. Currently we offer several explainers for convolutional networks including [Lime](https://github.com/marcotcr/lime). For example, below highlights the areas that causes Resnet50 model to make prediction for class 240:

<img src="docs/images/saliency.png" alt="t-sne visualization for MNIST" width="300"/>


## Tutorials

- [Simple Logging Tutorial](https://github.com/microsoft/tensorwatch/blob/master/notebooks/simple_logging.ipynb)

## Contribute

We would love your contributions, feedback and feature requests! Please [file a Github issue](https://github.com/microsoft/tensorwatch/issues/new) or send us a pull request. Review the [Microsoft Code of Conduct](https://opensource.microsoft.com/codeofconduct/) and [learn more](https://github.com/microsoft/tensorwatch/blob/master/CONTRIBUTING.md).

## Contact

Join the TensorWatch group on [Facebook](https://www.facebook.com/groups/378075159472803/) to stay up to date or ask any questions.

## License

This project is released under the MIT License. Please review the [License file](LICENSE.txt) for more details.
