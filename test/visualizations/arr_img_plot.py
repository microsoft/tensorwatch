import tensorwatch as tw
import numpy as np
import time
import torchvision.datasets as datasets

fruits_ds = datasets.ImageFolder(r'D:\datasets\fruits-360\Training')
mnist_ds = datasets.MNIST('../data', train=True, download=True)

images = [tw.ImageData(fruits_ds[i][0], title=str(i)) for i in range(5)] + \
         [tw.ImageData(mnist_ds[i][0], title=str(i)) for i in range(5)]

stream = tw.ArrayStream(images)

img_plot = tw.Visualizer(stream, vis_type='image', viz_img_scale=3)
img_plot.show()

tw.image_utils.plt_loop()