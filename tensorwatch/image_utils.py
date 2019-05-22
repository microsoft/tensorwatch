# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import math
import time

def guess_image_dims(img):
    if len(img.shape) == 1:
        # assume 2D monochrome (MNIST)
        width = height = round(math.sqrt(img.shape[0]))
        if width*height != img.shape[0]:
            # assume 3 channels (CFAR, ImageNet)
            width = height = round(math.sqrt(img.shape[0] / 3))
            if width*height*3 != img.shape[0]:
                raise ValueError("Cannot guess image dimensions for linearized pixels")
            return (3, height, width)
        return (1, height, width)
    return img.shape

def to_imshow_array(img, width=None, height=None):
    # array from Pytorch has shape: [[channels,] height, width]
    # image needed for imshow needs: [height, width, channels]
    from PIL import Image

    if img is not None:
        if isinstance(img, Image.Image):
            img = np.array(img)
            if len(img.shape) >= 2:
                return img # img is already compatible to imshow

        # force max 3 dimensions
        if len(img.shape) > 3:
            # TODO allow config
            # select first one in batch
            img = img[0:1,:,:] 

        if len(img.shape) == 1: # linearized pixels typically used for MLPs
            if not(width and height):
                # pylint: disable=unused-variable
                channels, height, width = guess_image_dims(img)
            img = img.reshape((-1, height, width))

        if len(img.shape) == 3:
            if img.shape[0] == 1: # single channel images
                img = img.squeeze(0)
            else:
                img = np.swapaxes(img, 0, 2) # transpose H,W for imshow
                img = np.swapaxes(img, 0, 1)
        elif len(img.shape) == 2:
            img = np.swapaxes(img, 0, 1) # transpose H,W for imshow
        else: #zero dimensions
            img = None

    return img

#width_dim=1 for imshow, 2 for pytorch arrays
def stitch_horizontal(images, width_dim=1):
    return np.concatenate(images, axis=width_dim)

def _resize_image(img, size=None):
    if size is not None or (hasattr(img, 'shape') and len(img.shape) == 1):
        if size is None:
            # make guess for 1-dim tensors
            h = int(math.sqrt(img.shape[0]))
            w = int(img.shape[0] / h)
            size = h,w
        img = np.reshape(img, size)
    return img

def show_image(img, size=None, alpha=None, cmap=None, 
               img2=None, size2=None, alpha2=None, cmap2=None, ax=None):
    import matplotlib.pyplot as plt # delayed import due to matplotlib threading issue

    img =_resize_image(img, size)
    img2 =_resize_image(img2, size2)

    (ax or plt).imshow(img, alpha=alpha, cmap=cmap)

    if img2 is not None:
        (ax or plt).imshow(img2, alpha=alpha2, cmap=cmap2)

    return ax or plt.show()

# convert_mode param is mode: https://pillow.readthedocs.io/en/5.1.x/handbook/concepts.html#modes
# use convert_mode='RGB' to force 3 channels
def open_image(path, resize=None, resample=1, convert_mode=None): # Image.ANTIALIAS==1
    from PIL import Image

    img = Image.open(path)
    if resize is not None:
        img = img.resize(resize, resample)
    if convert_mode is not None:
        img = img.convert(convert_mode)
    return img

def img2pyt(img, add_batch_dim=True, resize=None):
    from torchvision import transforms # expensive function-level import

    ts = []
    if resize is not None:
        ts.append(transforms.RandomResizedCrop(resize))
    ts.append(transforms.ToTensor())
    img_pyt = transforms.Compose(ts)(img)
    if add_batch_dim:
        img_pyt.unsqueeze_(0)
    return img_pyt

def linear_to_2d(img, size=None):
    if size is not None or (hasattr(img, 'shape') and len(img.shape) == 1):
        if size is None:
            # make guess for 1-dim tensors
            h = int(math.sqrt(img.shape[0]))
            w = int(img.shape[0] / h)
            size = h,w
        img = np.reshape(img, size)
    return img

def stack_images(imgs):
    return np.hstack(imgs)

def plt_loop(count=None, sleep_time=1, plt_pause=0.01):
    import matplotlib.pyplot as plt # delayed import due to matplotlib threading issue

    #plt.ion()
    #plt.show(block=False)
    while((count is None or count > 0) and not plt.waitforbuttonpress(plt_pause)):
        #plt.draw()
        plt.pause(plt_pause)
        time.sleep(sleep_time)
        if count is not None:
            count = count - 1

def get_cmap(name:str):
    import matplotlib.pyplot as plt # delayed import due to matplotlib threading issue
    return plt.cm.get_cmap(name=name)

