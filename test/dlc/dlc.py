import tensorwatch as tw
import time
import math
from tensorwatch import utils

utils.set_debug_verbosity(4)

def dlc_show_rand_outputs():
    cli = cli_train = tw.WatcherClient()
    imgs = cli.create_stream(event_name='batch', 
        expr="top(l, out_xform=pyt_img_img_out_xform, group_key=lambda x:'', topk=10, order='rnd')", 
        throttle=1)
    img_plot = tw.ImagePlot()
    img_plot.show(imgs, img_width=39, img_height=69, viz_img_scale=10)

    utils.wait_key()

def img2img_rnd():
    cli_train = tw.WatcherClient()
    cli = tw.WatcherClient()

    imgs = cli_train.create_stream(event_name='batch', 
        expr="top(l, out_xform=pyt_img_img_out_xform, group_key=lambda x:'', topk=2, order='rnd')", 
        throttle=1)
    img_plot = tw.ImagePlot()
    img_plot.show(imgs, img_width=100, img_height=100, viz_img_scale=3, cols=1)

    utils.wait_key()

dlc_show_rand_outputs()
img2img_rnd() 