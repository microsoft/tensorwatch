import tensorwatch as tw
import time
import math
from tensorwatch import utils

utils.set_debug_verbosity(4)

def img_in_class():
    cli_train = tw.WatcherClient()

    imgs = cli_train.create_stream(event_name='batch',
        expr="topk_all(l, batch_vals=lambda b: (b.batch.loss_all, (b.batch.input, b.batch.output), b.batch.target), \
            out_f=image_class_outf, order='dsc')", throttle=1)
    img_plot = tw.ImagePlot()
    img_plot.subscribe(imgs, viz_img_scale=3)
    img_plot.show()

    tw.image_utils.plt_loop()

def show_find_lr():
    cli_train = tw.WatcherClient()
    plot = tw.LinePlot()
    
    train_batch_loss = cli_train.create_stream(event_name='batch', 
        expr='lambda d:(d.tt.scheduler.get_lr()[0], d.metrics.batch_loss)')
    plot.subscribe(train_batch_loss, xtitle='Epoch', ytitle='Loss')
    
    utils.wait_key()

def plot_grads_plotly():
    train_cli = tw.WatcherClient()
    grads = train_cli.create_stream(event_name='batch', 
        expr='lambda d:grads_abs_mean(d.model)', throttle=1)
    p = tw.plotly.line_plot.LinePlot('Demo')
    p.subscribe(grads, xtitle='Layer', ytitle='Gradients', history_len=30, new_on_eval=True)
    utils.wait_key()


def plot_grads():
    train_cli = tw.WatcherClient()

    grads = train_cli.create_stream(event_name='batch', 
        expr='lambda d:grads_abs_mean(d.model)', throttle=1)
    grad_plot = tw.LinePlot()
    grad_plot.subscribe(grads, xtitle='Layer', ytitle='Gradients', clear_after_each=1, history_len=40, dim_history=True)
    grad_plot.show()

    tw.plt_loop()

def plot_weight():
    train_cli = tw.WatcherClient()

    params = train_cli.create_stream(event_name='batch', 
        expr='lambda d:weights_abs_mean(d.model)', throttle=1)
    params_plot = tw.LinePlot()
    params_plot.subscribe(params, xtitle='Layer', ytitle='avg |params|', clear_after_each=1, history_len=40, dim_history=True)
    params_plot.show()

    tw.plt_loop()

def epoch_stats():
    train_cli = tw.WatcherClient(port=0)
    test_cli = tw.WatcherClient(port=1)

    plot = tw.LinePlot()

    train_loss = train_cli.create_stream(event_name="epoch", 
        expr='lambda v:(v.metrics.epoch_index, v.metrics.epoch_loss)')
    plot.subscribe(train_loss, xtitle='Epoch', ytitle='Train Loss')
    
    test_acc = test_cli.create_stream(event_name="epoch", 
        expr='lambda v:(v.metrics.epoch_index, v.metrics.epoch_accuracy)')
    plot.subscribe(test_acc, xtitle='Epoch', ytitle='Test Accuracy', ylim=(0,1))

    plot.show()
    tw.plt_loop()


def batch_stats():
    train_cli = tw.WatcherClient()
    stream = train_cli.create_stream(event_name="batch", 
        expr='lambda v:(v.metrics.epochf, v.metrics.batch_loss)', throttle=0.75)

    train_loss = tw.Visualizer(stream, clear_after_end=False, vis_type='mpl-line',
        xtitle='Epoch', ytitle='Train Loss')
    
    #train_acc = tw.Visualizer('lambda v:(v.metrics.epochf, v.metrics.epoch_loss)', event_name="batch",
    #                     xtitle='Epoch', ytitle='Train Accuracy', clear_after_end=False, yrange=(0,1), 
    #                     vis=train_loss, vis_type='mpl-line')

    train_loss.show()
    tw.plt_loop()

def text_stats():
    train_cli = tw.WatcherClient()
    stream = train_cli.create_stream(event_name="batch", 
        expr='lambda d:(d.metrics.epoch_index, d.metrics.batch_loss)')

    trl = tw.Visualizer(stream, vis_type='text')
    trl.show()
    input('Paused...')



#epoch_stats()
#plot_weight()
#plot_grads()
#img_in_class()
text_stats()
#batch_stats()