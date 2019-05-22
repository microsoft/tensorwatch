import tensorwatch as tw


def main():
    w = tw.Watcher()
    s1 = w.create_stream()
    s2 = w.create_stream(name='accuracy', vis_args=tw.VisArgs(vis_type='line', xtitle='X-Axis', clear_after_each=False, history_len=2))
    s3 = w.create_stream(name='loss', expr='lambda d:d.loss')
    w.make_notebook()

main()

