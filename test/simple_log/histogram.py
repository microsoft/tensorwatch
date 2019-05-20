import tensorwatch as tw
import random, time

def static_hist():
    w = tw.Watcher()
    s = w.create_stream()

    v = tw.Visualizer(s, vis_type='histogram', bins=6)
    v.show()

    for i in range(100):
        i = float(i)
        s.write(int(random.random()*10))

    tw.plt_loop()


def dynamic_hist():
    w = tw.Watcher()
    s = w.create_stream()

    v = tw.Visualizer(s, vis_type='histogram', bins=6, clear_after_each=True)
    v.show()

    for i in range(100):
        s.write([int(random.random()*10) for i in range(100)])
        tw.plt_loop(count=3)

dynamic_hist()