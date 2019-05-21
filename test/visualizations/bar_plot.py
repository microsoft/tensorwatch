import tensorwatch as tw
import random, time

def static_bar():
    w = tw.Watcher()
    s = w.create_stream()

    v = tw.Visualizer(s, vis_type='bar')
    v.show()

    for i in range(10):
        s.write(int(random.random()*10))

    tw.plt_loop()


def dynamic_bar():
    w = tw.Watcher()
    s = w.create_stream()

    v = tw.Visualizer(s, vis_type='bar', clear_after_each=True)
    v.show()

    for i in range(100):
        s.write([('a'+str(i), random.random()*10) for i in range(10)])
        tw.plt_loop(count=3)

def dynamic_bar3d():
    w = tw.Watcher()
    s = w.create_stream()

    v = tw.Visualizer(s, vis_type='bar3d', clear_after_each=True)
    v.show()

    for i in range(100):
        s.write([(i, random.random()*10, z) for i in range(10)  for z in range(10)])
        tw.plt_loop(count=3)

static_bar()
#dynamic_bar()
#dynamic_bar3d()
