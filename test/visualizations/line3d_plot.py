import tensorwatch as tw
import random, time

# TODO: resolve problem with Axis3D?

def static_line3d():
    w = tw.Watcher()
    s = w.create_stream()

    v = tw.Visualizer(s, vis_type='line3d')
    v.show()

    for i in range(10):
        s.write((i, i*i, int(random.random()*10)))

    tw.plt_loop()

def dynamic_line3d():
    w = tw.Watcher()
    s = w.create_stream()

    v = tw.Visualizer(s, vis_type='line3d', clear_after_each=True)
    v.show()

    for i in range(100):
        s.write([(i, random.random()*10, z) for i in range(10)  for z in range(10)])
        tw.plt_loop(count=3)

static_line3d()
#dynamic_line3d()

