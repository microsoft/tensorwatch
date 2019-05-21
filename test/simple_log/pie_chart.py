import tensorwatch as tw
import random, time

def static_pie():
    w = tw.Watcher()
    s = w.create_stream()

    v = tw.Visualizer(s, vis_type='pie', bins=6)
    v.show()

    for i in range(6):
        s.write(('label'+str(i), random.random()*10, None, 0.5 if i==3 else 0))

    tw.plt_loop()


def dynamic_pie():
    w = tw.Watcher()
    s = w.create_stream()

    v = tw.Visualizer(s, vis_type='pie', bins=6, clear_after_each=True)
    v.show()

    for _ in range(100):
        s.write([('label'+str(i), random.random()*10, None, i*0.01) for i in range(12)])
        tw.plt_loop(count=3)

#static_pie()
dynamic_pie()
