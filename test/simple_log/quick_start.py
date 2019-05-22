import tensorwatch as tw
import time

w = tw.Watcher(filename='test.log')
s = w.create_stream(name='my_metric')
#w.make_notebook()

for i in range(1000):
    s.write((i, i*i)) 
    time.sleep(1)
