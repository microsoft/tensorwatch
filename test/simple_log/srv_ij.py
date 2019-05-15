import tensorwatch as tw
import time
import random
from tensorwatch import utils

utils.set_debug_verbosity(4)

srv = tw.Watcher()

while(True):
    for i in range(1000):
        srv.observe("ev_i", val=i*random.random(), x=i)
        print('sent ev_i ', i)
        time.sleep(1)
        for j in range(5):
            srv.observe("ev_j", x=j, val=j*random.random())
            print('sent ev_j ', j)
            time.sleep(0.5)
        srv.end_event("ev_j")
    srv.end_event("ev_i")