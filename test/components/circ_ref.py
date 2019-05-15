import tensorwatch as tw
import objgraph, time #pip install objgraph

cli = tw.WatcherClient()
time.sleep(10)
del cli

import gc
gc.collect()

import time
time.sleep(2)

objgraph.show_backrefs(objgraph.by_type('WatcherClient'), refcounts=True, filename='b.png')

