import time, random
import tensorwatch as tw

# create watcher object as usual
w = tw.Watcher()

weights = None
for i in range(10000):
    weights = [random.random() for _ in range(5)]

    # let watcher observe variables we have
    # this has almost no performance cost
    w.observe(weights=weights)

    time.sleep(1)

