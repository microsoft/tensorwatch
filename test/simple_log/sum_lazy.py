import time
import tensorwatch as tw
import random

# create watcher, notice that we are not logging anything
w = tw.Watcher()

weights = [random.random() for _ in range(5)]
for i in range(10000):
    weights = [random.random() for _ in range(5)]

    # we are just observing variables
    # observation has no cost, nothing gets logged anywhere
    w.observe(weights=weights)

    time.sleep(1)

