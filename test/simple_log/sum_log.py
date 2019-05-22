import time
import tensorwatch as tw

# create watcher and stream that we would use
# for logging
w = tw.Watcher()
s = w.create_stream('my_log')

sum = 0
for i in range(10000):
    sum += i
    # write tuple to our log
    s.write((i, sum))
    time.sleep(1)
