import time
import tensorwatch as tw

srv = tw.Watcher()

sum = 0
for i in range(10000):
    sum += i
    srv.observe(i=i, sum=sum)
    #print(i, sum)
    time.sleep(1)
