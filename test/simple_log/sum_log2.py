import time, random
import tensorwatch as tw

sum = 0
rsum = 0
for i in range(10000):
    sum += i
    rsum += random.random() * i
    tw.observe(i=i, sum=sum, rsum=rsum)
    time.sleep(1)

