import time
import tensorwatch as tw
from tensorwatch import utils
utils.set_debug_verbosity(4)

srv = tw.Watcher(filename=r'c:\temp\sum.log')
s1 = srv.create_stream('sum', expr='lambda v:(v.i, v.sum)')
s2 = srv.create_stream('sum_2', expr='lambda v:(v.i, v.sum/2)')

sum = 0
for i in range(10000):
    sum += i
    srv.observe(i=i, sum=sum)
    #print(i, sum)
    time.sleep(1)

