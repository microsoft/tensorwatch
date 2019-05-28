import time, random
import tensorwatch as tw

# we will create two streams, one for 
# sums of integers, other for random numbers
w = tw.Watcher()
st_isum = w.create_stream('isums')
st_rsum = w.create_stream('rsums')

isum, rsum = 0, 0
for i in range(10000):
    isum += i
    rsum += random.randint(0,i)

    # write to streams
    st_isum.write((i, isum))
    st_rsum.write((i, rsum))

    time.sleep(1)
