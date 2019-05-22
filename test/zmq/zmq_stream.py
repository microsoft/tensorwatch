from tensorwatch.watcher_base import WatcherBase
from tensorwatch.zmq_stream import ZmqStream

def main():
    watcher = WatcherBase()
    stream = watcher.create_stream(expr='lambda vars:vars.x**2')

    zmq_pub = ZmqStream(for_write=True, stream_name = 'ZmqPub', console_debug=True)
    zmq_pub.subscribe(stream)

    for i in range(5):
        watcher.observe(x=i)
    input('paused')

main()



