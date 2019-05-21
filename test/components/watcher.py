from tensorwatch.watcher_base import WatcherBase
from tensorwatch.stream import Stream

def main():
    watcher = WatcherBase()
    console_pub = Stream(stream_name = 'S1', console_debug=True)
    stream = watcher.create_stream(expr='lambda vars:vars.x**2')
    console_pub.subscribe(stream)

    for i in range(5):
        watcher.observe(x=i)

main()


