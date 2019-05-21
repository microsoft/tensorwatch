from tensorwatch.watcher_client import WatcherClient
import time

from tensorwatch import utils
utils.set_debug_verbosity(10)


def main():
    watcher = WatcherClient()
    stream = watcher.create_stream(expr='lambda vars:vars.x**2')
    stream.console_debug = True
    input('pause')

main()

