import tensorwatch as tw
import time
from tensorwatch.zmq_wrapper import ZmqWrapper
from tensorwatch import utils

class A:
    def on_event(self, obj):
        print(obj)

a = A()


utils.set_debug_verbosity(10)
sub = ZmqWrapper.Subscription(40859, "Topic1", a.on_event)
print("subscriber is waiting")

clisrv = ZmqWrapper.ClientServer(40860, False)
clisrv.send_obj("hello 1")
print('sleeping..')
time.sleep(10)
clisrv.send_obj("hello 2")

print('waiting for key..')
utils.wait_key()