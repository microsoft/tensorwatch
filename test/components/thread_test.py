import threading
import time
import sys

def handler(a,b=None):
    sys.exit(1)
def install_handler():
    if sys.platform == "win32":
        if sys.stdin is not None and sys.stdin.isatty():
            #this is Console based application
            import win32api
            win32api.SetConsoleCtrlHandler(handler, True)


def work():
    time.sleep(10000)        
t = threading.Thread(target=work, name='ThreadTest')
t.daemon = True
t.start()
while(True):
    t.join(0.1) #100ms ~ typical human response
# you will get KeyboardIntrupt exception

