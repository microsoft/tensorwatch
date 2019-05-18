# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import zmq
import errno
import pickle
from zmq.eventloop import ioloop, zmqstream
import zmq.utils.monitor
import functools, sys, logging
from threading import Thread, Event
from . import utils
import weakref, logging

class ZmqWrapper:

    _thread:Thread = None
    _ioloop:ioloop.IOLoop = None
    _start_event:Event = None
    _ioloop_block:Event = None # indicates if there is any blocking IOLoop call in progress

    @staticmethod
    def initialize():
        # create thread that will wait on IO Loop
        if ZmqWrapper._thread is None:
            ZmqWrapper._thread = Thread(target=ZmqWrapper._run_io_loop, name='ZMQIOLoop', daemon=True)
            ZmqWrapper._start_event = Event()
            ZmqWrapper._ioloop_block = Event()
            ZmqWrapper._ioloop_block.set() # no blocking call in progress right now
            ZmqWrapper._thread.start()
            # this is needed to make sure IO Loop has enough time to start
            ZmqWrapper._start_event.wait() 

    @staticmethod
    def close():
        # terminate the IO Loop
        if ZmqWrapper._thread is not None:
            ZmqWrapper._ioloop_block.set() # free any blocking call
            ZmqWrapper._ioloop.add_callback(ZmqWrapper._ioloop.stop)
            ZmqWrapper._thread = None
            ZmqWrapper._ioloop = None
            print('ZMQ IOLoop is now closed')

    @staticmethod
    def get_timer(secs, callback, start=True):
        utils.debug_log('Adding PeriodicCallback', secs)
        pc = ioloop.PeriodicCallback(callback, secs * 1e3)
        if (start):
            pc.start()
        return pc

    @staticmethod
    def _run_io_loop():
        if 'asyncio' in sys.modules:
            # tornado may be using asyncio,
            # ensure an eventloop exists for this thread
            import asyncio
            asyncio.set_event_loop(asyncio.new_event_loop())

        ZmqWrapper._ioloop = ioloop.IOLoop()
        ZmqWrapper._ioloop.make_current()
        while ZmqWrapper._thread is not None:
            try:
                ZmqWrapper._start_event.set()
                utils.debug_log('starting ioloop...')
                ZmqWrapper._ioloop.start()
            except zmq.ZMQError as ex:
                if ex.errno == errno.EINTR:
                    logging.exception('Cannot start IOLoop - ZMQError')
                    continue
                else:
                    raise

    # Utility method to run given function on IOLoop
    # this is blocking method if has_rresult=True
    # This can be called from any thread.
    @staticmethod
    def _io_loop_call(has_result, f, *kargs, **kwargs):
        class Result:
            def __init__(self, val=None):
                self.val = val

        def wrapper(f, r, *kargs, **kwargs):
            try:
                r.val = f(*kargs, **kwargs)
                ZmqWrapper._ioloop_block.set()
            except:
                logging.exception('Error in call scheduled in ioloop')


        # We will add callback in IO Loop and then wait for that
        # call back to be completed
        # If result is expected then we wait other wise fire and forget
        if has_result:
            if not ZmqWrapper._ioloop_block.is_set():
                # TODO: better way to raise this error?
                print('Previous blocking call on IOLoop is not yet complete!')
            ZmqWrapper._ioloop_block.clear()
            r = Result()
            f_wrapped = functools.partial(wrapper, f, r, *kargs, **kwargs)
            ZmqWrapper._ioloop.add_callback(f_wrapped)
            utils.debug_log('Waiting for call on ioloop', f, verbosity=5)
            ZmqWrapper._ioloop_block.wait()
            utils.debug_log('Call on ioloop done', f, verbosity=5)
            return r.val
        else:
            f_wrapped = functools.partial(f, *kargs, **kwargs)
            ZmqWrapper._ioloop.add_callback(f_wrapped)

    class Publication:
        def __init__(self, port, host='*', block_until_connected=True):
            # define vars
            self._socket = None
            self._mon_socket = None
            self._mon_stream = None

            ZmqWrapper.initialize()
            utils.debug_log('Creating Publication', port, verbosity=1)
            # make sure the call blocks until connection is made
            ZmqWrapper._io_loop_call(block_until_connected, self._start_srv, port, host)

        def _start_srv(self, port, host):
            context = zmq.Context()
            self._socket = context.socket(zmq.PUB)
            utils.debug_log('Binding socket', (host, port), verbosity=5)
            self._socket.bind('tcp://%s:%d' % (host, port))
            utils.debug_log('Bound socket', (host, port), verbosity=5)
            self._mon_socket = self._socket.get_monitor_socket(zmq.EVENT_CONNECTED | zmq.EVENT_DISCONNECTED)
            self._mon_stream = zmqstream.ZMQStream(self._mon_socket)
            self._mon_stream.on_recv(self._on_mon)

        def close(self):
            if self._socket:
                ZmqWrapper._io_loop_call(False, self._socket.close)

        # we need this wrapper method as self._socket might not be there yet
        def _send_multipart(self, parts):
            #utils.debug_log('_send_multipart', parts, verbosity=6)
            return self._socket.send_multipart(parts)

        def send_obj(self, obj, topic=''):
            ZmqWrapper._io_loop_call(False, self._send_multipart, 
                [topic.encode(), pickle.dumps(obj)])

        def _on_mon(self, msg):
            ev = zmq.utils.monitor.parse_monitor_message(msg)
            event = ev['event']
            endpoint = ev['endpoint']
            if event == zmq.EVENT_CONNECTED:
                utils.debug_log('Subscriber connect event', endpoint, verbosity=1)
            elif event == zmq.EVENT_DISCONNECTED:
                utils.debug_log('Subscriber disconnect event', endpoint, verbosity=1)


    class Subscription:
        # subscribe to topic, call callback when object is received on topic
        def __init__(self, port, topic='', callback=None, host='localhost'):
            self._socket = None
            self._stream = None
            self.topic = None

            ZmqWrapper.initialize()
            utils.debug_log('Creating Subscription', port, verbosity=1)
            ZmqWrapper._io_loop_call(False, self._add_sub,
                port, topic=topic, callback=callback, host=host)

        def close(self):
            if self._socket:
                ZmqWrapper._io_loop_call(False, self._socket.close)

        def _add_sub(self, port, topic, callback, host):
            def callback_wrapper(weak_callback, msg):
                [topic, obj_s] = msg # pylint: disable=unused-variable
                try:
                    if weak_callback and weak_callback():
                        weak_callback()(pickle.loads(obj_s))
                except Exception as ex:
                    logging.exception('Error in subscription callback')
                    raise

            # connect to stream socket
            context = zmq.Context()
            self.topic = topic.encode()
            self._socket = context.socket(zmq.SUB)

            utils.debug_log('Subscriber connecting...', (host, port), verbosity=1)
            self._socket.connect('tcp://%s:%d' % (host, port))
            utils.debug_log('Subscriber connected!', (host, port), verbosity=1)

            # setup socket filtering
            if topic != '':
                self._socket.setsockopt(zmq.SUBSCRIBE, self.topic)

            # if callback is specified then create a stream and set it 
            # for on_recv event - this would require running ioloop
            if callback is not None:
                self._stream = zmqstream.ZMQStream(self._socket)
                wr_cb = weakref.WeakMethod(callback)
                wrapper = functools.partial(callback_wrapper, wr_cb)
                self._stream.on_recv(wrapper)
            #else use receive_obj

        def _receive_obj(self):
            [topic, obj_s] = self._socket.recv_multipart() # pylint: disable=unbalanced-tuple-unpacking
            if topic != self.topic:
                raise ValueError('Expected topic: %s, Received topic: %s' % (topic, self.topic)) 
            return pickle.loads(obj_s)

        def receive_obj(self):
            return ZmqWrapper._io_loop_call(True, self._receive_obj)

        def _get_socket_identity(self):
            ep_id = self._socket.getsockopt(zmq.LAST_ENDPOINT)
            return ep_id

        def get_socket_identity(self):
            return ZmqWrapper._io_loop_call(True, self._get_socket_identity)


    class ClientServer:
        def __init__(self, port, is_server, callback=None, host=None):
            self._socket = None
            self._stream = None

            ZmqWrapper.initialize()
            utils.debug_log('Creating ClientServer', (is_server, port), verbosity=1)

            # make sure call blocks until connection is made
            # otherwise variables would not be available
            ZmqWrapper._io_loop_call(True, self._connect,
                port, is_server, callback, host)

        def close(self):
            if self._socket:
                ZmqWrapper._io_loop_call(False, self._socket.close)

        def _connect(self, port, is_server, callback, host):
            def callback_wrapper(callback, msg):
                utils.debug_log('Server received request...', verbosity=6)

                [obj_s] = msg
                try:
                    ret = callback(self, pickle.loads(obj_s))
                    # we must send reply to complete the cycle
                    self._socket.send_multipart([pickle.dumps((ret, None))])
                except Exception as ex:
                    logging.exception('ClientServer call raised exception')
                    # we must send reply to complete the cycle
                    self._socket.send_multipart([pickle.dumps((None, ex))])
                
                utils.debug_log('Server sent response', verbosity=6)
                
            context = zmq.Context()
            if is_server:
                host = host or '127.0.0.1'
                self._socket = context.socket(zmq.REP)
                utils.debug_log('Binding socket', (host, port), verbosity=5)
                self._socket.bind('tcp://%s:%d' % (host, port))
                utils.debug_log('Bound socket', (host, port), verbosity=5)
            else:
                host = host or 'localhost'
                self._socket = context.socket(zmq.REQ)
                self._socket.setsockopt(zmq.REQ_CORRELATE, 1)
                self._socket.setsockopt(zmq.REQ_RELAXED, 1)

                utils.debug_log('Client connecting...', verbosity=1)
                self._socket.connect('tcp://%s:%d' % (host, port))
                utils.debug_log('Client connected!', verbosity=1)

            if callback is not None:
                self._stream = zmqstream.ZMQStream(self._socket)
                wrapper = functools.partial(callback_wrapper, callback)
                self._stream.on_recv(wrapper)
            #else use receive_obj

        def send_obj(self, obj):
            ZmqWrapper._io_loop_call(False, self._socket.send_multipart,
                [pickle.dumps(obj)])

        def receive_obj(self):
            # pylint: disable=unpacking-non-sequence
            [obj_s] = ZmqWrapper._io_loop_call(True, self._socket.recv_multipart)
            return pickle.loads(obj_s)

        def request(self, req_obj):
            utils.debug_log('Client sending request...', verbosity=6)
            self.send_obj(req_obj)
            r = self.receive_obj()
            utils.debug_log('Client received response', verbosity=6)
            return r
