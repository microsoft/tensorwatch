# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np #pip install numpy
import math
import time
import sys
import os
import inspect
import re
import uuid
from collections import abc
import textwrap 

from functools import wraps
import gc
import timeit
def MeasureTime(f):
    @wraps(f)
    def _wrapper(*args, **kwargs):
        gcold = gc.isenabled()
        gc.disable()
        start_time = timeit.default_timer()
        try:
            result = f(*args, **kwargs)
        finally:
            elapsed = timeit.default_timer() - start_time
            if gcold:
                gc.enable()
            print('Function "{}": {}s'.format(f.__name__, elapsed))
        return result
    return _wrapper
class MeasureBlockTime:
    def __init__(self, name="(block)", no_print = False, disable_gc = True, format_str=":.2f"):
        self.name = name
        self.no_print = no_print
        self.disable_gc = disable_gc
        self.format_str = format_str
        self.gcold = None
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        if self.disable_gc:
            self.gcold = gc.isenabled()
            gc.disable()
        self.start_time = timeit.default_timer()
        return self
    def __exit__(self,ty,val,tb):
        self.elapsed = timeit.default_timer() - self.start_time
        if self.disable_gc and self.gcold:
            gc.enable()
        if not self.no_print:
            print(('{}: {' + self.format_str + '}s').format(self.name, self.elapsed))
        return False #re-raise any exceptions
def getTime():
    return timeit.default_timer()
def getElapsedTime(start_time):
    return timeit.default_timer() - start_time
def string_to_uint8_array(bstr):
    return np.fromstring(bstr, np.uint8)
    
def string_to_float_array(bstr):
    return np.fromstring(bstr, np.float32)
    
def list_to_2d_float_array(flst, width, height):
    return np.reshape(np.asarray(flst, np.float32), (height, width))
    
def get_pfm_array(response):
    return list_to_2d_float_array(response.image_data_float, response.width, response.height)

# creates same list as len of seq filled with val - if val is already not a list of same size
def fill_like(val, seq):
    l = len(seq)
    if is_array_like(val) and len(val) == l:
        return val
    return [val] * len(seq)

def is_array_like(obj, string_is_array=False, tuple_is_array=True):
    result = hasattr(obj, "__len__") and hasattr(obj, '__getitem__') 
    if result and not string_is_array and isinstance(obj, (str, abc.ByteString)):
        result = False
    if result and not tuple_is_array and isinstance(obj, tuple):
        result = False
    return result

def is_scalar(x):
    return x is None or np.isscalar(x)

def is_scaler_array(x): #detects (x,y) or [x, y]
    if is_array_like(x):
        if len(x) > 0:
            return len(x) if is_scalar(x[0]) else -1
        else:
            return 0
    else:
        return -1

def get_public_fields(obj):
    return [attr for attr in dir(obj)
                            if not (attr.startswith("_") 
                            or inspect.isbuiltin(attr)
                            or inspect.isfunction(attr)
                            or inspect.ismethod(attr))]

def set_default(dictionary, key, default_val, replace_none=True):
    if key not in dictionary or (replace_none and dictionary[key] is None):
        dictionary[key] = default_val

def to_array_like(val):
    if is_array_like(val):
        return val
    return [val]
    
def to_dict(obj):
    return dict([attr, getattr(obj, attr)] for attr in get_public_fields(obj))

    
def to_str(obj):
    return str(to_dict(obj))

    
def write_file(filename, bstr):
    with open(filename, 'wb') as afile:
        afile.write(bstr)

# helper method for converting getOrientation to roll/pitch/yaw
# https:#en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    
def has_method(o, name):
    return callable(getattr(o, name, None))

def to_eularian_angles(q):
    z = q.z_val
    y = q.y_val
    x = q.x_val
    w = q.w_val
    ysqr = y * y

    # roll (x-axis rotation)
    t0 = +2.0 * (w*x + y*z)
    t1 = +1.0 - 2.0*(x*x + ysqr)
    roll = math.atan2(t0, t1)

    # pitch (y-axis rotation)
    t2 = +2.0 * (w*y - z*x)
    if (t2 > 1.0):
        t2 = 1
    if (t2 < -1.0):
        t2 = -1.0
    pitch = math.asin(t2)

    # yaw (z-axis rotation)
    t3 = +2.0 * (w*z + x*y)
    t4 = +1.0 - 2.0 * (ysqr + z*z)
    yaw = math.atan2(t3, t4)

    return (pitch, roll, yaw)

    
# TODO: sync with AirSim utils.py
    
def wait_key(message = ''):
    ''' Wait for a key press on the console and return it. '''
    if message != '':
        print (message)

    result = None
    if os.name == 'nt':
        import msvcrt
        result = msvcrt.getch()
    else:
        # pylint: disable=import-error
        import termios # pylint: disable=import-error
        fd = sys.stdin.fileno()

        oldterm = termios.tcgetattr(fd)
        newattr = termios.tcgetattr(fd)
        newattr[3] = newattr[3] & ~termios.ICANON & ~termios.ECHO
        termios.tcsetattr(fd, termios.TCSANOW, newattr)

        try:
            result = sys.stdin.read(1)
        except IOError:
            pass
        finally:
            termios.tcsetattr(fd, termios.TCSAFLUSH, oldterm)

    return result

    
def read_pfm(file):
    """ Read a pfm file """
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    header = str(bytes.decode(header, encoding='utf-8'))
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    temp_str = str(bytes.decode(file.readline(), encoding='utf-8'))
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', temp_str)
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    # DEY: I don't know why this was there.
    #data = np.flipud(data)
    file.close()
    
    return data, scale

    
def write_pfm(file, image, scale=1):
    """ Write a pfm file """
    file = open(file, 'wb')

    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode('utf-8')  if color else 'Pf\n'.encode('utf-8'))
    temp_str = '%d %d\n' % (image.shape[1], image.shape[0])
    file.write(temp_str.encode('utf-8'))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    temp_str = '%f\n' % scale
    file.write(temp_str.encode('utf-8'))

    image.tofile(file)

    
def write_png(filename, image):
    """ image must be numpy array H X W X channels
    """
    import zlib, struct

    buf = image.flatten().tobytes()
    width = image.shape[1]
    height = image.shape[0]

    # reverse the vertical line order and add null bytes at the start
    width_byte_4 = width * 4
    raw_data = b''.join(b'\x00' + buf[span:span + width_byte_4]
                        for span in range((height - 1) * width_byte_4, -1, - width_byte_4))

    def png_pack(png_tag, data):
        chunk_head = png_tag + data
        return (struct.pack("!I", len(data)) +
                chunk_head +
                struct.pack("!I", 0xFFFFFFFF & zlib.crc32(chunk_head)))

    png_bytes = b''.join([
        b'\x89PNG\r\n\x1a\n',
        png_pack(b'IHDR', struct.pack("!2I5B", width, height, 8, 6, 0, 0, 0)),
        png_pack(b'IDAT', zlib.compress(raw_data, 9)),
        png_pack(b'IEND', b'')])

    write_file(filename, png_bytes)

def add_windows_ctrl_c():
    def handler(a,b=None): # pylint: disable=unused-argument
        sys.exit(1)
    add_windows_ctrl_c.is_handler_installed = \
        vars(add_windows_ctrl_c).setdefault('is_handler_installed',False)
    if sys.platform == "win32" and not add_windows_ctrl_c.is_handler_installed:
        if sys.stdin is not None and sys.stdin.isatty():
            #this is Console based application
            import win32api
            win32api.SetConsoleCtrlHandler(handler, True)
        #else do not install handler for non-console applications
        add_windows_ctrl_c.is_handler_installed = True

_utils_debug_verbosity=0
_utils_start_time = time.time()
def set_debug_verbosity(verbosity=0):
    global _utils_debug_verbosity # pylink: disable=global-statement
    _utils_debug_verbosity = verbosity
def debug_log(msg, param=None, verbosity=3):
    global _utils_debug_verbosity # pylink: disable=global-statement
    if _utils_debug_verbosity is not None and _utils_debug_verbosity >= verbosity:
        print("[Debug][{}]: {} : {} : t={:.2f}".format(verbosity, msg, param, time.time()-_utils_start_time))

def get_uuid(is_hex = False):
    return  str(uuid.uuid4()) if not is_hex else uuid.uuid4().hex

def is_uuid4(s, is_hex=False):
    try:
        val = uuid.UUID(s, version=4)
        return val.hex == s if is_hex else str(val) == s
    except ValueError:
        return False

def frange(start, stop=None, step=None, steps=None):
    if stop is None:
        start, stop = 0, start
    if steps is None:
        if step is None:
            step = 1
        steps = int((stop-start)/step)
    else:
        if step is not None:
            raise ValueError("Both step and steps cannot be specified")
        step = (stop-start)/steps
    for _ in range(steps):
        yield start
        start += step  

def wrap_string(s, chars_per_line=12):
    return "\n".join(textwrap.wrap(s, chars_per_line))

def is_eof(f):
    s = f.read(1)
    if s != b'':    # restore position
        f.seek(-1, os.SEEK_CUR)
    return s == b''

def str2identifier(s):

   # Remove invalid characters
   s = re.sub('[^0-9a-zA-Z_]', '', s)

   # Remove leading characters until we find a letter or underscore
   s = re.sub('^[^a-zA-Z_]+', '', s)

   return s