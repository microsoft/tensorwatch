import ipywidgets as widgets
from IPython import get_ipython

class PrinterX:
    def __init__(self):
        self.w = w=widgets.HTML()
    def show(self):
        return self.w
    def write(self,s):
        self.w.value = s

print("Running from within ipython?", get_ipython() is not None)
p=PrinterX()
p.show()
p.write('ffffffffff')
