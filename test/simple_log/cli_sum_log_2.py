import tensorwatch as tw
from tensorwatch import utils
utils.set_debug_verbosity(4)


#r = tw.Visualizer(vis_type='mpl-line')
#r.show()
#r2=tw.Visualizer('map(lambda x:math.sqrt(x.sum), l)', cell=r.cell)
#r3=tw.Visualizer('map(lambda x:math.sqrt(x.sum), l)', renderer=r2)

def show_mpl():
    cli = tw.WatcherClient(r'c:\temp\sum.log')
    s1 = cli.open_stream('sum')
    p = tw.LinePlot(title='Demo')
    p.subscribe(s1, xtitle='Index', ytitle='sqrt(ev_i)')
    s1.load()
    p.show()
    
    tw.plt_loop()

def show_text():
    cli = tw.WatcherClient(r'c:\temp\sum.log')
    s1 = cli.open_stream('sum_2')
    text = tw.Visualizer(s1)
    text.show()
    input('Waiting')

#show_text()
show_mpl()
