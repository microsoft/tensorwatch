import tensorwatch as tw
from tensorwatch import utils
utils.set_debug_verbosity(4)


#r = tw.Visualizer(vis_type='mpl-line')
#r.show()
#r2=tw.Visualizer('map(lambda x:math.sqrt(x.sum), l)', cell=r.cell)
#r3=tw.Visualizer('map(lambda x:math.sqrt(x.sum), l)', renderer=r2)

def show_mpl():
    cli = tw.WatcherClient()
    p = tw.mpl.LinePlot(title='Demo')
    s1 = cli.create_stream(expr='lambda v:(v.i, v.sum)')
    p.subscribe(s1, xtitle='Index', ytitle='sqrt(ev_i)')
    p.show()
    
    tw.plt_loop()

def show_text():
    cli = tw.WatcherClient()
    s1 = cli.create_stream(expr='lambda v:(v.i, v.sum)')
    text = tw.Visualizer(s1)
    text.show()
    input('Waiting')

#show_text()
show_mpl()