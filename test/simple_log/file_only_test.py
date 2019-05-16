import tensorwatch as tw

def writer():
    watcher = tw.Watcher(filename=r'c:\temp\test.log', port=None)
    with watcher.create_stream('metric1') as stream:
        for i in range(10):
            stream.write((i, i*i))

def reader1():
    watcher = tw.Watcher(filename=r'c:\temp\test.log', port=None)
    stream = watcher.open_stream('metric1')
    stream.console_debug = True
    stream.load()

def reader2():
    watcher = tw.Watcher(filename=r'c:\temp\test.log', port=None)
    stream = watcher.open_stream('metric1')
    for item in stream.read_all():
        print(item)

def reader3():
    watcher = tw.Watcher(filename=r'c:\temp\test.log', port=None)
    stream = watcher.open_stream('metric1')
    vis = tw.Visualizer(stream, vis_type='line')
    vis.show()
    tw.plt_loop()

writer()
reader1()
reader2()
reader3()

