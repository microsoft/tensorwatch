import tensorwatch as tw

w = tw.Watcher()
s = w.create_stream()

v = tw.Visualizer(s, vis_type='line')
v.show()

for i in range(10):
    i = float(i)
    s.write(tw.PointData(i, i*i, low=i*i-i, high=i*i+i))

tw.plt_loop()