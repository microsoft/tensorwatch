import tensorwatch as tw

stream = tw.ArrayStream([(i, i*i) for i in range(50)])
img_plot = tw.Visualizer(stream, vis_type='mpl-line', viz_img_scale=3, xtitle='Epochs', ytitle='Gain')
# img_plot.show()
# tw.plt_loop()
img_plot.save(r'c:\temp\fig1.png')
