conda install -c conda-forge jupyterlab nodejs
conda install ipywidgets
conda install -c plotly plotly-orca psutil

set NODE_OPTIONS=--max-old-space-size=4096
jupyter labextension install @jupyter-widgets/jupyterlab-manager --no-build
jupyter labextension install plotlywidget --no-build
jupyter labextension install @jupyterlab/plotly-extension --no-build
jupyter labextension install jupyterlab-chart-editor --no-build
jupyter lab build
set NODE_OPTIONS=