import pydot
from .summary_graph import SummaryGraph
from . import distiller
import torch
import collections.abc as abc
import os

class DotWrapper:
    def __init__(self, dot):
        self.dot = dot
    def _repr_svg_(self):
        """Allows Jupyter notebook to render the graph automatically."""
        return self.dot.create_svg().decode()
    def save(self, filename, format="png"):
        # self.dot.format = format
        # directory, file_name = os.path.split(path)
        # # Remove extension from file name. dot.render() adds it.
        # file_name = file_name.replace("." + format, "")
        # self.dot.render(file_name, directory=directory, cleanup=True)
        if filename is not None:
            png = self.dot.create_png()
            with open(os.path.expanduser(filename), 'wb') as fid:
                fid.write(png)   #

def draw_graph(model, args):
    if args is None:
        args = [1, 3, 224, 224] # assume ImageNet default

    # if args is not Tensor but is array like then convert it to torch tensor
    if not isinstance(args, torch.Tensor) and \
        hasattr(args, "__len__") and hasattr(args, '__getitem__') and \
        not isinstance(args, (str, abc.ByteString)):
        args = torch.ones(args)

    dot = draw_img_classifier(model, args)
    return DotWrapper(dot)

def draw_img_classifier(model, dataset=None, display_param_nodes=False,
                                rankdir='TB', styles=None, input_shape=None):
    """Draw a PyTorch image classifier to a PNG file.  This a helper function that
    simplifies the interface of draw_model_to_file().
    Args:
        model: PyTorch model instance
        png_fname (string): PNG file name
        dataset (string): one of 'imagenet' or 'cifar10'.  This is required in order to
                          create a dummy input of the correct shape.
        display_param_nodes (boolean): if True, draw the parameter nodes
        rankdir: diagram direction.  'TB'/'BT' is Top-to-Bottom/Bottom-to-Top
                 'LR'/'R/L' is Left-to-Rt/Rt-to-Left
        styles: a dictionary of styles.  Key is module name.  Value is
                a legal pydot style dictionary.  For example:
                styles['conv1'] = {'shape': 'oval',
                                   'fillcolor': 'gray',
                                   'style': 'rounded, filled'}
        input_shape (tuple): List of integers representing the input shape.
                             Used only if 'dataset' is None
    """
    dummy_input = distiller.get_dummy_input(dataset=dataset,
                                            device=distiller.model_device(model),
                                            input_shape=input_shape)
    try:
        non_para_model = distiller.make_non_parallel_copy(model)
        g = SummaryGraph(non_para_model, dummy_input)

        return sgraph2dot(g, display_param_nodes, rankdir, styles)
        print("Network PNG image generation completed")
    except FileNotFoundError:
        print("An error has occured while generating the network PNG image.")
        print("Please check that you have graphviz installed.")
        print("\t$ sudo apt-get install graphviz")
    finally:
        del non_para_model


def draw_model_to_file(sgraph, png_fname, display_param_nodes=False, rankdir='TB', styles=None):
    """Create a PNG file, containing a graphiz-dot graph of the netowrk represented
    by SummaryGraph 'sgraph'
    Args:
        sgraph (SummaryGraph): the SummaryGraph instance to draw.
        png_fname (string): PNG file name
        display_param_nodes (boolean): if True, draw the parameter nodes
        rankdir: diagram direction.  'TB'/'BT' is Top-to-Bottom/Bottom-to-Top
                 'LR'/'R/L' is Left-to-Rt/Rt-to-Left
        styles: a dictionary of styles.  Key is module name.  Value is
                a legal pydot style dictionary.  For example:
                styles['conv1'] = {'shape': 'oval',
                                   'fillcolor': 'gray',
                                   'style': 'rounded, filled'}
        """
    png = sgraph2dot(sgraph, display_param_nodes=display_param_nodes, rankdir=rankdir, styles=styles).create_png()
    with open(png_fname, 'wb') as fid:
        fid.write(png)

def sgraph2dot(sgraph, display_param_nodes=False, rankdir='TB', styles=None):
    """Create a PNG object containing a graphiz-dot graph of the network,
    as represented by SummaryGraph 'sgraph'.
    Args:
        sgraph (SummaryGraph): the SummaryGraph instance to draw.
        display_param_nodes (boolean): if True, draw the parameter nodes
        rankdir: diagram direction.  'TB'/'BT' is Top-to-Bottom/Bottom-to-Top
                 'LR'/'R/L' is Left-to-Rt/Rt-to-Left
        styles: a dictionary of styles.  Key is module name.  Value is
                a legal pydot style dictionary.  For example:
                styles['conv1'] = {'shape': 'oval',
                                   'fillcolor': 'gray',
                                   'style': 'rounded, filled'}
    """

    def annotate_op_node(op):
        if op['type'] == 'Conv':
            return ["sh={}".format(distiller.size2str(op['attrs']['kernel_shape'])),
                    "g={}".format(str(op['attrs']['group']))]
        return ''

    op_nodes = [op['name'] for op in sgraph.ops.values()]
    data_nodes = []
    param_nodes = []
    for id, param in sgraph.params.items():
        n_data = (id, str(distiller.volume(param['shape'])), str(param['shape']))
        if data_node_has_parent(sgraph, id):
            data_nodes.append(n_data)
        else:
            param_nodes.append(n_data)
    edges = sgraph.edges

    if not display_param_nodes:
        # Use only the edges that don't have a parameter source
        non_param_ids = op_nodes + [dn[0] for dn in data_nodes]
        edges = [edge for edge in sgraph.edges if edge.src in non_param_ids]
        param_nodes = None

    op_nodes_desc = [(op['name'], op['type'], *annotate_op_node(op)) for op in sgraph.ops.values()]
    pydot_graph = create_pydot_graph(op_nodes_desc, data_nodes, param_nodes, edges, rankdir, styles)
    return pydot_graph



def create_pydot_graph(op_nodes_desc, data_nodes, param_nodes, edges, rankdir='TB', styles=None):
    """Low-level API to create a PyDot graph (dot formatted).
    """
    pydot_graph = pydot.Dot('Net', graph_type='digraph', rankdir=rankdir)

    op_node_style = {'shape': 'record',
                     'fillcolor': '#6495ED',
                     'style': 'rounded, filled'}

    for op_node in op_nodes_desc:
        style = op_node_style
        # Check if we should override the style of this node.
        if styles is not None and op_node[0] in styles:
            style = styles[op_node[0]]
        pydot_graph.add_node(pydot.Node(op_node[0], **style, label="\n".join(op_node)))

    for data_node in data_nodes:
        pydot_graph.add_node(pydot.Node(data_node[0], label="\n".join(data_node[1:])))

    node_style = {'shape': 'oval',
                  'fillcolor': 'gray',
                  'style': 'rounded, filled'}

    if param_nodes is not None:
        for param_node in param_nodes:
            pydot_graph.add_node(pydot.Node(param_node[0], **node_style, label="\n".join(param_node[1:])))

    for edge in edges:
        pydot_graph.add_edge(pydot.Edge(edge[0], edge[1]))

    return pydot_graph

def data_node_has_parent(g, id):
    for edge in g.edges:
        if edge.dst == id:
            return True
    return False