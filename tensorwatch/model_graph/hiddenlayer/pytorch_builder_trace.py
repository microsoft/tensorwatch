from torch.utils.tensorboard._pytorch_graph import GraphPy, NodePyIO, NodePyOP
import torch

from . import transforms as ht
from collections import abc
import numpy as np

# PyTorch Graph Transforms
FRAMEWORK_TRANSFORMS = [
    # Hide onnx: prefix
    ht.Rename(op=r"onnx::(.*)", to=r"\1"),
    # ONNX uses Gemm for linear layers (stands for General Matrix Multiplication).
    # It's an odd name that noone recognizes. Rename it. 
    ht.Rename(op=r"Gemm", to=r"Linear"),
    # PyTorch layers that don't have an ONNX counterpart
    ht.Rename(op=r"aten::max\_pool2d\_with\_indices", to="MaxPool"),
    # Shorten op name
    ht.Rename(op=r"BatchNormalization", to="BatchNorm"),
]

def parse(graph, args=None, omit_useless_nodes=True):
    """This method parses an optimized PyTorch model graph and produces
    a list of nodes and node stats for eventual conversion to TensorBoard
    protobuf format.
    Args:
      graph (PyTorch module): The model to be parsed.
      args (tuple): input tensor[s] for the model.
      omit_useless_nodes (boolean): Whether to remove nodes from the graph.
    """
    n_inputs = len(args)

    scope = {}
    nodes_py = GraphPy()
    for i, node in enumerate(graph.inputs()):
        if omit_useless_nodes:
            if len(node.uses()) == 0:  # number of user of the node (= number of outputs/ fanout)
                continue

        if i < n_inputs:
            nodes_py.append(NodePyIO(node, 'input'))
        else:
            nodes_py.append(NodePyIO(node))  # parameter

    for node in graph.nodes():
        nodes_py.append(NodePyOP(node))

    for node in graph.outputs():  # must place last.
        NodePyIO(node, 'output')
    nodes_py.find_common_root()
    nodes_py.populate_namespace_from_OP_to_IO()
    return nodes_py


def graph(model, args, verbose=False):
    """
    This method processes a PyTorch model and produces a `GraphDef` proto
    that can be logged to TensorBoard.
    Args:
      model (PyTorch module): The model to be parsed.
      args (tuple): input tensor[s] for the model.
      verbose (bool): Whether to print out verbose information while
        processing.
    """
    with torch.onnx.set_training(model, False):  # TODO: move outside of torch.onnx?
        try:
            trace = torch.jit.trace(model, args)
            graph = trace.graph
        except RuntimeError as e:
            print(e)
            print('Error occurs, No graph saved')
            raise e

    if verbose:
        print(graph)
    return parse(graph, args)


def import_graph(hl_graph, model, args, input_names=None, verbose=False):
    # TODO: add input names to graph

    if args is None:
        args = [1, 3, 224, 224] # assume ImageNet default

    # if args is not Tensor but is array like then convert it to torch tensor
    if not isinstance(args, torch.Tensor) and \
        hasattr(args, "__len__") and hasattr(args, '__getitem__') and \
        not isinstance(args, (str, abc.ByteString)):
        args = torch.ones(args)

    graph_py = graph(model, args, verbose)

    # # Loop through nodes and build HL graph
    # nodes = list(torch_graph.nodes())
    # inps = [(n, [i.unique() for i in n.inputs()]) for n in nodes]
    # for i, torch_node in enumerate(nodes):
    #     # Op
    #     op = torch_node.kind()
    #     # Parameters
    #     params = {k: torch_node[k] for k in torch_node.attributeNames()} 
    #     # Inputs/outputs
    #     # TODO: inputs = [i.unique() for i in node.inputs()]
    #     outputs = [o.unique() for o in torch_node.outputs()]
    #     # Get output shape
    #     shape = get_shape(torch_node)
    #     # Add HL node
    #     hl_node = Node(uid=pytorch_id(torch_node), name=None, op=op, 
    #                    output_shape=shape, params=params)
    #     hl_graph.add_node(hl_node)
    #     # Add edges
    #     for target_torch_node,target_inputs in inps:
    #         if set(outputs) & set(target_inputs):
    #             hl_graph.add_edge_by_id(pytorch_id(torch_node), pytorch_id(target_torch_node), shape)
    return hl_graph
