"""
HiddenLayer

PyTorch graph importer.
 
Written by Waleed Abdulla
Licensed under the MIT License
"""

from __future__ import absolute_import, division, print_function
import re
from .graph import Graph, Node
from . import transforms as ht
import torch
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


def dump_pytorch_graph(graph):
    """List all the nodes in a PyTorch graph."""
    f = "{:25} {:40}   {} -> {}"
    print(f.format("kind", "scopeName", "inputs", "outputs"))
    for node in graph.nodes():
        print(f.format(node.kind(), node.scopeName(),
                       [i.unique() for i in node.inputs()],
                       [i.unique() for i in node.outputs()]
                       ))


def pytorch_id(node):
    """Returns a unique ID for a node."""
    # After ONNX simplification, the scopeName is not unique anymore
    # so append node outputs to guarantee uniqueness
    return node.scopeName() + "/outputs/" + "/".join([o.debugName() for o in node.outputs()])



def get_shape(torch_node):
    """Return the output shape of the given Pytorch node."""
    # Extract node output shape from the node string representation
    # This is a hack because there doesn't seem to be an official way to do it.
    # See my quesiton in the PyTorch forum:
    # https://discuss.pytorch.org/t/node-output-shape-from-trace-graph/24351/2
    # TODO: find a better way to extract output shape
    # TODO: Assuming the node has one output. Update if we encounter a multi-output node.
    m = re.match(r".*Float\(([\d\s\,]+)\).*", str(next(torch_node.outputs())))
    if m:
        shape = m.group(1)
        shape = shape.split(",")
        shape = tuple(map(int, shape))
    else:
        shape = None
    return shape

def calc_rf(model, input_shape):
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue;
        if 'bias' in n:
            p.data.fill_(0)
        elif 'weight' in n:
            p.data.fill_(1)

    input = torch.ones(input_shape, requires_grad=True)
    output = model(input)
    out_shape = output.size()
    ndims = len(out_shape)
    grad = torch.zeros(out_shape)
    l_tmp=[]
    for i in xrange(ndims):
        if i==0 or i==1:#batch or channel
            l_tmp.append(0)
        else:
            l_tmp.append(out_shape[i]/2)
            
    grad[tuple(l_tmp)] = 1
    output.backward(gradient=grad)
    grad_np = img_.grad[0,0].data.numpy()
    idx_nonzeros = np.where(grad_np!=0)
    RF=[np.max(idx)-np.min(idx)+1 for idx in idx_nonzeros]
    
    return RF

def import_graph(hl_graph, model, args, input_names=None, verbose=False):
    # TODO: add input names to graph

    if args is None:
        args = [1, 3, 224, 224] # assume ImageNet default

    # if args is not Tensor but is array like then convert it to torch tensor
    if not isinstance(args, torch.Tensor) and \
        hasattr(args, "__len__") and hasattr(args, '__getitem__') and \
        not isinstance(args, (str, abc.ByteString)):
        args = torch.ones(args)

    # Run the Pytorch graph to get a trace and generate a graph from it
    with torch.onnx.set_training(model, False):
        try:
            trace = torch.jit.trace(model, args)
            torch.onnx._optimize_trace(trace)
            torch_graph = trace.graph
        except RuntimeError as e:
            print(e)
            print('Error occured when creating jit trace for model.')
            raise e

    # Dump list of nodes (DEBUG only)
    if verbose:
        dump_pytorch_graph(torch_graph)

    # Loop through nodes and build HL graph
    nodes = list(torch_graph.nodes())
    inps = [(n, [i.unique() for i in n.inputs()]) for n in nodes]
    for i, torch_node in enumerate(nodes):
        # Op
        op = torch_node.kind()
        # Parameters
        params = {k: torch_node[k] for k in torch_node.attributeNames()} 
        # Inputs/outputs
        # TODO: inputs = [i.unique() for i in node.inputs()]
        outputs = [o.unique() for o in torch_node.outputs()]
        # Get output shape
        shape = get_shape(torch_node)
        # Add HL node
        hl_node = Node(uid=pytorch_id(torch_node), name=None, op=op, 
                       output_shape=shape, params=params)
        hl_graph.add_node(hl_node)
        # Add edges
        for target_torch_node,target_inputs in inps:
            if set(outputs) & set(target_inputs):
                hl_graph.add_edge_by_id(pytorch_id(torch_node), pytorch_id(target_torch_node), shape)
    return hl_graph
