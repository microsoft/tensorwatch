"""
HiddenLayer

Transforms that apply to and modify graph nodes.
 
Written by Waleed Abdulla
Licensed under the MIT License
"""

import re
import copy
from .graph import Node
from . import ge



###########################################################################
# Transforms
###########################################################################

def _concate_params(matches):
    combo_params = [match.params for match in matches]
    combo_params += [cb for match in matches if match.combo_params for cb in match.combo_params]
    return combo_params

class Fold():
    def __init__(self, pattern, op, name=None):
        # TODO: validate that op and name are valid
        self.pattern = ge.GEParser(pattern).parse()
        self.op = op
        self.name = name

    def apply(self, graph):
        # Copy the graph. Don't change the original.
        graph = copy.deepcopy(graph)

        while True:
            matches, _ = graph.search(self.pattern)
            if not matches:
                break

            # Replace pattern with new node
            if self.op == "__first__":
                combo = matches[0]
            elif self.op == "__last__":
                combo = matches[-1]
            else:
                combo = Node(uid=graph.sequence_id(matches),
                                name=self.name or " &gt; ".join([l.title for l in matches]),
                                op=self.op or self.pattern,
                                output_shape=matches[-1].output_shape,
                                combo_params=_concate_params(matches))
                combo._caption = "/".join(filter(None, [l.caption for l in matches]))
            graph.replace(matches, combo)
        return graph


class FoldId():
    def __init__(self, id_regex, op, name=None):
        # TODO: validate op and name are valid
        self.id_regex = re.compile(id_regex)
        self.op = op
        self.name = name

    def apply(self, graph):
        # Copy the graph. Don't change the original.
        graph = copy.deepcopy(graph)

        # Group nodes by the first matching group of the regex
        groups = {}
        for node in graph.nodes.values():
            m = self.id_regex.match(node.id)
            if not m:
                continue
            
            assert m.groups(), "Regular expression must have a matching group to avoid folding unrelated nodes."
            key = m.group(1)
            if key not in groups:
                groups[key] = []
            groups[key].append(node)
            
        # Fold each group of nodes together
        for key, nodes in groups.items():
            # Replace with a new node
            # TODO: Find last node in the sub-graph and get the output shape from it
            combo = Node(uid=key,
                         name=self.name,
                         op=self.op,
                         combo_params=_concate_params(nodes))
            graph.replace(nodes, combo)
        return graph


class Prune():
    def __init__(self, pattern):
        self.pattern = ge.GEParser(pattern).parse()

    def apply(self, graph):
        # Copy the graph. Don't change the original.
        graph = copy.deepcopy(graph)

        while True:
            matches, _ = graph.search(self.pattern)
            if not matches:
                break
            # Remove found nodes
            graph.remove(matches)
        return graph


class PruneBranch():
    def __init__(self, pattern):
        self.pattern = ge.GEParser(pattern).parse()

    def tag(self, node, tag, graph, conditional=False):
        # Return if the node is already tagged
        if hasattr(node, "__tag__") and node.__tag__ == "tag":
            return
        # If conditional, then tag the node if and only if all its
        # outgoing nodes already have the same tag.
        if conditional:
            # Are all outgoing nodes already tagged?
            outgoing = graph.outgoing(node)
            tagged = filter(lambda n: hasattr(n, "__tag__") and n.__tag__ == tag,
                            outgoing)
            if len(list(tagged)) != len(outgoing):
                # Not all outgoing are tagged
                return
        # Tag the node
        node.__tag__ = tag
        # Tag incoming nodes
        for n in graph.incoming(node):
            self.tag(n, tag, graph, conditional=True)

    def apply(self, graph):
        # Copy the graph. Don't change the original.
        graph = copy.deepcopy(graph)

        while True:
            matches, _ = graph.search(self.pattern)
            if not matches:
                break
            # Tag found nodes and their incoming branches
            for n in matches:
                self.tag(n, "delete", graph)
            # Find all tagged nodes and delete them
            tagged = [n for n in graph.nodes.values()
                      if hasattr(n, "__tag__") and n.__tag__ == "delete"]
            graph.remove(tagged)
        return graph


class FoldDuplicates():
    def apply(self, graph):
        # Copy the graph. Don't change the original.
        graph = copy.deepcopy(graph)

        matches = True
        while matches:
            for node in graph.nodes.values():
                pattern = ge.SerialPattern([ge.NodePattern(node.op), ge.NodePattern(node.op)])
                matches, _ = pattern.match(graph, node)
                if matches:
                    # Use op and name from the first node, and output_shape from the last
                    combo = Node(uid=graph.sequence_id(matches),
                                name=node.name,
                                op=node.op,
                                output_shape=matches[-1].output_shape,
                                combo_params=_concate_params(matches))
                    combo._caption = node.caption
                    combo.repeat = sum([n.repeat for n in matches])
                    graph.replace(matches, combo)
                    break
        return graph


class Rename():
    def __init__(self, op=None, name=None, to=None):
        assert op or name, "Either op or name must be provided"
        assert not(op and name), "Either op or name should be provided, but not both"
        assert bool(to), "The to parameter is required" 
        self.to = to
        self.op = re.compile(op) if op else None
        self.name = re.compile(name) if name else None
    
    def apply(self, graph):
        # Copy the graph. Don't change the original.
        graph = copy.deepcopy(graph)

        for node in graph.nodes.values():
            if self.op:
                node.op = self.op.sub(self.to, node.op)
            # TODO: name is not tested yet
            if self.name:
                node.name = self.name.sub(self.to, node.name)
        return graph


# Transforms to simplify graphs by folding layers that tend to be 
# used together often, such as Conv/BN/Relu.
# These transforms are used AFTER the framework specific transforms
# that map TF and PyTorch graphs to a common representation.
SIMPLICITY_TRANSFORMS = [
    Fold("Conv > Conv > BatchNorm > Relu", "ConvConvBnRelu"),
    Fold("Conv > BatchNorm > Relu", "ConvBnRelu"),
    Fold("Conv > BatchNorm", "ConvBn"),
    Fold("Conv > Relu", "ConvRelu"),
    Fold("Linear > Relu", "LinearRelu"),
    # Fold("ConvBnRelu > MaxPool", "ConvBnReluMaxpool"),
    # Fold("ConvRelu > MaxPool", "ConvReluMaxpool"),
    FoldDuplicates(),
]
