"""
HiddenLayer

Implementation graph expressions to find nodes in a graph based on a pattern.
 
Written by Waleed Abdulla
Licensed under the MIT License
"""

import re



class GEParser():
    def __init__(self, text):
        self.index = 0
        self.text = text

    def parse(self):
        return self.serial() or self.parallel() or self.expression()

    def parallel(self):
        index = self.index
        expressions = []
        while len(expressions) == 0 or self.token("|"):
            e = self.expression()
            if not e:
                break
            expressions.append(e)
        if len(expressions) >= 2:
            return ParallelPattern(expressions)
        # No match. Reset index
        self.index = index
    
    def serial(self):
        index = self.index
        expressions = []
        while len(expressions) == 0 or self.token(">"):
            e = self.expression()
            if not e:
                break
            expressions.append(e)

        if len(expressions) >= 2:
            return SerialPattern(expressions)
        self.index = index

    def expression(self):
        index = self.index
        
        if self.token("("):
            e = self.serial() or self.parallel() or self.op()
            if e and self.token(")"):
                return e
        self.index = index
        e = self.op()
        return e

    def op(self):
        t = self.re(r"\w+")
        if t:
            c = self.condition()
            return NodePattern(t, c)
    
    def condition(self):
        # TODO: not implemented yet. This function is a placeholder
        index = self.index
        if self.token("["):
            c = self.token("1x1") or self.token("3x3")
            if c:
                if self.token("]"):
                    return c
            self.index = index
    
    def token(self, s):
        return self.re(r"\s*(" + re.escape(s) + r")\s*", 1)

    def string(self, s):
        if s == self.text[self.index:self.index+len(s)]:
            self.index += len(s)
            return s

    def re(self, regex, group=0):
        m = re.match(regex, self.text[self.index:])
        if m:
            self.index += len(m.group(0))
            return m.group(group)
            

class NodePattern():
    def __init__(self, op, condition=None):
        self.op = op
        self.condition = condition  # TODO: not implemented yet
    
    def match(self, graph, node):
        if isinstance(node, list):
            return [], None
        if self.op == node.op:
            following = graph.outgoing(node)
            if len(following) == 1:
                following = following[0]
            return [node], following
        else:
            return [], None


class SerialPattern():
    def __init__(self, patterns):
        self.patterns = patterns

    def match(self, graph, node):
        all_matches = []
        for i, p in enumerate(self.patterns):
            matches, following = p.match(graph, node)
            if not matches:
                return [], None
            all_matches.extend(matches)
            if i < len(self.patterns) - 1:
                node = following  # Might be more than one node
        return all_matches, following


class ParallelPattern():
    def __init__(self, patterns):
        self.patterns = patterns

    def match(self, graph, nodes):
        if not nodes:
            return [], None
        nodes = nodes if isinstance(nodes, list) else [nodes]
        # If a single node, assume we need to match with its siblings
        if len(nodes) == 1:
            nodes = graph.siblings(nodes[0])
        else:
            # Verify all nodes have the same parent or all have no parent
            parents = [graph.incoming(n) for n in nodes]
            matches = [set(p) == set(parents[0]) for p in parents[1:]]
            if not all(matches):
                return [], None

        # TODO: If more nodes than patterns, we should consider
        #       all permutations of the nodes
        if len(self.patterns) != len(nodes):
            return [], None
        
        patterns = self.patterns.copy()
        nodes = nodes.copy()
        all_matches = []
        end_node = None
        for p in patterns:
            found = False
            for n in nodes:
                matches, following = p.match(graph, n)
                if matches:
                    found = True
                    nodes.remove(n)
                    all_matches.extend(matches)
                    # Verify all branches end in the same node
                    if end_node:
                        if end_node != following:
                            return [], None
                    else:
                        end_node = following
                    break
            if not found:
                return [], None
        return all_matches, end_node


