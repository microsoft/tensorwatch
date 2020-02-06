import pydot

g = pydot.Dot()
g.set_type('digraph')
node = pydot.Node('legend')
node.set("shape", 'box')
g.add_node(node)
node.set('label', 'mine')
s = g.to_string()
expected = 'digraph G {\nlegend [label=mine, shape=box];\n}\n'
assert s == expected
print(s)
png = g.create_png()
print(png)