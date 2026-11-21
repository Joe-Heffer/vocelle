"""Tiny helper DSL for building node trees from Python without boilerplate.

    t = Tree(node_group)
    d = t.math("LENGTH", t.node("GeometryNodeInputPosition"), vector=True)
    t.link(d, some_node.inputs["Value"])

Any argument that is a number/tuple is written to the socket's default value; any
`NodeSocket` or `Node` (first visible output) is linked.
"""

from __future__ import annotations

import bpy


def _visible(sockets):
    return [s for s in sockets if s.enabled and not getattr(s, "is_unavailable", False)]


def out(node, name: str | int | None = None):
    if isinstance(node, bpy.types.NodeSocket):
        return node
    socks = _visible(node.outputs)
    if name is None:
        return socks[0]
    if isinstance(name, int):
        return socks[name]
    return next(s for s in socks if s.name == name or s.identifier == name)


def inp(node, name: str | int):
    socks = _visible(node.inputs)
    if isinstance(name, int):
        return socks[name]
    return next(s for s in socks if s.name == name or s.identifier == name)


class Tree:
    def __init__(self, tree):
        self.tree = tree
        self.nodes = tree.nodes
        self.links = tree.links

    def node(self, kind: str, inputs: dict | None = None, **props):
        n = self.nodes.new(kind)
        for k, v in props.items():
            setattr(n, k, v)
        for k, v in (inputs or {}).items():
            self.set(inp(n, k), v)
        return n

    def set(self, socket, value):
        if isinstance(value, (bpy.types.NodeSocket, bpy.types.Node)):
            self.links.new(out(value), socket)
        else:
            socket.default_value = value

    def link(self, a, b):
        self.links.new(out(a), b if isinstance(b, bpy.types.NodeSocket) else inp(b, 0))

    # Shorthands -----------------------------------------------------------------------
    def math(self, op: str, a, b=0.0, c=0.0, clamp: bool = False):
        n = self.nodes.new("ShaderNodeMath")
        n.operation, n.use_clamp = op, clamp
        for i, v in enumerate((a, b, c)):
            self.set(n.inputs[i], v)
        return n

    def add(self, a, b):
        return self.math("ADD", a, b)

    def mul(self, a, b):
        return self.math("MULTIPLY", a, b)

    def madd(self, a, b, c):
        return self.math("MULTIPLY_ADD", a, b, c)

    def vmath(self, op: str, a, b=(0.0, 0.0, 0.0)):
        n = self.nodes.new("ShaderNodeVectorMath")
        n.operation = op
        self.set(n.inputs[0], a)
        self.set(n.inputs[1], b)
        return n

    def xyz(self, x=0.0, y=0.0, z=0.0):
        return self.node("ShaderNodeCombineXYZ", {"X": x, "Y": y, "Z": z})

    def sep(self, v):
        return self.node("ShaderNodeSeparateXYZ", {"Vector": v})
