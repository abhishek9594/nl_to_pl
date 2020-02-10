#adapted from https://github.com/pcyin/NL2code
from collections import Iterable

from lang.util import typename
from lang.py.grammar import is_builtin_type

class ASTNode(object):
    def __init__(self, node_type, label=None, value=None, children=None):
        self.type = node_type
        self.label = label
        self.value = value

        self.children = list()

        if children:
            if isinstance(children, Iterable):
                for child in children:
                    self.add_child(child)
            elif isinstance(children, ASTNode):
                self.add_child(children)
            else:
                raise AttributeError('Wrong type for child nodes')

        assert not (bool(children) and bool(value)), 'terminal node with a value cannot have children'

    @property
    def is_leaf(self):
        return len(self.children) == 0

    @property
    def is_preterminal(self):
        return len(self.children) == 1 and self.children[0].is_leaf

    @property
    def size(self):
        if self.is_leaf:
            return 1

        node_num = 1
        for child in self.children:
            node_num += child.size

        return node_num

    @property
    def nodes(self):
        """a generator that returns all the nodes"""

        yield self
        for child in self.children:
            for child_n in child.nodes:
                yield child_n

    @property
    def as_type_node(self):
        """return an ASTNode with type information only"""
        return ASTNode(self.type)

    def __repr__(self):
        repr_str = ''
        # if not self.is_leaf:
        repr_str += '('

        repr_str += typename(self.type)

        if self.label is not None:
            repr_str += '{%s}' % self.label

        if self.value is not None:
            repr_str += '{val=%s}' % self.value

        # if not self.is_leaf:
        for child in self.children:
            repr_str += ' ' + child.__repr__()
        repr_str += ')'

        return repr_str

    def __hash__(self):
        code = hash(self.type)
        if self.label is not None:
            code = code * 37 + hash(self.label)
        if self.value is not None:
            code = code * 37 + hash(self.value)
        for child in self.children:
            code = code * 37 + hash(child)

        return code

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if hash(self) != hash(other):
            return False

        if self.type != other.type:
            return False

        if self.label != other.label:
            return False

        if self.value != other.value:
            return False

        if len(self.children) != len(other.children):
            return False

        for i in xrange(len(self.children)):
            if self.children[i] != other.children[i]:
                return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __getitem__(self, child_type):
        return next(iter([c for c in self.children if c.type == child_type]))

    def __delitem__(self, child_type):
        tgt_child = [c for c in self.children if c.type == child_type]
        if tgt_child:
            assert len(tgt_child) == 1, 'unsafe deletion for more than one children'
            tgt_child = tgt_child[0]
            self.children.remove(tgt_child)
        else:
            raise KeyError

    def add_child(self, child):
        self.children.append(child)

    def get_child_id(self, child):
        for i, _child in enumerate(self.children):
            if child == _child:
                return i

        raise KeyError

    def to_rule(self):
        """
        transform the current AST node to a production rule
        """
        nodes = [self]
        rules = []
        while len(nodes):
            curr_node = nodes.pop()
            
            if curr_node.value is not None:
                #only builtin types have value
                rules.append('GenToken[' + str(curr_node.value) + ']')

            if curr_node.is_leaf:
                #either a builtin type (see above) or terminal type
                rules.append('GenToken[<eob>]')
            else:
                curr_node_type = typename(curr_node.type)
                curr_node_child = [typename(child.type) + ('' if child.label is None  else '[' + child.label + ']')  for child in curr_node.children]
                rules.append(curr_node_type + ' -> ' + ' '.join(curr_node_child))

                nodes.extend(curr_node.children[::-1])
        return rules

    def copy(self):
        new_tree = ASTNode(self.type, self.label, self.value)
        if self.is_leaf:
            return new_tree

        for child in self.children:
            new_tree.add_child(child.copy())

        return new_tree


if __name__ == '__main__':
    pass