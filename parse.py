#adapted from https://github.com/pcyin/NL2code
import ast
import astor
import re

from astnode import ASTNode
from lang.py.grammar import is_compositional_leaf, PY_AST_NODE_FIELDS, NODE_FIELD_BLACK_LIST, is_builtin_type, is_terminal_ast_type
from lang.util import typename, parse_rule, extract_val_GenToken


def python_ast_to_parse_tree(node):
    assert isinstance(node, ast.AST)

    node_type = type(node)
    tree = ASTNode(node_type)

    # it's a leaf AST node, e.g., ADD, Break, etc.
    if len(node._fields) == 0:
        return tree

    # if it's a compositional AST node with empty fields
    if is_compositional_leaf(node):
        epsilon = ASTNode('epsilon')
        tree.add_child(epsilon)

        return tree

    fields_info = PY_AST_NODE_FIELDS[node_type.__name__]

    for field_name, field_value in ast.iter_fields(node):
        # remove ctx stuff
        if field_name in NODE_FIELD_BLACK_LIST:
            continue

        # omit empty fields, including empty lists
        if field_value is None or (isinstance(field_value, list) and len(field_value) == 0):
            continue

        # now it's not empty!
        field_type = fields_info[field_name]['type']
        is_list_field = fields_info[field_name]['is_list']

        if isinstance(field_value, ast.AST):
            child = ASTNode(field_type, field_name)
            child.add_child(python_ast_to_parse_tree(field_value))
        elif type(field_value) is str or type(field_value) is int or \
                        type(field_value) is float or type(field_value) is object or \
                        type(field_value) is bool:
            child = ASTNode(type(field_value), field_name, value=field_value)
        elif is_list_field:
            list_node_type = typename(field_type) + '*'
            child = ASTNode(list_node_type, field_name)
            for n in field_value:
                if field_type in {ast.comprehension, ast.excepthandler, ast.arguments, ast.keyword, ast.alias}:
                    child.add_child(python_ast_to_parse_tree(n))
                else:
                    intermediate_node = ASTNode(field_type)
                    if field_type is str:
                        intermediate_node.value = n
                    else:
                        intermediate_node.add_child(python_ast_to_parse_tree(n))
                    child.add_child(intermediate_node)

        else:
            raise RuntimeError('unknown AST node field!')

        tree.add_child(child)

    return tree

def decode_rule_to_tree(rules, root, rule_num=0):
    rule = rules[rule_num]
    node_types_labels = parse_rule(rule)
    for node_type, node_label in node_types_labels:
        if is_builtin_type(node_type):
            node_val = extract_val_GenToken(rules[rule_num + 1])
            rule_num += 2 #skip node_val and GenToken[<eob>]
            child_node = ASTNode(node_type, node_label, node_type(node_val))
        elif is_terminal_ast_type(node_type) or node_type == 'epsilon':
            rule_num += 1 #skip GenToken[<eob>]
            child_node = ASTNode(node_type)
        else:            
            rule_num += 1
            child_node = ASTNode(node_type, node_label)
            child_node, rule_num = decode_rule_to_tree(rules, child_node, rule_num)
            
        root.add_child(child_node)
    return root, rule_num

def parse_tree_to_python_ast(tree):
    node_type = tree.type
    node_label = tree.label

    # remove root
    if node_type == 'root':
        return parse_tree_to_python_ast(tree.children[0])

    ast_node = node_type()
    node_type_name = typename(node_type)

    # if it's a compositional AST node, populate its children nodes,
    # fill fields with empty(default) values otherwise
    if node_type_name in PY_AST_NODE_FIELDS:
        fields_info = PY_AST_NODE_FIELDS[node_type_name]

        for child_node in tree.children:
            # if it's a compositional leaf
            if child_node.type == 'epsilon':
                continue

            field_type = child_node.type
            field_label = child_node.label
            field_entry = fields_info[field_label]
            is_list = field_entry['is_list']

            if is_list:
                field_type = field_entry['type']
                field_value = []

                if field_type in {ast.comprehension, ast.excepthandler, ast.arguments, ast.keyword, ast.alias}:
                    nodes_in_list = child_node.children
                    for sub_node in nodes_in_list:
                        sub_node_ast = parse_tree_to_python_ast(sub_node)
                        field_value.append(sub_node_ast)
                else:  # expr stuffs
                    inter_nodes = child_node.children
                    for inter_node in inter_nodes:
                        if inter_node.value is None:
                            assert len(inter_node.children) == 1
                            sub_node_ast = parse_tree_to_python_ast(inter_node.children[0])
                            field_value.append(sub_node_ast)
                        else:
                            assert len(inter_node.children) == 0
                            field_value.append(inter_node.value)
            else:
                # this node either holds a value, or is an non-terminal
                if child_node.value is None:
                    assert len(child_node.children) == 1
                    field_value = parse_tree_to_python_ast(child_node.children[0])
                else:
                    assert child_node.is_leaf
                    field_value = child_node.value

            setattr(ast_node, field_label, field_value)

    for field in ast_node._fields:
        if not hasattr(ast_node, field) and not field in NODE_FIELD_BLACK_LIST:
            if fields_info and fields_info[field]['is_list'] and not fields_info[field]['is_optional']:
                setattr(ast_node, field, list())
            else:
                setattr(ast_node, field, None)

    return ast_node

p_elif = re.compile(r'^elif\s?')
p_else = re.compile(r'^else\s?')
p_try = re.compile(r'^try\s?')
p_except = re.compile(r'^except\s?')
p_finally = re.compile(r'^finally\s?')
p_decorator = re.compile(r'^@.*')

def sugar_code(code):
    if p_elif.match(code):
        code = 'if True: pass\n' + code

    if p_else.match(code):
        code = 'if True: pass\n' + code

    if p_try.match(code):
        code = code + 'pass\nexcept: pass'
    elif p_except.match(code):
        code = 'try: pass\n' + code
    elif p_finally.match(code):
        code = 'try: pass\n' + code

    if p_decorator.match(code):
        code = code + '\ndef dummy(): pass'

    if code[-1] == ':':
        code = code + 'pass'

    return code


def de_sugar_code(code, ref_raw_code):
    code = code.strip()
    if code.endswith('def dummy():\n    pass'):
        code = code.replace('def dummy():\n    pass', '')

    if p_elif.match(ref_raw_code):
        # remove leading if true
        code = code.replace('if True:\n    pass', '')
    elif p_else.match(ref_raw_code):
        # remove leading if true
        code = code.replace('if True:\n    pass', '')

    # try/catch/except stuff
    if p_try.match(ref_raw_code):
        code = code.replace('\nexcept:\n    pass', '')
    elif p_except.match(ref_raw_code):
        code = code.replace('try:\n    pass', '')
    elif p_finally.match(ref_raw_code):
        code = code.replace('try:\n    pass', '')

    # remove ending pass
    if code.endswith(':\n    pass'):
        code = code[:-len('\n    pass')]

    return code.strip()


def add_root(tree):
    root_node = ASTNode('root')
    root_node.add_child(tree)

    return root_node


def parse(code):
    """
    parse a python code into a tree structure
    code -> AST tree -> AST tree to internal tree structure
    """

    code = sugar_code(code)
    py_ast = ast.parse(code)

    tree = python_ast_to_parse_tree(py_ast.body[0])

    tree = add_root(tree)

    return tree


def parse_raw(code):
    py_ast = ast.parse(code)

    tree = python_ast_to_parse_tree(py_ast.body[0])

    tree = add_root(tree)

    return tree

if __name__ == '__main__':
    code = """
class Demonwrath(SpellCard):
    def __init__(self):
        super().__init__("Demonwrath", 3, CHARACTER_CLASS.WARLOCK, CARD_RARITY.RARE)

    def use(self, player, game):
        super().use(player, game)
        targets = copy.copy(game.other_player.minions)
        targets.extend(game.current_player.minions)
        for minion in targets:
            if minion.card.minion_type is not MINION_TYPE.DEMON:
                minion.damage(player.effective_spell_damage(2), self)
"""
    code = """raise ImproperlyConfigured ( "You must define a '%s' cache" % DEFAULT_CACHE_ALIAS )"""
    parse_tree = parse(code)

    rules =  parse_tree.to_rules()

    root_node = ASTNode('root')
    root_node, _ = decode_rule_to_tree(rules, root_node)

    
    ast_tree = parse_tree_to_python_ast(root_node)
    out_code = astor.to_source(ast_tree)
    format_code = de_sugar_code(out_code, code)
    final_code = format_code.replace('\n', '')
    print(final_code)
