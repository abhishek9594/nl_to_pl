#adapted from https://github.com/pcyin/NL2code
from grammar import type_str_to_type

def typename(x):
    if isinstance(x, str):
        return x
    return x.__name__

def type_of(x):
    #x is a str variable eg.'4'
    if x in ['True', 'False']: return bool
    try:
        typed_var = int(x)
    except ValueError:
        try:
            typed_var = float(x)
        except ValueError:
            typed_var = x
    return type(typed_var)

def parse_rule(rule):
    """
    rule = head_node_tp -> child1_tp[label1?] child2_tp[label2?] ...
    """
    head_node_end = rule.find('->') - 1
    assert head_node_end > 0
    child_start = head_node_end + 4 #len(\s->\s) = 4

    children = rule[child_start: ].split(' ')
    children_types_labels = []
    for child in children:
        #child_type[child_label?]
        if '[' in child:
            child_type_end = child.find('[')
            child_type = type_str_to_type(child[: child_type_end])
            child_label = child[child_type_end + 1 : -1] #skip ']'
        else:
            child_type = type_str_to_type(child)
            child_label = None

        children_types_labels.append((child_type, child_label))
    return children_types_labels

def extract_val_GenToken(GenTokenStr):
    """
    GenTokenStr: GenToken[str]
    """
    return GenTokenStr[len('GenToken['): -1]
