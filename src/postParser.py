#!/usr/bin/env python
import ast

def parseBoolOp(boolop):
    if isinstance(boolop, ast.And):
        return 'And'
    else:
        return 'Or'

def parseOp(op):
    if isinstance(op, ast.Add):
        return 'Add'
    elif isinstance(op, ast.Sub):
        return 'Sub'
    elif isinstance(op, ast.Mult):
        return 'Mult'
    elif isinstance(op, ast.Div):
        return 'Div'
    elif isinstance(op, ast.Mod):
        return 'Mod'
    elif isinstance(op, ast.Pow):
        return 'Pow'
    elif isinstance(op, ast.LShift):
        return 'LShift'
    elif isinstance(op, ast.RShift):
        return 'RShift'
    elif isinstance(op, ast.BitOr):
        return 'BitOr'
    elif isinstance(op, ast.BitXor):
        return 'BitXor'
    elif isinstance(op, ast.BitAnd):
        return 'BitAnd'
    else:
        return 'FloorDiv'

def parseUnOp(unop):
    if isinstance(unop, ast.Invert):
        return 'Invert'
    elif isinstance(unop, ast.Not):
        return 'Not'
    elif isinstance(unop, ast.UAdd):
        return 'UAdd'
    else:
        return 'USub'

def parseCmpOp(cmpop):
    if isinstance(cmpop, ast.Eq):
        return 'Eq'
    elif isinstance(cmpop, ast.NotEq):
        return 'NotEq'
    elif isinstance(cmpop, ast.Lt):
        return 'Lt'
    elif isinstance(cmpop, ast.LtE):
        return 'LtE'
    elif isinstance(cmpop, ast.Gt):
        return 'Gt'
    elif isinstance(cmpop, ast.GtE):
        return 'GtE'
    elif isinstance(cmpop, ast.Is):
        return 'Is'
    elif isinstance(cmpop, ast.IsNot):
        return 'IsNot'
    elif isinstance(cmpop, ast.In):
        return 'In'
    else:
        return 'NotIn'

def parseBool(exp):
    """
    BoolOp(boolop op, expr * values)
    """
    return ' '.join([parseExp(value) for value in exp.values]) + ' ' + parseBoolOp(exp.op)

def parseBin(exp):
    """
    BinOp(expr left, operator op, expr right)
    """
    return parseExp(exp.left) + ' ' + parseExp(exp.right) + ' ' + parseOp(exp.op)

def parseUn(exp):
    """
    UnaryOp(unaryop op, expr operand)
    """
    return parseExp(exp.operand) + ' ' + parseUnOp(exp.op)

def parseLambda(exp):
    """
    Lambda(arguments args, expr body)
    """
    return parseArgs(exp.args) + ' ' + parseExp(exp.body)

def parseIf(exp):
    """
    IfExp(expr test, expr body, expr orelse)
    """
    return parseExp(exp.test) + ' ' + parseExp(exp.body) + ' ' + parseExp(exp.orelse)

def parseDict(exp):
    """
    Dict(expr* keys, expr* values)
    """
    return '( ' + ' '.join([parseExp(key) for key in exp.keys]) + ' ' + ' '.join([parseExp(value) for value in exp.values]) + ' Dict )'

def parseSet(exp):
    """
    Set(expr* elts)
    """
    return '( ' + ' '.join([parseExp(elt) for elt in exp.elts]) + ' Set )'

def parseListComp(exp):
    """
    ListComp(expr elt, comprehension* generators)
    """
    return '( ' + parseExp(exp.elt) + ' ' + ' '.join([parseComp(gen) for gen in exp.generators]) + ' List )'

def parseSetComp(exp):
    """
    SetComp(expr elt, comprehension* generators)
    """
    return '( ' + parseExp(exp.elt) + ' ' + ' '.join([parseComp(gen) for gen in exp.generators]) + ' Set )'

def parseDictComp(exp):
    """
    DictComp(expr key, expr value, comprehension* generators)
    """
    return '( ' + '( ' + parseExp(exp.key) + ' ' + parseExp(exp.value) + ' ) ' + ' '.join([parseComp(gen) for gen in exp.generators]) + ' Dict )'

def parseGenExp(exp):
    """
    GeneratorExp(expr elt, comprehension* generators)
    """
    return parseExp(exp.elt) + ' ' + ' '.join([parseComp(gen) for gen in exp.generators])

def parseYield(exp):
    """
    Yield(expr? value)
    """
    if exp.value: return parseExp(exp.value) + ' Yield'
    else: return 'Yield'

def parseCmp(exp):
    """
    Compare(expr left, cmpop* ops, expr* comparators)
    """
    return parseExp(exp.left) + ' ' + ' '.join([parseExp(comparator) for comparator in exp.comparators]) + ' ' +  ' '.join([parseCmpOp(cmpop) for cmpop in exp.ops])

def parseFunCall(exp):
    """
    Call(expr func, expr* args, keyword* keywords)
    """
    return parseExp(exp.func) + ' ' + ' '.join([parseExp(arg) for arg in exp.args]) + ' ' + ' '.join([parseKeyword(keyword) for keyword in exp.keywords]) + ' FunCall'

def parseNum(exp):
    """
    Num(object n)
    """
    return str(exp.n)

def parseRep(exp):
    """
    Repr(expr value)
    """
    return parseExp(exp.value)

def parseStr(exp):
    """
    Str(string s)
    """
    return '\'' + exp.s + '\''

def parseAttr(exp):
    """
    Attribute(expr value, ident attr, _)
    """
    return '( ' + parseExp(exp.value) + ' ' + exp.attr + ' . )'

def parseSubscript(exp):
    """
    Subscript(expr value, slice slice, _)
    """
    return '( ' + parseExp(exp.value) + ' ' + parseSlice(exp.slice) + ' _ ) '

def parseName(exp):
    """
    Name(ident id, _)
    """
    return exp.id

def parseList(exp):
    """
    List(expr* elts, _)
    """
    return 'List ( ' + ' '.join([parseExp(elt) for elt in exp.elts]) + ' )'

def parseTup(exp):
    """
    Tuple(expr* elts, _)
    """
    return '( ' + ' '.join([parseExp(elt) for elt in exp.elts]) + ' Tup )'

def parseSlice(exp):
    """
    Slice(expr? lower, expr? upper, expr? step)
    | ExtSlice(slice* dims)
    | Index(expr value)
    """
    if hasattr(exp, 'lower'):
        lower = parseExp(exp.lower) + ' ' if exp.lower else ''
        upper = parseExp(exp.upper) + ' ' if exp.upper else ''
        step = parseExp(exp.step) + ' ' if exp.step else ''
        return '( ' + lower + upper + step + ' : )'
    elif hasattr(exp, 'dims'):
        return ' '.join([parseSlice(dim) for dim in exp.dims])
    else:
        return parseExp(exp.value)

def parseArgs(args):
    """
    (expr* args, ident? vararg, ident? kwarg, expr* defaults)
    """
    vararg = args.vararg + ' ' if args.vararg else ''
    kwarg = args.kwarg + ' ' if args.kwarg else ''
    return ' '.join([parseExp(arg) for arg in args.args[:len(args.args) - len(args.defaults)]]) + ' ' + vararg + kwarg + ' '.join(['( ' + parseExp(defArg) + ' ' + parseExp(default) + ' = )' for defArg, default in zip(args.args[len(args.args) - len(args.defaults): ], args.defaults)])

def parseComp(comp):
    """
    (expr target, expr iter, expr* ifs)
    """
    return parseExp(comp.target) + ' ' + parseExp(comp.iter) + ' ' + ' '.join([parseExp(ifc) for ifc in comp.ifs])

def parseKeyword(keyword):
    """
    (ident arg, expr value)
    """
    return '( ' + str(keyword.arg) + ' ' + parseExp(keyword.value) + ' = )'

def parseAlias(alias):
    """
    (ident name, ident? asname)
    """
    if alias.asname:
        return '( ' + alias.name + ' ' + alias.asname + ' as )'
    else:
        return alias.name

def parseExp(exp):
    if isinstance(exp, ast.BoolOp):
        return parseBool(exp)
    elif isinstance(exp, ast.BinOp):
        return parseBin(exp)
    elif isinstance(exp, ast.UnaryOp):
        return parseUn(exp)
    elif isinstance(exp, ast.Lambda):
        return parseLambda(exp)
    elif isinstance(exp, ast.IfExp):
        return parseIf(exp) + ' If'
    elif isinstance(exp, ast.Dict):
        return parseDict(exp)
    elif isinstance(exp, ast.Set):
        return parseSet(exp)
    elif isinstance(exp, ast.ListComp):
        return parseListComp(exp)
    elif isinstance(exp, ast.SetComp):
        return parseSetComp(exp)
    elif isinstance(exp, ast.DictComp):
        return parseDictComp(exp)
    elif isinstance(exp, ast.GeneratorExp):
        return parseGenExp(exp)
    elif isinstance(exp, ast.Yield):
        return parseYield(exp)
    elif isinstance(exp, ast.Compare):
        return parseCmp(exp)
    elif isinstance(exp, ast.Call):
        return parseFunCall(exp)
    elif isinstance(exp, ast.Repr):
        return parseRep(exp)
    elif isinstance(exp, ast.Num):
        return parseNum(exp)
    elif isinstance(exp, ast.Str):
        return parseStr(exp)
    elif isinstance(exp, ast.Attribute):
        return parseAttr(exp)
    elif isinstance(exp, ast.Subscript):
        return parseSubscript(exp)
    elif isinstance(exp, ast.Name):
        return parseName(exp)
    elif isinstance(exp, ast.List):
        return parseList(exp)
    elif isinstance(exp, ast.Tuple):
        return parseTup(exp)
    else:
        #matches all the above case for 2.7.17 version
        return ' '

def parseSent(pl_sent):
    stmt = ast.parse(pl_sent).body[0]
    if isinstance(stmt, ast.FunctionDef):
        #FunctionDef(ident name, arguments args, _, _)
        return stmt.name + ' ' + parseArgs(stmt.args) + ' FunDef'

    elif isinstance(stmt, ast.ClassDef):
        #ClassDef(ident name, expr* bases, _, _)
        return stmt.name + ' ' + ' '.join([parseExp(base) for base in stmt.bases]) + ' ClassDef'

    elif isinstance(stmt, ast.Return):
        #Return(expr? value)
        if stmt.value: return parseExp(stmt.value) + ' Return'
        else: return 'Return'

    elif isinstance(stmt, ast.For):
        #For(expr target, expr iter, _, _)
        return parseExp(stmt.target) + ' ' + parseExp(stmt.iter) + ' For'

    elif isinstance(stmt, ast.While):
        #While(expr test, _, _)
        return parseExp(stmt.test) + ' While'

    elif isinstance(stmt, ast.If):
        return parseExp(stmt.test) + ' If'

    elif isinstance(stmt, ast.Assign):
        #Assign(expr* targets, expr value)
        return ' '.join([parseExp(target) for target in stmt.targets]) + ' ' + parseExp(stmt.value) + ' Assign'

    elif isinstance(stmt, ast.AugAssign):
        #AugAssign(expr target, operator op, expr value)
        return parseExp(stmt.target) + ' ' + parseOp(stmt.op) + ' ' + parseExp(stmt.value) + ' AugAssign'

    elif isinstance(stmt, ast.Expr):
        #Expr(expr value)
        return parseExp(stmt.value)

    elif isinstance(stmt, ast.Import):
        #Import(alias* names)
        return ' '.join([parseAlias(name) for name in stmt.names]) + ' Import'

    elif isinstance(stmt, ast.ImportFrom):
        #ImportFrom(ident? module, alias* names, int? level)
        module = stmt.module + ' ' if stmt.module else ''
        level = '( L ' + str(stmt.level) + ' ) ' if stmt.level else ''
        return module + ' '.join([parseAlias(name) for name in stmt.names]) + level + ' ImportFrom'

    else:
        return ' '
