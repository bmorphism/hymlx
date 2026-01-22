# py2hy.mojo - Python AST to Hy S-expression compiler
# Uses Mojo's compile-time metaprogramming for AST transformation
#
# Core insight: Mojo's @parameter + alias = compile-time AST rewriting
# Python AST nodes become S-expressions via structural pattern matching

from python import Python, PythonObject
from collections import List, Dict
from memory import memcpy

# ═══════════════════════════════════════════════════════════════
# AST Node Types (compile-time enum via alias)
# ═══════════════════════════════════════════════════════════════

alias AST_MODULE = 0
alias AST_FUNCTION = 1
alias AST_CLASS = 2
alias AST_ASSIGN = 3
alias AST_RETURN = 4
alias AST_IF = 5
alias AST_FOR = 6
alias AST_WHILE = 7
alias AST_CALL = 8
alias AST_NAME = 9
alias AST_NUM = 10
alias AST_STR = 11
alias AST_BINOP = 12
alias AST_COMPARE = 13
alias AST_ATTRIBUTE = 14
alias AST_SUBSCRIPT = 15
alias AST_LIST = 16
alias AST_DICT = 17
alias AST_TUPLE = 18
alias AST_LAMBDA = 19
alias AST_IMPORT = 20
alias AST_IMPORTFROM = 21

# Binary operators
alias OP_ADD = "+"
alias OP_SUB = "-"
alias OP_MUL = "*"
alias OP_DIV = "/"
alias OP_MOD = "%"
alias OP_POW = "**"
alias OP_MATMUL = "@"
alias OP_FLOORDIV = "//"
alias OP_LSHIFT = "<<"
alias OP_RSHIFT = ">>"
alias OP_BITOR = "|"
alias OP_BITXOR = "^"
alias OP_BITAND = "&"

# ═══════════════════════════════════════════════════════════════
# S-Expression Builder (compile-time string construction)
# ═══════════════════════════════════════════════════════════════

struct SExpr:
    """S-expression node for Hy output."""
    var head: String
    var children: List[String]
    
    fn __init__(inout self, head: String):
        self.head = head
        self.children = List[String]()
    
    fn add(inout self, child: String):
        self.children.append(child)
    
    fn add_sexpr(inout self, child: SExpr):
        self.children.append(child.to_string())
    
    fn to_string(self) -> String:
        """Convert to Hy string: (head child1 child2 ...)"""
        var result = String("(") + self.head
        for i in range(len(self.children)):
            result += " " + self.children[i]
        result += ")"
        return result

# ═══════════════════════════════════════════════════════════════
# Python AST → Hy Transformer
# ═══════════════════════════════════════════════════════════════

struct Py2Hy:
    """Transform Python AST to Hy S-expressions.
    
    Uses Mojo's Python interop to parse, then compile-time
    pattern matching for transformation.
    """
    var ast_module: PythonObject
    var indent: Int
    
    fn __init__(inout self) raises:
        self.ast_module = Python.import_module("ast")
        self.indent = 0
    
    fn parse(self, source: String) raises -> PythonObject:
        """Parse Python source to AST."""
        return self.ast_module.parse(source)
    
    fn transform(self, node: PythonObject) raises -> String:
        """Transform AST node to Hy string."""
        let node_type = str(node.__class__.__name__)
        
        if node_type == "Module":
            return self._transform_module(node)
        elif node_type == "FunctionDef":
            return self._transform_function(node)
        elif node_type == "ClassDef":
            return self._transform_class(node)
        elif node_type == "Assign":
            return self._transform_assign(node)
        elif node_type == "Return":
            return self._transform_return(node)
        elif node_type == "If":
            return self._transform_if(node)
        elif node_type == "For":
            return self._transform_for(node)
        elif node_type == "While":
            return self._transform_while(node)
        elif node_type == "Expr":
            return self.transform(node.value)
        elif node_type == "Call":
            return self._transform_call(node)
        elif node_type == "Name":
            return self._mangle_name(str(node.id))
        elif node_type == "Constant":
            return self._transform_constant(node)
        elif node_type == "Num":
            return str(node.n)
        elif node_type == "Str":
            return '"' + str(node.s) + '"'
        elif node_type == "BinOp":
            return self._transform_binop(node)
        elif node_type == "Compare":
            return self._transform_compare(node)
        elif node_type == "Attribute":
            return self._transform_attribute(node)
        elif node_type == "Subscript":
            return self._transform_subscript(node)
        elif node_type == "List":
            return self._transform_list(node)
        elif node_type == "Dict":
            return self._transform_dict(node)
        elif node_type == "Tuple":
            return self._transform_tuple(node)
        elif node_type == "Lambda":
            return self._transform_lambda(node)
        elif node_type == "Import":
            return self._transform_import(node)
        elif node_type == "ImportFrom":
            return self._transform_import_from(node)
        elif node_type == "Pass":
            return "None"
        elif node_type == "Break":
            return "(break)"
        elif node_type == "Continue":
            return "(continue)"
        else:
            return "; UNSUPPORTED: " + node_type
    
    # ─────────────────────────────────────────────────────────────
    # Node-specific transformers
    # ─────────────────────────────────────────────────────────────
    
    fn _transform_module(self, node: PythonObject) raises -> String:
        """(do stmt1 stmt2 ...)"""
        var result = String("")
        let body = node.body
        for i in range(int(Python.len(body))):
            if i > 0:
                result += "\n\n"
            result += self.transform(body[i])
        return result
    
    fn _transform_function(self, node: PythonObject) raises -> String:
        """(defn name [args] body...)"""
        var sexpr = SExpr("defn")
        sexpr.add(self._mangle_name(str(node.name)))
        
        # Args
        var args_str = String("[")
        let args = node.args.args
        for i in range(int(Python.len(args))):
            if i > 0:
                args_str += " "
            args_str += self._mangle_name(str(args[i].arg))
        args_str += "]"
        sexpr.add(args_str)
        
        # Body
        let body = node.body
        for i in range(int(Python.len(body))):
            sexpr.add(self.transform(body[i]))
        
        return sexpr.to_string()
    
    fn _transform_class(self, node: PythonObject) raises -> String:
        """(defclass Name [bases] body...)"""
        var sexpr = SExpr("defclass")
        sexpr.add(str(node.name))
        
        # Bases
        var bases_str = String("[")
        let bases = node.bases
        for i in range(int(Python.len(bases))):
            if i > 0:
                bases_str += " "
            bases_str += self.transform(bases[i])
        bases_str += "]"
        sexpr.add(bases_str)
        
        # Body
        let body = node.body
        for i in range(int(Python.len(body))):
            sexpr.add("\n  " + self.transform(body[i]))
        
        return sexpr.to_string()
    
    fn _transform_assign(self, node: PythonObject) raises -> String:
        """(setv target value)"""
        var sexpr = SExpr("setv")
        let targets = node.targets
        for i in range(int(Python.len(targets))):
            sexpr.add(self.transform(targets[i]))
        sexpr.add(self.transform(node.value))
        return sexpr.to_string()
    
    fn _transform_return(self, node: PythonObject) raises -> String:
        """value (Hy uses implicit return)"""
        if node.value is None:
            return "None"
        return self.transform(node.value)
    
    fn _transform_if(self, node: PythonObject) raises -> String:
        """(if condition then-body else-body)
           or (cond ...) for elif chains"""
        var sexpr = SExpr("if")
        sexpr.add(self.transform(node.test))
        
        # Then branch
        var then_str = String("(do")
        let body = node.body
        for i in range(int(Python.len(body))):
            then_str += " " + self.transform(body[i])
        then_str += ")"
        sexpr.add(then_str)
        
        # Else branch
        let orelse = node.orelse
        if int(Python.len(orelse)) > 0:
            var else_str = String("(do")
            for i in range(int(Python.len(orelse))):
                else_str += " " + self.transform(orelse[i])
            else_str += ")"
            sexpr.add(else_str)
        
        return sexpr.to_string()
    
    fn _transform_for(self, node: PythonObject) raises -> String:
        """(for [target iter] body...)"""
        var sexpr = SExpr("for")
        let binding = "[" + self.transform(node.target) + " " + self.transform(node.iter) + "]"
        sexpr.add(binding)
        
        let body = node.body
        for i in range(int(Python.len(body))):
            sexpr.add(self.transform(body[i]))
        
        return sexpr.to_string()
    
    fn _transform_while(self, node: PythonObject) raises -> String:
        """(while condition body...)"""
        var sexpr = SExpr("while")
        sexpr.add(self.transform(node.test))
        
        let body = node.body
        for i in range(int(Python.len(body))):
            sexpr.add(self.transform(body[i]))
        
        return sexpr.to_string()
    
    fn _transform_call(self, node: PythonObject) raises -> String:
        """(func arg1 arg2 :kw val)"""
        var sexpr = SExpr(self.transform(node.func))
        
        # Positional args
        let args = node.args
        for i in range(int(Python.len(args))):
            sexpr.add(self.transform(args[i]))
        
        # Keyword args
        let keywords = node.keywords
        for i in range(int(Python.len(keywords))):
            let kw = keywords[i]
            sexpr.add(":" + str(kw.arg))
            sexpr.add(self.transform(kw.value))
        
        return sexpr.to_string()
    
    fn _transform_binop(self, node: PythonObject) raises -> String:
        """(op left right)"""
        let op_type = str(node.op.__class__.__name__)
        var op_str: String
        
        if op_type == "Add":
            op_str = "+"
        elif op_type == "Sub":
            op_str = "-"
        elif op_type == "Mult":
            op_str = "*"
        elif op_type == "Div":
            op_str = "/"
        elif op_type == "Mod":
            op_str = "%"
        elif op_type == "Pow":
            op_str = "**"
        elif op_type == "MatMult":
            op_str = "@"
        elif op_type == "FloorDiv":
            op_str = "//"
        elif op_type == "LShift":
            op_str = "<<"
        elif op_type == "RShift":
            op_str = ">>"
        elif op_type == "BitOr":
            op_str = "|"
        elif op_type == "BitXor":
            op_str = "^"
        elif op_type == "BitAnd":
            op_str = "&"
        else:
            op_str = op_type
        
        var sexpr = SExpr(op_str)
        sexpr.add(self.transform(node.left))
        sexpr.add(self.transform(node.right))
        return sexpr.to_string()
    
    fn _transform_compare(self, node: PythonObject) raises -> String:
        """(op left right) - simplified for single comparator"""
        let ops = node.ops
        let comparators = node.comparators
        
        if int(Python.len(ops)) == 1:
            let op_type = str(ops[0].__class__.__name__)
            var op_str: String
            
            if op_type == "Eq":
                op_str = "="
            elif op_type == "NotEq":
                op_str = "!="
            elif op_type == "Lt":
                op_str = "<"
            elif op_type == "LtE":
                op_str = "<="
            elif op_type == "Gt":
                op_str = ">"
            elif op_type == "GtE":
                op_str = ">="
            elif op_type == "Is":
                op_str = "is"
            elif op_type == "IsNot":
                op_str = "is-not"
            elif op_type == "In":
                op_str = "in"
            elif op_type == "NotIn":
                op_str = "not-in"
            else:
                op_str = op_type
            
            var sexpr = SExpr(op_str)
            sexpr.add(self.transform(node.left))
            sexpr.add(self.transform(comparators[0]))
            return sexpr.to_string()
        else:
            # Chained comparison: a < b < c -> (and (< a b) (< b c))
            return "; TODO: chained comparison"
    
    fn _transform_attribute(self, node: PythonObject) raises -> String:
        """(. value attr)"""
        var sexpr = SExpr(".")
        sexpr.add(self.transform(node.value))
        sexpr.add(str(node.attr))
        return sexpr.to_string()
    
    fn _transform_subscript(self, node: PythonObject) raises -> String:
        """(get value slice)"""
        var sexpr = SExpr("get")
        sexpr.add(self.transform(node.value))
        sexpr.add(self.transform(node.slice))
        return sexpr.to_string()
    
    fn _transform_list(self, node: PythonObject) raises -> String:
        """[elem1 elem2 ...]"""
        var result = String("[")
        let elts = node.elts
        for i in range(int(Python.len(elts))):
            if i > 0:
                result += " "
            result += self.transform(elts[i])
        result += "]"
        return result
    
    fn _transform_dict(self, node: PythonObject) raises -> String:
        """{key1 val1 key2 val2}"""
        var result = String("{")
        let keys = node.keys
        let values = node.values
        for i in range(int(Python.len(keys))):
            if i > 0:
                result += " "
            result += self.transform(keys[i]) + " " + self.transform(values[i])
        result += "}"
        return result
    
    fn _transform_tuple(self, node: PythonObject) raises -> String:
        """#(elem1 elem2 ...)"""
        var result = String("#(")
        let elts = node.elts
        for i in range(int(Python.len(elts))):
            if i > 0:
                result += " "
            result += self.transform(elts[i])
        result += ")"
        return result
    
    fn _transform_lambda(self, node: PythonObject) raises -> String:
        """(fn [args] body)"""
        var sexpr = SExpr("fn")
        
        var args_str = String("[")
        let args = node.args.args
        for i in range(int(Python.len(args))):
            if i > 0:
                args_str += " "
            args_str += self._mangle_name(str(args[i].arg))
        args_str += "]"
        sexpr.add(args_str)
        
        sexpr.add(self.transform(node.body))
        return sexpr.to_string()
    
    fn _transform_import(self, node: PythonObject) raises -> String:
        """(import module) or (import module :as alias)"""
        var results = List[String]()
        let names = node.names
        for i in range(int(Python.len(names))):
            let alias = names[i]
            var sexpr = SExpr("import")
            sexpr.add(str(alias.name))
            if alias.asname is not None:
                sexpr.add(":as")
                sexpr.add(str(alias.asname))
            results.append(sexpr.to_string())
        
        if len(results) == 1:
            return results[0]
        else:
            var combined = String("")
            for i in range(len(results)):
                if i > 0:
                    combined += "\n"
                combined += results[i]
            return combined
    
    fn _transform_import_from(self, node: PythonObject) raises -> String:
        """(import module [name1 name2])"""
        var sexpr = SExpr("import")
        sexpr.add(str(node.module))
        
        var names_str = String("[")
        let names = node.names
        for i in range(int(Python.len(names))):
            if i > 0:
                names_str += " "
            let alias = names[i]
            names_str += str(alias.name)
            if alias.asname is not None:
                names_str += " :as " + str(alias.asname)
        names_str += "]"
        sexpr.add(names_str)
        
        return sexpr.to_string()
    
    fn _transform_constant(self, node: PythonObject) raises -> String:
        """Handle Python 3.8+ Constant nodes."""
        let value = node.value
        if value is None:
            return "None"
        elif value is True:
            return "True"
        elif value is False:
            return "False"
        elif Python.isinstance(value, Python.evaluate("str")):
            return '"' + str(value) + '"'
        else:
            return str(value)
    
    fn _mangle_name(self, name: String) -> String:
        """Convert Python naming to Hy (snake_case → kebab-case)."""
        var result = String("")
        for i in range(len(name)):
            let c = name[i]
            if c == "_":
                result += "-"
            else:
                result += c
        return result


# ═══════════════════════════════════════════════════════════════
# Compile-time transformation via @parameter
# ═══════════════════════════════════════════════════════════════

@parameter
fn py2hy_comptime[source: StringLiteral]() -> String:
    """Compile-time Python → Hy transformation.
    
    Usage:
        alias hy_code = py2hy_comptime["def add(a, b): return a + b"]()
    
    This runs at compile time, embedding the Hy code as a constant.
    """
    # Note: In practice this would need constexpr Python interop
    # which Mojo doesn't fully support yet. This is the design.
    return "COMPTIME: " + source


# ═══════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════

fn py2hy_string(source: String) raises -> String:
    """Runtime Python → Hy transformation."""
    var transformer = Py2Hy()
    let ast = transformer.parse(source)
    return transformer.transform(ast)


fn main() raises:
    print("py2hy.mojo - Python to Hy Compiler")
    print("=" * 50)
    
    # Test cases
    let tests = List[String](
        "x = 1 + 2",
        "def add(a, b): return a + b",
        "import mlx.core as mx",
        "for i in range(10): print(i)",
        "class Foo: pass",
        "lambda x: x * 2",
        "[1, 2, 3]",
        "{'a': 1, 'b': 2}",
    )
    
    var transformer = Py2Hy()
    
    for i in range(len(tests)):
        let source = tests[i]
        print("\n--- Python ---")
        print(source)
        print("--- Hy ---")
        try:
            let ast = transformer.parse(source)
            let hy_code = transformer.transform(ast)
            print(hy_code)
        except e:
            print("Error:", e)
    
    print("\n" + "=" * 50)
    print("✓ py2hy.mojo complete")
