import ast
import inspect
import ctypes

# ============================================================
# Native primitive kernels (precompiled shared library)
# ============================================================

lib = ctypes.CDLL("./liblambda.so")

lib.church_succ.argtypes = [ctypes.c_int]
lib.church_succ.restype = ctypes.c_int

lib.church_add.argtypes = [ctypes.c_int, ctypes.c_int]
lib.church_add.restype = ctypes.c_int


# ============================================================
# @kernel decorator: Python AST -> lowering -> native execution
# ============================================================

def kernel(func):
    """
    Treat the function body as an embedded DSL.
    The body is parsed, not executed.
    """
    # -------- Parse Python AST --------
    src = inspect.getsource(func)
    tree = ast.parse(src)
    func_def = tree.body[0]

    # Expect: return <expr>
    return_stmt = func_def.body[0]
    expr = return_stmt.value
    arg_names = [arg.arg for arg in func_def.args.args]

    # -------- Lowering: AST -> primitive ops --------
    def lower(node, env):
        if isinstance(node, ast.Name):
            return env[node.id]

        if isinstance(node, ast.Call):
            # Handle curried calls: ADD(m)(n)
            if isinstance(node.func, ast.Call):
                op_name = node.func.func.id
                if op_name == 'ADD':
                    m = lower(node.func.args[0], env)
                    n = lower(node.args[0], env)
                    return lib.church_add(m, n)

            # Handle simple calls: SUCC(n)
            if hasattr(node.func, 'id'):
                op_name = node.func.id
                if op_name == 'SUCC':
                    n = lower(node.args[0], env)
                    return lib.church_succ(n)

    # -------- Runtime wrapper --------
    def wrapper(*args):
        env = dict(zip(arg_names, args))
        result = lower(expr, env)
        return result

    return wrapper


# ============================================================
# Church Numerals (纯 Python，用于理解)
# ============================================================

ZERO = lambda f: lambda x: x
ONE = lambda f: lambda x: f(x)
TWO = lambda f: lambda x: f(f(x))
THREE = lambda f: lambda x: f(f(f(x)))

# Identity function
IDENTITY = lambda x: x

# Church Booleans
TRUE = lambda t: lambda f: t
FALSE = lambda t: lambda f: f

# IF condition: if p then a else b
# In Church encoding, the predicate p is a boolean that selects between a and b
IF = lambda p: lambda a: lambda b: p(a)(b)

SUCC = lambda n: lambda f: lambda x: f(n(f)(x))
ADD = lambda m: lambda n: lambda f: lambda x: m(f)(n(f)(x))

# 转换函数：Church numeral -> int
to_int = lambda n: n(lambda k: k + 1)(0)


# ============================================================
# User-defined kernels (使用 @kernel 加速)
# ============================================================

@kernel
def succ(n):
    return SUCC(n)

@kernel
def add(m, n):
    return ADD(m)(n)


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    # 纯 Python Lambda 演算
    print("=== Pure Python Lambda Calculus ===")
    print(f"to_int(ONE) = {to_int(ONE)}")
    print(f"to_int(TWO) = {to_int(TWO)}")
    print(f"to_int(SUCC(TWO)) = {to_int(SUCC(TWO))}")
    print(f"to_int(ADD(ONE)(TWO)) = {to_int(ADD(ONE)(TWO))}")

    # @kernel 加速版本
    print("\n=== @kernel Native Acceleration ===")
    print(f"succ(2) = {succ(2)}")
    print(f"add(1, 2) = {add(1, 2)}")
