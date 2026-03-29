import ast
import inspect
import ctypes

# ============================================================
# Native primitive kernels (precompiled shared library)
# ============================================================

lib = ctypes.CDLL("./liblambda.so")

lib.church_add.argtypes = [ctypes.c_int, ctypes.c_int]
lib.church_add.restype = ctypes.c_int

lib.church_mul.argtypes = [ctypes.c_int, ctypes.c_int]
lib.church_mul.restype = ctypes.c_int

lib.church_succ.argtypes = [ctypes.c_int]
lib.church_succ.restype = ctypes.c_int


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

        if isinstance(node, ast.Constant):
            return node.value

        if isinstance(node, ast.Call):
            # Handle curried calls: ADD(m)(n)
            if isinstance(node.func, ast.Call):
                outer = node.func
                if hasattr(outer.func, 'id'):
                    op_name = outer.func.id
                    if op_name in ['ADD', 'MUL']:
                        m = lower(outer.args[0], env)
                        n = lower(node.args[0], env)
                        if op_name == 'ADD':
                            return lib.church_add(m, n)
                        elif op_name == 'MUL':
                            return lib.church_mul(m, n)

            # Handle simple calls: SUCC(n)
            if hasattr(node.func, 'id'):
                op_name = node.func.id
                if op_name == 'SUCC':
                    n = lower(node.args[0], env)
                    return lib.church_succ(n)

        raise NotImplementedError("Unsupported AST node")

    # -------- Runtime wrapper --------
    def wrapper(*args):
        env = dict(zip(arg_names, args))
        result = lower(expr, env)
        return result

    return wrapper


# ============================================================
# User-defined kernels (pure Python syntax)
# ============================================================

@kernel
def succ(n):
    return SUCC(n)

@kernel
def add(m, n):
    return ADD(m)(n)

@kernel
def mul(m, n):
    return MUL(m)(n)


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    print("succ(2) =", succ(2))
    print("add(1, 2) =", add(1, 2))
    print("mul(2, 3) =", mul(2, 3))
