"""
Pure Python Lambda Calculus - Church Numerals & Booleans
纯 Lambda 演算实现：Church 数和布尔值
"""

# ============================================================
# Church Numerals
# ============================================================

ZERO = lambda f: lambda x: x
ONE = lambda f: lambda x: f(x)
TWO = lambda f: lambda x: f(f(x))
THREE = lambda f: lambda x: f(f(f(x)))

# Identity function
IDENTITY = lambda x: x

# Successor: SUCC(n) = n + 1
SUCC = lambda n: lambda f: lambda x: f(n(f)(x))

# Addition: ADD(m)(n) = m + n
ADD = lambda m: lambda n: lambda f: lambda x: m(f)(n(f)(x))

# Multiplication: MUL(m)(n) = m * n
MUL = lambda m: lambda n: lambda f: m(n(f))

# Exponentiation: POW(m)(n) = m^n
POW = lambda m: lambda n: n(m)

# Predecessor (decrement): PRED(n) = n - 1
PRED = lambda n: lambda f: lambda x: n(lambda g: lambda h: h(g(f)))(lambda u: x)(lambda u: u)

# Subtraction: SUB(m)(n) = m - n
SUB = lambda m: lambda n: n(PRED)(m)


# ============================================================
# Church Booleans
# ============================================================

# TRUE returns first argument
TRUE = lambda t: lambda f: t

# FALSE returns second argument
FALSE = lambda t: lambda f: f

# IF condition: IF(p)(a)(b) = if p then a else b
# In Church encoding, the predicate p is already a selector
IF = lambda p: lambda a: lambda b: p(a)(b)

# Logical operations
AND = lambda p: lambda q: p(q)(p)
OR = lambda p: lambda q: p(p)(q)
NOT = lambda p: p(FALSE)(TRUE)


# ============================================================
# Church Pairs
# ============================================================

PAIR = lambda x: lambda y: lambda f: f(x)(y)
FIRST = lambda p: p(TRUE)
SECOND = lambda p: p(FALSE)


# ============================================================
# Conversion Functions
# ============================================================

# Convert Church numeral to int
to_int = lambda n: n(lambda k: k + 1)(0)

# Convert int to Church numeral
def int_to_church(n):
    """Convert a Python int to Church numeral"""
    if n == 0:
        return ZERO
    return SUCC(int_to_church(n - 1))

# Convert Church boolean to bool
to_bool = lambda b: b(True)(False)

# Convert bool to Church boolean
def bool_to_church(b):
    return TRUE if b else FALSE


# ============================================================
# Test & Demo
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Pure Python Lambda Calculus Demo")
    print("=" * 50)

    # Test Church Numerals
    print("\n--- Church Numerals ---")
    for i, church in enumerate([ZERO, ONE, TWO, THREE]):
        print(f"{i} = {to_int(church)}")

    print(f"\nSUCC(THREE) = {to_int(SUCC(THREE))}")
    print(f"ADD(ONE)(TWO) = {to_int(ADD(ONE)(TWO))}")
    print(f"MUL(TWO)(THREE) = {to_int(MUL(TWO)(THREE))}")
    print(f"POW(TWO)(THREE) = {to_int(POW(TWO)(THREE))}")  # 2^3 = 8
    print(f"PRED(THREE) = {to_int(PRED(THREE))}")
    print(f"SUB(THREE)(ONE) = {to_int(SUB(THREE)(ONE))}")

    # Test Church Booleans
    print("\n--- Church Booleans ---")
    print(f"TRUE = {to_bool(TRUE)}")
    print(f"FALSE = {to_bool(FALSE)}")
    print(f"AND(TRUE)(FALSE) = {to_bool(AND(TRUE)(FALSE))}")
    print(f"OR(TRUE)(FALSE) = {to_bool(OR(TRUE)(FALSE))}")
    print(f"NOT(TRUE) = {to_bool(NOT(TRUE))}")

    # Test IF condition
    print("\n--- IF Condition ---")
    print(f"IF(TRUE)(1)(2) = {IF(TRUE)(1)(2)}")
    print(f"IF(FALSE)(1)(2) = {IF(FALSE)(1)(2)}")

    # Test Church Pairs
    print("\n--- Church Pairs ---")
    pair = PAIR(3)(5)
    print(f"PAIR(3)(5): FIRST = {FIRST(pair)}, SECOND = {SECOND(pair)}")

    # Dynamic computation
    print("\n--- Dynamic Computation ---")
    five = int_to_church(5)
    seven = int_to_church(7)
    print(f"5 + 7 = {to_int(ADD(five)(seven))}")
    print(f"5 * 7 = {to_int(MUL(five)(seven))}")
