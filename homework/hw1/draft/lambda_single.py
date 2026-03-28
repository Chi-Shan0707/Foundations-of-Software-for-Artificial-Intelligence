ZERO  = lambda f: (lambda x: x)         # 0 = λf.λx.x        (apply f zero times to x)
ONE   = lambda f: (lambda x: f(x))      # 1 = λf.λx.f x      (apply f once to x)
TWO   = lambda f: (lambda x: f(f(x)))   # 2 = λf.λx.f (f x)  (apply f twice to x)
SUCC  = lambda n: (lambda f: (lambda x: f(n(f)(x))))      # SUCC n = λf.λx. f (n f x) （successor: n -> n+1 (one more application of f) ）
ADD   = lambda m: (lambda n: (lambda f: (lambda x: m(f)(n(f)(x)))))# ADD m n = λf.λx. m f (n f x) （addition: apply n times, then m times; total m+n ）
to_int = lambda n: n(lambda k: k + 1)(0)

# Church-style application is curried: ADD(ONE)(TWO), not ADD(ONE, TWO).
THREE = ADD(ONE)(TWO)
THREE_INT = to_int(THREE)

import dis
dis.dis(to_int(THREE))

