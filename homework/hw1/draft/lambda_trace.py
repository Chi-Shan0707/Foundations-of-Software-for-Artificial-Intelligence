import sys
import trace


# Church numerals (same core content as lambda_single.py)
ZERO = lambda f: (lambda x: x)
ONE = lambda f: (lambda x: f(x))
TWO = lambda f: (lambda x: f(f(x)))
SUCC = lambda n: (lambda f: (lambda x: f(n(f)(x))))
ADD = lambda m: (lambda n: (lambda f: (lambda x: m(f)(n(f)(x)))))

to_int = lambda n: n(lambda k: k + 1)(0)


def run_demo() -> None:
    # Curried style: first ADD(ONE), then apply TWO.
    three = ADD(ONE)(TWO)
    four = SUCC(three)

    print("ADD(ONE)(TWO) =", to_int(three))
    print("SUCC(ADD(ONE)(TWO)) =", to_int(four))


def run_with_trace() -> None:
    # trace=True prints executed lines in runtime order.
    # ignoredirs filters most stdlib noise, so students focus on this file.
    tracer = trace.Trace(
        trace=True,
        count=False,
        ignoredirs=[sys.prefix, sys.exec_prefix],
    )
    tracer.runfunc(run_demo)


if __name__ == "__main__":
    run_with_trace()
