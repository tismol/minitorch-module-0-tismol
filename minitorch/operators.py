"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable, List, TypeVar


#
# Implementation of a prelude of elementary functions.

def mul(x: float, y: float) -> float:
    return x * y


def id(x: float) -> float:
    return x


def add(x: float, y: float) -> float:
    return x + y


def neg(x: float) -> float:
    return -x


def lt(x: float, y: float) -> bool:
    return x < y


def eq(x: float, y: float) -> bool:
    return x == y


def max(x: float, y: float) -> float:
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def relu(x: float) -> float:
    return x if x > 0.0 else 0.0


def log(x: float) -> float:
    return math.log(x)


def exp(x: float) -> float:
    return math.exp(x)


def log_back(x: float, grad: float) -> float:
    return grad / x


def inv(x: float) -> float:
    return 1.0 / x


def inv_back(x: float, grad: float) -> float:
    return -grad / (x * x)


def relu_back(x: float, grad: float) -> float:
    return grad if x > 0.0 else 0.0


#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


def map(fn: Callable[[T], U], xs: Iterable[T]) -> List[U]:
    return [fn(x) for x in xs]


def zipWith(fn: Callable[[T, U], V], xs: Iterable[T], ys: Iterable[U]) -> List[V]:
    out: List[V] = []
    for x, y in zip(xs, ys):
        out.append(fn(x, y))
    return out


def reduce(fn: Callable[[T, U], T], xs: Iterable[U], start: T) -> T:
    res = start
    for x in xs:
        res = fn(res, x)
    return res


def negList(xs: Iterable[float]) -> List[float]:
    return map(neg, xs)


def addLists(xs: Iterable[float], ys: Iterable[float]) -> List[float]:
    return zipWith(add, xs, ys)


def sum(xs: Iterable[float]) -> float:
    return reduce(add, xs, 0.0)


def prod(xs: Iterable[float]) -> float:
    return reduce(mul, xs, 1.0)
