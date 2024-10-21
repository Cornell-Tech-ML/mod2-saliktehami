"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, List


def mul(x: float, y: float) -> float:
    """Multiply two numbers.

    Args:
    ----
        x: A float.
        y: A float.

    Returns:
    -------
        A float representing the product of x and y.

    """
    return x * y


def id(x: float) -> float:
    """Return the identity of a number.

    Args:
    ----
        x: A float.

    Returns:
    -------
        A float representing the identity of x.

    """
    return x


def add(x: float, y: float) -> float:
    """Add two numbers.

    Args:
    ----
        x: A float.
        y: A float.

    Returns:
    -------
        A float representing the sum of x and y.

    """
    return float(x + y)


def neg(x: float) -> float:
    """Negate a number.

    Args:
    ----
        x: A float.

    Returns:
    -------
        A float representing the negation of x.

    """
    return -1.0 * x


def lt(x: float, y: float) -> float:
    """Compare two numbers.

    Args:
    ----
        x: A float.
        y: A float.

    Returns:
    -------
        A boolean representing whether x is less than y.

    """
    return float(x < y)


def eq(x: float, y: float) -> float:
    """Compare two numbers.

    Args:
    ----
        x: A float.
        y: A float.

    Returns:
    -------
        A boolean representing whether x is equal to y.

    """
    return float(x == y)


def max(x: float, y: float) -> float:
    """Return the maximum of two numbers.

    Args:
    ----
        x: A float.
        y: A float.

    Returns:
    -------
        A float representing the maximum of x and y.

    """
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Check if two numbers are close.

    Args:
    ----
        x: A float.
        y: A float.

    Returns:
    -------
        A boolean representing whether x and y are close.

    """
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Calculate the sigmoid of a number.

    Args:
    ----
        x: A float.

    Returns:
    -------
        A float representing the sigmoid of x.

    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def sigmoid_back(x: float) -> float:
    """Calculate the derivative of the sigmoid function.

    Args:
    ----
        x: A float.

    Returns:
    -------
        A float representing the derivative of the sigmoid function.

    """
    return x * (1 - x)


def log(x: float) -> float:
    """Calculate the natural logarithm of a number.

    Args:
    ----
        x: A float.

    Returns:
    -------
        A float representing the natural logarithm of x.

    """
    return math.log(x)


def exp(x: float) -> float:
    """Calculate the exponential of a number.

    Args:
    ----
        x: A float.

    Returns:
    -------
        A float representing the exponential of x.

    """
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    """Calculate the derivative of the logarithm function.

    Args:
    ----
        x (float): The input value for the logarithm.
        d (float): The derivative value.

    Returns:
    -------
        A float representing the derivative of the logarithm function.

    """
    return d * 1.0 / x


def inv(x: float) -> float:
    """Calculate the inverse of a number.

    Args:
    ----
        x: A float.

    Returns:
    -------
        A float representing the inverse of x.

    """
    return 1.0 / x


def inv_back(x: float, d: float) -> float:
    """Calculate the derivative of the inverse function.

    Args:
    ----
        x (float): The input value.
        d (float): The derivative value.

    Returns:
    -------
        float: A float representing the derivative of the inverse function.

    """
    return -1.0 * d / x**2.0


def relu(x: float) -> float:
    """Calculate the rectified linear unit of a number.

    Args:
    ----
        x: A float.

    Returns:
    -------
        A float representing the rectified linear unit of x.

    """
    return float(max(0.0, x))


def relu_back(x: float, d: float) -> float:
    """Calculate the derivative of the rectified linear unit function.

    Args:
    ----
        x: A float.
        d: A float.

    Returns:
    -------
        A float representing the derivative of the rectified linear unit function.

    """
    return d * (1.0 if x > 0.0 else 0.0)


def map(fn: Callable[[float], float], x: List[float]) -> List[float]:
    """Apply a function to a list.

    Args:
    ----
        fn: A function.
        x: A list of floats.

    Returns:
    -------
        A list of floats representing the application of fn to x.

    """
    return [fn(x[i]) for i in range(len(x))]


def zipWith(
    fn: Callable[[float, float], float], x: List[float], y: List[float]
) -> List[float]:
    """Apply a function to two lists.

    Args:
    ----
        fn: A function.
        x: A list of floats.
        y: A list of floats.

    Returns:
    -------
        A list of floats representing the application of fn to x and y.

    """
    return [fn(x[i], y[i]) for i in range(len(x))]


def reduce(
    fn: Callable[[float, float], float], x: List[float], initial: float
) -> float:
    """Reduce a list.

    Args:
    ----
        fn: A function.
        x: A list of floats.
        initial: A float.

    Returns:
    -------
        A float representing the reduction of x.

    """
    for i in range(len(x)):
        initial = fn(initial, x[i])
    return initial


def addLists(x: List[float], y: List[float]) -> List[float]:
    """Add two lists together.

    Args:
    ----
        x: A list of floats.
        y: A list of floats.

    Returns:
    -------
        A list of floats representing the sum of x and y.

    """
    return zipWith(add, x, y)


def negList(x: List[float]) -> List[float]:
    """Negate a list.

    Args:
    ----
        x: A list of floats.

    Returns:
    -------
        A list of floats representing the negation of x.

    """
    return map(neg, x)


def prod(x: List[float]) -> float:
    """Take the product of a list.

    Args:
    ----
        x: A list of floats.

    Returns:
    -------
        A float representing the product of x.

    """
    return reduce(mul, x, 1.0)


def sum(x: List[float]) -> float:
    """Sum a list.

    Args:
    ----
        x: A list of floats.

    Returns:
    -------
        A float representing the sum of x.

    """
    return reduce(add, x, 0.0)
