from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple, Type, Union

import numpy as np

from dataclasses import field
from .autodiff import Context, Variable, backpropagate, central_difference
from .scalar_functions import (
    EQ,
    LT,
    Add,
    Exp,
    Inv,
    Log,
    Mul,
    Neg,
    ReLU,
    ScalarFunction,
    Sigmoid,
)

ScalarLike = Union[float, int, "Scalar"]


@dataclass
class ScalarHistory:
    """`ScalarHistory` stores the history of `Function` operations that was
    used to construct the current Variable.

    Attributes
    ----------
        last_fn : The last Function that was called.
        ctx : The context for that Function.
        inputs : The inputs that were given when `last_fn.forward` was called.

    """

    last_fn: Optional[Type[ScalarFunction]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Scalar] = ()


# ## Task 1.2 and 1.4
# Scalar Forward and Backward

_var_count = 0


@dataclass
class Scalar:
    """A reimplementation of scalar values for autodifferentiation
    tracking. Scalar Variables behave as close as possible to standard
    Python numbers while also tracking the operations that led to the
    number's creation. They can only be manipulated by
    `ScalarFunction`.
    """

    data: float
    history: Optional[ScalarHistory] = field(default_factory=ScalarHistory)
    derivative: Optional[float] = None
    name: str = field(default="")
    unique_id: int = field(default=0)

    def __post_init__(self):
        global _var_count
        _var_count += 1
        object.__setattr__(self, "unique_id", _var_count)
        object.__setattr__(self, "name", str(self.unique_id))
        object.__setattr__(self, "data", float(self.data))
        print(
            f"Scalar created: {self}, is_leaf: {self.is_leaf()}, data: {self.data}, history: {self.history}"
        )

    def __repr__(self) -> str:
        return f"Scalar({self.data})"

    def __mul__(self, b: ScalarLike) -> Scalar:
        if isinstance(b, (int, float)):
            b = Scalar(b)
        return Mul.apply(self, b)

    def __truediv__(self, b: ScalarLike) -> Scalar:
        if isinstance(b, (int, float)):
            b = Scalar(b)
        return Mul.apply(self, Inv.apply(b))

    def __rtruediv__(self, b: ScalarLike) -> Scalar:
        if isinstance(b, (int, float)):
            b = Scalar(b)
        return Mul.apply(b, Inv.apply(self))

    def __bool__(self) -> bool:
        return bool(self.data)

    def __radd__(self, b: ScalarLike) -> Scalar:
        if isinstance(b, (int, float)):
            b = Scalar(b)
        return self + b

    def __rmul__(self, b: ScalarLike) -> Scalar:
        if isinstance(b, (int, float)):
            b = Scalar(b)
        return self * b

    def __lt__(self, b: ScalarLike) -> Scalar:
        if isinstance(b, (int, float)):
            b = Scalar(b)
        return LT.apply(self, b)

    def __gt__(self, b: ScalarLike) -> Scalar:
        if isinstance(b, (int, float)):
            b = Scalar(b)
        return LT.apply(b, self)

    def __sub__(self, b: ScalarLike) -> Scalar:
        if isinstance(b, (int, float)):
            b = Scalar(b)
        return Add.apply(self, Neg.apply(b))

    def __rsub__(self, b: ScalarLike) -> Scalar:
        if isinstance(b, (int, float)):
            b = Scalar(b)
        return Add.apply(b, Neg.apply(self))

    def __neg__(self) -> Scalar:
        return Neg.apply(self)

    def __add__(self, b: ScalarLike) -> Scalar:
        if isinstance(b, (int, float)):
            return Add.apply(self, Scalar(b))
        return Add.apply(self, b)

    def log(self) -> Scalar:
        """Applies the natural logarithm function to the scalar value.

        Returns
        -------
            A new Scalar object representing the natural logarithm of the original scalar value.

        """
        return Log.apply(self)

    def exp(self) -> Scalar:
        """Applies the exponential function to the scalar value.

        Returns
        -------
        Scalar
            A new Scalar object representing e raised to the power of the original scalar value.

        """
        return Exp.apply(self)

    def sigmoid(self) -> Scalar:
        """Applies the sigmoid function to the scalar value.

        The sigmoid function is defined as: f(x) = 1 / (1 + e^(-x))

        Returns
        -------
        Scalar
            A new Scalar object representing the result of the sigmoid function applied to the original scalar value.

        """
        return Sigmoid.apply(self)

    def relu(self) -> Scalar:
        """Applies the Rectified Linear Unit (ReLU) function to the scalar value.

        The ReLU function is defined as: f(x) = max(0, x)

        Returns
        -------
        Scalar
            A new Scalar object representing the result of the ReLU function applied to the original scalar value.

        """
        return ReLU.apply(self)

    def __eq__(self, b: ScalarLike) -> Scalar:
        """Checks equality between this scalar and another scalar-like value.

        Args:
        ----
        b : ScalarLike
            The value to compare with this scalar. Can be a Scalar, float, or int.

        Returns:
        -------
        Scalar
            A new Scalar object representing the result of the equality comparison (1.0 if equal, 0.0 if not equal).

        """
        if isinstance(b, (int, float)):
            b = Scalar(b)
        return EQ.apply(self, b)

    def __hash__(self) -> float:
        return hash(self.unique_id)

    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x: value to be accumulated

        """
        print(f"Accumulating derivative for Scalar with ID {id(self)}")
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.derivative is None:
            self.__setattr__("derivative", 0.0)
        self.__setattr__("derivative", self.derivative + x)

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """True if this scalar is a constant (i.e. its history is None)."""
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Returns:
        -------
        Iterable[Variable]: An iterable of the parent variables of this scalar.

        """
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d: Any) -> Iterable[Tuple[Variable, Any]]:
        """Applies the chain rule to compute gradients for the inputs of this Scalar's last operation.
        This method is a key part of the backpropagation process. It computes local gradients
        for each input of the last operation performed on this Scalar, pairs these gradients
        with their corresponding input variables, and filters out any constants.

        Args:
        ----
        d: The gradient flowing back from the next operation in the
                        computational graph. This is typically a float, but may be
                        a more complex type for operations with multiple outputs.

        Returns:
        -------
        Iterable[Tuple[Variable, Any]]: A list of tuples, each containing:
            - An input Variable to the last operation.
            - The gradient of the output with respect to that input.
          Constants (Variables that don't require gradients) are excluded from this list.

        Note:
        ----
        This method is typically called internally during the backward pass and is a crucial
        component in implementing reverse-mode automatic differentiation.

        """
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        local_grads = h.last_fn._backward(h.ctx, d)
        if not isinstance(local_grads, Iterable):
            local_grads = [local_grads]

        paired_grads = zip(h.inputs, local_grads)
        result = [(x, grad) for x, grad in paired_grads if not x.is_constant()]

        return result

    def backward(self, d_output: Optional[float] = None) -> None:
        """Calls autodiff to fill in the derivatives for the history of this object.

        Args:
        ----
            d_output (number, opt): starting derivative to backpropagate through the model
                                   (typically left out, and assumed to be 1.0).

        """
        if d_output is None:
            d_output = 1.0
        backpropagate(self, d_output)


def derivative_check(f: Any, *scalars: Scalar) -> None:
    """Checks that autodiff works on a python function.

    Args:
    ----
    f (Any): The function to check.
    *scalars (Scalar): Variable number of scalar inputs for the function.

    Asserts False if derivative is incorrect.

    """
    out = f(*scalars)
    out.backward()

    err_msg = """
Derivative check at arguments f(%s) and received derivative f'=%f for argument %d,
but was expecting derivative f'=%f from central difference."""
    for i, x in enumerate(scalars):
        check = central_difference(f, *scalars, arg=i)
        print(str([x.data for x in scalars]), x.derivative, i, check)
        assert x.derivative is not None
        np.testing.assert_allclose(
            x.derivative,
            check.data,
            1e-2,
            1e-2,
            err_msg=err_msg
            % (str([x.data for x in scalars]), x.derivative, i, check.data),
        )
