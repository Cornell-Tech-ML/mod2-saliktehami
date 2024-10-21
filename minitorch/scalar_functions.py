from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Applies the scalar function to the input values and returns the result as a new scalar variable.

        Args:
        ----
            vals (ScalarLike): The input values to the scalar function.

        Returns:
        -------
            Scalar: A new scalar variable representing the result of the scalar function applied to the input values.

        """
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Saves the input values for use in the backward pass and returns the result of the addition function.

        Args:
        ----
            ctx (Context): The context object to save values.
            a (float): The first input value to the function.
            b (float): The second input value to the function.

        Returns:
        -------
            float: The result of the addition function.

        """
        return operators.add(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Calculates the derivatives of the input values with respect to the output.

        Args:
        ----
            ctx (Context): The context object to save values.
            d_output (float): The derivative of the output with respect to the input.

        Returns:
        -------
            Tuple[float, ...]: The derivatives of the input values with respect to the output.

        """
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Saves the input value for use in the backward pass and returns the result of the log function.

        Args:
        ----
            ctx (Context): The context object to save values.
            a (float): The input value to the function.

        Returns:
        -------
            float: The result of the log function.

        """
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Args:
            ctx (Context): The context object to save values.
            d_output (float): The derivative of the output with respect to the input.

        Returns
        -------
            float: The derivative of the input with respect to the output.

        """
        (a,) = ctx.saved_values
        grad = operators.log_back(a, d_output)
        print(f"Log backward: d_output = {d_output}, a = {a}, grad = {grad}")
        return grad


class Mul(ScalarFunction):
    """Multiplication function $f(x,y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Saves the input values for use in the backward pass and returns the result of the multiplication function.

        Args:
        ----
            ctx (Context): The context object to save values.
            a (float): The first input value to the function.
            b (float): The second input value to the function.

        Returns:
        -------
            float: The result of the multiplication function.

        """
        ctx.save_for_backward(a, b)
        return operators.mul(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Calculates the derivatives of the input values with respect to the output.

        Args:
        ----
            ctx (Context): The context object to save values.
            d_output (float): The derivative of the output with respect to the input.

        Returns:
        -------
            Tuple[float, float]: The derivatives of the input values with respect to the output.

        """
        a, b = ctx.saved_values
        grad_a, grad_b = d_output * b, d_output * a
        print(
            f"Mul backward: d_output = {d_output}, a = {a}, b = {b}, grad_a = {grad_a}, grad_b = {grad_b}"
        )
        return grad_a, grad_b


class Inv(ScalarFunction):
    """Inverse function $f(x) = 1/x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Saves the input value for use in the backward pass and returns the result of the inverse function.

        Args:
        ----
            ctx (Context): The context object to save values.
            a (float): The input value to the function.

        Returns:
        -------
            float: The result of the inverse function.

        """
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Calculates the derivative of the input with respect to the output.

        Args:
        ----
            ctx (Context): The context object to save values.
            d_output (float): The derivative of the output with respect to the input.

        Returns:
        -------
            float: The derivative of the input with respect to the output.

        """
        (a,) = ctx.saved_values
        grad = operators.inv_back(a, d_output)
        print(f"Inv backward: d_output = {d_output}, a = {a}, grad = {grad}")
        return grad


class Neg(ScalarFunction):
    """Negation Function $f(x) = -1 * x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Saves the input value for use in the backward pass and returns the result of the negation function.

        Args:
        ----
            ctx (Context): The context object to save values.
            a (float): The input value to the function.

        Returns:
        -------
            float: The result of the negation function.

        """
        return operators.neg(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Calculates the derivative of the input with respect to the output.

        Args:
        ----
            ctx (Context): The context object to save values.
            d_output (float): The derivative of the output with respect to the input.

        Returns:
        -------
            float: Negation of the output derivative.

        """
        return operators.neg(d_output)


class Sigmoid(ScalarFunction):
    """Sigmoid function $f(x) = 1 / (1 + e^{-x})$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Saves the input value for use in the backward pass and returns the result of the sigmoid function.

        Args:
        ----
            ctx (Context): The context object to save values.
            a (float): The input value to the function.

        Returns:
        -------
            float: The result of the sigmoid function.

        """
        result = operators.sigmoid(a)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Calculates the derivative of the input with respect to the output.

        Args:
        ----
            ctx (Context): The context object to save values.
            d_output (float): The derivative of the output with respect to the input.

        Returns:
        -------
            float: The derivative of the input with respect to the output.

        """
        (a,) = ctx.saved_values
        grad = operators.sigmoid_back(a)
        return grad * d_output


class ReLU(ScalarFunction):
    """ReLu function f(x) = max(0,x)"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Saves the input value for use in the backward pass and returns the result of the ReLU function.

        Args:
        ----
            ctx (Context): The context object to save values.
            a (float): The input value to the function.

        Returns:
        -------
            float: The result of the ReLU function.

        """
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Calculates the derivative of the input with respect to the output.

        Args:
        ----
            ctx (Context): The context object to save values.
            d_output (float): The derivative of the output with respect to the input.

        Returns:
        -------
            float: Derivative of the Relu function of a with respect to the output.

        """
        (a,) = ctx.saved_values
        grad = operators.relu_back(a, d_output)
        return grad


class Exp(ScalarFunction):
    """Exponential function $f(x) = e^x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Saves the input value for use in the backward pass and returns the result of the exponential function.

        Args:
        ----
            ctx (Context): The context object to save values.
            a (float): The input value to the function.

        Returns:
        -------
            float: The result of the exponential function.

        """
        ctx.save_for_backward(a)
        return operators.exp(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Calculates the derivative of the input with respect to the output.

        Args:
        ----
            ctx (Context): The context object to save values.
            d_output (float): The derivative of the output with respect to the input.

        Returns:
        -------
            float: The derivative of the input with respect to the output.

        """
        (a,) = ctx.saved_values
        return d_output * operators.exp(a)


class LT(ScalarFunction):
    """Less than function $f(x,y) = x < y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Only returns the result of the less than function, no saving of values for backward pass.

        Args:
        ----
            ctx (Context): The context object to save values.
            a (float): The first input value to the function.
            b (float): The second input value to the function.

        Returns:
        -------
            float: The result of the less than function.

        """
        return operators.lt(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Computes the derivative of the less than function with respect to its inputs.

        Since the less than function is a comparator and does not have a derivative in the classical sense,
        this function returns 0.0 for both inputs. This is because the derivative of a comparator function
        is not defined in the context of calculus.

        Args:
        ----
            ctx (Context): The context object to save values.
            d_output (float): The gradient flowing back from the next operation in the computational graph.

        Returns:
        -------
            Tuple[float, float]: A tuple containing the derivatives of the less than function with respect to its inputs.
            Since the derivative is not defined, both values are 0.0.

        """
        return (0.0, 0.0)


class EQ(ScalarFunction):
    """Equal function $f(x,y) = x == y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Only returns the result of the equal function, no saving of values for backward pass.

        Args:
        ----
            ctx (Context): The context object to save values.
            a (float): The first input value to the function.
            b (float): The second input value to the function.

        Returns:
        -------
            float: The result of the equal function.

        """
        return operators.eq(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Computes the derivative of the equal function with respect to its inputs.

        Since the equal function is a comparator and does not have a derivative in the classical sense,
        this function returns 0.0 for both inputs. This is because the derivative of a comparator function
        is not defined in the context of calculus.

        Args:
        ----
            ctx (Context): The context object to save values.
            d_output (float): The gradient flowing back from the next operation in the computational graph.

        Returns:
        -------
            Tuple[float, float]: A tuple containing the derivatives of the equal function with respect to its inputs.
            Since the derivative is not defined, both values are 0.0.

        """
        return (0.0, 0.0)
