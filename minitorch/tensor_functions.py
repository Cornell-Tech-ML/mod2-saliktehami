"""Implementation of the autodifferentiation Functions for Tensor."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Optional

import numpy as np

import minitorch
from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    from typing import Any, List, Tuple

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x: Any) -> tuple:  # type: ignore
    """Convert a value to a tuple if it is not already a tuple.

    Args:
    ----
        x (Any): Input value.

    Returns:
    -------
        tuple: The input wrapped as a tuple, if necessary.

    """
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        """Call the forward function and track history"""
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        # assert isinstance(c, Tensor), "Expected return type Tensor got %s" % (
        #     type(c)
        # )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the negation of a tensor.

        Args:
        ----
            ctx (Context): Context object (unused in this case).
            t1 (Tensor): Input tensor.

        Returns:
        -------
            Tensor: Negated tensor.

        """
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for negation.

        Args:
        ----
            ctx (Context): Context object.
            grad_output (Tensor): Gradient of the output.

        Returns:
        -------
            Tensor: Gradient of the input.

        """
        return -1.0 * grad_output


class Inv(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the element-wise inverse of a tensor.

        Args:
        ----
            ctx (Context): Context object.
            t1 (Tensor): Input tensor.

        Returns:
        -------
            Tensor: Element-wise inverse of the input.

        """
        ctx.save_for_backward(t1)
        result = t1.f.inv_map(t1)
        return result

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for the inverse operation.

        Args:
        ----
            ctx (Context): Context with saved input tensor.
            grad_output (Tensor): Gradient of the output.

        Returns:
        -------
            Tensor: Gradient of the input.

        """
        (t1,) = ctx.saved_values
        result = grad_output.f.inv_back_zip(t1, grad_output)
        return result


class Add(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Performs the forward pass of the addition operation.

        Args:
        ----
            ctx (Context): Context object (unused in this case).
            t1 (Tensor): First input tensor.
            t2 (Tensor): Second input tensor.

        Returns:
        -------
            Tensor: A new tensor containing the element-wise sum of t1 and t2.

        """
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for addition.

        Args:
        ----
            ctx (Context): Context object.
            grad_output (Tensor): Gradient of the output.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Gradients of the inputs.

        """
        return grad_output, grad_output


class All(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Optional[Tensor] = None) -> Tensor:
        """Perform an all-reduction operation along a given dimension.

        Args:
        ----
            ctx (Context): Context object.
            a (Tensor): Input tensor.
            dim (Optional[Tensor]): Dimension along which to reduce.

        Returns:
        -------
            Tensor: Result of the all-reduction.

        """
        # Handle optional dimension input
        dim_value = int(dim.item()) if dim is not None else -1

        # Save the dimension value as a tensor for the backward pass
        ctx.save_for_backward(tensor([dim_value]))

        # If dim is -1, perform reduction over all elements
        if dim_value == -1:
            return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)
        else:
            return a.f.mul_reduce(a, dim_value)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, None]:
        """Backward pass for the all-reduction operation.

        Args:
        ----
            ctx (Context): Context with saved dimension.
            grad_output (Tensor): Gradient of the output.

        Returns:
        -------
            Tuple[Tensor, None]: Gradients of the input and dimension.

        """
        return grad_output, None


class Mul(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Compute element-wise multiplication of two tensors.

        Args:
        ----
            ctx (Context): Context object.
            a (Tensor): First input tensor.
            b (Tensor): Second input tensor.

        Returns:
        -------
            Tensor: Element-wise product of the inputs.

        """
        ctx.save_for_backward(a, b)
        result = a.f.mul_zip(a, b)
        return result

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for element-wise multiplication.

        Args:
        ----
            ctx (Context): Context with saved input tensors.
            grad_output (Tensor): Gradient of the output.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Gradients of the inputs.

        """
        a, b = ctx.saved_tensors
        result = grad_output * b, grad_output * a
        return result


class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Compute the sigmoid of the input tensor.

        Args:
        ----
            ctx (Context): Context object.
            a (Tensor): Input tensor.

        Returns:
        -------
            Tensor: Sigmoid of the input tensor.

        """
        out = a.f.sigmoid_map(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for sigmoid.

        Args:
        ----
            ctx (Context): Context with saved output tensor.
            grad_output (Tensor): Gradient of the output.

        Returns:
        -------
            Tensor: Gradient of the input.

        """
        (out,) = ctx.saved_tensors
        one = out._ensure_tensor(1.0)
        return grad_output * out * (one - out)


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Compute the ReLU (Rectified Linear Unit) of the input tensor.

        Args:
        ----
            ctx (Context): Context object.
            a (Tensor): Input tensor.

        Returns:
        -------
            Tensor: ReLU of the input tensor.

        """
        ctx.save_for_backward(a)
        return a.f.relu_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for ReLU.

        Args:
        ----
            ctx (Context): Context with saved input tensor.
            grad_output (Tensor): Gradient of the output.

        Returns:
        -------
            Tensor: Gradient of the input.

        """
        (a,) = ctx.saved_tensors
        return grad_output * (a > 0)


class Log(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Compute the natural logarithm of the input tensor.

        Args:
        ----
            ctx (Context): Context object.
            a (Tensor): Input tensor.

        Returns:
        -------
            Tensor: Logarithm of the input tensor.

        """
        ctx.save_for_backward(a)
        return a.f.log_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for logarithm.

        Args:
        ----
            ctx (Context): Context with saved input tensor.
            grad_output (Tensor): Gradient of the output.

        Returns:
        -------
            Tensor: Gradient of the input.

        """
        (a,) = ctx.saved_tensors
        return grad_output / a


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Compute the exponential of the input tensor.

        Args:
        ----
            ctx (Context): Context object.
            a (Tensor): Input tensor.

        Returns:
        -------
            Tensor: Exponential of the input tensor.

        """
        out = a.f.exp_map(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for exponential.

        Args:
        ----
            ctx (Context): Context with saved output tensor.
            grad_output (Tensor): Gradient of the output.

        Returns:
        -------
            Tensor: Gradient of the input.

        """
        (out,) = ctx.saved_tensors
        return grad_output * out


class Sum(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, dim: Optional[Tensor] = None) -> Tensor:
        """Computes the forward pass for summation.

        Args:
        ----
            ctx (Context): Context object.
            t1 (Tensor): Input tensor.
            dim (Optional[Tensor]): Dimension along which to sum.

        Returns:
        -------
            Tensor: Result of the summation.

        """
        # Convert dim to integer or None before saving
        dim_val = int(dim.item()) if dim is not None else None
        ctx.save_for_backward(t1.shape, dim_val)
        if dim is not None:
            dim_val = int(dim.item())
            result = t1.f.add_reduce(t1, dim_val)
        else:
            flattened = t1.contiguous().view(t1.size)
            result = t1.f.add_reduce(flattened, 0)
        return result

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, ...]:
        """Backward pass for summation.

        Args:
        ----
            ctx (Context): Context with saved shape and dimension.
            grad_output (Tensor): Gradient of the output.

        Returns:
        -------
            Tuple[Tensor, ...]: Gradients of the input and dimension.

        """
        shape, dim = ctx.saved_values
        grad = minitorch.zeros(shape, backend=grad_output.backend)
        grad_input = grad_output.f.add_zip(grad, grad_output)
        # Return a tensor for grad_input and an integer/None for dim
        if dim is not None:
            return (grad_input, 0)  # Return 0 instead of None for the dim gradient
        else:
            return (grad_input,)


class LT(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Compute element-wise less than comparison of two tensors.

        Args:
        ----
            ctx (Context): Context object.
            a (Tensor): First input tensor.
            b (Tensor): Second input tensor.

        Returns:
        -------
            Tensor: Element-wise less than comparison of the inputs.

        """
        ctx.save_for_backward(a, b)
        return a.f.lt_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for less than comparison.

        Args:
        ----
            ctx (Context): Context with saved input tensors.
            grad_output (Tensor): Gradient of the output.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Gradients of the inputs.

        """
        a, b = ctx.saved_tensors
        return a.zeros(a.shape), b.zeros(b.shape)


class EQ(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Compute element-wise equality comparison of two tensors.

        Args:
        ----
            ctx (Context): Context object.
            a (Tensor): First input tensor.
            b (Tensor): Second input tensor.

        Returns:
        -------
            Tensor: Element-wise equality comparison of the inputs.

        """
        ctx.save_for_backward(a, b)
        return a.f.eq_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for equality comparison.

        Args:
        ----
            ctx (Context): Context with saved input tensors.
            grad_output (Tensor): Gradient of the output.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Gradients of the inputs.

        """
        a, b = ctx.saved_tensors
        return a.zeros(a.shape), b.zeros(b.shape)


class IsClose(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Compute element-wise close comparison of two tensors.

        Args:
        ----
            ctx (Context): Context object.
            a (Tensor): First input tensor.
            b (Tensor): Second input tensor.

        Returns:
        -------
            Tensor: Element-wise close comparison of the inputs.

        """
        return a.f.is_close_zip(a, b)


class Permute(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, order: Tensor) -> Tensor:
        """Perform the forward pass for the Permute function.

        Args:
        ----
            ctx (Context): Context to store saved tensors.
            a (Tensor): Input tensor to be permuted.
            order (Tensor): Tensor specifying the permutation order.

        Returns:
        -------
            Tensor: The permuted tensor.

        """
        ctx.save_for_backward(order)
        order_list = [int(x) for x in order._tensor._storage.tolist()]
        new_shape = tuple(a._tensor.shape[i] for i in order_list)
        new_strides = tuple(a._tensor.strides[i] for i in order_list)
        permuted_data = minitorch.Tensor.make(
            a._tensor._storage, new_shape, new_strides, backend=a.backend
        )
        return permuted_data

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, None]:
        """Perform the backward pass for the Permute function.

        Args:
        ----
            ctx (Context): Context containing saved tensors.
            grad_output (Tensor): Gradient of the output tensor.

        Returns:
        -------
            Tuple[Tensor, None]: The gradient with respect to the input tensor and None for the order.

        """
        (order,) = ctx.saved_tensors
        order_list = [int(x) for x in order._tensor._storage.tolist()]

        # Compute the inverse permutation
        inv_order = [0] * len(order_list)
        for i, p in enumerate(order_list):
            inv_order[p] = i

        # Apply the inverse permutation to the gradient
        grad_a = grad_output.permute(*inv_order)

        # Return the gradient for 'a' and a zero tensor for 'order'
        zero_order_grad = minitorch.zeros(order.shape, backend=grad_output.backend)
        return grad_a, zero_order_grad


class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        """Reshape the input tensor.

        Args:
        ----
            ctx (Context): Context object.
            a (Tensor): Input tensor.
            shape (Tensor): New shape.

        Returns:
        -------
            Tensor: Reshaped tensor.

        """
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for reshaping.

        Args:
        ----
            ctx (Context): Context with saved original shape.
            grad_output (Tensor): Gradient of the output.

        Returns:
        -------
            Tuple[Tensor, float]: Gradient of the input and a placeholder float.

        """
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Id function makes contiguous"""
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Undo"""
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Matrix Multiply Forward (module 3)"""
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Matrix Multiply backward (module 3)"""
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a zero tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend

    Returns:
    -------
        new tensor

    """
    return minitorch.Tensor.make(
        [0.0] * int(operators.prod(shape)), shape, backend=backend
    )


def ones(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a ones tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend

    Returns:
    -------
        new tensor

    """
    return minitorch.Tensor.make(
        [1.0] * int(operators.prod(shape)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a random tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a tensor with data ls and shape `shape`.

    Args:
    ----
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
    -------
        new tensor

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Produce a tensor with data and shape from ls

    Args:
    ----
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    """Estimate gradient using the central difference method.

    Args:
    ----
        f (Any): The function to differentiate.
        *vals (Tensor): Input tensors.
        arg (int): Index of the argument to differentiate.
        epsilon (float): Small perturbation for numerical differentiation.
        ind (UserIndex): Index of the tensor element to perturb.

    Returns:
    -------
        float: The estimated gradient.

    """
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    """Check whether autodiff matches central difference."""
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )
