from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
       f : arbitrary function from n-scalar args to one value
       *vals : The point at which to evaluate f
       arg : the index of the argument to compute the derivative
       epsilon : a small constant

    Returns:
    -------
        An approximation of f'_i(x_0, \ldots, x_{n-1})$

    """
    vals1 = [v for v in vals]
    vals2 = [v for v in vals]
    vals1[arg] = vals1[arg] + epsilon
    vals2[arg] = vals2[arg] - epsilon
    return (f(*vals1) - f(*vals2)) / (2.0 * epsilon)
    # END ASSIGN 1.1


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Add gradient to the gradient accumulator.

        Args:
        ----
            x: Value to be accumulated

        """
        ...

    @property
    def unique_id(self) -> int:
        """Unique identifier for the variable."""
        ...

    def is_leaf(self) -> bool:
        """Check if the variable is a leaf node in the computation graph."""
        ...

    def is_constant(self) -> bool:
        """Check if the variable is a constant."""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Get the parent variables of the current variable."""
        ...

    def chain_rule(self, d: Any) -> Iterable[Tuple["Variable", Any]]:
        """Apply the chain rule to the variable and return the derivatives of the parents."""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    order: List[Variable] = []
    seen = set()

    def visit(var: Variable) -> None:
        if var.unique_id in seen or var.is_constant():
            return
        if not var.is_leaf():
            for m in var.parents:
                if not m.is_constant():
                    visit(m)
        seen.add(var.unique_id)
        order.insert(0, var)

    visit(variable)
    return order


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Run backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    """
    queue = topological_sort(variable)
    derivatives = {}
    derivatives[variable.unique_id] = deriv
    for var in queue:
        deriv = derivatives[var.unique_id]
        if var.is_leaf():
            var.accumulate_derivative(deriv)
        else:
            for v, d in var.chain_rule(deriv):
                if v.is_constant():
                    continue
                derivatives.setdefault(v.unique_id, 0.0)
                derivatives[v.unique_id] = derivatives[v.unique_id] + d


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Get previously saved values during forward pass."""
        return self.saved_values
