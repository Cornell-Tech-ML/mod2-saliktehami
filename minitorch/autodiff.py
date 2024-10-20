from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    args_plus = list(vals)
    args_minus = list(vals)
    args_plus[arg] += epsilon
    args_minus[arg] -= epsilon
    f_vals_plus, f_vals_minus = f(*args_plus), f(*args_minus)
    return (f_vals_plus - f_vals_minus) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None: ...  # noqa: D102

    @property
    def unique_id(self) -> int: ...  # noqa: D102

    def is_leaf(self) -> bool: ...  # noqa: D102

    def is_constant(self) -> bool: ...  # noqa: D102

    @property
    def parents(self) -> Iterable["Variable"]: ...  # noqa: D102

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]: ...  # noqa: D102


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """

    def dfs_topological_sort(variable: Variable) -> Iterable[Variable]:
        visited = set()
        result = []

        def dfs(v: Variable) -> None:
            if v in visited or v.is_constant():
                return
            visited.add(v)
            for parent in v.parents:
                dfs(parent)
            result.append(v)

        dfs(variable)
        return reversed(result)

    return dfs_topological_sort(variable)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
        variable: The variable to start backpropagation from.
        deriv: The derivative of the output with respect to the variable.

    """
    ordered_variables = [*topological_sort(variable)]
    print(f"TOPOLOGICALLY SORTED VARIABLES: {ordered_variables}")

    gradients = {variable: deriv}

    for var in ordered_variables:
        grad = gradients.get(var, 0)
        print(f"Processing variable: {var}, current gradient: {grad}")
        if var.is_leaf():
            var.accumulate_derivative(grad)
        else:
            # Ensure that we handle cases where the chain rule might return None
            for parent, parent_grad in var.chain_rule(grad):
                if parent_grad is None:
                    parent_grad = 0  # Default to 0 if no gradient is computed
                if parent not in gradients:
                    gradients[parent] = parent_grad
                else:
                    gradients[parent] += parent_grad
    print(f"FINAL GRADIENTS: {gradients}")


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
    def saved_tensors(self) -> Tuple[Any, ...]:  # noqa: D102
        return self.saved_values
