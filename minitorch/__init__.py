"""minitorch: A deep learning library.

This package provides a set of tools and modules for building and training neural networks from scratch. It includes:

- tensor: Core tensor operations and data structures.
- tensor_ops: Basic tensor operations.
- tensor_functions: Autograd-enabled tensor functions.
- datasets: Utilities for loading and handling datasets.
- optim: Optimization algorithms for training models.
- testing: Utilities for testing and validating models.
- module: Base class for all neural network modules.
- autodiff: Automatic differentiation for tensors.
- scalar: Scalar operations and data structures.
- scalar_functions: Functions for scalar operations.

"""

from .testing import MathTest, MathTestVariable  # type: ignore # noqa: F401,F403
from .tensor_data import *  # noqa: F401,F403
from .tensor import *  # noqa: F401,F403
from .tensor_ops import *  # noqa: F401,F403
from .tensor_functions import *  # noqa: F401,F403
from .datasets import *  # noqa: F401,F403
from .optim import *  # noqa: F401,F403
from .testing import *  # noqa: F401,F403
from .module import *  # noqa: F401,F403
from .autodiff import *  # noqa: F401,F403
from .scalar import *  # noqa: F401,F403
from .scalar_functions import *  # noqa: F401,F403
from .module import *  # noqa: F401,F403
