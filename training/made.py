import typing as tp

import elegy
import haiku as hk
import jax.numpy as jnp
import numpy as np
from elegy import module
from elegy.initializers import TruncatedNormal
from elegy.types import Initializer


class MaskInfo(tp.NamedTuple):
    mask: np.ndarray
    assignments: np.ndarray


class LinearMADE(module.Module):
    """Linear module."""

    w: np.ndarray
    b: np.ndarray
    mask_info: np.ndarray

    def __init__(
        self,
        output_size: int,
        n_features: int,
        is_output: bool = False,
        with_bias: bool = True,
        w_init: tp.Optional[Initializer] = None,
        b_init: tp.Optional[Initializer] = None,
        **kwargs
    ):
        """
        Constructs the Linear module.

        Arguments:
            output_size: Output dimensionality.
            with_bias: Whether to add a bias to the output.
            w_init: Optional initializer for weights. By default, uses random values
                from truncated normal, with stddev `1 / sqrt(fan_in)`. See
                https://arxiv.org/abs/1502.03167v3.
            b_init: Optional initializer for bias. By default, zero.
            kwargs: Additional keyword arguments passed to Module.
        """
        super().__init__(**kwargs)
        self.input_size = None
        self.output_size = output_size
        self.n_features = n_features
        self.is_output = is_output
        self.with_bias = with_bias
        self.w_init = w_init
        self.b_init = b_init or jnp.zeros

    @staticmethod
    def create_mask(
        output_size: int,
        n_features: int,
        input_assignments: np.ndarray,
        is_output: bool = False,
    ):
        """
        Creates a mask and unit assignments (for now it uses a very simple deterministic scheme).

        Arguments:
            output_size: Output dimensionality.
            n_features: Number of random variable in the input (input dimension).
            input_assignments: Unit assigments from the previous layer.
            is_output: Whether this is an output layer.

        Returns:
            Tuple of (y, assignments).
        """

        if not is_output:
            n_features -= 1

        input_size = input_assignments.shape[0]

        output_assignments = list(range(n_features)) * int(output_size / n_features + 1)
        output_assignments = np.array(output_assignments[:output_size])

        kernel_shape = (input_size, output_size)

        input_constraints = np.broadcast_to(input_assignments[:, None], kernel_shape)
        output_constraints = np.broadcast_to(output_assignments[None, :], kernel_shape)

        if is_output:
            kernel_mask = (output_constraints > input_constraints).astype(np.int32)
        else:
            kernel_mask = (output_constraints >= input_constraints).astype(np.int32)

        return MaskInfo(kernel_mask, output_assignments)

    def call(self, inputs: np.ndarray, input_assignments: np.ndarray) -> np.ndarray:
        """"""
        if not inputs.shape:
            raise ValueError("Input must not be scalar.")

        input_size = self.input_size = inputs.shape[-1]
        output_size = self.output_size
        dtype = inputs.dtype

        w_init = self.w_init

        if w_init is None:
            stddev = 1.0 / np.sqrt(self.input_size)
            w_init = TruncatedNormal(stddev=stddev)

        w = self.add_parameter(
            "w", [input_size, output_size], dtype, initializer=w_init
        )

        w_mask, output_assignments = self.add_parameter(
            "mask_info",
            initializer=lambda *args: self.create_mask(
                output_size=output_size,
                n_features=self.n_features,
                input_assignments=input_assignments,
                is_output=self.is_output,
            ),
            trainable=False,
        )

        w *= w_mask

        out = jnp.dot(inputs, w)

        if self.with_bias:
            b = self.add_parameter(
                "b", [self.output_size], dtype, initializer=self.b_init
            )
            b = jnp.broadcast_to(b, out.shape)
            out = out + b

        return out, output_assignments
