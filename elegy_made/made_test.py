import jax
import numpy as np
from elegy_made.linear import LinearMADE


def test_grads():

    x = np.random.uniform(size=(10, 3))

    def f(x):
        linear_m1 = LinearMADE(5, n_features=3)
        linear_m2 = LinearMADE(5, n_features=3)
        linear_m3 = LinearMADE(3, n_features=3, is_output=True)

        assignments = np.arange(3)
        x, assignments = linear_m1(x, assignments)
        x, assignments = linear_m2(x, assignments)
        x, assignments = linear_m3(x, assignments)

        return x

    for output_index in range(3):
        grads = jax.grad(lambda x: f(x)[0, output_index])(x)[0]

        for input_index in range(3):
            if input_index < output_index:
                assert grads[input_index] != 0
            else:
                assert grads[input_index] == 0
