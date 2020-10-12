import elegy
import jax
import numpy as np
import typer

from elegy_made.linear import LinearMADE


def main(debug: bool = False):

    if debug:
        import debugpy

        print("Waiting debugger...")
        debugpy.listen(5678)
        debugpy.wait_for_client()

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

    grads = jax.grad(lambda x: f(x)[0, 2])(x)[0]

    print(grads)


class MADE(elegy.Module):
    def __init__(self, n_features: int, **kwargs):
        super().__init__(**kwargs)
        self.n_features = n_features

    def call(self, x):
        assert self.n_features == x.shape[-1]

        assignments = np.arange(self.n_features)

        x, assignments = LinearMADE(32, n_features=self.n_features)(x, assignments)
        x = jax.nn.elu(x)

        x, assignments = LinearMADE(32, n_features=self.n_features)(x, assignments)
        x = jax.nn.elu(x)


if __name__ == "__main__":
    typer.run(main)
