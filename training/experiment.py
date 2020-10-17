import dataget
import elegy
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import scipy
import typer
from elegy_made.linear import LinearMADE
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from scipy.ndimage.filters import gaussian_filter


def main(
    debug: bool = False,
    lr: float = 0.002,
    batch_size: int = 32,
    epochs: int = 100,
    n_units: int = 64,
    n_components: int = 10,
    a1: float = 1.0,
    a2: float = 1.0,
    n_layers: int = 3,
    l2: float = 0.0005,
):

    if debug:
        import debugpy

        print("Waiting debugger...")
        debugpy.listen(5678)
        debugpy.wait_for_client()

    df_train, df_test = dataget.toy.spirals().get()

    X_train = df_train[["x0", "x1"]].to_numpy()
    X_train = MinMaxScaler().fit_transform(X_train)
    X_train = np.concatenate(
        [
            X_train,
            X_train + np.random.normal(scale=0.02, size=X_train.shape),
            X_train + np.random.normal(scale=0.02, size=X_train.shape),
            X_train + np.random.normal(scale=0.02, size=X_train.shape),
        ],
        axis=0,
    )

    module = MADE(
        n_units=n_units,
        n_features=X_train.shape[1],
        n_components=n_components,
        n_layers=n_layers,
    )

    model = Model(
        module=module,
        loss=[
            MixtureNLL4(a1, a2),
            elegy.regularizers.GlobalL2(l2),
        ],
        optimizer=optax.adam(lr),
        run_eagerly=False,
    )

    for i in range(epochs):
        model.fit(
            X_train,
            batch_size=batch_size,
            epochs=1000,
        )
        viz_component(X_train, model)


class MADE(elegy.Module):
    def __init__(
        self,
        n_units: int,
        n_features: int,
        n_components: int = 5,
        n_layers: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_units = n_units
        self.n_features = n_features
        self.n_components = n_components
        self.n_layers = n_layers

    def call(self, x):
        assert self.n_features == x.shape[-1]

        assignments = np.arange(self.n_features)

        for i in range(self.n_layers):
            x, assignments = LinearMADE(self.n_units, n_features=self.n_features)(
                x, assignments
            )
            x = jax.nn.relu(x)

        y: np.ndarray = jnp.stack(
            [
                jnp.stack(
                    [
                        LinearMADE(
                            self.n_features,
                            n_features=self.n_features,
                            is_output=True,
                        )(x, assignments)[0]
                        for _ in range(3)  # (mean, std, prob)
                    ],
                    axis=2,
                )
                for _ in range(self.n_components)
            ],
            axis=1,
        )

        y = jax.ops.index_update(y, jax.ops.index[..., 1], 1.0 + jax.nn.elu(y[..., 1]))
        y = jax.ops.index_update(
            y, jax.ops.index[..., 2], jax.nn.softmax(y[..., 2], axis=1)
        )

        return y


class Model(elegy.Model):
    def densities(self, x):
        y = self.predict(x)

        x = np.broadcast_to(x[:, None, :], y.shape[:-1])

        probs = y[..., 2]
        densities = scipy.stats.norm.pdf(x, loc=y[..., 0], scale=y[..., 1])

        return densities, probs


class MixtureNLL(elegy.Loss):
    def call(self, x, y_pred):
        mean = y_pred[..., 0]
        std = y_pred[..., 1]
        prob = y_pred[..., 2]

        x = x[:, None]
        x = jnp.broadcast_to(x, mean.shape)

        # step = self.add_parameter("step", initializer=jnp.array(0), trainable=False)
        # self.update_parameter("step", step + 1)

        # def train_params(inputs):
        #     mean, std, prob = inputs
        #     prob = jax.lax.stop_gradient(prob)
        #     return mean, std, prob

        # def train_probs(inputs):
        #     mean, std, prob = inputs
        #     mean = jax.lax.stop_gradient(mean)
        #     std = jax.lax.stop_gradient(std)
        #     return mean, std, prob

        # mean, std, prob = jax.lax.cond(
        #     step % 2 != 0,
        #     train_params,
        #     train_probs,
        #     (mean, std, prob),
        # )

        out = jnp.sum(
            -safe_log(
                jnp.sum(
                    prob * jax.scipy.stats.norm.pdf(x, loc=mean, scale=std),
                    axis=1,
                ),
            ),
            axis=1,
        )

        return out


class MixtureNLL2(elegy.Loss):
    def call(self, x, y_pred):
        mean = y_pred[..., 0]
        std = y_pred[..., 1]
        prob = y_pred[..., 2]

        x = x[:, None]

        x = jnp.broadcast_to(x, mean.shape)

        component_loss = -jax.scipy.stats.norm.logpdf(x, loc=mean, scale=std)
        min_loss_index = jnp.argmin(component_loss, axis=1)
        min_component_loss = jnp.min(component_loss, axis=1)
        prob_loss = -safe_log(
            jnp.take_along_axis(prob, min_loss_index[:, None], axis=1)
        )

        return min_component_loss + prob_loss


class MixtureNLL3(elegy.Loss):
    def call(self, x, y_pred):
        step = self.add_parameter("step", initializer=jnp.array(0), trainable=False)
        self.update_parameter("step", step + 1)

        return jax.lax.cond(
            step % 3 != 0,
            lambda t: MixtureNLL()(*t),
            lambda t: MixtureNLL2()(*t),
            (x, y_pred),
        )


class MixtureNLL4(elegy.Loss):
    def __init__(self, a1, a2, **kwargs):
        super().__init__(**kwargs)
        self.a1 = a1
        self.a2 = a2

    def call(self, x, y_pred):

        l1 = MixtureNLL()(x, y_pred)
        l2 = MixtureNLL2()(x, y_pred)

        return self.a1 * l1 + self.a2 * l2


def safe_log(x):
    return jnp.log(jnp.maximum(x, 1e-6))


def viz_component(X, model):

    x0_min = X[:, 0].min() - 0.2
    x0_max = X[:, 0].max() + 0.2

    x1_min = X[:, 1].min() - 0.2
    x1_max = X[:, 1].max() + 0.2

    ########################################
    # x0
    ########################################
    x0 = np.linspace(x0_min, x0_max, 100)
    x1 = np.zeros_like(x0, dtype=np.float32) + 0.5

    x = np.stack([x0, x1], axis=1)

    densities, probs = model.densities(x)
    feature_density = np.sum(densities * probs, axis=1)

    plt.figure(figsize=(16, 10))
    plt.subplot(2, 4, 1)
    plt.title("P_k(x0)")
    for module in range(densities.shape[1]):
        plt.plot(x0, densities[:, module, 0])

    plt.plot(x0, feature_density[:, 0], color="black", linewidth=3)

    plt.subplot(2, 4, 2)
    plt.title("P(x0)")
    plt.plot(x0, feature_density[:, 0], color="black", linewidth=3)

    plt.subplot(2, 4, 3)
    plt.title("hist(x0)")
    sns.kdeplot(x=X[:, 0], bw_adjust=0.2)
    sns.histplot(x=X[:, 0], stat="density", bins=40)

    ########################################
    # x1
    ########################################
    x1 = np.linspace(x1_min, x1_max, 100)
    x0 = np.zeros_like(x1, dtype=np.float32) + 0.5

    x = np.stack([x0, x1], axis=1)

    densities, probs = model.densities(x)
    feature_density = np.sum(densities * probs, axis=1)

    plt.subplot(2, 4, 5)
    plt.title("P_k(x1 | x0 = 0.5)")
    for module in range(densities.shape[1]):
        plt.plot(x1, densities[:, module, 1])

    plt.plot(x1, feature_density[:, 1], color="black", linewidth=3)

    plt.subplot(2, 4, 6)
    plt.title("P(x1 | x0 = 0.5)")
    plt.title(", ".join(f"{p:.3f}" for p in probs[0, :, 1]))
    plt.plot(x1, feature_density[:, 1], color="black", linewidth=3)

    plt.subplot(2, 4, 7)
    plt.title("hist(x1 | 0.475 < x0 < 0.525)")
    sns.kdeplot(x=X[(0.475 < X[:, 0]) & (X[:, 0] < 0.525), :][:, 1], bw_adjust=0.2)
    sns.histplot(
        x=X[(0.475 < X[:, 0]) & (X[:, 0] < 0.525), :][:, 1], stat="density", bins=40
    )

    ########################################
    # full
    ########################################
    x0 = np.linspace(x0_min, x0_max, 50)
    x1 = np.linspace(x1_min, x1_max, 50)

    x0v, x1v = np.meshgrid(x0, x1)

    x = np.stack([x0v.flatten(), x1v.flatten()], axis=1)
    densities, probs = model.densities(x)

    density = np.prod(np.sum(densities * probs, axis=1), axis=1)
    density = density.reshape(x0v.shape)

    plt.subplot(2, 4, 4)
    plt.title(f"P(x0, x1)")
    plt.pcolormesh(
        x0v,
        x1v,
        density,
        shading="gouraud",
    )

    plt.subplot(2, 4, 8)
    plt.title(f"P(x0, x1) + scatter(x0, x1)")
    sns.scatterplot(x=X[:, 0], y=X[:, 1], color="black")
    plt.pcolor(
        x0v,
        x1v,
        density,
        shading="auto",
        alpha=0.5,
    )
    plt.grid(False)

    plt.show()


if __name__ == "__main__":
    typer.run(main)
