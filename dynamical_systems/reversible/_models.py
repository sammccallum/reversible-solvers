import equinox as eqx
import jax.numpy as jnp
import jax.random as jr


class VectorField(eqx.Module):
    layers: list

    def __init__(self, y_dim, hidden_size, key):
        key1, key2 = jr.split(key, 2)
        self.layers = [
            eqx.nn.Linear(y_dim, hidden_size, use_bias=True, key=key1),
            eqx.nn.Linear(hidden_size, y_dim, use_bias=True, key=key2),
        ]

    def __call__(self, t, y, args):
        for layer in self.layers[:-1]:
            y = layer(y)
            y = jnp.tanh(y)
        y = self.layers[-1](y)
        return y


class TimeDependentVectorField(eqx.Module):
    layers: list

    def __init__(self, y_dim, hidden_size, key):
        key1, key2 = jr.split(key, 2)
        self.layers = [
            eqx.nn.Linear(y_dim + 1, hidden_size, use_bias=True, key=key1),
            eqx.nn.Linear(hidden_size, y_dim, use_bias=True, key=key2),
        ]

    def __call__(self, t, y, args):
        t = jnp.asarray(t)[None]
        y = jnp.concatenate([t, y])
        for layer in self.layers[:-1]:
            y = layer(y)
            y = jnp.tanh(y)
        y = self.layers[-1](y)
        return y
