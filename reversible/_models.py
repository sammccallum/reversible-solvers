import equinox as eqx
import jax.numpy as jnp
import jax.random as jr


class SIR_model(eqx.Module):
    layers: list

    def __init__(self, key):
        key1, key2 = jr.split(key, 2)
        self.layers = [
            eqx.nn.Linear(3, 10, use_bias=True, key=key1),
            eqx.nn.Linear(10, 3, use_bias=True, key=key2),
        ]

    def __call__(self, t, y, args):
        for layer in self.layers[:-1]:
            y = layer(y)
            y = jnp.tanh(y)
        y = self.layers[-1](y)
        return y
