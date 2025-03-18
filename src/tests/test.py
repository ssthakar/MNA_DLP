import jax.numpy as jnp

a = jnp.zeros((3, 1), float)
a = a.reshape((-1, 1))
b = jnp.zeros((3, 1), float)
b = b.reshape((-1, 1))
c = a + b
print(c.shape)
