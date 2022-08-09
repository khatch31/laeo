import jax
from jax.lib import xla_bridge

print("xla_bridge.get_backend().platform:", xla_bridge.get_backend().platform)
print("jax.devices():", jax.devices())
print("jax.default_backend():", jax.default_backend())
