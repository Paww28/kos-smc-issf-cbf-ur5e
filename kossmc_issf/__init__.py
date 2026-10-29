"""KOS-SMC + ISSf-CBF for the UR5e."""

# oscbf must be imported before any jnp array is created: its __init__ enables jax_enable_x64.
import oscbf  # noqa: F401
