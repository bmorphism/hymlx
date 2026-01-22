"""HyMLX: JAX-Style Transformations for MLX on Apple Silicon.

GF(3) Triadic Structure:
  PLUS (+1):    graphs.hy (generation), transforms.py
  MINUS (-1):   nn.py (validation/gradients)  
  ERGODIC (0):  jax_bridge.hy, telemetry.hy (coordination/observation)
  
Telemetry: Toad-traceable events for TUI/frontier model integration.
"""

# Import hy first to enable .hy file imports
import hy  # noqa: F401

from hymlx.transforms import grad, value_and_grad, jit, vmap, scan, fori_loop, force
from hymlx.splitmix import (
    splitmix64,
    seed_to_trit,
    seed_to_hue,
    seed_to_rgb,
    derive,
    derive_chain,
    check_gf3,
    spawn_triad,
    reafference,
)
from hymlx.nn import sequential, mlp_seeded, train_step

# Lazy imports for Hy modules (avoid import at module load time)
_lazy_modules = {}

def __getattr__(name: str):
    """Lazy loading for Hy modules."""
    import importlib
    
    hy_modules = {"arch", "graphs", "jax_bridge", "telemetry"}
    
    if name in hy_modules:
        if name not in _lazy_modules:
            _lazy_modules[name] = importlib.import_module(f"hymlx.{name}")
        return _lazy_modules[name]
    
    raise AttributeError(f"module 'hymlx' has no attribute {name!r}")

__version__ = "0.2.0"
__all__ = [
    # Transforms
    "grad",
    "value_and_grad", 
    "jit",
    "vmap",
    "scan",
    "fori_loop",
    "force",
    # SplitMix64
    "splitmix64",
    "seed_to_trit",
    "seed_to_hue",
    "seed_to_rgb",
    "derive",
    "derive_chain",
    "check_gf3",
    "spawn_triad",
    "reafference",
    # NN
    "sequential",
    "mlp_seeded",
    "train_step",
    # Hy modules (lazy)
    "arch",
    "graphs",
    "jax_bridge",
    "telemetry",
]
