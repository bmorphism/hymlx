# HyMLX

**JAX-Style Transforms for MLX on Apple Silicon + Hy S-expression DSL**

`hymlx` brings the functional elegance of JAX (`grad`, `jit`, `vmap`, `scan`) to Apple's [MLX](https://github.com/ml-explore/mlx) framework, wrapped in a [Hy](https://github.com/hylang/hy) (Lisp) DSL for expressive graph construction and deterministic color generation.

## Install

```bash
pip install hymlx
```

## Quick Start (uvx one-liners)

Run the deterministic color world or transform verification without installing:

```bash
# Run color world
uvx --from git+https://github.com/bmorphism/hymlx.git hymlx world

# Run JAX transform verification
uvx --from git+https://github.com/bmorphism/hymlx.git hymlx transforms

# Generate colors
uvx --from git+https://github.com/bmorphism/hymlx.git hymlx color 1069 5
```

## Core Features

### 1. JAX Transforms for MLX
Use `grad`, `jit`, `vmap`, and `scan` with native MLX arrays.

```python
from hymlx import grad, jit, vmap, scan

@jit
def loss(w, x):
    return (w * x).sum()

grad_fn = grad(loss)
batched_fn = vmap(loss)
```

### 2. Hy Graph DSL
Define graphs and GNNs using S-expressions.

```hy
(import hymlx.graphs [graph digraph topological-sort])

(setv g (digraph [0 1] [1 2] [2 0]))
(topological-sort g)
```

### 3. Deterministic Colors (Gay.jl Compatible)
Generate colors and GF(3) triads from seeds.

```python
from hymlx import splitmix64, seed_to_rgb
rgb = seed_to_rgb(splitmix64(1069))
```

## License
MIT