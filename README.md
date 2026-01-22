# HyMLX

> JAX-Style Transformations for MLX on Apple Silicon with Hy S-expression DSL

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![MLX](https://img.shields.io/badge/MLX-0.20+-orange.svg)](https://github.com/ml-explore/mlx)
[![Hy](https://img.shields.io/badge/Hy-1.0+-purple.svg)](https://github.com/hylang/hy)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/hymlx.svg)](https://pypi.org/project/hymlx/)

## Features

- **JAX-style transforms** (`grad`, `jit`, `vmap`, `scan`) for MLX
- **Hy S-expression DSL** for graph construction and GNN architectures
- **SplitMix64 RNG** compatible with Gay.jl for deterministic colors
- **GF(3) triadic structure** for balanced computation tracking
- **Optional JAX bridge** for JraphX, RLax, and MCTX integration

## Install

```bash
# Core (MLX + Hy)
pip install hymlx

# With graph support
pip install hymlx[graphs]

# With JAX ecosystem
pip install hymlx[jax]

# With everything
pip install hymlx[all]

# Development install
git clone https://github.com/Soft-Machine-io/soft-machine
cd soft-machine/hymlx
pip install -e ".[dev]"
```

## Quick Start

### Transforms

```python
from hymlx import grad, jit, vmap, scan

# Automatic differentiation
def loss(x):
    return (x ** 2).sum()

grad_loss = grad(loss)

# JIT compilation
@jit
def fast_fn(x):
    return x @ x.T

# Vectorized map (batching)
batched_op = vmap(lambda x: x.sum())

# Functional loop
def step(carry, x):
    return carry + x, carry

final, history = scan(step, 0.0, xs)
```

### Graph DSL (Hy)

```hy
;; In .hy files or via hy REPL
(import hymlx.graphs [graph digraph complete-graph])

;; Clojure Loom-style construction
(setv g (graph [0 1] [1 2] [2 0]))

;; Adjacency map
(setv g2 (graph {0 [1 2] 1 [2]}))

;; Algorithms
(import hymlx.graphs [topological-sort dfs bfs])
(topological-sort (digraph [0 1] [1 2]))  ; [0, 1, 2]

;; Convert to mlx_graphs
(import hymlx.graphs [->mlx-graph])
(setv mlx-data (->mlx-graph g))
```

### Deterministic Colors (Gay.jl Compatible)

```python
from hymlx import splitmix64, seed_to_rgb, spawn_triad, check_gf3

# Generate reproducible color
rgb = seed_to_rgb(splitmix64(1069))  # [r, g, b] in [0, 1]

# GF(3)-balanced triad
triad = spawn_triad(1069)
# Returns: {"minus": seed1, "zero": seed2, "plus": seed3}

# Verify conservation: sum of trits ≡ 0 (mod 3)
assert check_gf3([triad["minus"], triad["zero"], triad["plus"]])
```

### JAX Bridge (Optional)

```hy
;; Requires: pip install hymlx[jraphx]
(import hymlx.jax_bridge [->jraphx <-jraphx HyJaxGraph hyjax-graph])

;; Unified graph with both backends
(setv g (hyjax-graph [0 1] [1 2] [2 0]))

(.mlx g)  ; -> mlx_graphs.GraphData
(.jax g)  ; -> jraphx.data.Data

;; GF(3) triadic stack
(import hymlx.jax_bridge [TriadicStack])
(setv stack (TriadicStack))
(.add-plus stack "forward-pass")
(.add-minus stack "backward-pass")
(.balanced? stack)  ; True (1 + -1 = 0)
```

## CLI

```bash
# Run demo
hymlx demo

# Generate colors
hymlx color 1069 5

# Transform demos
hymlx transforms
```

## Architecture

```
hymlx/
├── transforms.py    # JAX-style transforms (grad, jit, vmap, scan)
├── splitmix.py      # SplitMix64 RNG, GF(3) colors
├── nn.py            # Neural network utilities
├── graphs.hy        # S-expression graph DSL
├── jax_bridge.hy    # JAX/JraphX integration
├── arch.hy          # Architecture DSL
└── cli.py           # Command-line interface
```

### GF(3) Triadic Structure

| Module | Trit | Role |
|--------|------|------|
| `graphs.hy` | PLUS (+1) | Generation |
| `nn.py` | MINUS (-1) | Validation/Gradients |
| `jax_bridge.hy` | ERGODIC (0) | Coordination |

## Optional Dependencies

| Extra | Packages | Use Case |
|-------|----------|----------|
| `[graphs]` | mlx-graphs | GNNs on Apple Silicon |
| `[jax]` | jax, flax, optax | JAX ecosystem |
| `[jraphx]` | jraphx | PyG-style GNNs in JAX |
| `[rl]` | rlax | Reinforcement learning |
| `[mcts]` | mctx | Monte Carlo tree search |
| `[full-jax]` | All JAX libs | Complete stack |
| `[dev]` | pytest, ruff, mypy | Development |

## Transform Comparison

| Transform | JAX | MLX | HyMLX |
|-----------|-----|-----|-------|
| `grad` | ✓ | ✓ | ✓ |
| `value_and_grad` | ✓ | ✓ | ✓ |
| `jit` | ✓ | `mx.compile` | ✓ |
| `vmap` | ✓ | ✗ | ✓ (unrolled) |
| `scan` | ✓ | ✗ | ✓ |
| `fori_loop` | ✓ | ✗ | ✓ |
| `pmap` | ✓ | ✗ | planned |

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run Hy tests
hy -c "(import pytest) (pytest.main [\"tests/test_graphs.hy\" \"-v\"])"

# Lint
ruff check src/

# Type check
mypy src/hymlx/
```

## License

MIT - see [LICENSE](LICENSE)

## Links

- [Repository](https://github.com/Soft-Machine-io/soft-machine)
- [MLX](https://github.com/ml-explore/mlx)
- [Hy](https://github.com/hylang/hy)
- [Gay.jl](https://github.com/TeglonLabs/Gay.jl) - Color generation inspiration
- [JraphX](https://dirt.design/jraphx/) - JAX GNN library
