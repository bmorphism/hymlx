# Changelog

All notable changes to HyMLX will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-01-21

### Added

- **graphs.hy**: S-expression DSL for graph construction
  - `HyGraph` immutable graph class with Clojure Loom-style API
  - Functional constructors: `graph`, `digraph`, `weighted-graph`
  - Graph algorithms: `topological-sort`, `dfs`, `bfs`, `strongly-connected-components`
  - Standard generators: `complete-graph`, `path-graph`, `cycle-graph`, `grid-graph`
  - `->mlx-graph` / `<-mlx-graph` bridges to mlx_graphs.GraphData
  - `NaryaGraphType` for type-theoretic graph property verification

- **jax_bridge.hy**: JAX/JraphX integration layer
  - `->jraphx` / `<-jraphx` bridges to jraphx.data.Data
  - `HyJaxGraph` unified graph supporting both MLX and JAX backends
  - `TriadicStack` for GF(3)-balanced computation tracking
  - RLax wrappers: `td-error`, `td-target`, `rlax-td-lambda`
  - MCTX integration: `mctx-policy-output`
  - Stellogen constellation to JAX graph bridge

- **Optional dependencies**: 
  - `[graphs]` for mlx-graphs
  - `[jax]` for JAX ecosystem
  - `[jraphx]` for JraphX GNN library
  - `[rl]` for RLax reinforcement learning
  - `[mcts]` for MCTX tree search
  - `[full-jax]` for complete JAX stack

- GF(3) triadic structure documentation in module docstrings

### Changed

- Version bump to 0.2.0
- Updated README with new features
- Added Hy and hyrule as core dependencies

## [0.1.0] - 2025-01-15

### Added

- Initial release
- **transforms.py**: JAX-style transformations for MLX
  - `grad` / `value_and_grad` - automatic differentiation
  - `jit` - JIT compilation via `mx.compile`
  - `vmap` - vectorized mapping (unrolled)
  - `scan` - functional loops
  - `fori_loop` - indexed loops

- **splitmix.py**: SplitMix64 deterministic RNG
  - Gay.jl-compatible color generation
  - `seed_to_trit`, `seed_to_hue`, `seed_to_rgb`
  - `spawn_triad` for GF(3)-balanced color triads
  - `check_gf3` conservation verification

- **nn.py**: Neural network utilities
  - `sequential` layer composition
  - `mlp_seeded` deterministic MLP construction
  - `train_step` functional training step

- **cli.py**: Command-line interface
  - `hymlx demo` - run demonstration
  - `hymlx color` - generate colors
  - `hymlx transforms` - show transform examples

## [Unreleased]

### Planned

- `arch.hy` - Architecture DSL for model definitions
- `pmap` - Parallel mapping across devices
- Extended JraphX layer support
- Narya formal verification integration
