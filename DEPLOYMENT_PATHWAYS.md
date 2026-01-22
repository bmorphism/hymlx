# Deployment Pathways

## Core Systems
- [x] **HyMLX**: JAX-style transforms and Hy DSL (Released v0.2.0)
  - Method: `pip install hymlx` / `uvx`
  - Repo: `bmorphism/hymlx`

## External Integrations & Servers
- [ ] **RGB (andrewgazelka/rgb)**: High-performance Rust Minecraft server
  - **Role**: Target environment for "World" simulations? (Potential integration)
  - **Deployment**:
    - Clone: `git clone https://github.com/andrewgazelka/rgb`
    - Build: `cargo build --release` (Requires Rust)
    - Run: `./target/release/rgb`
  - **Synergy**: Use `hymlx` to generate deterministic color palettes/blocks for RGB server plugins.
