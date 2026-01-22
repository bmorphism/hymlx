# Deployment Pathways

## Core Systems
- [x] **HyMLX**: JAX-style transforms and Hy DSL (Released v0.2.0)
  - Method: `pip install hymlx` / `uvx`
  - Repo: `bmorphism/hymlx`

## External Integrations & Servers
- [ ] **RGB (andrewgazelka/rgb)**: High-performance Rust Minecraft server
  - **Role**: Target environment for "World" simulations (Data-Parallel Entity Processing).
  - **Deployment**:
    - **Nix (Recommended)**: `nix run github:andrewgazelka/rgb`
    - **Manual**:
      - Clone: `git clone https://github.com/andrewgazelka/rgb`
      - CI/Test: `./ci.sh`
    - **Boxxy Deployment (Verified)**:
      - *Prerequisites*: Clone sibling repos `Flecs-Rust`, `nebari`, `chumsky` to parent dir. Patch `Cargo.toml` paths from `/Users/andrewgazelka/...` to relative `../../../...`.
      - *Command*: `nix --extra-experimental-features "nix-command flakes" develop --command cargo run --release --bin mc-server-runner`
  - **Synergy**: Use `hymlx` to generate deterministic color palettes/blocks for RGB server plugins. The server uses a Data-Parallel (R, G, B) architecture that aligns with hymlx's triadic structure.
