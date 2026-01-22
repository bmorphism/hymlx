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
    - **Boxxy Deployment**:
      - *Prerequisites*: Sufficient resources (High-perf Rust compilation) + "shenanigans" (AI-authored codebase handling).
      - *Command*: TBD (Requires integration with Boxxy agent context).
  - **Synergy**: Use `hymlx` to generate deterministic color palettes/blocks for RGB server plugins. The server uses a Data-Parallel (R, G, B) architecture that aligns with hymlx's triadic structure.
