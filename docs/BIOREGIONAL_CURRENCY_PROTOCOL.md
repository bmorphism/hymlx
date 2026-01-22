# The Bioregional Currency Protocol
> *Alterpolitics via Bicomodule Coherence*

**Status**: Experimental / Formalized
**Seed**: 1069 (Balanced Ternary: `[+1, -1, -1, +1, +1, +1, +1]`)
**Type**: Working Language System

## 1. The Core Thesis

The "Alterpolitics of Bioregional Currencies" is not just a policy goal; it is a **structural necessity** of a coherent Working Language system.

By formalizing the movement of information through a bicomodule structure ($C \triangleleft M \triangleleft D$), we prove that **sustainable resource management** (energy, water, food) requires a specific mathematical architecture:

1.  **No Extraction without Observation** ($\epsilon \circ \delta = id$): You cannot dispatch energy ($\epsilon$) without first observing the grid state ($\delta$).
2.  **Local Context Preservation** ($\delta$): Information must flow *upward* from the land (Data) to the interface (Market), adding context at each step.
3.  **Deterministic Trust** (Seed 1069): The rules of the system (the "Constitution") must be public, verifiable, and immutable, like the expansion of a mathematical seed.

## 2. The Formal Grammar of Currency

We define a "Currency" not as a store of value, but as a **Bicomodule Movement**.

| Component | Formal Symbol | Bioregional Role | `hymlx` Implementation |
| :--- | :--- | :--- | :--- |
| **Grid / Land** | $D$ (Data) | The physical reality (Watts, Joules, Liters). | `GridPoly` (Physical Flow) |
| **Market / Currency** | $M$ (Interface) | The signaling layer (Price, Credit, Mutual Credit). | `MarketPoly` (Signal) |
| **Protocol / Policy** | $C$ (Control) | The community governance logic. | `ProtocolPoly` (Strategy) |

### The Three Movements of Alterpolitics

1.  **$\delta$ (The Listening Move)**: *Upward Comultiplication*
    - **Logic**: $D \to M \triangleleft D$
    - **Meaning**: "Sensing the bioregion." The physical state of the land is lifted into the currency layer.
    - **Alterpolitics**: Prices are not set by distant markets, but discovered from local ecological constraints.

2.  **$\sigma$ (The Exchange Move)**: *Lateral Effect Handling*
    - **Logic**: $M \triangleleft D \to C \triangleleft M$
    - **Meaning**: "Negotiating value." The local currency mediates between physical needs and community governance.
    - **Alterpolitics**: This is the "Market Mechanism" - but one that is *handled* by the community's chosen logic ($C$), not an external extractor.

3.  **$\epsilon$ (The Action Move)**: *Downward Counit*
    - **Logic**: $M \triangleleft D \to D$
    - **Meaning**: "Dispatching resources." The decision turns back into physical action (turning on a pump, dispatching a battery).
    - **Alterpolitics**: "Symbols directing matter." The currency *causes* physical change in a verifiable loop.

## 3. The Infrastructure Stack

To build this, we need a "Working Language" that can execute these moves.

### A. `hymlx`: The Coordination Layer
*   **Role**: Ensures **GF(3) Balance** (`trit_sum ≡ 0`).
*   **Why**: A sustainable economy must be *ergodic* (time-average = ensemble-average). `hymlx` tracks this:
    *   **Generation (+)**: Creating value.
    *   **Validation (-)**: Consuming/Verifying value.
    *   **Coordination (0)**: Circulating value.
*   **Command**: `uvx --from git+https://github.com/bmorphism/hymlx.git hymlx world`

### B. `RGB`: The Simulation Layer
*   **Role**: High-performance, data-parallel entity simulation.
*   **Why**: Before deploying to the real grid, we simulate the bioregion in a "World" (Minecraft/Rust).
*   **Deployment**: `nix run github:andrewgazelka/rgb`

### C. `Plurigrid`: The Energy Layer
*   **Role**: The physical implementation of the protocol on microgrids.
*   **Why**: Energy is the fundamental currency of the bioregion.

## 4. The "Constitution" (Seed 1069)

Why Seed 1069? It represents the **Arbitrary but Shared** foundation of trust. 

*   It generates the **Balanced Ternary Pattern**: `[+1, -1, -1, +1, +1, +1, +1]`.
*   This pattern ensures that "Growth" (+1) is balanced by "Constraint" (-1) in a precise, predictable rhythm.
*   In an alterpolitical system, this replaces the "Central Bank." The monetary policy is code, visible to all, derived from a shared mathematical root.

## 5. Conclusion: From Code to Commons

This protocol does not just *describe* a bioregional economy; it **compiles** it.

By adopting the Bicomodule structure, we ensure that our economic software cannot structurally support extraction or dissociation. It *must* listen ($\delta$) before it acts ($\epsilon$). It *must* balance its books (GF(3)).

**This is the "Working Language" of the commons.**
