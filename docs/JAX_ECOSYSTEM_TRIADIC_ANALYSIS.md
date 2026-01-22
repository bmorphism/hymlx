# JAX Ecosystem Triadic Analysis for HyMLX Integration

## Three Maximally Oppositional Paradigms (from Stellogen)

Based on the `maximal_maximal.sg` exploration, we identify three fundamental paradigms that map to GF(3):

| Paradigm | Trit | Stellogen Example | JAX Equivalent |
|----------|------|-------------------|----------------|
| **GENERATION** (+1) | PLUS | `skill_plus`, `add` interaction nets | Forward models, generators |
| **VALIDATION** (-1) | MINUS | `skill_minus`, type system `spec nat` | Testing, verification, gradients |
| **COORDINATION** (0) | ERGODIC | `skill_zero`, database joins, SPI chains | Optimization, MCTS, equilibria |

---

## JAX Libraries by GF(3) Trit Assignment

### PLUS (+1): Generative / Forward Libraries

| Library | Stars | Description | HyMLX Integration |
|---------|-------|-------------|-------------------|
| **Flax** | ~6k | Neural network library | ⭐⭐⭐ Native MLX bridge exists |
| **Equinox** | ~2k | Elegant NN + scientific computing | ⭐⭐⭐ Pytree-compatible |
| **Diffrax** | ~1.5k | Differential equations | ⭐⭐ ODEs for dynamics |
| **EvoJAX** | ~900 | Neuroevolution | ⭐⭐ Population-based |
| **JaxLife** | new | Open-ended agentic simulator | ⭐⭐⭐ Emergent behavior |

### MINUS (-1): Validation / Backward Libraries

| Library | Stars | Description | HyMLX Integration |
|---------|-------|-------------|-------------------|
| **RLax** | ~1.4k | RL building blocks (TD, policy gradients) | ⭐⭐⭐ Core RL primitives |
| **Optax** | ~1.6k | Gradient transformations | ⭐⭐⭐ Already JAX-native |
| **kfac-jax** | ~500 | Second-order optimization | ⭐⭐ Curvature estimation |
| **Chex** | ~700 | Testing utilities | ⭐⭐⭐ Type assertions |
| **dm_pix** | ~400 | Image processing | ⭐⭐ Vision validation |

### ERGODIC (0): Coordination / Equilibrium Libraries

| Library | Stars | Description | HyMLX Integration |
|---------|-------|-------------|-------------------|
| **mctx** | ~2.3k | Monte Carlo tree search | ⭐⭐⭐ Planning/search |
| **JaxMARL** | ~1k | Multi-agent RL | ⭐⭐⭐ Agent coordination |
| **PGX** | ~1.2k | JAX game environments | ⭐⭐ Game equilibria |
| **Brax** | ~2.2k | Physics simulation | ⭐⭐ Sim environments |
| **Gymnax** | ~600 | JAX RL environments | ⭐⭐⭐ Fast envs |

---

## Graph Neural Network Libraries

| Library | Status | Backend | HyMLX Integration Notes |
|---------|--------|---------|------------------------|
| **jraph** | ⚠️ ARCHIVED (May 2025) | JAX | Legacy, avoid for new projects |
| **JraphX** | ✅ Active | JAX/Flax NNX | PyG-inspired, successor to jraph |
| **mlx_graphs** | ✅ Active | MLX | ⭐⭐⭐ **Primary target** - Apple Silicon native |
| **jax-gnn** | 🔸 Minimal | JAX | Too minimal, 1 star |

**Recommendation**: `mlx_graphs` is the primary target (already integrated in `graphs.hy`), with `JraphX` as JAX-side complement.

---

## Ground State Mixes (Triadic Combinations)

Following the Stellogen pattern where `(+gf3 plus minus) -> ergodic`:

### Mix 1: RL + Graphs = Graph RL
```
PLUS (Flax/Equinox) + MINUS (RLax) + ERGODIC (mctx)
= GraphRL constellation
```

**Implementation Path**:
```hy
;; HyMLX Graph RL pattern
(defn graph-policy [g features]
  "GNN-based policy for graph-structured MDPs"
  (let [node-embeds (gcn-layer g features)
        pooled (global-mean-pool node-embeds)]
    (policy-head pooled)))
```

### Mix 2: Evolution + Validation = Curriculum Learning
```
PLUS (EvoJAX) + MINUS (Chex) + ERGODIC (JaxMARL)
= Adaptive curriculum constellation
```

### Mix 3: Dynamics + Gradients = Neural ODEs
```
PLUS (Diffrax) + MINUS (Optax) + ERGODIC (Brax)
= Differentiable physics constellation
```

---

## Stellogen ↔ JAX Correspondence Table

| Stellogen Concept | JAX/HyMLX Equivalent |
|-------------------|---------------------|
| `(fire ...)` linear execution | Single forward pass, no reuse |
| `(exec ...)` non-linear execution | Training loop with reuse |
| `(+/- polarity)` rays | Forward/backward pass duality |
| `@focus` state distinction | `nnx.Param` vs `nnx.Variable` |
| `(spec ...)` type system | `chex.assert_*` / shape typing |
| Constellation composition | `nn.Sequential` / `haiku.transform` |
| Unification | Pattern matching / einsum |

---

## Priority Integration Ranking

### Tier 1: Immediate (Already Compatible)
1. **mlx_graphs** - ✅ Already in `graphs.hy`
2. **Optax** - Direct gradient transforms
3. **RLax** - RL primitives, pure functions

### Tier 2: High Value (Moderate Effort)
4. **mctx** - MCTS for planning, pure JAX
5. **Equinox** - Pytree-based, elegant bridge
6. **JraphX** - Modern GNN, PyG-style API

### Tier 3: Strategic (Requires Wrappers)
7. **Diffrax** - Neural ODEs
8. **JaxMARL** - Multi-agent
9. **EvoJAX** - Neuroevolution

---

## RLax Deep Dive

RLax provides the MINUS (-1) trit primitives:

```python
# Core value functions (validation/backward)
rlax.td_lambda()           # TD(λ) returns
rlax.q_learning()          # Q-value targets  
rlax.sarsa()               # On-policy TD
rlax.expected_sarsa()      # Expected SARSA

# Policy gradients (gradient flow)
rlax.policy_gradient_loss()
rlax.clipped_surrogate_pg_loss()  # PPO

# Distributional RL
rlax.categorical_td_learning()
rlax.quantile_q_learning()
```

**HyMLX Integration Pattern**:
```hy
;; rlax-style TD target in Hy
(defn td-target [reward gamma next-value]
  "Temporal difference target: r + γV(s')"
  (+ reward (* gamma next-value)))

(defn td-error [value target]
  "MINUS trit: validation signal"
  (- target value))
```

---

## Open Games ↔ JAX RL Mapping

From `maximal_maximal.sg`:
```stellogen
(def pd_cc [(+s1 c) (+s2 c) (+u 3)])
(def br [(+opp X) (+br d)])  ; Best response
```

JAX equivalent via compositional game theory:
```python
# Open game structure in JAX
@dataclass
class OpenGame:
    play: Callable[[State], Action]      # PLUS: forward
    coplay: Callable[[State, Costate], Costate]  # MINUS: backward
    
# Nash equilibrium = fixed point of best response
def nash_equilibrium(game, initial):
    return jax.lax.while_loop(
        lambda x: not converged(x),
        lambda x: best_response(game, x),
        initial
    )
```

---

## Summary: Triadic JAX Stack

```
┌─────────────────────────────────────────────────────────────┐
│                    ERGODIC (0)                               │
│         mctx · JaxMARL · PGX · Brax · Gymnax                │
│                   (Coordination Layer)                       │
├─────────────────────────────────────────────────────────────┤
│   PLUS (+1)              │           MINUS (-1)              │
│   Flax · Equinox         │     RLax · Optax · kfac-jax      │
│   Diffrax · EvoJAX       │     Chex · dm_pix                │
│   (Generation)           │     (Validation)                  │
├─────────────────────────────────────────────────────────────┤
│                      GRAPHS                                  │
│         mlx_graphs (MLX) · JraphX (JAX) · HyGraph (Hy)      │
│                   (Substrate Layer)                          │
└─────────────────────────────────────────────────────────────┘
```

The ground state is achieved when all three trits sum to 0 (mod 3):
- **Balanced stack**: 1 PLUS + 1 MINUS + 1 ERGODIC = 0 ✓
- **Example**: Flax (gen) + RLax (val) + mctx (coord) = stable RL agent
