#!/usr/bin/env hy
;; HyJAX Bridge - JraphX ↔ HyMLX Graph Integration
;;
;; Provides bidirectional bridges between:
;; - HyGraph (our s-expression DSL)
;; - mlx_graphs.GraphData (Apple Silicon)
;; - jraphx.data.Data (JAX/Flax NNX)
;;
;; GF(3) Trit Assignment: ERGODIC (0) - Coordination layer

(require hyrule [-> ->> unless])

;; ============================================
;; Conditional Imports
;; ============================================

(setv HAS_JAX False)
(setv HAS_JRAPHX False)
(setv HAS_RLAX False)
(setv HAS_MCTX False)

(try
  (import jax)
  (import jax.numpy :as jnp)
  (setv HAS_JAX True)
  (except [ImportError]))

(try
  (import jraphx.data [Data Batch])
  (import jraphx.nn.conv [GCNConv SAGEConv GATConv MessagePassing])
  (import jraphx.nn.models [GCN GAT GraphSAGE])
  (setv HAS_JRAPHX True)
  (except [ImportError]))

(try
  (import rlax)
  (setv HAS_RLAX True)
  (except [ImportError]))

(try
  (import mctx)
  (setv HAS_MCTX True)
  (except [ImportError]))

;; Import our local modules
(import hymlx.graphs [HyGraph graph digraph ->mlx-graph <-mlx-graph])
(import mlx.core :as mx)

;; ============================================
;; JraphX Data Bridge
;; ============================================

(defn ->jraphx [g [node-features None] [edge-features None] [y None]]
  "Convert HyGraph to jraphx.data.Data
   
   JraphX Data structure (PyG-style):
   - x: [num_nodes, num_features] node features
   - edge_index: [2, num_edges] COO format
   - edge_attr: [num_edges, edge_features] optional
   - y: target labels
   
   Example:
     (->jraphx (graph [0 1] [1 2] [2 0]))
  "
  (unless HAS_JRAPHX
    (raise (ImportError "jraphx not installed: pip install jraphx")))
  
  ;; Create node index mapping (sorted for determinism)
  (setv node-list (sorted (g.nodes)))
  (setv node-idx (dict (lfor #(i n) (enumerate node-list) #(n i))))
  (setv num-nodes (len node-list))
  
  ;; Build edge_index in COO [2, num_edges]
  (setv sources [])
  (setv targets [])
  (for [#(u v) (g.edges)]
    (.append sources (get node-idx u))
    (.append targets (get node-idx v)))
  
  (setv edge-index (jnp.array [sources targets] :dtype jnp.int32))
  
  ;; Default node features: one-hot identity
  (when (is node-features None)
    (setv node-features (jnp.eye num-nodes :dtype jnp.float32)))
  
  ;; Construct Data object
  (if (and (is edge-features None) (is y None))
      (Data :x node-features :edge_index edge-index)
      (Data :x node-features 
            :edge_index edge-index
            :edge_attr edge-features
            :y y)))

(defn <-jraphx [data [node-names None]]
  "Convert jraphx.data.Data back to HyGraph
   
   Example:
     (<-jraphx jraphx-data)
  "
  (setv edge-index (data.edge_index.tolist))
  (setv sources (get edge-index 0))
  (setv targets (get edge-index 1))
  
  (when (is node-names None)
    (setv n-nodes (int (+ 1 (max (max sources) (max targets)))))
    (setv node-names (list (range n-nodes))))
  
  (setv edges (lfor #(s t) (zip sources targets)
                    [(get node-names s) (get node-names t)]))
  (graph #* edges))

;; ============================================
;; MLX ↔ JAX Array Conversion
;; ============================================

(defn mx->jnp [arr]
  "Convert MLX array to JAX array via numpy"
  (unless HAS_JAX
    (raise (ImportError "jax not installed")))
  (jnp.array (.tolist arr)))

(defn jnp->mx [arr]
  "Convert JAX array to MLX array via numpy"
  (mx.array (.tolist arr)))

;; ============================================
;; Unified Graph Protocol
;; ============================================

(defclass HyJaxGraph []
  "Unified graph supporting both MLX and JAX backends.
   
   Holds the canonical HyGraph representation and lazily
   converts to backend-specific formats on demand.
   
   GF(3) Conservation: This is the ERGODIC (0) coordinator.
  "
  
  (defn __init__ [self hy-graph]
    (setv self._hy hy-graph)
    (setv self._mlx-cache None)
    (setv self._jax-cache None))
  
  (defn hy [self]
    "Get HyGraph (canonical form)"
    self._hy)
  
  (defn mlx [self [node-features None]]
    "Get mlx_graphs.GraphData (cached)"
    (when (is self._mlx-cache None)
      (setv self._mlx-cache (->mlx-graph self._hy node-features)))
    self._mlx-cache)
  
  (defn jax [self [node-features None]]
    "Get jraphx.data.Data (cached)"
    (when (is self._jax-cache None)
      (setv self._jax-cache (->jraphx self._hy node-features)))
    self._jax-cache)
  
  (defn invalidate [self]
    "Clear caches after mutation"
    (setv self._mlx-cache None)
    (setv self._jax-cache None))
  
  ;; Delegate common methods to HyGraph
  (defn nodes [self] (self._hy.nodes))
  (defn edges [self] (self._hy.edges))
  (defn successors [self node] (self._hy.successors node))
  (defn predecessors [self node] (self._hy.predecessors node)))

(defn hyjax-graph [#* specs]
  "Create unified graph from specs.
   
   (hyjax-graph [0 1] [1 2] [2 0])
  "
  (HyJaxGraph (graph #* specs)))

;; ============================================
;; GNN Layer Wrappers (Hy DSL)
;; ============================================

(defmacro defjax-gnn [name [#* layer-specs]]
  "Define a JraphX GNN model with Hy syntax.
   
   (defjax-gnn NodeClassifier
     (GCNConv 64 32)
     (GCNConv 32 num-classes))
   
   Expands to Flax NNX module.
  "
  (setv layers layer-specs)
  `(do
     (import flax.nnx :as nnx)
     (import jax.nn :as jnn)
     
     (defclass ~name [nnx.Module]
       (defn __init__ [self rngs]
         ~@(lfor #(i layer) (enumerate layers)
                 `(setv ~(hy.models.Symbol f"self.conv{i}") 
                        (~(get layer 0) ~@(cut layer 1) :rngs rngs))))
       
       (defn __call__ [self data]
         (setv x data.x)
         (setv edge-index data.edge_index)
         ~@(lfor #(i _) (enumerate (butlast layers))
                 `(setv x (jnn.relu (~(hy.models.Symbol f"self.conv{i}") x edge-index))))
         ;; Last layer without activation
         (~(hy.models.Symbol f"self.conv{(- (len layers) 1)}") x edge-index)))))

;; ============================================
;; RLax Integration (MINUS trit)
;; ============================================

(defn td-error [value target]
  "Temporal difference error (validation signal)"
  (unless HAS_JAX
    (raise (ImportError "jax not installed")))
  (jnp.subtract target value))

(defn td-target [reward gamma next-value]
  "TD target: r + γV(s')"
  (unless HAS_JAX
    (raise (ImportError "jax not installed")))
  (jnp.add reward (jnp.multiply gamma next-value)))

(defn rlax-td-lambda [rewards values discount lambda_]
  "RLax-style TD(λ) returns"
  (unless HAS_RLAX
    (raise (ImportError "rlax not installed: pip install rlax")))
  (rlax.td_lambda :r_t rewards 
                  :v_t values 
                  :discount_t discount
                  :lambda_ lambda_))

;; ============================================
;; MCTX Integration (ERGODIC trit)
;; ============================================

(defn mctx-policy-output [prior-logits value]
  "Create MCTX policy output for tree search"
  (unless HAS_MCTX
    (raise (ImportError "mctx not installed: pip install mctx")))
  (mctx.PolicyOutput prior-logits value))

;; ============================================
;; GF(3) Triadic Stack
;; ============================================

(defclass TriadicStack []
  "GF(3)-balanced JAX computation stack.
   
   Ensures PLUS + MINUS + ERGODIC = 0 (mod 3)
   
   PLUS (+1):   Forward/generation (GNN forward pass)
   MINUS (-1):  Backward/validation (gradients, TD error)
   ERGODIC (0): Coordination (MCTS, equilibria)
  "
  
  (defn __init__ [self]
    (setv self.plus-ops [])
    (setv self.minus-ops [])
    (setv self.ergodic-ops []))
  
  (defn add-plus [self op]
    "Add generation operation"
    (.append self.plus-ops op)
    self)
  
  (defn add-minus [self op]
    "Add validation operation"
    (.append self.minus-ops op)
    self)
  
  (defn add-ergodic [self op]
    "Add coordination operation"
    (.append self.ergodic-ops op)
    self)
  
  (defn balanced? [self]
    "Check GF(3) conservation"
    (setv total (+ (len self.plus-ops) 
                   (* -1 (len self.minus-ops))
                   (* 0 (len self.ergodic-ops))))
    (= (% total 3) 0))
  
  (defn trit-sum [self]
    "Current trit sum mod 3"
    (% (- (len self.plus-ops) (len self.minus-ops)) 3)))

;; ============================================
;; Stellogen Correspondence
;; ============================================

(defn stellogen->jax-graph [constellation]
  "Convert Stellogen constellation to JAX graph.
   
   Stellogen stars with (+node X) rays become nodes.
   Stars with (+edge X Y) rays become edges.
   
   This bridges logic programming (Stellogen) with 
   differentiable programming (JAX).
  "
  (setv nodes [])
  (setv edges [])
  
  (for [star constellation]
    (for [ray star]
      (when (and (isinstance ray tuple) (>= (len ray) 2))
        (cond
          (= (get ray 0) "+node")
          (.append nodes (get ray 1))
          
          (and (= (get ray 0) "+edge") (>= (len ray) 3))
          (.append edges [(get ray 1) (get ray 2)])))))
  
  (hyjax-graph #* edges))

;; ============================================
;; Exports
;; ============================================

(setv __all__ [
  ;; Bridge functions
  "->jraphx" "<-jraphx"
  "mx->jnp" "jnp->mx"
  ;; Unified graph
  "HyJaxGraph" "hyjax-graph"
  ;; GF(3) stack
  "TriadicStack"
  ;; RLax wrappers
  "td-error" "td-target" "rlax-td-lambda"
  ;; MCTX wrappers
  "mctx-policy-output"
  ;; Stellogen bridge
  "stellogen->jax-graph"
  ;; Feature flags
  "HAS_JAX" "HAS_JRAPHX" "HAS_RLAX" "HAS_MCTX"
])
