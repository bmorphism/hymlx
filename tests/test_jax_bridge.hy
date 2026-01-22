#!/usr/bin/env hy
;; Tests for HyJAX Bridge

(import pytest)
(import hymlx.graphs [graph digraph HyGraph])

;; Conditional imports
(setv HAS_JAX False)
(setv HAS_JRAPHX False)

(try
  (import jax)
  (import jax.numpy :as jnp)
  (setv HAS_JAX True)
  (except [ImportError]))

(try
  (import jraphx.data [Data])
  (setv HAS_JRAPHX True)
  (except [ImportError]))

;; ============================================
;; HyGraph Tests (no JAX required)
;; ============================================

(defn test-hygraph-basic []
  "Test basic HyGraph construction"
  (setv g (graph [0 1] [1 2] [2 0]))
  (assert (= (len (g.nodes)) 3))
  (assert (= (len (g.edges)) 3))
  (assert (g.has-node? 0))
  (assert (g.has-edge? 0 1)))

(defn test-hygraph-successors []
  "Test successors/predecessors"
  (setv g (graph [0 1] [0 2] [1 2]))
  (assert (= (g.successors 0) (frozenset [1 2])))
  (assert (= (g.predecessors 2) (frozenset [0 1]))))

;; ============================================
;; JraphX Bridge Tests
;; ============================================

(defn test-hyjax-graph-creation []
  "Test unified HyJaxGraph"
  (when (not HAS_JAX)
    (pytest.skip "JAX not installed"))
  
  (import hymlx.jax_bridge [HyJaxGraph hyjax-graph])
  
  (setv g (hyjax-graph [0 1] [1 2] [2 0]))
  
  ;; Check HyGraph access
  (assert (= (len (.nodes (.hy g))) 3))
  (assert (= (len (.edges (.hy g))) 3)))

(defn test-to-jraphx-conversion []
  "Test HyGraph -> JraphX Data conversion"
  (when (not HAS_JRAPHX)
    (pytest.skip "JraphX not installed"))
  
  (import hymlx.jax_bridge [->jraphx <-jraphx])
  
  (setv g (graph [0 1] [1 2] [2 0]))
  (setv data (->jraphx g))
  
  ;; Check Data structure
  (assert (= (tuple data.x.shape) #(3 3)))  ; 3 nodes, one-hot
  (assert (= (tuple data.edge_index.shape) #(2 3))))  ; 3 edges

(defn test-roundtrip-jraphx []
  "Test HyGraph -> JraphX -> HyGraph roundtrip"
  (when (not HAS_JRAPHX)
    (pytest.skip "JraphX not installed"))
  
  (import hymlx.jax_bridge [->jraphx <-jraphx])
  
  (setv g1 (graph [0 1] [1 2] [2 3]))
  (setv data (->jraphx g1))
  (setv g2 (<-jraphx data))
  
  ;; Check same structure
  (assert (= (len (g1.nodes)) (len (g2.nodes))))
  (assert (= (len (g1.edges)) (len (g2.edges)))))

;; ============================================
;; GF(3) Triadic Stack Tests
;; ============================================

(defn test-triadic-stack []
  "Test GF(3) balance tracking"
  (when (not HAS_JAX)
    (pytest.skip "JAX not installed"))
  
  (import hymlx.jax_bridge [TriadicStack])
  
  (setv stack (TriadicStack))
  
  ;; Empty stack is balanced (0 mod 3 = 0)
  (assert (.balanced? stack))
  
  ;; Add one PLUS -> sum = 1
  (.add-plus stack "forward")
  (assert (= (.trit-sum stack) 1))
  (assert (not (.balanced? stack)))
  
  ;; Add one MINUS -> sum = 0
  (.add-minus stack "backward")
  (assert (= (.trit-sum stack) 0))
  (assert (.balanced? stack))
  
  ;; Add ERGODIC doesn't change sum
  (.add-ergodic stack "coordinate")
  (assert (= (.trit-sum stack) 0))
  (assert (.balanced? stack)))

(defn test-triadic-full-cycle []
  "Test PLUS + PLUS + PLUS = 0 (mod 3)"
  (when (not HAS_JAX)
    (pytest.skip "JAX not installed"))
  
  (import hymlx.jax_bridge [TriadicStack])
  
  (setv stack (TriadicStack))
  (.add-plus stack "gen1")
  (.add-plus stack "gen2")
  (.add-plus stack "gen3")
  
  ;; 3 PLUS ops -> 3 mod 3 = 0 (balanced!)
  (assert (.balanced? stack)))

;; ============================================
;; Array Conversion Tests
;; ============================================

(defn test-mx-jnp-conversion []
  "Test MLX <-> JAX array conversion"
  (when (not HAS_JAX)
    (pytest.skip "JAX not installed"))
  
  (import mlx.core :as mx)
  (import hymlx.jax_bridge [mx->jnp jnp->mx])
  
  (setv arr-mx (mx.array [[1 2] [3 4]]))
  (setv arr-jax (mx->jnp arr-mx))
  (setv arr-back (jnp->mx arr-jax))
  
  ;; Check roundtrip
  (assert (= (.tolist arr-mx) (.tolist arr-back))))

;; ============================================
;; Stellogen Bridge Tests  
;; ============================================

(defn test-stellogen-to-jax []
  "Test Stellogen constellation -> JAX graph"
  (when (not HAS_JAX)
    (pytest.skip "JAX not installed"))
  
  (import hymlx.jax_bridge [stellogen->jax-graph])
  
  ;; Mock constellation (simplified structure)
  (setv constellation [
    [("+node" "a") ("+node" "b")]
    [("+edge" "a" "b")]
    [("+edge" "b" "c")]
  ])
  
  (setv g (stellogen->jax-graph constellation))
  
  ;; Should create a graph from the constellation
  (assert (isinstance (.hy g) HyGraph)))

;; ============================================
;; Run tests
;; ============================================

(when (= __name__ "__main__")
  (pytest.main [__file__ "-v"]))
