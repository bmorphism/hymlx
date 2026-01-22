#!/usr/bin/env hy
;; HyMLX Graphs - S-expression Graph DSL for mlx_graphs
;; 
;; Design principles:
;; - Clojure Loom: (graph [1 2] [2 3]) edge-pair syntax
;; - SRFI-234: Topological sort, functional graph patterns
;; - OCamlGraph: Functorial abstraction, persistent vs imperative
;; - Narya: Type-theoretic verification of graph properties

(import mlx.core :as mx)

;; Try to import mlx_graphs, provide stubs if not available
(try
  (import mlx_graphs [GraphData])
  (import mlx_graphs.nn [GCNConv SAGEConv GATConv])
  (import mlx_graphs.loaders [Planetoid])
  (setv HAS_MLX_GRAPHS True)
  (except [ImportError]
    (setv HAS_MLX_GRAPHS False)))

;; ============================================
;; Core Graph Data (Clojure Loom-style)
;; ============================================

(defclass HyGraph []
  "Immutable graph with s-expression construction.
   
   Inspired by:
   - Clojure Loom: (graph [1 2] [2 3] {3 [4 5]})
   - OCamlGraph: Persistent.Digraph.Abstract
   - SRFI-234: Topological ordering
  "
  
  (defn __init__ [self]
    "Empty graph"
    (setv self._nodes (frozenset))
    (setv self._edges (frozenset))
    (setv self._adj {})       ; node -> set of successors
    (setv self._pred {})      ; node -> set of predecessors
    (setv self._node-attrs {}) ; node -> attrs dict
    (setv self._edge-attrs {}) ; (u v) -> attrs dict
    (setv self._directed True))
  
  (defn nodes [self]
    "All nodes (SRFI-style accessor)"
    self._nodes)
  
  (defn edges [self]
    "All edges as (u v) pairs"
    self._edges)
  
  (defn successors [self node]
    "Direct successors (Loom-style)"
    (.get self._adj node (frozenset)))
  
  (defn predecessors [self node]
    "Direct predecessors"
    (.get self._pred node (frozenset)))
  
  (defn neighbors [self node]
    "All neighbors (undirected view)"
    (| (self.successors node) (self.predecessors node)))
  
  (defn out-degree [self node]
    (len (self.successors node)))
  
  (defn in-degree [self node]
    (len (self.predecessors node)))
  
  (defn has-node? [self node]
    (in node self._nodes))
  
  (defn has-edge? [self u v]
    (in #(u v) self._edges))
  
  (defn node-attr [self node key [default None]]
    "Get node attribute"
    (.get (.get self._node-attrs node {}) key default))
  
  (defn edge-attr [self u v key [default None]]
    "Get edge attribute"
    (.get (.get self._edge-attrs #(u v) {}) key default)))

;; ============================================
;; Functional Graph Constructors (Loom-style)
;; ============================================

(defn graph [#* specs]
  "Build graph from edges, adjacency maps, or other graphs.
   
   Clojure Loom syntax:
     (graph [1 2] [2 3])           ; edges
     (graph {1 [2 3] 2 [4]})       ; adjacency map  
     (graph [1 2 10])              ; weighted edge
     (graph other-graph [4 5])     ; merge
  "
  (setv g (HyGraph))
  (setv nodes (set))
  (setv edges (set))
  (setv adj {})
  (setv pred {})
  (setv node-attrs {})
  (setv edge-attrs {})
  
  (for [spec specs]
    (cond
      ;; Edge: [u v] or [u v weight]
      (and (isinstance spec list) (>= (len spec) 2) (<= (len spec) 3))
      (do
        (setv u (get spec 0))
        (setv v (get spec 1))
        (.add nodes u)
        (.add nodes v)
        (.add edges #(u v))
        (when (not (in u adj)) (setv (get adj u) (set)))
        (when (not (in v pred)) (setv (get pred v) (set)))
        (.add (get adj u) v)
        (.add (get pred v) u)
        (when (= (len spec) 3)
          (setv (get edge-attrs #(u v)) {"weight" (get spec 2)})))
      
      ;; Adjacency map: {1 [2 3] 2 [4]}
      (isinstance spec dict)
      (for [#(u vs) (.items spec)]
        (.add nodes u)
        (when (not (in u adj)) (setv (get adj u) (set)))
        (for [v (if (isinstance vs dict) (.keys vs) vs)]
          (.add nodes v)
          (.add edges #(u v))
          (.add (get adj u) v)
          (when (not (in v pred)) (setv (get pred v) (set)))
          (.add (get pred v) u)
          (when (isinstance vs dict)
            (setv (get edge-attrs #(u v)) {"weight" (get vs v)}))))
      
      ;; Another graph (check BEFORE single node)
      (isinstance spec HyGraph)
      (do
        (setv nodes (| nodes (set spec._nodes)))
        (setv edges (| edges (set spec._edges)))
        (for [#(n ss) (.items spec._adj)]
          (if (in n adj)
              (setv (get adj n) (| (get adj n) ss))
              (setv (get adj n) (set ss))))
        (for [#(n ps) (.items spec._pred)]
          (if (in n pred)
              (setv (get pred n) (| (get pred n) ps))
              (setv (get pred n) (set ps)))))
      
      ;; Single node (fallback)
      True
      (.add nodes spec)))
  
  ;; Freeze into immutable graph
  (setv g._nodes (frozenset nodes))
  (setv g._edges (frozenset edges))
  (setv g._adj (dict (lfor #(k v) (.items adj) #(k (frozenset v)))))
  (setv g._pred (dict (lfor #(k v) (.items pred) #(k (frozenset v)))))
  (setv g._node-attrs node-attrs)
  (setv g._edge-attrs edge-attrs)
  g)

(defn digraph [#* specs]
  "Directed graph (alias, all HyGraphs are directed)"
  (graph #* specs))

(defn weighted-graph [#* specs]
  "Weighted graph - edges are [u v weight]"
  (graph #* specs))

;; ============================================
;; Graph Transformations (OCaml functor style)
;; ============================================

(defn add-nodes [g #* nodes]
  "Return new graph with added nodes (persistent)"
  (graph g #* nodes))

(defn add-edges [g #* edges]
  "Return new graph with added edges (persistent)"
  (graph g #* edges))

(defn add-path [g #* nodes]
  "Add path connecting nodes in order"
  (setv edges (lfor #(u v) (zip nodes (cut nodes 1 None)) [u v]))
  (graph g #* edges))

(defn add-cycle [g #* nodes]
  "Add cycle connecting nodes"
  (setv path-edges (lfor #(u v) (zip nodes (cut nodes 1 None)) [u v]))
  (setv close-edge [[(get nodes -1) (get nodes 0)]])
  (graph g #* (+ path-edges close-edge)))

(defn subgraph [g node-pred]
  "Filter nodes by predicate (OCamlGraph: nodes_filtered_by)"
  (setv kept (frozenset (filter node-pred (g.nodes))))
  (setv new-edges (lfor #(u v) (g.edges) :if (and (in u kept) (in v kept)) [u v]))
  (graph #* new-edges))

(defn mapped-by [g f]
  "Map function over nodes (OCamlGraph: mapped_by)"
  (setv new-edges (lfor #(u v) (g.edges) [(f u) (f v)]))
  (graph #* new-edges))

(defn transpose [g]
  "Reverse all edges"
  (setv rev-edges (lfor #(u v) (g.edges) [v u]))
  (graph #* rev-edges))

;; ============================================
;; Graph Algorithms (SRFI-234 style)
;; ============================================

(defn topological-sort [g]
  "Kahn's algorithm for topological ordering (SRFI-234)"
  (setv in-degree (dict (lfor n (g.nodes) #(n (g.in-degree n)))))
  (setv queue (list (filter (fn [n] (= (get in-degree n) 0)) (g.nodes))))
  (setv result [])
  
  (while queue
    (setv node (.pop queue 0))
    (.append result node)
    (for [succ (g.successors node)]
      (setv (get in-degree succ) (- (get in-degree succ) 1))
      (when (= (get in-degree succ) 0)
        (.append queue succ))))
  
  (if (= (len result) (len (g.nodes)))
      result
      (raise (ValueError "Graph has cycle - no topological ordering"))))

(defn dfs [g start [visited None]]
  "Depth-first traversal - returns list"
  (when (is visited None)
    (setv visited (set)))
  (setv result [])
  (defn dfs-helper [node]
    (when (not (in node visited))
      (.add visited node)
      (.append result node)
      (for [succ (g.successors node)]
        (dfs-helper succ))))
  (dfs-helper start)
  result)

(defn bfs [g start]
  "Breadth-first traversal - returns list"
  (setv visited #{start})
  (setv queue [start])
  (setv result [])
  (while queue
    (setv node (.pop queue 0))
    (.append result node)
    (for [succ (g.successors node)]
      (when (not (in succ visited))
        (.add visited succ)
        (.append queue succ))))
  result)

(defn reachable-from [g start]
  "All nodes reachable from start"
  (frozenset (dfs g start)))

(defn strongly-connected-components [g]
  "Kosaraju's algorithm for SCCs"
  (setv visited (set))
  (setv finish-order [])
  
  ;; First DFS to get finish order
  (defn dfs1 [node]
    (when (not (in node visited))
      (.add visited node)
      (for [succ (g.successors node)]
        (dfs1 succ))
      (.append finish-order node)))
  
  (for [n (g.nodes)]
    (dfs1 n))
  
  ;; Second DFS on transpose
  (setv gt (transpose g))
  (setv visited (set))
  (setv sccs [])
  
  (defn dfs2 [node component]
    (when (not (in node visited))
      (.add visited node)
      (.append component node)
      (for [succ (gt.successors node)]
        (dfs2 succ component))))
  
  (for [node (reversed finish-order)]
    (when (not (in node visited))
      (setv component [])
      (dfs2 node component)
      (.append sccs component)))
  
  sccs)

;; ============================================
;; MLX-Graphs Bridge
;; ============================================

(defn ->mlx-graph [g [node-features None]]
  "Convert HyGraph to mlx_graphs.GraphData
   
   Maps our sexp graph to MLX tensor format:
   - edge_index: [2, num_edges] COO format
   - node_features: [num_nodes, dim]
  "
  (when (not HAS_MLX_GRAPHS)
    (raise (ImportError "mlx_graphs not installed")))
  
  ;; Create node index mapping
  (setv node-list (sorted (g.nodes)))
  (setv node-idx (dict (lfor #(i n) (enumerate node-list) #(n i))))
  
  ;; Build edge_index in COO format
  (setv sources [])
  (setv targets [])
  (for [#(u v) (g.edges)]
    (.append sources (get node-idx u))
    (.append targets (get node-idx v)))
  
  (setv edge-index (mx.array [sources targets]))
  
  ;; Default node features if not provided
  (when (is node-features None)
    (setv node-features (mx.eye (len node-list))))
  
  (GraphData edge-index node-features))

(defn <-mlx-graph [gd [node-names None]]
  "Convert mlx_graphs.GraphData back to HyGraph"
  (setv edge-index (.tolist gd.edge_index))
  (setv sources (get edge-index 0))
  (setv targets (get edge-index 1))
  
  (when (is node-names None)
    (setv n-nodes (int (+ 1 (max (max sources) (max targets)))))
    (setv node-names (list (range n-nodes))))
  
  (setv edges (lfor #(s t) (zip sources targets)
                    [(get node-names s) (get node-names t)]))
  (graph #* edges))

;; ============================================
;; GNN Layer Macros (Hy DSL)
;; ============================================

(defmacro defgnn [name layers]
  "Define a GNN model with layer spec.
   
   (defgnn MyGCN
     [(GCNConv 64 32)
      (GCNConv 32 16)
      (Linear 16 num-classes)])
  "
  `(defclass ~name [nn.Module]
     (defn __init__ [self]
       (.__init__ (super))
       ~@(lfor #(i layer) (enumerate layers)
               `(setv ~(hy.models.Symbol f"self.layer{i}") ~layer)))
     
     (defn __call__ [self graph]
       (setv x graph.node_features)
       (setv edge-index graph.edge_index)
       ~@(lfor #(i _) (enumerate layers)
               `(setv x (~(hy.models.Symbol f"self.layer{i}") x edge-index)))
       x)))

;; ============================================
;; Narya Bridge Types (Type-theoretic)
;; ============================================

(defclass NaryaGraphType []
  "Type-theoretic graph specification for Narya verification.
   
   Represents graph invariants that can be checked:
   - Acyclicity (DAG)
   - Connectivity
   - Bipartiteness
   - Planarity (structure only)
  "
  
  (defn __init__ [self g]
    (setv self.graph g)
    (setv self._acyclic None)
    (setv self._connected None)
    (setv self._bipartite None))
  
  (defn #^ bool acyclic? [self]
    "Is the graph a DAG? (checkable via topological sort)"
    (when (is self._acyclic None)
      (try
        (topological-sort self.graph)
        (setv self._acyclic True)
        (except [ValueError]
          (setv self._acyclic False))))
    self._acyclic)
  
  (defn #^ bool connected? [self]
    "Is the graph (weakly) connected?"
    (when (is self._connected None)
      (if (= (len (self.graph.nodes)) 0)
          (setv self._connected True)
          (do
            (setv start (next (iter (self.graph.nodes))))
            ;; Check reachability in undirected view
            (setv forward (lfor #(u v) (self.graph.edges) [u v]))
            (setv reverse (lfor #(u v) (self.graph.edges) [v u]))
            (setv undirected (graph #* (+ forward reverse)))
            (setv reached (reachable-from undirected start))
            (setv self._connected (= reached (self.graph.nodes))))))
    self._connected)
  
  (defn #^ bool bipartite? [self]
    "Is the graph 2-colorable?"
    (when (is self._bipartite None)
      (setv colors {})
      (setv is-bipartite True)
      (for [start (self.graph.nodes)]
        (when (and is-bipartite (not (in start colors)))
          (setv (get colors start) 0)
          (setv queue [start])
          (while (and queue is-bipartite)
            (setv node (.pop queue 0))
            (setv color (get colors node))
            (for [neighbor (self.graph.neighbors node)]
              (cond
                (not (in neighbor colors))
                (do
                  (setv (get colors neighbor) (- 1 color))
                  (.append queue neighbor))
                
                (!= (get colors neighbor) (- 1 color))
                (setv is-bipartite False))))))
      (setv self._bipartite is-bipartite))
    self._bipartite)
  
  (defn to-narya-spec [self]
    "Generate Narya type specification"
    (setv n (len (self.graph.nodes)))
    (setv acyc (if (self.acyclic?) "⊤" "⊥"))
    (setv conn (if (self.connected?) "⊤" "⊥"))
    (setv bip (if (self.bipartite?) "⊤" "⊥"))
    (.join "\n" [f"Graph : Type where"
                 f"  nodes : Fin {n}"
                 f"  edges : nodes -> nodes -> Bool"
                 f"  acyclic : {acyc}"
                 f"  connected : {conn}"
                 f"  bipartite : {bip}"])))

;; ============================================
;; Convenience Functions
;; ============================================

(defn complete-graph [n]
  "K_n complete graph"
  (setv edges (lfor i (range n) j (range n) :if (!= i j) [i j]))
  (graph #* edges))

(defn path-graph [n]
  "P_n path graph"
  (add-path (graph) #* (range n)))

(defn cycle-graph [n]
  "C_n cycle graph"
  (add-cycle (graph) #* (range n)))

(defn star-graph [n]
  "S_n star graph (center=0)"
  (setv edges (lfor i (range 1 (+ n 1)) [0 i]))
  (graph #* edges))

(defn grid-graph [rows cols]
  "Grid graph"
  (setv edges [])
  (for [r (range rows)]
    (for [c (range cols)]
      (setv node #(r c))
      (when (< (+ r 1) rows)
        (.append edges [node #((+ r 1) c)]))
      (when (< (+ c 1) cols)
        (.append edges [node #(r (+ c 1))]))))
  (graph #* edges))

;; ============================================
;; Pretty Printing
;; ============================================

(defn graph->sexp [g]
  "Convert graph back to s-expression"
  (setv edge-strs (lfor #(u v) (sorted (g.edges)) f"[{u} {v}]"))
  (+ "(graph " (.join " " edge-strs) ")"))

(defn graph->dot [g [name "G"]]
  "Export to Graphviz DOT format"
  (setv lines [f"digraph {name} {{"])
  (for [#(u v) (g.edges)]
    (.append lines f"  \"{u}\" -> \"{v}\";"))
  (.append lines "}")
  (.join "\n" lines))
