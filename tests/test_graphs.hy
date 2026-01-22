#!/usr/bin/env hy
;; Test suite for HyMLX Graphs

(import sys)
(sys.path.insert 0 "src")

(import hymlx.graphs [graph digraph add-nodes add-edges add-path add-cycle
                       subgraph mapped-by transpose
                       topological-sort dfs bfs reachable-from
                       strongly-connected-components
                       complete-graph path-graph cycle-graph star-graph grid-graph
                       graph->sexp graph->dot
                       NaryaGraphType])

;; ============================================
;; Basic Construction Tests
;; ============================================

(defn test-empty-graph []
  (print "Testing empty graph...")
  (setv g (graph))
  (assert (= (len (g.nodes)) 0))
  (assert (= (len (g.edges)) 0))
  (print "  ✓ Empty graph"))

(defn test-edge-construction []
  (print "Testing edge construction (Loom-style)...")
  ;; Clojure: (graph [1 2] [2 3] [3 4])
  (setv g (graph [1 2] [2 3] [3 4]))
  (assert (= (g.nodes) (frozenset [1 2 3 4])))
  (assert (= (len (g.edges)) 3))
  (assert (g.has-edge? 1 2))
  (assert (g.has-edge? 2 3))
  (assert (not (g.has-edge? 1 3)))
  (print "  ✓ Edge construction"))

(defn test-adjacency-map []
  (print "Testing adjacency map construction...")
  ;; Clojure: (graph {1 [2 3] 2 [4]})
  (setv g (graph {1 [2 3] 2 [4]}))
  (assert (= (g.successors 1) (frozenset [2 3])))
  (assert (= (g.successors 2) (frozenset [4])))
  (print "  ✓ Adjacency map"))

(defn test-weighted-edges []
  (print "Testing weighted edges...")
  ;; Clojure: (weighted-graph [:a :b 10] [:b :c 20])
  (setv g (graph ["a" "b" 10] ["b" "c" 20]))
  (assert (= (g.edge-attr "a" "b" "weight") 10))
  (assert (= (g.edge-attr "b" "c" "weight") 20))
  (print "  ✓ Weighted edges"))

;; ============================================
;; Graph Operations Tests
;; ============================================

(defn test-add-path []
  (print "Testing add-path...")
  (setv g (add-path (graph) 1 2 3 4 5))
  (assert (g.has-edge? 1 2))
  (assert (g.has-edge? 2 3))
  (assert (g.has-edge? 4 5))
  (assert (not (g.has-edge? 1 5)))
  (print "  ✓ add-path"))

(defn test-add-cycle []
  (print "Testing add-cycle...")
  (setv g (add-cycle (graph) 1 2 3))
  (assert (g.has-edge? 1 2))
  (assert (g.has-edge? 2 3))
  (assert (g.has-edge? 3 1))  ; closes the cycle
  (print "  ✓ add-cycle"))

(defn test-subgraph []
  (print "Testing subgraph (OCaml nodes_filtered_by)...")
  (setv g (graph [1 2] [2 3] [3 4] [4 5]))
  (setv sg (subgraph g (fn [n] (< n 4))))
  (assert (= (sg.nodes) (frozenset [1 2 3])))
  (assert (sg.has-edge? 1 2))
  (assert (sg.has-edge? 2 3))
  (assert (not (sg.has-edge? 3 4)))
  (print "  ✓ subgraph"))

(defn test-mapped-by []
  (print "Testing mapped-by (OCaml functor style)...")
  (setv g (graph [1 2] [2 3]))
  (setv mg (mapped-by g (fn [n] (* n 10))))
  (assert (mg.has-edge? 10 20))
  (assert (mg.has-edge? 20 30))
  (print "  ✓ mapped-by"))

(defn test-transpose []
  (print "Testing transpose...")
  (setv g (graph [1 2] [2 3]))
  (setv gt (transpose g))
  (assert (gt.has-edge? 2 1))
  (assert (gt.has-edge? 3 2))
  (assert (not (gt.has-edge? 1 2)))
  (print "  ✓ transpose"))

;; ============================================
;; Algorithm Tests (SRFI-234 style)
;; ============================================

(defn test-topological-sort []
  (print "Testing topological-sort (SRFI-234)...")
  (setv g (graph [1 2] [1 3] [2 4] [3 4]))
  (setv order (topological-sort g))
  (assert (< (.index order 1) (.index order 2)))
  (assert (< (.index order 1) (.index order 3)))
  (assert (< (.index order 2) (.index order 4)))
  (assert (< (.index order 3) (.index order 4)))
  (print f"  Order: {order}")
  (print "  ✓ topological-sort"))

(defn test-topological-sort-cycle []
  (print "Testing topological-sort with cycle...")
  (setv g (graph [1 2] [2 3] [3 1]))
  (try
    (topological-sort g)
    (assert False "Should have raised ValueError")
    (except [e ValueError]
      (print f"  Expected error: {e}")
      (print "  ✓ Cycle detection"))))

(defn test-dfs []
  (print "Testing DFS traversal...")
  (setv g (graph [1 2] [1 3] [2 4] [3 5]))
  (setv visited (dfs g 1))
  (assert (in 1 visited))
  (assert (in 2 visited))
  (assert (in 3 visited))
  (assert (= (get visited 0) 1))  ; starts at 1
  (print f"  DFS order: {visited}")
  (print "  ✓ DFS"))

(defn test-bfs []
  (print "Testing BFS traversal...")
  (setv g (graph [1 2] [1 3] [2 4] [3 5]))
  (setv visited (bfs g 1))
  (assert (= (get visited 0) 1))
  ;; BFS visits level by level
  (assert (in 2 (cut visited 1 3)))  ; 2 and 3 at level 1
  (assert (in 3 (cut visited 1 3)))
  (print f"  BFS order: {visited}")
  (print "  ✓ BFS"))

(defn test-reachable-from []
  (print "Testing reachable-from...")
  (setv g (graph [1 2] [2 3] [4 5]))  ; disconnected
  (setv r1 (reachable-from g 1))
  (assert (= r1 (frozenset [1 2 3])))
  (setv r4 (reachable-from g 4))
  (assert (= r4 (frozenset [4 5])))
  (print "  ✓ reachable-from"))

(defn test-scc []
  (print "Testing strongly-connected-components...")
  (setv g (graph [1 2] [2 3] [3 1]   ; SCC {1,2,3}
                 [3 4] [4 5] [5 4])) ; SCC {4,5}
  (setv sccs (strongly-connected-components g))
  (print f"  SCCs: {sccs}")
  (assert (= (len sccs) 2))
  (print "  ✓ SCC"))

;; ============================================
;; Standard Graph Tests
;; ============================================

(defn test-complete-graph []
  (print "Testing complete-graph K_5...")
  (setv k5 (complete-graph 5))
  (assert (= (len (k5.nodes)) 5))
  (assert (= (len (k5.edges)) 20))  ; n*(n-1) directed edges
  (print "  ✓ complete-graph"))

(defn test-path-graph []
  (print "Testing path-graph P_5...")
  (setv p5 (path-graph 5))
  (assert (= (len (p5.edges)) 4))
  (assert (p5.has-edge? 0 1))
  (assert (p5.has-edge? 3 4))
  (print "  ✓ path-graph"))

(defn test-cycle-graph []
  (print "Testing cycle-graph C_5...")
  (setv c5 (cycle-graph 5))
  (assert (= (len (c5.edges)) 5))
  (assert (c5.has-edge? 4 0))  ; closes the cycle
  (print "  ✓ cycle-graph"))

(defn test-grid-graph []
  (print "Testing grid-graph 3x3...")
  (setv grid (grid-graph 3 3))
  (assert (= (len (grid.nodes)) 9))
  ;; 2*3*2 = 12 edges (3 rows * 2 horizontal + 3 cols * 2 vertical)
  (assert (= (len (grid.edges)) 12))
  (print "  ✓ grid-graph"))

;; ============================================
;; Narya Type Tests
;; ============================================

(defn test-narya-acyclic []
  (print "Testing Narya acyclic? predicate...")
  (setv dag (graph [1 2] [2 3] [1 3]))
  (setv cyclic (graph [1 2] [2 3] [3 1]))
  
  (setv dag-type (NaryaGraphType dag))
  (setv cyclic-type (NaryaGraphType cyclic))
  
  (assert (dag-type.acyclic?))
  (assert (not (cyclic-type.acyclic?)))
  (print "  ✓ acyclic?"))

(defn test-narya-connected []
  (print "Testing Narya connected? predicate...")
  (setv conn (graph [1 2] [2 3]))
  (setv disconn (graph [1 2] [3 4]))
  
  (setv conn-type (NaryaGraphType conn))
  (setv disconn-type (NaryaGraphType disconn))
  
  (assert (conn-type.connected?))
  (assert (not (disconn-type.connected?)))
  (print "  ✓ connected?"))

(defn test-narya-bipartite []
  (print "Testing Narya bipartite? predicate...")
  (setv bip (graph [1 2] [2 3] [3 4]))  ; path is bipartite
  (setv non-bip (graph [1 2] [2 3] [3 1]))  ; triangle is not
  
  (setv bip-type (NaryaGraphType bip))
  (setv non-bip-type (NaryaGraphType non-bip))
  
  (assert (bip-type.bipartite?))
  (assert (not (non-bip-type.bipartite?)))
  (print "  ✓ bipartite?"))

(defn test-narya-spec []
  (print "Testing Narya spec generation...")
  (setv g (graph [1 2] [2 3]))
  (setv gt (NaryaGraphType g))
  (setv spec (gt.to-narya-spec))
  (print f"  Spec:\n{spec}")
  (assert (in "acyclic : ⊤" spec))
  (assert (in "connected : ⊤" spec))
  (print "  ✓ Narya spec"))

;; ============================================
;; Serialization Tests
;; ============================================

(defn test-graph-to-sexp []
  (print "Testing graph->sexp...")
  (setv g (graph [1 2] [2 3]))
  (setv sexp (graph->sexp g))
  (print f"  {sexp}")
  (assert (in "[1 2]" sexp))
  (assert (in "[2 3]" sexp))
  (print "  ✓ graph->sexp"))

(defn test-graph-to-dot []
  (print "Testing graph->dot...")
  (setv g (graph [1 2] [2 3]))
  (setv dot (graph->dot g "TestGraph"))
  (print f"  {dot}")
  (assert (in "digraph TestGraph" dot))
  (assert (in "\"1\" -> \"2\"" dot))
  (print "  ✓ graph->dot"))

;; ============================================
;; Main
;; ============================================

(defn run-all-tests []
  (print "🎨 HyMLX Graphs Test Suite")
  (print "=" 50)
  (print "\n== Basic Construction ==")
  (test-empty-graph)
  (test-edge-construction)
  (test-adjacency-map)
  (test-weighted-edges)
  
  (print "\n== Graph Operations ==")
  (test-add-path)
  (test-add-cycle)
  (test-subgraph)
  (test-mapped-by)
  (test-transpose)
  
  (print "\n== Algorithms (SRFI-234) ==")
  (test-topological-sort)
  (test-topological-sort-cycle)
  (test-dfs)
  (test-bfs)
  (test-reachable-from)
  (test-scc)
  
  (print "\n== Standard Graphs ==")
  (test-complete-graph)
  (test-path-graph)
  (test-cycle-graph)
  (test-grid-graph)
  
  (print "\n== Narya Type Verification ==")
  (test-narya-acyclic)
  (test-narya-connected)
  (test-narya-bipartite)
  (test-narya-spec)
  
  (print "\n== Serialization ==")
  (test-graph-to-sexp)
  (test-graph-to-dot)
  
  (print "\n" "=" 50)
  (print "✅ All tests passed!"))

(when (= __name__ "__main__")
  (run-all-tests))
