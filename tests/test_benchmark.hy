#!/usr/bin/env hy
;; HyMLX Benchmarks in Hy
;; Demonstrates Hy's utility for ML benchmarking

(import time)
(import mlx.core :as mx)
(import mlx.nn :as nn)

;; ============================================
;; Benchmark Utilities
;; ============================================

(defn benchmark [f args [warmup 3] [runs 10]]
  "Run benchmark with warmup, return timing dict"
  ;; Warmup
  (for [_ (range warmup)]
    (setv result (f #* args))
    (when (isinstance result mx.array)
      (mx.eval result)))
  
  ;; Timed runs
  (setv times [])
  (for [_ (range runs)]
    (setv start (time.perf_counter))
    (setv result (f #* args))
    (when (isinstance result mx.array)
      (mx.eval result))
    (.append times (- (time.perf_counter) start)))
  
  {"mean_ms" (* (/ (sum times) (len times)) 1000)
   "min_ms" (* (min times) 1000)
   "max_ms" (* (max times) 1000)})

(defmacro time-it [expr]
  "Time a single expression"
  `(do
     (setv _start (time.perf_counter))
     (setv _result ~expr)
     (when (isinstance _result mx.array)
       (mx.eval _result))
     (setv _elapsed (* (- (time.perf_counter) _start) 1000))
     (print (.format "{:.3f}ms" _elapsed))
     _result))

;; ============================================
;; Multi-Head Attention
;; ============================================

(defclass MultiHeadAttention [nn.Module]
  "Multi-head attention for benchmarking"
  
  (defn __init__ [self dim n-heads]
    (.__init__ (super))
    (setv self.dim dim
          self.n-heads n-heads
          self.head-dim (// dim n-heads)
          self.scale (** self.head-dim -0.5))
    (setv self.qkv (nn.Linear dim (* 3 dim)))
    (setv self.proj (nn.Linear dim dim)))
  
  (defn __call__ [self x]
    (setv #(B N _) x.shape)
    (setv qkv (self.qkv x))
    (setv qkv (mx.reshape qkv [B N 3 self.n-heads self.head-dim]))
    (setv qkv (mx.transpose qkv [2 0 3 1 4]))
    (setv #(q k v) [(get qkv 0) (get qkv 1) (get qkv 2)])
    (setv attn (* (mx.matmul q (mx.transpose k [0 1 3 2])) self.scale))
    (setv attn (mx.softmax attn :axis -1))
    (setv out (mx.matmul attn v))
    (setv out (mx.transpose out [0 2 1 3]))
    (setv out (mx.reshape out [B N self.dim]))
    (self.proj out)))

;; ============================================
;; SplitMix64 (Pure Hy)
;; ============================================

(setv GOLDEN 0x9E3779B97F4A7C15)
(setv MIX1 0xBF58476D1CE4E5B9)
(setv MIX2 0x94D049BB133111EB)
(setv MASK64 (- (<< 1 64) 1))

(defn splitmix64 [z]
  "Pure functional SplitMix64"
  (setv z (& (+ z GOLDEN) MASK64))
  (setv z (& (* (^ z (>> z 30)) MIX1) MASK64))
  (setv z (& (* (^ z (>> z 27)) MIX2) MASK64))
  (& (^ z (>> z 31)) MASK64))

(defn derive-chain [seed n]
  "Generate chain of n seeds - preallocated"
  (setv seeds (* [0] n))
  (setv current seed)
  (for [i (range n)]
    (setv (get seeds i) current)
    (setv current (splitmix64 current)))
  seeds)

;; ============================================
;; Transforms
;; ============================================

(defn jit [f]
  "JIT compile"
  (mx.compile f))

(defn grad [f]
  "Gradient"
  (mx.grad f))

(defn vmap [f [in-axes 0] [out-axes 0]]
  "Vectorized map using MLX native vmap"
  (mx.vmap f :in_axes in-axes :out_axes out-axes))

(defn scan [f init xs]
  "Functional scan - collect results then stack"
  (setv n (get xs.shape 0))
  (setv carry init)
  (setv ys (* [None] n))
  ;; Compile the step function  
  (setv compiled-f (mx.compile f))
  (for [i (range n)]
    (setv #(carry y) (compiled-f carry (get xs i)))
    (setv (get ys i) y))
  (mx.eval carry)
  #(carry (mx.stack ys)))

;; ============================================
;; Test Functions
;; ============================================

(defn test-splitmix []
  "Benchmark SplitMix64"
  (print "\n=== SplitMix64 ===")
  
  (for [n [1000 10000 100000]]
    (setv result (benchmark derive-chain [1069 n]))
    (setv throughput (/ n (/ (get result "mean_ms") 1000)))
    (print (.format "  {:,}:  {:.3f}ms ({:,.0f}/sec)" 
                    n (get result "mean_ms") throughput))))

(defn test-attention []
  "Benchmark attention"
  (print "\n=== Attention ===")
  
  (setv configs [[64 4 32 "small"]
                 [256 8 128 "medium"]
                 [768 12 512 "large"]])
  
  (for [#(dim heads seq name) configs]
    (setv attn (MultiHeadAttention dim heads))
    (setv x (mx.random.normal [1 seq dim]))
    (setv result (benchmark attn [x]))
    (print (.format "  {} ({}d, {}h, {}seq): {:.3f}ms"
                    name dim heads seq (get result "mean_ms")))))

(defn test-jit []
  "Benchmark JIT"
  (print "\n=== JIT ===")
  
  (setv a (mx.random.normal [256 256]))
  (setv b (mx.random.normal [256 256]))
  
  (defn matmul [a b] (@ a b))
  (setv jit-matmul (jit matmul))
  
  (setv eager (benchmark matmul [a b]))
  (setv compiled (benchmark jit-matmul [a b]))
  
  (print (.format "  Eager:   {:.3f}ms" (get eager "mean_ms")))
  (print (.format "  JIT:     {:.3f}ms" (get compiled "mean_ms")))
  (print (.format "  Speedup: {:.2f}x" 
                  (/ (get eager "mean_ms") (get compiled "mean_ms")))))

(defn test-grad []
  "Benchmark gradients"
  (print "\n=== Grad ===")
  
  (defn loss [x] (mx.sum (** x 2)))
  (setv grad-loss (grad loss))
  
  (setv x (mx.random.normal [1000]))
  (setv result (benchmark grad-loss [x]))
  (print (.format "  sum(x²) [1000]: {:.3f}ms" (get result "mean_ms"))))

(defn test-vmap []
  "Benchmark vmap"
  (print "\n=== vmap ===")
  
  (defn dot [x] (mx.sum (* x x)))
  (setv vdot (vmap dot))
  
  (setv x (mx.random.normal [100 64]))
  
  (setv vmapped (benchmark vdot [x]))
  (print (.format "  vmap dot [100,64]: {:.3f}ms" (get vmapped "mean_ms")))
  
  ;; Native comparison
  (defn native-dot [x] (mx.sum (* x x) :axis 1))
  (setv native (benchmark native-dot [x]))
  (print (.format "  native:            {:.3f}ms" (get native "mean_ms"))))

(defn test-scan []
  "Benchmark scan"
  (print "\n=== scan ===")
  
  (defn step [carry x]
    (setv new (+ carry x))
    #(new new))
  
  (setv xs (mx.random.normal [1000]))
  
  (defn run-scan [] (scan step (mx.array 0.0) xs))
  (setv scanned (benchmark run-scan []))
  (print (.format "  scan cumsum [1000]: {:.3f}ms" (get scanned "mean_ms")))
  
  ;; Native comparison
  (defn run-native [] (mx.cumsum xs))
  (setv native (benchmark run-native []))
  (print (.format "  native cumsum:      {:.3f}ms" (get native "mean_ms"))))

;; ============================================
;; Main
;; ============================================

(defn run-all-benchmarks []
  "Run all benchmarks"
  (print "🎨 HyMLX Benchmarks (Hy)")
  (print "=" 40)
  
  (test-splitmix)
  (test-attention)
  (test-jit)
  (test-grad)
  (test-vmap)
  (test-scan)
  
  (print "\n✅ Benchmarks complete!"))

(when (= __name__ "__main__")
  (run-all-benchmarks))
