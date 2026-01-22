#!/usr/bin/env hy
;; HyMLX Architectures - Lisp-native neural network definitions
;; Demonstrates instrumental utility of Hy for ML

(require hyrule [-> ->> unless])

(import mlx.core :as mx)
(import mlx.nn :as nn)

;; ============================================
;; 1. MACRO: Declarative Layer Composition
;; ============================================

;; In Python: layers = [Linear(784, 256), ReLU(), Linear(256, 10)]
;; In Hy: threading macro composes naturally

(defmacro -> [x #* forms]
  "Thread value through functions (like Clojure)"
  (if forms
    `(-> (~(get forms 0) ~x) ~@(cut forms 1 None))
    x))

(defmacro ->> [x #* forms]
  "Thread value as last arg"
  (if forms
    (do
      (setv form (get forms 0))
      (if (isinstance form hy.models.Expression)
        `(->> (~@form ~x) ~@(cut forms 1 None))
        `(->> (~form ~x) ~@(cut forms 1 None))))
    x))

;; ============================================
;; 2. MACRO: Automatic Residual Connections
;; ============================================

(defmacro residual [x #* layers]
  "Wrap layers in residual connection: x + f(x)"
  `(+ ~x (-> ~x ~@layers)))

(defmacro prenorm-residual [x norm-layer #* layers]
  "Pre-norm residual: x + f(norm(x))"
  `(+ ~x (-> (~norm-layer ~x) ~@layers)))

;; ============================================
;; 3. S-EXPRESSION: Network as Data
;; ============================================

(defn parse-layer [spec]
  "Convert s-expression to layer.
   
   Accepts both Hy symbols and Python strings as layer type:
     ['linear 64 128] or [\"linear\" 64 128]
  "
  (setv layer-type (str (get spec 0)))
  (cond
    (= layer-type "linear")
    (nn.Linear (get spec 1) (get spec 2))
    
    (= layer-type "conv2d")
    (nn.Conv2d (get spec 1) (get spec 2) (get spec 3))
    
    (= layer-type "relu")
    (nn.ReLU)
    
    (= layer-type "gelu")
    (nn.GELU)
    
    (= layer-type "dropout")
    (nn.Dropout (get spec 1))
    
    (= layer-type "layernorm")
    (nn.LayerNorm (get spec 1))
    
    (= layer-type "embed")
    (nn.Embedding (get spec 1) (get spec 2))
    
    True
    (raise (ValueError (.format "Unknown layer spec: {}" spec)))))

(defn build-sequential [specs]
  "Build Sequential from list of s-expression specs"
  (setv layers (lfor spec specs (parse-layer spec)))
  (nn.Sequential #* layers))

;; Example:
;; (build-sequential '[[linear 784 256] [relu] [linear 256 10]])

;; ============================================
;; 4. MACRO: Attention with Named Components
;; ============================================

(defclass MultiHeadAttention [nn.Module]
  "Multi-head attention with Hy-native initialization"
  
  (defn __init__ [self dim n-heads [dropout 0.0]]
    (.__init__ (super))
    (assert (= 0 (% dim n-heads)) "dim must be divisible by n_heads")
    (setv self.dim dim
          self.n-heads n-heads
          self.head-dim (// dim n-heads)
          self.scale (** self.head-dim -0.5))
    (setv self.qkv (nn.Linear dim (* 3 dim)))
    (setv self.proj (nn.Linear dim dim))
    (setv self.dropout (nn.Dropout dropout)))
  
  (defn __call__ [self x [mask None]]
    (setv #(B N _) x.shape)
    ;; QKV projection and reshape
    (setv qkv (self.qkv x))
    (setv qkv (mx.reshape qkv [B N 3 self.n-heads self.head-dim]))
    (setv qkv (mx.transpose qkv [2 0 3 1 4]))  ; [3, B, H, N, D]
    (setv #(q k v) qkv)
    ;; Attention
    (setv attn (* (mx.matmul q (mx.transpose k [0 1 3 2])) self.scale))
    (when (is-not mask None)
      (setv attn (+ attn mask)))
    (setv attn (mx.softmax attn :axis -1))
    (setv attn (self.dropout attn))
    ;; Combine heads
    (setv out (mx.matmul attn v))
    (setv out (mx.transpose out [0 2 1 3]))
    (setv out (mx.reshape out [B N self.dim]))
    (self.proj out)))

;; ============================================
;; 5. MACRO: Transformer Block as Template
;; ============================================

(defmacro defblock [name dim n-heads mlp-ratio [dropout 0.0]]
  "Define a transformer block with given hyperparams"
  `(defclass ~name [nn.Module]
     (defn __init__ [self]
       (.__init__ (super))
       (setv self.norm1 (nn.LayerNorm ~dim))
       (setv self.attn (MultiHeadAttention ~dim ~n-heads :dropout ~dropout))
       (setv self.norm2 (nn.LayerNorm ~dim))
       (setv self.mlp (nn.Sequential
                        (nn.Linear ~dim ~(* dim mlp-ratio))
                        (nn.GELU)
                        (nn.Linear ~(* dim mlp-ratio) ~dim)
                        (nn.Dropout ~dropout))))
     
     (defn __call__ [self x [mask None]]
       ;; Pre-norm transformer block
       (setv x (+ x (self.attn (self.norm1 x) :mask mask)))
       (setv x (+ x (self.mlp (self.norm2 x))))
       x)))

;; Usage: (defblock GPTBlock 768 12 4 :dropout 0.1)

;; ============================================
;; 6. QUASIQUOTE: Dynamic Architecture Gen
;; ============================================

(defn make-mlp-spec [layer-sizes [activation 'relu]]
  "Generate MLP spec programmatically"
  (setv specs [])
  (for [[in-dim out-dim] (zip layer-sizes (cut layer-sizes 1 None))]
    (.append specs ['linear in-dim out-dim])
    (.append specs [activation]))
  ;; Remove last activation
  (.pop specs)
  specs)

;; (make-mlp-spec [784 256 128 10])
;; => [[linear 784 256] [relu] [linear 256 128] [relu] [linear 128 10]]

;; ============================================
;; 7. RECURSIVE: Tree-structured Networks
;; ============================================

(defn make-unet-block [in-ch out-ch depth]
  "Recursively generate U-Net-like structure"
  (if (= depth 0)
    ;; Base: bottleneck
    [['conv2d in-ch out-ch 3] ['gelu]]
    ;; Recursive: down -> recurse -> up
    (+ [['conv2d in-ch out-ch 3] ['gelu]]  ; down
       (make-unet-block out-ch (* out-ch 2) (- depth 1))  ; recurse
       [["conv2d" (* out-ch 2) out-ch 3] ["gelu"]])))  ; up

;; ============================================
;; 8. PATTERN MATCHING: Layer Surgery
;; ============================================

(defn replace-activations [specs old-act new-act]
  "Replace all activations of one type with another.
   
   Handles both Hy symbols and Python strings.
  "
  (setv old-str (str old-act))
  (lfor spec specs
    (if (and (= (len spec) 1) (= (str (get spec 0)) old-str))
        [new-act]
        spec)))

;; (replace-activations mlp-spec 'relu 'gelu)

;; ============================================
;; 9. CLOSURE: Weight-tied Layers
;; ============================================

(defn make-weight-tied-decoder [encoder-embed]
  "Create decoder that shares weights with encoder embedding"
  (defclass TiedDecoder [nn.Module]
    (defn __init__ [self]
      (.__init__ (super))
      (setv self.embed encoder-embed))
    
    (defn __call__ [self x]
      ;; Transpose of embedding matrix
      (mx.matmul x (mx.transpose self.embed.weight [1 0]))))
  (TiedDecoder))

;; ============================================
;; 10. COMPOSE: Functional Pipeline
;; ============================================

(defn compose [#* fns]
  "Right-to-left function composition"
  (fn [x]
    (setv result x)
    (for [f (reversed fns)]
      (setv result (f result)))
    result))

(defn pipe [#* fns]
  "Left-to-right function composition"
  (fn [x]
    (setv result x)
    (for [f fns]
      (setv result (f result)))
    result))

;; (setv preprocess (pipe normalize augment tokenize))

;; ============================================
;; EXPORTS
;; ============================================

(setv __all__ [
  "parse-layer" "build-sequential" 
  "MultiHeadAttention"
  "make-mlp-spec" "make-unet-block"
  "replace-activations" "make-weight-tied-decoder"
  "compose" "pipe"
])
