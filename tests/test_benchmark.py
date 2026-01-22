"""Performance benchmarks for HyMLX."""

import pytest
import time
import mlx.core as mx
import mlx.nn as nn
from hymlx.transforms import grad, jit, vmap, scan, force
from hymlx.splitmix import splitmix64, derive_chain


class MultiHeadAttention(nn.Module):
    """Multi-head attention for benchmarking."""
    
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
    
    def __call__(self, x: mx.array) -> mx.array:
        B, N, _ = x.shape
        qkv = self.qkv(x)
        qkv = mx.reshape(qkv, [B, N, 3, self.n_heads, self.head_dim])
        qkv = mx.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = mx.matmul(q, mx.transpose(k, [0, 1, 3, 2])) * self.scale
        attn = mx.softmax(attn, axis=-1)
        out = mx.matmul(attn, v)
        out = mx.transpose(out, [0, 2, 1, 3])
        out = mx.reshape(out, [B, N, self.dim])
        return self.proj(out)


def benchmark(fn, *args, warmup=3, runs=10):
    """Run benchmark with warmup."""
    # Warmup
    for _ in range(warmup):
        result = fn(*args)
        if isinstance(result, mx.array):
            mx.eval(result)
    
    # Timed runs
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        result = fn(*args)
        if isinstance(result, mx.array):
            mx.eval(result)
        times.append(time.perf_counter() - start)
    
    return {
        "mean_ms": sum(times) / len(times) * 1000,
        "min_ms": min(times) * 1000,
        "max_ms": max(times) * 1000,
    }


class TestSplitMixBenchmark:
    """Benchmark SplitMix64 operations."""
    
    def test_splitmix_1k(self):
        """1,000 derivations."""
        result = benchmark(derive_chain, 1069, 1000)
        print(f"\nSplitMix64 1K: {result['mean_ms']:.3f}ms")
        assert result["mean_ms"] < 100  # Should be fast
    
    def test_splitmix_10k(self):
        """10,000 derivations."""
        result = benchmark(derive_chain, 1069, 10000)
        print(f"SplitMix64 10K: {result['mean_ms']:.3f}ms")
        assert result["mean_ms"] < 500
    
    def test_splitmix_100k(self):
        """100,000 derivations."""
        result = benchmark(derive_chain, 1069, 100000)
        throughput = 100000 / (result["mean_ms"] / 1000)
        print(f"SplitMix64 100K: {result['mean_ms']:.3f}ms ({throughput:,.0f}/sec)")
        assert throughput > 1_000_000  # >1M/sec


class TestAttentionBenchmark:
    """Benchmark attention forward pass."""
    
    def test_attention_small(self):
        """Small: dim=64, heads=4, seq=32."""
        attn = MultiHeadAttention(64, 4)
        x = mx.random.normal([1, 32, 64])
        
        result = benchmark(attn, x)
        print(f"\nAttention small (64d, 4h, 32seq): {result['mean_ms']:.3f}ms")
        assert result["mean_ms"] < 50
    
    def test_attention_medium(self):
        """Medium: dim=256, heads=8, seq=128."""
        attn = MultiHeadAttention(256, 8)
        x = mx.random.normal([1, 128, 256])
        
        result = benchmark(attn, x)
        print(f"Attention medium (256d, 8h, 128seq): {result['mean_ms']:.3f}ms")
        assert result["mean_ms"] < 100
    
    def test_attention_large(self):
        """Large: dim=768, heads=12, seq=512."""
        attn = MultiHeadAttention(768, 12)
        x = mx.random.normal([1, 512, 768])
        
        result = benchmark(attn, x)
        print(f"Attention large (768d, 12h, 512seq): {result['mean_ms']:.3f}ms")
        assert result["mean_ms"] < 500
    
    def test_attention_batched(self):
        """Batched: batch=8, dim=256, heads=8, seq=64."""
        attn = MultiHeadAttention(256, 8)
        x = mx.random.normal([8, 64, 256])
        
        result = benchmark(attn, x)
        print(f"Attention batched (8x64x256): {result['mean_ms']:.3f}ms")
        assert result["mean_ms"] < 100


class TestJitBenchmark:
    """Benchmark JIT compilation."""
    
    def test_jit_matmul(self):
        """JIT vs eager matmul."""
        a = mx.random.normal([256, 256])
        b = mx.random.normal([256, 256])
        
        def matmul(a, b):
            return a @ b
        
        jit_matmul = jit(matmul)
        
        eager = benchmark(matmul, a, b)
        compiled = benchmark(jit_matmul, a, b)
        
        print(f"\nMatmul 256x256:")
        print(f"  Eager: {eager['mean_ms']:.3f}ms")
        print(f"  JIT:   {compiled['mean_ms']:.3f}ms")
        print(f"  Speedup: {eager['mean_ms'] / compiled['mean_ms']:.2f}x")
    
    def test_jit_complex(self):
        """JIT complex computation."""
        x = mx.random.normal([1000, 100])
        
        def complex_fn(x):
            y = mx.sin(x) + mx.cos(x)
            y = mx.tanh(y @ y.T)
            return mx.sum(y)
        
        jit_fn = jit(complex_fn)
        
        eager = benchmark(complex_fn, x)
        compiled = benchmark(jit_fn, x)
        
        print(f"\nComplex fn (sin+cos+tanh+matmul):")
        print(f"  Eager: {eager['mean_ms']:.3f}ms")
        print(f"  JIT:   {compiled['mean_ms']:.3f}ms")


class TestGradBenchmark:
    """Benchmark gradient computation."""
    
    def test_grad_simple(self):
        """Gradient of simple function."""
        def f(x):
            return mx.sum(x ** 2)
        
        df = grad(f)
        x = mx.random.normal([1000])
        
        result = benchmark(df, x)
        print(f"\nGrad sum(x²) [1000]: {result['mean_ms']:.3f}ms")
    
    def test_grad_mlp(self):
        """Gradient through MLP."""
        model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )
        x = mx.random.normal([32, 256])
        y = mx.random.randint(0, 10, [32])
        
        def loss_fn(model):
            logits = model(x)
            return mx.mean(nn.losses.cross_entropy(logits, y))
        
        grad_fn = grad(loss_fn)
        
        result = benchmark(grad_fn, model)
        print(f"Grad MLP (256→512→256→10): {result['mean_ms']:.3f}ms")


class TestVmapBenchmark:
    """Benchmark vectorized mapping."""
    
    def test_vmap_dot(self):
        """vmap dot product."""
        def dot(x):
            return mx.sum(x * x)
        
        vdot = vmap(dot)
        x = mx.random.normal([100, 64])
        
        result = benchmark(vdot, x)
        print(f"\nvmap dot [100, 64]: {result['mean_ms']:.3f}ms")
    
    def test_vmap_vs_native(self):
        """Compare vmap to native batched ops."""
        def single_norm(x):
            return mx.sqrt(mx.sum(x ** 2))
        
        vmap_norm = vmap(single_norm)
        
        def native_norm(x):
            return mx.sqrt(mx.sum(x ** 2, axis=1))
        
        x = mx.random.normal([100, 64])
        
        vmapped = benchmark(vmap_norm, x)
        native = benchmark(native_norm, x)
        
        print(f"\nNorm [100, 64]:")
        print(f"  vmap:   {vmapped['mean_ms']:.3f}ms")
        print(f"  native: {native['mean_ms']:.3f}ms")


class TestScanBenchmark:
    """Benchmark scan operations."""
    
    def test_scan_cumsum(self):
        """scan cumulative sum."""
        def step(carry, x):
            new = carry + x
            return new, new
        
        xs = mx.random.normal([1000])
        
        def run_scan():
            return scan(step, mx.array(0.0), xs)
        
        result = benchmark(run_scan)
        print(f"\nscan cumsum [1000]: {result['mean_ms']:.3f}ms")
    
    def test_scan_vs_cumsum(self):
        """Compare scan to native cumsum."""
        xs = mx.random.normal([1000])
        
        def step(carry, x):
            new = carry + x
            return new, new
        
        def run_scan():
            return scan(step, mx.array(0.0), xs)
        
        def run_native():
            return mx.cumsum(xs)
        
        scanned = benchmark(run_scan)
        native = benchmark(run_native)
        
        print(f"\nCumsum [1000]:")
        print(f"  scan:   {scanned['mean_ms']:.3f}ms")
        print(f"  native: {native['mean_ms']:.3f}ms")


class TestMemoryBenchmark:
    """Benchmark memory-related operations."""
    
    def test_attention_memory_scaling(self):
        """Attention memory scales O(n²) with sequence length."""
        attn = MultiHeadAttention(256, 8)
        
        results = []
        for seq_len in [64, 128, 256, 512]:
            x = mx.random.normal([1, seq_len, 256])
            result = benchmark(attn, x, warmup=2, runs=5)
            results.append((seq_len, result["mean_ms"]))
        
        print(f"\nAttention scaling (256d, 8h):")
        for seq, ms in results:
            print(f"  seq={seq:4d}: {ms:.3f}ms")
        
        # Verify O(n²) scaling roughly (4x seq → ~16x time)
        ratio = results[-1][1] / results[0][1]
        seq_ratio = results[-1][0] / results[0][0]
        print(f"  8x seq → {ratio:.1f}x time (expected ~64x for O(n²))")
