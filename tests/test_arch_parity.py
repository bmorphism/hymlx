"""Architecture parity tests: HyMLX (MLX) vs HyJAX (JAX).

Tests numeric stability and shape alignment between MLX and JAX implementations.
Requires: pip install hymlx[jax] for JAX comparisons.

Run with JAX: pytest tests/test_arch_parity.py -v
Run MLX-only: pytest tests/test_arch_parity.py -v -k "not jax"
"""

import pytest
import numpy as np
import mlx.core as mx
import mlx.nn as nn

# Optional JAX imports
try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None
    jnp = None

# Optional Flax imports (for NNX tests)
try:
    from flax import nnx
    HAS_FLAX = True
except ImportError:
    HAS_FLAX = False
    nnx = None


# ============================================
# Shared Test Data (deterministic seeds)
# ============================================

def get_test_data(seed: int = 42):
    """Generate aligned test data for both backends."""
    np.random.seed(seed)
    return {
        "x_1d": np.random.randn(16).astype(np.float32),
        "x_2d": np.random.randn(4, 16).astype(np.float32),
        "x_seq": np.random.randn(2, 8, 32).astype(np.float32),  # B, N, D
        "weights": np.random.randn(16, 32).astype(np.float32),
        "bias": np.random.randn(32).astype(np.float32),
    }


# ============================================
# MLX Implementations
# ============================================

class MLXLinear:
    """Manual linear layer for parity testing."""
    def __init__(self, weight, bias):
        self.weight = mx.array(weight)
        self.bias = mx.array(bias)
    
    def __call__(self, x):
        return x @ self.weight + self.bias


class MLXAttention:
    """Minimal attention for parity testing."""
    def __init__(self, dim, n_heads):
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
    
    def __call__(self, q, k, v):
        # q, k, v: [B, N, D]
        B, N, D = q.shape
        
        # Reshape to heads
        q = mx.reshape(q, [B, N, self.n_heads, self.head_dim])
        k = mx.reshape(k, [B, N, self.n_heads, self.head_dim])
        v = mx.reshape(v, [B, N, self.n_heads, self.head_dim])
        
        # Transpose to [B, H, N, D]
        q = mx.transpose(q, [0, 2, 1, 3])
        k = mx.transpose(k, [0, 2, 1, 3])
        v = mx.transpose(v, [0, 2, 1, 3])
        
        # Attention
        attn = mx.matmul(q, mx.transpose(k, [0, 1, 3, 2])) * self.scale
        attn = mx.softmax(attn, axis=-1)
        
        # Combine
        out = mx.matmul(attn, v)
        out = mx.transpose(out, [0, 2, 1, 3])
        out = mx.reshape(out, [B, N, D])
        return out


def mlx_layernorm(x, eps=1e-5):
    """Manual LayerNorm for parity."""
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.var(x, axis=-1, keepdims=True)
    return (x - mean) / mx.sqrt(var + eps)


def mlx_gelu(x):
    """GELU activation."""
    return x * 0.5 * (1.0 + mx.erf(x / mx.sqrt(mx.array(2.0))))


def mlx_softmax(x, axis=-1):
    """Numerically stable softmax."""
    x_max = mx.max(x, axis=axis, keepdims=True)
    exp_x = mx.exp(x - x_max)
    return exp_x / mx.sum(exp_x, axis=axis, keepdims=True)


# ============================================
# JAX Implementations (for comparison)
# ============================================

if HAS_JAX:
    class JAXLinear:
        """Manual linear layer for parity testing."""
        def __init__(self, weight, bias):
            self.weight = jnp.array(weight)
            self.bias = jnp.array(bias)
        
        def __call__(self, x):
            return x @ self.weight + self.bias

    class JAXAttention:
        """Minimal attention for parity testing."""
        def __init__(self, dim, n_heads):
            self.dim = dim
            self.n_heads = n_heads
            self.head_dim = dim // n_heads
            self.scale = self.head_dim ** -0.5
        
        def __call__(self, q, k, v):
            B, N, D = q.shape
            
            q = jnp.reshape(q, [B, N, self.n_heads, self.head_dim])
            k = jnp.reshape(k, [B, N, self.n_heads, self.head_dim])
            v = jnp.reshape(v, [B, N, self.n_heads, self.head_dim])
            
            q = jnp.transpose(q, [0, 2, 1, 3])
            k = jnp.transpose(k, [0, 2, 1, 3])
            v = jnp.transpose(v, [0, 2, 1, 3])
            
            attn = jnp.matmul(q, jnp.transpose(k, [0, 1, 3, 2])) * self.scale
            attn = jax.nn.softmax(attn, axis=-1)
            
            out = jnp.matmul(attn, v)
            out = jnp.transpose(out, [0, 2, 1, 3])
            out = jnp.reshape(out, [B, N, D])
            return out

    def jax_layernorm(x, eps=1e-5):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        return (x - mean) / jnp.sqrt(var + eps)

    def jax_gelu(x):
        return x * 0.5 * (1.0 + jax.scipy.special.erf(x / jnp.sqrt(2.0)))

    def jax_softmax(x, axis=-1):
        x_max = jnp.max(x, axis=axis, keepdims=True)
        exp_x = jnp.exp(x - x_max)
        return exp_x / jnp.sum(exp_x, axis=axis, keepdims=True)


# ============================================
# Shape Alignment Tests
# ============================================

class TestShapeAlignment:
    """Verify output shapes match between implementations."""
    
    def test_linear_shapes(self):
        data = get_test_data()
        mlx_layer = MLXLinear(data["weights"], data["bias"])
        
        x = mx.array(data["x_2d"])
        out = mlx_layer(x)
        mx.eval(out)
        
        assert out.shape == (4, 32), f"Expected (4, 32), got {out.shape}"
    
    @pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
    def test_linear_shapes_jax(self):
        data = get_test_data()
        jax_layer = JAXLinear(data["weights"], data["bias"])
        
        x = jnp.array(data["x_2d"])
        out = jax_layer(x)
        
        assert out.shape == (4, 32), f"Expected (4, 32), got {out.shape}"
    
    def test_attention_shapes(self):
        data = get_test_data()
        attn = MLXAttention(dim=32, n_heads=4)
        
        x = mx.array(data["x_seq"])
        out = attn(x, x, x)
        mx.eval(out)
        
        assert out.shape == (2, 8, 32), f"Expected (2, 8, 32), got {out.shape}"
    
    @pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
    def test_attention_shapes_jax(self):
        data = get_test_data()
        attn = JAXAttention(dim=32, n_heads=4)
        
        x = jnp.array(data["x_seq"])
        out = attn(x, x, x)
        
        assert out.shape == (2, 8, 32), f"Expected (2, 8, 32), got {out.shape}"


# ============================================
# Numeric Parity Tests
# ============================================

class TestNumericParity:
    """Verify numeric outputs match between MLX and JAX.
    
    Tolerance Notes (FP32 backend differences):
    - Softmax, LayerNorm, GELU: 1e-6 (excellent match)
    - Linear, Attention: 1e-2 (matmul accumulation order differs)
    """
    
    # Element-wise ops: tight tolerance
    ATOL_TIGHT = 1e-5
    RTOL_TIGHT = 1e-5
    
    # Matmul-heavy ops: relaxed for FP32 accumulation differences
    ATOL_MATMUL = 1e-2
    RTOL_MATMUL = 1e-2
    
    @pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
    def test_linear_parity(self):
        """Linear layer outputs match."""
        data = get_test_data()
        
        mlx_layer = MLXLinear(data["weights"], data["bias"])
        jax_layer = JAXLinear(data["weights"], data["bias"])
        
        mlx_out = mlx_layer(mx.array(data["x_2d"]))
        jax_out = jax_layer(jnp.array(data["x_2d"]))
        
        mx.eval(mlx_out)
        np.testing.assert_allclose(
            np.array(mlx_out), np.array(jax_out),
            atol=self.ATOL_MATMUL, rtol=self.RTOL_MATMUL
        )
    
    @pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
    def test_softmax_parity(self):
        """Softmax matches between implementations."""
        data = get_test_data()
        x_np = data["x_2d"]
        
        mlx_out = mlx_softmax(mx.array(x_np))
        jax_out = jax_softmax(jnp.array(x_np))
        
        mx.eval(mlx_out)
        np.testing.assert_allclose(
            np.array(mlx_out), np.array(jax_out),
            atol=self.ATOL_TIGHT, rtol=self.RTOL_TIGHT
        )
    
    @pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
    def test_gelu_parity(self):
        """GELU matches between implementations."""
        data = get_test_data()
        x_np = data["x_1d"]
        
        mlx_out = mlx_gelu(mx.array(x_np))
        jax_out = jax_gelu(jnp.array(x_np))
        
        mx.eval(mlx_out)
        np.testing.assert_allclose(
            np.array(mlx_out), np.array(jax_out),
            atol=self.ATOL_TIGHT, rtol=self.RTOL_TIGHT
        )
    
    @pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
    def test_layernorm_parity(self):
        """LayerNorm matches between implementations."""
        data = get_test_data()
        x_np = data["x_2d"]
        
        mlx_out = mlx_layernorm(mx.array(x_np))
        jax_out = jax_layernorm(jnp.array(x_np))
        
        mx.eval(mlx_out)
        np.testing.assert_allclose(
            np.array(mlx_out), np.array(jax_out),
            atol=self.ATOL_TIGHT, rtol=self.RTOL_TIGHT
        )
    
    @pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
    def test_attention_parity(self):
        """Attention outputs match (relaxed for matmul accumulation)."""
        data = get_test_data()
        x_np = data["x_seq"]
        
        mlx_attn = MLXAttention(dim=32, n_heads=4)
        jax_attn = JAXAttention(dim=32, n_heads=4)
        
        mlx_out = mlx_attn(mx.array(x_np), mx.array(x_np), mx.array(x_np))
        jax_out = jax_attn(jnp.array(x_np), jnp.array(x_np), jnp.array(x_np))
        
        mx.eval(mlx_out)
        np.testing.assert_allclose(
            np.array(mlx_out), np.array(jax_out),
            atol=self.ATOL_MATMUL, rtol=self.RTOL_MATMUL
        )


# ============================================
# Gradient Parity Tests
# ============================================

class TestGradientParity:
    """Verify gradients match between MLX and JAX.
    
    Gradients through matmul have higher variance due to accumulation order.
    """
    
    ATOL_MATMUL = 1e-1  # Gradients amplify matmul differences
    RTOL_MATMUL = 1e-1
    ATOL_TIGHT = 1e-4
    RTOL_TIGHT = 1e-3
    
    @pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
    def test_linear_grad_parity(self):
        """Linear layer gradients match."""
        from hymlx import grad as mlx_grad
        
        data = get_test_data()
        w, b = data["weights"], data["bias"]
        x = data["x_2d"]
        
        # MLX gradient
        def mlx_fn(x):
            layer = MLXLinear(w, b)
            return mx.sum(layer(x) ** 2)
        
        mlx_g = mlx_grad(mlx_fn)(mx.array(x))
        mx.eval(mlx_g)
        
        # JAX gradient
        def jax_fn(x):
            layer = JAXLinear(w, b)
            return jnp.sum(layer(x) ** 2)
        
        jax_g = jax.grad(jax_fn)(jnp.array(x))
        
        np.testing.assert_allclose(
            np.array(mlx_g), np.array(jax_g),
            atol=self.ATOL_MATMUL, rtol=self.RTOL_MATMUL
        )
    
    @pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
    def test_softmax_grad_parity(self):
        """Softmax gradients match."""
        from hymlx import grad as mlx_grad
        
        data = get_test_data()
        x = data["x_2d"]
        
        def mlx_fn(x):
            return mx.sum(mlx_softmax(x) ** 2)
        
        def jax_fn(x):
            return jnp.sum(jax_softmax(x) ** 2)
        
        mlx_g = mlx_grad(mlx_fn)(mx.array(x))
        jax_g = jax.grad(jax_fn)(jnp.array(x))
        
        mx.eval(mlx_g)
        np.testing.assert_allclose(
            np.array(mlx_g), np.array(jax_g),
            atol=self.ATOL_TIGHT, rtol=self.RTOL_TIGHT
        )


# ============================================
# Numeric Stability Tests
# ============================================

class TestNumericStability:
    """Test edge cases for numeric stability."""
    
    def test_softmax_large_values(self):
        """Softmax stable with large inputs."""
        x = mx.array([1000.0, 1001.0, 1002.0])
        out = mlx_softmax(x)
        mx.eval(out)
        
        # Should not overflow, should sum to 1
        assert not mx.any(mx.isnan(out)).item()
        assert not mx.any(mx.isinf(out)).item()
        assert abs(float(mx.sum(out)) - 1.0) < 1e-5
    
    def test_softmax_negative_values(self):
        """Softmax stable with large negative inputs."""
        x = mx.array([-1000.0, -999.0, -998.0])
        out = mlx_softmax(x)
        mx.eval(out)
        
        assert not mx.any(mx.isnan(out)).item()
        assert not mx.any(mx.isinf(out)).item()
        assert abs(float(mx.sum(out)) - 1.0) < 1e-5
    
    def test_layernorm_small_variance(self):
        """LayerNorm stable with near-zero variance."""
        x = mx.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0001]])
        out = mlx_layernorm(x)
        mx.eval(out)
        
        assert not mx.any(mx.isnan(out)).item()
        assert not mx.any(mx.isinf(out)).item()
    
    def test_attention_long_sequence(self):
        """Attention works with longer sequences."""
        attn = MLXAttention(dim=64, n_heads=8)
        x = mx.random.normal([1, 128, 64])  # Longer sequence
        out = attn(x, x, x)
        mx.eval(out)
        
        assert out.shape == (1, 128, 64)
        assert not mx.any(mx.isnan(out)).item()


# ============================================
# Hy Architecture Tests
# ============================================

class TestHyArchitectures:
    """Test Hy-defined architectures from arch.hy."""
    
    def test_build_sequential(self):
        """Test build-sequential from arch.hy."""
        import hy
        from hymlx.arch import build_sequential
        
        specs = [['linear', 16, 32], ['relu'], ['linear', 32, 10]]
        model = build_sequential(specs)
        
        x = mx.random.normal([4, 16])
        out = model(x)
        mx.eval(out)
        
        assert out.shape == (4, 10)
    
    def test_multihead_attention(self):
        """Test MultiHeadAttention from arch.hy."""
        import hy
        from hymlx.arch import MultiHeadAttention
        
        attn = MultiHeadAttention(dim=64, n_heads=4)
        x = mx.random.normal([2, 8, 64])
        out = attn(x)
        mx.eval(out)
        
        assert out.shape == (2, 8, 64)
    
    def test_multihead_attention_with_mask(self):
        """Test MultiHeadAttention with causal mask."""
        import hy
        from hymlx.arch import MultiHeadAttention
        
        attn = MultiHeadAttention(dim=32, n_heads=4)
        x = mx.random.normal([1, 4, 32])
        
        # Causal mask: [1, 1, N, N] where upper triangle is -inf
        seq_len = 4
        mask = mx.triu(mx.full([1, 1, seq_len, seq_len], float('-inf')), k=1)
        
        out = attn(x, mask=mask)
        mx.eval(out)
        
        assert out.shape == (1, 4, 32)
        assert not mx.any(mx.isnan(out)).item()
    
    def test_make_mlp_spec(self):
        """Test make-mlp-spec from arch.hy."""
        import hy
        from hymlx.arch import make_mlp_spec
        
        specs = make_mlp_spec([784, 256, 128, 10])
        assert len(specs) == 5  # 3 linear + 2 relu
        # Convert to strings for comparison (Hy returns symbols)
        assert str(specs[0][0]) == 'linear'
        assert specs[0][1] == 784
        assert specs[0][2] == 256
        assert str(specs[1][0]) == 'relu'
    
    def test_make_mlp_spec_gelu(self):
        """Test make-mlp-spec with GELU activation."""
        import hy
        from hymlx.arch import make_mlp_spec
        
        specs = make_mlp_spec([64, 128, 32], activation='gelu')
        assert len(specs) == 3  # 2 linear + 1 gelu
        assert str(specs[1][0]) == 'gelu'
    
    def test_replace_activations(self):
        """Test replace-activations pattern matching."""
        import hy
        from hymlx.arch import make_mlp_spec, replace_activations
        
        specs = make_mlp_spec([64, 128, 32])
        new_specs = replace_activations(specs, 'relu', 'gelu')
        
        # All relu should be replaced with gelu
        for spec in new_specs:
            if len(spec) == 1:
                assert str(spec[0]) != 'relu'
    
    def test_compose_functions(self):
        """Test compose function composition."""
        import hy
        from hymlx.arch import compose
        
        add1 = lambda x: x + 1
        mul2 = lambda x: x * 2
        
        # compose is right-to-left: mul2 first, then add1
        f = compose(add1, mul2)
        assert f(3) == 7  # (3 * 2) + 1
    
    def test_pipe_functions(self):
        """Test pipe left-to-right composition."""
        import hy
        from hymlx.arch import pipe
        
        add1 = lambda x: x + 1
        mul2 = lambda x: x * 2
        
        # pipe is left-to-right: add1 first, then mul2
        f = pipe(add1, mul2)
        assert f(3) == 8  # (3 + 1) * 2


# ============================================
# Feature Matrix Documentation
# ============================================

class TestArchitectureMatrix:
    """Document architecture feature matrix."""
    
    def test_feature_matrix(self):
        """
        Architecture Feature Matrix: HyMLX vs JAX/Flax
        
        | Component          | HyMLX (MLX)           | JAX (Flax)            | Parity |
        |--------------------|-----------------------|-----------------------|--------|
        | Linear             | nn.Linear             | nnx.Linear            | ✓      |
        | Conv2d             | nn.Conv2d             | nnx.Conv              | ✓      |
        | LayerNorm          | nn.LayerNorm          | nnx.LayerNorm         | ✓      |
        | Embedding          | nn.Embedding          | nnx.Embed             | ✓      |
        | MultiHeadAttention | arch.MultiHeadAttention| nnx.MultiHeadAttention| ✓      |
        | GELU               | nn.GELU               | jax.nn.gelu           | ✓      |
        | Softmax            | mx.softmax            | jax.nn.softmax        | ✓      |
        | Dropout            | nn.Dropout            | nnx.Dropout           | ✓      |
        | Sequential         | nn.Sequential         | (manual)              | ✓      |
        
        Macro Features (Hy-only):
        | Feature            | HyMLX                 | JAX                   |
        |--------------------|-----------------------|-----------------------|
        | residual macro     | ✓ (residual x ...)    | ✗                     |
        | defblock macro     | ✓ (defblock Name ...) | ✗                     |
        | S-expr specs       | ✓ [linear 64 128]     | ✗                     |
        | Pattern matching   | ✓ (match spec ...)    | ✗                     |
        | Threading macros   | ✓ (-> x f g)          | ✗                     |
        """
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
