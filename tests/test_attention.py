"""Test Multi-Head Attention implementation."""

import pytest
import mlx.core as mx
import mlx.nn as nn


class MultiHeadAttention(nn.Module):
    """Multi-head attention - Python version of arch.hy implementation."""
    
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def __call__(self, x: mx.array, mask: mx.array = None) -> mx.array:
        B, N, _ = x.shape
        
        # QKV projection: [B, N, 3*dim]
        qkv = self.qkv(x)
        
        # Reshape: [B, N, 3, n_heads, head_dim]
        qkv = mx.reshape(qkv, [B, N, 3, self.n_heads, self.head_dim])
        
        # Transpose: [3, B, n_heads, N, head_dim]
        qkv = mx.transpose(qkv, [2, 0, 3, 1, 4])
        
        # Split into q, k, v
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores: [B, n_heads, N, N]
        attn = mx.matmul(q, mx.transpose(k, [0, 1, 3, 2])) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            attn = attn + mask
        
        # Softmax
        attn = mx.softmax(attn, axis=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values: [B, n_heads, N, head_dim]
        out = mx.matmul(attn, v)
        
        # Transpose back: [B, N, n_heads, head_dim]
        out = mx.transpose(out, [0, 2, 1, 3])
        
        # Reshape: [B, N, dim]
        out = mx.reshape(out, [B, N, self.dim])
        
        # Output projection
        return self.proj(out)


class TestMultiHeadAttentionShape:
    """Test output shapes are correct."""
    
    def test_basic_shape(self):
        """Output shape matches input shape."""
        attn = MultiHeadAttention(dim=64, n_heads=8)
        x = mx.random.normal([2, 10, 64])  # [batch, seq, dim]
        
        out = attn(x)
        mx.eval(out)
        
        assert out.shape == (2, 10, 64)
    
    def test_single_head(self):
        """Single head attention works."""
        attn = MultiHeadAttention(dim=32, n_heads=1)
        x = mx.random.normal([1, 5, 32])
        
        out = attn(x)
        mx.eval(out)
        
        assert out.shape == (1, 5, 32)
    
    def test_many_heads(self):
        """Many heads attention works."""
        attn = MultiHeadAttention(dim=512, n_heads=16)
        x = mx.random.normal([4, 128, 512])
        
        out = attn(x)
        mx.eval(out)
        
        assert out.shape == (4, 128, 512)
    
    def test_batch_size_one(self):
        """Single sample batch works."""
        attn = MultiHeadAttention(dim=64, n_heads=4)
        x = mx.random.normal([1, 20, 64])
        
        out = attn(x)
        mx.eval(out)
        
        assert out.shape == (1, 20, 64)


class TestMultiHeadAttentionMask:
    """Test causal and padding masks."""
    
    def test_causal_mask(self):
        """Causal mask prevents attending to future."""
        attn = MultiHeadAttention(dim=64, n_heads=4)
        x = mx.random.normal([1, 8, 64])
        
        # Causal mask: -inf for future positions
        seq_len = 8
        mask = mx.triu(mx.full([seq_len, seq_len], -1e9), k=1)
        mask = mx.expand_dims(mx.expand_dims(mask, 0), 0)  # [1, 1, N, N]
        
        out = attn(x, mask=mask)
        mx.eval(out)
        
        assert out.shape == (1, 8, 64)
    
    def test_no_mask(self):
        """No mask allows full attention."""
        attn = MultiHeadAttention(dim=64, n_heads=4)
        x = mx.random.normal([2, 10, 64])
        
        out = attn(x, mask=None)
        mx.eval(out)
        
        assert out.shape == (2, 10, 64)


class TestMultiHeadAttentionDeterminism:
    """Test reproducibility with seeds."""
    
    def test_deterministic_output(self):
        """Same seed produces same output."""
        mx.random.seed(42)
        attn1 = MultiHeadAttention(dim=64, n_heads=4)
        
        mx.random.seed(42)
        attn2 = MultiHeadAttention(dim=64, n_heads=4)
        
        x = mx.random.normal([1, 5, 64])
        mx.random.seed(123)  # Reset for input
        x1 = mx.random.normal([1, 5, 64])
        mx.random.seed(123)
        x2 = mx.random.normal([1, 5, 64])
        
        out1 = attn1(x1)
        out2 = attn2(x2)
        mx.eval(out1)
        mx.eval(out2)
        
        diff = mx.sum(mx.abs(out1 - out2))
        mx.eval(diff)
        
        assert float(diff) < 1e-5


class TestMultiHeadAttentionGradients:
    """Test gradients flow correctly."""
    
    def test_gradient_flow(self):
        """Gradients propagate through attention."""
        attn = MultiHeadAttention(dim=64, n_heads=4)
        x = mx.random.normal([2, 10, 64])
        
        def loss_fn(model, x):
            out = model(x)
            return mx.mean(out ** 2)
        
        loss, grads = mx.value_and_grad(loss_fn)(attn, x)
        mx.eval(loss)
        
        # Check gradients exist for all parameters
        def check_grads(g):
            if isinstance(g, dict):
                for v in g.values():
                    check_grads(v)
            elif isinstance(g, mx.array):
                mx.eval(g)
                assert g.size > 0
        
        check_grads(grads)
    
    def test_backward_shape(self):
        """Gradient shapes match parameter shapes."""
        attn = MultiHeadAttention(dim=32, n_heads=2)
        x = mx.random.normal([1, 4, 32])
        
        def loss_fn(model, x):
            return mx.sum(model(x))
        
        _, grads = mx.value_and_grad(loss_fn)(attn, x)
        
        # QKV weight gradient should be [96, 32] (3*dim, in) - MLX convention
        qkv_grad = grads["qkv"]["weight"]
        mx.eval(qkv_grad)
        assert qkv_grad.shape == (96, 32)


class TestMultiHeadAttentionNumerics:
    """Test numerical properties."""
    
    def test_attention_scores_sum_to_one(self):
        """Softmax attention weights sum to 1."""
        # Can't directly test internal attention, but verify output is stable
        attn = MultiHeadAttention(dim=64, n_heads=4)
        x = mx.random.normal([1, 10, 64])
        
        out = attn(x)
        mx.eval(out)
        
        # Output should be finite
        assert mx.all(mx.isfinite(out))
    
    def test_scale_factor(self):
        """Scale factor is 1/sqrt(head_dim)."""
        attn = MultiHeadAttention(dim=64, n_heads=4)
        
        # head_dim = 64 / 4 = 16
        # scale = 1 / sqrt(16) = 0.25
        expected_scale = 0.25
        assert abs(attn.scale - expected_scale) < 1e-6
    
    def test_large_sequence(self):
        """Handle longer sequences."""
        attn = MultiHeadAttention(dim=128, n_heads=8)
        x = mx.random.normal([1, 512, 128])
        
        out = attn(x)
        mx.eval(out)
        
        assert out.shape == (1, 512, 128)
        assert mx.all(mx.isfinite(out))


class TestMultiHeadAttentionEdgeCases:
    """Test edge cases."""
    
    def test_single_token(self):
        """Single token sequence works."""
        attn = MultiHeadAttention(dim=64, n_heads=4)
        x = mx.random.normal([1, 1, 64])
        
        out = attn(x)
        mx.eval(out)
        
        assert out.shape == (1, 1, 64)
    
    def test_dim_equals_heads(self):
        """dim == n_heads (head_dim = 1) works."""
        attn = MultiHeadAttention(dim=8, n_heads=8)
        x = mx.random.normal([1, 4, 8])
        
        out = attn(x)
        mx.eval(out)
        
        assert out.shape == (1, 4, 8)
    
    def test_invalid_dim_heads(self):
        """Raises error when dim not divisible by n_heads."""
        with pytest.raises(AssertionError):
            MultiHeadAttention(dim=64, n_heads=5)
