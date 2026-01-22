"""Feature parity tests between HyMLX and mlegls/hyjax.

HyJAX (github.com/mlegls/hyjax) provides:
- defn/j: JIT-compiled function definition
- mapv: vmap wrapper  
- if/j: lax.cond wrapper
- cond/j: multi-branch conditional
- lcond: lax.cond with explicit bindings

HyMLX provides MLX equivalents:
- jit: JIT compilation via mx.compile
- vmap: vectorized map (unrolled for MLX)
- grad: automatic differentiation
- value_and_grad: value + gradient
- scan: functional loop with carry
- fori_loop: indexed for loop

This file tests that HyMLX behavior matches expected JAX semantics.
"""

import pytest
import mlx.core as mx
from hymlx.transforms import grad, value_and_grad, jit, vmap, scan, fori_loop, force


class TestDefnJParity:
    """Test jit decorator matches defn/j behavior."""
    
    def test_selu_jit(self):
        """HyJAX example: (defn/j selu [x] ...)"""
        @jit
        def selu(x, alpha=1.67, lmbda=1.05):
            return lmbda * mx.where(x > 0, x, alpha * (mx.exp(x) - alpha))
        
        x = mx.array([1.0, -1.0, 0.0])
        result = selu(x)
        force(result)
        
        # Verify shape preserved
        assert result.shape == (3,)
        # Positive input unchanged (scaled)
        assert float(result[0]) > 0
        # Negative input transformed
        assert float(result[1]) < 0
    
    def test_jit_determinism(self):
        """JIT produces identical results on repeated calls."""
        @jit
        def f(x):
            return mx.sum(x ** 2)
        
        x = mx.array([1.0, 2.0, 3.0])
        r1 = f(x)
        r2 = f(x)
        force(r1)
        force(r2)
        
        assert abs(float(r1) - float(r2)) < 1e-10


class TestMapvParity:
    """Test vmap matches HyJAX's mapv behavior."""
    
    def test_mapv_apply_matrix(self):
        """HyJAX example: (mapv apply_matrix v_batched)"""
        mx.random.seed(0)
        mat = mx.random.normal([150, 100])
        batched_x = mx.random.normal([10, 100])
        
        def apply_matrix(v):
            return mat @ v
        
        # HyJAX: (mapv apply_matrix batched_x)
        vmapped = vmap(apply_matrix)
        result = vmapped(batched_x)
        force(result)
        
        assert result.shape == (10, 150)
    
    def test_mapv_preserves_batch(self):
        """vmap preserves batch dimension like HyJAX mapv."""
        def f(x):
            return mx.sum(x)
        
        batch = mx.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = vmap(f)(batch)
        force(result)
        
        assert result.shape == (3,)
        assert abs(float(result[0]) - 3.0) < 1e-5
        assert abs(float(result[1]) - 7.0) < 1e-5
        assert abs(float(result[2]) - 11.0) < 1e-5


class TestGradParity:
    """Test grad matches JAX grad behavior."""
    
    def test_sum_logistic_grad(self):
        """HyJAX example: (grad sum_logistic)"""
        def sum_logistic(x):
            return mx.sum(1.0 / (1.0 + mx.exp(-x)))
        
        x_small = mx.arange(3.0)
        derivative_fn = grad(sum_logistic)
        result = derivative_fn(x_small)
        force(result)
        
        # Gradient of sigmoid: σ(x) * (1 - σ(x))
        # At x=0: σ(0)*(1-σ(0)) = 0.5 * 0.5 = 0.25
        # At x=1: σ(1)*(1-σ(1)) ≈ 0.731 * 0.269 ≈ 0.197
        assert result.shape == (3,)
        assert abs(float(result[0]) - 0.25) < 0.01  # x=0
        assert abs(float(result[1]) - 0.197) < 0.01  # x=1
    
    def test_grad_scalar(self):
        """Basic scalar gradient."""
        def f(x):
            return x ** 2
        
        df = grad(f)
        result = df(mx.array(3.0))
        force(result)
        
        assert abs(float(result) - 6.0) < 1e-5


class TestCondParity:
    """Test conditional execution.
    
    HyJAX uses lax.cond for JIT-compatible conditionals.
    MLX handles this natively, but we verify behavior matches.
    """
    
    def test_if_j_simple(self):
        """HyJAX: (if/j (< x 3) (* 3 (** x 2)) (* -4 x))"""
        @jit
        def test_if(x):
            return mx.where(x < 3, 3 * (x ** 2), -4 * x)
        
        assert abs(float(test_if(mx.array(2.0))) - 12.0) < 1e-5  # 3 * 4
        assert abs(float(test_if(mx.array(5.0))) - (-20.0)) < 1e-5  # -4 * 5
    
    def test_nested_cond(self):
        """HyJAX: nested if/j"""
        @jit
        def test_nested(x):
            return mx.where(
                x < 3, 
                3 * (x ** 2),
                mx.where(x < 5, -4 * x, 5 * x)
            )
        
        assert abs(float(test_nested(mx.array(2.0))) - 12.0) < 1e-5
        assert abs(float(test_nested(mx.array(4.0))) - (-16.0)) < 1e-5
        assert abs(float(test_nested(mx.array(6.0))) - 30.0) < 1e-5


class TestMissingInHyjax:
    """Features HyMLX has that HyJAX lacks."""
    
    def test_scan_cumsum(self):
        """HyJAX TODO: lax.while_loop binding - HyMLX has scan."""
        def step(carry, x):
            new = carry + x
            return new, new
        
        xs = mx.array([1.0, 2.0, 3.0, 4.0, 5.0])
        final, history = scan(step, mx.array(0.0), xs)
        force(history)
        
        expected = [1.0, 3.0, 6.0, 10.0, 15.0]
        for i, exp in enumerate(expected):
            assert abs(float(history[i]) - exp) < 1e-5
    
    def test_fori_loop(self):
        """HyJAX TODO: lax.fori_loop binding - HyMLX has fori_loop."""
        def body(i, carry):
            return carry + i
        
        result = fori_loop(0, 10, body, mx.array(0.0))
        force(result)
        
        assert abs(float(result) - 45.0) < 1e-5
    
    def test_value_and_grad(self):
        """HyMLX provides value_and_grad not in HyJAX."""
        def f(x):
            return mx.sum(x ** 2)
        
        vg = value_and_grad(f)
        x = mx.array([1.0, 2.0, 3.0])
        val, g = vg(x)
        force(val)
        force(g)
        
        assert abs(float(val) - 14.0) < 1e-5
        assert abs(float(g[1]) - 4.0) < 1e-5


class TestFeatureMatrix:
    """Document feature parity matrix."""
    
    def test_feature_coverage(self):
        """
        Feature Matrix:
        
        | Feature        | HyJAX (JAX)     | HyMLX (MLX)      |
        |----------------|-----------------|------------------|
        | defn/j / jit   | ✓ @jit          | ✓ mx.compile     |
        | mapv / vmap    | ✓ jax.vmap      | ✓ unrolled       |
        | grad           | ✓ jax.grad      | ✓ mx.grad        |
        | value_and_grad | (implicit)      | ✓ mx.value_and_grad |
        | if/j           | ✓ lax.cond      | ✓ mx.where       |
        | cond/j         | ✓ lax.cond      | ✓ mx.where       |
        | while_loop     | TODO            | ✗                |
        | fori_loop      | TODO            | ✓ manual         |
        | scan           | (not listed)    | ✓ manual         |
        """
        # This test documents the feature matrix
        assert True
