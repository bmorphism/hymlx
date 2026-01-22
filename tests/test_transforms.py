"""Test JAX-style transforms against expected behavior."""

import pytest
import mlx.core as mx
from hymlx.transforms import grad, value_and_grad, jit, vmap, scan, fori_loop, force


class TestGrad:
    """Test automatic differentiation matches JAX semantics."""
    
    def test_scalar_grad(self):
        """grad of x² at x=3 should be 2*3=6."""
        def f(x):
            return x ** 2
        
        df = grad(f)
        result = df(mx.array(3.0))
        force(result)
        assert abs(float(result) - 6.0) < 1e-5
    
    def test_sum_of_squares_grad(self):
        """grad of sum(x²) should be 2x."""
        def f(x):
            return mx.sum(x ** 2)
        
        df = grad(f)
        x = mx.array([1.0, 2.0, 3.0])
        result = df(x)
        force(result)
        
        expected = [2.0, 4.0, 6.0]
        for i in range(3):
            assert abs(float(result[i]) - expected[i]) < 1e-5
    
    def test_value_and_grad(self):
        """value_and_grad returns both value and gradient."""
        def f(x):
            return mx.sum(x ** 2)
        
        vg = value_and_grad(f)
        x = mx.array([1.0, 2.0, 3.0])
        val, g = vg(x)
        force(val)
        force(g)
        
        assert abs(float(val) - 14.0) < 1e-5  # 1 + 4 + 9
        assert abs(float(g[0]) - 2.0) < 1e-5
        assert abs(float(g[1]) - 4.0) < 1e-5
        assert abs(float(g[2]) - 6.0) < 1e-5
    
    def test_grad_chain_rule(self):
        """Test chain rule: d/dx sin(x²) = 2x*cos(x²)."""
        def f(x):
            return mx.sin(x ** 2)
        
        df = grad(f)
        x = mx.array(1.0)
        result = df(x)
        force(result)
        
        # 2 * 1 * cos(1) ≈ 1.0806
        expected = 2.0 * 1.0 * float(mx.cos(mx.array(1.0)))
        assert abs(float(result) - expected) < 1e-5


class TestJit:
    """Test JIT compilation."""
    
    def test_jit_correctness(self):
        """JIT doesn't change computation results."""
        def f(x):
            return mx.sum(x ** 2 + mx.sin(x))
        
        jit_f = jit(f)
        x = mx.array([1.0, 2.0, 3.0])
        
        result_eager = f(x)
        result_jit = jit_f(x)
        force(result_eager)
        force(result_jit)
        
        assert abs(float(result_eager) - float(result_jit)) < 1e-5
    
    def test_jit_determinism(self):
        """JIT produces same result on repeated calls."""
        @jit
        def f(x):
            return x @ x.T
        
        x = mx.array([[1.0, 2.0], [3.0, 4.0]])
        r1 = f(x)
        r2 = f(x)
        force(r1)
        force(r2)
        
        for i in range(2):
            for j in range(2):
                assert abs(float(r1[i, j]) - float(r2[i, j])) < 1e-5


class TestVmap:
    """Test vectorized mapping."""
    
    def test_vmap_single_arg(self):
        """vmap over single argument."""
        def f(x):
            return mx.sum(x)
        
        vf = vmap(f)
        batch = mx.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = vf(batch)
        force(result)
        
        assert abs(float(result[0]) - 3.0) < 1e-5
        assert abs(float(result[1]) - 7.0) < 1e-5
        assert abs(float(result[2]) - 11.0) < 1e-5
    
    def test_vmap_preserves_batch_size(self):
        """Output batch size matches input."""
        def f(x):
            return x * 2
        
        vf = vmap(f)
        batch = mx.random.normal([7, 3])
        result = vf(batch)
        force(result)
        
        assert result.shape[0] == 7


class TestScan:
    """Test functional scan loop."""
    
    def test_scan_cumsum(self):
        """scan can compute cumulative sum."""
        def step(carry, x):
            new_carry = carry + x
            return new_carry, new_carry
        
        xs = mx.array([1.0, 2.0, 3.0, 4.0, 5.0])
        final, history = scan(step, mx.array(0.0), xs)
        force(final)
        force(history)
        
        assert abs(float(final) - 15.0) < 1e-5
        expected = [1.0, 3.0, 6.0, 10.0, 15.0]
        for i, exp in enumerate(expected):
            assert abs(float(history[i]) - exp) < 1e-5
    
    def test_scan_running_max(self):
        """scan can compute running maximum."""
        def step(carry, x):
            new_carry = mx.maximum(carry, x)
            return new_carry, new_carry
        
        xs = mx.array([1.0, 5.0, 3.0, 7.0, 2.0])
        final, history = scan(step, mx.array(-float('inf')), xs)
        force(history)
        
        expected = [1.0, 5.0, 5.0, 7.0, 7.0]
        for i, exp in enumerate(expected):
            assert abs(float(history[i]) - exp) < 1e-5


class TestForiLoop:
    """Test for-loop with carry."""
    
    def test_fori_sum(self):
        """fori_loop can sum a range."""
        def body(i, carry):
            return carry + i
        
        result = fori_loop(0, 10, body, mx.array(0.0))
        force(result)
        
        assert abs(float(result) - 45.0) < 1e-5  # 0+1+...+9
    
    def test_fori_factorial(self):
        """fori_loop can compute factorial."""
        def body(i, carry):
            return carry * (i + 1)
        
        result = fori_loop(0, 5, body, mx.array(1.0))
        force(result)
        
        assert abs(float(result) - 120.0) < 1e-5  # 5!
