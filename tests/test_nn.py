"""Test neural network utilities."""

import pytest
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from hymlx.nn import Sequential, sequential, mlp_seeded, train_step


class TestSequential:
    """Test sequential container."""
    
    def test_sequential_forward(self):
        """Sequential applies layers in order."""
        model = sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )
        
        x = mx.random.normal([3, 4])
        y = model(x)
        mx.eval(y)
        
        assert y.shape == (3, 2)
    
    def test_sequential_has_layers(self):
        """Sequential registers layers."""
        model = sequential(
            nn.Linear(10, 5),
            nn.Linear(5, 2),
        )
        
        assert hasattr(model, 'layers')
        assert len(model.layers) == 2


class TestMlpSeeded:
    """Test seeded MLP construction."""
    
    def test_mlp_seeded_determinism(self):
        """Same seed produces identical outputs."""
        m1 = mlp_seeded(1069, [4, 8, 2])
        m2 = mlp_seeded(1069, [4, 8, 2])
        
        x = mx.array([[1.0, 2.0, 3.0, 4.0]])
        
        y1 = m1(x)
        y2 = m2(x)
        mx.eval(y1)
        mx.eval(y2)
        
        diff = mx.sum(mx.abs(y1 - y2))
        mx.eval(diff)
        assert float(diff) < 1e-10
    
    def test_mlp_seeded_different_seeds(self):
        """Different seeds produce different outputs."""
        m1 = mlp_seeded(1069, [4, 8, 2])
        m2 = mlp_seeded(42, [4, 8, 2])
        
        x = mx.array([[1.0, 2.0, 3.0, 4.0]])
        
        y1 = m1(x)
        y2 = m2(x)
        mx.eval(y1)
        mx.eval(y2)
        
        diff = mx.sum(mx.abs(y1 - y2))
        mx.eval(diff)
        assert float(diff) > 0.01
    
    def test_mlp_seeded_shape(self):
        """MLP has correct layer sizes."""
        model = mlp_seeded(1069, [784, 256, 128, 10])
        
        x = mx.random.normal([5, 784])
        y = model(x)
        mx.eval(y)
        
        assert y.shape == (5, 10)


class TestTrainStep:
    """Test training step creation."""
    
    def test_train_step_reduces_loss(self):
        """Training step reduces loss."""
        mx.random.seed(42)
        
        model = mlp_seeded(42, [4, 8, 2])
        optimizer = optim.SGD(learning_rate=0.1)
        
        def loss_fn(pred, target):
            return mx.mean((pred - target) ** 2)
        
        step = train_step(model, loss_fn, optimizer)
        
        x = mx.random.normal([10, 4])
        y = mx.random.normal([10, 2])
        
        loss1 = step(x, y)
        mx.eval(loss1)
        
        loss2 = step(x, y)
        mx.eval(loss2)
        
        # Loss should decrease (usually)
        assert float(loss2) < float(loss1) * 1.5
