"""Neural network utilities for HyMLX."""

from typing import Callable
import mlx.core as mx
import mlx.nn as nn


class Sequential(nn.Module):
    """Sequential container for layers."""
    
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)
        for i, layer in enumerate(self.layers):
            setattr(self, f"layer_{i}", layer)
    
    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


def sequential(*layers) -> Sequential:
    """Build sequential model from layers."""
    return Sequential(*layers)


def mlp_seeded(seed: int, layer_sizes: list[int], activation: str = "relu") -> Sequential:
    """Build MLP with deterministic Gay.jl seed."""
    mx.random.seed(seed)
    
    layers = []
    act_fn = nn.GELU() if activation == "gelu" else nn.ReLU()
    
    for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
        layers.append(nn.Linear(in_size, out_size))
        layers.append(act_fn)
    
    # Remove last activation
    layers.pop()
    
    return Sequential(*layers)


def train_step(
    model: nn.Module,
    loss_fn: Callable[[mx.array, mx.array], mx.array],
    optimizer
) -> Callable[[mx.array, mx.array], mx.array]:
    """Create training step function."""
    
    def step(x: mx.array, y: mx.array) -> mx.array:
        def compute_loss(model):
            return loss_fn(model(x), y)
        
        loss, grads = mx.value_and_grad(compute_loss)(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters())
        return loss
    
    return step
