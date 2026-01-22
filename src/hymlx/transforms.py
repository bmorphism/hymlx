"""JAX-style transformations for MLX."""

from typing import Any, Callable, TypeVar
import mlx.core as mx
import time

F = TypeVar("F", bound=Callable[..., Any])

# Telemetry integration (lazy to avoid circular import)
_telemetry_ctx = None

def _get_telemetry():
    """Lazy telemetry context."""
    global _telemetry_ctx
    if _telemetry_ctx is None:
        try:
            from hymlx import telemetry
            _telemetry_ctx = telemetry.get_context()
        except ImportError:
            _telemetry_ctx = False  # Mark as unavailable
    return _telemetry_ctx if _telemetry_ctx else None

def _emit(name: str, trit: int, payload: dict = None):
    """Emit telemetry event if available."""
    ctx = _get_telemetry()
    if ctx:
        ctx.emit(name, trit, payload)


def grad(f: F, argnums: int = 0) -> F:
    """Automatic differentiation - returns gradient function."""
    return mx.grad(f, argnums=argnums)


def value_and_grad(f: F, argnums: int = 0) -> Callable[..., tuple[Any, Any]]:
    """Returns both value and gradient."""
    return mx.value_and_grad(f, argnums=argnums)


def jit(f: F) -> F:
    """JIT compilation via mx.compile."""
    return mx.compile(f)


def vmap(f: F, in_axes: int | tuple = 0, out_axes: int = 0) -> F:
    """Vectorizing map - auto-batch over axis.
    
    MLX lacks native vmap, so we unroll manually.
    """
    def vmapped(*args):
        if isinstance(in_axes, int):
            batch_size = args[in_axes].shape[0]
            axes = [in_axes] * len(args)
        else:
            first_batched = next(i for i, ax in enumerate(in_axes) if ax is not None)
            batch_size = args[first_batched].shape[0]
            axes = list(in_axes)
        
        results = []
        for i in range(batch_size):
            sliced = []
            for j, arg in enumerate(args):
                if axes[j] is None:
                    sliced.append(arg)
                else:
                    sliced.append(arg[i])
            results.append(f(*sliced))
        
        return mx.stack(results, axis=out_axes)
    
    return vmapped


def scan(
    f: Callable[[Any, Any], tuple[Any, Any]], 
    init: Any, 
    xs: Any
) -> tuple[Any, Any]:
    """Functional loop with carry state.
    
    f: (carry, x) -> (new_carry, y)
    Returns: (final_carry, stacked_ys)
    """
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, mx.stack(ys)


def fori_loop(
    lower: int, 
    upper: int, 
    body: Callable[[int, Any], Any], 
    init: Any
) -> Any:
    """For loop with carry: body(i, carry) -> new_carry."""
    carry = init
    for i in range(lower, upper):
        carry = body(i, carry)
    return carry


def force(x: mx.array) -> mx.array:
    """Force evaluation of lazy computation."""
    mx.eval(x)
    return x


# ============================================================
# Traced Variants (with telemetry)
# ============================================================

def traced_grad(f: F, argnums: int = 0, name: str = None) -> F:
    """Gradient with telemetry tracing."""
    grad_fn = mx.grad(f, argnums=argnums)
    trace_name = name or getattr(f, "__name__", "anon")
    
    def traced(*args, **kwargs):
        _emit(f"grad:enter:{trace_name}", -1, {"argnums": argnums})  # MINUS for backward
        start = time.time()
        result = grad_fn(*args, **kwargs)
        _emit(f"grad:exit:{trace_name}", -1, {"duration_ms": (time.time() - start) * 1000})
        return result
    
    return traced


def traced_jit(f: F, name: str = None) -> F:
    """JIT with telemetry tracing."""
    compiled = mx.compile(f)
    trace_name = name or getattr(f, "__name__", "anon")
    
    def traced(*args, **kwargs):
        _emit(f"jit:enter:{trace_name}", 1)  # PLUS for forward/generation
        start = time.time()
        result = compiled(*args, **kwargs)
        _emit(f"jit:exit:{trace_name}", 1, {"duration_ms": (time.time() - start) * 1000})
        return result
    
    return traced


def traced_scan(
    f: Callable[[Any, Any], tuple[Any, Any]], 
    init: Any, 
    xs: Any,
    name: str = "scan"
) -> tuple[Any, Any]:
    """Scan with per-step telemetry."""
    _emit(f"scan:enter:{name}", 0, {"steps": len(xs) if hasattr(xs, "__len__") else "unknown"})
    start = time.time()
    
    carry = init
    ys = []
    for i, x in enumerate(xs):
        _emit(f"scan:step:{name}", 0, {"step": i})
        carry, y = f(carry, x)
        ys.append(y)
    
    _emit(f"scan:exit:{name}", 0, {"duration_ms": (time.time() - start) * 1000})
    return carry, mx.stack(ys)


def traced_fori_loop(
    lower: int, 
    upper: int, 
    body: Callable[[int, Any], Any], 
    init: Any,
    name: str = "fori_loop"
) -> Any:
    """For loop with telemetry."""
    _emit(f"fori:enter:{name}", 0, {"lower": lower, "upper": upper})
    start = time.time()
    
    carry = init
    for i in range(lower, upper):
        _emit(f"fori:iter:{name}", 0, {"i": i})
        carry = body(i, carry)
    
    _emit(f"fori:exit:{name}", 0, {"duration_ms": (time.time() - start) * 1000})
    return carry
