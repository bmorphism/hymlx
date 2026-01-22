#!/usr/bin/env python3
"""HyMLX CLI - JAX-style transforms for MLX."""

import argparse
import time
import mlx.core as mx

from hymlx.splitmix import (
    splitmix64, seed_to_trit, seed_to_hue, seed_to_rgb,
    derive, derive_chain, check_gf3, spawn_triad, reafference
)


def world(seed: int = 1069):
    """Run world - deterministic color worlding."""
    print("🌍 HyMLX World\n")
    print("Same algorithm as Gay.jl - deterministic colors on Apple Silicon\n")
    print(f"Base seed: {seed}")
    print("-" * 40)

    # Derivation chain
    print("\n📊 Derivation Chain:")
    chain = derive_chain(seed, 8)
    
    trit_symbols = {-1: "−", 0: "○", 1: "+"}
    for i, s in enumerate(chain):
        trit = seed_to_trit(s)
        hue = seed_to_hue(s)
        print(f"  [{i}] {s:20} | trit: {trit_symbols[trit]} | hue: {hue:3}°")

    # GF(3) triad
    print("\n🔺 GF(3) Balanced Triad:")
    triad = spawn_triad(seed)
    print(f"  MINUS: {triad['minus']}")
    print(f"  ZERO:  {triad['zero']}")
    print(f"  PLUS:  {triad['plus']}")
    
    seeds = [triad["minus"], triad["zero"], triad["plus"]]
    conserved = check_gf3(seeds)
    print(f"  Sum ≡ 0 (mod 3): {conserved} {'✓' if conserved else '✗'}")

    # RGB colors
    print("\n🌈 RGB Colors:")
    for i in range(5):
        s = derive(seed, i)
        rgb = seed_to_rgb(s)
        mx.eval(rgb)
        r, g, b = int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
        print(f"  [{i}] RGB({r:3}, {g:3}, {b:3}) | #{r:02x}{g:02x}{b:02x}")

    # Immune system
    print("\n🛡️ Immune System (Self/Non-Self):")
    self_result = reafference(seed, seed, 42)
    print(f"  Same seed: {self_result['status']} (free energy: {self_result['free_energy']:.4f})")
    
    other_result = reafference(seed, 9999, 42)
    print(f"  Diff seed: {other_result['status']} (free energy: {other_result['free_energy']:.4f})")

    # Benchmark
    print("\n⏱️ Benchmark:")
    n = 10000
    start = time.time()
    _ = derive_chain(seed, n)
    mx.eval(_)
    elapsed = time.time() - start
    print(f"  {n:,} derivations in {elapsed:.4f}s")
    print(f"  Throughput: {int(n / elapsed):,} seeds/sec")

    print("\n✅ World complete!")


def demo_transforms():
    """Demo JAX-style transforms."""
    from hymlx import grad, jit, vmap, scan, force
    
    print("🔧 HyMLX Transforms Demo\n")
    
    # grad
    print("1. grad - Automatic Differentiation:")
    def f(x):
        return mx.sum(x ** 2)
    
    grad_f = grad(f)
    x = mx.array([1.0, 2.0, 3.0])
    g = grad_f(x)
    force(g)
    print(f"   f(x) = sum(x²), x = {list(x)}")
    print(f"   ∇f(x) = {list(g)}")
    
    # jit
    print("\n2. jit - Just-In-Time Compilation:")
    @jit
    def fast_matmul(a, b):
        return a @ b
    
    a = mx.random.normal([100, 100])
    b = mx.random.normal([100, 100])
    start = time.time()
    for _ in range(100):
        _ = fast_matmul(a, b)
        force(_)
    print(f"   100 matmuls (100x100): {(time.time() - start)*1000:.2f}ms")
    
    # vmap
    print("\n3. vmap - Vectorized Map:")
    def single_dot(x):
        return mx.sum(x * x)
    
    batched_dot = vmap(single_dot)
    batch = mx.random.normal([5, 10])
    result = batched_dot(batch)
    force(result)
    print(f"   Batched dot products: {list(result)[:3]}...")
    
    # scan
    print("\n4. scan - Functional Loop:")
    def accumulate(carry, x):
        new_carry = carry + x
        return new_carry, new_carry
    
    xs = mx.array([1.0, 2.0, 3.0, 4.0, 5.0])
    final, running = scan(accumulate, mx.array(0.0), xs)
    force(running)
    print(f"   Running sum of {list(xs)}: {list(running)}")
    
    print("\n✅ Transforms demo complete!")


def main():
    parser = argparse.ArgumentParser(
        description="HyMLX: JAX-Style Transformations for MLX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  hymlx world              Run world (deterministic colors)
  hymlx world --seed 42    Use custom seed
  hymlx transforms         Demo grad/jit/vmap/scan
  hymlx color 1069 5       Generate 5 colors from seed 1069
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # world command
    world_parser = subparsers.add_parser("world", help="Run world (deterministic colors)")
    world_parser.add_argument("--seed", type=int, default=1069, help="Base seed")
    
    # transforms command
    subparsers.add_parser("transforms", help="Demo JAX-style transforms")
    
    # color command
    color_parser = subparsers.add_parser("color", help="Generate colors")
    color_parser.add_argument("seed", type=int, help="Base seed")
    color_parser.add_argument("count", type=int, nargs="?", default=1, help="Number of colors")
    
    args = parser.parse_args()
    
    if args.command == "world":
        world(args.seed)
    elif args.command == "transforms":
        demo_transforms()
    elif args.command == "color":
        chain = derive_chain(args.seed, args.count)
        mx.eval(chain)
        for i, s in enumerate(chain):
            rgb = seed_to_rgb(s)
            mx.eval(rgb)
            r, g, b = int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
            print(f"#{r:02x}{g:02x}{b:02x}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
