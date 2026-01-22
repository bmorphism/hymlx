"""SplitMix64 PRNG - Gay.jl compatible deterministic colors.

Uses pure Python for uint64 operations since MLX uint64 has issues.
"""

import mlx.core as mx

# Constants (same as Gay.jl / JAX)
GOLDEN = 0x9E3779B97F4A7C15
MIX1 = 0xBF58476D1CE4E5B9
MIX2 = 0x94D049BB133111EB
MASK64 = (1 << 64) - 1


def splitmix64(z: int) -> int:
    """Pure functional SplitMix64 - deterministic PRNG step."""
    z = (z + GOLDEN) & MASK64
    z = ((z ^ (z >> 30)) * MIX1) & MASK64
    z = ((z ^ (z >> 27)) * MIX2) & MASK64
    return (z ^ (z >> 31)) & MASK64


def seed_to_trit(seed: int) -> int:
    """Convert seed to GF(3) trit: {-1, 0, +1}."""
    return (seed % 3) - 1


def seed_to_hue(seed: int) -> int:
    """Convert seed to hue in [0, 360)."""
    return seed % 360


def seed_to_rgb(seed: int, saturation: float = 0.7, lightness: float = 0.55) -> mx.array:
    """Convert seed to RGB via HSL."""
    hue = seed_to_hue(seed) / 360.0
    c = saturation * (1.0 - abs(2.0 * lightness - 1.0))
    x = c * (1.0 - abs((hue * 6.0) % 2.0 - 1.0))
    m = lightness - c / 2.0
    
    h_sector = int(hue * 6.0)
    if h_sector < 1:
        rgb = (c, x, 0.0)
    elif h_sector < 2:
        rgb = (x, c, 0.0)
    elif h_sector < 3:
        rgb = (0.0, c, x)
    elif h_sector < 4:
        rgb = (0.0, x, c)
    elif h_sector < 5:
        rgb = (x, 0.0, c)
    else:
        rgb = (c, 0.0, x)
    
    return mx.array([rgb[0] + m, rgb[1] + m, rgb[2] + m], dtype=mx.float32)


def derive(seed: int, index: int) -> int:
    """Derive seed at specific index (XOR then hash)."""
    return splitmix64(seed ^ index)


def derive_chain(seed: int, n: int) -> list[int]:
    """Generate chain of n seeds from initial seed."""
    seeds = []
    current = seed
    for _ in range(n):
        seeds.append(current)
        current = splitmix64(current)
    return seeds


def check_gf3(seeds: list[int]) -> bool:
    """Verify sum of trits ≡ 0 (mod 3)."""
    total = sum(seed_to_trit(s) for s in seeds)
    return total % 3 == 0


def spawn_triad(base_seed: int) -> dict[str, int | None]:
    """Spawn balanced triad with trits (-1, 0, +1)."""
    found: dict[str, int | None] = {"minus": None, "zero": None, "plus": None}
    
    for i in range(1000):
        seed = derive(base_seed, i)
        trit = seed_to_trit(seed)
        
        if trit == -1 and found["minus"] is None:
            found["minus"] = seed
        elif trit == 0 and found["zero"] is None:
            found["zero"] = seed
        elif trit == 1 and found["plus"] is None:
            found["plus"] = seed
        
        if all(v is not None for v in found.values()):
            break
    
    return found


def reafference(host_seed: int, sample_seed: int, index: int) -> dict:
    """Self/non-self discrimination via prediction matching."""
    predicted = derive(host_seed, index)
    observed = derive(sample_seed, index)
    
    pred_hue = seed_to_hue(predicted)
    obs_hue = seed_to_hue(observed)
    
    diff = abs(pred_hue - obs_hue)
    hue_diff = min(diff, 360 - diff)
    free_energy = hue_diff / 180.0
    
    if free_energy < 0.01:
        status = "SELF"
    elif free_energy < 0.3:
        status = "BOUNDARY"
    else:
        status = "NON_SELF"
    
    return {
        "match": predicted == observed,
        "free_energy": free_energy,
        "is_self": free_energy < 0.01,
        "status": status,
    }
