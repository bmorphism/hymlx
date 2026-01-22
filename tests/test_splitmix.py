"""Test SplitMix64 against known JAX/Gay.jl values."""

import pytest
from hymlx.splitmix import (
    splitmix64, seed_to_trit, seed_to_hue, derive, derive_chain,
    check_gf3, spawn_triad, GOLDEN, MIX1, MIX2, MASK64
)


class TestConstants:
    """Verify constants match JAX/Gay.jl exactly."""
    
    def test_golden(self):
        assert GOLDEN == 0x9E3779B97F4A7C15
    
    def test_mix1(self):
        assert MIX1 == 0xBF58476D1CE4E5B9
    
    def test_mix2(self):
        assert MIX2 == 0x94D049BB133111EB
    
    def test_mask64(self):
        assert MASK64 == (1 << 64) - 1


class TestSplitMix64:
    """Test splitmix64 produces exact JAX-compatible values."""
    
    # Known values verified against reference implementation
    KNOWN_VALUES = [
        (0, 16294208416658607535),
        (1, 10451216379200822465),
        (42, 13679457532755275413),
        (1069, 10908445076698799222),
    ]
    
    @pytest.mark.parametrize("seed,expected", KNOWN_VALUES)
    def test_known_values(self, seed, expected):
        assert splitmix64(seed) == expected
    
    def test_chain_determinism(self):
        """Same seed always produces same chain."""
        chain1 = derive_chain(1069, 10)
        chain2 = derive_chain(1069, 10)
        assert chain1 == chain2
    
    def test_chain_values(self):
        """Verify specific chain values."""
        chain = derive_chain(1069, 5)
        assert chain[0] == 1069
        assert chain[1] == 10908445076698799222
        assert chain[2] == 14533190410327020252
        assert chain[3] == 8231059306442291956
        assert chain[4] == 11119431197519591511


class TestDerive:
    """Test indexed derivation."""
    
    def test_derive_index_0(self):
        # derive(seed, 0) = splitmix64(seed ^ 0) = splitmix64(seed)
        assert derive(1069, 0) == splitmix64(1069)
    
    def test_derive_different_indices(self):
        """Different indices produce different values."""
        values = [derive(1069, i) for i in range(10)]
        assert len(set(values)) == 10
    
    def test_derive_commutativity(self):
        """XOR is commutative in derive."""
        assert derive(1069, 42) == derive(42, 1069)


class TestGF3:
    """Test GF(3) trit operations."""
    
    def test_trit_range(self):
        """Trits are always in {-1, 0, +1}."""
        for seed in derive_chain(1069, 100):
            trit = seed_to_trit(seed)
            assert trit in (-1, 0, 1)
    
    def test_known_trits(self):
        """Verify specific trit values."""
        assert seed_to_trit(1069) == 0  # 1069 % 3 = 2, 2 - 1 = 1? No: 1069 % 3 = 1, 1-1=0
        assert seed_to_trit(10908445076698799222) == 1  # % 3 = 2, 2-1=1
        assert seed_to_trit(14533190410327020252) == -1  # % 3 = 0, 0-1=-1
    
    def test_triad_conservation(self):
        """Spawned triad always sums to 0 mod 3."""
        for base in [0, 1, 42, 1069, 999999]:
            triad = spawn_triad(base)
            seeds = [triad["minus"], triad["zero"], triad["plus"]]
            assert check_gf3(seeds), f"Triad not balanced for seed {base}"
    
    def test_triad_trit_values(self):
        """Triad contains exactly one of each trit."""
        triad = spawn_triad(1069)
        assert seed_to_trit(triad["minus"]) == -1
        assert seed_to_trit(triad["zero"]) == 0
        assert seed_to_trit(triad["plus"]) == 1


class TestHue:
    """Test hue generation."""
    
    def test_hue_range(self):
        """Hue is always in [0, 360)."""
        for seed in derive_chain(1069, 100):
            hue = seed_to_hue(seed)
            assert 0 <= hue < 360
    
    def test_known_hues(self):
        """Verify specific hue values."""
        assert seed_to_hue(1069) == 349  # 1069 % 360
        assert seed_to_hue(10908445076698799222) == 62
        assert seed_to_hue(14533190410327020252) == 252
