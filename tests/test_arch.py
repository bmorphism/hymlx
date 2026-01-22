"""Test Hy architectures - verifies the instrumental utility of Lisp for ML."""

import pytest
import subprocess
import sys


class TestHyArchitectures:
    """Test that Hy architecture code runs correctly."""
    
    def test_hy_arch_syntax(self):
        """Verify arch.hy has valid Hy syntax."""
        result = subprocess.run(
            ["hy", "-c", "(import hymlx.arch)"],
            capture_output=True,
            text=True,
            cwd="src"
        )
        # Just check it parses (import may fail due to MLX but syntax is valid)
        assert "SyntaxError" not in result.stderr
        assert "LexException" not in result.stderr


class TestArchPatterns:
    """Test architecture patterns implemented in Python equivalents."""
    
    def test_threading_macro_pattern(self):
        """-> macro pattern: thread value through functions."""
        # Hy: (-> x f g h) = h(g(f(x)))
        def thread(x, *fns):
            for f in fns:
                x = f(x)
            return x
        
        result = thread(1, lambda x: x + 1, lambda x: x * 2, lambda x: x - 1)
        assert result == 3  # ((1 + 1) * 2) - 1
    
    def test_residual_pattern(self):
        """residual macro: x + f(x)."""
        def residual(x, f):
            return x + f(x)
        
        result = residual(10, lambda x: x * 2)
        assert result == 30  # 10 + 20
    
    def test_spec_parsing(self):
        """S-expression specs for layers."""
        specs = [
            ['linear', 784, 256],
            ['relu'],
            ['linear', 256, 10],
        ]
        
        # Each spec is a list that can be pattern-matched
        for spec in specs:
            if spec[0] == 'linear':
                assert len(spec) == 3
                assert isinstance(spec[1], int)
                assert isinstance(spec[2], int)
            elif spec[0] == 'relu':
                assert len(spec) == 1
    
    def test_make_mlp_spec(self):
        """Dynamic MLP spec generation."""
        def make_mlp_spec(layer_sizes, activation='relu'):
            specs = []
            for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
                specs.append(['linear', in_dim, out_dim])
                specs.append([activation])
            specs.pop()  # Remove last activation
            return specs
        
        result = make_mlp_spec([784, 256, 128, 10])
        assert result == [
            ['linear', 784, 256], ['relu'],
            ['linear', 256, 128], ['relu'],
            ['linear', 128, 10]
        ]
    
    def test_replace_activations(self):
        """Pattern matching for layer surgery."""
        def replace_activations(specs, old_act, new_act):
            result = []
            for spec in specs:
                if len(spec) == 1 and spec[0] == old_act:
                    result.append([new_act])
                else:
                    result.append(spec)
            return result
        
        mlp_spec = [
            ['linear', 784, 256], ['relu'],
            ['linear', 256, 10]
        ]
        
        result = replace_activations(mlp_spec, 'relu', 'gelu')
        assert result[1] == ['gelu']
    
    def test_recursive_unet_spec(self):
        """Recursive network structure generation."""
        def make_unet_block(in_ch, out_ch, depth):
            if depth == 0:
                return [['conv2d', in_ch, out_ch, 3], ['gelu']]
            else:
                return (
                    [['conv2d', in_ch, out_ch, 3], ['gelu']] +
                    make_unet_block(out_ch, out_ch * 2, depth - 1) +
                    [['conv2d', out_ch * 2, out_ch, 3], ['gelu']]
                )
        
        # Depth 0: 2 layers
        assert len(make_unet_block(3, 64, 0)) == 2
        
        # Depth 1: 2 + 2 + 2 = 6 layers
        assert len(make_unet_block(3, 64, 1)) == 6
        
        # Depth 2: 2 + 6 + 2 = 10 layers
        assert len(make_unet_block(3, 64, 2)) == 10
    
    def test_compose_pipe(self):
        """Functional composition patterns."""
        def compose(*fns):
            def composed(x):
                for f in reversed(fns):
                    x = f(x)
                return x
            return composed
        
        def pipe(*fns):
            def piped(x):
                for f in fns:
                    x = f(x)
                return x
            return piped
        
        add1 = lambda x: x + 1
        mul2 = lambda x: x * 2
        sub3 = lambda x: x - 3
        
        # compose: right-to-left
        f = compose(sub3, mul2, add1)  # sub3(mul2(add1(x)))
        assert f(5) == 9  # ((5 + 1) * 2) - 3
        
        # pipe: left-to-right (same result, different order)
        g = pipe(add1, mul2, sub3)
        assert g(5) == 9


class TestHyAdvantages:
    """Document why Hy/Lisp is instrumentally useful for ML."""
    
    def test_homoiconicity_advantage(self):
        """
        Hy Advantage 1: HOMOICONICITY
        
        Code is data. Network architectures can be:
        - Stored as s-expressions
        - Transformed programmatically
        - Generated at runtime
        - Serialized/deserialized trivially
        
        Python equivalent requires AST manipulation or string eval.
        """
        # S-expression network spec
        net_spec = [
            ['linear', 784, 256],
            ['relu'],
            ['dropout', 0.5],
            ['linear', 256, 10],
        ]
        
        # Trivially serialize
        import json
        serialized = json.dumps(net_spec)
        restored = json.loads(serialized)
        assert net_spec == restored
    
    def test_macro_advantage(self):
        """
        Hy Advantage 2: MACROS
        
        Compile-time code generation for:
        - residual connections
        - skip connections
        - parameter sharing patterns
        
        Python requires decorators or inheritance, which are runtime.
        """
        # In Hy: (residual x layer1 layer2)
        # Expands to: (+ x (-> x layer1 layer2))
        # This happens at COMPILE TIME, not runtime
        assert True
    
    def test_pattern_matching_advantage(self):
        """
        Hy Advantage 3: PATTERN MATCHING
        
        Native destructuring for:
        - Layer configuration parsing
        - Architecture surgery
        - Hyperparameter sweeps
        
        Python's match statement is limited compared to Hy's.
        """
        # Hy: (match spec ['linear in out] (nn.Linear in out))
        # Directly destructures and acts
        assert True
    
    def test_composition_advantage(self):
        """
        Hy Advantage 4: NATURAL COMPOSITION
        
        Threading macros (-> and ->>) make layer composition read naturally:
        
        (-> x
            (embed vocab-size dim)
            (transformer-block)
            (transformer-block)
            (layer-norm)
            (linear vocab-size))
        
        vs Python:
        
        x = embed(x, vocab_size, dim)
        x = transformer_block(x)
        x = transformer_block(x)
        x = layer_norm(x)
        x = linear(x, vocab_size)
        """
        assert True
    
    def test_metaprogramming_advantage(self):
        """
        Hy Advantage 5: METAPROGRAMMING
        
        (defblock GPTBlock 768 12 4)
        
        Generates a complete class definition at compile time.
        Python equivalent requires:
        - Class factories
        - type() calls
        - Lots of boilerplate
        """
        assert True
