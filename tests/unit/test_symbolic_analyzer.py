"""Tests for symbolic analyzer optimizations."""

from code_explainer.symbolic.analyzer import (
    SymbolicAnalyzer,
    _MIN_CODE_LENGTH_FOR_CACHE,
    _MAX_CODE_LENGTH_FOR_CACHE,
    _DEFAULT_AST_CACHE_SIZE,
)


class TestSymbolicAnalyzerConstants:
    """Tests for symbolic analyzer constants."""
    
    def test_cache_constants_defined(self):
        """Test that cache constants are properly defined."""
        assert _MIN_CODE_LENGTH_FOR_CACHE == 50
        assert _MAX_CODE_LENGTH_FOR_CACHE == 5000
        assert _DEFAULT_AST_CACHE_SIZE == 256
    
    def test_cache_constants_are_integers(self):
        """Test that cache constants are integers."""
        assert isinstance(_MIN_CODE_LENGTH_FOR_CACHE, int)
        assert isinstance(_MAX_CODE_LENGTH_FOR_CACHE, int)
        assert isinstance(_DEFAULT_AST_CACHE_SIZE, int)
    
    def test_cache_limit_initialized(self):
        """Test that analyzer cache limit is initialized correctly."""
        analyzer = SymbolicAnalyzer()
        assert analyzer._cache_size_limit == _DEFAULT_AST_CACHE_SIZE


class TestSymbolicAnalyzerCaching:
    """Tests for AST caching behavior."""
    
    def test_short_code_not_cached(self):
        """Test that very short code is not cached."""
        analyzer = SymbolicAnalyzer()
        
        short_code = "x = 1"  # Less than MIN_CODE_LENGTH_FOR_CACHE
        assert len(short_code) < _MIN_CODE_LENGTH_FOR_CACHE
        
        analyzer.analyze_code(short_code)
        # Cache should still be empty for short code
        assert len(analyzer._ast_cache) == 0
    
    def test_longer_code_cached(self):
        """Test that longer code is cached."""
        analyzer = SymbolicAnalyzer()
        
        # Create code longer than minimum cache length
        longer_code = "def function_with_many_lines():\n" + "    x = 1\n" * 10
        assert len(longer_code) >= _MIN_CODE_LENGTH_FOR_CACHE
        
        analyzer.analyze_code(longer_code)
        # Cache should have one entry
        assert len(analyzer._ast_cache) == 1
    
    def test_very_long_code_not_cached(self):
        """Test that very long code is not cached."""
        analyzer = SymbolicAnalyzer()
        
        # Create code longer than max cache length
        very_long_code = "x = 1\n" * (_MAX_CODE_LENGTH_FOR_CACHE // 6 + 100)
        assert len(very_long_code) >= _MAX_CODE_LENGTH_FOR_CACHE
        
        analyzer.analyze_code(very_long_code)
        # Cache should still be empty for very long code
        assert len(analyzer._ast_cache) == 0
    
    def test_cache_hit(self):
        """Test that repeated analysis uses cache."""
        analyzer = SymbolicAnalyzer()
        
        code = "def function_with_body():\n    return 42\n" * 3
        assert len(code) >= _MIN_CODE_LENGTH_FOR_CACHE
        
        # First analysis - should cache
        result1 = analyzer.analyze_code(code)
        cache_size_after_first = len(analyzer._ast_cache)
        
        # Second analysis - should hit cache
        result2 = analyzer.analyze_code(code)
        cache_size_after_second = len(analyzer._ast_cache)
        
        # Cache size should not change
        assert cache_size_after_first == cache_size_after_second == 1

    def test_cache_hit_refreshes_lru_position(self):
        """Test that cache hits refresh entry recency for eviction."""
        analyzer = SymbolicAnalyzer()
        analyzer._cache_size_limit = 2

        code_a = "def function_a():\n" + "    x = 1\n" * 10
        code_b = "def function_b():\n" + "    y = 2\n" * 10
        code_c = "def function_c():\n" + "    z = 3\n" * 10

        ast_a = analyzer._get_or_parse_ast(code_a)
        ast_b = analyzer._get_or_parse_ast(code_b)
        analyzer._get_or_parse_ast(code_a)
        ast_c = analyzer._get_or_parse_ast(code_c)

        assert len(analyzer._ast_cache) == 2
        assert ast_a in analyzer._ast_cache.values()
        assert ast_c in analyzer._ast_cache.values()
        assert ast_b not in analyzer._ast_cache.values()


class TestSymbolicAnalyzerSlots:
    """Tests for __slots__ optimization."""
    
    def test_analyzer_has_slots(self):
        """Test that SymbolicAnalyzer uses __slots__."""
        assert hasattr(SymbolicAnalyzer, '__slots__')
    
    def test_analyzer_slots_contents(self):
        """Test that analyzer slots contain expected attributes."""
        slots = SymbolicAnalyzer.__slots__
        assert 'control_flow' in slots
        assert 'variable_assignments' in slots
        assert '_ast_cache' in slots
        assert '_cache_size_limit' in slots
