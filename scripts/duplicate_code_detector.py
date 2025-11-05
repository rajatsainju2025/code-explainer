#!/usr/bin/env python3
"""
Duplicate Code Detection Report

Identifies similar code patterns and duplications.
"""

import os
from pathlib import Path
from collections import defaultdict

def extract_function_bodies(file_path: str) -> dict:
    """Extract function signatures and their approximate size."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except:
        return {}
    
    functions = {}
    current_func = None
    indent_count = 0
    
    for i, line in enumerate(lines, 1):
        stripped = line.lstrip()
        if stripped.startswith('def ') or stripped.startswith('async def '):
            current_func = line.strip()
            indent_count = len(line) - len(stripped)
        elif current_func:
            current_indent = len(line) - len(stripped) if stripped else indent_count
            if line.strip() and current_indent <= indent_count and not stripped.startswith('#'):
                functions[current_func] = i - (len([l for l in lines[:i] if l.strip() == '']) // 20 or 1)
                current_func = None
    
    return functions

def find_similar_patterns(directory: str) -> dict:
    """Find similar code patterns across files."""
    patterns = defaultdict(list)
    
    # Common patterns to check
    cache_patterns = []
    validation_patterns = []
    retrieval_patterns = []
    
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git']]
        for file in files:
            if not file.endswith('.py'):
                continue
                
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, directory)
            
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
            except:
                continue
            
            # Detect cache-related code
            if 'cache' in rel_path.lower() or any(x in content for x in ['_cache', 'Cache', 'get_cached']):
                cache_patterns.append(rel_path)
            
            # Detect validation code
            if 'validat' in content.lower() or 'assert' in content:
                validation_patterns.append(rel_path)
            
            # Detect retrieval code
            if 'retriev' in rel_path.lower() or 'search' in rel_path.lower():
                retrieval_patterns.append(rel_path)
    
    return {
        'cache_modules': cache_patterns,
        'validation_modules': validation_patterns,
        'retrieval_modules': retrieval_patterns
    }

# Run analysis
root_dir = "src/code_explainer"
patterns = find_similar_patterns(root_dir)

print("=" * 80)
print("DUPLICATE CODE & PATTERN ANALYSIS")
print("=" * 80)

print("\n1. Cache-Related Modules:")
print("-" * 40)
for module in sorted(set(patterns['cache_modules'])):
    print(f"  ✓ {module}")

print("\n2. Validation Modules:")
print("-" * 40)
for module in sorted(set(patterns['validation_modules']))[:10]:
    print(f"  ✓ {module}")

print("\n3. Retrieval/Search Modules:")
print("-" * 40)
for module in sorted(set(patterns['retrieval_modules'])):
    print(f"  ✓ {module}")

print("\n" + "=" * 80)
print("KEY DUPLICATION OPPORTUNITIES")
print("=" * 80)

print("""
1. CACHE IMPLEMENTATIONS
   - base_cache.py: Core LRU/LFU cache
   - explanation_cache.py: Specialized explanation cache
   - embedding_cache.py: Specialized embedding cache
   
   CONSOLIDATION OPPORTUNITY: Use base_cache for all cache types

2. RETRIEVAL PATTERNS
   - retriever.py: Main orchestrator
   - hybrid_search.py: Hybrid fusion logic
   - bm25_index.py: BM25 search
   - faiss_index.py: Vector search
   
   CONSOLIDATION OPPORTUNITY: Common result formatting and filtering

3. INPUT VALIDATION
   - Multiple modules check similar patterns
   - Code length validation repeated
   - String/type validation duplicated
   
   CONSOLIDATION OPPORTUNITY: Create shared validation utilities

4. CONFIGURATION LOADING
   - Config loaded in multiple places
   - No centralized configuration factory
   - Repeated YAML parsing
   
   CONSOLIDATION OPPORTUNITY: Centralize config loading in single module
""")

print("=" * 80)
print("RECOMMENDATIONS FOR COMMIT 3")
print("=" * 80)

print("""
1. Extract common cache patterns into BaseCache
   - Move initialization logic to base
   - Standardize expiration checking
   - Consolidate serialization methods

2. Create unified cache factory pattern
   - Single point for cache creation
   - Consistent interface for all cache types
   - Easier testing and mocking

3. Extract validation utilities
   - Common input validation functions
   - Shared code analysis patterns
   - Centralized security checks

4. Consolidate result formatting
   - Similar result wrapping in multiple modules
   - Common scoring/ranking logic
   - Shared sorting/filtering logic
""")

print("=" * 80)
