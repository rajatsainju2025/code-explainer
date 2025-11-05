#!/usr/bin/env python3
"""
Import Cleanup Audit Report

Comprehensive analysis of imports across the codebase.
"""

import os
import ast
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, List, Tuple

def analyze_file_imports(file_path: str) -> Dict[str, any]:
    """Analyze imports in a Python file."""
    try:
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())
    except:
        return None
    
    imports = []
    import_from = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                import_from.append(f"{node.module}.{alias.name}" if node.module else alias.name)
    
    return {"imports": imports, "from_imports": import_from}

def scan_directory(directory: str) -> Dict[str, List[str]]:
    """Scan all Python files in directory."""
    results = {}
    for root, dirs, files in os.walk(directory):
        # Skip __pycache__ and .git
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', '.pytest_cache']]
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, directory)
                analysis = analyze_file_imports(file_path)
                if analysis:
                    all_imports = analysis["imports"] + analysis["from_imports"]
                    results[rel_path] = all_imports
    
    return results

# Run analysis
root_dir = "src/code_explainer"
results = scan_directory(root_dir)

# Statistics
print("=" * 80)
print("IMPORT CLEANUP AUDIT REPORT")
print("=" * 80)

all_imports = defaultdict(int)
for file_imports in results.values():
    for imp in file_imports:
        all_imports[imp] += 1

# Find most common imports
print("\nMost Common Imports:")
print("-" * 40)
for imp, count in sorted(all_imports.items(), key=lambda x: x[1], reverse=True)[:15]:
    print(f"  {imp}: {count} files")

# Find duplicate imports
print("\n\nPotential Import Issues:")
print("-" * 40)

# Check for redundant imports
logging_imports = []
typing_imports = []
for file_path, imports in results.items():
    for imp in imports:
        if 'logging' in imp:
            logging_imports.append(file_path)
        if 'typing' in imp:
            typing_imports.append(file_path)

print(f"\n✓ Logging used in {len(set(logging_imports))} files")
print(f"✓ Typing used in {len(set(typing_imports))} files")

# Find files with many imports
print("\n\nLarge Import Sets (> 10 imports per file):")
print("-" * 40)
large_imports = [(f, len(i)) for f, i in results.items() if len(i) > 10]
for file_path, count in sorted(large_imports, key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {file_path}: {count} imports")

print("\n" + "=" * 80)
print("Summary Statistics:")
print(f"  Total Python files: {len(results)}")
print(f"  Unique import modules: {len(all_imports)}")
print(f"  Average imports per file: {sum(len(i) for i in results.values()) / len(results):.1f}")
print("=" * 80)
