#!/usr/bin/env python3
"""
Codebase Analysis Script

Analyzes the codebase to identify:
1. Files that are actually imported/used
2. Dead code that can be safely removed
3. Files with high complexity that should be refactored
4. Dependencies between modules
"""

import os
import re
import ast
import sys
from pathlib import Path
from typing import Dict, Set, List, Tuple, Any
from collections import defaultdict


class CodebaseAnalyzer:
    """Analyzes the codebase structure and dependencies."""
    
    __slots__ = ('root_path', 'modules', 'imports', 'exports', 'file_sizes', 'complexity_scores')

    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.modules: Dict[str, Path] = {}
        self.imports: Dict[str, Set[str]] = defaultdict(set)
        self.exports: Dict[str, Set[str]] = defaultdict(set)
        self.file_sizes: Dict[str, int] = {}
        self.complexity_scores: Dict[str, int] = {}

    def scan_files(self):
        """Scan all Python files in the codebase."""
        for py_file in self.root_path.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            module_name = self._path_to_module(py_file)
            self.modules[module_name] = py_file

            # Analyze file
            self._analyze_file(py_file, module_name)

    def _path_to_module(self, path: Path) -> str:
        """Convert file path to module name."""
        rel_path = path.relative_to(self.root_path)
        return str(rel_path).replace("/", ".").replace("\\", ".").replace(".py", "")

    def _analyze_file(self, file_path: Path, module_name: str):
        """Analyze a single Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            self.file_sizes[module_name] = len(content.splitlines())

            # Parse AST for imports and complexity
            tree = ast.parse(content, filename=str(file_path))
            self._extract_imports(tree, module_name)
            self._extract_exports(tree, module_name)
            self._calculate_complexity(tree, module_name)

        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")

    def _extract_imports(self, tree: ast.AST, module_name: str):
        """Extract import statements from AST."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_module = alias.name
                    if imported_module.startswith('code_explainer'):
                        # Extract the submodule part after 'code_explainer.'
                        submodule = imported_module.split('code_explainer.', 1)[1] if 'code_explainer.' in imported_module else imported_module
                        # Take only the first part after code_explainer
                        submodule = submodule.split('.')[0]
                        self.imports[module_name].add(submodule)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    # For relative imports, node.module is like 'cache', 'model', etc.
                    # For absolute imports, node.module is like 'code_explainer.cache'
                    if node.module.startswith('code_explainer'):
                        submodule = node.module.split('code_explainer.', 1)[1] if 'code_explainer.' in node.module else node.module
                        submodule = submodule.split('.')[0]
                    else:
                        # Relative import or external import
                        submodule = node.module.split('.')[0]
                    self.imports[module_name].add(submodule)

    def _extract_exports(self, tree: ast.AST, module_name: str):
        """Extract __all__ exports and class/function definitions."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == '__all__':
                        if isinstance(node.value, ast.List):
                            for item in node.value.elts:
                                if isinstance(item, ast.Str):
                                    self.exports[module_name].add(item.s)
                                elif isinstance(item, ast.Constant) and isinstance(item.value, str):
                                    self.exports[module_name].add(item.value)
            elif isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                self.exports[module_name].add(node.name)

    def _calculate_complexity(self, tree: ast.AST, module_name: str):
        """Calculate a simple complexity score based on AST nodes."""
        complexity = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
                complexity += 1
            elif isinstance(node, ast.FunctionDef):
                complexity += len(node.args.args)  # Parameter complexity
        self.complexity_scores[module_name] = complexity

    def analyze_usage(self) -> Dict[str, Any]:
        """Analyze which modules are used and which are dead code."""
        used_modules = set()
        all_modules = set(self.modules.keys())

        # Find root entry points
        entry_points = {
            '__init__', 'cli', 'model', 'api.server',
            'trainer', 'research_evaluation_orchestrator'
        }

        # BFS to find all reachable modules
        queue = list(entry_points)
        visited = set()

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            if current in self.imports:
                for imported in self.imports[current]:
                    if imported not in visited:
                        queue.append(imported)
                        # Also add the package __init__ if importing a package
                        package_init = f"{imported}.__init__"
                        if package_init in all_modules and package_init not in visited:
                            queue.append(package_init)

        used_modules = visited.intersection(all_modules)
        dead_modules = all_modules - used_modules

        return {
            'used_modules': used_modules,
            'dead_modules': dead_modules,
            'entry_points': entry_points,
            'total_modules': len(all_modules),
            'used_count': len(used_modules),
            'dead_count': len(dead_modules)
        }

    def get_large_files(self, threshold: int = 500) -> List[Tuple[str, int]]:
        """Get files that exceed the line threshold."""
        return [(module, size) for module, size in self.file_sizes.items() if size > threshold]

    def get_complex_files(self, threshold: int = 50) -> List[Tuple[str, int]]:
        """Get files with high complexity scores."""
        return [(module, score) for module, score in self.complexity_scores.items() if score > threshold]

    def generate_report(self) -> str:
        """Generate a comprehensive analysis report."""
        usage = self.analyze_usage()
        large_files = self.get_large_files()
        complex_files = self.get_complex_files()

        report = ["# Codebase Analysis Report\n"]

        # Summary
        report.append("## Summary\n")
        report.append(f"- **Total modules**: {usage['total_modules']}")
        report.append(f"- **Used modules**: {usage['used_count']}")
        report.append(f"- **Dead modules**: {usage['dead_count']}")
        report.append(f"- **Dead code ratio**: {usage['dead_count']/usage['total_modules']:.1%}")
        report.append("")

        # Dead modules
        if usage['dead_modules']:
            report.append("## Dead Code (Safe to Remove)\n")
            for module in sorted(usage['dead_modules']):
                size = self.file_sizes.get(module, 0)
                report.append(f"- `{module}` ({size} lines)")
            report.append("")

        # Large files
        if large_files:
            report.append("## Large Files (Consider Splitting)\n")
            for module, size in sorted(large_files, key=lambda x: x[1], reverse=True):
                report.append(f"- `{module}`: {size} lines")
            report.append("")

        # Complex files
        if complex_files:
            report.append("## High Complexity Files (Consider Refactoring)\n")
            for module, score in sorted(complex_files, key=lambda x: x[1], reverse=True):
                report.append(f"- `{module}`: complexity score {score}")
            report.append("")

        # Recommendations
        report.append("## Recommendations\n")

        if usage['dead_count'] > 10:
            report.append("### Critical: Remove Dead Code")
            report.append(f"- {usage['dead_count']} modules appear to be unused and can be safely removed")
            report.append("- This would reduce codebase size by ~{:.0f}%".format(
                sum(self.file_sizes.get(m, 0) for m in usage['dead_modules']) /
                sum(self.file_sizes.values()) * 100))
            report.append("")

        if large_files:
            report.append("### High Priority: Split Large Files")
            report.append("- Files over 500 lines should be split into smaller, focused modules")
            report.append("")

        if complex_files:
            report.append("### Medium Priority: Reduce Complexity")
            report.append("- High complexity files should be refactored for better maintainability")
            report.append("")

        return "\n".join(report)


def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_codebase.py <path_to_src>")
        sys.exit(1)

    analyzer = CodebaseAnalyzer(sys.argv[1])
    analyzer.scan_files()
    report = analyzer.generate_report()
    print(report)


if __name__ == "__main__":
    main()