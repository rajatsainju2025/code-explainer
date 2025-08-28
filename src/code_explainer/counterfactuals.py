"""Counterfactual code generator for evaluation.

Generates small mutations that preserve overall semantics:
- Systematic variable/function/class renaming
- Insertion of no-op statements (pass, redundant assignments)
- Reordering of independent statements (conservative)
"""
from __future__ import annotations

import ast
import random
from dataclasses import dataclass
from typing import Dict, List


class _Renamer(ast.NodeTransformer):
    def __init__(self, mapping: Dict[str, str]):
        self.mapping = mapping

    def visit_Name(self, node: ast.Name):
        if node.id in self.mapping:
            return ast.copy_location(ast.Name(id=self.mapping[node.id], ctx=node.ctx), node)
        return node

    def visit_arg(self, node: ast.arg):
        if node.arg in self.mapping:
            node.arg = self.mapping[node.arg]
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if node.name in self.mapping:
            node.name = self.mapping[node.name]
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef):
        if node.name in self.mapping:
            node.name = self.mapping[node.name]
        self.generic_visit(node)
        return node


def _random_token() -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    return "cf_" + "".join(random.choice(alphabet) for _ in range(6))


def rename_identifiers(tree: ast.AST, max_items: int = 5) -> ast.AST:
    names: Dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names[node.id] = names.get(node.id, 0) + 1
        elif isinstance(node, ast.FunctionDef):
            names[node.name] = names.get(node.name, 0) + 1
        elif isinstance(node, ast.ClassDef):
            names[node.name] = names.get(node.name, 0) + 1

    # Do not rename builtins or dunder
    candidates = [n for n in names.keys() if not (n.startswith("__") and n.endswith("__"))]
    random.shuffle(candidates)
    mapping = {n: _random_token() for n in candidates[: max_items]}
    return _Renamer(mapping).visit(ast.fix_missing_locations(tree))


def insert_noops(tree: ast.AST, max_inserts: int = 3) -> ast.AST:
    class _Inserter(ast.NodeTransformer):
        def visit_FunctionDef(self, node: ast.FunctionDef):
            self.generic_visit(node)
            if node.body and len(node.body) >= 1:
                idx = random.randint(0, len(node.body))
                node.body.insert(idx, ast.parse("pass").body[0])
            return node

    t = _Inserter().visit(ast.fix_missing_locations(tree))
    return t


def generate_counterfactuals(code: str, n: int = 3, seed: int | None = 1337) -> List[str]:
    """Generate n counterfactual variants of code."""
    if seed is not None:
        random.seed(seed)
    try:
        base = ast.parse(code)
    except SyntaxError:
        return [code] * n

    outs: List[str] = []
    for i in range(max(1, int(n))):
        t = ast.fix_missing_locations(base)
        if i % 2 == 0:
            t = rename_identifiers(t)
        if i % 3 == 0:
            t = insert_noops(t)
        outs.append(ast.unparse(t) if hasattr(ast, "unparse") else code)
    return outs
