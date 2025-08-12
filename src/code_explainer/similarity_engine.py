"""Advanced code similarity and clustering engine."""

import logging
import ast
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import re

logger = logging.getLogger(__name__)


@dataclass
class CodeSnippet:
    """Represents a code snippet with metadata."""
    code: str
    ast_features: Dict[str, Any]
    semantic_features: Dict[str, Any]
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None


@dataclass
class SimilarityResult:
    """Represents a similarity comparison result."""
    snippet1_id: str
    snippet2_id: str
    similarity_score: float
    similarity_type: str
    details: Dict[str, Any]


class CodeFeatureExtractor:
    """Extracts various features from code for similarity analysis."""
    
    def __init__(self):
        """Initialize the feature extractor."""
        self.tfidf_vectorizer = TfidfVectorizer(
            token_pattern=r'\b\w+\b',
            ngram_range=(1, 2),
            max_features=1000,
            stop_words='english'
        )
    
    def extract_ast_features(self, code: str) -> Dict[str, Any]:
        """Extract AST-based features from code.
        
        Args:
            code: Python code string
            
        Returns:
            Dictionary of AST features
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return {"valid_syntax": False}
        
        features = {
            "valid_syntax": True,
            "num_nodes": len(list(ast.walk(tree))),
            "num_functions": 0,
            "num_classes": 0,
            "num_imports": 0,
            "num_loops": 0,
            "num_conditionals": 0,
            "num_assignments": 0,
            "max_depth": 0,
            "function_names": [],
            "class_names": [],
            "variable_names": set(),
            "imported_modules": [],
            "complexity_score": 1
        }
        
        # Analyze AST nodes
        current_depth = 0
        max_depth = 0
        
        for node in ast.walk(tree):
            # Track depth
            if hasattr(node, 'lineno'):
                current_depth = getattr(node, '_depth', 0)
                max_depth = max(max_depth, current_depth)
            
            # Count different node types
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                features["num_functions"] += 1
                features["function_names"].append(node.name)
                features["complexity_score"] += self._calculate_function_complexity(node)
            elif isinstance(node, ast.ClassDef):
                features["num_classes"] += 1
                features["class_names"].append(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                features["num_imports"] += 1
                if isinstance(node, ast.Import):
                    features["imported_modules"].extend([alias.name for alias in node.names])
                else:
                    if node.module:
                        features["imported_modules"].append(node.module)
            elif isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
                features["num_loops"] += 1
                features["complexity_score"] += 1
            elif isinstance(node, ast.If):
                features["num_conditionals"] += 1
                features["complexity_score"] += 1
            elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                features["num_assignments"] += 1
                # Extract variable names
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            features["variable_names"].add(target.id)
        
        features["max_depth"] = max_depth
        features["variable_names"] = list(features["variable_names"])
        
        return features
    
    def _calculate_function_complexity(self, func_node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity for a function."""
        complexity = 1
        for node in ast.walk(func_node):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.comprehension):
                complexity += 1
        return complexity
    
    def extract_semantic_features(self, code: str) -> Dict[str, Any]:
        """Extract semantic features from code.
        
        Args:
            code: Python code string
            
        Returns:
            Dictionary of semantic features
        """
        features = {
            "line_count": len(code.split('\n')),
            "char_count": len(code),
            "avg_line_length": 0,
            "comment_ratio": 0,
            "docstring_ratio": 0,
            "keyword_density": {},
            "identifier_patterns": [],
            "string_literals": [],
            "numeric_literals": []
        }
        
        lines = code.split('\n')
        features["line_count"] = len(lines)
        features["char_count"] = len(code)
        
        # Calculate average line length
        non_empty_lines = [line for line in lines if line.strip()]
        if non_empty_lines:
            features["avg_line_length"] = sum(len(line) for line in non_empty_lines) / len(non_empty_lines)
        
        # Count comments and docstrings
        comment_chars = sum(len(re.findall(r'#.*$', line)) for line in lines)
        docstring_chars = len(re.findall(r'""".*?"""', code, re.DOTALL))
        docstring_chars += len(re.findall(r"'''.*?'''", code, re.DOTALL))
        
        total_chars = len(code.replace(' ', '').replace('\n', ''))
        if total_chars > 0:
            features["comment_ratio"] = comment_chars / total_chars
            features["docstring_ratio"] = docstring_chars / total_chars
        
        # Extract keywords and patterns
        python_keywords = [
            'def', 'class', 'if', 'else', 'elif', 'for', 'while', 'try', 'except',
            'import', 'from', 'return', 'yield', 'with', 'as', 'lambda', 'async', 'await'
        ]
        
        for keyword in python_keywords:
            count = len(re.findall(r'\b' + keyword + r'\b', code))
            if count > 0:
                features["keyword_density"][keyword] = count
        
        # Extract identifiers, strings, and numbers
        features["identifier_patterns"] = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code)
        features["string_literals"] = re.findall(r'"[^"]*"|\'[^\']*\'', code)
        features["numeric_literals"] = re.findall(r'\b\d+\.?\d*\b', code)
        
        return features
    
    def extract_all_features(self, code: str, metadata: Optional[Dict[str, Any]] = None) -> CodeSnippet:
        """Extract all features from a code snippet.
        
        Args:
            code: Python code string
            metadata: Optional metadata dictionary
            
        Returns:
            CodeSnippet object with all features
        """
        ast_features = self.extract_ast_features(code)
        semantic_features = self.extract_semantic_features(code)
        
        return CodeSnippet(
            code=code,
            ast_features=ast_features,
            semantic_features=semantic_features,
            metadata=metadata or {}
        )


class CodeSimilarityEngine:
    """Advanced code similarity analysis and clustering engine."""
    
    def __init__(self):
        """Initialize the similarity engine."""
        self.feature_extractor = CodeFeatureExtractor()
        self.snippets: Dict[str, CodeSnippet] = {}
        self.similarity_cache: Dict[Tuple[str, str], float] = {}
    
    def add_snippet(self, snippet_id: str, code: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a code snippet to the engine.
        
        Args:
            snippet_id: Unique identifier for the snippet
            code: Python code string
            metadata: Optional metadata
        """
        snippet = self.feature_extractor.extract_all_features(code, metadata)
        self.snippets[snippet_id] = snippet
        
        # Clear similarity cache when new snippets are added
        self.similarity_cache.clear()
    
    def calculate_ast_similarity(self, snippet1: CodeSnippet, snippet2: CodeSnippet) -> float:
        """Calculate AST-based similarity between two snippets.
        
        Args:
            snippet1: First code snippet
            snippet2: Second code snippet
            
        Returns:
            Similarity score between 0 and 1
        """
        features1 = snippet1.ast_features
        features2 = snippet2.ast_features
        
        # Handle syntax errors
        if not features1.get("valid_syntax") or not features2.get("valid_syntax"):
            return 0.0
        
        # Compare structural features
        similarities = []
        
        # Numeric feature similarity
        numeric_features = [
            "num_functions", "num_classes", "num_imports", "num_loops",
            "num_conditionals", "num_assignments", "max_depth", "complexity_score"
        ]
        
        for feature in numeric_features:
            val1 = features1.get(feature, 0)
            val2 = features2.get(feature, 0)
            if val1 == 0 and val2 == 0:
                similarities.append(1.0)
            else:
                # Normalized similarity
                max_val = max(val1, val2)
                min_val = min(val1, val2)
                similarities.append(min_val / max_val if max_val > 0 else 0.0)
        
        # Name-based similarities
        func_sim = self._calculate_name_similarity(
            features1.get("function_names", []),
            features2.get("function_names", [])
        )
        class_sim = self._calculate_name_similarity(
            features1.get("class_names", []),
            features2.get("class_names", [])
        )
        var_sim = self._calculate_name_similarity(
            features1.get("variable_names", []),
            features2.get("variable_names", [])
        )
        module_sim = self._calculate_name_similarity(
            features1.get("imported_modules", []),
            features2.get("imported_modules", [])
        )
        
        similarities.extend([func_sim, class_sim, var_sim, module_sim])
        
        return np.mean(similarities) if similarities else 0.0
    
    def calculate_semantic_similarity(self, snippet1: CodeSnippet, snippet2: CodeSnippet) -> float:
        """Calculate semantic similarity between two snippets.
        
        Args:
            snippet1: First code snippet
            snippet2: Second code snippet
            
        Returns:
            Similarity score between 0 and 1
        """
        features1 = snippet1.semantic_features
        features2 = snippet2.semantic_features
        
        similarities = []
        
        # Compare size-related features
        size_features = [
            ("line_count", 50),  # Normalize by expected max
            ("char_count", 2000),
            ("avg_line_length", 100)
        ]
        
        for feature, max_val in size_features:
            val1 = min(features1.get(feature, 0), max_val)
            val2 = min(features2.get(feature, 0), max_val)
            if val1 == 0 and val2 == 0:
                similarities.append(1.0)
            else:
                similarities.append(1.0 - abs(val1 - val2) / max_val)
        
        # Compare keyword densities
        keywords1 = features1.get("keyword_density", {})
        keywords2 = features2.get("keyword_density", {})
        all_keywords = set(keywords1.keys()) | set(keywords2.keys())
        
        if all_keywords:
            keyword_similarities = []
            for keyword in all_keywords:
                count1 = keywords1.get(keyword, 0)
                count2 = keywords2.get(keyword, 0)
                if count1 == 0 and count2 == 0:
                    keyword_similarities.append(1.0)
                else:
                    max_count = max(count1, count2)
                    min_count = min(count1, count2)
                    keyword_similarities.append(min_count / max_count if max_count > 0 else 0.0)
            
            similarities.append(np.mean(keyword_similarities))
        
        # Compare identifier patterns (simplified)
        identifiers1 = set(features1.get("identifier_patterns", []))
        identifiers2 = set(features2.get("identifier_patterns", []))
        if identifiers1 or identifiers2:
            jaccard_sim = len(identifiers1 & identifiers2) / len(identifiers1 | identifiers2)
            similarities.append(jaccard_sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_name_similarity(self, names1: List[str], names2: List[str]) -> float:
        """Calculate similarity between two lists of names."""
        if not names1 and not names2:
            return 1.0
        if not names1 or not names2:
            return 0.0
        
        set1 = set(names1)
        set2 = set(names2)
        
        # Jaccard similarity
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_similarity(
        self, 
        snippet1_id: str, 
        snippet2_id: str, 
        weights: Optional[Dict[str, float]] = None
    ) -> SimilarityResult:
        """Calculate overall similarity between two snippets.
        
        Args:
            snippet1_id: ID of first snippet
            snippet2_id: ID of second snippet
            weights: Optional weights for different similarity types
            
        Returns:
            SimilarityResult object
        """
        if weights is None:
            weights = {"ast": 0.6, "semantic": 0.4}
        
        # Check cache
        cache_key = tuple(sorted([snippet1_id, snippet2_id]))
        if cache_key in self.similarity_cache:
            cached_score = self.similarity_cache[cache_key]
            return SimilarityResult(
                snippet1_id=snippet1_id,
                snippet2_id=snippet2_id,
                similarity_score=cached_score,
                similarity_type="cached",
                details={}
            )
        
        snippet1 = self.snippets[snippet1_id]
        snippet2 = self.snippets[snippet2_id]
        
        # Calculate different types of similarity
        ast_sim = self.calculate_ast_similarity(snippet1, snippet2)
        semantic_sim = self.calculate_semantic_similarity(snippet1, snippet2)
        
        # Weighted combination
        overall_sim = (
            weights.get("ast", 0.6) * ast_sim +
            weights.get("semantic", 0.4) * semantic_sim
        )
        
        # Cache the result
        self.similarity_cache[cache_key] = overall_sim
        
        return SimilarityResult(
            snippet1_id=snippet1_id,
            snippet2_id=snippet2_id,
            similarity_score=overall_sim,
            similarity_type="combined",
            details={
                "ast_similarity": ast_sim,
                "semantic_similarity": semantic_sim,
                "weights": weights
            }
        )
    
    def find_similar_snippets(
        self, 
        query_id: str, 
        top_k: int = 5, 
        min_similarity: float = 0.1
    ) -> List[SimilarityResult]:
        """Find snippets similar to the query snippet.
        
        Args:
            query_id: ID of the query snippet
            top_k: Number of top results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of SimilarityResult objects, sorted by similarity
        """
        if query_id not in self.snippets:
            raise ValueError(f"Snippet {query_id} not found")
        
        results = []
        for snippet_id in self.snippets:
            if snippet_id != query_id:
                similarity = self.calculate_similarity(query_id, snippet_id)
                if similarity.similarity_score >= min_similarity:
                    results.append(similarity)
        
        # Sort by similarity score (descending)
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return results[:top_k]
    
    def cluster_snippets(self, n_clusters: int = 5) -> Dict[int, List[str]]:
        """Cluster code snippets based on similarity.
        
        Args:
            n_clusters: Number of clusters to create
            
        Returns:
            Dictionary mapping cluster IDs to lists of snippet IDs
        """
        if len(self.snippets) < n_clusters:
            # If we have fewer snippets than clusters, each gets its own cluster
            return {i: [snippet_id] for i, snippet_id in enumerate(self.snippets.keys())}
        
        # Create feature matrix for clustering
        snippet_ids = list(self.snippets.keys())
        n_snippets = len(snippet_ids)
        
        # Calculate pairwise similarities
        similarity_matrix = np.zeros((n_snippets, n_snippets))
        
        for i, id1 in enumerate(snippet_ids):
            for j, id2 in enumerate(snippet_ids):
                if i != j:
                    sim = self.calculate_similarity(id1, id2).similarity_score
                    similarity_matrix[i][j] = sim
                else:
                    similarity_matrix[i][j] = 1.0
        
        # Convert similarity to distance for clustering
        distance_matrix = 1.0 - similarity_matrix
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(distance_matrix)
        
        # Group snippets by cluster
        clusters = {}
        for i, cluster_id in enumerate(cluster_labels):
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(snippet_ids[i])
        
        return clusters
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get analytics about the code corpus.
        
        Returns:
            Dictionary with corpus analytics
        """
        if not self.snippets:
            return {"message": "No snippets in corpus"}
        
        analytics = {
            "total_snippets": len(self.snippets),
            "avg_functions_per_snippet": 0,
            "avg_classes_per_snippet": 0,
            "avg_complexity": 0,
            "most_common_functions": {},
            "most_common_imports": {},
            "language_patterns": {}
        }
        
        total_functions = 0
        total_classes = 0
        total_complexity = 0
        all_function_names = []
        all_imports = []
        
        for snippet in self.snippets.values():
            ast_features = snippet.ast_features
            if ast_features.get("valid_syntax"):
                total_functions += ast_features.get("num_functions", 0)
                total_classes += ast_features.get("num_classes", 0)
                total_complexity += ast_features.get("complexity_score", 1)
                all_function_names.extend(ast_features.get("function_names", []))
                all_imports.extend(ast_features.get("imported_modules", []))
        
        n_snippets = len(self.snippets)
        analytics["avg_functions_per_snippet"] = total_functions / n_snippets
        analytics["avg_classes_per_snippet"] = total_classes / n_snippets
        analytics["avg_complexity"] = total_complexity / n_snippets
        
        # Count frequencies
        from collections import Counter
        function_counts = Counter(all_function_names)
        import_counts = Counter(all_imports)
        
        analytics["most_common_functions"] = dict(function_counts.most_common(10))
        analytics["most_common_imports"] = dict(import_counts.most_common(10))
        
        return analytics
    
    def save_corpus(self, filepath: str) -> None:
        """Save the corpus to a file.
        
        Args:
            filepath: Path to save the corpus
        """
        corpus_data = {
            "snippets": {},
            "analytics": self.get_analytics()
        }
        
        for snippet_id, snippet in self.snippets.items():
            corpus_data["snippets"][snippet_id] = {
                "code": snippet.code,
                "ast_features": snippet.ast_features,
                "semantic_features": snippet.semantic_features,
                "metadata": snippet.metadata
            }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(corpus_data, f, indent=2, default=str)
        
        logger.info(f"Corpus saved to {filepath}")
    
    def load_corpus(self, filepath: str) -> None:
        """Load a corpus from a file.
        
        Args:
            filepath: Path to load the corpus from
        """
        with open(filepath, 'r') as f:
            corpus_data = json.load(f)
        
        self.snippets.clear()
        self.similarity_cache.clear()
        
        for snippet_id, data in corpus_data["snippets"].items():
            snippet = CodeSnippet(
                code=data["code"],
                ast_features=data["ast_features"],
                semantic_features=data["semantic_features"],
                metadata=data["metadata"]
            )
            self.snippets[snippet_id] = snippet
        
        logger.info(f"Loaded {len(self.snippets)} snippets from {filepath}")


def main():
    """CLI entry point for similarity analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Code similarity analysis")
    parser.add_argument("command", choices=["analyze", "cluster", "find"], help="Command to run")
    parser.add_argument("--corpus", required=True, help="Path to code corpus JSON file")
    parser.add_argument("--query", help="Query snippet ID for similarity search")
    parser.add_argument("--clusters", type=int, default=5, help="Number of clusters")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top results")
    parser.add_argument("--output", help="Output file path")
    
    args = parser.parse_args()
    
    engine = CodeSimilarityEngine()
    
    # Load corpus if it exists
    if Path(args.corpus).exists():
        engine.load_corpus(args.corpus)
    else:
        print(f"Corpus file {args.corpus} not found")
        return
    
    if args.command == "analyze":
        analytics = engine.get_analytics()
        print(json.dumps(analytics, indent=2))
    
    elif args.command == "cluster":
        clusters = engine.cluster_snippets(args.clusters)
        print(f"Created {len(clusters)} clusters:")
        for cluster_id, snippet_ids in clusters.items():
            print(f"  Cluster {cluster_id}: {len(snippet_ids)} snippets")
            for snippet_id in snippet_ids[:3]:  # Show first 3
                code_preview = engine.snippets[snippet_id].code[:50].replace('\n', ' ')
                print(f"    {snippet_id}: {code_preview}...")
    
    elif args.command == "find":
        if not args.query:
            print("Query snippet ID required for similarity search")
            return
        
        results = engine.find_similar_snippets(args.query, args.top_k)
        print(f"Top {len(results)} similar snippets to '{args.query}':")
        for result in results:
            print(f"  {result.snippet2_id}: {result.similarity_score:.3f}")


if __name__ == "__main__":
    main()
