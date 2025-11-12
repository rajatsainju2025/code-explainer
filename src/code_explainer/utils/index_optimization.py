"""Index optimization utilities for fast lookups and filtering.

This module provides efficient indexing structures for large datasets
to enable O(1) or O(log n) lookups instead of O(n) scans.
"""

from typing import Any, Dict, List, Set, Optional, Tuple, Callable
import bisect
import threading


class InvertedIndex:
    """Inverted index for fast reverse lookups."""
    
    __slots__ = ('_forward', '_reverse', '_lock')
    
    def __init__(self):
        """Initialize inverted index."""
        self._forward: Dict[Any, Set[Any]] = {}  # key -> {values}
        self._reverse: Dict[Any, Set[Any]] = {}  # value -> {keys}
        self._lock = threading.RLock()
    
    def add(self, key: Any, value: Any) -> None:
        """Add key-value pair."""
        with self._lock:
            if key not in self._forward:
                self._forward[key] = set()
            self._forward[key].add(value)
            
            if value not in self._reverse:
                self._reverse[value] = set()
            self._reverse[value].add(key)
    
    def get_by_key(self, key: Any) -> Set[Any]:
        """Get all values for key."""
        with self._lock:
            return self._forward.get(key, set()).copy()
    
    def get_by_value(self, value: Any) -> Set[Any]:
        """Get all keys for value (reverse lookup)."""
        with self._lock:
            return self._reverse.get(value, set()).copy()
    
    def remove(self, key: Any, value: Any) -> None:
        """Remove key-value pair."""
        with self._lock:
            if key in self._forward:
                self._forward[key].discard(value)
            if value in self._reverse:
                self._reverse[value].discard(key)
    
    def clear(self) -> None:
        """Clear index."""
        with self._lock:
            self._forward.clear()
            self._reverse.clear()


class RangeIndex:
    """Efficient range queries using sorted lists."""
    
    __slots__ = ('_data', '_lock')
    
    def __init__(self):
        """Initialize range index."""
        self._data: List[Tuple[Any, Any]] = []  # [(key, value), ...]
        self._lock = threading.RLock()
    
    def add(self, key: Any, value: Any) -> None:
        """Add key-value pair (maintains sorted order)."""
        with self._lock:
            pos = bisect.bisect_left(self._data, (key, value))
            self._data.insert(pos, (key, value))
    
    def range_query(self, key_min: Any, key_max: Any) -> List[Tuple[Any, Any]]:
        """Query range of keys.
        
        Returns:
            List of (key, value) tuples in range
        """
        with self._lock:
            left = bisect.bisect_left(self._data, (key_min, None))
            right = bisect.bisect_right(self._data, (key_max, None))
            return self._data[left:right]
    
    def get_values_by_key_range(self, key_min: Any, key_max: Any) -> List[Any]:
        """Get values in key range."""
        return [v for k, v in self.range_query(key_min, key_max)]


class BloomFilter:
    """Probabilistic set membership test (fast negative checks)."""
    
    __slots__ = ('_bits', '_size', '_hash_count', '_lock')
    
    def __init__(self, size: int = 10000, hash_count: int = 3):
        """Initialize bloom filter.
        
        Args:
            size: Size of bit array
            hash_count: Number of hash functions
        """
        self._bits = bytearray(size // 8 + 1)
        self._size = size
        self._hash_count = hash_count
        self._lock = threading.RLock()
    
    def _hash(self, item: Any, seed: int) -> int:
        """Generate hash for item."""
        return (hash(item) ^ seed) % self._size
    
    def add(self, item: Any) -> None:
        """Add item to filter."""
        with self._lock:
            for i in range(self._hash_count):
                idx = self._hash(item, i)
                byte_idx = idx // 8
                bit_idx = idx % 8
                self._bits[byte_idx] |= (1 << bit_idx)
    
    def might_contain(self, item: Any) -> bool:
        """Check if item might be in set (can have false positives)."""
        with self._lock:
            for i in range(self._hash_count):
                idx = self._hash(item, i)
                byte_idx = idx // 8
                bit_idx = idx % 8
                if not (self._bits[byte_idx] & (1 << bit_idx)):
                    return False
        return True
    
    def clear(self) -> None:
        """Clear filter."""
        with self._lock:
            self._bits = bytearray(len(self._bits))


class TrieIndex:
    """Trie (prefix tree) for prefix-based searches."""
    
    __slots__ = ('_root', '_lock')
    
    def __init__(self):
        """Initialize trie."""
        self._root: Dict = {}
        self._lock = threading.RLock()
    
    def add(self, key: str, value: Any) -> None:
        """Add string key with value."""
        with self._lock:
            node = self._root
            for char in key:
                if char not in node:
                    node[char] = {}
                node = node[char]
            node['$value'] = value
    
    def find_prefix(self, prefix: str) -> List[Any]:
        """Find all values with given prefix.
        
        Returns:
            List of values matching prefix
        """
        with self._lock:
            node = self._root
            for char in prefix:
                if char not in node:
                    return []
                node = node[char]
            
            # Collect all values under this node
            results = []
            self._collect_values(node, results)
            return results
    
    def _collect_values(self, node: Dict, results: List) -> None:
        """Recursively collect values from node."""
        if '$value' in node:
            results.append(node['$value'])
        for key, child in node.items():
            if key != '$value' and isinstance(child, dict):
                self._collect_values(child, results)
    
    def clear(self) -> None:
        """Clear trie."""
        with self._lock:
            self._root.clear()


class MultiFieldIndex:
    """Index supporting queries on multiple fields."""
    
    __slots__ = ('_indices', '_lock')
    
    def __init__(self):
        """Initialize multi-field index."""
        self._indices: Dict[str, Dict[Any, Set[int]]] = {}
        self._lock = threading.RLock()
    
    def add_index_field(self, field_name: str) -> None:
        """Add indexable field.
        
        Args:
            field_name: Field name to index
        """
        with self._lock:
            if field_name not in self._indices:
                self._indices[field_name] = {}
    
    def index_document(self, doc_id: int, field_name: str, value: Any) -> None:
        """Index document field value.
        
        Args:
            doc_id: Document ID
            field_name: Field name
            value: Field value
        """
        with self._lock:
            if field_name in self._indices:
                if value not in self._indices[field_name]:
                    self._indices[field_name][value] = set()
                self._indices[field_name][value].add(doc_id)
    
    def search(self, field_name: str, value: Any) -> Set[int]:
        """Search for document IDs matching field value.
        
        Returns:
            Set of matching document IDs
        """
        with self._lock:
            if field_name in self._indices:
                return self._indices[field_name].get(value, set()).copy()
        return set()
    
    def multi_search(self, queries: Dict[str, Any]) -> Set[int]:
        """Search with multiple field constraints (AND logic).
        
        Args:
            queries: Dict mapping field_name -> value
            
        Returns:
            Intersection of matching document IDs
        """
        results = None
        
        for field_name, value in queries.items():
            doc_ids = self.search(field_name, value)
            if results is None:
                results = doc_ids
            else:
                results = results.intersection(doc_ids)
        
        return results or set()
    
    def clear(self) -> None:
        """Clear all indices."""
        with self._lock:
            self._indices.clear()


# Global instances
_inverted_index = InvertedIndex()
_bloom_filter = BloomFilter(size=100000, hash_count=3)


def get_inverted_index() -> InvertedIndex:
    """Get global inverted index."""
    return _inverted_index


def get_bloom_filter() -> BloomFilter:
    """Get global bloom filter."""
    return _bloom_filter
