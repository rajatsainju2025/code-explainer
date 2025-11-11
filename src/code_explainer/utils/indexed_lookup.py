"""Index-based lookup optimization utilities."""

from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict


class IndexedLookup:
    """Build and maintain indexes for fast lookups."""
    
    def __init__(self):
        """Initialize indexed lookup."""
        self.indexes: Dict[str, Dict[Any, List[int]]] = defaultdict(lambda: defaultdict(list))
        self.data: List[Any] = []
    
    def add(self, item: Any, **index_fields):
        """Add item and build indices.
        
        Args:
            item: Item to add
            **index_fields: Fields to index (name=value)
        """
        idx = len(self.data)
        self.data.append(item)
        
        # Build indices
        for field_name, value in index_fields.items():
            self.indexes[field_name][value].append(idx)
    
    def lookup(self, field: str, value: Any) -> List[Any]:
        """Look up items by indexed field.
        
        Args:
            field: Field name
            value: Value to find
        
        Returns:
            List of items matching the criteria
        """
        indices = self.indexes.get(field, {}).get(value, [])
        return [self.data[i] for i in indices]
    
    def lookup_range(self, field: str, min_val: Any, max_val: Any) -> List[Any]:
        """Look up items in range (for comparable values).
        
        Args:
            field: Field name
            min_val: Minimum value
            max_val: Maximum value
        
        Returns:
            List of items within range
        """
        results = []
        for value, indices in self.indexes.get(field, {}).items():
            if min_val <= value <= max_val:
                results.extend([self.data[i] for i in indices])
        return results
    
    def count(self, field: str, value: Any) -> int:
        """Count items with field value.
        
        Args:
            field: Field name
            value: Value to count
        
        Returns:
            Count of matching items
        """
        return len(self.indexes.get(field, {}).get(value, []))


class CachedIndexer:
    """Cache indices of values for quick filtering."""
    
    def __init__(self, items: List[Any], key_func):
        """Initialize cached indexer.
        
        Args:
            items: Items to index
            key_func: Function to extract key from item
        """
        self.items = items
        self.key_func = key_func
        # Build index at creation time
        self.index: Dict[Any, List[int]] = defaultdict(list)
        for i, item in enumerate(items):
            key = key_func(item)
            self.index[key].append(i)
    
    def find_all(self, key: Any) -> List[Any]:
        """Find all items with given key.
        
        Args:
            key: Key to search
        
        Returns:
            List of items with matching key
        """
        indices = self.index.get(key, [])
        return [self.items[i] for i in indices]
    
    def find_first(self, key: Any) -> Optional[Any]:
        """Find first item with given key.
        
        Args:
            key: Key to search
        
        Returns:
            First matching item or None
        """
        indices = self.index.get(key, [])
        return self.items[indices[0]] if indices else None
    
    def exists(self, key: Any) -> bool:
        """Check if key exists.
        
        Args:
            key: Key to check
        
        Returns:
            True if key exists
        """
        return key in self.index


class MultiKeyIndex:
    """Index with multiple keys for complex queries."""
    
    def __init__(self):
        """Initialize multi-key index."""
        self.items: List[Dict[str, Any]] = []
        self.indices: Dict[str, Dict[Any, List[int]]] = defaultdict(lambda: defaultdict(list))
    
    def add(self, item_dict: Dict[str, Any]):
        """Add item with indexed fields.
        
        Args:
            item_dict: Dictionary with item data
        """
        idx = len(self.items)
        self.items.append(item_dict)
        
        # Index all fields
        for field, value in item_dict.items():
            self.indices[field][value].append(idx)
    
    def query(self, **filters) -> List[Dict[str, Any]]:
        """Query items with multiple filters (AND logic).
        
        Args:
            **filters: Field=value filters
        
        Returns:
            Items matching all filters
        """
        if not filters:
            return self.items
        
        # Start with first filter results
        field, value = next(iter(filters.items()))
        result_indices = set(self.indices[field].get(value, []))
        
        # Intersect with other filters
        for field, value in list(filters.items())[1:]:
            filter_indices = set(self.indices[field].get(value, []))
            result_indices &= filter_indices
        
        return [self.items[i] for i in result_indices]
    
    def count(self, **filters) -> int:
        """Count items matching filters.
        
        Args:
            **filters: Field=value filters
        
        Returns:
            Count of matching items
        """
        return len(self.query(**filters))
