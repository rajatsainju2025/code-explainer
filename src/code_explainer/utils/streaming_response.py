"""Streaming response builders for memory-efficient large response handling.

This module provides streaming and incremental response building to reduce
memory footprint for large responses.
"""

import json
import io
from typing import Any, Dict, List, Optional, Iterator
from dataclasses import dataclass, asdict
import threading


@dataclass
class StreamingMetadata:
    """Metadata for streaming responses."""
    total_items: int
    current_item: int
    is_streaming: bool
    buffer_size: int


class IncrementalJSONBuilder:
    """Builds large JSON responses incrementally without loading entire structure."""
    
    __slots__ = ('_buffer', '_lock', '_flush_threshold', '_indent')
    
    def __init__(self, flush_threshold: int = 8192, indent: Optional[int] = None):
        """Initialize incremental JSON builder.
        
        Args:
            flush_threshold: Buffer size before automatic flush
            indent: JSON indentation level (None for compact)
        """
        self._buffer = io.StringIO()
        self._lock = threading.RLock()
        self._flush_threshold = flush_threshold
        self._indent = indent
    
    def append_object(self, obj: Dict[str, Any]) -> None:
        """Append an object to the JSON stream."""
        with self._lock:
            json_str = json.dumps(obj, separators=(',', ':'), indent=self._indent)
            self._buffer.write(json_str)
            
            if self._buffer.tell() > self._flush_threshold:
                self._buffer.write('\n')  # Add newline for streaming
    
    def append_array_item(self, item: Any) -> None:
        """Append item to JSON array."""
        with self._lock:
            json_str = json.dumps(item, separators=(',', ':'), indent=self._indent)
            self._buffer.write(json_str)
            self._buffer.write(',')
    
    def get_content(self) -> str:
        """Get accumulated content."""
        with self._lock:
            return self._buffer.getvalue()
    
    def reset(self) -> None:
        """Reset buffer for reuse."""
        with self._lock:
            self._buffer.seek(0)
            self._buffer.truncate()
    
    def get_size(self) -> int:
        """Get current buffer size in bytes."""
        with self._lock:
            return len(self._buffer.getvalue())


class StreamingResponseBuilder:
    """Builds streaming responses with chunked output."""
    
    __slots__ = ('_chunks', '_lock', '_chunk_size', '_metadata')
    
    def __init__(self, chunk_size: int = 4096):
        """Initialize streaming response builder.
        
        Args:
            chunk_size: Size of chunks in bytes
        """
        self._chunks: List[bytes] = []
        self._lock = threading.RLock()
        self._chunk_size = chunk_size
        self._metadata: Optional[StreamingMetadata] = None
    
    def add_chunk(self, data: Any, is_json: bool = True) -> None:
        """Add data chunk to response.
        
        Args:
            data: Data to add
            is_json: Whether to serialize as JSON
        """
        with self._lock:
            if is_json:
                content = json.dumps(data, separators=(',', ':'))
            else:
                content = str(data)
            
            chunk = content.encode('utf-8')
            self._chunks.append(chunk)
    
    def add_string_chunk(self, data: str) -> None:
        """Add string chunk directly."""
        with self._lock:
            self._chunks.append(data.encode('utf-8'))
    
    def get_chunks(self) -> Iterator[bytes]:
        """Get iterator over accumulated chunks."""
        with self._lock:
            for chunk in self._chunks:
                yield chunk
    
    def get_all_bytes(self) -> bytes:
        """Get all content as bytes."""
        with self._lock:
            return b''.join(self._chunks)
    
    def reset(self) -> None:
        """Reset builder for reuse."""
        with self._lock:
            self._chunks.clear()
    
    def get_size(self) -> int:
        """Get total size in bytes."""
        with self._lock:
            return sum(len(chunk) for chunk in self._chunks)


class LazyResponseBuilder:
    """Builds responses lazily, deferring computation until needed."""
    
    __slots__ = ('_builders', '_lock', '_computed')
    
    def __init__(self):
        """Initialize lazy response builder."""
        self._builders: Dict[str, callable] = {}
        self._lock = threading.RLock()
        self._computed: Dict[str, Any] = {}
    
    def add_lazy_field(self, key: str, builder_fn: callable) -> 'LazyResponseBuilder':
        """Add a lazily-computed field.
        
        Args:
            key: Field name
            builder_fn: Callable that returns field value
        """
        with self._lock:
            self._builders[key] = builder_fn
        return self
    
    def add_eager_field(self, key: str, value: Any) -> 'LazyResponseBuilder':
        """Add an eagerly-computed field.
        
        Args:
            key: Field name
            value: Field value
        """
        with self._lock:
            self._computed[key] = value
        return self
    
    def build(self) -> Dict[str, Any]:
        """Build the complete response, computing lazy fields."""
        with self._lock:
            result = self._computed.copy()
            
            # Compute lazy fields on demand
            for key, builder_fn in self._builders.items():
                if key not in result:
                    result[key] = builder_fn()
            
            return result
    
    def build_partial(self, keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """Build partial response with only specified keys.
        
        Args:
            keys: List of keys to include. If None, include all.
        """
        with self._lock:
            result = {}
            
            # Add computed fields
            if keys is None:
                result.update(self._computed)
            else:
                for key in keys:
                    if key in self._computed:
                        result[key] = self._computed[key]
            
            # Compute requested lazy fields
            for key, builder_fn in self._builders.items():
                if keys is None or key in keys:
                    result[key] = builder_fn()
            
            return result
    
    def reset(self) -> None:
        """Reset builder for reuse."""
        with self._lock:
            self._builders.clear()
            self._computed.clear()


class CompressedJSONBuilder:
    """Builds JSON with compression hints for transport."""
    
    __slots__ = ('_content', '_lock', '_is_compressible')
    
    def __init__(self):
        """Initialize compressed JSON builder."""
        self._content: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._is_compressible = True
    
    def add_field(self, key: str, value: Any) -> 'CompressedJSONBuilder':
        """Add field to JSON."""
        with self._lock:
            self._content[key] = value
        return self
    
    def build_compact(self) -> str:
        """Build compact JSON (no whitespace)."""
        with self._lock:
            return json.dumps(self._content, separators=(',', ':'), ensure_ascii=True)
    
    def build_pretty(self) -> str:
        """Build pretty JSON (for debugging)."""
        with self._lock:
            return json.dumps(self._content, indent=2)
    
    def estimate_size(self) -> int:
        """Estimate size of compact JSON in bytes."""
        compact = self.build_compact()
        return len(compact.encode('utf-8'))
    
    def get_compression_ratio(self) -> Dict[str, float]:
        """Get estimated compression ratio for different methods."""
        compact = self.build_compact()
        size = len(compact.encode('utf-8'))
        
        # Rough estimates
        return {
            'raw_bytes': size,
            'gzip_ratio': 0.7 if self._is_compressible else 0.95,  # Typical ratios
            'brotli_ratio': 0.5 if self._is_compressible else 0.9,
        }
    
    def reset(self) -> None:
        """Reset builder for reuse."""
        with self._lock:
            self._content.clear()


def create_streaming_response(items: List[Any], chunk_size: int = 4096) -> Iterator[bytes]:
    """Create streaming response from list of items.
    
    Args:
        items: List of items to stream
        chunk_size: Size of chunks in bytes
        
    Yields:
        Chunks of bytes
    """
    builder = StreamingResponseBuilder(chunk_size=chunk_size)
    
    # Start array
    builder.add_string_chunk('[')
    
    for i, item in enumerate(items):
        if i > 0:
            builder.add_string_chunk(',')
        builder.add_chunk(item, is_json=True)
    
    # End array
    builder.add_string_chunk(']')
    
    return builder.get_chunks()


def build_large_response(data: Dict[str, Any], max_inline_size: int = 1024) -> Dict[str, Any]:
    """Build response with streaming hints for large data.
    
    Args:
        data: Response data
        max_inline_size: Maximum size to inline (larger gets streaming hint)
        
    Returns:
        Response with streaming metadata if applicable
    """
    builder = CompressedJSONBuilder()
    
    for key, value in data.items():
        builder.add_field(key, value)
    
    # Add streaming hint if large
    size = builder.estimate_size()
    if size > max_inline_size:
        builder.add_field('_streaming_hint', {
            'size_bytes': size,
            'recommended_chunk_size': 4096,
            'use_gzip': True
        })
    
    return builder._content
