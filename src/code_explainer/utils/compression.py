"""Data compression utilities for efficient serialization and transport.

This module provides optimized compression strategies for different data types
to minimize network transfer and storage overhead.
"""

import zlib
import gzip
import io
import json
from typing import Any, Dict, Optional, Tuple


class CompressionStrategy:
    """Base class for compression strategies."""
    
    def compress(self, data: bytes) -> bytes:
        """Compress data."""
        raise NotImplementedError
    
    def decompress(self, data: bytes) -> bytes:
        """Decompress data."""
        raise NotImplementedError
    
    def get_compression_ratio(self, data: bytes) -> float:
        """Get compression ratio."""
        compressed = self.compress(data)
        return len(compressed) / len(data) if data else 1.0


class DeflateCompression(CompressionStrategy):
    """DEFLATE compression (RFC 1951)."""
    
    def __init__(self, level: int = 6):
        """Initialize deflate compression.
        
        Args:
            level: Compression level (1-9)
        """
        self.level = level
    
    def compress(self, data: bytes) -> bytes:
        """Compress using DEFLATE."""
        return zlib.compress(data, self.level)
    
    def decompress(self, data: bytes) -> bytes:
        """Decompress DEFLATE."""
        return zlib.decompress(data)


class GZipCompression(CompressionStrategy):
    """GZip compression."""
    
    def __init__(self, level: int = 6):
        """Initialize gzip compression.
        
        Args:
            level: Compression level (1-9)
        """
        self.level = level
    
    def compress(self, data: bytes) -> bytes:
        """Compress using GZip."""
        out = io.BytesIO()
        with gzip.GzipFile(fileobj=out, mode='wb', compresslevel=self.level) as f:
            f.write(data)
        return out.getvalue()
    
    def decompress(self, data: bytes) -> bytes:
        """Decompress GZip."""
        return gzip.decompress(data)


class NoCompression(CompressionStrategy):
    """No compression (passthrough)."""
    
    def compress(self, data: bytes) -> bytes:
        """Return data unchanged."""
        return data
    
    def decompress(self, data: bytes) -> bytes:
        """Return data unchanged."""
        return data
    
    def get_compression_ratio(self, data: bytes) -> float:
        """Always 1.0 (no compression)."""
        return 1.0


class AdaptiveCompression:
    """Automatically selects best compression strategy."""
    
    __slots__ = ('_strategies', '_size_threshold')
    
    def __init__(self, size_threshold: int = 1024):
        """Initialize adaptive compression.
        
        Args:
            size_threshold: Minimum size to compress
        """
        self._strategies = {
            'deflate': DeflateCompression(level=6),
            'gzip': GZipCompression(level=6),
            'none': NoCompression()
        }
        self._size_threshold = size_threshold
    
    def compress_adaptive(self, data: bytes) -> Tuple[bytes, str]:
        """Compress data with best strategy.
        
        Args:
            data: Data to compress
            
        Returns:
            Tuple of (compressed_data, strategy_name)
        """
        if len(data) < self._size_threshold:
            return data, 'none'
        
        best_strategy = 'none'
        best_data = data
        best_ratio = 1.0
        
        for name, strategy in self._strategies.items():
            if name == 'none':
                continue
            try:
                compressed = strategy.compress(data)
                ratio = len(compressed) / len(data)
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_data = compressed
                    best_strategy = name
            except Exception:
                continue
        
        return best_data, best_strategy
    
    def decompress(self, data: bytes, strategy: str) -> bytes:
        """Decompress data.
        
        Args:
            data: Compressed data
            strategy: Strategy used for compression
            
        Returns:
            Decompressed data
        """
        if strategy in self._strategies:
            return self._strategies[strategy].decompress(data)
        return data


class CompressedResponse:
    """Response wrapper with compression support."""
    
    __slots__ = ('_data', '_compressed', '_strategy', '_original_size')
    
    def __init__(self, data: bytes):
        """Initialize compressed response.
        
        Args:
            data: Response data
        """
        self._data = data
        self._original_size = len(data)
        self._compressed = False
        self._strategy = 'none'
    
    def compress(self, strategy: str = 'deflate') -> None:
        """Compress the response.
        
        Args:
            strategy: Compression strategy to use
        """
        strategies = {
            'deflate': DeflateCompression(),
            'gzip': GZipCompression(),
            'none': NoCompression()
        }
        
        if strategy in strategies:
            self._data = strategies[strategy].compress(self._data)
            self._compressed = True
            self._strategy = strategy
    
    def get_data(self) -> bytes:
        """Get response data."""
        return self._data
    
    def get_compression_ratio(self) -> float:
        """Get compression ratio."""
        return len(self._data) / self._original_size if self._original_size else 1.0
    
    def get_savings_bytes(self) -> int:
        """Get bytes saved by compression."""
        return self._original_size - len(self._data)
    
    def get_headers(self) -> Dict[str, str]:
        """Get headers for compressed response."""
        headers = {'Content-Length': str(len(self._data))}
        if self._compressed:
            headers['Content-Encoding'] = self._strategy
        return headers


class JSONCompression:
    """Optimized JSON compression."""
    
    __slots__ = ('_compressor',)
    
    def __init__(self):
        """Initialize JSON compression."""
        self._compressor = AdaptiveCompression()
    
    def compress_json(self, data: Dict[str, Any]) -> Tuple[bytes, str]:
        """Compress JSON data efficiently.
        
        Args:
            data: Dictionary to compress
            
        Returns:
            Tuple of (compressed_data, strategy_name)
        """
        # Use compact JSON format
        json_bytes = json.dumps(data, separators=(',', ':')).encode('utf-8')
        return self._compressor.compress_adaptive(json_bytes)
    
    def decompress_json(self, data: bytes, strategy: str) -> Dict[str, Any]:
        """Decompress and parse JSON.
        
        Args:
            data: Compressed data
            strategy: Compression strategy
            
        Returns:
            Decompressed dictionary
        """
        json_bytes = self._compressor.decompress(data, strategy)
        return json.loads(json_bytes.decode('utf-8'))


# Global compression instances
_adaptive_compression = AdaptiveCompression()
_json_compression = JSONCompression()


def compress_adaptive(data: bytes) -> Tuple[bytes, str]:
    """Compress data adaptively.
    
    Args:
        data: Data to compress
        
    Returns:
        Tuple of (compressed_data, strategy)
    """
    return _adaptive_compression.compress_adaptive(data)


def compress_json_adaptive(data: Dict[str, Any]) -> Tuple[bytes, str]:
    """Compress JSON data adaptively.
    
    Args:
        data: Data to compress
        
    Returns:
        Tuple of (compressed_data, strategy)
    """
    return _json_compression.compress_json(data)
