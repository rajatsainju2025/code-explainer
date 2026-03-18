"""Tests for logging configuration optimizations."""

import pytest
from unittest.mock import patch, MagicMock


class TestStructuredLoggerSlots:
    """Tests for StructuredLogger __slots__ optimization."""
    
    def test_structured_logger_has_slots(self):
        """Test that StructuredLogger uses __slots__."""
        from code_explainer.logging_config import StructuredLogger
        
        assert hasattr(StructuredLogger, '__slots__')
        assert 'logger' in StructuredLogger.__slots__
    
    @patch('code_explainer.logging_config.os.makedirs')
    @patch('code_explainer.logging_config.RotatingFileHandler')
    def test_logger_initialization(self, mock_handler, mock_makedirs):
        """Test that logger initializes with only required attributes."""
        from code_explainer.logging_config import StructuredLogger
        
        mock_handler.return_value = MagicMock()
        
        logger = StructuredLogger("test", "DEBUG")
        
        # Should only have 'logger' attribute due to __slots__
        assert hasattr(logger, 'logger')
        # Should not have __dict__ due to __slots__
        assert not hasattr(logger, '__dict__')


class TestLoggingTypeSafety:
    """Tests for logging type hint correctness."""
    
    @patch('code_explainer.logging_config.os.makedirs')
    @patch('code_explainer.logging_config.RotatingFileHandler')
    def test_add_context_accepts_none(self, mock_handler, mock_makedirs):
        """Test that _add_context accepts None for extra parameter."""
        from code_explainer.logging_config import StructuredLogger
        
        mock_handler.return_value = MagicMock()
        logger = StructuredLogger("test")
        
        # Should not raise TypeError
        context = logger._add_context(None)
        
        assert "service" in context
        assert context["service"] == "code-explainer"
    
    @patch('code_explainer.logging_config.os.makedirs')
    @patch('code_explainer.logging_config.RotatingFileHandler')
    def test_add_context_with_extra(self, mock_handler, mock_makedirs):
        """Test that _add_context merges extra context."""
        from code_explainer.logging_config import StructuredLogger
        
        mock_handler.return_value = MagicMock()
        logger = StructuredLogger("test")
        
        extra = {"custom_field": "custom_value"}
        context = logger._add_context(extra)
        
        assert context["custom_field"] == "custom_value"
        assert "service" in context
