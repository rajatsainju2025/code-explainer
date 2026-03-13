"""Tests for logging helpers."""

import logging

import pytest

from code_explainer.logging_helpers import get_logger, log_operation, log_error


def test_get_logger_returns_logger():
    logger = get_logger("test_module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_module"


def test_get_logger_has_handlers():
    logger = get_logger("test_module_with_handlers")
    assert len(logger.handlers) > 0


def test_log_operation(caplog):
    logger = get_logger("test_op_logger")
    caplog.clear()
    
    with caplog.at_level(logging.INFO, logger=logger.name):
        log_operation(logger, "Test operation")
    
    assert "Test operation" in caplog.text


def test_log_operation_with_metadata(caplog):
    logger = get_logger("test_metadata_logger")
    caplog.clear()
    
    with caplog.at_level(logging.INFO, logger=logger.name):
        log_operation(logger, "Test operation", metadata={"key": "value"})
    
    assert "Test operation" in caplog.text
    assert "key=value" in caplog.text


def test_log_error(caplog):
    logger = get_logger("test_error_logger")
    caplog.clear()
    
    error = ValueError("Test error")
    with caplog.at_level(logging.ERROR, logger=logger.name):
        log_error(logger, "Operation failed", error)
    
    assert "Operation failed" in caplog.text
    assert "ValueError" in caplog.text
    assert "Test error" in caplog.text


def test_log_error_with_metadata(caplog):
    logger = get_logger("test_error_metadata_logger")
    caplog.clear()
    
    error = RuntimeError("Runtime issue")
    with caplog.at_level(logging.ERROR, logger=logger.name):
        log_error(logger, "Failed operation", error, metadata={"status": "failed"})
    
    assert "Failed operation" in caplog.text
    assert "RuntimeError" in caplog.text
    assert "status=failed" in caplog.text
