"""Tests for configuration utilities."""

import os
import tempfile
import pytest
from pathlib import Path

from code_explainer.utils.config import load_config, _interpolate_env_vars


class TestConfigLoading:
    """Tests for configuration loading."""
    
    def test_load_json_config(self, tmp_path):
        """Test loading JSON configuration."""
        config_file = tmp_path / "config.json"
        config_file.write_text('{"model": "codet5", "batch_size": 32}')
        
        config = load_config(str(config_file))
        
        assert config["model"] == "codet5"
        assert config["batch_size"] == 32
    
    def test_load_yaml_config(self, tmp_path):
        """Test loading YAML configuration."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("model: codet5\nbatch_size: 32\n")
        
        config = load_config(str(config_file))
        
        assert config["model"] == "codet5"
        assert config["batch_size"] == 32
    
    def test_config_not_found(self, tmp_path):
        """Test error handling for missing config file."""
        with pytest.raises(FileNotFoundError):
            load_config(str(tmp_path / "nonexistent.json"))
    
    def test_unsupported_format(self, tmp_path):
        """Test error handling for unsupported config format."""
        config_file = tmp_path / "config.txt"
        config_file.write_text("some content")
        
        with pytest.raises(ValueError, match="Unsupported config file format"):
            load_config(str(config_file))


class TestEnvInterpolation:
    """Tests for environment variable interpolation."""
    
    def test_interpolate_string(self, monkeypatch):
        """Test interpolating environment variables in strings."""
        monkeypatch.setenv("TEST_VAR", "test_value")
        
        result = _interpolate_env_vars("prefix_${TEST_VAR}_suffix")
        
        assert result == "prefix_test_value_suffix"
    
    def test_interpolate_dict(self, monkeypatch):
        """Test interpolating environment variables in dicts."""
        monkeypatch.setenv("DB_HOST", "localhost")
        
        data = {"host": "${DB_HOST}", "port": 5432}
        result = _interpolate_env_vars(data)
        
        assert result["host"] == "localhost"
        assert result["port"] == 5432
    
    def test_interpolate_list(self, monkeypatch):
        """Test interpolating environment variables in lists."""
        monkeypatch.setenv("ITEM", "test")
        
        data = ["${ITEM}", "other"]
        result = _interpolate_env_vars(data)
        
        assert result == ["test", "other"]
    
    def test_missing_env_var_unchanged(self):
        """Test that missing env vars are left unchanged."""
        result = _interpolate_env_vars("${UNDEFINED_VAR_12345}")
        
        assert result == "${UNDEFINED_VAR_12345}"


class TestConfigCaching:
    """Tests for configuration caching behavior."""
    
    def test_config_cached(self, tmp_path):
        """Test that config loading is cached."""
        config_file = tmp_path / "config.json"
        config_file.write_text('{"value": 1}')
        
        # Clear cache
        load_config.cache_clear()
        
        config1 = load_config(str(config_file))
        config2 = load_config(str(config_file))
        
        # Should be same object due to caching
        assert config1 is config2
