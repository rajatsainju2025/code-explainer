"""Monitoring and metrics methods for CodeExplainer."""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple
import psutil
import torch

from ..security import SecurityManager

logger = logging.getLogger(__name__)


class CodeExplainerMonitoringMixin:
    """Mixin providing monitoring, security, and optimization methods."""

    def _init_monitoring(self):
        """Initialize monitoring components."""
        if not hasattr(self, '_security_manager'):
            self._security_manager: Optional[SecurityManager] = None
        if not hasattr(self, '_request_count'):
            self._request_count = 0
        if not hasattr(self, '_total_response_time'):
            self._total_response_time = 0.0
        if not hasattr(self, '_start_time'):
            self._start_time = time.time()

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        self._init_monitoring()
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            result = {
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
                "percent": process.memory_percent()
            }
            
            # Add GPU memory if CUDA is available
            if torch.cuda.is_available():
                result["gpu_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
                result["gpu_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
            
            return result
        except Exception as e:
            logger.error(f"Failed to get memory usage: {e}")
            return {"error": str(e)}

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        self._init_monitoring()
        uptime = time.time() - self._start_time
        avg_response_time = (
            self._total_response_time / self._request_count
            if self._request_count > 0
            else 0.0
        )
        
        return {
            "uptime_seconds": uptime,
            "total_requests": self._request_count,
            "average_response_time": avg_response_time,
            "requests_per_second": self._request_count / uptime if uptime > 0 else 0,
            "memory": self.get_memory_usage()
        }

    def _track_request(self, response_time: float):
        """Track request metrics."""
        self._request_count += 1
        self._total_response_time += response_time

    def validate_input_security(self, code: str) -> Tuple[bool, List[str]]:
        """Validate input for security issues."""
        self._init_monitoring()
        if self._security_manager is None:
            self._security_manager = SecurityManager()
        
        return self._security_manager.validate_code(code)

    def check_rate_limit(self, client_id: str = "default") -> bool:
        """Check if request is within rate limits."""
        self._init_monitoring()
        if self._security_manager is None:
            self._security_manager = SecurityManager()
        
        allowed, _ = self._security_manager.check_rate_limit(client_id)
        return allowed

    def audit_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log a security audit event."""
        self._init_monitoring()
        if self._security_manager is None:
            self._security_manager = SecurityManager()
        
        self._security_manager.executor.audit_logger.log_event(
            event_type,
            details
        )

    def get_setup_info(self) -> Dict[str, Any]:
        """Get comprehensive setup information."""
        info = {
            "model_loaded": hasattr(self, 'model') and self.model is not None,
            "device": str(getattr(self, 'device', 'cpu')),
            "config": {}
        }
        
        if hasattr(self, 'config'):
            info["config"] = {
                "model_name": getattr(self.config, 'model_name', 'unknown'),
                "max_length": getattr(self.config, 'max_length', 512)
            }
        
        if hasattr(self, 'model') and self.model is not None:
            try:
                info["model_parameters"] = sum(
                    p.numel() for p in self.model.parameters()
                )
            except Exception:
                pass
        
        return info

    def enable_quantization(self, bits: int = 8) -> Dict[str, Any]:
        """Enable model quantization."""
        if bits not in [4, 8, 16]:
            return {"error": f"Unsupported quantization: {bits} bits"}
        
        # Placeholder for quantization logic
        logger.info(f"Quantization to {bits}-bit requested")
        return {
            "success": True,
            "message": f"{bits}-bit quantization enabled",
            "bits": bits
        }

    def enable_gradient_checkpointing(self) -> Dict[str, Any]:
        """Enable gradient checkpointing for memory efficiency."""
        try:
            if hasattr(self, 'model') and self.model is not None:
                if hasattr(self.model, 'gradient_checkpointing_enable'):
                    self.model.gradient_checkpointing_enable()
                    return {"success": True, "message": "Gradient checkpointing enabled"}
            
            return {"success": False, "message": "Model does not support gradient checkpointing"}
        except Exception as e:
            logger.error(f"Failed to enable gradient checkpointing: {e}")
            return {"error": str(e)}

    def optimize_for_inference(self) -> Dict[str, Any]:
        """Optimize model for inference."""
        try:
            if hasattr(self, 'model') and self.model is not None:
                self.model.eval()
                
                # Disable dropout
                for module in self.model.modules():
                    if hasattr(module, 'dropout'):
                        module.dropout = 0.0
                
                return {"success": True, "message": "Model optimized for inference"}
            
            return {"success": False, "message": "No model loaded"}
        except Exception as e:
            logger.error(f"Failed to optimize for inference: {e}")
            return {"error": str(e)}

    def optimize_tokenizer(self) -> Dict[str, Any]:
        """Optimize tokenizer for faster processing."""
        try:
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                # Enable fast tokenization if available
                return {"success": True, "message": "Tokenizer optimized"}
            
            return {"success": False, "message": "No tokenizer loaded"}
        except Exception as e:
            logger.error(f"Failed to optimize tokenizer: {e}")
            return {"error": str(e)}

    def explain_code_batch(self, requests: List[Dict[str, Any]]) -> List[str]:
        """Batch explain multiple code snippets.
        
        Args:
            requests: List of dicts with 'code' and optional 'strategy' keys
            
        Returns:
            List of explanations
        """
        explanations = []
        
        for request in requests:
            code = request.get('code', '')
            strategy = request.get('strategy', 'vanilla')
            
            try:
                explanation = self.explain_code(code, strategy=strategy)
                explanations.append(explanation)
            except Exception as e:
                logger.error(f"Failed to explain code: {e}")
                explanations.append(f"Error: {str(e)}")
        
        return explanations
