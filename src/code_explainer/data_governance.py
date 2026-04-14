"""Data governance, lineage tracking, and retention management.

Optimized for:
- orjson for faster provenance card serialization
- __slots__ for memory-efficient config objects
- Environment-based configuration loading
"""

import os
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List
import logging


logger = logging.getLogger(__name__)


class DataGovernanceConfig:
    """Configuration for data governance policies."""
    
    __slots__ = ('retention_days', 'cleanup_enabled', 'storage_disabled', 'audit_log_path')
    
    def __init__(
        self,
        retention_days: int = 30,
        cleanup_enabled: bool = True,
        storage_disabled: bool = False,
        audit_log_path: str = "logs/data-audit.log"
    ):
        """Initialize data governance configuration.
        
        Args:
            retention_days: Days to keep user-provided data (default: 30)
            cleanup_enabled: Enable automatic cleanup (default: True)
            storage_disabled: Disable all data persistence (default: False)
            audit_log_path: Path to audit log file
        """
        self.retention_days = retention_days
        self.cleanup_enabled = cleanup_enabled
        self.storage_disabled = storage_disabled
        self.audit_log_path = audit_log_path
    
    @classmethod
    def from_env(cls) -> "DataGovernanceConfig":
        """Load configuration from environment variables.
        
        Environment variables:
            CODE_EXPLAINER_DATA_RETENTION_DAYS: Retention period in days
            CODE_EXPLAINER_CLEANUP_ENABLED: Enable cleanup (0 or 1)
            CODE_EXPLAINER_DATA_STORAGE_DISABLED: Disable storage (0 or 1)
            CODE_EXPLAINER_DATA_AUDIT_LOG: Path to audit log
        
        Returns:
            Configured DataGovernanceConfig instance
        """
        return cls(
            retention_days=int(
                os.getenv("CODE_EXPLAINER_DATA_RETENTION_DAYS", "30")
            ),
            cleanup_enabled=os.getenv(
                "CODE_EXPLAINER_CLEANUP_ENABLED", "1"
            ) == "1",
            storage_disabled=os.getenv(
                "CODE_EXPLAINER_DATA_STORAGE_DISABLED", "0"
            ) == "1",
            audit_log_path=os.getenv(
                "CODE_EXPLAINER_DATA_AUDIT_LOG",
                "logs/data-audit.log"
            ),
        )


def log_data_access(
    request_id: str,
    operation: str,
    data_type: str,
    size_bytes: int = 0,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Log data access for audit trail.
    
    Args:
        request_id: Unique request identifier
        operation: Operation type (STORE, RETRIEVE, DELETE)
        data_type: Type of data accessed
        size_bytes: Size of data in bytes
        metadata: Optional metadata to include
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    
    parts = [
        timestamp,
        f"| {request_id}",
        f"| {operation}",
        f"| {data_type}",
    ]
    
    if size_bytes:
        parts.append(f"| {size_bytes} bytes")
    
    if metadata:
        parts.append(f"| {metadata}")
    
    message = " ".join(parts)
    logger.info(message)


def calculate_expiration(retention_days: int) -> datetime:
    """Calculate data expiration date.
    
    Args:
        retention_days: Number of days to retain data
    
    Returns:
        Expiration datetime
    """
    return datetime.now(timezone.utc) + timedelta(days=retention_days)


def is_data_expired(timestamp: float, retention_days: int) -> bool:
    """Check if data entry has expired based on retention policy.
    
    Args:
        timestamp: Unix timestamp of data creation
        retention_days: Retention period in days
    
    Returns:
        True if data is past expiration, False otherwise
    """
    expiration = datetime.fromtimestamp(timestamp, tz=timezone.utc) + timedelta(
        days=retention_days
    )
    return datetime.now(timezone.utc) > expiration


def log_data_lineage(
    operation: str,
    input_datasets: List[str],
    output_datasets: List[str],
    timestamp: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Log data lineage for reproducibility.
    
    Args:
        operation: Operation name (e.g., 'model_training')
        input_datasets: List of input dataset identifiers
        output_datasets: List of output dataset identifiers
        timestamp: ISO-8601 timestamp (auto-generated if None)
        metadata: Optional metadata about the operation
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()
    
    lineage = {
        "timestamp": timestamp,
        "operation": operation,
        "inputs": input_datasets,
        "outputs": output_datasets,
        "metadata": metadata or {},
    }
    
    logger.info("Data lineage: %s", lineage)


class DataProvenance:
    """Manages data provenance and provenance cards."""
    
    __slots__ = ('provenance_dir',)
    
    def __init__(self, provenance_dir: str = "data/provenance"):
        """Initialize provenance manager.
        
        Args:
            provenance_dir: Directory to store provenance cards
        """
        self.provenance_dir = provenance_dir
        os.makedirs(provenance_dir, exist_ok=True)
    
    def create_provenance_card(
        self,
        dataset_name: str,
        description: str,
        source: str,
        composition: Dict[str, Any],
        license_id: str = "CC-BY-4.0",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a provenance card for a dataset.
        
        Args:
            dataset_name: Name of dataset
            description: Dataset description
            source: Source URL or identifier
            composition: Dataset composition details
            license_id: SPDX license identifier
            metadata: Additional metadata
        
        Returns:
            Provenance card dictionary
        """
        card = {
            "name": dataset_name,
            "description": description,
            "source": source,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "composition": composition,
            "license": license_id,
            "metadata": metadata or {},
        }
        
        return card
    
    def save_provenance_card(
        self,
        dataset_name: str,
        card: Dict[str, Any]
    ) -> str:
        """Save provenance card to file.
        
        Args:
            dataset_name: Name of dataset
            card: Provenance card data
        
        Returns:
            Path to saved card
        """
        from .utils.hashing import json_dumps
        
        card_path = os.path.join(
            self.provenance_dir,
            f"{dataset_name}_provenance.json"
        )
        
        with open(card_path, "w") as f:
            # Use orjson for faster serialization (fallback to stdlib)
            # Note: indent not supported in orjson, but we prioritize speed
            f.write(json_dumps(card))
        
        logger.info("Saved provenance card: %s", card_path)
        return card_path
