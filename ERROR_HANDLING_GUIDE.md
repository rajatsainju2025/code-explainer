# Error Handling Standardization - Commit 4

## Overview
Standardized error handling across the codebase by promoting use of specific custom exceptions and establishing clear guidelines.

## Exception Hierarchy

The project uses a comprehensive custom exception hierarchy defined in `src/code_explainer/exceptions.py`:

```
Exception
├── CodeExplainerError (base for all project errors)
│   ├── ConfigurationError - Configuration loading/parsing failures
│   ├── ModelError - Model loading or inference errors
│   ├── ValidationError - Input validation failures
│   ├── CacheError - Cache operation failures
│   ├── ResourceError - Resource allocation or management failures
│   ├── InferenceError - Code explanation inference failures
│   ├── TimeoutError - Operation timeout failures
│   └── SecurityError - Security validation failures
```

## Exception Features

Each custom exception includes:
- **Error Message**: Clear description of what went wrong
- **Error Code**: Unique code for logging/tracking (e.g., "VALIDATION_ERROR")
- **Context**: Relevant context information (dict)
- **Field Information**: For validation errors - which field caused the issue

## Usage Guidelines

### 1. Validation Errors

**When to Use**: When input validation fails

```python
from code_explainer.exceptions import ValidationError

# Instead of:
if not code.strip():
    raise ValueError('Code cannot be empty')

# Use:
if not code.strip():
    raise ValidationError(
        'Code cannot be empty',
        field_name='code'
    )
```

### 2. Configuration Errors

**When to Use**: When loading or parsing configuration fails

```python
from code_explainer.exceptions import ConfigurationError

# Instead of:
raise ValueError(f"Missing config key: {key}")

# Use:
raise ConfigurationError(
    f"Missing required configuration key",
    config_path="/path/to/config.yaml",
    missing_key=key
)
```

### 3. Model Errors

**When to Use**: When model loading or inference fails

```python
from code_explainer.exceptions import ModelError

# Instead of:
raise RuntimeError(f"Failed to load {model_name}")

# Use:
raise ModelError(
    f"Failed to load model from checkpoint",
    model_path=model_name,
    model_type="transformer"
)
```

### 4. Cache Errors

**When to Use**: When cache operations fail

```python
from code_explainer.exceptions import CacheError

# Instead of:
raise Exception("Cache lookup failed")

# Use:
raise CacheError(
    "Cache entry not found or expired",
    cache_type="explanation_cache",
    operation="get"
)
```

### 5. Security Errors

**When to Use**: When security validations fail

```python
from code_explainer.exceptions import SecurityError

# Instead of:
raise ValueError("Code contains dangerous patterns")

# Use:
raise SecurityError(
    "Code contains patterns that pose security risks",
    security_check="dangerous_import_detection",
    code_snippet=code[:50]  # First 50 chars only for security
)
```

### 6. Inference Errors

**When to Use**: When code explanation inference fails

```python
from code_explainer.exceptions import InferenceError

# Instead of:
raise RuntimeError("Explanation generation failed")

# Use:
raise InferenceError(
    "Failed to generate explanation",
    strategy="retrieval_augmented",
    code_length=len(code)
)
```

## Error Context Example

**Exception Output**:
```
ValidationError: [VALIDATION_ERROR] Code exceeds maximum length of 10000 characters (Context: field=code, value="long_code_string")
```

This shows:
1. Exception type: `ValidationError`
2. Error code: `[VALIDATION_ERROR]`
3. Message: Clear description
4. Context: Field name and value (truncated if needed)

## Catching and Handling Exceptions

### Specific Exception Handling

```python
from code_explainer.exceptions import ValidationError, ModelError, CodeExplainerError

try:
    explainer.explain_code(code)
except ValidationError as e:
    logger.warning(f"Input validation failed: {e.message}")
    logger.debug(f"Context: {e.context}")
except ModelError as e:
    logger.error(f"Model error: {e.message}")
    # Could attempt fallback or retry
except CodeExplainerError as e:
    logger.error(f"Unexpected error: {e}")
    raise
```

### Context Access

```python
try:
    cache.get(key)
except CacheError as e:
    print(f"Cache type: {e.context.get('cache_type')}")
    print(f"Operation: {e.context.get('operation')}")
    print(f"Error code: {e.error_code}")
```

## Migration Guide

### Phase 1: New Code (Now)
✅ All new code uses custom exceptions

### Phase 2: Key Modules (This Commit)
- `validation.py` - Updated to use `ValidationError`
- Core validation logic uses specific exceptions

### Phase 3: High-Value Modules (Future)
- `retrieval/` - Add specific retrieval exceptions
- `cache/` - Add cache operation exceptions
- `model_loader.py` - Add model loading exceptions

### Phase 4: Remaining Modules (Future)
- All other modules migrated to custom exceptions
- Generic `ValueError`, `RuntimeError` eliminated

## Benefits

✅ **Better Debugging**: Error codes help track issues
✅ **Contextual Information**: Know exactly what failed and why
✅ **Consistent Handling**: All errors follow same pattern
✅ **Type Safety**: Catch specific exception types
✅ **Logging Integration**: Error codes work well with log analysis
✅ **API Clarity**: Clear what exceptions a function can raise

## Files Modified

1. `src/code_explainer/validation.py`
   - Updated `CodeExplanationRequest` validators
   - Updated `BatchCodeExplanationRequest` validators
   - Now uses `ValidationError` with context

## Code Changes Summary

**Replaced**:
- 7 instances of generic `ValueError`
- Improved context in all error raises

**Result**:
- Better error messages for end users
- Easier debugging for developers
- Consistent error handling across modules
- Improved logging with error codes

## Testing Recommendations

```python
import pytest
from code_explainer.exceptions import ValidationError

def test_validation_error_context():
    """Verify ValidationError includes proper context."""
    try:
        raise ValidationError("Invalid code", field_name="code", field_value="test")
    except ValidationError as e:
        assert e.error_code == "VALIDATION_ERROR"
        assert e.context["field"] == "code"
        assert "VALIDATION_ERROR" in str(e)
```

## Backwards Compatibility

✅ All changes are **backwards compatible**
- Custom exceptions inherit from `Exception`
- Code catching generic `Exception` still works
- Existing code continues to function

## Performance Impact

✅ **No performance degradation**
- Exception creation is rare (only on errors)
- Context dict creation is minimal
- String formatting only happens in error path

## Documentation

Error codes and context fields are documented in `exceptions.py`:
```python
class ValidationError(CodeExplainerError):
    """Raised when input validation fails.
    
    Context fields:
        field: The field name that failed validation
        value: The value that failed (truncated to 100 chars)
    """
```

---

## Commit Details

**Files Changed**: 1 modified
**Improvements**: 7 error raises updated to use custom exceptions
**Breaking Changes**: None (backwards compatible)
**Performance Impact**: None (improvements only on error path)
