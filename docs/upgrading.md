# Upgrading MLArena

This guide helps you upgrade between major versions of MLArena that contain breaking changes.

## Table of Contents
- [Upgrading to v0.3.0](#upgrading-to-v030)
- [Future Versions](#future-versions)  
- [Need Help?](#need-help)

## Upgrading to v0.3.0

### ⚠️ Breaking Changes
- **Class renamed**: `ML_PIPELINE` → `MLPipeline`
- **why?**: Follow Python PEP 8 naming conventions (classes use CapWords)

### 🔧 Action recommended:
- Find: `ML_PIPELINE`  
- Replace: `MLPipeline`

### 📅 Timeline

- **v0.3.0**: `ML_PIPELINE` functional with deprecated warning
- **v0.4.0**: `ML_PIPELINE` support will be removed

### 📝 Example

```python
# Before (v0.2.x)
from mlarena import ML_PIPELINE
pipeline = ML_PIPELINE(model=your_model)
results = ML_PIPELINE.tune(X, y, algorithm, preprocessor, param_ranges)

# After (v0.3.0+)  
from mlarena import MLPipeline
pipeline = MLPipeline(model=your_model)
results = MLPipeline.tune(X, y, algorithm, preprocessor, param_ranges)
```

Backward Compatibility: Your existing code still works! `ML_PIPELINE` shows a deprecation warning but functions normally:

```python
# This still works but shows a warning
from mlarena import ML_PIPELINE  # ⚠️ DeprecationWarning
```

## Future Versions

This section will be updated with upgrade instructions for future breaking changes.

## Need Help?

- Check the [Changelog](../CHANGELOG.md) for detailed version notes
- Open an [issue](https://github.com/MenaWANG/mlarena/issues) if you need assistance
- Review the [API documentation](api.rst) for current method signatures 