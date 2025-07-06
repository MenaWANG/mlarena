# Wrapper Feature Selection Implementation Plan

## Overview
Add wrapper-based feature selection capability to the MLArena package, integrated into the PreProcessor module for seamless ML pipeline workflow.

## Design Decisions

### 1. Implementation Location: PreProcessor Module
- **Rationale**: Logical fit between data preprocessing and model training
- **Consistency**: Aligns with existing `filter_feature_selection()` method
- **User Experience**: Single location for all data preparation steps

### 2. Method Type: Static Method
- **Rationale**: Consistency with existing PreProcessor analysis methods
- **Benefits**: 
  - Functional independence (no instance state modification needed)
  - Can be used without creating PreProcessor instance
  - Follows existing pattern of `filter_feature_selection()`, `encoding_recommendations()`

### 3. Algorithm Choice: RFE (Recursive Feature Elimination)
- **Rationale**: More widely adopted, better scikit-learn support, granular control
- **Implementation**: Use `RFECV` from scikit-learn with cross-validation

### 4. Default Parameters
- **n_max_features**: `n_train // 10` (prevents overfitting, computationally reasonable)
- **cv_variance_penalty**: `0.1` (consistent with MLPipeline.tune())
- **min_features_to_select**: `2` (minimum meaningful feature set)

## Implementation Steps

### Step 1: Core Method Implementation ✅
- [x] Add `wrapper_feature_selection` static method to PreProcessor
- [x] Implement RFE with cross-validation and variance penalty
- [x] Add comprehensive error handling and validation
- [x] Include detailed docstring with examples

### Step 2: Visualization ✅
- [x] Create summary plot showing performance vs number of features
- [x] Highlight optimal feature count with error bars
- [x] Match MLArena's existing plot styling

### Step 3: Testing ✅
- [x] Unit tests for basic functionality
- [x] Integration tests with different estimators
- [x] Edge case testing (small datasets, n_max_features smaller than min_features_to_select，etc)
- [x] Performance validation tests

### Step 5: Further optimization
- [ ] Set default scoring to be AUC and RMSE respectively
- [ ] Consider whether it is a good idea to offer default estimator
- [ ] Consider use estimator attribute for task detection 

### Step 6: MLPipeline Integration
- [ ] Add optional `feature_selection_params` to `MLPipeline.tune()`
- [ ] Consider the optimal integration strategy
- [ ] Implement seamless integration workflow
- [ ] Update documentation and examples

## Method Signature

```python
@staticmethod
def wrapper_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    estimator,
    n_max_features: int = None,
    min_features_to_select: int = 2,
    step: int = 1,
    cv: int = 5,
    cv_variance_penalty: float = 0.1,
    scoring: str = None,
    random_state: int = 42,
    visualize: bool = True,
    verbose: bool = True
) -> dict:
```

## Usage Patterns

### Independent Usage
```python
# Pure feature selection analysis
feature_results = PreProcessor.wrapper_feature_selection(
    X_train, y_train, 
    estimator=lgb.LGBMClassifier(),
    n_max_features=20,
    cv_variance_penalty=0.1
)

selected_features = feature_results['selected_features']
X_train_selected = X_train[selected_features]
```

### Integrated with MLPipeline.tune()
```python
# Option 1: Feature selection first, then tune
feature_results = PreProcessor.wrapper_feature_selection(X, y, base_estimator)
selected_features = feature_results['selected_features']
X_selected = X[selected_features]

tune_results = MLPipeline.tune(
    X_selected, y, 
    algorithm=lgb.LGBMClassifier,
    preprocessor=PreProcessor(),
    param_ranges=param_ranges
)

# Option 2: Enhanced tune() method (future implementation)
tune_results = MLPipeline.tune(
    X, y,
    algorithm=lgb.LGBMClassifier,
    preprocessor=PreProcessor(),
    param_ranges=param_ranges,
    feature_selection_params={'n_max_features': 20}
)
```

## Return Value Structure

```python
{
    'selected_features': List[str],           # Names of selected features
    'n_features_selected': int,               # Number of features selected
    'optimal_score': float,                   # Best cross-validation score
    'optimal_score_std': float,               # Standard deviation of best score
    'penalized_score': float,                 # Score with variance penalty applied
    'cv_scores': List[float],                 # All CV scores by feature count
    'cv_scores_std': List[float],             # Standard deviations
    'feature_rankings': List[int],            # Feature importance rankings
    'rfecv_object': RFECV,                    # Full RFECV object for advanced users
    'selection_params': dict                  # Parameters used for selection
}
```

