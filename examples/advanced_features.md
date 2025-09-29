# Advanced Code Explanation Examples

This document demonstrates the new state-of-the-art features added to the code explainer.

## Symbolic Explanations

Symbolic explanations provide formal analysis of code properties and generate executable property-based tests.

### Example 1: Function with Assertions

```python
def validate_age(age):
    """Validate user age for system access."""
    assert age >= 0, "Age cannot be negative"
    assert age <= 150, "Age cannot exceed 150"

    if age < 18:
        return "minor"
    elif age < 65:
        return "adult"
    else:
        return "senior"
```

**Command:**
```bash
python -m code_explainer.cli explain --symbolic 'def validate_age(age):
    assert age >= 0, "Age cannot be negative"
    assert age <= 150, "Age cannot exceed 150"
    if age < 18:
        return "minor"
    elif age < 65:
        return "adult"
    else:
        return "senior"'
```

**Symbolic Analysis Output:**
- **Preconditions**: `age >= 0`, `age <= 150`
- **Postconditions**: Returns classification string
- **Property Tests**: Age validation boundary tests
- **Complexity**: O(1) time, low cyclomatic complexity

### Example 2: Sorting Algorithm

```python
def bubble_sort(arr):
    """Sort array using bubble sort algorithm."""
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
```

**Symbolic Analysis:**
- **Time Complexity**: O(n²) due to nested loops
- **Invariants**: Array length preserved, elements permuted
- **Property Tests**: Sorted output verification, permutation check

## Multi-Agent Collaborative Explanations

Multi-agent explanations provide comprehensive analysis from multiple specialized perspectives.

### Example 3: Complex Algorithm

```python
def find_prime_factors(n):
    """Find all prime factors of a number."""
    if n <= 1:
        return []

    factors = []
    d = 2

    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1

    if n > 1:
        factors.append(n)

    return factors
```

**Command:**
```bash
python -m code_explainer.cli explain --multi-agent 'def find_prime_factors(n):
    if n <= 1:
        return []
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors'
```

**Multi-Agent Analysis:**

1. **Structural Agent**:
   - Function structure analysis
   - Nested loops detected
   - Time complexity estimation

2. **Semantic Agent**:
   - Prime factorization algorithm explanation
   - Mathematical logic understanding

3. **Context Agent**:
   - Mathematical algorithm pattern recognition
   - Best practices suggestions

4. **Verification Agent**:
   - Property-based test generation
   - Edge case identification

## API Usage Examples

### Symbolic Analysis via API

```python
import requests

response = requests.post("http://localhost:8000/explain", json={
    "code": """
def factorial(n):
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers")
    if n <= 1:
        return 1
    return n * factorial(n - 1)
""",
    "symbolic": True
})

print(response.json()["explanation"])
```

### Multi-Agent Analysis via API

```python
response = requests.post("http://localhost:8000/explain", json={
    "code": """
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
""",
    "multi_agent": True  # This would be implemented in future API updates
})
```

## Prompt Strategy Combinations

You can combine new features with existing prompt strategies:

### AST-Augmented + Symbolic

```bash
python -m code_explainer.cli explain --prompt-strategy ast_augmented --symbolic 'def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)'
```

### Execution Trace + Multi-Agent

```bash
python -m code_explainer.cli explain --prompt-strategy execution_trace --multi-agent 'def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)'
```

## Comparison with Traditional Explanations

### Traditional Output (vanilla strategy):
"This function implements a bubble sort algorithm that iterates through the array..."

### Symbolic + Multi-Agent Output:
```
# Multi-Agent Code Explanation

**Code Structure Analysis:**
- Functions: bubble_sort
- Cyclomatic Complexity: 3
- Estimated Time Complexity: O(n²)
- Number of Loops: 2

**Semantic Analysis:**
This implements the bubble sort algorithm, which repeatedly steps through the list,
compares adjacent elements and swaps them if they are in wrong order...

**Symbolic Analysis:**
Input Conditions:
- arr: list of comparable elements

Preconditions:
- Array elements must support comparison operators

Postconditions:
- returns sorted(arr)
- len(result) == len(arr)

Property-Based Tests:
- Sorted output should be permutation of input
- Function should not crash with valid inputs

**Contextual Information:**
- Implements a sorting algorithm
- Uses proper nested loop structure

**Verification and Testing:**
Property-based tests that could be generated:
- Sorted output verification
- Array length preservation check
- Element permutation validation
```

## Performance Considerations

### Feature Comparison

| Feature | Speed | Comprehensiveness | Use Case |
|---------|-------|-------------------|----------|
| Vanilla | Fastest | Basic | Quick explanations |
| Symbolic | Medium | High | Formal verification |
| Multi-Agent | Slowest | Highest | Complex code analysis |

### Recommended Usage

- **Development/Learning**: Multi-agent + symbolic
- **Production/API**: Vanilla or symbolic only
- **Code Review**: Multi-agent for complex algorithms
- **Educational**: All features combined

## Integration Examples

### Jupyter Notebook

```python
from code_explainer.model import CodeExplainer

explainer = CodeExplainer(config_path="configs/codet5-small.yaml")

code = """
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)
"""

# Get enhanced explanation
explanation = explainer.explain_code_with_symbolic(code, include_symbolic=True)
print(explanation)

# Get collaborative explanation
collaborative = explainer.explain_code_multi_agent(code)
print(collaborative)
```

### CI/CD Integration

```yaml
# .github/workflows/code-analysis.yml
- name: Analyze Code with AI
  run: |
    python -m code_explainer.cli explain --symbolic --multi-agent \
      --prompt-strategy ast_augmented "$(cat new_algorithm.py)"
```

This demonstrates how the new features position our code explainer as a state-of-the-art system that goes beyond simple text generation to provide formal, verifiable, and comprehensive code analysis.
