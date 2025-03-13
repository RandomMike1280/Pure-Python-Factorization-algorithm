# Pure-Python-Factorization-algorithm
No C, C++, no sympy, gmpy2, numpy, itertools,... just pure python and the default math library

## Features

- **Multiple Sieve Methods**: Implements both Sieve of Atkin and Segmented Sieve for efficient prime number generation
- **Dynamic Search Strategy**: Adapts search limits based on the size of the remaining number to factorize
- **Optimized for Different Ranges**:
  - Special handling for small numbers (< 5M)
  - Cube root optimization for very large numbers
  - Fallback strategy using sqrt(n) range when needed
- **Memory Efficient**: Uses segmented approach to handle large ranges without excessive memory usage

## Algorithm Details

The implementation uses a multi-phase approach:

1. **Initial Phase**
   - For small numbers (< 5M): Uses segmented sieve up to sqrt(n)
   - For larger numbers: Starts with cube root approach

2. **Fallback Strategy**
   - If no factors found in initial phase, searches around sqrt(n)
   - Uses dynamic width based on number size

3. **Dynamic Search**
   - Adjusts search limits based on remainder size
   - Uses different exponents (k) for different number ranges:
     - k ≈ 0.2 for n > 10^20 (close to cube root)
     - k ≈ 0.3 for 10^14 < n ≤ 10^20
     - k ≈ 0.4 for 10^7 < n ≤ 10^14
     - k ≈ 0.5 for smaller numbers (close to square root)

## Usage

```python
# Import the module
from Judrajim import factor_number

# Factor a number
n = 1000000007
factors = factor_number(n)
print('*'.join(map(str, factors)))
```

## Performance Characteristics

- **Small Numbers**: Highly efficient for numbers < 5M using segmented sieve
- **Medium Numbers**: Balanced approach using cube root strategy
- **Large Numbers**: Adaptive search with dynamic limits
- **Memory Usage**: Optimized for large numbers through segmented approach
- **Progress Tracking**: Provides detailed progress information during factorization

## Implementation Details

### Key Components

1. **Sieve of Atkin** (`sieve_of_atkin`)
   - More efficient than Sieve of Eratosthenes for larger numbers
   - Implements quadratic forms to identify prime candidates

2. **Segmented Sieve** (`segmented_sieve`)
   - Memory-efficient prime generation
   - Processes sieve in segments

3. **Range-based Prime Finding** (`primes_in_range`)
   - Optimized for finding primes in specific ranges
   - Uses segmented approach

4. **Prime Testing** (`is_prime`)
   - Quick primality test for individual numbers
   - Uses optimized trial division

### Optimization Techniques

- Dynamic adjustment of search limits based on number size
- Fallback strategies for hard-to-factor numbers
- Progress monitoring and reporting
- Memory-efficient sieve implementations

## Example Output

```python
# Example with timing
import timeit
start_time = timeit.default_timer()
factors = factor_number(1000000007)
print('*'.join(map(str, factors)))
print(f"Time taken: {timeit.default_timer()-start_time} seconds")
```

This implementation is particularly useful for:
- Cryptographic applications
- Number theory research
- Performance-critical factorization tasks
- Educational purposes in understanding prime factorization algorithms

# Test cases, all beloww as done on an intel core i5 12450H
 - int(1e16)+69420 took 0.05s to factorize
 - int(1e21)+69420 took 2.1s to factorize
 - int(1e30)+69420 took 4.5 minutes to factorize
