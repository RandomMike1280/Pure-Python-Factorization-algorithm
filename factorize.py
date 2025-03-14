import math
from collections import Counter

def sieve_of_atkin(limit):
    """
    Implementation of the Sieve of Atkin for finding primes up to a limit.
    More efficient than Sieve of Eratosthenes for larger numbers.
    
    Args:
        limit (int): Upper bound for prime generation
        
    Returns:
        list: All prime numbers up to the limit
    """
    # Initialize the sieve
    sieve = [False] * (limit + 1)
    
    # Put in candidate primes: 2, 3
    if limit >= 2:
        sieve[2] = True
    if limit >= 3:
        sieve[3] = True
    
    # Main part of the Sieve of Atkin algorithm
    for x in range(1, int(math.sqrt(limit)) + 1):
        for y in range(1, int(math.sqrt(limit)) + 1):
            # First quadratic: 4x² + y² = n where n mod 12 = 1, 5
            n = 4 * x * x + y * y
            if n <= limit and (n % 12 == 1 or n % 12 == 5):
                sieve[n] = not sieve[n]
            
            # Second quadratic: 3x² + y² = n where n mod 12 = 7
            n = 3 * x * x + y * y
            if n <= limit and n % 12 == 7:
                sieve[n] = not sieve[n]
            
            # Third quadratic: 3x² - y² = n where n mod 12 = 11 and x > y
            n = 3 * x * x - y * y
            if x > y and n <= limit and n % 12 == 11:
                sieve[n] = not sieve[n]
    
    # Mark all multiples of squares as non-prime
    for i in range(5, int(math.sqrt(limit)) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i * i):
                sieve[j] = False
    
    # Collect the primes
    primes = []
    for i in range(2, limit + 1):
        if sieve[i]:
            primes.append(i)
    
    return primes

def segmented_sieve(limit):
    """
    Optimized segmented sieve for finding primes up to a limit.
    Uses memory efficiently by processing segments of the sieve.
    
    Args:
        limit (int): Upper bound for prime generation
        
    Returns:
        list: All prime numbers up to the limit
    """
    primes = []
    size = int(math.sqrt(limit)) + 1
    is_prime = [True] * size
    is_prime[0] = is_prime[1] = False

    # Find primes up to sqrt(limit) using simple sieve
    for i in range(2, size):
        if is_prime[i]:
            primes.append(i)
            for j in range(i * i, size, i):
                is_prime[j] = False

    # Process segments
    low = 2
    block_size = int(math.sqrt(limit))
    high = min(low + block_size, limit)

    while low < limit:
        is_prime_segment = [True] * (high - low)
        
        for prime in primes:
            start = max(prime * prime, ((low + prime - 1) // prime) * prime)
            for i in range(start, high, prime):
                is_prime_segment[i - low] = False
        
        for i in range(high - low):
            if is_prime_segment[i]:
                primes.append(i + low)
        
        low = high
        high = min(high + block_size, limit)

    return primes

def primes_in_range(low, high):
    """
    Finds all prime numbers in the range [low, high], inclusive.
    Uses a segmented approach based on the Sieve of Eratosthenes,
    but optimized to only find primes within the specified range.
    
    Args:
        low (int): The lower bound of the range (inclusive)
        high (int): The upper bound of the range (inclusive)
        
    Returns:
        list: A list of all prime numbers in the range [low, high]
    """
    # Handle edge cases
    if high < 2:
        return []
    
    # Adjust low if needed
    low = max(low, 2)
    
    # Size of the sieve array
    size = high - low + 1
    
    # Generate primes up to sqrt(high) to use as base primes for sieving
    limit = int(high**0.5) + 1
    base_primes = []
    
    # Simple sieve to find base primes
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    # Collect base primes
    for i in range(2, limit + 1):
        if is_prime[i]:
            base_primes.append(i)
    
    # Sieve for the target range
    segment = [True] * size
    
    # Special case for when 'low' is 1
    if low == 1:
        segment[0] = False
    
    # Mark composites in the target range using the base primes
    for p in base_primes:
        # Find the first multiple of p that is >= low
        start = max(p * p, ((low + p - 1) // p) * p)
        
        # Mark all multiples of p in the range as composite
        for i in range(start, high + 1, p):
            segment[i - low] = False
    
    # Collect the primes in the range
    primes = [i + low for i in range(size) if segment[i]]
    
    return primes

def is_prime(n):
    """
    Check if a number is prime using trial division up to sqrt(n)
    
    Args:
        n (int): Number to check for primality
        
    Returns:
        bool: True if n is prime, False otherwise
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    # Only check odd numbers up to sqrt(n)
    for i in range(3, int(n ** 0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def factor_number(n):
    """
    Factor a number into its prime components using an optimized approach.
    
    For small numbers (< 5M), uses segmented sieve up to sqrt(n).
    For larger numbers, uses Sieve of Atkin up to cube root initially,
    then dynamically adjusts search limits based on remainder size.
    Includes fallback strategy for hard-to-factor numbers.
    
    Args:
        n (int): Number to factorize
        
    Returns:
        list: List of prime factors of n
    """
    factors = []
    total_primes_checked = 0  # Counter for total primes checked
    
    # Special case for small numbers
    if n < 5_000_000:
        primes = segmented_sieve(n)
        
        for prime in primes:
            total_primes_checked += 1
            if prime * prime > n:
                break
                
            while n % prime == 0:
                factors.append(prime)
                n //= prime
                
        if n > 1:
            factors.append(n)
        print(f"Total primes checked: {total_primes_checked}")
        return factors
    
    # For larger numbers, start with cube root approach
    if n > 1e20:
        cbrt_n = int(n ** (1/(math.log10(n)/10+1)))
    else:
        cbrt_n = int(n ** (1/3))
    sqrt_n = int(math.sqrt(n))
    range_width = sqrt_n // a
    
    # Use Sieve of Atkin for initial prime generation
    try:
        primes = sieve_of_atkin(cbrt_n)
    except:
        primes = sieve_of_atkin(int(n ** (1/4)))
    print(f"Initial primes up to cube root: {len(primes)}")
    
    # First pass: try to factor using primes up to cube root
    i = 0
    first_pass_checked = 0
    while i < len(primes) and n > 1:
        prime = primes[i]
        first_pass_checked += 1
        if prime * prime > n:
            break
            
        while n % prime == 0:
            factors.append(prime)
            n //= prime
        i += 1
    
    total_primes_checked += first_pass_checked
    print(f"First pass primes checked: {first_pass_checked}")
    
    # If no factors found, try the fallback strategy
    if len(factors) == 0:
        print("Initiating fallback strategy - searching around sqrt(n)")
        fallback_widths = [n**0.999, n**0.9, n**0.5, n**0.3, 1000, 500, 200, 100, 50, 20, 10, 7.5, 6, 5, 4, 4.75, 4.5, 4.25, 3, 2.75, 2.5, 2.25, 2]  # Different width multipliers to try
        fallback_checked = 0
        prev_range = sqrt_n
        for width_multiplier in fallback_widths:
            range_width = sqrt_n // width_multiplier
            print(f"Trying fallback with width multiplier {width_multiplier}")
            fallback_primes = primes_in_range(int(sqrt_n - range_width), int(prev_range))
            print(f"Fallback primes in range: {len(fallback_primes)}")
            
            old_n = n  # To check if we made progress
            phase_checked = 0
            for prime in fallback_primes:
                phase_checked += 1
                if prime * prime > n:
                    break
                    
                while n % prime == 0:
                    factors.append(prime)
                    n //= prime
            
            fallback_checked += phase_checked
            print(f"Primes checked with multiplier {width_multiplier}: {phase_checked}")
            prev_range = sqrt_n - range_width
            if n != old_n:  # If we found factors, break the loop
                break
        
        total_primes_checked += fallback_checked
        print(f"Total fallback primes checked: {fallback_checked}")
    
    # If we still have a large remainder, continue factoring with dynamic limits
    dynamic_checked = 0
    while n > 1 and not is_prime(n):
        # Adjust search limit based on remainder size
        if n > 1e20:
            k = 0.2  # Close to cube root
        elif n > 1e14:
            k = 0.3
        elif n > 1e7:
            k = 0.4
        else:
            k = 0.5  # Close to square root
        
        search_limit = int(n ** k)
        new_primes = sieve_of_atkin(search_limit)
        print(f"Dynamic search with limit {search_limit}: {len(new_primes)} primes")
        
        old_n = n  # To check if we made progress
        phase_checked = 0
        for prime in new_primes:
            phase_checked += 1
            if prime * prime > n:
                break
                
            while n % prime == 0:
                factors.append(prime)
                n //= prime
        
        dynamic_checked += phase_checked
        print(f"Primes checked in this phase: {phase_checked}")
        
        # If we didn't make progress, increase search limit gradually
        if old_n == n:
            k_increment = 0.025
            max_attempts = 10
            attempts = 0
            
            while old_n == n and attempts < max_attempts:
                k += k_increment
                search_limit = int(n ** k)
                print(f"Increasing search limit to {search_limit}")
                
                new_primes = sieve_of_atkin(search_limit)
                attempt_checked = 0
                for prime in new_primes:
                    attempt_checked += 1
                    if prime * prime > n:
                        break
                        
                    while n % prime == 0:
                        factors.append(prime)
                        n //= prime
                
                dynamic_checked += attempt_checked
                print(f"Primes checked in attempt {attempts+1}: {attempt_checked}")
                attempts += 1
            
            # If we still couldn't factor after multiple attempts, assume it's prime
            if old_n == n:
                break
    
    total_primes_checked += dynamic_checked
    print(f"Dynamic search primes checked: {dynamic_checked}")
    
    # If there's still a remainder, it must be prime
    if n > 1:
        factors.append(n)
    
    print(f"Total primes checked across all phases: {total_primes_checked}")
    return factors

def print_factors(factors):
    """Prints a list of factors, using exponents for repeated factors.

    Args:
        factors: A list of integer factors (ideally prime factors).
    """

    if not factors:
        print("1")  # Handle the case of an empty list (e.g., factorizing 1)
        return

    # Use Counter to count the occurrences of each factor
    factor_counts = Counter(factors)

    output_parts = []
    for factor, count in factor_counts.items():
        if count > 1:
            output_parts.append(f"{factor}^{count}")
        else:
            output_parts.append(str(factor))

    print(" * ".join(output_parts))

# Run the factorization
import timeit
import random
# n = 1000000016000000063
# n = 100000000000001300000000000004209
# n = random.randint(int(1e20), int(1e30))
# n = 19284928471927379
# n = 3000
n = 8
a = 2.5  # Weight for the fallback range width
print(f"Factoring: {n}")
start_time = timeit.default_timer()
factors = factor_number(n)
print("Factored into:")
print_factors(factors)
print(f"Time taken: {timeit.default_timer()-start_time:.5f} seconds")
