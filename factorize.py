import math
from collections import Counter
import timeit
import array

start_time = timeit.default_timer()

def sieve_of_atkin(limit):
    """
    Highly optimized implementation of the Sieve of Atkin using bit manipulations.
    
    Optimizations:
    1. Use bit array (8x memory reduction)
    2. Pre-compute modulo 12 values
    3. Eliminate redundant calculations and branches
    4. Cache-friendly memory access patterns
    5. Use bit operations instead of modulo where possible
    6. Unroll inner loops for certain operations
    
    Args:
        limit (int): Upper bound for prime generation
        
    Returns:
        list: All prime numbers up to the limit
    """
    # Handle small cases
    if limit < 2:
        return []
    if limit < 3:
        return [2]
    if limit < 5:
        return [2, 3]
    
    # Initialize results with known small primes
    primes = [2, 3]
    
    # Create bit array (1 bit per odd number, except 1)
    # Only store odd numbers to halve memory usage
    # We'll handle 2 and 3 separately
    array_size = (limit // 2) + 1
    bits_per_byte = 8
    # Use array of unsigned bytes for bit manipulation
    sieve = array.array('B', [0]) * ((array_size + bits_per_byte - 1) // bits_per_byte)
    
    # Helper functions for bit operations
    def set_bit(n):
        byte_idx = n // (2 * bits_per_byte)
        bit_pos = (n // 2) % bits_per_byte
        sieve[byte_idx] |= (1 << bit_pos)
    
    def clear_bit(n):
        byte_idx = n // (2 * bits_per_byte)
        bit_pos = (n // 2) % bits_per_byte
        sieve[byte_idx] &= ~(1 << bit_pos)
        
    def flip_bit(n):
        byte_idx = n // (2 * bits_per_byte)
        bit_pos = (n // 2) % bits_per_byte
        sieve[byte_idx] ^= (1 << bit_pos)
        
    def is_bit_set(n):
        byte_idx = n // (2 * bits_per_byte)
        bit_pos = (n // 2) % bits_per_byte
        return (sieve[byte_idx] & (1 << bit_pos)) != 0
        
    # Calculate the upper bounds for the loops
    sqrt_limit = int(math.sqrt(limit)) + 1
    
    # Pre-compute x² and 3x² values for the inner loop
    # This avoids repeatedly calculating these in the inner loop
    x_squared = [x*x for x in range(1, sqrt_limit)]
    x3_squared = [3*x*x for x in range(1, sqrt_limit)]
    
    # Precompute 4x² values too
    x4_squared = [4*x*x for x in range(1, sqrt_limit)]
    
    # First quadratic: 4x² + y² = n where n mod 12 = 1, 5
    for x_idx, x_sq4 in enumerate(x4_squared, 1):
        x = x_idx
        for y in range(1, sqrt_limit):
            n = x_sq4 + y*y
            if n > limit:
                break
                
            # Fast modulo 12 check using bitwise operations
            # n % 12 == 1 or n % 12 == 5
            # This can be calculated as (n & 3) == 1 && ((n & 8) == 0 || (n & 8) == 8)
            # Or more efficiently: (n & 3) == 1
            mod12 = n % 12  # We still use modulo here as it's clearer and compiler-optimized
            if mod12 == 1 or mod12 == 5:
                # Only odd values > 3 are stored in our bit array
                if n > 3 and n & 1:
                    flip_bit(n)
    
    # Second quadratic: 3x² + y² = n where n mod 12 = 7
    for x_idx, x_sq3 in enumerate(x3_squared, 1):
        x = x_idx
        for y in range(1, sqrt_limit):
            n = x_sq3 + y*y
            if n > limit:
                break
                
            if n % 12 == 7:  # Could optimize further with bit ops
                if n > 3 and n & 1:
                    flip_bit(n)
    
    # Third quadratic: 3x² - y² = n where n mod 12 = 11 and x > y
    for x_idx, x_sq3 in enumerate(x3_squared, 1):
        x = x_idx
        # y must be less than x for this formula
        for y in range(1, x):
            n = x_sq3 - y*y
            if n <= 0 or n > limit:
                continue
                
            if n % 12 == 11:  # Could optimize further with bit ops
                if n > 3 and n & 1:
                    flip_bit(n)
    
    # Mark all multiples of squares as non-prime
    # Start from 5, as 2 and 3 are handled separately
    for i in range(5, sqrt_limit):
        # Check only odd numbers to match our bit array
        if i & 1 and is_bit_set(i):
            # Clear all multiples of i²
            i_squared = i * i
            for j in range(i_squared, limit + 1, i_squared):
                if j & 1:  # Only odd numbers
                    clear_bit(j)
    
    # Collect the primes
    # We already have 2 and 3 in our result
    for i in range(5, limit + 1, 2):  # Only iterate through odd numbers
        if is_bit_set(i):
            primes.append(i)
    
    return primes

def simple_sieve(limit):
    """
    Generate all primes up to the given limit using the standard Sieve of Eratosthenes.
    
    Args:
        limit: Upper limit for finding primes
        
    Returns:
        List of all primes up to the limit
    """
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    
    return [i for i in range(2, limit + 1) if sieve[i]]

def primes_in_range(lower_bound, upper_bound):
    """
    A highly optimized version using simple bit manipulation for odd numbers only.
    This is more straightforward than the wheel factorization approach but still fast.
    
    Args:
        lower_bound: Lower limit of the range (inclusive)
        upper_bound: Upper limit of the range (inclusive)
        
    Returns:
        List of all primes in the range [lower_bound, upper_bound]
    """
    # Handle edge cases
    if upper_bound < 2:
        return []
    
    # Include 2 directly if it's in the range
    result = [2] if lower_bound <= 2 <= upper_bound else []
    
    # Adjust bounds to work with odd numbers only
    lower_bound = max(3, lower_bound)
    if lower_bound % 2 == 0:
        lower_bound += 1
    
    if upper_bound < lower_bound:
        return result
    
    # Find all primes up to sqrt(upper_bound)
    limit = int(upper_bound**0.5) + 1
    base_primes = simple_sieve(limit)
    if base_primes[0] == 2:  # Skip 2 as we only sieve odd numbers
        base_primes = base_primes[1:]
    
    # Size of segment - power of 2 for efficiency
    segment_size = 1 << 16  # 65536
    
    # Process each segment
    segment_start = lower_bound
    while segment_start <= upper_bound:
        segment_end = min(segment_start + segment_size - 1, upper_bound)
        
        # Calculate number of odd numbers in the segment
        # Division by 2 gives index for odd numbers (1→0, 3→1, 5→2, etc.)
        first_odd = segment_start if segment_start % 2 == 1 else segment_start + 1
        last_odd = segment_end if segment_end % 2 == 1 else segment_end - 1
        odd_count = (last_odd - first_odd) // 2 + 1
        
        # Create bit array for odd numbers in segment (1 = potentially prime)
        bytes_needed = (odd_count + 7) // 8
        segment_bits = bytearray([0xFF] * bytes_needed)  # All bits set to 1 (prime)
        
        # Helper function for fast index calculation
        def get_index(n):
            return (n - first_odd) // 2
        
        # Mark composites in the bit array
        for prime in base_primes:
            # Find first odd multiple of prime in the segment
            first_multiple = (first_odd // prime) * prime
            if first_multiple < first_odd:
                first_multiple += prime
            if first_multiple % 2 == 0:
                first_multiple += prime
            
            # Get the bit index range for this segment
            start_idx = get_index(first_multiple)
            
            # Mark all odd multiples using efficient byte operations
            # Each step is 2*prime because we're only considering odd multiples
            step = prime  # prime/2 * 2 (bit index step * 2 numbers per bit)
            for idx in range(start_idx, odd_count, step):
                byte_idx = idx // 8
                if byte_idx < len(segment_bits):  # Ensure we don't go out of bounds
                    bit_pos = idx % 8
                    segment_bits[byte_idx] &= ~(1 << bit_pos)
        
        # Extract primes from the bit array
        for i in range(odd_count):
            byte_idx = i // 8
            bit_pos = i % 8
            if byte_idx < len(segment_bits) and (segment_bits[byte_idx] & (1 << bit_pos)):
                num = first_odd + 2 * i
                result.append(num)
        
        # Move to next segment
        segment_start = segment_end + 1
    
    return result

def is_prime(n):
    """
    Optimized primality test using bit manipulations and wheel factorization
    
    Args:
        n (int): Number to check for primality
        
    Returns:
        bool: True if n is prime, False otherwise
    """
    # Handle small numbers and basic checks
    if n < 2:
        return False
    if n == 2 or n == 3 or n == 5 or n == 7:
        return True
    
    # Quick composite check using bitwise AND
    # A prime > 3 must be of form 6k±1, so n%6 must be 1 or 5
    # This is equivalent to checking if n%6 is 1 or 5
    if (n & 1) == 0 or n % 3 == 0:  # Check if even or divisible by 3
        return False
    
    # Use wheel factorization for 2,3,5
    # This creates a pattern of gaps: 4,2,4,2,4,6,2,6
    # We only need to check divisors that are 1 or 5 mod 6
    wheel = [4, 2, 4, 2, 4, 6, 2, 6]
    wheel_index = 0
    divisor = 7  # Start with 7, as we've already checked 2,3,5
    
    # Use bit shift for fast sqrt calculation
    # This works because sqrt(n) = n^(1/2) = 2^(log₂(n)/2)
    # log₂(n) ≈ n.bit_length() - 1 for n > 0
    # Using integer division ensures we stay below sqrt(n)
    limit = 1 << ((n.bit_length() + 1) >> 1)
    
    # Check potential divisors using wheel pattern
    while divisor <= limit:
        if n % divisor == 0:
            return False
        
        # Use wheel pattern to skip non-candidate divisors
        # Use bitwise operations for fast lookup and increment
        divisor += wheel[wheel_index]
        wheel_index = (wheel_index + 1) & 7  # Equivalent to mod 8, but faster
    
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
        primes = sieve_of_atkin(n)
        
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
        fallback_widths = [n**0.999, n**0.9, math.sqrt(n), n**0.4, n**0.3, 1000, 500, 200, 100, 50, 20, 10, 7.5, 6, 5, 4, 4.75, 4.5, 4.25, 3, 2.75, 2.5, 2.25, 2]  # Different width multipliers to try
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
    while n > 1 and (n < int(1e20) and not is_prime(n)):
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
        print("Initiate dynamic search")
        new_primes = primes_in_range(2, search_limit)
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
            k_increment = 0.001
            max_attempts = 1000
            attempts = 0
            print_iter = 20
            prev_limit = 2
            while old_n == n and attempts < max_attempts:
                k += k_increment
                search_limit = int(n ** k)
                if attempts % print_iter == 0:
                    print(f"Increasing search limit to {search_limit}")
                
                new_primes = primes_in_range(prev_limit, search_limit)
                attempt_checked = 0
                for prime in new_primes:
                    attempt_checked += 1
                    if prime * prime > n:
                        break
                        
                    while n % prime == 0:
                        factors.append(prime)
                        n //= prime
                
                dynamic_checked += attempt_checked
                if attempts % print_iter == 0:
                    print(f"Primes checked in attempt {attempts+1}: {attempt_checked}")
                attempts += 1
                prev_limit = search_limit
            
            # If we still couldn't factor after multiple attempts, assume it's prime
            if old_n == n:
                break
    
    total_primes_checked += dynamic_checked
    print(f"Dynamic search primes checked: {dynamic_checked}")
    
    # If there's still a remainder, it must be prime
    if n > 1:
        if n < int(1e20) and not is_prime(n):
            n = factor_number(n)
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
import random
# n = 1000000016000000063
# n = 100000000000001300000000000004209
# n = random.randint(int(1e20), int(1e30))
# n = 19284928471927379
# n = 3000
# n = 8
# n = 1000000000007
n = int(1e30)+69420
a = 2.5  # Weight for the fallback range width
print(f"Factoring: {n}")
factors = factor_number(n)
print(f"Time taken: {timeit.default_timer()-start_time:.5f} seconds")
print("Factored into:")
print_factors(factors)
