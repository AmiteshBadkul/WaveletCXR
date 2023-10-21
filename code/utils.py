import os
import psutil
import time

def measure_computational_load(func):
    """A decorator to measure and return the computational load metrics
    (execution time and memory usage) of a function."""

    def wrapper(*args, **kwargs):
        # Measure initial memory usage
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss

        # Measure execution time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        # Measure final memory usage
        end_memory = process.memory_info().rss

        # Calculate metrics
        execution_time = end_time - start_time
        memory_usage = (end_memory - start_memory) / (1024 * 1024)  # Convert to MB

        return result, {"execution_time (s)": execution_time, "memory_usage (MB)": memory_usage}

    return wrapper
