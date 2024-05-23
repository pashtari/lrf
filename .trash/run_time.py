import cProfile
import pstats
import io

def my_function():
    total = 0
    for i in range(10000):
        total += i
    return total

# Create a profiler instance
profiler = cProfile.Profile()

# Profile the function within the context manager
with profiler:
    my_function()

# Create a stream to hold the profiling statistics
stream = io.StringIO()
stats = pstats.Stats(profiler, stream=stream)

# Sort the statistics by the cumulative time and print them
stats.sort_stats(pstats.SortKey.CUMULATIVE)
stats.print_stats()

# Get the profiling results as a string
profile_results = stream.getvalue()
print(profile_results)
