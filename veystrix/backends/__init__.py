"""Model runtimes. Each one satisfies veystrix.backends.base.Backend.

These are the only modules that import MLX. The rest of veystrix stays pure
Python so it can be imported by every environment, which matters here
because the two models need different ones.
"""
