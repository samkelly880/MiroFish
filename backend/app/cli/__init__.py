"""MiroFish CLI orchestration layer (thin adapter over existing services)."""

__all__ = ["main"]


def main(argv=None):
    from .main import main as _main

    return _main(argv)
