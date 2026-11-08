"""Offline table printer. The model itself lives in models/flashnext/prefill_budget.py (production)."""
from __future__ import annotations

from models.flashnext.prefill_budget import *          # noqa: F401,F403  (names the tests and callers use)
from models.flashnext.prefill_budget import (GB, METAL_LIMIT, GUARD_HEADROOM, budget, state_and_kv)


def table(sizes=(1024, 2048, 4096, 8192, 10_000)) -> str:
    lines = [f"{'tokens':>7} {'as is':>8} {'fits':>5} {'conv fix':>9} {'fits':>5} "
             f"{'retained':>9} {'state+KV':>9} {'logits':>7} {'transient':>10} {'scores':>7}"]
    for tokens in sizes:
        b = budget(tokens)
        lines.append(
            f"{tokens:>7} {b.peak(False) / GB:7.2f}G {'yes' if b.fits(False) else 'NO':>5} "
            f"{b.peak(True) / GB:8.2f}G {'yes' if b.fits(True) else 'NO':>5} "
            f"{b.retained_as_is / GB:8.2f}G {state_and_kv(tokens) / GB:8.2f}G "
            f"{b.logits / GB:6.2f}G {b.transient / GB:9.2f}G {b.attention_scores / GB:6.2f}G")
    return "\n".join(lines)


def table_chunked(sizes=(2048, 4096, 8192, 10_000), chunks=(2048, 1024)) -> str:
    lines = [f"{'tokens':>7} " + " ".join(f"{'chunk ' + str(c):>12}" for c in chunks)]
    for tokens in sizes:
        cells = [f"{budget(tokens, c).peak(True) / GB:11.2f}G" for c in chunks]
        lines.append(f"{tokens:>7} " + " ".join(cells))
    return "\n".join(lines)


if __name__ == "__main__":
    print(table())
    print()
    print("chunked prefill (the retained-input fix is part of every chunk):")
    print(table_chunked())
    print(f"limit {METAL_LIMIT / GB:.2f} GB, stop at {(METAL_LIMIT - GUARD_HEADROOM) / GB:.2f} GB; "
          "the decode pool (2 GB) is created after the scratch ring is freed")
