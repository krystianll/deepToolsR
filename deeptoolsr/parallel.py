"""Ordered outer parallelism with an explicit inner native budget."""

from concurrent.futures import ThreadPoolExecutor


def parallel_map(fn, items, threads):
    values = list(items)
    if threads < 1:
        raise ValueError('threads must be positive')
    if threads == 1 or len(values) <= 1:
        return [fn(item, threads) for item in values]
    with ThreadPoolExecutor(max_workers=min(threads, len(values))) as executor:
        return list(executor.map(lambda item: fn(item, 1), values))
