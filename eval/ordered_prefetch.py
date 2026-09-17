"""Bounded thread prefetch for deterministic, ordered image decoding."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import islice


def ordered_prefetch(function, items, workers=0):
    if workers < 0:
        raise ValueError('workers must be nonnegative')
    if workers == 0:
        yield from map(function, items)
        return
    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = deque(executor.submit(function, item) for item in islice(iterator, workers * 2))
        while pending:
            value = pending.popleft().result()
            try:
                item = next(iterator)
            except StopIteration:
                pass
            else:
                pending.append(executor.submit(function, item))
            yield value
