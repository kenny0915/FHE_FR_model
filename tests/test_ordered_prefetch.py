import pytest
from eval.ordered_prefetch import ordered_prefetch


@pytest.mark.parametrize('workers', [0, 1, 4])
def test_order_and_bounded_consumption(workers):
    consumed = []
    def source():
        for i in range(100):
            consumed.append(i)
            yield i
    iterator = ordered_prefetch(lambda i: (i, i * i), source(), workers)
    assert next(iterator) == (0, 0)
    assert len(consumed) <= max(1, 2 * workers + 1)
    assert list(iterator) == [(i, i * i) for i in range(1, 100)]


def test_prefetch_propagates_decode_failures():
    def read(i):
        if i == 3:
            raise FileNotFoundError('bad image')
        return i
    with pytest.raises(FileNotFoundError, match='bad image'):
        list(ordered_prefetch(read, range(10), 2))
