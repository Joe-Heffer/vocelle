from vocelle.cli import split_frames


def test_split_covers_range_contiguously():
    for first, last, n in [(1, 100, 3), (5, 5, 4), (1, 7, 7), (10, 20, 1), (1, 3, 10)]:
        chunks = split_frames(first, last, n)
        assert chunks[0][0] == first and chunks[-1][1] == last
        assert all(b[0] == a[1] + 1 for a, b in zip(chunks, chunks[1:], strict=False))
        assert all(f0 <= f1 for f0, f1 in chunks)
        assert len(chunks) <= n


def test_split_balanced():
    sizes = [f1 - f0 + 1 for f0, f1 in split_frames(1, 100, 3)]
    assert max(sizes) - min(sizes) <= 1
