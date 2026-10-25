def merge_intervals(intervals):
    """Merge overlapping or touching intervals.

    Takes a list of (start, end) tuples with start <= end, in any order, and returns a new
    list of (start, end) tuples sorted by start, where intervals that overlap or touch
    (one ends exactly where the next starts) are merged. The input is not modified.

    >>> merge_intervals([(5, 6), (1, 3), (2, 4)])
    [(1, 4), (5, 6)]
    >>> merge_intervals([(1, 2), (2, 3)])
    [(1, 3)]
    >>> merge_intervals([])
    []
    """
    raise NotImplementedError
