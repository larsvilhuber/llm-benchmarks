import unittest

from intervals import merge_intervals


class MergeTest(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(merge_intervals([]), [])

    def test_unsorted_overlap(self):
        self.assertEqual(merge_intervals([(5, 6), (1, 3), (2, 4)]), [(1, 4), (5, 6)])

    def test_touching(self):
        self.assertEqual(merge_intervals([(1, 2), (2, 3)]), [(1, 3)])

    def test_nested(self):
        self.assertEqual(merge_intervals([(1, 10), (2, 3), (4, 5)]), [(1, 10)])

    def test_input_not_modified(self):
        data = [(3, 4), (1, 2)]
        merge_intervals(data)
        self.assertEqual(data, [(3, 4), (1, 2)])


if __name__ == "__main__":
    unittest.main()
