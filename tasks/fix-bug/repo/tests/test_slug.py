import unittest

from slug import slugify


class SlugTest(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(slugify("Hello, World!"), "hello-world")

    def test_whitespace(self):
        self.assertEqual(slugify("  A   B "), "a-b")

    def test_symbols(self):
        self.assertEqual(slugify("C++ rocks"), "c-rocks")


if __name__ == "__main__":
    unittest.main()
