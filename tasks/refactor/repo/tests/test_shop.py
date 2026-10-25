import unittest

from shop.cart import Cart
from shop.pricing import compute_total
from shop.report import summary


class ShopTest(unittest.TestCase):
    def test_compute_total(self):
        self.assertEqual(compute_total([("a", 2.0, 3)], 0.5), 9.0)

    def test_cart(self):
        cart = Cart(tax_rate=0.1)
        cart.add("pen", 1.5, 2)
        self.assertEqual(cart.total(), 3.3)

    def test_report(self):
        self.assertEqual(summary({"b": [("x", 1.0, 1)], "a": [("y", 2.0, 2)]}), "a: 4.00\nb: 1.00")


if __name__ == "__main__":
    unittest.main()
