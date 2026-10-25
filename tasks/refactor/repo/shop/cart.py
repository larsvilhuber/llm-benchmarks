from shop.pricing import calc


class Cart:
    def __init__(self, tax_rate=0.0):
        self.items = []
        self.tax_rate = tax_rate

    def add(self, name, unit_price, quantity=1):
        self.items.append((name, unit_price, quantity))

    def total(self):
        return calc(self.items, self.tax_rate)
