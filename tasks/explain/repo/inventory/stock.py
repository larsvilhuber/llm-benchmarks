from inventory.config import DEFAULT_REORDER_THRESHOLD
from inventory.errors import InsufficientStockError


class Stock:
    def __init__(self):
        self.levels = {}

    def add_stock(self, sku, units):
        self.levels[sku] = self.levels.get(sku, 0) + units

    def remove_stock(self, sku, units):
        have = self.levels.get(sku, 0)
        if units > have:
            raise InsufficientStockError(f"{sku}: wanted {units}, have {have}")
        self.levels[sku] = have - units

    def needs_reorder(self, sku, threshold=DEFAULT_REORDER_THRESHOLD):
        return self.levels.get(sku, 0) < threshold
