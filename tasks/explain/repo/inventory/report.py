from inventory.stock import Stock


def low_items(stock: Stock):
    return sorted(sku for sku in stock.levels if stock.needs_reorder(sku))
