def calc(items, tax_rate=0.0):
    """Total price of (name, unit_price, quantity) items, plus tax."""
    subtotal = sum(price * qty for _, price, qty in items)
    return round(subtotal * (1 + tax_rate), 2)
