from shop import pricing


def summary(orders):
    """One line per order: its total. `orders` maps an order id to a list of items."""
    lines = []
    for order_id, items in sorted(orders.items()):
        lines.append(f"{order_id}: {pricing.calc(items):.2f}")
    return "\n".join(lines)
