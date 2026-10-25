import re


def slugify(text):
    """Lowercase text, replace runs of non-alphanumerics with a single dash, no leading/trailing dash."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]", "-", text)
    return text
