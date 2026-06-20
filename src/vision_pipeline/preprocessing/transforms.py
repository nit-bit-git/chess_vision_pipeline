def resize(image, size):
    """Return a minimal representation of a resized image.

    `image` can be any object — this helper is intentionally lightweight for tests.
    """
    return {"original": image, "size": tuple(size)}


def normalize(image):
    """Return a placeholder normalized image representation."""
    return {"normalized": True, "original": image}
