def draw_box(image, box, color=(255, 0, 0)):
    """Placeholder for drawing a box on an image.

    This is a stub so downstream code and tests can import it without imaging deps.
    """
    return {"image": image, "box": box, "color": color}


def get_logger(name="vision_pipeline"):
    import logging

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
