def track(detections):
    """Return detections augmented with tracking ids (stub)."""
    return [{**d, "track_id": idx} for idx, d in enumerate(detections)]
