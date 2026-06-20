import pathlib


def load_config(path=None):
    """Load YAML config if available, otherwise return defaults."""
    defaults = {"camera": 0, "input_size": [640, 480], "model_path": "models/yolov5_chess.pt"}
    if path is None:
        path = pathlib.Path(__file__).parent.parent / "config" / "pipeline_config.yaml"
    p = pathlib.Path(path)
    if not p.exists():
        return defaults
    try:
        import yaml

        with p.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return data.get("default", defaults)
    except Exception:
        return defaults
