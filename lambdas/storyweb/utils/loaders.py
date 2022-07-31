from jinja2 import Environment, FileSystemLoader
from pathlib import Path


def get_jinja_environment():
    current_dir = Path(__file__)
    jinja_dir = current_dir.parent.parent / "assets" / "jinja"
    environment = Environment(loader=FileSystemLoader(jinja_dir))
    return environment
