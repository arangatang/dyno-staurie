from pathlib import Path


def load_css(filename: str):
    print("fetching css for", filename)
    print("__file__", __file__)
    current_dir = Path(__file__)
    assets_dir = current_dir.parent.parent / "assets"
    css_file = assets_dir / f"{filename}.css"

    with open(css_file, "r") as fp:
        css = fp.read()

    with open(assets_dir / "main.css") as fp:
        main_css = fp.read()
    final_css = "\n".join([main_css, css])
    return final_css


if __name__ == "__main__":
    load_css("read_story_lambda")