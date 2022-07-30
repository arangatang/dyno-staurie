from pathlib import Path


def dump(html: str):
    dumpdir = Path("/tmp/storyweb")
    dumpdir.mkdir(parents=True, exist_ok=True)
    with open(dumpdir / "index.html", "w") as fp:
        fp.write(html.get("body"))
