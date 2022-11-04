from pathlib import Path
import json


def dump(html: str):
    dumpdir = Path("/tmp/storyweb")
    dumpdir.mkdir(parents=True, exist_ok=True)
    with open(dumpdir / "index.html", "w") as fp:
        try:
            fp.write(html.get("body"))
        except:
            fp.write(json.dumps(html.get("body"), indent=1))
