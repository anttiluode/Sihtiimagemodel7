"""Build web/index.html from web/aura_src.html + web/atoms.json + web/sample.b64."""
from pathlib import Path
here = Path(__file__).parent
src = (here / "aura_src.html").read_text()
src = src.replace("/*ATOMS*/[]", (here / "atoms.json").read_text().strip())
src = src.replace("/*SAMPLE*/", (here / "sample.b64").read_text().strip())
(here / "index.html").write_text(src)
(here.parent / "docs" / "index.html").write_text(src)   # GitHub Pages (main, /docs)
print("wrote web/index.html and docs/index.html", len(src), "bytes")
