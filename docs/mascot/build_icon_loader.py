#!/usr/bin/env python3
"""Embed icon + cutout PNGs into the icon/loader showcase."""
import base64
from pathlib import Path

SP = Path(__file__).parent
TEMPLATE = SP / "lumen-icon-loader-template.html"
OUT = SP / "lumen-icon-loader.html"

html = TEMPLATE.read_text(encoding="utf-8")
assets = {
    "{{NAVY}}": SP / "icons" / "lumen-icon-navy-512.png",
    "{{CREAM}}": SP / "icons" / "lumen-icon-cream-512.png",
    "{{MASK}}": SP / "icons" / "lumen-icon-maskable-512.png",
    "{{FULL}}": SP / "icons" / "web-full.png",
}
for tag, path in assets.items():
    b64 = base64.b64encode(path.read_bytes()).decode()
    uri = f"data:image/png;base64,{b64}"
    if tag not in html:
        raise SystemExit(f"placeholder {tag} not found")
    html = html.replace(tag, uri)

if "{{" in html:
    raise SystemExit("unreplaced placeholder remains")
OUT.write_text(html, encoding="utf-8")
print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")
