"""
generate_dashboard_svg.py
─────────────────────────────────────────────────────────────────────
Run this script from the ROOT of your repo:

    python generate_dashboard_svg.py

It reads  assets/dash1.png … assets/dash5.png,
embeds them as base64 into dashboard.svg,
and writes the final self-contained dashboard.svg to the repo root.

The README.md then just references:
    ![Dashboard](dashboard.svg)
"""

import base64, sys, textwrap
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────
ASSETS_DIR  = Path("assets")
OUTPUT_SVG  = Path("dashboard.svg")
IMAGE_FILES = [ASSETS_DIR / f"dash{i}.png" for i in range(1, 6)]
LABELS      = ["Overview", "Performance Trends", "Grade Distribution",
               "Attendance Analysis", "Subject Comparison"]

# ── Load & encode images ─────────────────────────────────────────────
imgs_b64 = []
for p in IMAGE_FILES:
    if not p.exists():
        print(f"  ERROR: {p} not found. Place dash1.png–dash5.png in assets/")
        sys.exit(1)
    with open(p, "rb") as f:
        imgs_b64.append(base64.b64encode(f.read()).decode())
    print(f"  ✓  {p}  ({len(imgs_b64[-1])//1024} KB base64)")

# ── Build image + label elements ─────────────────────────────────────
image_tags = ""
label_tags = ""
for i, (b64, lbl) in enumerate(zip(imgs_b64, LABELS)):
    opacity = "1" if i == 0 else "0"
    image_tags += f'''    <image id="si{i}" class="img-slide" opacity="{opacity}"
           href="data:image/png;base64,{b64}"
           x="44" y="118" width="872" height="390"
           preserveAspectRatio="xMidYMid slice"/>\n'''
    label_tags += f'''  <text class="lbl" id="lbl{i}" x="70" y="498"
        font-family="'Segoe UI',Arial,sans-serif"
        font-size="11" font-weight="600" fill="#fff" letter-spacing="0.3"
        opacity="{opacity}"
        style="animation:lbl{i} 17.5s ease-in-out infinite">{lbl}</text>\n'''

# ── Build CSS keyframes ───────────────────────────────────────────────
N         = 5
DUR       = 17.5   # total cycle seconds
HOLD      = 0.771  # fraction of each slot = hold time
FADE      = 0.171  # fade-out as fraction of slot

css_frames = ""
for i in range(N):
    slot      = 100.0 / N          # 20 % each
    start     = i * slot
    hold_end  = start + slot * HOLD * 100 / 100
    fade_end  = start + slot

    # clamp
    s  = f"{start:.2f}"
    h  = f"{hold_end:.2f}"
    fe = f"{fade_end:.2f}"

    if i == 0:
        css_frames += f"""@keyframes img{i}{{
  0%{{opacity:1}} {h}%{{opacity:1}} {fe}%,100%{{opacity:0}}
}}\n"""
        css_frames += f"""@keyframes lbl{i}{{
  0%{{opacity:1}} {h}%{{opacity:1}} {fe}%,100%{{opacity:0}}
}}\n"""
    else:
        prev_end = f"{(i*slot):.2f}"
        css_frames += f"""@keyframes img{i}{{
  0%,{prev_end}%{{opacity:0}} {s}%{{opacity:1}} {h}%{{opacity:1}} {fe}%,100%{{opacity:0}}
}}\n"""
        css_frames += f"""@keyframes lbl{i}{{
  0%,{prev_end}%{{opacity:0}} {s}%{{opacity:1}} {h}%{{opacity:1}} {fe}%,100%{{opacity:0}}
}}\n"""

dot_frames = ""
for i in range(N):
    slot  = 100.0 / N
    start = i * slot
    end   = start + slot
    if i == 0:
        dot_frames += f"""@keyframes dot{i}{{
  0%{{opacity:1;r:7}} {end:.2f}%,100%{{opacity:.22;r:5}}
}}\n"""
    else:
        dot_frames += f"""@keyframes dot{i}{{
  0%,{start:.2f}%{{opacity:.22;r:5}} {start:.2f}%{{opacity:1;r:7}} {end:.2f}%,100%{{opacity:.22;r:5}}
}}\n"""

# ── Assemble SVG ──────────────────────────────────────────────────────
prog_steps = " ".join(
    f"{i*100/N:.2f}%{{width:{int(i*872/N)}px}}" for i in range(N+1)
)

svg = f"""<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
     viewBox="0 0 960 580" width="960" height="580">
<defs>
  <linearGradient id="bgG" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0%"  stop-color="#090b1a"/>
    <stop offset="100%" stop-color="#0a1320"/>
  </linearGradient>
  <linearGradient id="cardG" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%"  stop-color="#13152e"/>
    <stop offset="100%" stop-color="#0d1126"/>
  </linearGradient>
  <linearGradient id="accG" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0%"  stop-color="#7b6fff"/>
    <stop offset="50%" stop-color="#00e0b5"/>
    <stop offset="100%" stop-color="#7b6fff"/>
  </linearGradient>
  <linearGradient id="progG" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0%"  stop-color="#7b6fff"/>
    <stop offset="100%" stop-color="#00e0b5"/>
  </linearGradient>
  <linearGradient id="botFadeG" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%"  stop-color="#090b1a" stop-opacity="0"/>
    <stop offset="100%" stop-color="#090b1a" stop-opacity="0.42"/>
  </linearGradient>
  <filter id="cardF" x="-4%" y="-4%" width="108%" height="116%">
    <feDropShadow dx="0" dy="8" stdDeviation="20"
                  flood-color="#000" flood-opacity="0.6"/>
  </filter>
  <pattern id="gridP" x="0" y="0" width="48" height="48" patternUnits="userSpaceOnUse">
    <path d="M48 0L0 0 0 48" fill="none" stroke="#fff"
          stroke-width="0.4" stroke-opacity="0.03"/>
  </pattern>
  <clipPath id="imgClip">
    <rect x="44" y="118" width="872" height="390" rx="10"/>
  </clipPath>
</defs>

<style>
.img-slide{{
  animation-duration:{DUR}s;
  animation-timing-function:ease-in-out;
  animation-iteration-count:infinite;
}}
#si0{{animation-name:img0;}}
#si1{{animation-name:img1;}}
#si2{{animation-name:img2;}}
#si3{{animation-name:img3;}}
#si4{{animation-name:img4;}}
#dot0{{animation:dot0 {DUR}s ease-in-out infinite;}}
#dot1{{animation:dot1 {DUR}s ease-in-out infinite;}}
#dot2{{animation:dot2 {DUR}s ease-in-out infinite;}}
#dot3{{animation:dot3 {DUR}s ease-in-out infinite;}}
#dot4{{animation:dot4 {DUR}s ease-in-out infinite;}}
@keyframes prog{{ {prog_steps} }}
#prog{{animation:prog {DUR}s linear infinite;}}
{css_frames}
{dot_frames}
</style>

<!-- ── Background ── -->
<rect width="960" height="580" fill="url(#bgG)"/>
<rect width="960" height="580" fill="url(#gridP)"/>
<circle cx="-30" cy="-30" r="260" fill="#7b6fff" fill-opacity="0.05"/>
<circle cx="990" cy="610" r="240" fill="#00e0b5" fill-opacity="0.04"/>

<!-- ── Card shell ── -->
<rect x="24" y="24" width="912" height="532" rx="20"
      fill="url(#cardG)" filter="url(#cardF)"
      stroke="#fff" stroke-opacity="0.055" stroke-width="1"/>
<rect x="24" y="24" width="912" height="4" rx="2" fill="url(#accG)"/>

<!-- ── Header ── -->
<rect x="44" y="50" width="36" height="36" rx="9"
      fill="#7b6fff" fill-opacity="0.16"
      stroke="#7b6fff" stroke-opacity="0.42" stroke-width="0.8"/>
<polygon points="62,58 75,63.5 62,69 49,63.5" fill="#9d97ff" fill-opacity="0.92"/>
<rect x="71" y="64" width="2.5" height="6" rx="1.2" fill="#9d97ff" fill-opacity="0.7"/>
<line x1="62" y1="69" x2="62" y2="76" stroke="#9d97ff" stroke-width="1.6"
      stroke-linecap="round" stroke-opacity="0.55"/>
<ellipse cx="62" cy="76" rx="5" ry="2.5" fill="none" stroke="#9d97ff"
         stroke-width="1.2" stroke-opacity="0.5"/>
<text x="90" y="65"
      font-family="'Segoe UI','Helvetica Neue',Arial,sans-serif"
      font-size="16" font-weight="700" fill="#ffffff" letter-spacing="0.3">
  Students Academic Performance Analysis
</text>
<rect x="90" y="73" width="156" height="17" rx="8.5"
      fill="#7b6fff" fill-opacity="0.14"
      stroke="#7b6fff" stroke-opacity="0.38" stroke-width="0.8"/>
<text x="168" y="84.5" font-family="'Segoe UI',Arial,sans-serif"
      font-size="9.5" font-weight="600" fill="#a8a3ff"
      text-anchor="middle" letter-spacing="1.1">DASHBOARD PREVIEW</text>
<line x1="44" y1="108" x2="916" y2="108"
      stroke="url(#accG)" stroke-opacity="0.22" stroke-width="0.8"/>

<!-- ── Image area ── -->
<rect x="44" y="118" width="872" height="390" rx="10" fill="#080a18"/>

<g clip-path="url(#imgClip)">
{image_tags}
</g>

<!-- ── Overlays ── -->
<rect x="44" y="448" width="872" height="60"
      fill="url(#botFadeG)" pointer-events="none"/>
<rect x="44" y="118" width="872" height="390" rx="10"
      fill="none" stroke="#fff" stroke-opacity="0.07" stroke-width="1"
      pointer-events="none"/>

<!-- ── Slide label ── -->
<rect x="58" y="482" width="220" height="24" rx="7"
      fill="#000" fill-opacity="0.60"
      stroke="#fff" stroke-opacity="0.10" stroke-width="0.6"/>
{label_tags}

<!-- ── Footer ── -->
<line x1="44" y1="512" x2="916" y2="512"
      stroke="#fff" stroke-opacity="0.06" stroke-width="0.8"/>
<rect x="44" y="520" width="872" height="4" rx="2" fill="#fff" fill-opacity="0.07"/>
<rect id="prog" x="44" y="520" width="0" height="4" rx="2" fill="url(#progG)"/>

<!-- ── Nav dots ── -->
<circle id="dot0" cx="436" cy="546" r="7" fill="#00e0b5" fill-opacity="1"/>
<circle id="dot1" cx="458" cy="546" r="5" fill="#00e0b5" fill-opacity=".22"/>
<circle id="dot2" cx="480" cy="546" r="5" fill="#00e0b5" fill-opacity=".22"/>
<circle id="dot3" cx="502" cy="546" r="5" fill="#00e0b5" fill-opacity=".22"/>
<circle id="dot4" cx="524" cy="546" r="5" fill="#00e0b5" fill-opacity=".22"/>

<text x="44" y="556" font-family="'Segoe UI',Arial,sans-serif"
      font-size="9.5" fill="#fff" fill-opacity="0.22" letter-spacing="0.8">AUTO-ROTATING</text>
<text x="916" y="556" font-family="'Segoe UI',Arial,sans-serif"
      font-size="9.5" fill="#fff" fill-opacity="0.22"
      text-anchor="end" letter-spacing="0.8">EVERY 3.5s</text>

</svg>"""

OUTPUT_SVG.write_text(svg, encoding="utf-8")
size_kb = OUTPUT_SVG.stat().st_size // 1024
print(f"\n  ✅  Written → {OUTPUT_SVG}  ({size_kb} KB)\n")
print("  Drop dashboard.svg into your repo root.")
print("  In README.md use:  ![Dashboard](dashboard.svg)\n")
