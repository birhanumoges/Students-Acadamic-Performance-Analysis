"""
scripts/build_svg.py
════════════════════════════════════════════════════════════════════════
Reads  Assets/dash1.png … Assets/dash5.png  from the repo root,
embeds them as base64 data-URIs inside a self-contained SVG,
and writes  dashboard.svg  to the repo root.

Run locally:   python scripts/build_svg.py
Run on CI:     triggered automatically by GitHub Actions on push.
"""

import base64, sys
from pathlib import Path

# ── Paths (relative to repo root) ────────────────────────────────────
ROOT       = Path(__file__).parent.parent
ASSETS     = ROOT / "Assets"
OUT        = ROOT / "dashboard.svg"
IMAGES     = [ASSETS / f"dash{i}.png" for i in range(1, 6)]
LABELS     = [
    "Overview",
    "Performance Trends",
    "Grade Distribution",
    "Attendance Analysis",
    "Subject Comparison",
]

# ── Validate & encode images ──────────────────────────────────────────
b64 = []
for p in IMAGES:
    if not p.exists():
        print(f"  ✗  Missing: {p}")
        print("     Place dash1.png – dash5.png inside Assets/ and try again.")
        sys.exit(1)
    data = base64.b64encode(p.read_bytes()).decode()
    b64.append(data)
    print(f"  ✓  {p.name}  ({len(data)//1024} KB base64)")

# ── Animation constants ───────────────────────────────────────────────
N        = 5          # number of slides
DUR      = 17.5       # total cycle duration in seconds
SLOT     = 100.0 / N  # % each slide occupies  (20 %)
HOLD_PCT = 77.1       # % of slot where slide is fully visible
FADE_PCT = 85.7       # % of slot where slide starts fading to 0

# ── Helper: build @keyframes for one slide ────────────────────────────
def img_kf(i):
    s  = i * SLOT
    h  = s + SLOT * HOLD_PCT / 100
    fe = s + SLOT * FADE_PCT / 100
    if i == 0:
        return (
            f"@keyframes img{i}{{0%{{opacity:1}}"
            f"{h:.2f}%{{opacity:1}}"
            f"{fe:.2f}%,100%{{opacity:0}}}}"
        )
    prev = i * SLOT
    return (
        f"@keyframes img{i}{{0%,{prev:.2f}%{{opacity:0}}"
        f"{s:.2f}%{{opacity:1}}"
        f"{h:.2f}%{{opacity:1}}"
        f"{fe:.2f}%,100%{{opacity:0}}}}"
    )

def dot_kf(i):
    s = i * SLOT
    e = s + SLOT
    if i == 0:
        return (
            f"@keyframes dot{i}{{0%{{opacity:1;r:7}}"
            f"{e:.2f}%,100%{{opacity:.22;r:5}}}}"
        )
    return (
        f"@keyframes dot{i}{{0%,{s:.2f}%{{opacity:.22;r:5}}"
        f"{s:.2f}%{{opacity:1;r:7}}"
        f"{e:.2f}%,100%{{opacity:.22;r:5}}}}"
    )

# ── Build <image> elements ────────────────────────────────────────────
image_els = ""
for i, data in enumerate(b64):
    op = "1" if i == 0 else "0"
    image_els += (
        f'    <image id="si{i}" class="img-slide" opacity="{op}"\n'
        f'           href="data:image/png;base64,{data}"\n'
        f'           x="44" y="118" width="872" height="390"\n'
        f'           preserveAspectRatio="xMidYMid slice"/>\n'
    )

# ── Build label elements ──────────────────────────────────────────────
label_els = ""
for i, lbl in enumerate(LABELS):
    op = "1" if i == 0 else "0"
    label_els += (
        f'  <text class="lbl" x="70" y="498"\n'
        f'        font-family="\'Segoe UI\',Arial,sans-serif"\n'
        f'        font-size="11" font-weight="600" fill="#fff"\n'
        f'        letter-spacing="0.3" opacity="{op}"\n'
        f'        style="animation:img{i} {DUR}s ease-in-out infinite">'
        f'{lbl}</text>\n'
    )

# ── Progress keyframes ────────────────────────────────────────────────
prog_kf = "  ".join(
    f"{i*SLOT:.2f}%{{width:{int(i*872/N)}px}}" for i in range(N + 1)
)

# ── Dot & image keyframes ─────────────────────────────────────────────
all_kf = "\n".join(img_kf(i) for i in range(N))
all_kf += "\n" + "\n".join(dot_kf(i) for i in range(N))

# ── Nav dot elements ──────────────────────────────────────────────────
dot_cx   = [436, 458, 480, 502, 524]
dot_els  = ""
for i, cx in enumerate(dot_cx):
    op = "1" if i == 0 else ".22"
    r  = "7" if i == 0 else "5"
    dot_els += (
        f'<circle id="dot{i}" cx="{cx}" cy="546" r="{r}" '
        f'fill="#00e0b5" fill-opacity="{op}"/>\n'
    )

# ── Assemble SVG ──────────────────────────────────────────────────────
svg = f"""\
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
     viewBox="0 0 960 580" width="960" height="580">
<defs>
  <linearGradient id="bgG" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0%"   stop-color="#090b1a"/>
    <stop offset="100%" stop-color="#0a1320"/>
  </linearGradient>
  <linearGradient id="cardG" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%"   stop-color="#13152e"/>
    <stop offset="100%" stop-color="#0d1126"/>
  </linearGradient>
  <linearGradient id="accG" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0%"   stop-color="#7b6fff"/>
    <stop offset="50%"  stop-color="#00e0b5"/>
    <stop offset="100%" stop-color="#7b6fff"/>
  </linearGradient>
  <linearGradient id="progG" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0%"   stop-color="#7b6fff"/>
    <stop offset="100%" stop-color="#00e0b5"/>
  </linearGradient>
  <linearGradient id="fadeG" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%"   stop-color="#090b1a" stop-opacity="0"/>
    <stop offset="100%" stop-color="#090b1a" stop-opacity="0.42"/>
  </linearGradient>
  <filter id="shadow" x="-4%" y="-4%" width="108%" height="116%">
    <feDropShadow dx="0" dy="8" stdDeviation="20"
                  flood-color="#000" flood-opacity="0.6"/>
  </filter>
  <pattern id="grid" x="0" y="0" width="48" height="48" patternUnits="userSpaceOnUse">
    <path d="M48 0L0 0 0 48" fill="none" stroke="#fff"
          stroke-width="0.4" stroke-opacity="0.03"/>
  </pattern>
  <clipPath id="imgClip">
    <rect x="44" y="118" width="872" height="390" rx="10"/>
  </clipPath>
</defs>

<style>
/* ── Slide images ── */
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
/* ── Nav dots ── */
#dot0{{animation:dot0 {DUR}s ease-in-out infinite;}}
#dot1{{animation:dot1 {DUR}s ease-in-out infinite;}}
#dot2{{animation:dot2 {DUR}s ease-in-out infinite;}}
#dot3{{animation:dot3 {DUR}s ease-in-out infinite;}}
#dot4{{animation:dot4 {DUR}s ease-in-out infinite;}}
/* ── Progress bar ── */
@keyframes prog{{ {prog_kf} }}
#prog{{animation:prog {DUR}s linear infinite;}}
/* ── Slide & label keyframes (auto-generated) ── */
{all_kf}
</style>

<!-- ══ Background ══════════════════════════════════════════════════════ -->
<rect width="960" height="580" fill="url(#bgG)"/>
<rect width="960" height="580" fill="url(#grid)"/>
<circle cx="-30"  cy="-30"  r="260" fill="#7b6fff" fill-opacity="0.05"/>
<circle cx="990" cy="610"  r="240" fill="#00e0b5" fill-opacity="0.04"/>

<!-- ══ Card ════════════════════════════════════════════════════════════ -->
<rect x="24" y="24" width="912" height="532" rx="20"
      fill="url(#cardG)" filter="url(#shadow)"
      stroke="#fff" stroke-opacity="0.055" stroke-width="1"/>
<rect x="24" y="24" width="912" height="4" rx="2" fill="url(#accG)"/>

<!-- ══ Header ══════════════════════════════════════════════════════════ -->
<rect x="44" y="50" width="36" height="36" rx="9"
      fill="#7b6fff" fill-opacity="0.16"
      stroke="#7b6fff" stroke-opacity="0.42" stroke-width="0.8"/>
<polygon points="62,58 75,63.5 62,69 49,63.5" fill="#9d97ff" fill-opacity="0.92"/>
<rect x="71" y="64" width="2.5" height="6" rx="1.2" fill="#9d97ff" fill-opacity="0.7"/>
<line x1="62" y1="69" x2="62" y2="76" stroke="#9d97ff" stroke-width="1.6"
      stroke-linecap="round" stroke-opacity="0.55"/>
<ellipse cx="62" cy="76" rx="5" ry="2.5" fill="none"
         stroke="#9d97ff" stroke-width="1.2" stroke-opacity="0.5"/>
<text x="90" y="65"
      font-family="'Segoe UI','Helvetica Neue',Arial,sans-serif"
      font-size="16" font-weight="700" fill="#fff" letter-spacing="0.3">
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

<!-- ══ Slideshow ═══════════════════════════════════════════════════════ -->
<rect x="44" y="118" width="872" height="390" rx="10" fill="#080a18"/>
<g clip-path="url(#imgClip)">
{image_els}
</g>

<!-- ══ Overlays ════════════════════════════════════════════════════════ -->
<rect x="44" y="448" width="872" height="60"
      fill="url(#fadeG)" pointer-events="none"/>
<rect x="44" y="118" width="872" height="390" rx="10"
      fill="none" stroke="#fff" stroke-opacity="0.07" stroke-width="1"
      pointer-events="none"/>

<!-- ══ Slide label badge ════════════════════════════════════════════════ -->
<rect x="58" y="482" width="220" height="24" rx="7"
      fill="#000" fill-opacity="0.60"
      stroke="#fff" stroke-opacity="0.10" stroke-width="0.6"/>
{label_els}

<!-- ══ Footer ══════════════════════════════════════════════════════════ -->
<line x1="44" y1="512" x2="916" y2="512"
      stroke="#fff" stroke-opacity="0.06" stroke-width="0.8"/>
<!-- progress track -->
<rect x="44" y="520" width="872" height="4" rx="2"
      fill="#fff" fill-opacity="0.07"/>
<!-- progress fill -->
<rect id="prog" x="44" y="520" width="0" height="4" rx="2"
      fill="url(#progG)"/>

<!-- ══ Nav dots ══════════════════════════════════════════════════════════ -->
{dot_els}

<!-- ══ Footer hints ══════════════════════════════════════════════════════ -->
<text x="44"  y="556" font-family="'Segoe UI',Arial,sans-serif"
      font-size="9.5" fill="#fff" fill-opacity="0.22"
      letter-spacing="0.8">AUTO-ROTATING</text>
<text x="916" y="556" font-family="'Segoe UI',Arial,sans-serif"
      font-size="9.5" fill="#fff" fill-opacity="0.22"
      text-anchor="end" letter-spacing="0.8">EVERY 3.5s</text>

</svg>
"""

OUT.write_text(svg, encoding="utf-8")
kb = OUT.stat().st_size // 1024
print(f"\n  ✅  {OUT}  ({kb} KB)\n")
