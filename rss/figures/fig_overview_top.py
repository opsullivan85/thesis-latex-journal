"""Rebuild the top row of figures/fig_overview.svg as two panels:
(a) GaitNet, the per-candidate network, and (b) CADENCE = GaitNet + history + candidate mixer.
The bottom panels are relabelled (b)->(c), (c)->(d), (d)->(e). Units: mm, same layer transform."""
import sys
from PIL import ImageFont
SVG = sys.argv[1]
FIRA = '/usr/share/fonts/opentype/fira/FiraSans-Regular.otf'
FIRAB = '/usr/share/fonts/opentype/fira/FiraSans-SemiBold.otf'
LMM = '/usr/share/texmf/fonts/opentype/public/lm-math/latinmodern-math.otf'
_f = {}
def width(txt, path, size):
    key = (path,)
    if key not in _f: _f[key] = ImageFont.truetype(path, 1000)
    return _f[key].getlength(txt) * size / 1000.0

INK, GREY, PURPLE = '#1F2937', '#4B5563', '#4F46E5'
TEAL, OR = '#0D9488', '#D97706'
LAV_F, LAV_S, YEL_F, YEL_S = '#E0E7FF', '#A7A3F2', '#FEF3C7', '#F2B43A'
FS, FSS, MS, MSS = 2.820, 2.400, 3.328, 2.396
out = []
def rect(x, y, w, h, rx=0.7, fill='white', stroke=PURPLE, sw=0.22, extra=''):
    out.append(f'<rect x="{x:.3f}" y="{y:.3f}" width="{w:.3f}" height="{h:.3f}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{extra}/>')
def runs_width(runs):
    w = 0
    for kind, t, size, *_ in runs:
        if kind in ('sub', 'sup'):
            w += width(t, LMM, size)
        else:
            w += width(t, LMM if kind == 'math' else (FIRAB if kind == 'bold' else FIRA), size)
    return w
def text(cx, y, runs, fill=INK, anchor='middle', x0=None):
    """runs: (kind, text, size): kind in text|bold|math|sub (sub = math subscript, baseline +0.592)."""
    w = runs_width(runs)
    x = cx - w / 2 if anchor == 'middle' else x0
    spans = []
    for kind, t, size, *col in runs:
        fam = 'Latin Modern Math' if kind in ('math', 'sub', 'sup') else 'Fira Sans'
        yy = y + (0.592 if kind == 'sub' else (-1.185 if kind == 'sup' else 0))
        wt = ' font-weight="600"' if kind == 'bold' else ''
        c = f' fill="{col[0]}"' if col else ''
        spans.append(f'<tspan x="{x:.3f}" y="{yy:.3f}" font-family="{fam}" font-size="{size:.3f}"{wt}{c}>{t}</tspan>')
        x += width(t, LMM if kind in ('math', 'sub', 'sup') else (FIRAB if kind == 'bold' else FIRA), size)
    out.append(f'<text fill="{fill}" xml:space="preserve">' + ''.join(spans) + '</text>')
def arrow(pts, color=INK, sw=0.3):
    (x1, y1), (x2, y2) = pts[-2], pts[-1]
    # arrowhead of the existing figure: length 1.5, half width 0.575, notch 1.05
    if abs(y2 - y1) < 1e-6:
        d = 1 if x2 > x1 else -1
        tip = (x2, y2); line_end = (x2 - d * 1.125, y2)
        head = [(x2, y2), (x2 - d * 1.5, y2 + 0.575), (x2 - d * 1.05, y2), (x2 - d * 1.5, y2 - 0.575)]
    else:
        d = 1 if y2 > y1 else -1
        line_end = (x2, y2 - d * 1.125)
        head = [(x2, y2), (x2 + 0.575, y2 - d * 1.5), (x2, y2 - d * 1.05), (x2 - 0.575, y2 - d * 1.5)]
    p = pts[:-1] + [line_end]
    out.append('<polyline points="' + ' '.join(f'{a:.3f},{b:.3f}' for a, b in p) + f'" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linejoin="round"/>')
    out.append('<polygon points="' + ' '.join(f'{a:.3f},{b:.3f}' for a, b in head) + f'" fill="{color}"/>')
def panel(x, w, fill, stroke, label, title):
    rect(x, 0.25, w, 29.3, rx=2.0, fill=fill, stroke=stroke, sw=0.27, extra=' stroke-dasharray="1.2,0.8"')
    text(None, 3.35, [('bold', label, FS), ('text', ' ' + title, FS)], anchor='left', x0=x + 1.6)
def stacked(x, y, w, h, fill, stroke):
    for k in (0.8, 0.4, 0.0):
        rect(x - k, y - k, w, h, fill=fill, stroke=stroke)

# ---------------- (a) GaitNet
out.append('<g inkscape:label="(a) GaitNet: per-candidate network">')
panel(0.25, 66.0, '#E6F5F1', '#5FBFAE', '(a)', 'GaitNet')
rows = (7.4, 15.5, 23.6)
rect(1.6, rows[0] - 2.4, 17.0, 4.8, fill=LAV_F, stroke=LAV_S)
text(10.1, rows[0] + 0.733, [('text', 'state ', FS), ('math', '𝐱', MS), ('sub', '𝑔', MSS)])
stacked(1.6, rows[1] - 2.4, 17.0, 4.8, YEL_F, YEL_S)
text(10.1, rows[1] + 0.733, [('text', 'candidate ', FS), ('math', '𝐪', MS), ('sub', '𝑖', MSS)])
rect(1.6, rows[2] - 2.4, 17.0, 4.8, fill=YEL_F, stroke=YEL_S)
text(10.1, rows[2] + 0.815, [('text', 'no step ', FS), ('math', '⌀', MS)])
for r, lab in zip(rows, ([('math', '𝑓', MS), ('sub', '𝑠', MSS)], [('math', '𝑓', MS), ('sub', '𝑐', MSS)], [('math', '𝐞', MS), ('sub', '⌀', MSS)])):
    arrow([(18.6, r), (22.6, r)])
    rect(22.6, r - 2.4, 6.8, 4.8, stroke=TEAL)
    text(26.0, r + 0.733, lab)
    arrow([(29.4, r), (33.2, r)])
rect(33.2, 5.0, 8.0, 21.0, rx=1.3, stroke=TEAL, sw=0.3)
text(37.2, 15.0, [('math', '𝑔', 3.6)])
text(37.2, 19.2, [('text', 'shared', FSS)], fill=GREY)
text(37.2, 21.8, [('text', 'MLP', FSS)], fill=GREY)
arrow([(41.2, 15.5), (44.6, 15.5)])
rect(44.6, 5.0, 20.0, 21.0, rx=1.3, stroke=TEAL, sw=0.3)
text(54.6, 13.6, [('text', 'score ', FS), ('math', '𝛼', MS), ('sub', '𝑖', MSS)])
text(54.6, 17.5, [('text', 'swing time ', FS), ('math', '𝑇', MS), ('sub', '𝑖', MSS)])
text(54.6, 22.6, [('text', 'each option alone', FSS)], fill=GREY)
out.append('</g>')

# ---------------- (b) CADENCE
out.append('<g inkscape:label="(b) CADENCE = GaitNet + history + candidate mixer (orange)">')
panel(67.75, 113.0, '#EEF1FD', '#A9B4F4', '(b)', 'CADENCE: GaitNet + history + candidate mixer')
rect(150.6, 1.45, 4.0, 2.05, rx=0.5, stroke=OR, sw=0.32)
text(None, 3.05, [('text', 'added to GaitNet', FSS)], anchor='left', x0=155.4)
top, mid = 8.6, 20.5
stacked(69.8, top - 2.4, 17.0, 4.8, LAV_F, LAV_S)
text(78.3, top + 0.765, [('text', 'history ', FS), ('math', '𝐡', MS)])
rect(69.8, mid - 2.4, 17.0, 4.8, fill=YEL_F, stroke=YEL_S)
text(78.3, mid + 0.733, [('math', '𝐱', MS), ('sub', '𝑔', MSS), ('math', ', 𝐪', MS), ('sub', '𝑖', MSS), ('math', ', ⌀', MS)])
arrow([(86.8, top), (90.0, top)])
rect(90.0, top - 2.4, 6.8, 4.8, stroke=OR, sw=0.32)
text(93.4, top + 0.733, [('math', '𝑓', MS), ('sub', 'ℎ', MSS)])
arrow([(86.8, mid), (90.0, mid)])
rect(90.0, mid - 3.9, 15.0, 7.8, stroke=TEAL, sw=0.3)
text(97.5, mid - 0.7, [('bold', 'GaitNet', FSS, TEAL)])
text(97.5, mid + 2.3, [('text', 'encoders', FSS)])
cx = 109.3
out.append(f'<circle cx="{cx}" cy="{mid}" r="1.45" fill="white" stroke="{OR}" stroke-width="0.32"/>')
out.append(f'<path d="M {cx-0.8:.3f},{mid} h 1.6 M {cx},{mid-0.8:.3f} v 1.6" stroke="{INK}" stroke-width="0.28"/>')
arrow([(105.0, mid), (cx - 1.45, mid)])
arrow([(96.8, top), (cx, top), (cx, mid - 1.45)], color=OR)
text(102.6, top - 1.0, [('math', '𝑊', MS), ('sub', 'ℎ', MSS)])
arrow([(cx + 1.45, mid), (113.6, mid)])
rect(113.6, 14.5, 12.0, 12.0, rx=1.3, sw=0.3)
text(119.6, 19.9, [('text', 'tokens', FS)])
text(119.6, 23.4, [('math', '𝐭', MS), ('sub', '𝑖', MSS)])
arrow([(125.6, mid), (128.6, mid)])
rect(128.6, 12.8, 17.0, 15.4, rx=1.3, stroke=OR, sw=0.4)
text(137.1, 17.8, [('bold', 'candidate', FS)], fill=PURPLE)
text(137.1, 21.2, [('bold', 'mixer', FS)], fill=PURPLE)
text(137.1, 24.4, [('text', '2 transformer', FSS)], fill=GREY)
text(137.1, 26.8, [('text', 'layers', FSS)], fill=GREY)
arrow([(145.6, mid), (148.6, mid)])
rect(148.6, mid - 3.9, 12.6, 7.8, stroke=TEAL, sw=0.3)
text(154.9, mid - 0.7, [('bold', 'GaitNet', FSS, TEAL)])
text(154.9, mid + 2.3, [('text', 'heads', FSS)])
arrow([(161.2, mid), (164.2, mid)])
rect(164.2, mid - 4.435, 15.904, 8.870, fill=YEL_F, stroke=YEL_S)
text(172.152, mid - 0.902, [('math', '(ℓ, 𝐮', MS), ('sup', '⋆', MSS), ('math', ', 𝑇)', MS)])
text(172.152, mid + 2.933, [('text', 'or ', FS), ('math', '⌀', MS)])
out.append('</g>')

s = open(SVG).read()
a = s.index('<g inkscape:groupmode="layer" inkscape:label="(a) architecture"')
a_open_end = s.index('>', a) + 1
b = s.index('<g inkscape:groupmode="layer"', a_open_end)
close = s.rindex('</g>', a_open_end, b)
s = s[:a] + '<g inkscape:groupmode="layer" inkscape:label="(a) GaitNet and (b) CADENCE" transform="translate(1.3575 0) scale(0.985)">\n' + '\n'.join(out) + '\n' + s[close:]
for old, new in (('(d)', '(e)'), ('(c)', '(d)'), ('(b)', '(c)')):
    tag = f'font-size="2.820" font-weight="600">{old}</tspan>'
    i = s.index('y="33.850"')  # bottom-row titles share this baseline
    cnt = s.count(f'y="33.850" font-family="Fira Sans" {tag}')
    assert cnt == 1, (old, cnt)
    s = s.replace(f'y="33.850" font-family="Fira Sans" {tag}', f'y="33.850" font-family="Fira Sans" font-size="2.820" font-weight="600">{new}</tspan>')
s = s.replace('Panel (a): the dashed teal box is GaitNet,\n     orange edges mark what CADENCE adds (history branch, candidate mixer).',
              'Panel (a): GaitNet, the per-candidate network; panel (b): CADENCE, with\n     its additions to GaitNet (history branch, candidate mixer) in orange; panels (c)-(e) as before.')
open(SVG, 'w').write(s)
print('top row rebuilt')
