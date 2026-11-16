// SPDX-License-Identifier: Apache-2.0
/**
 * Current Figure 3 B: prefix interaction and recorded capacity metric cards.
 *
 * node paper/figures/fig3_b.js
 * node paper/figures/fig3_b.js --update4
 * node paper/figures/fig3_b.js --ttft-units
 *
 * Output is always an intermediate native PPTX in the figure build directory.
 * Existing images, editable presentations, manuscript sources and PDFs are not
 * overwritten. Candidate output is build/kvfold_update4_b/fig3/fig3_b_update4.pptx.
 * The default and --update4 generate the current figure, with Inspector MiB
 * labels calculated from archived byte counts. Install pinned dependencies with
 * npm ci --prefix paper before running this CPU-only generator.
 * --ttft-units writes a separate fig3_b_units.pptx candidate whose four TTFT
 * value labels include the suffix "s"; measurements and geometry are unchanged.
 *
 * The accepted layout uses Arial, monospaced code, neutral vector shadows,
 * white action buttons, and blue Pool/orange Tier. Metric inputs are kept in
 * figures/fig3_b_metrics.json, independently of retired figure sources.
 * The six metric cards preserve every value and scale, including the shared
 * seven-second TTFT maximum. Their open arrows are reduced to 62% in size.
 *
 * Prefix strips use one 656-token scale. Measured prompt/cached counts come
 * from paper/evidence/live-run.json; no operation-specific explanations,
 * additional records/layers, NVMe budget, or Restore TTFT are introduced.
 * Canvas: 178.6 x 114.996 mm, including complete frame shadows.
 */
'use strict';

const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { execFileSync } = require('node:child_process');
const PptxGenJS = require('pptxgenjs');

const root = path.resolve(__dirname, '../..');
if (process.argv.slice(2).some(arg => !['--update4', '--ttft-units'].includes(arg))) {
  throw new Error('Usage: node paper/figures/fig3_b.js [--update4] [--ttft-units]');
}
const ttftUnits = process.argv.includes('--ttft-units');
const work = path.join(root, 'paper/build/kvfold_update4_b/fig3');
const output = path.join(work, ttftUnits ? 'fig3_b_units.pptx' : 'fig3_b_update4.pptx');
const sourcePath = path.join(__dirname, 'fig3_b_metrics.json');
const source = fs.readFileSync(sourcePath, 'utf8');
const sourceHash = crypto.createHash('sha256').update(source).digest('hex');

const MM = 1 / 25.4;
const TEX_TO_OFFICE_PT = 72 / 72.27;
const XS = 10; // Original TikZ x centimetres to millimetres.
const YS = 7.4; // Preserve Update 3 geometry compression, not font compression.
const METRIC_TRIM = .32; // Remove 2.368 mm from each metric panel and card.
const FOOTER_TRIM = .50; // Shorten both footer cards and the canvas by 3.7 mm.
const SHADOW_X = .06;
const SHADOW_Y = .06;
// Extra right/bottom space balances the existing left/top frame margins.
const WIDTH = (17.8 + SHADOW_X) * XS;
const TOP = 16.62;
const HEIGHT = (TOP - 2 * METRIC_TRIM - FOOTER_TRIM + SHADOW_Y) * YS;
const SANS = 'Arial';
const MONO = 'Courier New';
// Native bold glyphs emphasize the compressed modes; no duplicate text strokes.
const MODE_SANS = 'DejaVu Sans';
// DejaVu Sans contains U+2460--U+2463; only these four glyphs use this face.
const CIRCLED = 'DejaVu Sans';
const C = {
  ink: '1F2A3B', muted: '526074', raw: '8A819B', rawInk: '655C77', blue: '286DCD',
  reused: '159B9E',
  cyan: '00798C', cyanFill: 'E8F7F9', gpuFill: 'F8FAFE',
  orange: 'ED953D', orangeInk: 'CC6D1D',
  line: 'DFE5EF', frame: 'CFD6E2', track: 'EFF2F4',
  blueFill: 'EFF4FD', orangeFill: 'FFF4E8',
  shadow: 'E7EBF0', badge: 'F4F6F8', white: 'FFFFFF',
};

const metricData = JSON.parse(source).metrics;
if (!Array.isArray(metricData) || metricData.length !== 6) {
  throw new Error('Expected six metric cards.');
}
const metrics = metricData.map(m => ({ ...m, color: C[m.color], ink: C[m.ink] }));
for (const m of metrics) {
  if (![m.x, m.top, m.raw, m.compressed, m.maximum].every(Number.isFinite) ||
      m.maximum <= 0 || m.raw < 0 || m.compressed < 0 ||
      Math.max(m.raw, m.compressed) > m.maximum ||
      !['up', 'down'].includes(m.direction) || !['Pool', 'Tier'].includes(m.mode) ||
      !m.color || !m.ink) {
    throw new Error(`Invalid source metric: ${m.label}`);
  }
}


const interactionSource = path.join(root, 'paper/evidence/live-run.json');
const interactionJSON = fs.readFileSync(interactionSource, 'utf8');
const interactionHash = crypto.createHash('sha256').update(interactionJSON).digest('hex');
const captured = new Map();
function collectInteractions(value) {
  if (Array.isArray(value)) {
    value.forEach(collectInteractions);
  } else if (value && typeof value === 'object') {
    if (value.action && Number.isInteger(value.prompt_tokens) && !captured.has(value.action)) {
      captured.set(value.action, value);
    }
    Object.values(value).forEach(collectInteractions);
  }
}
collectInteractions(JSON.parse(interactionJSON));
// After Build supplies exact bytes; never convert the rounded MiB labels.
const verification = captured.get('build')?.verification;
const displayMiB = bytes => {
  if (!Number.isSafeInteger(bytes) || bytes <= 0) {
    throw new Error('The saved Inspector size must contain a positive integer byte count.');
  }
  return `${(bytes / (1024 ** 2)).toFixed(2)} MiB`;
};
const inspectorSizes = {
  raw: displayMiB(verification?.raw_bytes),
  encoded: displayMiB(verification?.encoded_bytes),
};
const actions = [
  { key: 'build', label: 'Build' }, { key: 'append', label: 'Append' },
  { key: 'edit', label: 'Edit early' }, { key: 'restore', label: 'Restore' },
].map(action => {
  const value = captured.get(action.key);
  if (!value) throw new Error(`Missing saved interaction: ${action.key}`);
  return { ...action, prompt: value.prompt_tokens, cached: value.cached_tokens,
    computed: value.computed_tokens };
});
const expectedCounts = [[534, 0, 534], [656, 512, 144], [534, 0, 534], [534, 512, 22]];
if (JSON.stringify(actions.map(a => [a.prompt, a.cached, a.computed])) !== JSON.stringify(expectedCounts)) {
  throw new Error('The saved example changed; review the intended prefix strip values.');
}
const PREFIX_MAX = Math.max(...actions.map(a => a.prompt));
for (const action of actions) {
  if (action.cached + action.computed !== action.prompt || action.cached < 0 || action.computed < 0) {
    throw new Error(`Inconsistent saved token counts: ${action.key}`);
  }
}

const pptx = new PptxGenJS();
pptx.defineLayout({ name: 'KVFoldFigure3Interaction', width: WIDTH * MM, height: HEIGHT * MM });
pptx.layout = 'KVFoldFigure3Interaction';
pptx.author = 'KVFold authors';
pptx.subject = 'Standalone interaction-focused Figure 3; existing manuscript unchanged';
pptx.title = 'KVFold: Prefix reuse and restore, GPU capacity, and history retention';
pptx.company = '';
pptx.lang = 'en-US';
pptx.theme = { headFontFace: SANS, bodyFontFace: SANS, lang: 'en-US' };
const slide = pptx.addSlide();
slide.background = { color: C.white };
slide.addNotes('Figure 3 alternative: reference-layout panel (a), approved six-card panels (b,c).\n' +
  `Metric source SHA-256: ${sourceHash}\nSaved interaction SHA-256: ${interactionHash}\n` +
  'Prefix strips share the same zero-to-656-token scale. White separators mark 128-token boundaries.\n' +
  'Inspector is the independent After Build export/import integrity check.\n' +
  'Both Engine TTFT charts retain a zero-to-seven-second scale.\n' +
  'All visible elements are native editable PowerPoint shapes, lines, and text.');
let group = 'panel-a/chrome';
const name = id => `${group}::${id}`;
const pt = value => value * TEX_TO_OFFICE_PT;
const mx = value => value * XS * MM;
const my = value => (TOP - value) * YS * MM;

function rect(id, x, y, w, h, fill = C.white, stroke = C.line, weight = .55, radius = 2) {
  // Only independent containers get shadows. Bands, lanes, badges, bars and
  // tracks remain flat; block glyphs retain their smaller existing shadows.
  if (weight > 0 && (id === 'frame' || id === 'card' || id.endsWith('-card'))) {
    rect(`${id}-shadow`, x + SHADOW_X, y - SHADOW_Y, w, h,
      C.shadow, C.shadow, 0, radius);
  }
  slide.addShape(radius ? pptx.ShapeType.roundRect : pptx.ShapeType.rect, {
    x: mx(x), y: my(y + h), w: w * XS * MM, h: h * YS * MM,
    rectRadius: radius * TEX_TO_OFFICE_PT / 72,
    fill: { color: fill },
    line: { color: stroke, width: pt(weight), transparency: weight === 0 ? 100 : 0 },
    objectName: name(id),
  });
}

function line(id, x1, y1, x2, y2, color = C.line, weight = .55, opts = {}) {
  const reversed = x2 < x1;
  slide.addShape(pptx.ShapeType.line, {
    x: mx(Math.min(x1, x2)), y: my(Math.max(y1, y2)),
    w: Math.abs(x2 - x1) * XS * MM, h: Math.abs(y2 - y1) * YS * MM,
    flipV: opts.flipV ?? ((y2 < y1) !== reversed),
    line: { color, width: pt(weight),
      ...(opts.dash ? { dashType: 'dash' } : {}),
      ...(opts.arrow ? reversed ? { beginArrowType: 'stealth' } : { endArrowType: 'stealth' } : {}) },
    objectName: name(id),
  });
}

/** Text height is physical millimetres and is never compressed with geometry. */
function text(id, value, x, y, w, opts = {}) {
  const { anchor = 'center', size = 8.5, bold = false, color = C.ink,
    height = 4.2, font = SANS } = opts;
  const left = anchor === 'west' ? x : anchor === 'east' ? x - w : x - w / 2;
  slide.addText(value, {
    x: mx(left), y: my(y) - height * MM / 2, w: w * XS * MM, h: height * MM,
    fontFace: font, fontSize: pt(size), bold, color,
    align: anchor === 'west' ? 'left' : anchor === 'east' ? 'right' : 'center',
    valign: 'mid', margin: 0, breakLine: false, paraSpaceAfterPt: 0,
    fit: 'none', wrap: false, charSpacing: 0, objectName: name(id),
  });
}

function title(id, number, value, y) {
  text(id, [{ text: `${number} `, options: { bold: true } },
    { text: value, options: { bold: true } }],
    .45, y, 12, { anchor: 'west', size: 11.2, height: 5 });
}

/** A full-width title band shares the frame's upper corners and border. */
function panelFrame(bottom, top, tint) {
  const x = .05;
  const width = 17.70;
  const radius = pt(4) / 72;
  const bandWidth = width * XS * MM;
  const bandHeight = .80 * YS * MM;
  rect('frame', x, bottom, width, top - bottom, C.white, C.frame, .65, 4);
  // The band has only two rounded upper corners. Its sides and top coincide
  // with the frame, rather than leaving a white inset around a rounded pill.
  slide.addShape(pptx.ShapeType.custGeom, {
    x: mx(x), y: my(top), w: bandWidth, h: bandHeight,
    points: [
      { x: 0, y: radius },
      { x: radius, y: 0,
        curve: { type: 'arc', hR: radius, wR: radius, stAng: 180, swAng: 90 } },
      { x: bandWidth - radius, y: 0 },
      { x: bandWidth, y: radius,
        curve: { type: 'arc', hR: radius, wR: radius, stAng: 270, swAng: 90 } },
      { x: bandWidth, y: bandHeight }, { x: 0, y: bandHeight }, { close: true },
    ],
    fill: { color: tint }, line: { color: tint, transparency: 100 },
    objectName: name('title-band'),
  });
  // Keep the original border continuous above the fill, including its corners.
  slide.addShape(pptx.ShapeType.roundRect, {
    x: mx(x), y: my(top), w: bandWidth, h: (top - bottom) * YS * MM,
    rectRadius: radius, fill: { color: C.white, transparency: 100 },
    line: { color: C.frame, width: pt(.65) }, objectName: name('frame-outline'),
  });
}

panelFrame(8.57 + FOOTER_TRIM, 16.57, C.blueFill);
title('title', '1.', 'Prefix reuse and restore', 16.17);

// The action flow carries names only. No implementation or operation subtitles.
actions.forEach((action, index) => {
  group = `panel-a/action-${action.key}`;
  const x = .35 + 4.44 * index;
  const width = 3.78;
  rect('card', x, 14.76, width, .80);
  text('label', [
    { text: '①②③④'[index], options: { fontFace: CIRCLED, bold: false } },
    { text: ` ${action.label}` },
  ], x + width / 2, 15.16, width - .44,
    { size: 8.9, bold: true });
  if (index < actions.length - 1) {
    line('next-action', x + width + .16, 15.16, x + 4.44 - .12, 15.16,
      C.muted, .65, { arrow: true });
  }
});

group = 'panel-a/prompt-excerpt';
rect('card', .35, 11.05, 7.10, 3.40);
text('heading', 'Prompt excerpt', .63, 14.04, 6.54,
  { anchor: 'west', size: 8.9, bold: true });
text('instruction-1', '“You are maintaining a Python repository.', .63, 13.34,
  6.54, { anchor: 'west' });
text('instruction-2', 'Diagnose the timeout failure below.”', .63, 12.86,
  6.54, { anchor: 'west' });
text('code-1', 'deadline = monotonic() + timeout', .63, 12.12,
  6.54, { anchor: 'west', size: 8, font: MONO });
text('code-2', 'connection = await client.acquire()', .63, 11.64,
  6.54, { anchor: 'west', size: 8, font: MONO });

group = 'panel-a/prefix-reuse';
rect('card', 7.75, 11.05, 9.70, 3.40);
text('heading', 'Prefix reuse & restore', 8.03, 14.04, 4.50,
  { anchor: 'west', size: 8.9, bold: true });
// The header and all four counts share one column center. Its two short lines
// remain clear of both the data strips and the left-shifted legend group.
const countColumnCenter = 16.295;
const countColumnWidth = 1.75;
text('ratio-label-top', 'Cached /', countColumnCenter, 14.04, 1.90,
  { size: 7.8, color: C.muted });
text('ratio-label-bottom', 'prompt tokens', countColumnCenter, 13.52, 1.90,
  { size: 7.8, color: C.muted });
// Visible text widths at the unchanged 7.8 TeX pt size, measured with the
// local Arial-compatible Liberation Sans face. Extra text-box slack is not
// included in centering: the colored swatches and actual words form one group.
const legendItems = [
  { label: 'Reused', color: C.reused, textWidthMm: 9.449559 },
  { label: 'Restored', color: C.blue, textWidthMm: 11.124546 },
  { label: 'Computed', color: C.orange, textWidthMm: 12.648757 },
];
const legendSwatchMm = 1.50;
const legendTextGapMm = .90;
const legendItemGapMm = 6.00;
const legendWidthMm = legendItems.reduce((sum, item) =>
  sum + legendSwatchMm + legendTextGapMm + item.textWidthMm, 0) +
  legendItemGapMm * (legendItems.length - 1);
const legendShiftMm = 6.00;
let legendX = 7.75 + (9.70 - legendWidthMm / XS) / 2 - legendShiftMm / XS;
for (const item of legendItems) {
  rect(`legend-${item.label}-swatch`, legendX, 13.43, legendSwatchMm / XS, .18,
    item.color, item.color, 0, 0);
  text(`legend-${item.label}`, item.label,
    legendX + (legendSwatchMm + legendTextGapMm) / XS, 13.52,
    (item.textWidthMm + .80) / XS,
    { anchor: 'west', size: 7.8 });
  legendX += (legendSwatchMm + legendTextGapMm + item.textWidthMm + legendItemGapMm) / XS;
}
const stripX = 9.83;
const stripWidth = 5.30;
const stripHeight = .28;
const tokenX = tokens => stripX + stripWidth * tokens / PREFIX_MAX;
actions.forEach((action, index) => {
  group = `panel-a/prefix-reuse/${action.key}`;
  const y = 13.00 - .52 * index;
  text('action-label', action.label, 8.03, y, 1.62, { anchor: 'west' });
  rect('full-scale-track', stripX, y - stripHeight / 2, stripWidth, stripHeight,
    C.track, C.track, 0, 0);
  if (action.cached) {
    const color = action.key === 'restore' ? C.blue : C.reused;
    rect('cached-segment', stripX, y - stripHeight / 2,
      stripWidth * action.cached / PREFIX_MAX, stripHeight, color, color, 0, 0);
  }
  rect('computed-segment', tokenX(action.cached), y - stripHeight / 2,
    stripWidth * action.computed / PREFIX_MAX, stripHeight, C.orange, C.orange, 0, 0);
  for (let token = 128; token < action.prompt; token += 128) {
    line(`block-boundary-${token}`, tokenX(token), y - stripHeight / 2,
      tokenX(token), y + stripHeight / 2, C.white, .5);
  }
  text('count', `${action.cached}/${action.prompt}`, countColumnCenter, y, countColumnWidth,
    { size: 8.9, bold: true });
});

// Compact, aligned bottom cards retain the current response and Inspector scope.
const cardBottom = 8.72 + FOOTER_TRIM;
const cardHeight = 2.09 - FOOTER_TRIM;
const headingY = 10.53;
group = 'panel-a/restore-response';
rect('card', .35, cardBottom, 10.50, cardHeight);
text('heading', 'Restore response', .63, headingY, 5.80,
  { anchor: 'west', size: 8.9, bold: true });
text('excerpt', [
  { text: '“… it is not enforced during the ' },
  { text: 'client.acquire()', options: { fontFace: MONO } }, { text: ' call.”' },
], 5.60, cardBottom + .55, 9.94);

group = 'panel-a/inspector-after-build';
rect('card', 11.10, cardBottom, 6.35, cardHeight);
text('heading', 'KV Inspector', 11.38, headingY, 3.25,
  { anchor: 'west', size: 8.9, bold: true });
rect('status-badge', 15.14, headingY - .20, 2.03, .40, C.badge, C.badge, 0, 1);
text('status', 'After Build', 16.155, headingY, 1.93,
  { size: 7.8, color: C.muted, height: 3.0 });
text('raw-format', 'BF16', 12.78, headingY - .46, 2.30, { size: 7.8, color: C.muted });
text('encoded-format', 'Encoded', 15.83, headingY - .46, 2.30, { size: 7.8, color: C.muted });
text('raw-size', inspectorSizes.raw, 12.78, headingY - .92, 2.30, { size: 9, bold: true });
line('size-arrow', 13.99, headingY - .92, 14.62, headingY - .92, C.blue, .85, { arrow: true });
text('encoded-size', inspectorSizes.encoded, 15.83, headingY - .92, 2.30, { size: 9, bold: true });

for (const p of [
  { id: 'b', bottom: 4.27 + METRIC_TRIM, top: 8.12, tint: C.blueFill, label: 'GPU capacity',
    heading: 7.72, conditions: 'Raw vs Pool mode · 54 GiB KV / GPU' },
  { id: 'c', bottom: .05 + 2 * METRIC_TRIM, top: 3.90 + METRIC_TRIM,
    tint: C.orangeFill, label: 'History retention',
    heading: 3.50 + METRIC_TRIM, conditions: 'Raw vs Tier mode · 400 GiB DRAM' },
]) {
  group = `panel-${p.id}/chrome`;
  panelFrame(p.bottom + FOOTER_TRIM, p.top + FOOTER_TRIM, p.tint);
  title('title', p.id === 'b' ? '2.' : '3.', p.label, p.heading + FOOTER_TRIM);
  text('conditions', p.conditions, 17.35, p.heading + FOOTER_TRIM, 8,
    { anchor: 'east', size: 7.8, color: C.muted });
}

metrics.forEach((m, index) => {
  group = `panel-${index < 3 ? 'b' : 'c'}/metric-${index + 1}`;
  const top = m.top + (index >= 3 ? METRIC_TRIM : 0) + FOOTER_TRIM;
  const height = 2.62 - METRIC_TRIM;
  rect('card', m.x, top - height, 5.42, height, C.white, C.line, .55, 3);
  text('name', m.label, m.x + .28, top - .42, 4.22,
    { anchor: 'west', size: 8.9, bold: true });
  // A broad open head and slim shaft stay readable without a heavy block fill.
  // Dimensions are physical millimetres, independent of the panel compression.
  const direction = m.direction === 'up' ? 1 : -1;
  const arrowX = m.x + 5;
  const arrowY = top - .42;
  const arrowScale = .62;
  const tipY = arrowY + direction * 2.1 * arrowScale / YS;
  const tailY = arrowY - direction * 2.1 * arrowScale / YS;
  const shoulderY = tipY - direction * 1.5 * arrowScale / YS;
  line('direction-stem', arrowX, tailY, arrowX, tipY, m.ink, .65);
  // The figure's upward Y axis reverses the native diagonal-line orientation.
  line('direction-head-left', arrowX - 1.5 * arrowScale / XS, shoulderY, arrowX, tipY, m.ink, .65,
    { flipV: direction > 0 });
  line('direction-head-right', arrowX + 1.5 * arrowScale / XS, shoulderY, arrowX, tipY, m.ink, .65,
    { flipV: direction < 0 });
  for (const item of [
    { key: 'raw', dy: 1.15, label: 'Raw', value: m.raw, shown: m.rawText,
      fill: C.raw, ink: C.rawInk, labelInk: C.rawInk },
    { key: 'mode', dy: 1.88, label: m.mode, value: m.compressed,
      shown: m.compressedText, fill: m.color, ink: m.ink, labelInk: m.ink },
  ]) {
    const isMode = item.key === 'mode';
    text(`${item.key}-label`, item.label, m.x + .28, top - item.dy, .88,
      { anchor: 'west', color: item.labelInk, bold: isMode,
        font: isMode ? MODE_SANS : SANS, size: isMode ? 9.0 : 8.5 });
    rect(`${item.key}-track`, m.x + 1.31, top - item.dy - .14, 2.35, .28,
      C.track, C.track, 0, .6);
    rect(`${item.key}-bar`, m.x + 1.31, top - item.dy - .14,
      2.35 * item.value / m.maximum, .28, item.fill, item.fill, 0, .6);
    const shown = ttftUnits && m.label === 'Engine TTFT' ? `${item.shown}s` : item.shown;
    text(`${item.key}-value`, shown, m.x + 5.14, top - item.dy, 1.24,
      { anchor: 'east', size: isMode ? 9.2 : 9, bold: isMode, color: item.ink,
        font: isMode ? MODE_SANS : SANS });
  }
});

// PptxGenJS keeps objects native but has no public grouping API. Add native
// DrawingML groups while preserving each child's coordinates and stable names.
const groupAndValidate = String.raw`
import json, sys, zipfile, xml.etree.ElementTree as ET
from pathlib import Path

pptx_path = Path(sys.argv[1])
audit_path = Path(sys.argv[2])
P = 'http://schemas.openxmlformats.org/presentationml/2006/main'
A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
ET.register_namespace('p', P)
ET.register_namespace('a', A)
ET.register_namespace('r', R)
ns = {'p':P,'a':A}
tag = lambda uri, name: '{' + uri + '}' + name
with zipfile.ZipFile(pptx_path) as archive:
    files = {name:archive.read(name) for name in archive.namelist()}
root = ET.fromstring(files['ppt/slides/slide1.xml'])
tree = root.find('p:cSld/p:spTree', ns)
objects = [e for e in list(tree) if e.tag in (tag(P,'sp'),tag(P,'cxnSp'),tag(P,'pic'))]
maximum_id = max(int(e.get('id')) for e in root.findall('.//p:cNvPr', ns))
groups = {}
new_id = maximum_id
def group_for(key):
    global new_id
    if key in groups: return groups[key]
    parent_key = key.rsplit('/',1)[0] if '/' in key else None
    parent = group_for(parent_key) if parent_key else tree
    g = ET.SubElement(parent, tag(P,'grpSp'))
    nv = ET.SubElement(g, tag(P,'nvGrpSpPr'))
    new_id += 1
    ET.SubElement(nv,tag(P,'cNvPr'),id=str(new_id),name=key)
    ET.SubElement(nv,tag(P,'cNvGrpSpPr'))
    ET.SubElement(nv,tag(P,'nvPr'))
    ET.SubElement(g,tag(P,'grpSpPr'))
    groups[key] = g
    return g
for obj in objects:
    nv = obj.find('.//p:cNvPr', ns)
    key = nv.get('name').split('::',1)[0]
    tree.remove(obj)
    group_for(key).append(obj)
for key,g in reversed(list(groups.items())):
    bounds = []
    for obj in g.findall('.//p:sp',ns) + g.findall('.//p:cxnSp',ns):
        xf = obj.find('p:spPr/a:xfrm',ns)
        if xf is None: continue
        off,ext = xf.find('a:off',ns),xf.find('a:ext',ns)
        x,y,w,h = [int(v) for v in (off.get('x'),off.get('y'),ext.get('cx'),ext.get('cy'))]
        bounds.append((x,y,x+w,y+h))
    x,y = min(b[0] for b in bounds),min(b[1] for b in bounds)
    w,h = max(b[2] for b in bounds)-x,max(b[3] for b in bounds)-y
    xf = ET.SubElement(g.find('p:grpSpPr',ns),tag(A,'xfrm'))
    for name,attrs in [('off',{'x':str(x),'y':str(y)}),('ext',{'cx':str(w),'cy':str(h)}),
                       ('chOff',{'x':str(x),'y':str(y)}),('chExt',{'cx':str(w),'cy':str(h)})]:
        ET.SubElement(xf,tag(A,name),**attrs)
files['ppt/slides/slide1.xml'] = ET.tostring(root,encoding='utf-8',xml_declaration=True)
text_nodes = root.findall('.//a:t',ns)
media = [n for n in files if n.startswith('ppt/media/') and not n.endswith('/')]
presentation = ET.fromstring(files['ppt/presentation.xml'])
size = presentation.find('p:sldSz',ns)
W,H = int(size.get('cx')),int(size.get('cy'))
outside = []
for obj in root.findall('.//p:sp',ns) + root.findall('.//p:cxnSp',ns):
    xf = obj.find('p:spPr/a:xfrm',ns)
    if xf is None: continue
    off,ext = xf.find('a:off',ns),xf.find('a:ext',ns)
    x,y,w,h = [int(v) for v in (off.get('x'),off.get('y'),ext.get('cx'),ext.get('cy'))]
    if min(x,y,w,h)<0 or x+w>W or y+h>H:
        outside.append(obj.find('.//p:cNvPr',ns).get('name'))
audit = {
    'slides':len(presentation.findall('p:sldIdLst/p:sldId',ns)),
    'native_shapes':len(root.findall('.//p:sp',ns)),
    'native_connectors':len(root.findall('.//p:cxnSp',ns)),
    'native_lines':len(root.findall('.//a:prstGeom[@prst="line"]',ns)),
    'semantic_groups':len(root.findall('.//p:grpSp',ns)),
    'live_text_boxes':len(root.findall('.//p:txBody',ns)),
    'text_runs':len(text_nodes),
    'picture_shapes':len(root.findall('.//p:pic',ns)),
    'embedded_media':media,
    'out_of_slide_objects':outside,
    'canvas_mm':[W/36000,H/36000],
    'all_text':[n.text or '' for n in text_nodes],
}
assert audit['slides']==1 and audit['native_shapes']>100 and audit['live_text_boxes']>=60
assert not media and not audit['picture_shapes'] and not outside,audit
temp = pptx_path.with_suffix('.tmp.pptx')
with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as archive:
    for name,data in files.items(): archive.writestr(name,data)
temp.replace(pptx_path)
audit_path.write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in audit.items() if k!='all_text'},ensure_ascii=False))
`;


(async () => {
  fs.mkdirSync(work, { recursive: true });
  await pptx.writeFile({ fileName: output, compression: true });
  const auditPath = path.join(work, ttftUnits
    ? 'fig3_b_units-native-object-audit.json' : 'native-object-audit.json');
  process.stdout.write(execFileSync('python3', ['-c', groupAndValidate, output, auditPath], { encoding: 'utf8' }));
  const dataPath = path.join(work, ttftUnits
    ? 'fig3_b_units-source-data-check.json' : 'source-data-check.json');
  fs.writeFileSync(dataPath, JSON.stringify({
    metricSource: path.relative(root, sourcePath), metricSourceSha256: sourceHash,
    interactionSource: path.relative(root, interactionSource), interactionSha256: interactionHash,
    canvasMm: [WIDTH, HEIGHT], minimumFontPt: 7.8 * TEX_TO_OFFICE_PT,
    prefixScale: PREFIX_MAX, prefixStripWidthMm: stripWidth * XS,
    interactions: actions,
    inspectorSizes,
    ...(ttftUnits ? { ttftUnitSuffix: 's' } : {}),
    metrics: metrics.map(m => ({ label: m.label, mode: m.mode, raw: m.raw,
      compressed: m.compressed, maximum: m.maximum })),
  }, null, 2) + '\n');
  process.stdout.write(`Saved ${path.relative(root, output)}\n`);
})().catch(error => { console.error(error); process.exitCode = 1; });
