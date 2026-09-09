"use strict";

const TAU = Math.PI * 2;
const DEFAULTS = Object.freeze({
  s: 1.0,
  q: 0.10,
  rho: 0.04,
  xs: -0.075,
  ys: 0.090,
});

const COLORS = Object.freeze({
  background: "#ffffff",
  panel: "#ffffff",
  grid: "#e9eef5",
  gridStrong: "#d5dee9",
  text: "#172338",
  muted: "#718096",
  caustic: "#b7841f",
  causticFill: "rgba(244, 210, 126, 0.18)",
  critical: "#8d9aaa",
  outside: "#287cc8",
  inside: "#26945c",
  sourceOutside: "rgba(40, 124, 200, 0.17)",
  sourceInside: "rgba(38, 148, 92, 0.27)",
  centre: "#d93b49",
  auxiliary: "#dc8619",
  active: "#a64bb4",
  lens: "#172338",
});

const canvas = document.getElementById("plot");
const ctx = canvas.getContext("2d");
const controls = {
  s: document.getElementById("separation"),
  q: document.getElementById("massRatio"),
  rho: document.getElementById("radius"),
  xs: document.getElementById("sourceX"),
  ys: document.getElementById("sourceY"),
};
const outputs = {
  s: document.getElementById("separationValue"),
  q: document.getElementById("massRatioValue"),
  rho: document.getElementById("radiusValue"),
  xs: document.getElementById("sourceXValue"),
  ys: document.getElementById("sourceYValue"),
};
let state = { ...DEFAULTS };
let scene = null;
let layout = null;
let resizeObserver = null;
let recomputeTimer = null;
let dragMode = null;
let dragPointer = null;
const viewState = {
  source: { zoom: 1, centerX: 0, centerY: 0 },
  image: { zoom: 1, centerX: 0, centerY: 0 },
};
function signed(value, digits) {
  const text = Number(value).toFixed(digits);
  return text.startsWith("-") ? `−${text.slice(1)}` : text;
}

function updateControlText() {
  outputs.s.textContent = state.s.toFixed(2);
  outputs.q.textContent = state.q.toFixed(2);
  outputs.rho.textContent = state.rho.toFixed(3);
  outputs.xs.textContent = signed(state.xs, 3);
  outputs.ys.textContent = signed(state.ys, 3);
  for (const key of Object.keys(controls)) {
    controls[key].value = String(state[key]);
  }
}

function readControls() {
  state = {
    s: Number(controls.s.value),
    q: Number(controls.q.value),
    rho: Number(controls.rho.value),
    xs: Number(controls.xs.value),
    ys: Number(controls.ys.value),
  };
  updateControlText();
}

function lensConfig(values) {
  const m1 = 1 / (1 + values.q);
  const m2 = values.q * m1;
  return {
    ...values,
    m1,
    m2,
    x1: -m2 * values.s,
    x2: m1 * values.s,
  };
}

function lensMap(x, y, p) {
  let sx = x;
  let sy = y;
  for (const [mass, lx] of [[p.m1, p.x1], [p.m2, p.x2]]) {
    const dx = x - lx;
    const dy = y;
    const r2 = dx * dx + dy * dy;
    if (r2 < 1e-12) return { x: Number.NaN, y: Number.NaN };
    sx -= mass * dx / r2;
    sy -= mass * dy / r2;
  }
  return { x: sx, y: sy };
}

function criticalValue(x, y, p) {
  let gx = 0;
  let gy = 0;
  for (const [mass, lx] of [[p.m1, p.x1], [p.m2, p.x2]]) {
    const dx = x - lx;
    const dy = y;
    const r2 = dx * dx + dy * dy;
    if (r2 < 1e-12) return 1e6;
    const r4 = r2 * r2;
    gx += mass * (dy * dy - dx * dx) / r4;
    gy += mass * (-2 * dx * dy) / r4;
  }
  return Math.hypot(gx, gy) - 1;
}

function imageDomain(p) {
  const range = Math.max(
    1.35,
    Math.abs(p.x1) + 1.05,
    Math.abs(p.x2) + 1.05,
    Math.abs(p.xs) + p.rho + 0.9,
    Math.abs(p.ys) + p.rho + 0.9,
  );
  return { minX: -range, maxX: range, minY: -range, maxY: range, range };
}

function interpolatePoint(a, b, va, vb) {
  const denominator = va - vb;
  const t = Math.abs(denominator) < 1e-12 ? 0.5 : va / denominator;
  const clamped = Math.max(0, Math.min(1, t));
  return {
    x: a.x + (b.x - a.x) * clamped,
    y: a.y + (b.y - a.y) * clamped,
  };
}

function contourSegments(p, domain, resolution = 152) {
  const n = resolution;
  const values = new Float64Array((n + 1) * (n + 1));
  const dx = (domain.maxX - domain.minX) / n;
  const dy = (domain.maxY - domain.minY) / n;
  const index = (ix, iy) => iy * (n + 1) + ix;

  for (let iy = 0; iy <= n; iy += 1) {
    const y = domain.minY + iy * dy;
    for (let ix = 0; ix <= n; ix += 1) {
      const x = domain.minX + ix * dx;
      values[index(ix, iy)] = criticalValue(x, y, p);
    }
  }

  const segments = [];
  const edgePoint = (edge, x, y, v00, v10, v11, v01) => {
    const p00 = { x, y };
    const p10 = { x: x + dx, y };
    const p11 = { x: x + dx, y: y + dy };
    const p01 = { x, y: y + dy };
    if (edge === 0) return interpolatePoint(p00, p10, v00, v10);
    if (edge === 1) return interpolatePoint(p10, p11, v10, v11);
    if (edge === 2) return interpolatePoint(p11, p01, v11, v01);
    return interpolatePoint(p01, p00, v01, v00);
  };
  const pairs = [
    [], [[3, 0]], [[0, 1]], [[3, 1]], [[1, 2]],
    [[3, 2], [0, 1]], [[0, 2]], [[3, 2]],
    [[2, 3]], [[2, 0]], [[0, 3], [1, 2]], [[1, 2]],
    [[1, 3]], [[0, 1]], [[3, 0]], [],
  ];

  for (let iy = 0; iy < n; iy += 1) {
    const y = domain.minY + iy * dy;
    for (let ix = 0; ix < n; ix += 1) {
      const x = domain.minX + ix * dx;
      const v00 = values[index(ix, iy)];
      const v10 = values[index(ix + 1, iy)];
      const v11 = values[index(ix + 1, iy + 1)];
      const v01 = values[index(ix, iy + 1)];
      let code = 0;
      if (v00 < 0) code |= 1;
      if (v10 < 0) code |= 2;
      if (v11 < 0) code |= 4;
      if (v01 < 0) code |= 8;
      if (code === 5 || code === 10) {
        // Resolve saddle cells with the cell-centre sign so the contour does
        // not jump unpredictably when q or s changes by a tiny amount.
        const centre = (v00 + v10 + v11 + v01) * 0.25;
        if (code === 5 && centre < 0) pairs[code] = [[3, 0], [1, 2]];
        if (code === 5 && centre >= 0) pairs[code] = [[3, 2], [0, 1]];
        if (code === 10 && centre < 0) pairs[code] = [[0, 1], [2, 3]];
        if (code === 10 && centre >= 0) pairs[code] = [[0, 3], [1, 2]];
      }
      for (const [a, b] of pairs[code]) {
        segments.push({
          a: edgePoint(a, x, y, v00, v10, v11, v01),
          b: edgePoint(b, x, y, v00, v10, v11, v01),
        });
      }
    }
  }
  return { segments, step: Math.max(dx, dy) };
}

function joinContourSegments(segments, step) {
  const tolerance = Math.max(step * 0.55, 1e-4);
  const key = (point) => `${Math.round(point.x / tolerance)},${Math.round(point.y / tolerance)}`;
  const links = new Map();
  const addLink = (point, segment, end) => {
    const pointKey = key(point);
    if (!links.has(pointKey)) links.set(pointKey, []);
    links.get(pointKey).push({ segment, end });
  };
  segments.forEach((segment, i) => {
    addLink(segment.a, i, 0);
    addLink(segment.b, i, 1);
  });

  const used = new Uint8Array(segments.length);
  const result = [];
  const distance = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
  const extend = (polyline, atEnd) => {
    for (;;) {
      const endpoint = atEnd ? polyline[polyline.length - 1] : polyline[0];
      const candidates = links.get(key(endpoint)) || [];
      const next = candidates.find((candidate) => !used[candidate.segment]);
      if (!next) return;
      used[next.segment] = 1;
      const segment = segments[next.segment];
      const other = next.end === 0 ? segment.b : segment.a;
      if (atEnd) polyline.push(other);
      else polyline.unshift(other);
    }
  };

  segments.forEach((segment, i) => {
    if (used[i]) return;
    used[i] = 1;
    const polyline = [segment.a, segment.b];
    extend(polyline, true);
    extend(polyline, false);
    const closed = distance(polyline[0], polyline[polyline.length - 1]) <= step * 4;
    if (closed) polyline.push({ ...polyline[0] });
    if (polyline.length >= 8) result.push({ points: polyline, closed });
  });
  return result;
}

function buildCriticalGeometry(p) {
  const domain = imageDomain(p);
  const { segments, step } = contourSegments(p, domain);
  const curves = joinContourSegments(segments, step);
  const caustics = curves
    .filter((curve) => curve.closed && curve.points.length > 18)
    .map((curve) => curve.points.map((point) => lensMap(point.x, point.y, p)));
  return { domain, criticalCurves: curves, caustics, step };
}

function pointInPolygon(x, y, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const a = polygon[i];
    const b = polygon[j];
    const intersects = ((a.y > y) !== (b.y > y))
      && (x < (b.x - a.x) * (y - a.y) / ((b.y - a.y) || 1e-12) + a.x);
    if (intersects) inside = !inside;
  }
  return inside;
}

function insideCaustic(x, y, caustics) {
  return caustics.some((polygon) => pointInPolygon(x, y, polygon));
}

function sourceRange(p, caustics) {
  let range = Math.max(0.62, Math.abs(p.xs) + p.rho + 0.18, Math.abs(p.ys) + p.rho + 0.18);
  for (const polygon of caustics) {
    for (const point of polygon) {
      range = Math.max(range, Math.abs(point.x) + 0.12, Math.abs(point.y) + 0.12);
    }
  }
  return range;
}

function sampleInverseSupport(p, geometry, resolution = 240) {
  const domain = geometry.domain;
  const outside = [];
  const inside = [];
  const all = [];
  const samplesByGridIndex = new Array(resolution * resolution);
  const hit = new Uint8Array(resolution * resolution);
  for (let iy = 0; iy < resolution; iy += 1) {
    const y = domain.minY + (iy + 0.5) * (domain.maxY - domain.minY) / resolution;
    for (let ix = 0; ix < resolution; ix += 1) {
      const x = domain.minX + (ix + 0.5) * (domain.maxX - domain.minX) / resolution;
      const source = lensMap(x, y, p);
      if (!Number.isFinite(source.x)) continue;
      const distance2 = (source.x - p.xs) ** 2 + (source.y - p.ys) ** 2;
      if (distance2 > p.rho * p.rho) continue;
      const gridIndex = iy * resolution + ix;
      hit[gridIndex] = 1;
      const sample = {
        x,
        y,
        gridIndex,
        sourceInside: insideCaustic(source.x, source.y, geometry.caustics),
      };
      all.push(sample);
      samplesByGridIndex[gridIndex] = sample;
      (sample.sourceInside ? inside : outside).push(sample);
    }
  }

  // Label image-plane support components.  This lets the probe seeds be
  // compared by topology, rather than by a fragile distance threshold: an
  // ordinary image can move a long way while remaining on the same branch.
  const labels = new Int32Array(resolution * resolution);
  const queue = new Int32Array(resolution * resolution);
  let componentCount = 0;
  for (let start = 0; start < hit.length; start += 1) {
    if (!hit[start] || labels[start]) continue;
    componentCount += 1;
    let head = 0;
    let tail = 0;
    queue[tail++] = start;
    labels[start] = componentCount;
    while (head < tail) {
      const current = queue[head++];
      const cx = current % resolution;
      const cy = Math.floor(current / resolution);
      for (let oy = -1; oy <= 1; oy += 1) {
        for (let ox = -1; ox <= 1; ox += 1) {
          if (!ox && !oy) continue;
          const nx = cx + ox;
          const ny = cy + oy;
          if (nx < 0 || nx >= resolution || ny < 0 || ny >= resolution) continue;
          const next = ny * resolution + nx;
          if (hit[next] && !labels[next]) {
            labels[next] = componentCount;
            queue[tail++] = next;
          }
        }
      }
    }
  }
  for (const sample of all) sample.component = labels[sample.gridIndex];
  return {
    outside,
    inside,
    all,
    samplesByGridIndex,
    components: componentCount,
    hit,
    labels,
    resolution,
    cellWidth: (domain.maxX - domain.minX) / resolution,
    cellHeight: (domain.maxY - domain.minY) / resolution,
  };
}

function nearestSupportSample(root, support) {
  let bestDistance = Infinity;
  let bestSample = null;
  for (const sample of support.all) {
    const distance2 = (root.x - sample.x) ** 2 + (root.y - sample.y) ** 2;
    if (distance2 < bestDistance) {
      bestDistance = distance2;
      bestSample = sample;
    }
  }
  return bestSample;
}

function nearestSupportComponent(root, support) {
  return nearestSupportSample(root, support)?.component ?? null;
}

function jacobian(x, y, p) {
  let a = 1;
  let b = 0;
  let c = 0;
  let d = 1;
  for (const [mass, lx] of [[p.m1, p.x1], [p.m2, p.x2]]) {
    const dx = x - lx;
    const dy = y;
    const r2 = dx * dx + dy * dy;
    if (r2 < 1e-10) return null;
    const r4 = r2 * r2;
    const da = (dy * dy - dx * dx) / r4;
    const db = -2 * dx * dy / r4;
    const dd = (dx * dx - dy * dy) / r4;
    a -= mass * da;
    b -= mass * db;
    c -= mass * db;
    d -= mass * dd;
  }
  return { a, b, c, d };
}

function findPointImages(p, target, domain) {
  const starts = [];
  const startCount = 15;
  for (let iy = 0; iy < startCount; iy += 1) {
    const y = domain.minY + (iy + 0.5) * (domain.maxY - domain.minY) / startCount;
    for (let ix = 0; ix < startCount; ix += 1) {
      const x = domain.minX + (ix + 0.5) * (domain.maxX - domain.minX) / startCount;
      starts.push({ x, y });
    }
  }
  for (const lx of [p.x1, p.x2]) {
    for (let k = 0; k < 16; k += 1) {
      const angle = TAU * k / 16;
      starts.push({ x: lx + 0.08 * Math.cos(angle), y: 0.08 * Math.sin(angle) });
      starts.push({ x: lx + 0.25 * Math.cos(angle), y: 0.25 * Math.sin(angle) });
    }
  }

  const roots = [];
  for (const start of starts) {
    let x = start.x;
    let y = start.y;
    let converged = false;
    for (let iteration = 0; iteration < 28; iteration += 1) {
      const mapped = lensMap(x, y, p);
      if (!Number.isFinite(mapped.x)) break;
      const fx = mapped.x - target.x;
      const fy = mapped.y - target.y;
      if (Math.hypot(fx, fy) < 1e-7) {
        converged = true;
        break;
      }
      const j = jacobian(x, y, p);
      if (!j) break;
      const determinant = j.a * j.d - j.b * j.c;
      if (Math.abs(determinant) < 1e-10) break;
      let stepX = (j.d * fx - j.b * fy) / determinant;
      let stepY = (-j.c * fx + j.a * fy) / determinant;
      const stepLength = Math.hypot(stepX, stepY);
      if (stepLength > 0.7) {
        const factor = 0.7 / stepLength;
        stepX *= factor;
        stepY *= factor;
      }
      x -= stepX;
      y -= stepY;
      if (Math.abs(x) > domain.range * 1.35 || Math.abs(y) > domain.range * 1.35) break;
    }
    if (!converged) {
      const mapped = lensMap(x, y, p);
      converged = Number.isFinite(mapped.x)
        && Math.hypot(mapped.x - target.x, mapped.y - target.y) < 3e-6;
    }
    if (!converged) continue;
    if (Math.min(Math.hypot(x - p.x1, y), Math.hypot(x - p.x2, y)) < 1e-4) continue;
    if (roots.every((root) => Math.hypot(root.x - x, root.y - y) > 2e-4)) {
      roots.push({ x, y });
    }
  }
  roots.sort((a, b) => a.x - b.x || a.y - b.y);
  return roots;
}

function chooseInteriorProbe(p, caustics) {
  if (insideCaustic(p.xs, p.ys, caustics)) return null;
  const candidates = [];
  for (let radial = 0.25; radial <= 0.98; radial += 0.09) {
    for (let k = 0; k < 96; k += 1) {
      const angle = TAU * k / 96;
      const x = p.xs + radial * p.rho * Math.cos(angle);
      const y = p.ys + radial * p.rho * Math.sin(angle);
      if (insideCaustic(x, y, caustics)) candidates.push({ x, y, radial });
    }
  }
  if (!candidates.length) return null;
  // Use the first source point just across the caustic.  Its three ordinary
  // images stay close to the centre-image roots, leaving the fold-born pair
  // as the visibly new auxiliary seeds.
  candidates.sort((a, b) => a.radial - b.radial);
  return { x: candidates[0].x, y: candidates[0].y };
}

function buildScene() {
  const p = lensConfig(state);
  const geometry = buildCriticalGeometry(p);
  const support = sampleInverseSupport(p, geometry);
  const probe = chooseInteriorProbe(p, geometry.caustics);
  const centreTarget = { x: p.xs, y: p.ys };
  const centreRoots = findPointImages(p, centreTarget, geometry.domain);
  const probeRoots = probe ? findPointImages(p, probe, geometry.domain) : [];
  const centreComponents = new Set(
    centreRoots.map((root) => nearestSupportComponent(root, support)),
  );
  const auxiliaryRoots = probeRoots.filter((root) => (
    !centreComponents.has(nearestSupportComponent(root, support))
  ));
  return {
    p,
    geometry,
    support,
    probe,
    centreRoots,
    auxiliaryRoots,
    sourceRange: sourceRange(p, geometry.caustics),
    centreInside: insideCaustic(p.xs, p.ys, geometry.caustics),
  };
}

function makeViewport(panel, range) {
  const padding = 34;
  const size = Math.max(80, Math.min(panel.w - padding * 2, panel.h - padding * 2));
  const transform = viewState[panel.kind];
  return {
    ...panel,
    cx: panel.x + panel.w / 2,
    cy: panel.y + panel.h / 2 + 8,
    size,
    scale: size / (2 * range / transform.zoom),
    centerX: transform.centerX,
    centerY: transform.centerY,
    zoom: transform.zoom,
    range,
  };
}

function getLayout(width, height, currentScene) {
  const stacked = width < 720;
  const marginX = stacked ? 14 : 17;
  const top = 42;
  const bottom = 25;
  const gap = stacked ? 40 : 48;
  if (stacked) {
    const panelH = Math.max(180, (height - top - bottom - gap) / 2);
    const panelW = width - marginX * 2;
    const sourcePanel = { kind: "source", x: marginX, y: top, w: panelW, h: panelH };
    const imagePanel = { kind: "image", x: marginX, y: top + panelH + gap, w: panelW, h: panelH };
    return {
      source: makeViewport(sourcePanel, currentScene.sourceRange),
      image: makeViewport(imagePanel, currentScene.geometry.domain.range),
      sourcePanel,
      imagePanel,
      stacked,
    };
  }
  const panelW = (width - marginX * 2 - gap) / 2;
  const panelH = height - top - bottom;
  const sourcePanel = { kind: "source", x: marginX, y: top, w: panelW, h: panelH };
  const imagePanel = { kind: "image", x: marginX + panelW + gap, y: top, w: panelW, h: panelH };
  return {
    source: makeViewport(sourcePanel, currentScene.sourceRange),
    image: makeViewport(imagePanel, currentScene.geometry.domain.range),
    sourcePanel,
    imagePanel,
    stacked,
  };
}

function toScreen(view, point) {
  return {
    x: view.cx + (point.x - view.centerX) * view.scale,
    y: view.cy - (point.y - view.centerY) * view.scale,
  };
}

function fromScreen(view, x, y) {
  return {
    x: view.centerX + (x - view.cx) / view.scale,
    y: view.centerY + (view.cy - y) / view.scale,
  };
}

function drawLinePath(view, points, close = false) {
  if (!points.length) return;
  ctx.beginPath();
  const first = toScreen(view, points[0]);
  ctx.moveTo(first.x, first.y);
  for (let i = 1; i < points.length; i += 1) {
    const point = toScreen(view, points[i]);
    ctx.lineTo(point.x, point.y);
  }
  if (close) ctx.closePath();
}

function drawGrid(view) {
  const left = view.cx - view.size / 2;
  const right = view.cx + view.size / 2;
  const top = view.cy - view.size / 2;
  const bottom = view.cy + view.size / 2;
  ctx.fillStyle = COLORS.panel;
  ctx.fillRect(left, top, view.size, view.size);
  ctx.strokeStyle = COLORS.grid;
  ctx.lineWidth = 1;
  const tick = view.range > 2.1 ? 1 : 0.5;
  const visibleRange = view.range / view.zoom;
  const minCoordinate = Math.floor((view.centerX - visibleRange) / tick) * tick;
  const maxCoordinate = Math.ceil((view.centerX + visibleRange) / tick) * tick;
  for (let value = minCoordinate; value <= maxCoordinate + 1e-8; value += tick) {
    const x = toScreen(view, { x: value, y: 0 }).x;
    const y = toScreen(view, { x: 0, y: value }).y;
    ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, bottom); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(right, y); ctx.stroke();
  }
  ctx.strokeStyle = COLORS.gridStrong;
  const origin = toScreen(view, { x: 0, y: 0 });
  if (origin.y >= top && origin.y <= bottom) {
    ctx.beginPath(); ctx.moveTo(left, origin.y); ctx.lineTo(right, origin.y); ctx.stroke();
  }
  if (origin.x >= left && origin.x <= right) {
    ctx.beginPath(); ctx.moveTo(origin.x, top); ctx.lineTo(origin.x, bottom); ctx.stroke();
  }
}

function drawPolygon(view, polygon, fill, stroke, lineWidth = 1.4, dash = []) {
  drawLinePath(view, polygon, true);
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.save();
  ctx.setLineDash(dash);
  ctx.strokeStyle = stroke;
  ctx.lineWidth = lineWidth;
  ctx.stroke();
  ctx.restore();
}

function drawStar(point, radius, color) {
  ctx.beginPath();
  for (let i = 0; i < 10; i += 1) {
    const angle = -Math.PI / 2 + i * Math.PI / 5;
    const r = i % 2 ? radius * 0.45 : radius;
    const x = point.x + Math.cos(angle) * r;
    const y = point.y + Math.sin(angle) * r;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
  ctx.strokeStyle = COLORS.background;
  ctx.lineWidth = 1.5;
  ctx.stroke();
}

function drawDiamond(point, radius, color) {
  ctx.beginPath();
  ctx.moveTo(point.x, point.y - radius);
  ctx.lineTo(point.x + radius, point.y);
  ctx.lineTo(point.x, point.y + radius);
  ctx.lineTo(point.x - radius, point.y);
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
  ctx.strokeStyle = COLORS.background;
  ctx.lineWidth = 1.4;
  ctx.stroke();
}

function drawSource(view, currentScene) {
  const { p, geometry } = currentScene;
  drawGrid(view);
  for (const caustic of geometry.caustics) {
    drawPolygon(view, caustic, COLORS.causticFill, COLORS.caustic, 1.5, [5, 4]);
  }

  const centre = toScreen(view, { x: p.xs, y: p.ys });
  const radius = p.rho * view.scale;
  ctx.beginPath();
  ctx.arc(centre.x, centre.y, radius, 0, TAU);
  ctx.fillStyle = COLORS.sourceOutside;
  ctx.fill();
  ctx.save();
  ctx.beginPath();
  ctx.arc(centre.x, centre.y, radius, 0, TAU);
  ctx.clip();
  for (const caustic of geometry.caustics) {
    drawLinePath(view, caustic, true);
    ctx.fillStyle = COLORS.sourceInside;
    ctx.fill();
  }
  ctx.restore();
  ctx.beginPath();
  ctx.arc(centre.x, centre.y, radius, 0, TAU);
  ctx.strokeStyle = COLORS.text;
  ctx.lineWidth = 1.4;
  ctx.stroke();
  drawStar(centre, 6, COLORS.centre);

  if (currentScene.probe) {
    drawDiamond(toScreen(view, currentScene.probe), 5, COLORS.auxiliary);
  }

}

function drawImage(view, currentScene) {
  const { p, geometry, support } = currentScene;
  drawGrid(view);

  ctx.save();
  ctx.setLineDash([5, 5]);
  ctx.strokeStyle = COLORS.critical;
  ctx.lineWidth = 1.1;
  for (const curve of geometry.criticalCurves) {
    drawLinePath(view, curve.points);
    ctx.stroke();
  }
  ctx.restore();

  const drawSamples = (samples, color, alpha = 1, predicate = () => true) => {
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.fillStyle = color;
    const width = Math.max(1, support.cellWidth * view.scale * 1.06);
    const height = Math.max(1, support.cellHeight * view.scale * 1.06);
    for (const sample of samples) {
      if (!predicate(sample)) continue;
      const point = toScreen(view, sample);
      ctx.fillRect(point.x - width * 0.5, point.y - height * 0.5, width, height);
    }
    ctx.restore();
  };
  drawSamples(support.outside, COLORS.outside);
  drawSamples(support.inside, COLORS.inside);

  for (const root of currentScene.centreRoots) {
    const point = toScreen(view, root);
    ctx.beginPath();
    ctx.arc(point.x, point.y, 4.3, 0, TAU);
    ctx.fillStyle = COLORS.centre;
    ctx.fill();
    ctx.strokeStyle = COLORS.background;
    ctx.lineWidth = 1.4;
    ctx.stroke();
  }
  for (const root of currentScene.auxiliaryRoots) {
    drawDiamond(toScreen(view, root), 5.2, COLORS.auxiliary);
  }

  const lensRadius = Math.max(3.5, Math.min(8, 0.035 * view.scale));
  for (const x of [p.x1, p.x2]) {
    const point = toScreen(view, { x, y: 0 });
    ctx.beginPath();
    ctx.arc(point.x, point.y, lensRadius, 0, TAU);
    ctx.fillStyle = COLORS.lens;
    ctx.fill();
  }
}

function draw() {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (!width || !height || !scene) return;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = COLORS.background;
  ctx.fillRect(0, 0, width, height);
  layout = getLayout(width, height, scene);
  ctx.save();
  ctx.beginPath();
  ctx.rect(layout.sourcePanel.x, layout.sourcePanel.y, layout.sourcePanel.w, layout.sourcePanel.h);
  ctx.clip();
  drawSource(layout.source, scene);
  ctx.restore();
  ctx.save();
  ctx.beginPath();
  ctx.rect(layout.imagePanel.x, layout.imagePanel.y, layout.imagePanel.w, layout.imagePanel.h);
  ctx.clip();
  drawImage(layout.image, scene);
  ctx.restore();
}

function recompute() {
  window.setTimeout(() => {
    scene = buildScene();
    draw();
  }, 0);
}

function scheduleRecompute() {
  window.clearTimeout(recomputeTimer);
  recomputeTimer = window.setTimeout(recompute, 55);
}

function pointerPosition(event) {
  const rect = canvas.getBoundingClientRect();
  return { x: event.clientX - rect.left, y: event.clientY - rect.top };
}

function panelAt(pointer) {
  if (!layout) return null;
  for (const key of ["source", "image"]) {
    const panel = layout[`${key}Panel`];
    if (
      pointer.x >= panel.x && pointer.x <= panel.x + panel.w
      && pointer.y >= panel.y && pointer.y <= panel.y + panel.h
    ) return key;
  }
  return null;
}

function updateSourceFromPointer(event) {
  if (!layout || !scene) return;
  const pointer = pointerPosition(event);
  const view = layout.source;
  const source = fromScreen(view, pointer.x, pointer.y);
  const x = Math.max(-2, Math.min(2, source.x));
  const y = Math.max(-2, Math.min(2, source.y));
  state.xs = Math.round(x / 0.005) * 0.005;
  state.ys = Math.round(y / 0.005) * 0.005;
  updateControlText();
  scheduleRecompute();
}

canvas.addEventListener("pointerdown", (event) => {
  if (!layout || !scene) return;
  const pointer = pointerPosition(event);
  const panelKey = panelAt(pointer);
  if (!panelKey) return;
  const source = toScreen(layout.source, { x: state.xs, y: state.ys });
  const hitRadius = Math.max(18, state.rho * layout.source.scale + 8);
  if (panelKey === "source" && Math.hypot(pointer.x - source.x, pointer.y - source.y) <= hitRadius) {
    dragMode = "source";
    canvas.style.cursor = "grabbing";
    canvas.setPointerCapture(event.pointerId);
    updateSourceFromPointer(event);
    return;
  }
  dragMode = "pan";
  dragPointer = { ...pointer, panelKey };
  canvas.style.cursor = "grabbing";
  canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener("pointermove", (event) => {
  if (dragMode === "source") {
    updateSourceFromPointer(event);
    return;
  }
  if (dragMode !== "pan" || !dragPointer || !layout) return;
  const pointer = pointerPosition(event);
  const view = layout[dragPointer.panelKey];
  const transform = viewState[dragPointer.panelKey];
  transform.centerX -= (pointer.x - dragPointer.x) / view.scale;
  transform.centerY += (pointer.y - dragPointer.y) / view.scale;
  dragPointer = { ...pointer, panelKey: dragPointer.panelKey };
  draw();
});
canvas.addEventListener("pointerup", (event) => {
  dragMode = null;
  dragPointer = null;
  canvas.style.cursor = "grab";
  if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
});
canvas.addEventListener("pointercancel", () => {
  dragMode = null;
  dragPointer = null;
  canvas.style.cursor = "grab";
});

canvas.addEventListener("wheel", (event) => {
  if (!layout || !scene) return;
  const pointer = pointerPosition(event);
  const panelKey = panelAt(pointer);
  if (!panelKey) return;
  event.preventDefault();
  const view = layout[panelKey];
  const panel = layout[`${panelKey}Panel`];
  const transform = viewState[panelKey];
  const anchor = fromScreen(view, pointer.x, pointer.y);
  const delta = event.deltaMode === 1 ? event.deltaY * 16 : event.deltaY;
  const factor = Math.exp(-delta * (event.ctrlKey ? 0.012 : 0.004));
  const nextZoom = Math.max(1, Math.min(16, transform.zoom * factor));
  const nextScale = view.size / (2 * view.range / nextZoom);
  const panelCentreX = panel.x + panel.w / 2;
  const panelCentreY = panel.y + panel.h / 2 + 8;
  transform.zoom = nextZoom;
  transform.centerX = anchor.x - (pointer.x - panelCentreX) / nextScale;
  transform.centerY = anchor.y + (pointer.y - panelCentreY) / nextScale;
  draw();
}, { passive: false });

canvas.addEventListener("dblclick", (event) => {
  const panelKey = panelAt(pointerPosition(event));
  if (!panelKey) return;
  viewState[panelKey] = { zoom: 1, centerX: 0, centerY: 0 };
  draw();
});

for (const key of Object.keys(controls)) {
  controls[key].addEventListener("input", () => {
    readControls();
    scheduleRecompute();
  });
}

document.getElementById("resetPreset").addEventListener("click", () => {
  state = { ...DEFAULTS };
  viewState.source = { zoom: 1, centerX: 0, centerY: 0 };
  viewState.image = { zoom: 1, centerX: 0, centerY: 0 };
  updateControlText();
  recompute();
});

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.max(1, Math.round(rect.width * pixelRatio));
  canvas.height = Math.max(1, Math.round(rect.height * pixelRatio));
  ctx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  draw();
}

if ("ResizeObserver" in window) {
  resizeObserver = new ResizeObserver(resizeCanvas);
  resizeObserver.observe(canvas);
} else {
  window.addEventListener("resize", resizeCanvas);
}

updateControlText();
recompute();
