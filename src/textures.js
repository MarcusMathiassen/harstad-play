// Procedural textures, drawn once at load on OffscreenCanvases in the render worker. These are the same
// drawings src/world.js makes, kept in JavaScript on purpose: they are Canvas 2D calls (gradients, curves,
// compositing), there is nothing here for C to speed up, and they run while the build worker is busy.
// Every function returns { width, height, data: Uint8Array of RGBA rows, first row at v = 0 }.

export function mulberry32(seed) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// `gpu`: drawn as src/world.js draws it, on a canvas left to the graphics card, whose edges come out a shade apart
// from the software rasteriser's. The facades need it: a window cell's light is weighed against a threshold.
function canvas(w, h, gpu) {
  const c = new OffscreenCanvas(w, h);
  return [c, c.getContext('2d', { willReadFrequently: !gpu })];
}
const pixels = (ctx, w, h) => new Uint8Array(ctx.getImageData(0, 0, w, h).data.buffer);

// 4x4 cells of wall (3.4 m x 2.8 m each): three rows with one window per cell, and a top row (v >= 0.75) of
// bare cladding for gables, knee walls and outbuildings. Returns [colour, glow]; colour alpha = 1 where the
// wall colour may tint. `timber` gives clapboard with cased sash windows, else rendered blocks.
// The light each cell of a facade atlas gives off through its glow map, linear and averaged over the cell, as 16 cells
// of r g b indexed [row up the texture * 4 + column] (src/world.js, cellLight): the C lays each lit window's patch on
// the snow from it, so the light outside matches the rooms the wall shows lit.
export function cellLight({ data, width: S }) {
  const C = S / 4, cells = new Float64Array(48);
  const lin = Array.from({ length: 256 }, (_, v) => { v /= 255; return v < 0.04045 ? v * 0.0773993808 : Math.pow(v * 0.9478672986 + 0.0521327014, 2.4); });
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) {
    const e = [0, 0, 0];
    for (let y = r * C; y < (r + 1) * C; y++) for (let o = (y * S + c * C) * 4, end = o + C * 4; o < end; o += 4) {
      e[0] += lin[data[o]]; e[1] += lin[data[o + 1]]; e[2] += lin[data[o + 2]];
    }
    cells.set(e.map(v => v / (C * C)), (r * 4 + c) * 3);
  }
  return cells;
}

// `glowOnly`: for the build worker, which weighs the glow's cells and never looks at the colour: that is painted all the same
// (the two share one run of random numbers) but not read back.
export function facadeTextures(timber, glowOnly) {
  const S = 1024, N = 4, C = S / N;
  const ctxs = [0, 1, 2].map(() => canvas(S, S, true)[1]);
  const [col, mask, glow] = ctxs;
  // rows are not flipped on upload, so flip here and draw with y pointing down the wall
  for (const c of ctxs) { c.translate(0, S); c.scale(1, -1); }
  const rng = mulberry32(timber ? 3 : 4);
  col.fillStyle = '#fff'; col.fillRect(0, 0, S, S);
  mask.fillStyle = '#fff'; mask.fillRect(0, 0, S, S);
  glow.fillStyle = '#000'; glow.fillRect(0, 0, S, S);
  if (timber) {
    for (let y = 0; y < S; y += 16) { // overlapping boards: shadow under each lap, highlight on its edge
      col.fillStyle = 'rgba(0,0,0,.24)'; col.fillRect(0, y, S, 2);
      col.fillStyle = 'rgba(0,0,0,.07)'; col.fillRect(0, y + 2, S, 4);
      col.fillStyle = 'rgba(255,255,255,.5)'; col.fillRect(0, y + 14, S, 1);
    }
    for (let i = 0; i < 500; i++) { // board joints and weathering
      col.fillStyle = `rgba(0,0,0,${0.03 + rng() * 0.05})`;
      col.fillRect(rng() * S, (rng() * 64 | 0) * 16, 1 + rng() * 2, 16);
    }
  } else {
    for (let i = 0; i < 6000; i++) { // render grain and faint streaks
      col.fillStyle = `rgba(0,0,0,${rng() * 0.05})`;
      col.fillRect(rng() * S, rng() * S, 2 + rng() * 3, 2 + rng() * 14);
    }
    for (let y = 0; y < S; y += C) { col.fillStyle = 'rgba(0,0,0,.2)'; col.fillRect(0, y + C - 5, S, 5); }
  }

  const WARM = ['#ffb65c', '#ffc77a', '#ffa94a', '#ffd9a0'], xmas = mulberry32(timber ? 24 : 25);
  for (let cy = 1; cy < N; cy++) for (let cx = 0; cx < N; cx++) {
    const w = timber ? 88 : 150, h = timber ? 122 : 112;
    const x = cx * C + (C - w) / 2, y = cy * C + (timber ? 58 : 66);
    const trim = timber ? 10 : 5;
    mask.fillStyle = '#000'; mask.fillRect(x - trim - 3, y - trim - 3, w + trim * 2 + 6, h + trim * 2 + 10);
    if (timber) {
      col.fillStyle = '#efece3'; col.fillRect(x - trim, y - trim - 3, w + trim * 2, h + trim * 2 + 3);
      col.fillStyle = '#fbf9f3'; col.fillRect(x - trim - 3, y + h + trim - 2, w + trim * 2 + 6, 7); // sill
      col.fillStyle = 'rgba(0,0,0,.3)'; col.fillRect(x - trim - 3, y + h + trim + 5, w + trim * 2 + 6, 2);
    } else {
      col.fillStyle = '#2e3338'; col.fillRect(x - trim, y - trim, w + trim * 2, h + trim * 2);
      col.fillStyle = '#c9cdd1'; col.fillRect(x - trim - 3, y + h + trim, w + trim * 2 + 6, 5);
    }
    const lit = rng() < (timber ? 0.42 : 0.36);
    const tv = !lit && rng() < 0.07;
    const night = col.createLinearGradient(0, y, 0, y + h);
    night.addColorStop(0, '#0a0f18'); night.addColorStop(1, '#17212e');
    col.fillStyle = night; col.fillRect(x, y, w, h);
    if (lit || tv) {
      const dim = 0.55 + rng() * 0.45, tone = tv ? '#7fa6ff' : WARM[rng() * WARM.length | 0];
      for (const c of [col, glow]) {
        const g = c.createLinearGradient(0, y, 0, y + h);
        g.addColorStop(0, tone); g.addColorStop(1, tv ? '#3c5fae' : '#e8862e');
        c.globalAlpha = tv ? 0.5 : dim;
        c.fillStyle = g; c.fillRect(x, y, w, h);
        c.globalAlpha = 1;
      }
      const style = rng(), blind = 0.3 + rng() * 0.3, sill = rng() < 0.5, sx = x + w * (0.25 + rng() * 0.5);
      if (!tv) for (const c of [col, glow]) {
        if (style < 0.45) { // curtains drawn back to either side
          c.fillStyle = c === col ? 'rgba(255,240,220,.75)' : 'rgba(120,96,70,.6)';
          for (const s of [0, 1]) {
            c.beginPath();
            c.moveTo(x + s * w, y); c.lineTo(x + s * w + (s ? -1 : 1) * w * 0.34, y);
            c.quadraticCurveTo(x + s * w + (s ? -1 : 1) * w * 0.3, y + h * 0.6, x + s * w, y + h);
            c.fill();
          }
        } else if (style < 0.7) { // half-lowered blind
          c.fillStyle = c === col ? 'rgba(250,238,215,.8)' : 'rgba(110,90,66,.65)';
          c.fillRect(x, y, w, h * blind);
        }
        if (sill) { // something on the sill: a lamp or a pot plant in silhouette
          c.fillStyle = c === col ? '#2a1c12' : '#000';
          c.beginPath(); c.arc(sx, y + h - 12, 9, 0, 6.3); c.fill();
          c.fillRect(sx - 5, y + h - 8, 10, 8);
        }
      }
    } else { // a cold reflection of the sky
      col.fillStyle = 'rgba(120,150,200,.10)';
      col.beginPath(); col.moveTo(x, y + h * 0.7); col.lineTo(x + w * 0.6, y); col.lineTo(x + w, y); col.lineTo(x, y + h); col.fill();
    }
    for (const c of [col, glow]) { // glazing bars
      c.fillStyle = c === col ? (timber ? '#efece3' : '#2e3338') : '#000';
      if (timber) { c.fillRect(x + w / 2 - 3, y, 6, h); c.fillRect(x, y + h * 0.36, w, 5); }
      else { c.fillRect(x + w / 3 - 2, y, 4, h); c.fillRect(x + w * 2 / 3 - 2, y, 4, h); }
    }
    // Advent: a paper star hung in the window, or a seven-armed candle bridge on the sill. Own random
    // stream, so which rooms are lit stays as it was.
    const advent = xmas();
    if (advent < 0.5) for (const c of [col, glow]) {
      if (advent < 0.28) {
        const sx = x + w * (timber ? 0.27 : 0.5), sy = y + h * 0.3, r = timber ? 15 : 19;
        c.fillStyle = advent < 0.1 ? '#ff5236' : advent < 0.2 ? '#ffb347' : '#fff1cf';
        c.beginPath();
        for (let i = 0; i < 14; i++) {
          const a = i / 14 * Math.PI * 2 - Math.PI / 2, q = i % 2 ? r * 0.45 : r;
          c.lineTo(sx + Math.cos(a) * q, sy + Math.sin(a) * q);
        }
        c.fill();
      } else {
        c.fillStyle = '#ffe7a6';
        for (let i = -3; i <= 3; i++) { c.beginPath(); c.arc(x + w / 2 + i * 10, y + h - 9 - (3 - Math.abs(i)) * 6, 3.2, 0, 6.3); c.fill(); }
      }
    }
  }
  if (glowOnly) return [null, { width: S, height: S, data: pixels(glow, S, S) }];
  const colour = pixels(col, S, S), m = pixels(mask, S, S);
  for (let i = 0; i < colour.length; i += 4) colour[i + 3] = m[i]; // alpha from the mask, so it is never premultiplied
  return [{ width: S, height: S, data: colour }, { width: S, height: S, data: pixels(glow, S, S) }];
}

/* ---- facade styles (src/sentrum.js) ----
 * Atlases laid out as facadeTextures: 4 x 4 cells of 3.4 m x 2.8 m, rows of windows below a top row
 * of bare wall (v >= 0.75). Colour alpha is where the building's own colour tints; glow is what lights at night. */

// boards_v, boards_h: bare timber cladding, standing and lying boards, no windows (src/dalsletta.js puts its own in)
export const STYLES = ['classic', 'brick', 'ribbon', 'fins', 'panel', 'glass', 'boards_v', 'boards_h'];

function styleTextures(name, seed, glowOnly) {
  const S = 512, N = 4, C = S / N, k = S / 1024; // drawn in the units of world.js's 1024 px atlas
  const ctxs = [0, 1, 2].map(() => canvas(S, S)[1]);
  const [cx, mask, glow] = ctxs;
  for (const c of ctxs) { c.translate(0, S); c.scale(1, -1); c.scale(k, k); } // y down the wall, as world.js draws
  const rng = mulberry32(seed);
  const U = 1024, CU = U / N;
  cx.fillStyle = '#fff'; cx.fillRect(0, 0, U, U);
  mask.fillStyle = '#fff'; mask.fillRect(0, 0, U, U);
  glow.fillStyle = '#000'; glow.fillRect(0, 0, U, U);
  const both = (f) => { for (const c of [cx, glow]) f(c, c === glow); };
  const rect = (c, x, y, w, h, s) => { c.fillStyle = s; c.fillRect(x, y, w, h); };
  const noMask = (x, y, w, h) => rect(mask, x, y, w, h, '#000');

  // the wall
  if (name === 'brick') {
    for (let y = 0; y < U; y += 16) for (let x = -(y / 16 % 2) * 22; x < U; x += 44) {
      const v = 200 + rng() * 40 | 0;
      rect(cx, x + 1, y + 1, 42, 14, `rgb(${v},${v - 8},${v - 14})`);
    }
    for (let y = 0; y < U; y += 16) rect(cx, 0, y, U, 1, 'rgba(255,255,255,.6)');
  } else if (name === 'boards_v') { // standing boards, 0.2 m, a narrow batten over every joint (75 px a metre)
    for (let x = 0; x < U; x += 15) { rect(cx, x, 0, 5, U, 'rgba(255,255,255,.5)'); rect(cx, x + 5, 0, 2, U, 'rgba(0,0,0,.32)'); rect(cx, x + 7, 0, 8, U, `rgba(0,0,0,${0.02 + rng() * 0.04})`); }
  } else if (name === 'boards_h') { // lying weatherboards, 0.17 m courses, each lapping the one below with a shadow line
    for (let y = 0; y < U; y += 13) { rect(cx, 0, y, U, 3, 'rgba(0,0,0,.3)'); rect(cx, 0, y + 3, U, 10, `rgba(255,255,255,${0.1 + rng() * 0.1})`); }
  } else if (name === 'panel') { // large fibre-cement panels with open joints
    for (let y = 0; y < U; y += CU / 2) rect(cx, 0, y, U, 4, 'rgba(0,0,0,.28)');
    for (let x = 0; x < U; x += CU / 2) rect(cx, x, 0, 4, U, 'rgba(0,0,0,.22)');
  } else {
    for (let i = 0; i < 5000; i++) rect(cx, rng() * U, rng() * U, 2 + rng() * 3, 2 + rng() * 12, `rgba(0,0,0,${rng() * 0.05})`);
  }
  if (name === 'classic') for (let y = 0; y < U; y += CU) rect(cx, 0, y + CU - 12, U, 12, 'rgba(255,255,255,.55)'), rect(cx, 0, y + CU - 2, U, 4, 'rgba(0,0,0,.25)'); // string course at each floor

  const WARM = ['#ffb65c', '#ffc77a', '#ffa94a', '#ffd9a0'], COOL = ['#dfe8f0', '#cfdbe6', '#f0f0e6'];
  const lights = (x, y, w, h, share, office) => { // a pane, dark or lit, with what shows inside
    const g = cx.createLinearGradient(0, y, 0, y + h);
    g.addColorStop(0, '#0a0f18'); g.addColorStop(1, '#16202c');
    rect(cx, x, y, w, h, g);
    if (rng() < share) {
      const tone = office ? COOL[rng() * COOL.length | 0] : WARM[rng() * WARM.length | 0], dim = 0.55 + rng() * 0.4;
      both((c, gl) => { c.globalAlpha = dim * (office ? 0.75 : 1); rect(c, x, y, w, h, tone); c.globalAlpha = 1;
        if (!office && rng() < 0.5) { c.fillStyle = gl ? 'rgba(110,90,66,.6)' : 'rgba(250,238,215,.8)'; c.fillRect(x, y, w, h * (0.25 + rng() * 0.3)); } // a blind
        if (office) { c.fillStyle = gl ? '#000' : '#2a2e33'; for (let q = x + w / 4; q < x + w; q += w / 4) c.fillRect(q - 1, y, 2, h); } });
    } else {
      cx.fillStyle = 'rgba(120,150,200,.1)'; cx.beginPath(); cx.moveTo(x, y + h * 0.7); cx.lineTo(x + w * 0.6, y); cx.lineTo(x + w, y); cx.lineTo(x, y + h); cx.fill();
    }
  };
  for (let cy = 1; cy < N; cy++) for (let cxl = 0; cxl < N; cxl++) {
    const x0 = cxl * CU, y0 = cy * CU;
    if (name === 'classic' || name === 'brick') { // tall sash windows in a moulded surround, a sill, a head
      const w = 112, h = 150, x = x0 + (CU - w) / 2, y = y0 + 52;
      noMask(x - 16, y - 22, w + 32, h + 40);
      rect(cx, x - 14, y - 20, w + 28, h + 36, name === 'brick' ? '#e9e4d8' : '#f4f1ea');
      rect(cx, x - 18, y + h + 6, w + 36, 10, '#faf8f2'); rect(cx, x - 18, y + h + 16, w + 36, 3, 'rgba(0,0,0,.3)');
      if (name === 'brick') { cx.fillStyle = '#e9e4d8'; cx.beginPath(); cx.arc(x + w / 2, y, w / 2 + 14, Math.PI, 0); cx.fill(); noMask(x - 16, y - w / 2 - 18, w + 32, w / 2 + 20); }
      else rect(cx, x - 22, y - 30, w + 44, 12, '#faf8f2'); // a cornice over the window
      lights(x, y, w, h, 0.42, false);
      both((c, gl) => { rect(c, x + w / 2 - 4, y, 8, h, gl ? '#000' : '#f4f1ea'); rect(c, x, y + h * 0.36, w, 6, gl ? '#000' : '#f4f1ea'); });
    } else if (name === 'ribbon') { // continuous window bands under solid spandrels
      const y = y0 + 70, h = 120;
      noMask(x0, y - 6, CU, h + 12);
      rect(cx, x0, y - 6, CU, h + 12, '#2c3036');
      for (const [x, w] of [[x0 + 4, CU / 2 - 6], [x0 + CU / 2 + 2, CU / 2 - 6]]) lights(x, y, w, h, 0.3, true);
      rect(cx, x0, y + h + 6, CU, 18, 'rgba(0,0,0,.18)');
    } else if (name === 'fins') { // narrow windows between deep vertical fins
      for (const x of [x0 + 16, x0 + CU / 2 + 16]) {
        const w = CU / 2 - 44, y = y0 + 40, h = 190;
        noMask(x - 4, y - 4, w + 8, h + 8);
        rect(cx, x - 4, y - 4, w + 8, h + 8, '#2a2e33');
        lights(x, y, w, h, 0.28, true);
      }
      for (const x of [x0, x0 + CU / 2]) { rect(cx, x - 6, y0, 12, CU, 'rgba(255,255,255,.6)'); rect(cx, x + 6, y0, 6, CU, 'rgba(0,0,0,.35)'); }
    } else if (name === 'panel') { // square windows punched in the panels, deep reveals
      const w = 140, h = 140, x = x0 + (CU - w) / 2, y = y0 + 56;
      noMask(x - 8, y - 8, w + 16, h + 16);
      rect(cx, x - 8, y - 8, w + 16, h + 16, '#3a3f45');
      lights(x, y, w, h, 0.5, false);
      both((c, gl) => rect(c, x + w * 0.62, y, 6, h, gl ? '#000' : '#3a3f45'));
    } else if (name === 'glass') { // curtain wall: mullions and transoms on a grid, every pane glass
      noMask(x0, y0, CU, CU);
      for (const [x, w] of [[x0, CU / 2], [x0 + CU / 2, CU / 2]]) lights(x + 4, y0 + 4, w - 8, CU - 8, 0.45, rng() < 0.6);
      // the frames take the building's own colour (the Kulturhus tower's are red), the glass does not
      for (const c of [cx, glow, mask]) { const m = c === glow ? '#000' : '#fff'; rect(c, x0, y0, CU, 8, m); rect(c, x0, y0, 8, CU, m); rect(c, x0 + CU / 2 - 4, y0, 8, CU, m); }
    }
  }
  if (glowOnly) return [null, { width: S, height: S, data: pixels(glow, S, S) }];
  const colour = pixels(cx, S, S), alpha = pixels(mask, S, S);
  for (let i = 0; i < colour.length; i += 4) colour[i + 3] = alpha[i];
  return [{ width: S, height: S, data: colour }, { width: S, height: S, data: pixels(glow, S, S) }];
}

// One [colour, glow] pair per entry of STYLES, layers 2 and up of the walls' atlases.
export function facadeStyles(glowOnly) {
  return STYLES.map((name, i) => styleTextures(name, 51 + i, glowOnly));
}

// Winter road surface, u across the carriageway, v along it. kind 0: a side street, packed snow with gritted, greyer
// tracks. 1: a main road, salted down to wet asphalt in the wheel tracks of its two lanes. 2: one salted lane on its own (a
// one-way carriageway). Paint is not in the texture: road.c lays the lines, and knows where they stop.
export function roadTexture(kind) {
  const W = 256, H = 1024;
  const [, x] = canvas(W, H);
  const rng = mulberry32(kind === 1 ? 11 : kind === 2 ? 13 : 12);
  // wandering function of y that tiles over the texture height: long swells only, so an edge meanders and never saws
  const wander = () => {
    const k = [1, 2, 3, 5, 8].map(f => [f, rng() * 6.283, (0.4 + rng()) / f]);
    return y => k.reduce((s, [f, ph, a]) => s + Math.sin(6.283185 * f * y / H + ph) * a, 0) / 1.6;
  };
  const band = (u, half, rgb, alpha, wobble) => { // stripe along the road with uneven, smoothly wandering, soft edges
    const centre = wander(), width = wander();
    for (const [grow, share] of [[1.14, 0.6], [0.9, 0.8]]) { // a wider faint pass under a narrower one: the edge feathers
      x.fillStyle = `rgba(${rgb},${alpha * share})`;
      x.beginPath();
      for (let y = -4; y <= H + 4; y += 4) x.lineTo((u + centre(y) * wobble - half * grow * (1 + 0.28 * width(y))) * W, y);
      for (let y = H + 4; y >= -4; y -= 4) x.lineTo((u + centre(y) * wobble + half * grow * (1 + 0.28 * width(y))) * W, y);
      x.fill();
    }
  };
  if (kind) {
    x.fillStyle = '#3b3e45'; x.fillRect(0, 0, W, H);
    for (let i = 0; i < 7000; i++) {
      const v = 45 + rng() * 60 | 0;
      x.fillStyle = `rgba(${v},${v},${v + 6},${0.3 + rng() * 0.3})`;
      x.fillRect(rng() * W, rng() * H, 1 + rng() * 2, 1 + rng() * 3);
    }
    if (kind === 1) for (const u of [0.25, 0.5, 0.75]) { // slush left between the wheel tracks, heaviest on the centre line
      band(u, 0.045, '140,150,162', u === 0.5 ? 0.42 : 0.2, 0.012);
      band(u, 0.02, '200,210,221', u === 0.5 ? 0.6 : 0.22, 0.01);
    } else { // one lane: a strip of slush between its two tracks
      band(0.5, 0.05, '140,150,162', 0.16, 0.014);
      band(0.5, 0.02, '200,210,221', 0.14, 0.012);
    }
    const edge = kind === 1 ? 1 : 1.5; // the ploughed margins are as wide in metres on the narrower lane
    band(0.03 * edge, 0.08 * edge, '226,233,241', 0.95, 0.008); band(1 - 0.03 * edge, 0.08 * edge, '226,233,241', 0.95, 0.008);
    band(0.105 * edge, 0.03 * edge, '160,168,178', 0.5, 0.012); band(1 - 0.105 * edge, 0.03 * edge, '160,168,178', 0.5, 0.012);
  } else {
    x.fillStyle = '#d3dae3'; x.fillRect(0, 0, W, H);
    for (let i = 0; i < 9000; i++) {
      const v = 150 + rng() * 90 | 0;
      x.fillStyle = `rgba(${v},${v + 4},${v + 10},${0.2 + rng() * 0.3})`;
      x.fillRect(rng() * W, rng() * H, 1 + rng() * 3, 1 + rng() * 4);
    }
    for (const u of [0.3, 0.7]) { // polished, gritted tracks
      band(u, 0.085, '120,126,134', 0.3, 0.015);
      band(u, 0.045, '96,98,102', 0.26, 0.012);
    }
    for (let i = 0; i < 2500; i++) { // strewn grit
      x.fillStyle = `rgba(70,60,50,${0.25 + rng() * 0.3})`;
      x.fillRect((0.2 + rng() * 0.6) * W, rng() * H, 1.5, 1.5);
    }
  }
  const fade = x.createLinearGradient(0, 0, W, 0); // feather the verge into the snow
  fade.addColorStop(0, 'rgba(0,0,0,0)'); fade.addColorStop(0.04, 'rgba(0,0,0,1)');
  fade.addColorStop(0.96, 'rgba(0,0,0,1)'); fade.addColorStop(1, 'rgba(0,0,0,0)');
  x.globalCompositeOperation = 'destination-in';
  x.fillStyle = fade; x.fillRect(0, 0, W, H);
  // three uploaded this canvas bottom row first; do the same so the surface runs the same way along the road
  const top = pixels(x, W, H), data = new Uint8Array(top.length);
  for (let y = 0; y < H; y++) data.set(top.subarray((H - 1 - y) * W * 4, (H - y) * W * 4), y * W * 4);
  return { width: W, height: H, data };
}

export function noiseTexture() {
  const S = 256, d = new Uint8Array(S * S * 4), rng = mulberry32(5);
  for (let i = 0; i < S * S; i++) {
    const v = 96 + rng() * 64 + (rng() < 0.08 ? rng() * 90 - 45 : 0) | 0;
    d[i * 4] = d[i * 4 + 1] = d[i * 4 + 2] = v; d[i * 4 + 3] = 255;
  }
  return { width: S, height: S, data: d };
}

// Tileable snow relief, stored as its slope (dh/du, dh/dv in red and green, 0.5 = level) so mip-mapping flattens it
// with distance instead of letting it shimmer. Rolling swells under sharp-crested ridges: what wind makes of loose
// snow. `slope` is the steepest slope, in height units per tile, that 1.0 stands for. As snowRelief in src/world.js.
let relief = null;
export function snowRelief() {
  if (relief) return relief;
  const S = 256, rng = mulberry32(31), waves = [], h = new Float32Array(S * S);
  for (let k = 0; k < 28; k++) {
    const fx = Math.round((rng() * 2 - 1) * 9), fy = Math.round((rng() * 2 - 1) * 9) || 1;
    waves.push([fx, fy, rng() * 6.283, Math.pow(Math.hypot(fx, fy), -1.3)]);
  }
  for (let j = 0; j < S; j++) for (let i = 0; i < S; i++) {
    let s = 0;
    for (const [fx, fy, ph, a] of waves) s += Math.sin(6.283185 * (fx * i + fy * j) / S + ph) * a;
    h[j * S + i] = s;
  }
  let lo = Infinity, hi = -Infinity;
  for (const v of h) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  for (let k = 0; k < h.length; k++) { // half swell, half ridge: the crests where the field crosses its middle come out sharp
    const s = (h[k] - lo) / (hi - lo) * 2 - 1;
    h[k] = 0.5 * s + 0.5 * (1 - Math.abs(s));
  }
  const gx = new Float32Array(S * S), gy = new Float32Array(S * S);
  let top = 0;
  for (let j = 0; j < S; j++) for (let i = 0; i < S; i++) {
    const k = j * S + i;
    gx[k] = (h[j * S + (i + 1 & S - 1)] - h[j * S + (i - 1 & S - 1)]) * S / 2;
    gy[k] = (h[(j + 1 & S - 1) * S + i] - h[(j - 1 & S - 1) * S + i]) * S / 2;
    top = Math.max(top, Math.abs(gx[k]), Math.abs(gy[k]));
  }
  const d = new Uint8Array(S * S * 4);
  for (let k = 0; k < S * S; k++) { d[k * 4] = 127.5 + gx[k] / top * 127; d[k * 4 + 1] = 127.5 + gy[k] / top * 127; d[k * 4 + 3] = 255; }
  return relief = { width: S, height: S, data: d, slope: top };
}

// Folkeparken's trails: four surfaces across one image. Groomed for skating: corduroy, with a classic track pressed
// in along the right-hand edge. Classic alone: one pair of grooves down the middle of a narrow packed strip.
// Trodden: boot prints, where the C side's random stream says (`prints`, x y pairs). Landing hill: packed smooth.
// Each fades out at its edges into the snow lying around it.
export function trailTexture(prints) {
  const BANDS = 4, LANDING = 3, W = 256, H = 128;
  const [, x] = canvas(W * BANDS, H);
  const strip = (band, alpha, tone) => {
    const g = x.createLinearGradient(band * W, 0, (band + 1) * W, 0), rgb = `${tone},${tone + 5},${tone + 14}`;
    g.addColorStop(0.02, `rgba(${rgb},0)`); g.addColorStop(0.1, `rgba(${rgb},${alpha})`);
    g.addColorStop(0.9, `rgba(${rgb},${alpha})`); g.addColorStop(0.98, `rgba(${rgb},0)`);
    x.fillStyle = g; x.fillRect(band * W, 0, W, H);
  };
  const corduroy = (band, from, to, pitch) => {
    x.fillStyle = 'rgba(150,165,200,0.2)';
    for (let u = from; u < to; u += pitch) x.fillRect(band * W + u, 0, pitch / 2, H);
  };
  const grooves = (band, at, gap, wide) => {
    for (const s of [-1, 1]) {
      x.fillStyle = 'rgba(110,126,165,0.7)'; x.fillRect(band * W + at + s * gap - wide / 2, 0, wide, H);
      x.fillStyle = 'rgba(250,252,255,0.7)'; x.fillRect(band * W + at + s * gap + wide / 2, 0, 1.5, H); // the far wall catches the light
    }
  };
  strip(0, 0.92, 236); corduroy(0, 26, 186, 4); grooves(0, 212, 6, 4);   // 5 m: 2 cm a texel
  strip(1, 0.9, 234); corduroy(1, 30, 226, 8); grooves(1, 128, 12, 7);   // 2.6 m: 1 cm a texel
  strip(2, 0.5, 214);
  x.fillStyle = 'rgba(120,135,170,0.4)';
  for (let i = 0; i + 1 < prints.length; i += 2) x.fillRect(2 * W + prints[i], prints[i + 1], 22, 4); // a boot: 12 cm by 25 cm
  strip(LANDING, 0.94, 240);
  return { width: W * BANDS, height: H, data: pixels(x, W * BANDS, H) };
}

// Tileable fractal value noise: the ragged, wispy edge of each cloud puff.
export function cloudNoise() {
  const S = 128, d = new Uint8Array(S * S * 4), rng = mulberry32(31);
  const octave = n => {
    const g = new Float32Array(n * n);
    for (let i = 0; i < g.length; i++) g[i] = rng();
    return (u, v) => {
      const x = u * n, y = v * n, i = Math.floor(x), j = Math.floor(y);
      let fx = x - i, fy = y - j;
      fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy);
      const at = (a, b) => g[(b % n) * n + (a % n)];
      return (at(i, j) * (1 - fx) + at(i + 1, j) * fx) * (1 - fy) + (at(i, j + 1) * (1 - fx) + at(i + 1, j + 1) * fx) * fy;
    };
  };
  const octaves = [4, 8, 16, 32].map(octave);
  for (let j = 0; j < S; j++) for (let i = 0; i < S; i++) {
    let v = 0, a = 0.5;
    for (const o of octaves) { v += o(i / S, j / S) * a; a *= 0.5; }
    const k = (j * S + i) * 4;
    d[k] = d[k + 1] = d[k + 2] = v / 0.9375 * 255; d[k + 3] = 255;
  }
  return { width: S, height: S, data: d };
}

// A tileable cell pattern for the sea's floes (src/water.js, floeTexture): R nearness to the cell's centre (1 there, 0
// half a cell out), G the cell's own random.
export function floePattern() {
  const N = 128, C = 6, pts = [];
  let seed = 77; const rnd = () => (seed = Math.imul(seed ^ seed >>> 15, 2246822519) + 0x9e3779b9 | 0, (seed >>> 0) / 4294967296);
  for (let j = 0; j < C; j++) for (let i = 0; i < C; i++) pts.push([(i + 0.15 + rnd() * 0.7) / C, (j + 0.15 + rnd() * 0.7) / C, rnd()]);
  const d = new Uint8Array(N * N * 4);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const u = (x + 0.5) / N, v = (y + 0.5) / N;
    let a = 9, id = 0;
    for (const [px, py, r] of pts) for (let oy = -1; oy <= 1; oy++) for (let ox = -1; ox <= 1; ox++) {
      const q = Math.hypot(u - px - ox, v - py - oy);
      if (q < a) { a = q; id = r; }
    }
    const o = (y * N + x) * 4;
    d[o] = Math.max(0, 1 - a * C * 2) * 255; d[o + 1] = id * 255; d[o + 3] = 255;
  }
  return { width: N, height: N, data: d };
}

// Tunnel surfaces, as src/tunnels.js draws them: bare wet asphalt, and the two kinds of wall (u runs up one wall,
// over the ceiling and down the other). One random stream runs through all three. Returned at one size, so they
// can share an array texture: [floor, rock tunnel wall, lidded tunnel wall].
export function tunnelTextures() {
  const rng = mulberry32(83), W = 512, H = 256;
  const texture = (w, h, draw) => {
    const [c, x] = canvas(w, h);
    draw(x);
    // the luminaire sits over the middle of each repeat: what is far from it along the tunnel is dimmer
    const g = x.createLinearGradient(0, 0, 0, h);
    g.addColorStop(0, '#9a9a9a'); g.addColorStop(0.5, '#fff'); g.addColorStop(1, '#9a9a9a');
    x.globalCompositeOperation = 'multiply';
    x.fillStyle = g; x.fillRect(0, 0, w, h);
    const [, out] = canvas(W, H);
    out.drawImage(c, 0, 0, W, H);
    const top = pixels(out, W, H), data = new Uint8Array(top.length); // bottom row first, as three uploaded it
    for (let y = 0; y < H; y++) data.set(top.subarray((H - 1 - y) * W * 4, (H - y) * W * 4), y * W * 4);
    return { width: W, height: H, data };
  };
  const floor = texture(256, 256, x => {
    x.fillStyle = '#34363a'; x.fillRect(0, 0, 256, 256);
    for (let i = 0; i < 2600; i++) {
      const v = 40 + rng() * 40 | 0;
      x.fillStyle = `rgba(${v},${v},${v + 4},.5)`;
      x.fillRect(rng() * 256, rng() * 256, 1, 1);
    }
    for (const u of [0.28, 0.72]) { x.fillStyle = 'rgba(20,21,24,.35)'; x.fillRect(u * 256 - 22, 0, 44, 256); } // wheel tracks
    x.fillStyle = '#c9c9c2'; x.fillRect(6, 0, 4, 256); x.fillRect(246, 0, 4, 256);
    x.fillStyle = '#c9a53a'; x.fillRect(123, 0, 3, 256); x.fillRect(130, 0, 3, 256);
  });
  const wall = bands => texture(512, 256, x => {
    bands.forEach(([u, colour], i) => {
      x.fillStyle = colour;
      x.fillRect(u * 512, 0, ((bands[i + 1] || [1])[0] - u) * 512 + 1, 256);
    });
    for (let i = 0; i < 5000; i++) {
      x.fillStyle = `rgba(0,0,0,${rng() * 0.05})`;
      x.fillRect(rng() * 512, rng() * 256, 1, 1 + rng() * 4);
    }
    x.fillStyle = 'rgba(0,0,0,.3)'; x.fillRect(0, 0, 512, 2); x.fillRect(0, 128, 512, 2); // joints between wall elements
  });
  // rock: white concrete elements up the walls, dark sprayed vault above. Lid: bare concrete all round.
  return [floor, wall([[0, '#c9cbc8'], [0.21, '#3a3d40'], [0.79, '#c9cbc8']]), wall([[0, '#9a9c9b'], [0.3, '#77797a'], [0.7, '#9a9c9b']])];
}

// Paving, each 512 square and repeating: [quay apron, the square's bare setts, the promenade's trodden setts,
// the centre's setts, its pavement] (enum gfx_paving in gfx.h).
export function pavingTextures() {
  const S = 512;
  // Eight metres square of quay apron in winter: ploughed, the asphalt showing dark and wet through what the
  // blade left, packed snow where it missed.
  const apron = rng => {
    const [, x] = canvas(S, S);
    x.fillStyle = '#bcc4cf'; x.fillRect(0, 0, S, S);
    const blot = (n, r0, r1, tone) => {
      for (let i = 0; i < n; i++) {
        const px = rng() * S, py = rng() * S, r = r0 + rng() * (r1 - r0);
        const g = x.createRadialGradient(px, py, 0, px, py, r);
        g.addColorStop(0, tone); g.addColorStop(1, tone.replace(/[\d.]+\)$/, '0)'));
        x.fillStyle = g;
        for (const [ox, oy] of [[0, 0], [S, 0], [-S, 0], [0, S], [0, -S]]) x.fillRect(px - r + ox, py - r + oy, r * 2, r * 2);
      }
    };
    blot(22, 30, 80, 'rgba(84,88,97,.3)');
    blot(22, 20, 60, 'rgba(226,232,240,.4)');
    for (let i = 0; i < 9000; i++) {
      const v = 120 + rng() * 110 | 0;
      x.fillStyle = `rgba(${v},${v + 4},${v + 10},${0.12 + rng() * 0.2})`;
      x.fillRect(rng() * S, rng() * S, 1 + rng() * 4, 1 + rng() * 3);
    }
    return x;
  };
  // Four metres square of grey setts. `snow` 0: bare and wet, as the heated square is kept; 1: trodden snow
  // lying over all but the worn middle of each stone, as on the promenade.
  const setts = (rng, snow) => {
    const [, x] = canvas(S, S), ROWS = 32, W = 24;
    x.fillStyle = '#33373d'; x.fillRect(0, 0, S, S);
    for (let j = 0; j < ROWS; j++) for (let i = -1; i < S / W + 1; i++) {
      const v = 62 + rng() * 26 | 0;
      x.fillStyle = `rgb(${v},${v + 2},${v + 7})`;
      x.fillRect(i * W + (j % 2) * W / 2 + 1, j * 16 + 1, W - 2, 14);
    }
    for (let i = 0; i < 26 + snow * 60; i++) { // standing water; or snow packed down
      const px = rng() * S, py = rng() * S, r = 20 + rng() * (50 + snow * 40), tone = snow ? '205,213,225' : '14,17,22';
      const g = x.createRadialGradient(px, py, 0, px, py, r);
      g.addColorStop(0, `rgba(${tone},${snow ? 0.8 : 0.55})`); g.addColorStop(1, `rgba(${tone},0)`);
      x.fillStyle = g;
      for (const [ox, oy] of [[0, 0], [S, 0], [-S, 0], [0, S], [0, -S]]) x.fillRect(px - r + ox, py - r + oy, r * 2, r * 2);
    }
    for (let i = 0; i < 260 + snow * 900; i++) { // slush in the joints
      x.fillStyle = `rgba(190,200,214,${0.1 + rng() * 0.2 + snow * 0.3})`;
      x.fillRect(rng() * S, (rng() * ROWS | 0) * 16 - 1, 4 + rng() * 22, 2);
    }
    return x;
  };
  // bottom row first, as three uploaded its canvases
  const flipped = x => { const top = pixels(x, S, S), data = new Uint8Array(top.length); for (let y = 0; y < S; y++) data.set(top.subarray((S - 1 - y) * S * 4, (S - y) * S * 4), y * S * 4); return { width: S, height: S, data }; };
  // Six metres square of pavement in winter: shovelled and trodden down to a thin hard layer, gritted, the slabs
  // showing through where it has worn off.
  const pavement = rng => {
    const [, x] = canvas(S, S);
    x.fillStyle = '#c3cbd6'; x.fillRect(0, 0, S, S);
    for (let i = 0; i < 9000; i++) {
      const v = 180 + rng() * 50 | 0;
      x.fillStyle = `rgba(${v},${v + 4},${v + 10},${0.15 + rng() * 0.2})`;
      x.fillRect(rng() * S, rng() * S, 1 + rng() * 4, 1 + rng() * 4);
    }
    for (let i = 0; i < 22; i++) { // worn patches: grey slabs and their joints
      const px = rng() * S, py = rng() * S, r = 24 + rng() * 60;
      const g = x.createRadialGradient(px, py, 0, px, py, r);
      g.addColorStop(0, 'rgba(120,125,134,.28)'); g.addColorStop(1, 'rgba(120,125,134,0)');
      x.fillStyle = g;
      for (const [ox, oy] of [[0, 0], [S, 0], [-S, 0], [0, S], [0, -S]]) x.fillRect(px - r + ox, py - r + oy, r * 2, r * 2);
    }
    x.fillStyle = 'rgba(60,64,72,.16)';
    for (let k = 0; k < S; k += S / 12) { x.fillRect(k, 0, 1.5, S); x.fillRect(0, k, S, 1.5); } // half-metre slabs
    for (let i = 0; i < 1500; i++) { // strewn grit
      x.fillStyle = `rgba(80,70,60,${0.15 + rng() * 0.25})`;
      x.fillRect(rng() * S, rng() * S, 1.5, 1.5);
    }
    return x;
  };
  return [apron(mulberry32(41)), setts(mulberry32(42), 0), setts(mulberry32(43), 1), setts(mulberry32(31), 0), pavement(mulberry32(32))].map(flipped);
}

// Shop signs. The lettering is the browser's, so the glue does both halves: measure each name (the C side lays the
// signs out from the widths), then letter the atlas from the table the C side hands back.
export const SIGN_FONT = `600 ${64 * 0.62}px "Helvetica Neue", Helvetica, Arial, sans-serif`;
const SIGN_FACES = [['#1d150e', '#ffc46b'], ['#0f2c4d', '#f4f6f8'], ['#0d5a34', '#f4f6f8'], ['#14161a', '#fff3d8'], ['#24272b', '#c9cfd6'], // food, bank, pharmacy, hotel, office
  ['#17191c', '#f6efe0'], ['#5b1414', '#f6efe0'], ['#12303f', '#f6efe0'], ['#1f3a2a', '#f6efe0'], ['#e9e5d8', '#22252a']];                  // and five for plain shops
export function measureSigns(names) {
  const [, x] = canvas(8, 8);
  x.font = SIGN_FONT;
  return names.map(n => Math.ceil(x.measureText(n).width));
}
// signs: [{ name, x, y, w, face }] in atlas pixels
export function signAtlas(signs) {
  const [, x] = canvas(2048, 1024);
  x.font = SIGN_FONT;
  x.textBaseline = 'middle';
  for (const s of signs) {
    const [bg, fg] = SIGN_FACES[s.face] || SIGN_FACES[5];
    x.fillStyle = bg; x.fillRect(s.x, s.y, s.w, 64);
    x.fillStyle = fg; x.fillText(s.name, s.x + 64 * 0.35, s.y + 64 * 0.53);
  }
  return { width: 2048, height: 1024, data: pixels(x, 2048, 1024) };
}

/* ---- the centre's shop fronts (src/sentrum.js) ---- */

const TILE = 128;          // px per display tile in the window atlas (one bay of glass, about 2.8 x 2.4 m)
const COLS = 16, ROWS = 4; // tiles across and down the atlas

// What a display can show; the index is its tile in the window atlas. One row of lit displays in each of the
// first three rows (with variants), the fourth row for closed and dark frontages.
const DISPLAY = {
  fashion: 0, shoes: 1, cafe: 2, restaurant: 3, bakery: 4, pharmacy: 5, optician: 6, books: 7, gifts: 8, grocery: 9,
  kiosk: 10, bank: 11, office: 12, pub: 13, hair: 14, sport: 15, phone: 16, flowers: 17, jeweller: 18, toys: 19,
  yarn: 20, interior: 21, lobby: 22, beauty: 23, fastfood: 24, mall: 25, cinema: 26, travel: 27,
  shutter: 48, dark: 49, paper: 50, door: 51, doordark: 52, blinds: 53,
};

/* ---- shop windows: one atlas of small lit dioramas ----
 * Each tile is a bay of glass seen at night from the pavement: the room behind lit, what is for sale in
 * silhouette and colour, the glazing bars, and for Advent a string of lights along the head, a paper star or a
 * candle bridge. Drawn in canvas y-down, tile origin top left. */

export function displayAtlas(rng = mulberry32(1907)) {
  const [c, x] = canvas(TILE * COLS, TILE * ROWS);
  const T = TILE, r = rng;
  const pick = a => a[r() * a.length | 0];
  const room = (wall, floor, glow = 0.35) => { // the back wall, the floor, a pool of light from the ceiling
    const g = x.createLinearGradient(0, 0, 0, T);
    g.addColorStop(0, wall[0]); g.addColorStop(0.72, wall[1]); g.addColorStop(0.73, floor[0]); g.addColorStop(1, floor[1]);
    x.fillStyle = g; x.fillRect(0, 0, T, T);
    x.fillStyle = 'rgba(48,30,14,.22)'; x.fillRect(0, 0, T, T); // seen through glass at night: warmer and duller than the paint
    const s = x.createRadialGradient(T / 2, T * 0.15, 4, T / 2, T * 0.3, T * 0.75);
    s.addColorStop(0, `rgba(255,240,215,${glow})`); s.addColorStop(1, 'rgba(255,240,215,0)');
    x.fillStyle = s; x.fillRect(0, 0, T, T);
  };
  const box = (px, py, w, h, f) => { x.fillStyle = f; x.fillRect(px, py, w, h); };
  const figure = (px, base, h, top, legs) => { // a mannequin: head, shoulders, garment, stand
    box(px - 1, base - 6, 2, 6, '#2a2a2a');
    box(px - 7, base - 7, 14, 2, '#2a2a2a');
    x.fillStyle = top; x.beginPath();
    x.moveTo(px - h * 0.16, base - h * 0.78); x.lineTo(px + h * 0.16, base - h * 0.78);
    x.lineTo(px + h * 0.2, base - h * (legs ? 0.42 : 0.12)); x.lineTo(px - h * 0.2, base - h * (legs ? 0.42 : 0.12)); x.fill();
    if (legs) box(px - h * 0.13, base - h * 0.42, h * 0.26, h * 0.36, legs);
    x.fillStyle = '#e8dccb'; x.beginPath(); x.arc(px, base - h * 0.88, h * 0.07, 0, 6.3); x.fill();
  };
  const shelves = (x0, x1, y0, y1, n, items, back = '#d9d2c6') => {
    box(x0, y0, x1 - x0, y1 - y0, back);
    for (let k = 0; k < n; k++) {
      const y = y0 + (y1 - y0) * (k + 1) / n;
      for (let px = x0 + 2; px < x1 - 4;) {
        const w = 3 + r() * 6, h = 5 + r() * (y1 - y0) / n * 0.55;
        box(px, y - h - 1, w, h, pick(items));
        px += w + 1 + r() * 2;
      }
      box(x0, y - 1, x1 - x0, 2, '#7a6d5e');
    }
  };
  const table = (px, py, w, cloth) => {
    box(px - w / 2, py, w, 3, cloth); box(px - 1, py + 3, 2, 14, '#2b2017');
    x.fillStyle = '#ffd88a'; x.beginPath(); x.arc(px, py - 3, 2.2, 0, 6.3); x.fill(); // a candle
    for (const s of [-1, 1]) { box(px + s * (w / 2 + 3) - 2, py - 8, 4, 22, '#3a2a1e'); }
  };
  const pendant = (px, py) => {
    box(px, 0, 1, py, '#222');
    x.fillStyle = '#ffe2a8'; x.beginPath(); x.arc(px + 0.5, py + 2, 5, Math.PI, 0); x.fill();
    const g = x.createRadialGradient(px, py + 6, 0, px, py + 10, 28);
    g.addColorStop(0, 'rgba(255,214,150,.5)'); g.addColorStop(1, 'rgba(255,214,150,0)');
    x.fillStyle = g; x.fillRect(px - 30, py, 60, 40);
  };
  const people = (n, y) => { for (let k = 0; k < n; k++) {
    const px = 12 + r() * (T - 24), h = 26 + r() * 10;
    x.fillStyle = `rgba(30,22,18,${0.75 + r() * 0.2})`;
    x.beginPath(); x.arc(px, y - h, 5, 0, 6.3); x.fill();
    x.fillRect(px - 7, y - h + 5, 14, h * 0.55);
  } };

  const draw = {
    fashion: () => { room(['#f3e6d2', '#d9c3a6'], ['#bfa88c', '#8f7a63'], 0.5);
      for (const px of [26, 64, 102]) figure(px, T * 0.86, 62 + r() * 8, pick(['#8c1d1d', '#1f2f4a', '#d9d4c7', '#3b3b3b', '#6b4b2a', '#c45a3a']), r() < 0.6 ? pick(['#20242c', '#3b3b3b', '#5a4a3a']) : null); },
    shoes: () => { room(['#efe6da', '#d6c8b4'], ['#a08c74', '#7a6957'], 0.45); shelves(10, 118, 22, 92, 4, ['#3a2a1e', '#7a2d1d', '#1c1c1c', '#c9b79a', '#5d4a3a', '#8a8a8a']); },
    cafe: () => { room(['#c98f55', '#8a5a33'], ['#5b3d26', '#3a271a'], 0.55); pendant(36, 24); pendant(92, 20);
      box(0, 70, T, 3, '#3a2616'); table(40, 86, 22, '#efe3cf'); table(96, 88, 22, '#efe3cf'); people(2, 98); },
    restaurant: () => { room(['#b5643a', '#6e3a20'], ['#4a2a18', '#2c180d'], 0.4); pendant(64, 18);
      table(34, 84, 26, '#f4ece0'); table(94, 84, 26, '#f4ece0'); people(1 + (r() * 2 | 0), 96); },
    bakery: () => { room(['#f6e4c4', '#e2c49a'], ['#b08a62', '#86664a'], 0.55); shelves(14, 114, 14, 54, 2, ['#c98a3e', '#a8662a', '#e0b06a', '#8a4f20']);
      box(8, 70, 112, 26, '#efe9df'); box(12, 66, 104, 6, '#d8e6ec'); // the counter with its glass display
      for (let k = 0; k < 9; k++) { x.fillStyle = pick(['#c98a3e', '#e8c27a', '#7a3f1c', '#f0d9a8']); x.beginPath(); x.ellipse(18 + k * 11, 78, 5, 3.5, 0, 0, 6.3); x.fill(); } },
    pharmacy: () => { room(['#f4f8f6', '#dfe8e4'], ['#c7d0cc', '#a5aeaa'], 0.6); shelves(6, 122, 16, 98, 5, ['#2e8b57', '#ffffff', '#3a7ac0', '#e8e8e8', '#d23c3c', '#f2c94c'], '#f1f5f3');
      box(56, 24, 16, 6, '#1f9c55'); box(61, 19, 6, 16, '#1f9c55'); },
    optician: () => { room(['#f7f4ee', '#e4ded2'], ['#c9c0b2', '#a69d8f'], 0.55);
      for (let k = 0; k < 4; k++) for (let i = 0; i < 5; i++) { x.strokeStyle = pick(['#222', '#5a3a22', '#2a3f6a', '#8a1d1d']); x.lineWidth = 2;
        x.beginPath(); x.arc(20 + i * 22, 26 + k * 18, 4.5, 0, 6.3); x.arc(31 + i * 22, 26 + k * 18, 4.5, 0, 6.3); x.stroke(); } },
    books: () => { room(['#e9dcc4', '#c9b08a'], ['#8f7454', '#6a543c'], 0.45); shelves(8, 120, 10, 70, 3, ['#8a1d1d', '#1f3f6a', '#d9a441', '#2e5e3e', '#f1ece0', '#4a3a6a', '#c45a3a']);
      box(30, 78, 68, 14, '#5a4030'); for (let k = 0; k < 6; k++) box(34 + k * 10, 70, 8, 8, pick(['#d23c3c', '#2a5a8a', '#f2c94c', '#efe3cf'])); },
    gifts: () => { room(['#f0e2cc', '#d3b994'], ['#a2876a', '#7a644c'], 0.5); shelves(10, 118, 18, 66, 2, ['#c0392b', '#f1ece0', '#2e5e3e', '#d9a441', '#b0b0b0']);
      box(26, 80, 76, 12, '#6a4a30'); for (let k = 0; k < 5; k++) { x.fillStyle = pick(['#c0392b', '#e8e0d0', '#2e5e3e']); x.beginPath(); x.arc(36 + k * 14, 76, 5, 0, 6.3); x.fill(); } },
    grocery: () => { room(['#f8faf6', '#e6ebe4'], ['#cdd3cd', '#aab1ab'], 0.75); shelves(4, 124, 14, 100, 5, ['#d23c3c', '#f2c94c', '#2e8b57', '#3a7ac0', '#f08a24', '#ffffff', '#7a3a8a'], '#f4f6f2'); },
    kiosk: () => { room(['#fbfbf6', '#ebebe2'], ['#cfcfc5', '#adada3'], 0.8); shelves(6, 122, 10, 60, 3, ['#d23c3c', '#2a5a8a', '#f2c94c', '#1f1f1f', '#f08a24', '#ffffff']);
      box(10, 70, 108, 24, '#2b2b2b'); box(14, 66, 100, 4, '#d23c3c'); },
    bank: () => { room(['#9ea6ad', '#7f878e'], ['#6a7178', '#565c62'], 0.2); // frosted lower half, a desk beyond, lit low after hours
      box(0, 52, T, 48, 'rgba(150,158,166,.92)'); box(0, 50, T, 3, '#9aa6b0'); box(40, 34, 48, 12, '#6b7680'); },
    office: () => { room(['#aab0b4', '#90969a'], ['#7a8084', '#62686c'], 0.2);
      for (let k = 0; k < 9; k++) box(0, 6 + k * 10, T, 6, 'rgba(176,176,170,.92)'); },
    pub: () => { room(['#7a3a18', '#3e1c0c'], ['#2c150a', '#1a0c05'], 0.35); pendant(30, 22); pendant(98, 22);
      shelves(14, 114, 18, 52, 2, ['#3a7a3a', '#8a5a1a', '#d9a441', '#6a1a1a', '#e0e0d0'], '#4a2410');
      box(0, 72, T, 26, '#3a1f0e'); box(0, 70, T, 3, '#8a5a2a'); people(2, 74); },
    hair: () => { room(['#f2efe9', '#dedad2'], ['#9a948a', '#7a756c'], 0.55);
      for (const px of [24, 64, 104]) { box(px - 11, 18, 22, 34, '#bcc9d2'); box(px - 12, 17, 24, 2, '#222'); box(px - 8, 66, 16, 14, '#1f1f1f'); box(px - 2, 80, 4, 14, '#555'); } },
    sport: () => { room(['#eef0f2', '#d6dbe0'], ['#a9b0b6', '#868d93'], 0.6);
      for (let k = 0; k < 6; k++) box(10 + k * 7, 14, 4, 76, pick(['#d23c3c', '#1f3f6a', '#f2c94c', '#2e8b57'])); // skis on the wall
      figure(84, T * 0.86, 66, pick(['#d23c3c', '#1f3f6a', '#f08a24']), '#20242c'); figure(112, T * 0.86, 60, '#2e8b57', '#3b3b3b'); },
    phone: () => { room(['#1b2430', '#121821'], ['#0e131a', '#080b10'], 0.15);
      for (let k = 0; k < 5; k++) { box(10 + k * 23, 40, 15, 28, '#0a0d12'); box(12 + k * 23, 42, 11, 24, pick(['#5aa0ff', '#7ad0ff', '#c0e4ff', '#ff8ad0'])); }
      box(8, 76, 112, 8, '#e8eef4'); },
    flowers: () => { room(['#f2efe2', '#d8d2b8'], ['#8a7a5a', '#6a5c42'], 0.5);
      for (let k = 0; k < 14; k++) { const px = 8 + r() * 112, py = 40 + r() * 46; box(px - 4, py, 8, 14, '#6a4a2a');
        x.fillStyle = pick(['#d23c3c', '#c0185a', '#f2f2ea', '#2e7a3e', '#e04a2a']); x.beginPath(); x.arc(px, py - 4, 6 + r() * 4, 0, 6.3); x.fill(); } },
    jeweller: () => { room(['#2a2420', '#1a1512'], ['#120e0b', '#0a0806'], 0.2);
      for (let k = 0; k < 3; k++) { box(14 + k * 36, 52, 28, 22, '#e8e0d0'); for (let i = 0; i < 4; i++) box(18 + k * 36 + i * 6, 58, 3, 3, '#ffe9a0'); } },
    toys: () => { room(['#fff2d8', '#f2d9a8'], ['#c9a46a', '#a0804a'], 0.6); shelves(8, 120, 14, 90, 3, ['#d23c3c', '#2a7ad0', '#f2c94c', '#2e8b57', '#f08a24', '#a03ab0']); },
    yarn: () => { room(['#f0e6d6', '#d9c8ae'], ['#a08a6a', '#7a684e'], 0.5);
      for (let k = 0; k < 4; k++) for (let i = 0; i < 9; i++) { x.fillStyle = pick(['#c0392b', '#e8d8b8', '#2e5e7e', '#7a9a5a', '#d9a441', '#8a4a7a', '#f2f2ea']);
        x.beginPath(); x.arc(12 + i * 13, 22 + k * 18, 6, 0, 6.3); x.fill(); } },
    interior: () => { room(['#efe6d8', '#d4c4ac'], ['#9a8266', '#76624a'], 0.5); pendant(64, 16);
      box(14, 64, 56, 20, '#7a6a5a'); box(14, 56, 10, 28, '#7a6a5a'); box(82, 50, 30, 40, '#3a2a1e'); box(86, 54, 22, 14, '#d9cbb4'); },
    lobby: () => { room(['#e9d7b8', '#c6a67a'], ['#7a5a3a', '#5a4028'], 0.6); pendant(64, 14);
      box(24, 62, 80, 30, '#5a3a22'); box(20, 58, 88, 5, '#d9c3a0'); people(1, 62); },
    beauty: () => { room(['#f6ece8', '#e2d0c8'], ['#b8a49a', '#968278'], 0.6); shelves(10, 118, 16, 70, 3, ['#ffffff', '#e0a0b0', '#c9a0d0', '#f2c94c', '#d0d0d0']); },
    fastfood: () => { room(['#f6f0e0', '#e0d6bc'], ['#b0a68c', '#8c826a'], 0.8); box(0, 10, T, 18, '#1f1f1f');
      for (let k = 0; k < 4; k++) box(6 + k * 30, 13, 26, 12, pick(['#f2c94c', '#d23c3c', '#f08a24'])); box(6, 66, 116, 28, '#c03020'); people(2, 66); },
    mall: () => { room(['#f2f0ea', '#dcd8cc'], ['#c9c4b6', '#a8a396'], 0.7); // an arcade: shop fronts down a lit passage
      x.fillStyle = '#b8b2a4'; x.beginPath(); x.moveTo(0, 0); x.lineTo(44, 34); x.lineTo(44, 80); x.lineTo(0, T); x.fill();
      x.beginPath(); x.moveTo(T, 0); x.lineTo(84, 34); x.lineTo(84, 80); x.lineTo(T, T); x.fill();
      box(44, 34, 40, 46, '#fff6e0'); for (let k = 0; k < 3; k++) box(4 + k * 12, 30 + k * 6, 6, 50 - k * 10, pick(['#d23c3c', '#2a5a8a', '#f2c94c'])); people(2, 96); },
    cinema: () => { room(['#2a1418', '#160a0c'], ['#0e0607', '#080304'], 0.2);
      for (let k = 0; k < 3; k++) { box(12 + k * 38, 20, 28, 40, '#0a0a0a'); box(14 + k * 38, 22, 24, 36, pick(['#c03a2a', '#2a4a8a', '#d9a441', '#3a7a5a'])); }
      for (let k = 0; k < 12; k++) { x.fillStyle = '#ffe9a0'; x.beginPath(); x.arc(8 + k * 10.5, 8, 1.6, 0, 6.3); x.fill(); } },
    travel: () => { room(['#e6f0f6', '#c8dae6'], ['#a0b0bc', '#80909c'], 0.6); box(14, 18, 100, 44, '#3a8ac0'); box(14, 46, 100, 16, '#e8d8a8'); box(40, 72, 48, 16, '#5a6a76'); },
    // closed and dark frontages
    shutter: () => { box(0, 0, T, T, '#7d8288'); for (let k = 0; k < T; k += 6) { box(0, k, T, 1, '#5d6268'); box(0, k + 1, T, 1, '#a2a7ad'); } },
    dark: () => { const g = x.createLinearGradient(0, 0, T, T); g.addColorStop(0, '#0d131b'); g.addColorStop(1, '#1a2430'); x.fillStyle = g; x.fillRect(0, 0, T, T);
      x.fillStyle = 'rgba(120,150,190,.09)'; x.beginPath(); x.moveTo(10, T); x.lineTo(70, 0); x.lineTo(96, 0); x.lineTo(36, T); x.fill(); },
    paper: () => { box(0, 0, T, T, '#d8d0bc'); for (let k = 0; k < 30; k++) box(r() * T, r() * T, 10 + r() * 30, 8 + r() * 20, `rgba(150,140,120,${r() * 0.3})`); box(30, 40, 68, 30, '#f4f0e6'); box(36, 48, 56, 4, '#c03020'); },
    door: () => { room(['#f2e4c8', '#d6bf98'], ['#a0866a', '#7a654e'], 0.5); box(0, 0, T, T, 'rgba(0,0,0,0)'); people(1, 100); },
    doordark: () => { box(0, 0, T, T, '#10161e'); },
    blinds: () => { box(0, 0, T, T, '#3a3a36'); for (let k = 0; k < T; k += 5) box(0, k, T, 3, '#cfc9ba'); },
  };
  const advent = (k) => { // string lights along the head, and now and then a star or a candle bridge
    const lit = r();
    for (let i = 0; i < 12; i++) {
      const px = 6 + i * 10.5, py = 6 + Math.sin(i * 0.9) * 2;
      x.fillStyle = 'rgba(255,214,140,.35)'; x.beginPath(); x.arc(px, py, 3.6, 0, 6.3); x.fill();
      x.fillStyle = lit < 0.7 ? '#fff4d6' : pick(['#ff4d3a', '#ffd166', '#7fd36b', '#6ab0ff']); x.beginPath(); x.arc(px, py, 1.6, 0, 6.3); x.fill();
    }
    if (k % 3 === 0) { // a paper star hung in the window
      const sx = 30 + r() * 68, sy = 26 + r() * 10, rr = 9;
      x.fillStyle = pick(['#ff5236', '#ffb347', '#fff1cf']); x.beginPath();
      for (let i = 0; i < 14; i++) { const a = i / 14 * Math.PI * 2 - Math.PI / 2, q = i % 2 ? rr * 0.45 : rr; x.lineTo(sx + Math.cos(a) * q, sy + Math.sin(a) * q); }
      x.fill();
    } else if (k % 3 === 1) { // a seven-armed candle bridge on the sill
      box(40, T - 10, 48, 3, '#3a2a1a');
      for (let i = -3; i <= 3; i++) { x.fillStyle = '#ffe7a6'; x.beginPath(); x.arc(64 + i * 7, T - 14 - (3 - Math.abs(i)) * 3, 2, 0, 6.3); x.fill(); }
    }
  };
  const glazing = (light) => { // the glass itself: the frame round the tile, a transom, a faint sky reflection
    x.fillStyle = 'rgba(150,175,205,.08)'; x.beginPath(); x.moveTo(0, T * 0.8); x.lineTo(T * 0.5, 0); x.lineTo(T * 0.62, 0); x.lineTo(0, T); x.fill();
    box(0, 0, T, 3, light); box(0, T - 3, T, 3, light); box(0, 0, 3, T, light); box(T - 3, 0, 3, T, light);
    box(0, T * 0.16, T, 2, light);
  };
  for (const [name, k] of Object.entries(DISPLAY)) {
    x.save();
    x.translate((k % COLS) * T, (k / COLS | 0) * T);
    x.beginPath(); x.rect(0, 0, T, T); x.clip();
    draw[name]();
    if (k < 48 && name !== 'bank' && name !== 'office') advent(k);
    if (name !== 'shutter' && name !== 'paper' && name !== 'blinds') glazing(k < 48 ? '#1c1d20' : '#26282c');
    x.restore();
  }
  // the top of the picture is the last row: v runs up, as the fronts' tiles count it
  const W = TILE * COLS, H = TILE * ROWS, top = pixels(x, W, H), data = new Uint8Array(top.length);
  for (let y = 0; y < H; y++) data.set(top.subarray((H - 1 - y) * W * 4, (H - y) * W * 4), y * W * 4);
  return { width: W, height: H, data };
}

// The centre's surveyed signs, lettered as src/sentrum.js letters them: semibold on a band of its own colour, or bold
// letters glowing on nothing. As with the shop signs the C side lays them out from widths measured here.
const labelFont = bold => `${bold ? 700 : 600} ${64 * 0.62}px "Helvetica Neue", Helvetica, Arial, sans-serif`;
// texts: [{ text, bold }]
export function measureLabels(texts) {
  const [, x] = canvas(8, 8);
  return texts.map(t => { x.font = labelFont(t.bold); return Math.ceil(x.measureText(t.text).width); });
}
// labels: [{ name, x, y, w, fg, bg, letters }] in atlas pixels, colours 0xRRGGBB
export function labelAtlas(labels) {
  const [, x] = canvas(2048, 512), css = v => '#' + v.toString(16).padStart(6, '0');
  for (const s of labels) {
    x.font = labelFont(s.letters);
    if (!s.letters) { x.fillStyle = css(s.bg); x.fillRect(s.x, s.y, s.w, 64); }
    else { x.shadowColor = css(s.fg); x.shadowBlur = 8; }
    x.fillStyle = css(s.fg); x.textBaseline = 'middle';
    x.fillText(s.name, s.x + 64 * 0.35, s.y + 64 * 0.53);
    x.shadowBlur = 0;
  }
  return { width: 2048, height: 512, data: pixels(x, 2048, 512) };
}

// The number plate.
export function plateTexture(text) {
  const [, x] = canvas(256, 64);
  x.fillStyle = '#f4f4f0'; x.fillRect(0, 0, 256, 64);
  x.fillStyle = '#1d3f8f'; x.fillRect(0, 0, 26, 64);
  x.strokeStyle = '#111'; x.lineWidth = 3; x.strokeRect(1.5, 1.5, 253, 61);
  x.fillStyle = '#111'; x.font = '600 44px "Helvetica Neue", Arial, sans-serif';
  x.textAlign = 'center'; x.textBaseline = 'middle';
  x.fillText(text, 141, 35);
  return { width: 256, height: 64, data: pixels(x, 256, 64) };
}

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

/* ---- the car's instruments (tools/car/car.js) ----
 * The instrument cluster: drawn as the user's own photo of this car's shows it at dusk ---- */

const GAUGE_W = 1024, GAUGE_H = 320;          // canvas; the dials sit on a 0.5 × 0.156 m face
const DIAL_FROM = Math.PI * 0.77, DIAL_SWEEP = Math.PI * 1.44; // the big dials run from low left round over the top to low right
const SPEEDO = [330, 150, 140], REVS = [668, 150, 140], POD_Y = 172, ARROW_Y = 80; // big dials; the small pods and the turn arrows sit low enough to clear the hood
const RED_LINE = 6.5, REV_SCALE = 6600; // r/min where the red band starts; what the engine's revs 0..1 span
const ODO_START = 246879, OUTSIDE = '-2°C', FUEL = 0.85; // the photographed car's odometer, and a Harstad winter night
const BACKLIT = '#f5ead6', NEEDLE = '#ff4d1a', NEEDLE_GLOW = 'rgba(255,77,26,.28)', ARROW = '#56ff3c', LCD = '#c4c75a', LCD_INK = '#2c2e0e';

function clusterFace() {
  const c = new OffscreenCanvas(1, 1);
  c.width = GAUGE_W; c.height = GAUGE_H;
  const x = c.getContext('2d');
  x.fillStyle = '#040506'; x.fillRect(0, 0, GAUGE_W, GAUGE_H);
  x.textAlign = 'center'; x.textBaseline = 'middle';
  const polar = (cx, cy, r, a) => [cx + Math.cos(a) * r, cy + Math.sin(a) * r];
  const tick = (cx, cy, r0, r1, a, w) => { x.lineWidth = w; x.beginPath(); x.moveTo(...polar(cx, cy, r0, a)); x.lineTo(...polar(cx, cy, r1, a)); x.stroke(); };
  for (const [cx, cy, r] of [SPEEDO, REVS]) { x.strokeStyle = '#101113'; x.lineWidth = 3; x.beginPath(); x.arc(cx, cy, r + 4, 0, 7); x.stroke(); }
  x.strokeStyle = x.fillStyle = BACKLIT;
  // speedometer: 0–240 km/h, a tick every 10, figures every 20
  {
    const [cx, cy, r] = SPEEDO;
    for (let v = 0; v <= 240; v += 10) {
      const a = DIAL_FROM + DIAL_SWEEP * v / 240, major = v % 20 === 0;
      tick(cx, cy, r - (major ? 22 : 14), r - 4, a, major ? 6 : 4);
      if (major) { x.font = '500 25px "Helvetica Neue", Arial, sans-serif'; x.fillText(v, ...polar(cx, cy, r - 42, a)); }
    }
    x.font = '15px "Helvetica Neue", Arial, sans-serif'; x.fillText('km/h', cx + 18, cy + 50);
  }
  // rev counter: 0–8 ×1000 r/min, ticks every 250, the red band from 6,500
  {
    const [cx, cy, r] = REVS;
    x.strokeStyle = '#ff2a12'; x.lineWidth = 10; x.beginPath();
    x.arc(cx, cy, r - 12, DIAL_FROM + DIAL_SWEEP * RED_LINE / 8, DIAL_FROM + DIAL_SWEEP, false); x.stroke();
    x.strokeStyle = BACKLIT;
    for (let v = 0; v <= 32; v++) {
      const a = DIAL_FROM + DIAL_SWEEP * v / 32, major = v % 4 === 0;
      if (v / 4 < RED_LINE || major) tick(cx, cy, r - (major ? 24 : 14), r - 4, a, major ? 6 : 3.5);
      if (major) { x.font = '500 34px "Helvetica Neue", Arial, sans-serif'; x.fillText(v / 4, ...polar(cx, cy, r - 48, a)); }
    }
    x.font = '15px "Helvetica Neue", Arial, sans-serif'; x.fillText('x1000r/min', cx + 10, cy + 54);
  }
  // fuel at the left, its needle pivoting from outside the dial; temperature mirrored at the right
  for (const [px, s, top, bottom] of [[40, 1, 'F', 'E'], [GAUGE_W - 40, -1, 'H', 'C']]) {
    for (let k = 0; k <= 4; k++) {
      const a = (s > 0 ? 0 : Math.PI) + s * (k / 4 - 0.5) * 1.15;
      tick(px, POD_Y, 70, k % 2 ? 80 : 86, a, k % 2 ? 3 : 4.5);
    }
    x.font = '500 22px "Helvetica Neue", Arial, sans-serif';
    x.fillText(top, px + s * 66, POD_Y - 68); x.fillText(bottom, px + s * 66, POD_Y + 70);
  }
  x.fillRect(90, POD_Y - 10, 11, 17); x.fillRect(88, POD_Y + 7, 15, 3); x.fillRect(103, POD_Y - 6, 3, 9); // fuel pump
  x.fillStyle = '#7a7466'; x.font = '11px "Helvetica Neue", Arial, sans-serif';
  x.fillText('TOTAL', 232, 278); x.fillText('TRIP', 232, 290); x.fillText('VDO', REVS[0], 304);
  // warning lamps, unlit: they show as dark symbols until one has reason to light
  x.fillStyle = '#17181a';
  for (const [wx, wy] of [[499, 210], [499, 245], [499, 280], [150, 285], [870, 285]]) { x.beginPath(); x.arc(wx, wy, 11, 0, 7); x.fill(); }
  return c;
}

// The cluster's live parts over its face: needles, turn arrows, warning lamps, odometer, clock and outside temperature.
// the soft green bloom a lit turn arrow throws up the face above it
function arrowBloom() {
  const c = new OffscreenCanvas(1, 1), x = c.getContext('2d');
  c.width = c.height = 68;
  const g = x.createRadialGradient(34, 34, 2, 34, 34, 34);
  g.addColorStop(0, 'rgba(86,255,60,.55)'); g.addColorStop(1, 'rgba(86,255,60,0)');
  x.fillStyle = g; x.fillRect(0, 0, 68, 68);
  return c;
}

// needles: glowing orange-red, a broad faint stroke under a bright one, from under a black hub
function needle(x, cx, cy, from, len, tail, a, hub) {
  const c = Math.cos(a), s = Math.sin(a);
  for (let k = 0; k < 2; k++) {
    x.strokeStyle = k ? NEEDLE : NEEDLE_GLOW; x.lineWidth = k ? 5 : 11;
    x.beginPath(); x.moveTo(cx + c * (from - tail), cy + s * (from - tail)); x.lineTo(cx + c * len, cy + s * len); x.stroke();
  }
  if (hub) { x.fillStyle = '#070708'; x.beginPath(); x.arc(cx, cy, hub, 0, 7); x.fill(); }
}
function arrow(x, bloom, on, ax, dir) {
  if (on) { x.drawImage(bloom, ax - 34, 14, 68, 52); x.fillStyle = ARROW; } else x.fillStyle = '#0c120b';
  const y = ARROW_Y;
  x.beginPath(); x.moveTo(ax + dir * 16, y); x.lineTo(ax, y - 14); x.lineTo(ax, y - 6); x.lineTo(ax - dir * 14, y - 6);
  x.lineTo(ax - dir * 14, y + 6); x.lineTo(ax, y + 6); x.lineTo(ax, y + 14); x.closePath(); x.fill();
}

// The cluster as a picture: its face with the arrows, the two LCDs and the small pods' needles, which never move. With `lit` it is
// the same picture with both turn arrows and the brake warning lamp alight. The car's shader (shaders.js, CAR) picks between the
// two where a lamp is, and draws the two big needles itself, so nothing here is painted again but an LCD when its text changes.
const ODO_LCD = [SPEEDO[0] - 52, 246, 104, 46], CLOCK_LCD = [REVS[0] - 54, 246, 108, 50]; // x, y, width, height
function odoLcd(x, odo) {
  x.fillStyle = LCD; x.fillRect(...ODO_LCD);
  x.fillStyle = LCD_INK; x.textAlign = 'left';
  x.font = '13px "Helvetica Neue", Arial, sans-serif'; x.fillText('total', SPEEDO[0] - 46, 256);
  x.font = '500 27px Menlo, "Courier New", monospace'; x.textAlign = 'center';
  x.fillText(odo, SPEEDO[0], 277);
}
function clockLcd(x, time) {
  x.fillStyle = LCD; x.fillRect(...CLOCK_LCD);
  x.fillStyle = LCD_INK; x.textAlign = 'center';
  x.font = '500 21px Menlo, "Courier New", monospace';
  x.fillText(time, REVS[0], 260);
  x.fillText(OUTSIDE, REVS[0], 283);
}
function drawCluster(x, face, bloom, lit, odo, time) {
  x.drawImage(face, 0, 0);
  x.lineCap = 'round';
  arrow(x, bloom, lit, 470, -1);
  arrow(x, bloom, lit, 550, 1);
  if (lit) {
    x.strokeStyle = x.fillStyle = '#ff3220'; x.lineWidth = 3; x.textAlign = 'center';
    x.beginPath(); x.arc(499, 280, 9, 0, 7); x.stroke();
    x.font = 'bold 13px Arial, sans-serif'; x.fillText('!', 499, 281);
  }
  odoLcd(x, odo); clockLcd(x, time); // yellow-green backlight, dark segments
  needle(x, 40, POD_Y, 30, 82, 0, (0.5 - FUEL) * 1.15, 0);      // fuel: most of a tank
  needle(x, GAUGE_W - 40, POD_Y, 30, 82, 0, Math.PI, 0);        // temperature: warmed up, on the middle mark
  x.lineCap = 'butt';
}
const clockText = minute => `${Math.floor(minute / 60) % 24}:${String(minute % 60).padStart(2, '0')}`;

/* ---- the lit panels round the cabin: one small atlas ---- */

const PANEL_W = 512, PANEL_H = 256;
// cells [x, y, w, h]: heater controls, radio, clock, hazard switch, selector gate, switches left of the wheel,
// window switches, stalk pictograms
const CELL = { heater: [0, 0, 256, 80], radio: [0, 80, 256, 56], clock: [256, 0, 96, 32], hazard: [352, 0, 32, 32],
  gate: [384, 0, 48, 144], left: [256, 32, 128, 48], windows: [256, 80, 112, 48], stalk: [432, 0, 64, 32] };
const SYMBOL = '#e9e0cc', SELECTOR = ['P', 'R', 'N', 'D', '2', 'L'];

// The two cells that change: each wipes its own cell and draws it again, so only that much of the atlas need be sent.
// clock over the vents: green digits
function panelClock(x, minute) {
  const [cx, cy, w, h] = CELL.clock;
  x.fillStyle = '#0b0c0e'; x.fillRect(cx, cy, w, h);
  x.textAlign = 'center'; x.textBaseline = 'middle';
  x.fillStyle = '#58ffb0'; x.font = '500 22px Menlo, "Courier New", monospace';
  x.fillText(clockText(minute), cx + w / 2, cy + h / 2 + 1);
}
// selector gate: P R N D 2 L, the engaged one bright
function panelGate(x, gear) {
  const [gx, gy, w, h] = CELL.gate;
  x.fillStyle = '#0b0c0e'; x.fillRect(gx, gy, w, h);
  x.textAlign = 'center'; x.textBaseline = 'middle';
  x.font = 'bold 18px "Helvetica Neue", Arial, sans-serif';
  SELECTOR.forEach((l, k) => {
    const on = l === (gear < 0 ? 'R' : 'D');
    if (on) { x.fillStyle = '#ffb347'; x.fillRect(gx + 8, gy + 4 + k * 23, 32, 21); }
    x.fillStyle = on ? '#1b0f00' : 'rgba(233,224,204,.55)'; x.fillText(l, gx + 24, gy + 15 + k * 23);
  });
}

function drawPanels(x, minute, gear) {
  x.fillStyle = '#0b0c0e'; x.fillRect(0, 0, PANEL_W, PANEL_H);
  x.textAlign = 'center'; x.textBaseline = 'middle';
  const dim = 'rgba(233,224,204,.55)';
  // heater: fan speed, temperature (blue to red), air direction; a green lamp in the A/C button
  {
    const [cx0, cy0] = CELL.heater;
    x.strokeStyle = x.fillStyle = SYMBOL; x.lineWidth = 2;
    for (let k = 0; k < 3; k++) {
      const cx = cx0 + 44 + k * 84, cy = cy0 + 38;
      x.fillStyle = '#151618'; x.beginPath(); x.arc(cx, cy, 24, 0, 7); x.fill();
      for (let m = 0; m <= 4; m++) {
        const a = Math.PI * (0.75 + m * 0.375), r0 = 28, r1 = 33;
        x.strokeStyle = k === 1 ? `rgb(${80 + m * 40},${110 - m * 10},${240 - m * 50})` : SYMBOL;
        x.beginPath(); x.moveTo(cx + Math.cos(a) * r0, cy + Math.sin(a) * r0); x.lineTo(cx + Math.cos(a) * r1, cy + Math.sin(a) * r1); x.stroke();
      }
    }
    x.fillStyle = SYMBOL; x.font = '10px Arial, sans-serif';
    x.fillText('0 1 2 3', cx0 + 44, cy0 + 72); x.fillText('A/C', cx0 + 128, cy0 + 72); x.fillText('▲ ◢ ▼', cx0 + 212, cy0 + 72);
    x.fillStyle = '#38ff6a'; x.fillRect(cx0 + 146, cy0 + 68, 7, 4);
  }
  // radio-cassette: orange display, faint button legends
  {
    const [rx, ry] = CELL.radio;
    x.fillStyle = '#ff7b22'; x.fillRect(rx + 70, ry + 8, 116, 26);
    x.fillStyle = '#3a1404'; x.font = '500 17px Menlo, "Courier New", monospace'; x.fillText('FM1  98.6', rx + 128, ry + 22);
    x.fillStyle = dim; x.font = '9px Arial, sans-serif';
    for (let k = 0; k < 6; k++) x.fillText(k + 1, rx + 70 + k * 23 + 6, ry + 46);
    x.fillText('VOL', rx + 30, ry + 22); x.fillText('TUNE', rx + 226, ry + 22);
  }
  panelClock(x, minute);
  // hazard switch: a red triangle
  {
    const [hx, hy] = CELL.hazard;
    x.strokeStyle = '#ff3b2a'; x.lineWidth = 3;
    x.beginPath(); x.moveTo(hx + 16, hy + 6); x.lineTo(hx + 27, hy + 25); x.lineTo(hx + 5, hy + 25); x.closePath(); x.stroke();
  }
  panelGate(x, gear);
  // left of the wheel: headlamp levelling 0–3, rear fog, mirror adjuster
  {
    const [lx, ly] = CELL.left;
    x.fillStyle = SYMBOL; x.font = '10px Arial, sans-serif';
    x.fillText('0 1 2 3', lx + 24, ly + 40);
    x.strokeStyle = SYMBOL; x.lineWidth = 1.5;
    x.beginPath(); x.arc(lx + 20, ly + 20, 7, Math.PI * 0.5, Math.PI * 1.5); x.closePath(); x.stroke();
    x.beginPath(); x.arc(lx + 64, ly + 20, 6, Math.PI * 0.5, Math.PI * 1.5); x.closePath(); x.stroke();
    for (let k = 0; k < 3; k++) { x.beginPath(); x.moveTo(lx + 71, ly + 15 + k * 5); x.lineTo(lx + 80, ly + 15 + k * 5); x.stroke(); }
    x.fillText('L  R', lx + 106, ly + 40); x.fillText('◀ ▲ ▼ ▶', lx + 106, ly + 20);
  }
  // window switches: four, and the lock
  {
    const [wx, wy] = CELL.windows;
    x.fillStyle = SYMBOL; x.font = '12px Arial, sans-serif';
    for (let k = 0; k < 4; k++) x.fillText('▲', wx + 14 + k * 24, wy + 16);
    x.fillRect(wx + 44, wy + 32, 12, 9); x.strokeStyle = SYMBOL; x.lineWidth = 2; x.beginPath(); x.arc(wx + 50, wy + 32, 4, Math.PI, 0); x.stroke(); // lock
  }
  // stalk: headlamp and fog pictograms
  {
    const [sx, sy] = CELL.stalk;
    x.strokeStyle = SYMBOL; x.lineWidth = 1.6;
    x.beginPath(); x.arc(sx + 18, sy + 16, 6, Math.PI * 0.5, Math.PI * 1.5); x.closePath(); x.stroke();
    for (let k = 0; k < 3; k++) { x.beginPath(); x.moveTo(sx + 26, sy + 11 + k * 5); x.lineTo(sx + 32, sy + 11 + k * 5); x.stroke(); }
    x.beginPath(); x.arc(sx + 48, sy + 16, 5, Math.PI * 0.5, Math.PI * 1.5); x.closePath(); x.stroke();
    x.beginPath(); x.moveTo(sx + 53, sy + 9); x.bezierCurveTo(sx + 56, sy + 14, sx + 56, sy + 18, sx + 53, sy + 23); x.stroke();
  }
}

// The cluster and the panels, painted once on canvases left to the graphics card and handed over as they are (reading them back
// stalled a frame). `layers` are the cluster's two pictures, unlit and lit. After that only what changes is painted again, a small
// rectangle at a time, and each of these returns the [x, y, width, height] to copy from its canvas: odo(text) and clock(minute) on
// layers[0], panelClock(minute) and gate(1 or -1 for the gear) on `panels`.
export function dashPainter() {
  const [gc, gctx] = canvas(GAUGE_W, GAUGE_H, true), [lc, lctx] = canvas(GAUGE_W, GAUGE_H, true), [pc, pctx] = canvas(PANEL_W, PANEL_H, true);
  const face = clusterFace(), bloom = arrowBloom();
  drawCluster(gctx, face, bloom, false, String(ODO_START), clockText(0));
  drawCluster(lctx, face, bloom, true, String(ODO_START), clockText(0));
  drawPanels(pctx, 0, 1);
  return {
    odoStart: ODO_START, layers: [gc, lc], panels: pc,
    odo(text) { odoLcd(gctx, text); return ODO_LCD; },
    clock(minute) { clockLcd(gctx, clockText(minute)); return CLOCK_LCD; },
    panelClock(minute) { panelClock(pctx, minute); return CELL.clock; },
    gate(gear) { panelGate(pctx, gear); return CELL.gate; },
  };
}

// The flakes of every layer: a position in the unit box and a seed each, then which layer. The vertex shader
// does the falling. One random stream runs through the layers in order, as in the JavaScript game.
export function snowFlakes(layers) {
  const n = layers.reduce((s, l) => s + l.count, 0), d = new Float32Array(n * 5), rng = mulberry32(99);
  let i = 0;
  layers.forEach((l, k) => { for (let c = 0; c < l.count; c++, i++) { for (let q = 0; q < 4; q++) d[i * 5 + q] = rng(); d[i * 5 + 4] = k; } });
  return d;
}
