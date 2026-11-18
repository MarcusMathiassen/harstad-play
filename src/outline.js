// The map in outline (data/outline.bin, made by tools/outline.py): drawn on the page while the town loads, under the
// very eye the opening zoom starts from (wasm/src/cam.c, cam_intro_eye), so that when the town is ready the one fades
// into the other. The coast and the lakes are there from the first; the roads and the houses are there faintly, and come
// up to their full strength from the centre outwards as the load goes on: as far out as it has come.

export const INTRO_FOV = 58, INTRO_FIT = 1.06; // CAM_INTRO_FOV, CAM_INTRO_FIT (wasm/include/harstad/cam.h)
const CLASSES = ['coast', 'lake', 'major', 'street', 'path'];
// each class's line: colour, width in CSS pixels, and its strength before the load has reached it (0: as it is)
const LOOK = {
  coast: ['200, 218, 240', 1.0, 0.75, 0], lake: ['200, 218, 240', 0.7, 0.5, 0],
  major: ['255, 200, 140', 1.3, 0.9, 0.14], street: ['255, 222, 186', 0.75, 0.55, 0.1], path: ['205, 214, 230', 0.5, 0.3, 0.07],
};
const HOUSE = ['255, 214, 160', 0.55, 0.08]; // colour, strength reached and not
const SIDES = [3, 5, 7, 10, 14, 20, 28, 40];   // tools/outline.py

/** Read data/outline.bin: the bounds, the centre it fills in from, and for each class its segments (x0 y0 z0 x1 y1 z1),
 *  and the houses (x y z side), each ordered by how far it lies from the centre, with that distance. */
export function readOutline(buf) {
  const v = new DataView(buf), b = new Uint8Array(buf);
  if (String.fromCharCode(...b.subarray(0, 4)) !== 'HSO1') throw new Error('not an outline');
  const f = o => v.getFloat32(o, true);
  const [bx, bz, w, d, unit, rise, cx, cz] = [4, 8, 12, 16, 20, 24, 28, 32].map(f);
  const nl = v.getUint32(36, true), nh = v.getUint32(40, true), gw = v.getUint16(44, true), gh = v.getUint16(46, true), cell = f(48);
  const grid = b.subarray(52, 52 + gw * gh);
  let o = 52 + gw * gh;
  const uint = () => { let r = 0, s = 1, c; do { c = b[o++]; r += (c & 127) * s; s *= 128; } while (c & 128); return r; };
  const zig = n => n % 2 ? -(n + 1) / 2 : n / 2;
  const ground = (x, z) => { // the height grid, bilinear
    const i = Math.min(gw - 1.001, Math.max(0, (x - bx) / cell)), j = Math.min(gh - 1.001, Math.max(0, (z - bz) / cell));
    const a = i | 0, c = j | 0, fi = i - a, fj = j - c, at = (p, q) => grid[q * gw + p] * rise;
    return (at(a, c) * (1 - fi) + at(a + 1, c) * fi) * (1 - fj) + (at(a, c + 1) * (1 - fi) + at(a + 1, c + 1) * fi) * fj;
  };
  const segs = Object.fromEntries(CLASSES.map(k => [k, []]));
  let px = 0, pz = 0;
  for (let i = 0; i < nl; i++) {
    const cls = CLASSES[b[o++]], n = uint(), flat = cls === 'coast';
    let x0, y0, z0;
    for (let k = 0; k < n; k++) {
      px += zig(uint()); pz += zig(uint());
      const x = bx + px * unit, z = bz + pz * unit, y = flat ? 0 : ground(x, z);
      if (k) segs[cls].push([Math.hypot((x0 + x) / 2 - cx, (z0 + z) / 2 - cz), x0, y0, z0, x, y, z]);
      x0 = x; y0 = y; z0 = z;
    }
  }
  const houses = [];
  px = 0; pz = 0;
  for (let i = 0; i < nh; i++) {
    px += zig(uint());
    const m = uint(), side = SIDES[m % 8];
    pz += zig(Math.floor(m / 8));
    const x = bx + px * unit, z = bz + pz * unit;
    houses.push([Math.hypot(x - cx, z - cz), x, ground(x, z), z, side]);
  }
  const order = (a, c) => a[0] - c[0];
  for (const k of CLASSES) segs[k].sort(order);
  houses.sort(order);
  const reach = Math.max(...CLASSES.map(k => segs[k].at(-1)?.[0] || 0), houses.at(-1)?.[0] || 0);
  return { bx, bz, w, d, segs, houses, reach };
}

/** Where a point of the map comes on a canvas `width` x `height`, seen as the zoom's first frame sees it. */
export function introEye(map, width, height) {
  const ty = Math.tan(INTRO_FOV * Math.PI / 360), tx = ty * width / height;
  const h = INTRO_FIT * Math.max(map.w / (2 * tx), map.d / (2 * ty)), ex = map.bx + map.w / 2, ez = map.bz + map.d / 2;
  return (x, y, z) => { const k = 1 / (h - y); return [width / 2 * (1 + (x - ex) * k / tx), height / 2 * (1 + (z - ez) * k / ty)]; };
}

/** Draws `map` on a 2D canvas, filled in to `share` (0..1) of the way out from the centre; call again as it grows. */
export function outlineDrawer(canvas, map, ratio) {
  const g = canvas.getContext('2d');
  let at = null, eye = null, reached = -1, mpp = 1;
  const done = Object.fromEntries(CLASSES.map(k => [k, 0]));
  let doneHouses = 0;
  function lines(cls, from, to, alpha) {
    const [rgb, width] = LOOK[cls], list = map.segs[cls];
    g.strokeStyle = `rgba(${rgb}, ${alpha})`;
    g.lineWidth = width * ratio;
    g.beginPath();
    for (let i = from; i < to; i++) {
      const s = list[i], [ax, ay] = eye(s[1], s[2], s[3]), [bx, by] = eye(s[4], s[5], s[6]);
      g.moveTo(ax, ay); g.lineTo(bx, by);
    }
    g.stroke();
  }
  function houses(from, to, alpha) {
    g.fillStyle = `rgba(${HOUSE[0]}, ${alpha})`;
    g.beginPath();
    for (let i = from; i < to; i++) {
      const h = map.houses[i], [x, y] = eye(h[1], h[2], h[3]), r = Math.max(0.6 * ratio, Math.min(3 * ratio, h[4] / mpp)) / 2;
      g.rect(x - r, y - r, 2 * r, 2 * r);
    }
    g.fill();
  }
  // the classes drawn in this order, the finer under the coarser
  const ORDER = ['path', 'street', 'major', 'lake', 'coast'];
  function all() {
    const W = canvas.width, H = canvas.height;
    eye = introEye(map, W, H);
    mpp = map.w / Math.abs(eye(map.bx + map.w, 0, 0)[0] - eye(map.bx, 0, 0)[0]);
    const sky = g.createRadialGradient(W / 2, H / 2, 0, W / 2, H / 2, Math.hypot(W, H) / 2);
    sky.addColorStop(0, '#0d121c'); sky.addColorStop(1, '#070a10');
    g.fillStyle = sky;
    g.fillRect(0, 0, W, H);
    g.lineCap = 'round'; g.lineJoin = 'round';
    houses(0, map.houses.length, HOUSE[2]);
    for (const k of ORDER) if (LOOK[k][3]) lines(k, 0, map.segs[k].length, LOOK[k][3]);
    for (const k of ORDER) if (!LOOK[k][3]) lines(k, 0, map.segs[k].length, LOOK[k][2]);
    for (const k of CLASSES) done[k] = 0;
    doneHouses = 0;
    reached = -1;
  }
  return function draw(share) {
    const size = `${canvas.width}x${canvas.height}`;
    if (size !== at) { at = size; all(); }
    const r = Math.min(1, share) * map.reach * 1.0001;
    if (r <= reached) return;
    reached = r;
    const upto = list => { let i = 0, j = list.length; while (i < j) { const m = (i + j) >> 1; if (list[m][0] <= r) i = m + 1; else j = m; } return i; };
    const n = upto(map.houses);
    if (n > doneHouses) { houses(doneHouses, n, HOUSE[1] - HOUSE[2]); doneHouses = n; }
    for (const k of ORDER) {
      if (!LOOK[k][3]) continue;
      const m = upto(map.segs[k]);
      if (m > done[k]) { lines(k, done[k], m, LOOK[k][2]); done[k] = m; }
    }
  };
}
