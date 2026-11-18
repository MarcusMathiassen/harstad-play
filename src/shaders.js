// WGSL for every pipeline. One shared prelude (the frame's uniforms, the painted lamp light, the night's two
// real lights, tone mapping, haze), then a vertex and fragment pair per kind of thing.
//
// The look is the three.js game's, reproduced term by term: Lambert under a hemisphere light and the moon,
// plus the lamp map by world position; the eye and an extended Reinhard tone map (tone()); sRGB encoding done by hand into a
// plain (non-sRGB) target, with the haze mixed in after it, as three does. Depth is reversed (1 near, 0 far).

import { snowRelief, FLIGHT } from './textures.js';

// For looking into the frame (?off=a,b on the page): the shaders are made with the named parts changed. Nothing is in play.
const OFF = new Set((new URL(import.meta.url).searchParams.get('off') || '').split(','));
// The aurora (aurora_state): ?aurora=0 none, ?aurora=1 an event from the moment play starts (?clock=, else 0; or from
// ?auroraat=), then as they come; ?aurora=storm a breakup that never ends. Without it the events come as the game's clock has them.
const AURORA_ARG = new URL(import.meta.url).searchParams.get('aurora');
// ?hdr=1 (src/worker.js imports the shaders so where its canvas is HDR): the light may rise over white to HDR.peak (tone)
const HDR_ON = new URL(import.meta.url).searchParams.get('hdr') === '1';
const AURORA_MODE = AURORA_ARG === '0' ? 0 : AURORA_ARG === '1' ? 1 : AURORA_ARG === 'storm' ? 2 : 3;
const AURORA_ARGS = new URL(import.meta.url).searchParams; // ?auroraat=: the forced event's start, for pictures of all of it
const AURORA_AT = +(AURORA_ARGS.get('auroraat') || AURORA_ARGS.get('clock') || 0) || 0;
// The aurora's look: the radiance of a curtain at full strength; how far its folds swing it across (in p, the sky
// over the horizon's plane: 1 is 45 degrees up at the horizon's foot) when most active; the lower border's N2 at its
// most active, of the green; the pulsating patches, of a curtain. Its events (aurora_state), in s: one starts in each
// slot of the clock, somewhere in the first \`jitter\` after \`lead\`; it grows for \`grow\` (and up to as much again),
// breaks up for \`breakup\` (and up to half as much again), and recovers for \`recover\` (and up to half again).
const AURORA = { bright: 0.2, fold: 0.45, rim: 0.85, patch: 1.0, slot: 630, lead: 40, jitter: 200, grow: 90, breakup: 60, recover: 100 };
const AURORA_LONGEST = 2 * AURORA.grow + 1.5 * AURORA.breakup + 1.5 * AURORA.recover; // s: an event at its longest, 420: over before the next slot's can start

const srgb = h => [h >> 16 & 255, h >> 8 & 255, h & 255].map(v => { v /= 255; return v < 0.04045 ? v * 0.0773993808 : Math.pow(v * 0.9478672986 + 0.0521327014, 2.4); });
const vec3 = a => `vec3f(${a.map(v => v.toFixed(6)).join(', ')})`;
// The twilight's light (struct clock_light, dusk): its colour, the clear sky's after sunset, blue (scaled to a luminance of
// 1); what the snow gives back of it from below; and what the clear air holds of it, seen across the fjord
const DUSK = { colour: [0.62, 0.77, 1.0].map(v => v / (0.2126 * 0.62 + 0.7152 * 0.77 + 0.0722)), ground: 0.75, haze: 0.04 };
const scaled = (h, k) => vec3(srgb(h).map(v => v * k));
const MOON = [0.46, 0.62, 0.64].map(v => v / Math.hypot(0.46, 0.62, 0.64)); // high in the south-east (wasm/include/harstad/moon.h)
const lum = c => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
// ---- one photometric scale (wasm/include/harstad/gfx.h, the same numbers: test_photometry) ----
// A unit of the scene's linear radiance is SCENE_CD cd/m2, for every light alike: the moon's and the sky's, the street
// lamps' (STREET_LUX at the lamp map's full, LAMP_STREET_LUX), the lit windows' (WINDOW_CD through their curtains), the
// lenses' (gfx_lens_gain). The screen shows it at the exposure, with a neutral shoulder over a knee (tone(), DISPLAY).
const PHOTO = [16, 20, 40, 0.14, 0.81]; // and the full moon's light on the ground, lux, and its own share of it (clock.h)
const [SCENE_CD, WINDOW_CD, STREET_LUX] = PHOTO;
export const CD_LAMP = 1.5e4; // cd/m2: a street lamp's, a flood's or a mast's lens (wasm/include/harstad/gfx.h, GFX_CD_LAMP)
export const CD_BEACON = 6e4; // cd/m2: an LED beacon's lens (wasm/include/harstad/gfx.h, GFX_CD_BEACON), the blue lights'
export const LENS_FLOOR = 0.06; // GFX_LENS_FLOOR
export const LENS_WHITE = 2.0; // GFX_LAMP_LENS_WHITE: a lens's colour at this luminance burns in full, its share of that being its luminance over it
// The night's own light: the full moon's on the ground (clock.h, CLOCK_MOON_LUX, of which CLOCK_MOON_SHARE its own),
// which MOON_FULL (its colour at 0.22 NIGHT) lays at the moon's height, in units. It scales the moonlight, the sky's and
// the ground's ambient light, and the sky itself with its cloud, its stars, the aurora and the moon's aureole. Those
// were drawn with NIGHT 0.45 when a unit was 3.4 cd/m2, against lamps too dim and lenses too bright for that: MOONLIT is
// the share of how they were drawn that is their true light now.
const MOON_LUX = PHOTO[3] * PHOTO[4];
const NIGHT = MOON_LUX / SCENE_CD / (lum(srgb(0xb4c2ee)) * 0.22 * MOON[1]);
const MOONLIT = NIGHT / 0.45;
// The clouds' shadow from the moon (CLOUD_SHADE): a map CLOUD_SHADE_PX texels a side over CLOUD_SHADE_SPAN m round the
// eye, on the plane at sea level along the moon's way: twice the reach the clouds are drawn to, 32 m a texel, finer than
// the cells of a cloud (95 m and up) and as fine as the moon's light under a cloud 400 m up can tell from the noise's mips
export const CLOUD_SHADE_PX = 1024;
const CLOUD_SHADE_SPAN = 32768;
const CLOUD_SHADE_AT = /* wgsl */`
fn cloud_shade_origin() -> vec2f { return floor(F.cam_pos.xz * ${(CLOUD_SHADE_PX / CLOUD_SHADE_SPAN).toFixed(6)}) * ${(CLOUD_SHADE_SPAN / CLOUD_SHADE_PX).toFixed(1)} - ${(CLOUD_SHADE_SPAN / 2).toFixed(1)}; }`;
// The moon's field (wasm/src/moon.c): metres to its texel; the farthest occluder its reach tells apart, m; the moon's
// angular radius, rad. A surface reads it a metre out along its normal, where its own texel cannot shade it; its edge is
// half a texel's rise soft, and softer by the moon's disc at the distance of what casts it.
const MOON_CELL = 1, MOON_REACH_FAR = 500, MOON_RADIUS = 0.0045, MOON_STANDOFF = 1;
// The tiles both town-wide maps are kept in (wasm/include/harstad/tiles.h): texels a tile covers, its slot, the slots
// along a layer's side, a layer's side, and the page table's mark of a tile that is all one value.
const TILE_STEPS = 256, TILE_SIDE = TILE_STEPS + 1; // the map's tiles: terrain steps along a side, and posts (wasm/include/harstad/tile.h)
const TILES_SIDE = 62, TILES_SLOT = 64, TILES_LAYER = 1024, TILES_ACROSS = TILES_LAYER / TILES_SLOT, TILES_PER_LAYER = TILES_ACROSS ** 2, TILES_CONSTANT = 0xffff;
// The frame's page tables' entries in TABLES (TABLES_WGSL): each table's first, its levels' count's, and how many there are
export const TABLES = { map: 0, lamp: 16, tint: 32, reach: 48, tiles: 64, back: 80, sea: 96, levels: 15, reachFirst: 112, skyFirst: 113, glowFirst: 114, entries: 128 };
// the moon's field's first layer in moon_map, after the clouds' shade's two
export const MOON_FIRST = 2;
const LAMP_PX = 1.5; // m to the lamp map's texel (wasm/include/harstad/lamp.h)
const LAMP_SW_HEAD = 4, LAMP_SW_FIRST = 2, LAMP_SW_PALETTE = 255;
const SKYGLOW_PX = 8, SKYGLOW_COARSE = 4; // the town glow's lamp pixels to a texel, and its texels to a coarse one (wasm/include/harstad/skyglow.h) // the switches' buffer's words before its levels' (wasm/include/harstad/lamp.h, LAMP_SW_HEAD)
// The haze's map in tiles (lamp.h, LAMP_GLOW_*): its texels a tile covers along a side, the dense map's more round
// them in its slot, the slot, and a layer's side
const GLOW_SIDE = 248, GLOW_RIM = 4, GLOW_SLOT = GLOW_SIDE + 2 * GLOW_RIM, GLOW_LAYER = 1024, GLOW_ACROSS = GLOW_LAYER / GLOW_SLOT;
// A bilinear look into the lamp map as if it were whole, at s in its texels less a half (where the sampler would take
// it): into the slot of the tile of floor(s), which holds the texels round it that the look weighs (tiles.h, as map_at),
// or the one colour the tile holds throughout (a page's g and b, its low half first). Its alpha is the floor the light
// falls on: 1 the ground, less where a long bridge's deck stands over it (wasm/src/lamp.c, lamp__floors; lamp_floor_above)
// (dims, load: the page table's size and a texel of it: the frame's in TABLES, LAMP_GLOW's its own texture)
const lampLook = (dims, load) => /* wgsl */`
fn lamp_look(s: vec2f) -> vec4f {
  let t = lamp_tile(s);
  return lamp_look_in(s, t, ${load('t')});
}
// the tile of the dense map's tiles that a look at s takes
fn lamp_tile(s: vec2f) -> vec2i { return clamp(vec2i(floor((floor(s) + 0.5) * ${(1 / TILES_SIDE).toPrecision(9)})), vec2i(0), ${dims} - 1); }
// the look at s in tile t, whose page entry is p
fn lamp_look_in(s: vec2f, t: vec2i, p: vec4u) -> vec4f {
  if (p.x == ${TILES_CONSTANT}u) { return unpack4x8unorm(p.y | (p.z << 16u)); }
  let k = p.x | (p.y << 16u); // a slot: its number's low half, then its high
  let at = s - vec2f(t * ${TILES_SIDE} - 1) + 0.5;
  let uv = (vec2f(f32(k % ${TILES_ACROSS}u), f32(k / ${TILES_ACROSS}u % ${TILES_ACROSS}u)) * ${TILES_SLOT}.0 + at) * ${(1 / TILES_LAYER).toExponential()};
  return textureSampleLevel(lamp_map, samp_clamp, uv, i32(k / ${TILES_PER_LAYER}u), 0.0);
}
// The same look through a cubic B-spline over the 4 x 4 texels round s, in four bilinear looks (each weighed pair of
// texels along an axis one look placed between them, in the tile of its own floor): a bilinear look magnified, as the
// ground near by is some twenty pixels to a 1.5 m texel, shows its texels' grid as creases, and the bytes' faint skirts
// (a byte is 0.16 lux, about the moon's light) as flat steps with square edges. The spline is smooth across texels, so a
// pool is smooth at every scale; it keeps the map's total, and moves light by under a texel.
fn lamp_smooth(s: vec2f) -> vec4f {
  let i = floor(s); let f = s - i; let g = 1.0 - f;
  let w0 = g * g * g / 6.0; let w3 = f * f * f / 6.0; let w1 = (4.0 - 6.0 * f * f + 3.0 * f * f * f) / 6.0; let w2 = 1.0 - w0 - w1 - w3;
  let a = w0 + w1; let b = w2 + w3;
  let p = i - 1.0 + w1 / a; let r = i + 1.0 + w3 / b;
  // (the four looks' page entries read once for the tile they share, as they mostly do: a tile's slot holds a texel more
  // either way, so the spline's four 4 x 4 corners lie in one tile unless it is at the tile's edge)
  let s0 = p; let s1 = vec2f(r.x, p.y); let s2 = vec2f(p.x, r.y); let s3 = r;
  let t0 = lamp_tile(s0); let e0 = ${load('t0')};
  let t1 = lamp_tile(s1); var e1 = e0; if (any(t1 != t0)) { e1 = ${load('t1')}; }
  let t2 = lamp_tile(s2); var e2 = e0; if (any(t2 != t0)) { e2 = ${load('t2')}; }
  let t3 = lamp_tile(s3); var e3 = e0; if (any(t3 != t0)) { e3 = select(${load('t3')}, e2, all(t3 == t2)); }
  return (lamp_look_in(s0, t0, e0) * a.x + lamp_look_in(s1, t1, e1) * b.x) * a.y + (lamp_look_in(s2, t2, e2) * a.x + lamp_look_in(s3, t3, e3) * b.x) * b.y;
}
`;
// The town-wide maps' coarser levels (src/worker.js, pyramid): over a map's first level, levels l = 1, 2, ... of
// ceil(size 2^-l) texels across and down, each texel the mean of the 2 x 2 under it, packed in tiles as the first level
// is (tiles.h) into slots after the first level's. Level l's tiles are the page table's mip l: the table is a power of
// two across and down, so each level's tiles fit its mip.
// A page entry's channels chan and chan + 1 are a slot (low half, then high), or TILES_CONSTANT and the tile's value:
// kind 0 a half float, 1 four bytes, 2 four bytes in this channel and the next (low half first).
const PYR_PAGES = /* wgsl */`
fn pyr_size(size0: vec2f, l: u32) -> vec2f { return ceil(size0 * exp2(-f32(l))); } // level l's texels, of the first's size0
fn pyr_const(p: vec4u, chan: u32, kind: u32) -> vec4f {
  let c = select(p.y, p.w, chan == 2u);
  if (kind == 0u) { return vec4f(unpack2x16float(c).x); }
  if (kind == 1u) { return unpack4x8unorm(c); }
  return unpack4x8unorm(p.y | (p.z << 16u));
}
// texel c of level l, unfiltered
fn pyr_load(atlas: texture_2d_array<f32>, pages: texture_2d<u32>, l: u32, c: vec2u, chan: u32, kind: u32) -> vec4f {
  let t = c / ${TILES_SIDE}u;
  let p = textureLoad(pages, t, l);
  let k = select(p.x, p.z, chan == 2u);
  if (k == ${TILES_CONSTANT}u) { return pyr_const(p, chan, kind); }
  let n = k | (select(p.y, p.w, chan == 2u) << 16u);
  let at = vec2u(n % ${TILES_ACROSS}u, n / ${TILES_ACROSS}u % ${TILES_ACROSS}u) * ${TILES_SLOT}u + c - t * ${TILES_SIDE}u + 1u;
  return textureLoad(atlas, at, i32(n / ${TILES_PER_LAYER}u), 0);
}
`;
// A level of a town-wide map made from the level under it (src/worker.js, pyramid): each slot a quad TILES_SLOT pixels
// square in a layer laid out as the atlas's, each texel the mean of the four under it (clamped to that level's edges).
// U: the level, the page channel, the kind of constant, the steps to dither to (0: none); the first level's texels across and down.
export const PYR_MAKE = /* wgsl */`
@group(0) @binding(0) var src: texture_2d_array<f32>;
@group(0) @binding(1) var pages: texture_2d<u32>;
@group(0) @binding(3) var<uniform> U: array<vec4u, 2>;
@group(0) @binding(4) var<storage, read> slots: array<vec4u>; // each slot made: its number, its tile's column and row
${PYR_PAGES}
fn slot_at(k: u32) -> vec2f { return vec2f(f32(k % ${TILES_ACROSS}u), f32(k / ${TILES_ACROSS}u % ${TILES_ACROSS}u)) * ${TILES_SLOT}.0; }
struct V { @builtin(position) pos: vec4f, @location(0) @interpolate(flat) slot: vec4u }
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> V {
  let corner = array<vec2f, 6>(vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0), vec2f(0.0, 1.0));
  let s = slots[ii];
  let px = (slot_at(s.x) + corner[vi] * ${TILES_SLOT}.0) * ${(1 / TILES_LAYER).toExponential()};
  return V(vec4f(px.x * 2.0 - 1.0, 1.0 - px.y * 2.0, 0.0, 1.0), s);
}
@fragment fn fs(v: V) -> @location(0) vec4f {
  let l = U[0].x; let size0 = vec2f(U[1].xy); let size = vec2i(pyr_size(size0, l)); let under = vec2i(pyr_size(size0, l - 1u)) - 1;
  let c = clamp(vec2i(v.slot.yz * ${TILES_SIDE}u) - 1 + vec2i(v.pos.xy - slot_at(v.slot.x)), vec2i(0), size - 1);
  var sum = vec4f(0.0);
  for (var j = 0; j < 2; j++) { for (var i = 0; i < 2; i++) { sum += pyr_load(src, pages, l - 1u, vec2u(min(c * 2 + vec2i(i, j), under)), U[0].y, U[0].z); } }
  // U[0].w steps: the mean rounded to them through a 2 x 2 ordered dither (-3/8, -1/8, 1/8, 3/8 of a step), unbiased,
  // so a faint level keeps its light on the whole; a constant stays itself
  let k = vec2u(c) & vec2u(1u);
  let dither = select(0.0, (f32(k.x * 2u + (k.x ^ k.y)) - 1.5) * 0.25 / f32(max(U[0].w, 1u)), U[0].w > 0u);
  return sum * 0.25 + dither;
}
`;
// A look into the haze's map (lamp_glow, the lamp map's first mip and the one under it) as if it were whole, at world xz
// and level l: s = uv size - 0.5 in that level's texels, the dense level's look clamped to its edges; the tile of floor(s)
// (GLOW_SIDE >> l of its texels a side), its slot from channel 2 of the tint's page table at the tile's first tint tile,
// which holds every texel the look weighs, each level being the dense map's moved by whole texels; or black throughout.
// The dense map was looked into at levels 0.5 and 1.5 with the sampler's nearest mip, which took levels 0 and 1.
const GLOW_LOOK = /* wgsl */`
fn glow_s(xz: vec2f, l: u32) -> vec2f {
  let size = max(floor(floor(round(F.lamp_rect.zw / ${LAMP_PX}) * 0.5) / f32(1u << l)), vec2f(1.0));
  return clamp((xz - F.lamp_rect.xy) / F.lamp_rect.zw, vec2f(0.0), vec2f(1.0)) * size - 0.5;
}
fn glow_look(xz: vec2f, l: u32) -> vec3f { return glow_at(glow_s(xz, l), l); }
// the same through a cubic B-spline over the 4 x 4 texels round it (as lamp_smooth): level 0's texel is 3 m, and the snow
// a few metres off sees it magnified some fifty times, where a bilinear look shows its grid as creases and squares
fn glow_smooth(xz: vec2f) -> vec3f {
  let s = glow_s(xz, 0u); let i = floor(s); let f = s - i; let g = 1.0 - f;
  let w0 = g * g * g / 6.0; let w3 = f * f * f / 6.0; let w1 = (4.0 - 6.0 * f * f + 3.0 * f * f * f) / 6.0; let w2 = 1.0 - w0 - w1 - w3;
  let a = w0 + w1; let b = w2 + w3;
  let p = i - 1.0 + w1 / a; let r = i + 1.0 + w3 / b;
  return (glow_at(p, 0u) * a.x + glow_at(vec2f(r.x, p.y), 0u) * b.x) * a.y + (glow_at(vec2f(p.x, r.y), 0u) * a.x + glow_at(r, 0u) * b.x) * b.y;
}
// level l's look at s in its texels less a half
fn glow_at(s: vec2f, l: u32) -> vec3f {
  let side = f32(${GLOW_SIDE}u >> l);
  let g = clamp(vec2i(floor((floor(s) + 0.5) / side)), vec2i(0), vec2i(ceil(floor(round(F.lamp_rect.zw / ${LAMP_PX}) * 0.5) * ${(1 / GLOW_SIDE).toPrecision(9)})) - 1);
  let k = tab_load(${TABLES.tint}u, g * 4).z;
  if (k == ${TILES_CONSTANT}u) { return vec3f(0.0); }
  let q = k % ${GLOW_ACROSS * GLOW_ACROSS}u;
  let at = vec2f(f32(q % ${GLOW_ACROSS}u), f32(q / ${GLOW_ACROSS}u)) * f32(${GLOW_SLOT}u >> l) + s - (vec2f(g) * side - f32(${GLOW_RIM}u >> l)) + 0.5;
  return textureSampleLevel(lamp_map, samp_clamp, at / f32(${GLOW_LAYER}u >> l), tab_first(${TABLES.glowFirst}u) + i32(k / ${GLOW_ACROSS * GLOW_ACROSS}u), f32(l)).rgb;
}
`;
const MOON_SOFT = 0.5 * MOON_CELL * MOON[1] / Math.hypot(MOON[0], MOON[2]);
// The sky the ground sees (wasm/src/ao.c), as it is let into the light: what stands within `high` m of the ground reads
// it, fully up to the first and not at all above the second; a surface reads it `standoff` m out along its normal, so a
// wall reads the snow at its foot; what faces sideways sees none of the sky behind it, which its foot's texel counts
// against it, so it takes `side` times that texel's share. The sky's and the moon's scattered light take it whole, the
// painted lamplight `lamps` of it: the pools are painted with the buildings' shadows already.
// Light inside the snow (subsurface scattering): snow is ice grains and air, so light goes in, wanders and comes out
// elsewhere. Where it has wandered far (sheltered hollows, a wall's foot, ruts) it comes out bluer, ice taking a little
// of the red over a long path: albedo times SSS.deep by how little sky the place sees, SSS.deep_k as much at the most.
// And a lamp's pool on snow spreads past its painted edge a little and glows there: the lamp map blurred (SSS.lod
// mips, some 4 m) less the pool itself, SSS.bleed of it. ?off=sss leaves both out.
const SSS = { deep: [0.66, 0.84, 1.16], deep_k: 1.6, bleed: 0.9, lod: 1.5 };
const SKY_SEEN = { high: [0.5, 4.0], standoff: 0.5, side: 1.3, lamps: 0.8, deepen: 1.8 }; // deepen: the map's value to this power, so the corners and the walls' feet read at night
const SHADOW_HALF = 300, SHADOW_FADE = 50; // the near map's reach to each side of its centre (wasm/src/game.c), and the last of it the field fades in over
// The snow's surface (src/world.js, snowSurface): drifts lying across the wind, one relief tile spanning so many metres
// along and across it, and their height scale; and sastrugi, the small sharp ridges the wind carves along itself.
const DRIFT = [11, 36, 0.45], RIDGE = [6, 1.7, 0.022], RELIEF_TOP = snowRelief().slope;
// Where the snow's finer looks give out, by how they show on screen rather than how far off they are: the sastrugi and
// the drifts fade once a wave of either is under RIDGE_PX or DRIFT_PX pixels long (where they were faded at 120 and 700 m
// on a screen 1514 pixels high), and glints once GLINT_SPAN m is under a pixel (350 m there). A bigger or sharper screen
// shows them further off, a smaller one less far: they go where they would no longer be told apart, not at a distance.
const RIDGE_PX = 19.4, DRIFT_PX = 21.5, GLINT_SPAN = 0.256;
const GLINT_MIN = 0.04; // light below which no glints are looked for (faint window spill, moonlight)
// Rime (rime_share, rime_glints): the most of the way to its albedo a surface goes, of that what PLAIN's surfaces take (a
// mix of thin things and a shed's walls) and the trees' (whose models are painted with rime already: tree.c), fine trim
// taking more; the share of it all round (the rest faces the wind); the smallest cell a needle sits in (m), the share of cells that hold one, how bright one is,
// and the patch of rime that must be a pixel or more for any to be looked for (m; some 60 m off on a screen 1514 high)
const RIME = { white: 0.15, plain: 0.5, tree: 1.0, fine: 0.7, round: 0.35, cell: 0.012, grains: 0.35, bright: 6.0, span: 0.045 };
// The clear air (aerial(), air_tau, air_lit): its extinction a metre at sea level (some 20 km of meteorological range,
// a winter's coastal haze), thinning by e every 300 m up; the share of what it takes out that it scatters, its albedo,
// and its phase's asymmetry, an aerosol's, forward; the town's light in it, of the light leaving the town's ground
// under it (town_glow: air lit from below by a bright ground, scattering it evenly, has a radiance of 1 / 2 pi of the
// ground's exitance); how far out over the horizon the sky's dome of that light is looked for (m), and how much of it
// the dome shows; the most the sea smoke takes
const LAMP_LENS = 0.7; // m: across a street lamp's lens, which its light leaves from (lamp_heads_up)
// the street lamps' light (street_light()): the lamp map's full is STREET_LUX on the ground, which a white surface gives
// back as STREET_LUX / pi cd/m2; what it lights takes it times 1.6 (lamp_tilt's full, and the gain the ground's shading
// gives it), so 0.497 of the map's full: a lit Norwegian street at night, 6 to 13 lux, not a floodlit one
const STREET = STREET_LUX / (1.6 * Math.PI * SCENE_CD);
const AIR = { sigma: 1.8e-4, h: 300, albedo: 0.95, g: 0.5, glow: 1 / (2 * Math.PI), dome: [500, 1800, 5000], most: 0.55, smoke: 0.3 };
// The town's own light given back round it (wasm/src/skyglow.c, the map's texel and how far it reaches past the lamp
// map's): the light leaving its ground, spread as the air and the cloud over it see it, of which the clear air sends
// `back` down again (skyglow.h, SKYGLOW_BACK: some 0.02 lux over the middle of the town, a fifth of the full moon's;
// a small town's skyglow on the ground is 0.001 to 0.05 lux); from below, by the snow round the lamps' pools. Off
// the map it fades on by e every `fade` m, as the widest of its spreads does. `below` of the light from below is let in
// past what a surface facing down takes of the pools under it already (lamp_tilt's half); `ground` of the light from
// above the snow gives back from under, its albedo. WILD: of the sky's and the ground's light as they were everywhere
// (dd15f21), what is the night's own, the moon's light scattered in the air and off the snow: the rest was the town's,
// and is now its own, where it is.
const TOWN = { cell: 12, margin: 3000, fade: 1000, back: 0.015, below: 0.25, ground: 0.75 };
const WILD = 0.2;
// the full moon's light on the ground and its sky's, as luminance: what the twilight's is counted in (clock.h, CLOCK_MOON_LUX)
const DUSK_REF = lum(srgb(0xb4c2ee).map(v => v * 0.22 * NIGHT)) * MOON[1] + lum(srgb(0xa3a9c8).map(v => v * 0.21 * NIGHT * WILD));
// the ripple tiles' size and how many waves each sums; the size of the sky the water mirrors
export const RIPPLE_N = 128, RIPPLE_WAVES = 28, SKY_W = 256, SKY_H = 64;
const reach = L => `vec2f(${L[0].toFixed(5)}, ${L[1].toFixed(5)})`, steep = L => `vec2f(${(RELIEF_TOP * L[2] / L[0]).toFixed(5)}, ${(RELIEF_TOP * L[2] / L[1]).toFixed(5)})`;

// The jets' light (car_jets): from no nearer than `near` m, gone by `reach` m; and how much of it what faces away
// still gets, round the plume's length; how wide a flame is as the water mirrors it, m
const JET_LIGHT = { near: 0.3, reach: 18, wrap: 0.1, girth: 0.35 };
// A tail lamp: a few candela of deep red behind its lens, the brake lamp some ten times that. lens: the radiance a lit
// lens is drawn at, running (the car's own, tools/car/car.js's RED; the traffic's burn as they truly do, lamp_true), low enough that
// the clip at white keeps it a saturated red, under braking too; pool: the strength of the light both lamps lay behind
// the car, running (wasm/src/game.c, TAIL_GLOW: the car's own); wet: of that, the power their reflection on a wet road
// has (WET_BIN). Braking, the traffic's burn TAIL_BRAKE times as bright (game.c's cars_dir z), the car's own as avensis.c
// and game.c have it. All of it shines back from the lamps and out to the sides: none of it falls on the car's own body.
const TAIL = { lens: [0.2, 0.002, 0.001], pool: 0.55, wet: 0.1 };
// Full beam: m its main beams light the road out to (wasm/src/game.c, GAME_FULL_REACH), and how much brighter the
// headlamps' lenses burn on it than dipped (wasm/src/avensis.c, FULL_LENS)
const FULL_LENS = 3.0;
// A traffic lamp's lens as it truly burns (wasm/src/traffic.c, traffic_draw: gfx_lens_gain of GFX_CD_HEAD and GFX_CD_TAIL,
// wasm/include/harstad/gfx.h): its colour scaled to its luminance in cd/m2, over SCENE_CD. The glow sprite draws a lens
// that far off at this radiance times its beam towards the eye; near by, where the lens is drawn by the car's own mesh
// (lamp_true), the mesh burns at the same, so the lens does not dim as it grows from a point to a disc to the lamp
export const LENS_TRUE = (() => {
  const at = (c, cd) => c.map(v => v * cd / (SCENE_CD * lum(c)));
  return { head: at([1.7, 1.55, 1.3], 1e5), tail: at([2.0, 0.025, 0.012], 1e3), running: 0.6 }; // a tail lamp running, of braking
})();
// The lamps' light on the scene (wasm/include/harstad/gfx.h, the same numbers: test_light_falloff): the illuminance under
// which a lamp's light is gone (lux), how near it its light starts (m), and in candela a dipped halogen's hot spot, a
// main beam on its axis and a tail lamp running. A lamp of I cd reaches sqrt(I / LUX_SEEN) m (gfx_light_reach): a car's
// pair of dipped beams DIP_REACH, of main beams FULL_REACH (game.c, GAME_FULL_REACH).
const LIGHT = [1.0, 0.1, 12000, 51200, 5];
const [LUX_SEEN, LIGHT_NEAR, I_DIP, I_FULL, I_TAIL] = LIGHT;
const lightReach = cd => Math.sqrt(cd / LUX_SEEN);
const DIP_REACH = lightReach(2 * I_DIP), FULL_REACH = lightReach(2 * I_FULL);
// rad: how far across from its axis a headlamp's beam's skirt has faded to nothing (gfx.h, GFX_HEAD_SPREAD)
const HEAD_SPREAD = 1.4;
// cd: a plough's LED work lights, the roof's on the blade and the road ahead, and the wing's and the spreader's; and its
// beacons' beams (wasm/src/plough.c, BEACON_CD), its colour F.plough's. Its lights reach as far as the roof's does
// (plough.c, LIGHTS_REACH: F.plough_at's w).
const PLOUGH_WORK = [2500, 600, 1000];
// a colour given a luminance of 1, so that times I cd it is I's luminous intensity
const luminous = c => vec3(c.map(v => v / lum(c)));
// the tail lamps' pool, in cd of their colour (TAIL_RED times TAIL.pool, F.tail's and the traffic's) as a pair running
const TAIL_CD = 2 * I_TAIL / (TAIL.pool * lum([1.0, 0.01033, 0.003677]));
// The moving lights' shadows over the buildings (wasm/include/harstad/seen.h): sectors across a car's beams, all round a
// lamp, m a step of a sector's distance; the round lights after the traffic's, in the order game.h lists them
const SEEN = { car: 48, round: 64, step: 0.5, carHalf: 0.85 }; // carHalf: a car's sectors' half width, 80 degrees (seen.h, SEEN_CAR_HALF)
// The light list (wasm/include/harstad/shine.h): its grid's buckets, its sectors (the car's own beams, then the fixed
// round lights': the ploughs' cabs 0 + k, hoppers 3 + k, the truck's 6 + i, the Julehus's 10 + i; then each car's, then
// each event light's), and where its index list starts in the grid's words
const SHINE = { buckets: 32768, cell: 32, cars: 3584, events: 160, heads: 16384 };
const GAME_LIT_GRID = 256; // the grid of where the moving lights reach, cells a side (wasm/include/harstad/game.h)
SHINE.sectors = SEEN.car + 12 * SEEN.round + SHINE.cars * SEEN.car + SHINE.events * SEEN.round;
SHINE.sec0 = SHINE.buckets + 1; SHINE.index0 = SHINE.sec0 + SHINE.sectors / 2;
SHINE.heads0 = SHINE.index0 + (1 << 21); SHINE.hindex0 = SHINE.heads0 + SHINE.buckets + 1; // the street lamps' heads' grid (shine.h, SHINE_HEADS0)
SHINE.boxes = 736; SHINE.clouds0 = SHINE.hindex0 + (1 << 18); SHINE.cindex0 = SHINE.clouds0 + SHINE.boxes + 1; // each cloud's lights (shine.h, SHINE_CLOUDS0)
export const SHINE_LIGHTS = SHINE.cars + SHINE.events + SHINE.heads, SHINE_GRID_WORDS = SHINE.cindex0 + (1 << 14), SHINE_LIGHT_BYTES = 80, SHINE_BUCKETS = SHINE.buckets, SHINE_INDEX0 = SHINE.index0;
export const LAMP_SW0 = SHINE_GRID_WORDS; // where the lamp switches' words begin in shine_grid's buffer
export const SHINE_MOVING = SHINE.cars + SHINE.events, SHINE_HEADS0 = SHINE.heads0, SHINE_CLOUDS0 = SHINE.clouds0, SHINE_CLOUD_BOXES = SHINE.boxes;
const seenRound = k => `${SEEN.car + k * SEEN.round}u`; // the k-th fixed round light's first sector
// The lighting everything shares (lit_over, in PRELUDE). gated: the same for a shader that has asked lights_reach where
// it stands and set lit_moving by it, so that where no moving light reaches none is looked for; only such a shader
// carries it, as the test costs every other one more than it saves.
const litOver = (name, gated) => /* wgsl */`// above: w's height over the ground, w.y - ground_at(w.xz), which the painted light, the sky's share and the light inside
// the snow all go by: looked up once, by whoever needs it first
fn ${name}(albedo: vec3f, n0: vec3f, w: vec3f, gain: f32, cover_in: f32, relief: f32, dx: vec2f, dy: vec2f, moon_floor: f32, reach: vec3f, own: f32, above: f32) -> vec3f {
  var cover = cover_in;
  if (cover < 0.0) { cover = smoothstep(0.42, 0.6, min(albedo.r, min(albedo.g, albedo.b))) * smoothstep(-0.03, 0.02, albedo.b - albedo.r); }
  cover *= smoothstep(0.5, 0.75, n0.y);
  var n = n0;
  let dist = distance(w, F.cam_pos.xyz);
  if (cover > 0.0${OFF.has('relief') ? ' && false' : ''}) { // the wind relief tilts the normal before any light is gathered
    let slope = snow_slope(w.xz, dist, dx, dy) * cover * relief;
    n = normalize(n0 - vec3f(slope.x, 0.0, slope.y));
  }
${gated ? `  var lamps = vec3f(0.0); var under = 1.0; var wash = vec3f(0.0);
  if (lit_moving) {
    lamps = car_lamps(w, n);
    under = traffic_shade * truck_shade(w, n);
    wash = tower_wash(w, n);
  }
  let seen = sky_seen_over(w, n, above) * own;
  let open = seen * under;
  let direct = moon_light(w, n, moon_floor * open) + lamps + wash;` : `  let lamps = car_lamps(w, n); // first: it finds the traffic's shade
  let seen = sky_seen_over(w, n, above) * own; let under = traffic_shade * truck_shade(w, n); // the cars stand between the snow and the lamps overhead too
  let open = seen * under;
  let direct = moon_light(w, n, moon_floor * open) + lamps + tower_wash(w, n);`}
  let lamp_in = reach * lamp_tilt(n) * (gain * 1.6 * mix(1.0, seen, ${SKY_SEEN.lamps.toFixed(2)}) * under);
  var rime = 0.0; // what lies as snow carries none of its own (rime_share)
  if (rime_on > 0.0${OFF.has('rime') ? ' && false' : ''}) { // nor does snow on a steep face, whose colour says it is snow all the same
    let pale = smoothstep(0.42, 0.6, min(albedo.r, min(albedo.g, albedo.b))) * smoothstep(-0.03, 0.02, albedo.b - albedo.r);
    rime = rime_share(n, seen) * (1.0 - max(cover, pale));
  }
  var c = max(albedo, mix(albedo, RIME_ALBEDO, rime * RIME_WHITE)) * (((sky_light_over(w.xz, n.y, reach) + aurora_light(n) * seen) * open + direct) / PI + lamp_in);
${OFF.has('sss') ? '' : `  if (cover > 0.0) { // the light inside the snow (SSS)
    let blue = mix(vec3f(1.0), vec3f(${SSS.deep.map(v => v.toFixed(3)).join(', ')}), clamp((1.0 - seen) * ${SSS.deep_k.toFixed(2)}, 0.0, 1.0) * cover);
    c *= blue;
    if (above < 1.5) {
      let m = glow_smooth(w.xz); // (level 0.5, as the dense map was looked into: the nearest mip, 0; smooth across its texels)
      let spread = street_light(m) + vec3f(1.0, 0.58, 0.26) * m.g * 1.25;
      c += albedo * blue * max(spread - reach, vec3f(0.0)) * (${SSS.bleed.toFixed(2)} * gain * 1.6 * cover * under * (1.0 - smoothstep(0.5, 1.5, above)));
    }
  }`}
  if (cover > 0.0${OFF.has('glints') ? ' && false' : ''}) { c += snow_glints(w, n, lamp_in, direct / PI, dx, dy, dist) * cover; }
  if (rime > 0.0${OFF.has('glints') ? ' && false' : ''}) { c += rime_glints(w, n, lamp_in, direct / PI, dist) * rime; }
  return c;
}
`;
// The frame's page tables in one texture (src/worker.js, tableAtlas), so that a shader reads no more textures than
// WebGPU's default sixteen a stage: each table's level l at entry TABLES.<table> + l, its levels' count at entry
// TABLES.<table> + 15. An entry is the first row's texel at its number: the corner where that level lies, its width and
// height. tab_load reads a texel of it as textureLoad read the table's own texture, its dimensions as textureDimensions
// did; the tiles' layers are i32 and the backdrop's heights f32, as their bits.
const TABLES_WGSL = /* wgsl */`
fn tab(e: u32) -> vec4u { return textureLoad(tables, vec2u(e, 0u), 0); }
fn tab_dims(e: u32) -> vec2i { return vec2i(tab(e).zw); }
fn tab_levels(e: u32) -> u32 { return tab(e + ${TABLES.levels}u).x; }
fn tab_first(e: u32) -> i32 { return i32(tab(e).x); } // a map's first layer in the texture it shares
fn tab_load(e: u32, t: vec2i) -> vec4u { return textureLoad(tables, vec2i(tab(e).xy) + t, 0); }
`;
// The ground's height at xz as the height map holds it, in tiles (wasm/src/wasm.c, hs_tile_layer): a layer of map a
// tile the map holds, TILE_SIDE posts a side from its first, so the four posts of each of its cells are in one layer
// and the look is bilinear in it, as in the one map over the tiles' bounding box that the layers took the place of
// (clamped at its edges as that was); where no tile is, the backdrop's triangles as world_grid_at has them over outer.bin
// (back, a post round its edge: texel i + 1 is post i). tiles: which layer each tile of the tiles' grid is, -1 for
// none. rect, at, bk: F.ground_rect, F.ground_tiles, F.ground_back (struct frame_uniforms, wasm/include/harstad/game.h).
// The look itself, as it reads the tiles' layers (dims, layer) and the backdrop's posts (back): GROUND_LOOK with them as
// textures, ground_at (the frame's) with them in TABLES
const groundLook = (head, { map, samp, rect, at, bk }, dims, layer, back) => /* wgsl */`
${head}
  let p = clamp((xz - ${rect}.xy) / ${rect}.zw * ${at}.zw - 0.5, vec2f(0.0), ${at}.zw - 1.0); // the grid's posts, the map's texels
  let t = clamp(vec2i(floor((p - ${at}.xy) / ${TILE_STEPS}.0)), vec2i(0), ${dims} - 1);
  let layer = ${layer('t')};
  if (layer >= 0) { return textureSampleLevel(${map}, ${samp}, (p - ${at}.xy - vec2f(t) * ${TILE_STEPS}.0 + 0.5) / ${TILE_SIDE}.0, layer, 0.0).r; }
  let g = clamp((${rect}.xy + (p + 0.5) * ${rect}.zw / ${at}.zw - ${bk}.xy) / ${bk}.z, vec2f(0.0), vec2f(${bk}.w - 1.0001));
  let i = vec2i(g) + 1; let f = g - floor(g);
  let a = ${back('i')}; let b = ${back('i + vec2i(1, 0)')};
  let c = ${back('i + vec2i(0, 1)')}; let d = ${back('i + vec2i(1, 1)')};
  return select(d + (c - d) * (1.0 - f.x) + (b - d) * (1.0 - f.y), a + (b - a) * f.x + (c - a) * f.y, f.x + f.y <= 1.0);
}
`;
const GROUND_LOOK = groundLook('fn ground_look(map: texture_2d_array<f32>, tiles: texture_2d<i32>, back: texture_2d_array<f32>, samp: sampler, rect: vec4f, at: vec4f, bk: vec4f, xz: vec2f) -> f32 {',
  { map: 'map', samp: 'samp', rect: 'rect', at: 'at', bk: 'bk' }, 'vec2i(textureDimensions(tiles))', t => `textureLoad(tiles, ${t}, 0).r`, i => `textureLoad(back, ${i}, 0, 0).r`);
const GROUND_AT = groundLook('fn ground_at(xz: vec2f) -> f32 {', { map: 'ground_map', samp: 'samp_clamp', rect: 'F.ground_rect', at: 'F.ground_tiles', bk: 'F.ground_back' },
  `tab_dims(${TABLES.tiles}u)`, t => `bitcast<i32>(tab_load(${TABLES.tiles}u, ${t}).x)`, i => `bitcast<f32>(tab_load(${TABLES.back}u, ${i}).x)`);
// HDR output (src/worker.js): the second shoulder's start, of white, and the peak the light may rise to, of SDR white.
// It starts where SDR's own shoulder has all but reached white (0.99, a luminance of 2.26 at the exposure): a lit window,
// a lamp-lit wall, the snow under a lamp stay as SDR shows them, under white; only what is far brighter than white, a
// lamp's lens, a headlamp, rises over it (a street lamp's lens to some 5.6 of white, a headlamp to the peak)
export const HDR = { shoulder: 0.99, peak: 6.0 };
// The frame onto the canvas (src/worker.js, present): the scene is drawn into a float texture as the screen shows it
// (encode(tone()), as an 8-bit canvas took it before, so that what is blended over it blends as it did), and taken from
// there onto the canvas: on an HDR screen (a float canvas, extended tone mapping) as extended sRGB, still encoded with
// the sRGB transfer curve (a WebGPU canvas's colour space, 'srgb' or 'display-p3', always carries it, float or not), whose
// highlights go over white; on an SDR one, or into a grab's 8 bits, as it is, with a dither of a 255th (triangular, by
// the pixel) so that the sky's and the sea's slow gradients show no steps.
export const PRESENT = /* wgsl */`
@group(0) @binding(0) var scene: texture_2d<f32>;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f { return vec4f(f32(i & 1u) * 4.0 - 1.0, f32(i >> 1u) * 4.0 - 1.0, 0.0, 1.0); }
fn ign(p: vec2f) -> f32 { return fract(52.9829189 * fract(dot(p, vec2f(0.06711056, 0.00583715)))); }
@fragment fn hdr(@builtin(position) p: vec4f) -> @location(0) vec4f {
  // the same dither as the SDR's: the screen composites the float canvas at 8 or 10 bits, where the night's slow
  // gradients would step
  let n = ign(p.xy) + ign(p.yx + vec2f(37.0, 11.0)) - 1.0;
  return vec4f(max(textureLoad(scene, vec2i(p.xy), 0).rgb + n * ${(1 / 255).toFixed(8)}, vec3f(0.0)), 1.0);
}
@fragment fn sdr(@builtin(position) p: vec4f) -> @location(0) vec4f {
  let c = textureLoad(scene, vec2i(p.xy), 0).rgb;
  let n = ign(p.xy) + ign(p.yx + vec2f(37.0, 11.0)) - 1.0;
  return vec4f(clamp(c, vec3f(0.0), vec3f(1.0)) + n * ${(1 / 255).toFixed(8)}, 1.0);
}
`;
export const PRELUDE = /* wgsl */`
struct Frame {
  view_proj: mat4x4f,
  cam_pos: vec4f,     // w: time
  cam_right: vec4f,   // scaled by tan(fov_x / 2); w: tan(fov_y / 2)
  cam_up: vec4f,      // scaled by tan(fov_y / 2); w: viewport height
  cam_fwd: vec4f,     // w: viewport width
  lamp_rect: vec4f,
  ground_rect: vec4f,
  ground_tiles: vec4f, // the height map in tiles (GROUND_LOOK): the grid post at the tiles' corner; the grid's posts across, down
  ground_back: vec4f,  // the backdrop's heights where no tile is: their first post x, z; their step; posts a side
  car_pos: vec4f,     // w: heading
  car_dir: vec4f,     // w: speed
  car_up: vec4f,       // w: the aurora the page's desk asks for (aurora_state)
  misc: vec4f,
  sun: vec4f,         // the way to the sun (wasm/include/harstad/clock.h); w: the twilight's light on the ground, of the full moon's
  moon_now: vec4f,    // the moon's light now, of MOON_FULL's, per channel; w: its phase angle, rad
  sky_now: vec4f,     // the night sky's light, of SKY_NIGHT's; the exposure, of the night's; the share of the town's homes lit; the hour
  town_now: vec4f,    // what the town's lights go by (wasm/include/harstad/lights.h): the night's hour, a working day that afternoon and the morning after, the shops lit
  cam_vel: vec4f,     // the camera's own motion, m/s; w: the ground's height under it
  tail: vec4f,        // the tail lamps' light on the road: colour times strength
  shadow_vp: mat4x4f, // world to the moon's shadow map
  fog: vec4f,         // the haze: its colour as the screen shows it, density (thicker and paler inside a cloud)
  cloud: vec4f,       // w: how deep the eye is in a cloud
  wash_at: vec4f,     // the TV tower's coloured light: where from; w: its reach
  wash: vec4f,        // its colour times its strength
  tunnel_at: vec4f,   // the tunnel luminaires' light on the car: where; w: its reach, 0 in the open
  tunnel: vec4f,      // its colour times its strength
  beam_vp: array<mat4x4f, 2>, // world to each headlamp's shadow map
  shadow_eye: vec4f,  // where the moon's shadow map is centred
  jet: vec4f,         // the jets' light: its colour, 0 when none burns; w: how hard they stir the air
  jet_at: array<vec4f, 5>, // each wing's flames as a lamp, and the boost turbine's: where they shine from; w: their strength (wasm/src/rocket.c)
  jet_aim: array<vec4f, 5>, // the way each shines, out of its nozzle; w: how far that way only, 0 all round
  jul_at: array<vec4f, 2>, // the Julehus's coloured lights (wasm/src/julehus.c): where from; w: reach, 0 for none
  jul: array<vec4f, 2>,    // their colours times their strengths
  truck_at: array<vec4f, 4>, // the Christmas truck's glow (wasm/src/traffic.c): ahead of it, each side of its trailer, behind it; w: reach, 0 for none
  truck: array<vec4f, 4>,    // their colours times their strengths
  heli_at: vec4f,  // the air ambulance's searchlight (wasm/src/heli.c): where from; w: reach, 0 when off
  heli_aim: vec4f, // the way it shines; w: the cosine of its cone's edge
  heli: vec4f,     // its colour times its strength; w: the cosine of its core
  plough_at: array<vec4f, 3>,   // the winter road crews (wasm/src/plough.c): each one's beacons on its cab, where; w: their reach, 0 for none
  plough_back: array<vec4f, 3>, // the beacon on its hopper; w: its work lights' strength
  plough_turn: array<vec4f, 3>, // the way each of its three beacons points, and its heading (rad, 0 along +z, rising to the left)
  plough: array<vec4f, 3>,      // the beacons' colour times their strength; w: 1 where it has a blade and wing
  grit: vec4f,                  // the gritted track's ring: its newest section, how many it holds; z: the night's snowfall, m; w: the word the roads' snow cells start at
  snowing: vec4f,               // the snow falling now (wasm/include/harstad/winter.h): flakes a cubic metre, the veil they make (per m), the snowfall (m/s), the height it falls from (m)
  showers: array<vec4f, 8>,      // the snow showers over the fjord (wasm/src/shower.c): x, z, radius, strength; cloud base, seed, wind x, z
  lit_at: vec4f,                 // the grid over the map of where the moving lights reach (game.h, GAME_LIT_GRID): x, z of its corner, cells a metre
  lit_grid: array<vec4u, ${GAME_LIT_GRID * GAME_LIT_GRID / 128}>, // GAME_LIT_GRID cells a side, a bit each, a row's columns in GAME_LIT_GRID / 32 words
  shine: vec4u,                  // the light list (wasm/include/harstad/shine.h): its cars, all its lights
}
@group(0) @binding(0) var<uniform> F: Frame;
// Every light that moves (wasm/src/shine.c), the cars first: where it shines from, w how far any of it reaches; a round
// lamp's colour times its strength, w its kind (0 a car, 1 round); a car's way x z, tail lamps' strength, out of the
// centre (2 more on full beam), a round lamp's aim x; where its sectors were worked out from, w the first (u32 bits);
// the way their middle pointed then
struct Light { at: vec4f, col: vec4f, dir: vec4f, seen: vec4f, seen_m: vec4f, }
@group(0) @binding(23) var<storage, read> shine_lights: array<Light>;
// ${SHINE.buckets} + 1 offsets, the sectors (two a word), the index list; and from LAMP_SW0 the low lights' switches, whose
// light the green is (sw_on, struct lamp_switches): one buffer, so that a shader of the frame's reads three storage buffers
@group(0) @binding(24) var<storage, read> shine_grid: array<u32>;
// The lamp map's tiles (wasm/src/lamp.c, tiles.c; its page table TABLES.lamp) from layer 0, and from tab_first(TABLES.glowFirst)
// its first mip and the next, in tiles (LAMP_GLOW, glow_look): the town's light held in the haze
@group(0) @binding(1) var lamp_map: texture_2d_array<f32>;
@group(0) @binding(2) var ground_map: texture_2d_array<f32>; // the terrain's heights, a layer a tile (GROUND_LOOK), and in TABLES which layer each tile is and the backdrop's heights where none is
@group(0) @binding(3) var detail_map: texture_2d_array<f32>; // the fine noise (layer 0) and the snow's relief (1), both repeating
@group(0) @binding(4) var samp_clamp: sampler;
@group(0) @binding(5) var samp_repeat: sampler;
@group(0) @binding(6) var shadow_map: texture_depth_2d;
@group(0) @binding(7) var samp_shadow: sampler_comparison;
@group(0) @binding(8) var beam_map: texture_depth_2d_array;
// The two-channel maps in tiles, one after another (src/worker.js, joinMaps): the low light's colour (wasm/src/lamp.c,
// lamp_tint_tiles; its page table TABLES.tint) from layer 0; how far off what casts the moon's shadow is, sqrt(m /
// MOON_REACH_FAR), and the tops (r, g; wasm/src/moon.c; TABLES.reach) from tab_first(TABLES.reachFirst); how much of the
// sky the ground sees, in r (wasm/src/ao.c; TABLES.map's b and a) from tab_first(TABLES.skyFirst)
@group(0) @binding(10) var pair_map: texture_2d_array<f32>;
@group(0) @binding(11) var<storage, read> sky_rows: SkyRows;
// Every page table of the maps above (their slots in tiles, or their one value: tiles.h), each of its levels, and the
// ground's tiles and backdrop, in one texture (TABLES; src/worker.js, tableAtlas): its first row says where each is
@group(0) @binding(16) var tables: texture_2d<u32>;
${TABLES_WGSL}
// The moon's light through the clouds over the ground (CLOUD_SHADE: layers 0 and 1), and from layer MOON_FIRST the lowest
// height the moon is seen at, over ground_at (wasm/src/moon.c), in tiles (its page table TABLES.map's r and g)
@group(0) @binding(25) var moon_map: texture_2d_array<f32>;
@group(0) @binding(17) var glow_map: texture_2d<f32>; // the town's light given back round it (wasm/src/skyglow.c): rgb from above, a from below

const PI = 3.141592653589793;
const MOON_DIR = ${vec3(MOON)};
// The night's light as it was tuned, under a full moon 38 degrees up: what the town's clock scales (wasm/include/harstad/clock.h,
// struct clock_light; F.moon_now, F.sky_now, F.sun)
const MOON_FULL = ${scaled(0xb4c2ee, 0.22 * NIGHT)};
const SKY_NIGHT = ${scaled(0xa3a9c8, 0.21 * NIGHT * WILD)};   // soft and only faintly blue, so lamplight reads warm
const GROUND_NIGHT = ${scaled(0x8f8f9f, 0.21 * NIGHT * WILD)};
// The twilight's light on the ground at as much light as that full moon and its sky give it (F.sun.w is how many times
// that): the clear sky's after sunset, blue
const DUSK_LIGHT = ${vec3(DUSK.colour.map(v => v * DUSK_REF))};
const HAZE_NIGHT = ${scaled(0x12151c, MOONLIT)};     // in clear air; F.fog is the haze where the eye is
const HAZE_DENSITY = 0.00016;                   // thin haze only: the far shore and mountains stay in view
const CLOUD_TINT = ${scaled(0x343945, MOONLIT)};
const EXPOSURE = 1.25; // the night's: F.sky_now.y scales it as the eye adapts to the light it is in
const DISPLAY = 0.5;   // the scene's light at that exposure that is the screen's: lamp-lit snow readable, not blown out
const KNEE = 0.55;     // the screen's luminance up to which it shows the light as it is (tone())
// On an HDR screen (src/worker.js, hdr) the shoulder is the same up to HDR_T1 of white, and over it a second shoulder,
// as steep where it starts, rises towards the screen's peak (hdr_peak) instead of white: HDR_L1 the luminance where the
// first reaches HDR_T1, HDR_S1 its slope there (gfx_tone_hdr in the C)
const HDR_T1 = ${HDR.shoulder.toFixed(2)};
const HDR_L1 = ${(0.55 - 0.45 * Math.log(1 - (HDR.shoulder - 0.55) / 0.45)).toFixed(6)};
const HDR_S1 = ${(1 - (HDR.shoulder - 0.55) / 0.45).toFixed(6)};
// The light now, by the town's clock: the moon's, the sky's from above and from below (the snow gives back the twilight's)
fn moon_now() -> vec3f { return MOON_FULL * F.moon_now.rgb; }
fn sky_now() -> vec3f { return SKY_NIGHT * F.sky_now.x + DUSK_LIGHT * F.sun.w; }
fn ground_now() -> vec3f { return GROUND_NIGHT * F.sky_now.x + DUSK_LIGHT * (F.sun.w * ${DUSK.ground.toFixed(2)}); }
// the clear air's own colour: the night's, and the twilight's in it
fn haze_now() -> vec3f { return HAZE_NIGHT * F.sky_now.x + DUSK_LIGHT * (F.sun.w * ${DUSK.haze.toFixed(3)}); }

// colours arrive as sqrt(c / scale) in rgb and scale * 16 in a (gfx_rgb in the C)
fn unpack(c: vec4f) -> vec3f { return c.rgb * c.rgb * (c.a * 15.9375); }

// The screen's linear light: the scene's at the exposure (F.sky_now.y as the eye adapts to the light it is in), as it
// is up to a luminance of KNEE; over it a soft shoulder on the luminance alone, KNEE + (1 - KNEE)(1 - exp(-(l - KNEE) /
// (1 - KNEE))), smooth at the knee and coming up to white, the colour scaled with it so its hue and saturation are kept;
// and a colour whose brightest channel is still over white runs into white. Nothing else: no toe, no colour matrix, no
// tint, so black is black, the dark is as dark as it is and a shadow keeps its own colour. gfx_eye, gfx_tone in the C.
fn screen_gain() -> f32 { return EXPOSURE * F.sky_now.y / 0.6 * DISPLAY; }
// The screen's white, of SDR white, that the light may come up to: 1 on an SDR screen; on an HDR one HDR.peak. A
// constant of the shaders as they are made (HDR_ON), not a uniform: tone() is in every shader, and with both shoulders
// in it every one of them was slower (street 45 -> 54 ms, aerial 83 -> 105)
const HDR_PEAK = ${HDR_ON ? HDR.peak.toFixed(1) : '1.0'};
fn hdr_peak() -> f32 { return HDR_PEAK; }
// the shoulder on a luminance l over KNEE, to white or on an HDR screen to its peak
fn tone_l(l: f32, peak: f32) -> f32 {
${HDR_ON ? '' : '  if (true) { return KNEE + (1.0 - KNEE) * (1.0 - exp((KNEE - l) / (1.0 - KNEE))); } // (SDR: the one shoulder)'}
  if (peak <= 1.0 || l <= HDR_L1) { return KNEE + (1.0 - KNEE) * (1.0 - exp((KNEE - l) / (1.0 - KNEE))); }
  return HDR_T1 + (peak - HDR_T1) * (1.0 - exp(HDR_S1 * (HDR_L1 - l) / (peak - HDR_T1)));
}
// and back: the luminance the shoulder shows as t
fn tone_l_inv(t: f32, peak: f32) -> f32 {
  if (t <= KNEE) { return t; }
  if (peak <= 1.0 || t <= HDR_T1) { return KNEE - (1.0 - KNEE) * log(max(1.0 - (t - KNEE) / (1.0 - KNEE), 1e-4)); }
  return HDR_L1 - (peak - HDR_T1) / HDR_S1 * log(max(1.0 - (t - HDR_T1) / (peak - HDR_T1), 1e-4));
}
fn tone(c: vec3f) -> vec3f {
  let e = max(c, vec3f(0.0)) * screen_gain();
  let l = dot(e, vec3f(0.2126, 0.7152, 0.0722));
  let peak = hdr_peak();
  let o = select(e, e * (tone_l(l, peak) / max(l, 1e-6)), l > KNEE);
  let m = max(max(o.r, o.g), o.b);
  if (m <= peak) { return o; }
  return mix(o / m, vec3f(1.0), saturate((m - peak) * 0.5 / peak)) * peak;
}
fn encode(c: vec3f) -> vec3f {
  return select(pow(c, vec3f(0.41666)) * 1.055 - 0.055, c * 12.92, c <= vec3f(0.0031308));
}
// A light seen directly (a lamp's lens): the exposure and the clip at white (on an HDR screen its peak) on the brightest
// channel alone, so a light that clips keeps its hue (sodium orange, a tail lamp's red) and does not run into white
fn tone_hue(c: vec3f) -> vec3f {
  let m = max(max(c.r, c.g), c.b);
  return select(vec3f(0.0), c / m * tone(vec3f(m)).g, m > 0.0);
}
// What the air lets through of a light at the point at, to the eye: the haze's density falls off with height (HAZE_TOP), so a
// light high in the sky is seen through less of it than one as far off along the ground (the mean of its density along
// the way), and the clear air's share (FAR_AIR), any snow shower and the falling snow's veil. One formula for the glow
// sprite (GLOW: a lens far off) and the lens a mesh draws near by (lens_burn): so a lens does not change as it takes over.
const HAZE_TOP = 1500.0; // m: the height over which the haze thins by e, the winter's low boundary layer
fn lens_thru(at: vec3f) -> f32 {
  let to = at - F.cam_pos.xyz; let dist = length(to);
  let y0 = max(F.cam_pos.y, 0.0); let y1 = max(at.y, 0.0); let e0 = exp(-y0 / HAZE_TOP); let e1 = exp(-y1 / HAZE_TOP);
  let dense = select(HAZE_TOP * (e0 - e1) / max(y1 - y0, 1e-3), 0.5 * (e0 + e1), abs(y1 - y0) < 1.0);
  return exp(-air_tau(F.cam_pos.xyz, to) - F.fog.a * dist * FAR_AIR * dense - shower_tau(F.cam_pos.xyz, to / max(dist, 1e-3), dist)
    - F.snowing.y * snow_through(F.cam_pos.xyz, to / max(dist, 1e-3), dist));
}
// A lens a mesh draws (GFX_LENS_LAMP), drawn as the glow sprite draws it far off: its colour's hue at the radiance it truly
// has, cd cd/m2 (gfx.h, GFX_CD_LAMP), through the air, clipped by its brightest channel alone (tone_hue) so that it keeps
// its hue and, on an HDR screen, rises over white by its light. share: how much of that this one burns (a lamp failing).
// Returned as the screen takes it, to lay over what the mesh shows as the brighter, as the sprite is.
fn lens_burn(hue: vec3f, at: vec3f, cd: f32, share: f32) -> vec3f {
  let l = max(dot(hue, vec3f(0.2126, 0.7152, 0.0722)), 1e-5);
  return encode(tone_hue(hue / l * (cd / ${SCENE_CD.toFixed(1)} * share * lens_thru(at))));
}
// The town's own light given back round xz, as irradiance (TOWN): rgb from above, a the brightest channel of the light
// from below, whose colour is rgb's
fn town_light(xz: vec2f) -> vec4f { return town_up(xz) * vec4f(vec3f(${TOWN.back.toFixed(4)}), 1.0); }
// coarse texel c's shares that burn by a switch, from above and from below (skyglow.h): two to a texel under the map
fn town_share(c: vec2u, cw: u32, w: u32, h: u32) -> vec2f {
  let n = c.y * cw + c.x; let t = n / 2u;
  let p = textureLoad(glow_map, vec2u(t % w, h + t / w), 0);
  return select(p.xy, p.zw, (n & 1u) != 0u);
}
// the map as it is: rgb the light leaving the town's ground, a the light from below
fn town_up(xz: vec2f) -> vec4f {
  // the map, w by h (as skyglow.c lays it out from the lamp map's size); under it the shares that burn by a switch
  let all = textureDimensions(glow_map);
  let dims = ceil(round(F.lamp_rect.zw / ${LAMP_PX}) / ${SKYGLOW_PX}.0) + ${2 * TOWN.margin / TOWN.cell}.0;
  let lo = F.lamp_rect.xy - ${TOWN.margin.toFixed(1)}; let size = dims * ${TOWN.cell.toFixed(1)};
  let uv = clamp((xz - lo) / size, vec2f(0.0), vec2f(1.0));
  let v = clamp(uv.y * dims.y, 0.5, dims.y - 0.5) / f32(all.y); // (never across into the rows under it)
  let g = textureSampleLevel(glow_map, samp_clamp, vec2f(uv.x, v), 0.0);
  // the shares, on the coarse grid between its texels' middles, as the light from above was read back
  let cn = vec2u(ceil(dims / ${SKYGLOW_COARSE}.0));
  let c = clamp(uv * dims / ${SKYGLOW_COARSE}.0 - 0.5, vec2f(0.0), vec2f(cn - 1u)); let c0 = vec2u(c); let c1 = min(c0 + 1u, cn - 1u); let cf = c - floor(c);
  let k = mix(mix(town_share(c0, cn.x, all.x, u32(dims.y)), town_share(vec2u(c1.x, c0.y), cn.x, all.x, u32(dims.y)), cf.x),
              mix(town_share(vec2u(c0.x, c1.y), cn.x, all.x, u32(dims.y)), town_share(c1, cn.x, all.x, u32(dims.y)), cf.x), cf.y);
  let dark = 1.0 - F.sky_now.z; // the windows' share of it dims as the town's homes go dark (wasm/src/skyglow.c)
  let off = max(max(lo - xz, xz - lo - size), vec2f(0.0));
  return vec4f(g.rgb * (1.0 - k.x * dark), g.a * (1.0 - k.y * dark)) * exp(-length(off) / ${TOWN.fade.toFixed(1)});
}
// The sky's light on a surface facing ny up, from the town's light g there: the night's own, and the town's from above;
// from below, what the snow gives back of that, and \`below\` of the town's own from below
fn sky_from(g: vec4f, ny: f32, below: f32) -> vec3f {
  let hue = g.rgb / max(max(g.r, max(g.g, g.b)), 1e-6);
  return mix(ground_now() + g.rgb * ${TOWN.ground.toFixed(2)} + hue * below, sky_now() + g.rgb, 0.5 * ny + 0.5);
}
// what of the light from below is let in at a place whose lamps' reach (lamp_reach) is \`reach\`: past the half of the
// pools a surface facing down takes already (lamp_tilt), 1.6 pi times the reach as irradiance
fn town_below(g: vec4f, reach: vec3f) -> f32 { return max(g.a * ${TOWN.below.toFixed(2)} - ${(0.8 * Math.PI).toFixed(4)} * max(reach.r, max(reach.g, reach.b)), 0.0); }
fn sky_light_over(xz: vec2f, ny: f32, reach: vec3f) -> vec3f { let g = town_light(xz); return sky_from(g, ny, town_below(g, reach)); }
// the same where the reach is not at hand: the pools' on the ground there
fn sky_light(xz: vec2f, ny: f32) -> vec3f { let m = lamps_at(xz); return sky_light_over(xz, ny, street_light(m) + vec3f(m.g * 1.25)); }
// the sky's light from straight above
fn sky_up(xz: vec2f) -> vec3f { return sky_now() + town_light(xz).rgb; }
// The town's light in the air over that part of it: the haze lit from below by the light leaving its ground (AIR.glow)
fn town_glow(xz: vec2f) -> vec3f { return town_up(xz).rgb * ${AIR.glow.toFixed(4)}; }
// The town's light in the low air seen along the horizontal way \`flat\` from xz, a dome over the horizon where the town
// lies: the air at three distances out, where the haze along a low ray mostly is
fn town_dome(xz: vec2f, flat: vec2f) -> vec3f {
  return (town_glow(xz + flat * ${AIR.dome[0].toFixed(1)}) + town_glow(xz + flat * ${AIR.dome[1].toFixed(1)}) + town_glow(xz + flat * ${AIR.dome[2].toFixed(1)})) * ${(1 / 3).toFixed(5)};
}
// The falling snow (winter.h, winter_veil): F.snowing.y of extinction a metre, from the ground up to the cloud it falls
// from, F.snowing.w m up. Its flakes far off are too small to see one by one, so it is seen as a veil, lit as the
// flakes near by are (SNOW): by the moon, the sky and the town's light in the air. Everything seen through it is
// veiled by it, the sky and the far lights too, as thick as the snow falls: a heavy spell closes the view in.
// m of the way from e along d, len long, that lies in the falling snow
fn snow_through(e: vec3f, d: vec3f, len: f32) -> f32 {
  let top = F.snowing.w;
  if (e.y < top) { return select(len, min(len, (top - e.y) / d.y), d.y > 1e-5); }
  if (d.y > -1e-5) { return 0.0; }
  return max(len - (e.y - top) / -d.y, 0.0);
}
// a flake's light, from all round: the moon's, the sky's from above and the town's in the air (as a shower's curtain)
fn snow_lit(xz: vec2f) -> vec3f { return moon_now() * 0.4 + sky_up(xz) * 0.28 + town_glow(xz) * 0.6; }
fn snow_veil(c: vec3f, e: vec3f, d: vec3f, len: f32) -> vec3f {
${OFF.has('veil') ? '  if (true) { return c; }' : ''}
  if (F.snowing.y <= 0.0) { return c; }
  // in two parts: the last few metres before what is seen, down among its street lamps and lit by them as the flakes
  // there are (lamp_reach: a lit street glows through the snow over it, a dark field does not); and the rest, lit where
  // most of it lies along the way
  let through = snow_through(e, d, len); let tail = select(0.0, min(through, 10.0), e.y + d.y * len < F.snowing.w); // (a ray out of the snow ends above it)
  let mid = e + d * (len - tail * 0.5);
  let near = mix(c, snow_lit(mid.xz) + lamp_reach(mid) * 0.6, 1.0 - exp(-F.snowing.y * tail));
  return mix(near, snow_lit(e.xz + d.xz * min(len, 1.0 / F.snowing.y) * 0.5), 1.0 - exp(-F.snowing.y * (through - tail)));
}
// the veil between the eye and w as a colour and its share, for what blends itself over the scene (snow_veil's mix)
fn snow_veil_at(w: vec3f) -> vec4f {
  if (F.snowing.y <= 0.0) { return vec4f(0.0); }
  let to = w - F.cam_pos.xyz; let len = length(to); let d = to / max(len, 1e-4);
  let v = 1.0 - exp(-F.snowing.y * snow_through(F.cam_pos.xyz, d, len));
  return vec4f(snow_veil(vec3f(0.0), F.cam_pos.xyz, d, len) / max(v, 1e-6), v); // (what it adds over black, as a colour and its share)
}
// The clear air's optical depth from e along \`to\`: AIR.sigma a metre at sea level, thinning by e every AIR.h m up (the
// mean of its density along the way, as a share of e's), F.misc.w the player's setting (1 as tuned)
fn air_tau(e: vec3f, to: vec3f) -> f32 {
  let k = to.y / ${AIR.h.toFixed(1)};
  let along = select((1.0 - exp(-k)) / k, 1.0, abs(k) < 1e-3);
  return ${AIR.sigma.toExponential(3)} * F.misc.w * length(to) * exp(-max(e.y, 0.0) / ${AIR.h.toFixed(1)}) * along;
}
// What the clear air at \`at\` scatters towards an eye looking along d, the radiance of air deep enough to hide what is
// behind it, lit as the snow's veil is (snow_lit) by the same light on the one scale: the moon's by the haze's phase
// (forward, AIR.g), the sky's from above and the snow's from below each scattered evenly (a hemisphere of radiance E /
// pi, half of it each way), of AIR.albedo; and the town's light in the air (town_glow). Under the full moon some 0.012
// cd/m2, as the sky at the horizon is drawn; at noon the twilight's.
fn air_lit(at: vec3f, d: vec3f) -> vec3f {
  let up = town_up(at.xz).rgb; // sky_up's town_light and town_glow, from the one look into the map
  return (moon_now() * hg_phase(dot(d, MOON_DIR), ${AIR.g.toFixed(2)}) + (sky_now() + up * ${TOWN.back.toFixed(4)} + ground_now()) * ${(1 / (2 * Math.PI)).toFixed(6)}) * ${AIR.albedo.toFixed(2)}
    + up * ${AIR.glow.toFixed(4)};
}
// Aerial perspective, in linear light before the eye: the clear air between the eye and w takes out exp(-tau) of what
// is behind it (air_tau) and scatters in its own light (air_lit), as the falling snow's veil does after it.
fn aerial(c: vec3f, w: vec3f) -> vec3f {
${OFF.has('aerial') ? '  if (true) { return c; }' : ''}
  let to = w - F.cam_pos.xyz; let dist = length(to); let d = to / max(dist, 1e-4);
  let tau = air_tau(F.cam_pos.xyz, to);
  if (tau < 0.002) { return snow_veil(c, F.cam_pos.xyz, d, dist); }
  return snow_veil(mix(c, air_lit(F.cam_pos.xyz + to * 0.5, d), 1.0 - exp(-tau)), F.cam_pos.xyz, d, dist);
}
// the air (aerial(): the clear air's haze in linear light), tone map, encode, and only then a cloud's own haze where the
// eye is in one (F.fog: nothing in clear air, clouds.c), with the car's lamps in it
fn finish(linear: vec3f, w: vec3f) -> vec3f {
  let to = w - F.cam_pos.xyz; let dist = length(to);
  let depth = dot(to, F.cam_fwd.xyz);
  let fog = 1.0 - exp(-F.fog.a * F.fog.a * depth * depth);
  return lamps_over(mix(encode(tone(aerial(linear, w))), F.fog.rgb, fog), cloud_lamps(to / max(dist, 1e-4), dist));
}
fn hg_phase(c: f32, g: f32) -> f32 { let x = 1.0 + g * g - 2.0 * g * c; return (1.0 - g * g) / (${(4 * Math.PI).toFixed(6)} * x * sqrt(x)); }
// a cloud's droplets: most of the light on forward, some back (k narrows both for the octaves that have spread out)
fn cloud_phase(c: f32, k: f32) -> f32 { return mix(hg_phase(c, -0.25 * k), hg_phase(c, 0.8 * k), 0.8); }
fn decode(c: vec3f) -> vec3f { // encode's inverse
  return select(pow((c + 0.055) / 1.055, vec3f(2.4)), c / 12.92, c <= vec3f(0.04045));
}
// The player's car's lamps as the air round them takes them: the light that falls at p from its beams, and from its
// tail lamps and jets, each through the cloud droplets' phase towards the eye, which looks along d. (car_beams_at and
// car_tail_at with the surface turned to face each: the air has no side turned away.)
// m from the car its furthest lamp reaches (car_beams): the dipped beams', or on full beam (F.tail.w) the main beams'
fn car_reach() -> f32 { return select(${(DIP_REACH + 4).toFixed(1)}, ${(FULL_REACH + 4).toFixed(1)}, F.tail.w > 0.0); }
fn car_air(p: vec3f, d: vec3f) -> vec3f {
  let up = F.car_up.xyz; let fwd = F.car_dir.xyz;
  let to_f = F.car_pos.xyz + up * 0.73 + fwd * 2.3 - p; let to_b = F.car_pos.xyz + up * 0.85 - fwd * 2.3 - p;
  let lf = to_f / max(length(to_f), 1e-3); let lb = to_b / max(length(to_b), 1e-3);
  return car_beams_at(p, lf) * cloud_phase(dot(d, lf), 1.0) + car_tail_at(p, lb) * cloud_phase(dot(d, lb), 1.0);
}
// The car's lamps in the haze of a cloud the eye is in (F.cloud.w): what they scatter towards the eye along d, out to
// \`far\` m, as linear light. What lies at t is seen through the haze's exp(-(a t)^2) (finish), so the share of the light
// at t scattered towards the eye is that curve's slope there, 2 a^2 t exp(-(a t)^2): the haze that hides what is behind
// it is the haze lit. Only the stretch of the ray within car_reach() of the car is looked at, in 16 steps.
fn cloud_lamps(d: vec3f, far: f32) -> vec3f {
${OFF.has('cloudlamps') ? '  if (true) { return vec3f(0.0); }' : ''}
  if (F.cloud.w < 0.002) { return vec3f(0.0); }
  let rel = F.car_pos.xyz - F.cam_pos.xyz; let tc = dot(rel, d); let off = dot(rel, rel) - tc * tc;
  let ra = car_reach();
  if (off >= ra * ra) { return vec3f(0.0); }
  let h = sqrt(ra * ra - off);
  let t0 = max(tc - h, 0.0); let t1 = min(tc + h, far);
  if (t1 <= t0) { return vec3f(0.0); }
  let a2 = F.fog.a * F.fog.a; let dt = (t1 - t0) / 16.0;
  var sum = vec3f(0.0);
  for (var i = 0; i < 16; i++) {
    let t = t0 + (f32(i) + 0.5) * dt;
    sum += car_air(F.cam_pos.xyz + d * t, d) * (2.0 * a2 * t * exp(-a2 * t * t) * dt);
  }
  return sum;
}
// Light added to a colour already encoded for the screen, as it adds: at the exposure, and over the knee through the
// shoulder, from the light tone() shows as that colour (the shoulder's inverse)
fn lamps_over(c: vec3f, lamps: vec3f) -> vec3f {
  if (all(lamps <= vec3f(0.0))) { return c; }
  let base = decode(c);
  let l = dot(base, vec3f(0.2126, 0.7152, 0.0722));
  let up = select(1.0, tone_l_inv(l, hdr_peak()) / max(l, 1e-6), l > KNEE);
  let at = base * up / screen_gain();
  return encode(base + max(tone(at + lamps) - tone(at), vec3f(0.0)));
}
// small whole powers as products: pow() is exp2(log2) on the GPU, two transcendentals where two or three multiplies do
fn pow3(x: f32) -> f32 { return x * x * x; }
fn pow4(x: f32) -> f32 { let y = x * x; return y * y; }
fn pow5(x: f32) -> f32 { let y = x * x; return y * y * x; }
fn pow8(x: f32) -> f32 { let y = x * x; let z = y * y; return z * z; }
fn display(linear: vec3f) -> vec3f { return encode(tone(linear)); }

// The snow showers over the fjord (wasm/src/shower.c): how much snow a ray from o along the unit d crosses in its first
// len m. Each is a column, densest at its axis and thinning as a gaussian of its radius outwards, from the sea up to
// near its cloud's base: along a straight ray that is a gaussian in how near the ray passes and the difference of two
// error functions in where along it that is, taken at the height where it passes nearest.
const FAR_AIR = 0.25; // the clear air a far light (and a shower) is seen through, of the haze's density (GLOW)
const SHOWER_THICK = ${(1.1e-3).toExponential(2)}; // per m at a shower's axis at full strength: a light shower, some three kilometres' sight in it
fn erf_near(x_in: f32) -> f32 { let x = clamp(x_in, -3.0, 3.0); return tanh(x * (1.1283792 + 0.1009102 * x * x)); } // tanh of a large number is NaN on some GPUs
fn shower_tau(o: vec3f, d: vec3f, len: f32) -> f32 {
  let h = length(d.xz);
  if (h < 1e-4) { return 0.0; }
  let u = d.xz / h; var tau = 0.0;
  for (var k = 0; k < ${4}; k++) {
    let a = F.showers[k * 2]; let b = F.showers[k * 2 + 1];
    if (a.w <= 0.0) { continue; }
    let c = a.xy - o.xz; let s0 = dot(c, u); let miss = c.x * u.y - c.y * u.x;
    let e = miss * miss / (a.z * a.z);
    if (e > 9.0) { continue; }
    let along = 0.5 * (erf_near((len * h - s0) / a.z) + erf_near(s0 / a.z)); // of the whole column the ray crosses, 0..1
    let y = o.y + d.y * max(s0, 0.0) / h; // its height where it passes nearest the axis
    let high = (1.0 - smoothstep(b.x * 0.4, b.x, y)) * (1.0 - 0.45 * y / b.x); // thicker low down, thinning into its cloud
    tau += SHOWER_THICK * a.w * exp(-e) * a.z * 1.7724539 * along * high / h;
  }
  return tau;
}


// The street lamps' light in the lamp map's red (sodium) and blue (cool white), as it lands: STREET of the strength they
// were painted at
fn street_light(m: vec3f) -> vec3f { return (vec3f(1.0, 0.58, 0.24) * m.r + vec3f(0.80, 0.92, 1.0) * m.b) * ${STREET.toFixed(3)}; }
fn lamps_at(xz: vec2f) -> vec3f { return lamps_floor_at(xz, 1.0).rgb; }
fn room_hash(x: u32) -> u32 { var h = x * 747796405u + 2891336453u; h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u; return (h >> 22u) ^ h; }
fn room_rand(id: u32, k: u32) -> f32 { return f32(room_hash(id + k * 0x9e3779b9u) >> 8u) * (1.0 / 16777216.0); }
// When a room is lit, by the town's clock (wasm/include/harstad/lights.h, the same from the same numbers in f32): a home's
// hours in the night's own (F.town_now.x, from 15:00), each room going dark at its own minute round the home's bedtime;
// an office's by the hour of the day; a window's Christmas string or Advent star on its timer. A home is the wall's (its
// quad's first corner across the ground), and in a block of flats each storey of it.
struct RoomHours { back: f32, bed: f32, wake: f32, leave: f32, back_day: f32 }
fn room_hours(home: u32) -> RoomHours {
  let we = F.town_now.y > 0.5; let wm = F.town_now.z > 0.5;
  let inside = room_rand(home, 40u) < 0.35;
  let early = room_rand(home, 49u) < 0.3;
  var o: RoomHours;
  o.back = select(0.0, select(room_rand(home, 41u) * 2.25 - 0.25, 0.0, early), we && !inside);
  o.bed = 7.0 + 2.0 * room_rand(home, 42u);
  if (room_rand(home, 43u) < 0.125) { o.bed += 2.5 * room_rand(home, 44u); }
  o.wake = 15.0 + 1.75 * room_rand(home, 45u) + select(1.5, 0.0, wm);
  o.leave = select(24.0, o.wake + 0.75 + 0.75 * room_rand(home, 46u), wm && !inside);
  o.back_day = select(24.0, 21.5 + 2.5 * room_rand(home, 41u), early);
  return o;
}
fn room_on(home: u32, id: u32) -> bool {
  let n = F.town_now.x; let o = room_hours(home);
  let bed = o.bed + room_rand(id, 47u) - 0.667;
  return (n >= o.back && n < bed) || (room_rand(id, 48u) < 0.6 && ((n >= o.wake && n < o.leave) || n >= o.back_day));
}
fn office_on(id: u32) -> bool {
  let u = room_rand(id, 50u); let hour = F.sky_now.w;
  if (u < 1.0 / 12.0) { return true; }
  if (F.town_now.y < 0.5 || (F.sky_now.w < 15.0 && F.town_now.z < 0.5)) { return false; } // (the morning's day is the one the hour is on)
  if (u < 1.0 / 12.0 + 0.125 && hour >= 18.0 && hour < 20.0) { return true; }
  return hour >= 7.5 && hour < 15.5 + 2.0 * room_rand(id, 51u);
}
fn string_on(id: u32) -> bool {
  let n = F.town_now.x;
  return room_rand(id, 53u) < 1.0 / 3.0 || n < 8.5 + 1.5 * room_rand(id, 52u) || n >= 15.0;
}
// ---- the low lights' switches (wasm/include/harstad/lamp.h, lights.h): what of the lamp map's green burns now ----
// lamp_sw: the table of switches and every level's tiles of whose light each texel's green is (struct lamp_switches).
// A texel's word: its switch's word in the table (low 20 bits), that switch's share of the green and the other
// switched lights' share, in 63rds; 0 where no switched light reaches. The green burning is share * the switch's own
// (sw_on) + other * the town's share of homes lit (F.sky_now.z) + the rest, which always burns.
fn sw_on(at: u32) -> f32 {
  let b = shine_grid[${LAMP_SW0}u + 1u] + at; let kind = shine_grid[${LAMP_SW0}u + b] & 255u;
  if (kind == 2u) { return select(0.0, 1.0, F.town_now.w > 0.5); } // a shop's display: lit while the shops are
  if (kind != 1u) { return 1.0; }
  return bitcast<f32>(shine_grid[${LAMP_SW0}u + b + 1u]); // the rooms' share burning, worked out this frame (SW_NOW)
}
// texel c of level l (LAMP_SW_FIRST on), clamped to the level's edges: its switch's word in the table, its share and
// the other share in 15ths (lamp.h): 0 where no switched light reaches
// The four texels of level l (LAMP_SW_FIRST on) at i, i + (1, 0), (0, 1), (1, 1), clamped to the level's edges: each its
// switch's word in the table, its share and the other share in 15ths (lamp.h); 0 where no switched light reaches. The
// level's header is read once, and a tile's page entry once for all of the four in it (most often all four are).
fn sw_quad(l: u32, i: vec2i) -> array<vec3u, 4> {
  let h = ${LAMP_SW_HEAD}u + min(l, shine_grid[${LAMP_SW0}u + 0u] - 1u) * 4u;
  let most = vec2i(i32(shine_grid[${LAMP_SW0}u + h]), i32(shine_grid[${LAMP_SW0}u + h + 1u])) - 1;
  let across = shine_grid[${LAMP_SW0}u + h + 2u]; let page = shine_grid[${LAMP_SW0}u + h + 3u];
  var o: array<vec3u, 4>;
  var t0 = vec2u(0xffffffffu); var e = 0u;
  for (var k = 0; k < 4; k++) {
    let at = vec2u(clamp(i + vec2i(k & 1, k >> 1), vec2i(0), most));
    let t = at / ${TILES_SIDE}u;
    if (any(t != t0)) { t0 = t; e = shine_grid[${LAMP_SW0}u + page + t.y * across + t.x]; }
    if (e == 0u) { o[k] = vec3u(0u); continue; }
    let r = at - t * ${TILES_SIDE}u; let n = r.y * ${TILES_SIDE}u + r.x;
    let v = (shine_grid[${LAMP_SW0}u + e + ${1 + LAMP_SW_PALETTE}u + n / 2u] >> ((n & 1u) * 16u)) & 0xffffu;
    let kk = v & 255u;
    o[k] = vec3u(select(0u, shine_grid[${LAMP_SW0}u + e + kk], kk != 0u), (v >> 8u) & 15u, v >> 12u);
  }
  return o;
}
// what of a texel's green burns now; own 0 leaves out all the switched light (the windows' and displays'), lit or not
fn sw_now(t: vec3u, own: f32) -> f32 {
  if (all(t.yz == vec2u(0u))) { return 1.0; }
  let sh = f32(t.y) * (1.0 / 15.0); let r = f32(t.z) * (1.0 / 15.0);
  return 1.0 - sh - r + (select(0.0, sh * sw_on(t.x), t.y != 0u) + r * F.sky_now.z) * own;
}
// The share of the green burning now at level l (LAMP_SW_FIRST on) round q (the first level's texels from the map's
// corner): the four texels round it, each by its own switch, weighed as a bilinear look weighs them, as smooth as the
// green itself is looked at (pyr_look)
fn sw_bilinear(l: u32, q: vec2f, own: f32) -> f32 {
  let s = q * exp2(-f32(l)) - 0.5; let i = vec2i(floor(s)); let a = s - floor(s);
  let quad = sw_quad(l, i); let t0 = quad[0]; let t1 = quad[1]; let t2 = quad[2]; let t3 = quad[3];
  if (all(vec4u(t0.y | t0.z, t1.y | t1.z, t2.y | t2.z, t3.y | t3.z) == vec4u(0u))) { return 1.0; }
  let n0 = sw_now(t0, own);
  let n1 = select(sw_now(t1, own), n0, all(t1 == t0));
  let n2 = select(select(sw_now(t2, own), n1, all(t2 == t1)), n0, all(t2 == t0));
  let n3 = select(select(select(sw_now(t3, own), n2, all(t3 == t2)), n1, all(t3 == t1)), n0, all(t3 == t0));
  return mix(mix(n0, n1, a.x), mix(n2, n3, a.x), a.y);
}
// The share of the green burning now at q, looked at with level of detail x as lamps_floor_at looks. Under
// LAMP_SW_FIRST the levels are not kept (lamp.h): the first kept, filtered; over it the two levels pyr_tri weighs, each
// filtered as it does.
fn sw_share(q: vec2f, x: f32, own: f32) -> f32 {
  let L = shine_grid[${LAMP_SW0}u + 0u];
  if (L <= ${LAMP_SW_FIRST}u) { return 1.0; }
  if (x < ${LAMP_SW_FIRST}.0) { return sw_bilinear(${LAMP_SW_FIRST}u, q, own); }
  let l = u32(x); let f = x - f32(l);
  if (f <= 0.0 || l + 1u >= L) { return sw_bilinear(min(l, L - 1u), q, own); }
  return mix(sw_bilinear(l, q, own), sw_bilinear(l + 1u, q, own), f);
}
// the same with the floor the light falls on (lamp_floor_above); own as sw_now
fn lamps_floor_at(xz: vec2f, own: f32) -> vec4f {
  // the map's outermost texels are never painted, so a clamped lookup reads black off the map
  let q = clamp((xz - F.lamp_rect.xy) / F.lamp_rect.zw, vec2f(0.0), vec2f(1.0)) * round(F.lamp_rect.zw / ${LAMP_PX});
  let x = pyr_lod(${TABLES.lamp}u, foot_xz(xz), ${LAMP_PX});
  var m = vec4f(0.0);
  if (x < 1.0) { m = lamp_smooth(q - 0.5); }
  if (x > 0.0) { m = pyr_tri(lamp_map, 0, ${TABLES.lamp}u, round(F.lamp_rect.zw / ${LAMP_PX}), q, x, 0u, 2u, m); }
  if (m.g > 0.0) { m.g *= sw_share(q, x, own); } // the windows and displays whose light it is, lit or not now
  return m;
}
// w's height over the floor its lamps light, given its height over the ground and the lamp map's alpha there: on a long
// bridge's deck (alpha 1 less its height in quarter metres over 255) its height over the deck; under it out of reach.
fn lamp_floor_above(w: vec3f, a: f32, above: f32) -> f32 {
  let deck = (1.0 - a) * 63.75;
  if (deck < 0.25) { return above; }
  return select(w.y - deck, 1e3, w.y < deck - 2.0);
}
${lampLook(`tab_dims(${TABLES.lamp}u)`, t => `tab_load(${TABLES.lamp}u, ${t})`)}
${GLOW_LOOK}
// Colour of the low light (the lamp map's green) at world xz, brightest channel 1: the tint (wasm/src/lamp.c, src/tint.js)
// holds its chromaticity, three metres to the texel over the lamp map's rectangle, half the lamp map's texels across
// and down, rounded up. It is kept in tiles (tiles.h): the dense tint's look at s = uv size - 0.5, clamped to its edge
// texels as the sampler clamped it, taken in the slot of floor(s)'s tile (as map_at), or the tile's one colour.
fn low_tint(xz: vec2f) -> vec3f {
  let size = ceil(round(F.lamp_rect.zw / ${LAMP_PX}) * 0.5);
  let s = clamp((xz - F.lamp_rect.xy + 1.5) / (size * 3.0) * size - 0.5, vec2f(-0.5), size - 0.5);
  let x = pyr_lod(${TABLES.tint}u, foot_xz(xz), ${(2 * LAMP_PX).toFixed(1)});
  var c = vec2f(0.0);
  if (x < 1.0) {
    let k = clamp(vec2i(floor((floor(s) + 0.5) * ${(1 / TILES_SIDE).toPrecision(9)})), vec2i(0), tab_dims(${TABLES.tint}u) - 1);
    let p = tab_load(${TABLES.tint}u, k);
    c = unpack4x8unorm(p.y).xy;
    if (p.x != ${TILES_CONSTANT}u) { let n = p.x | (p.y << 16u); c = textureSampleLevel(pair_map, samp_clamp, map_uv(n, s - vec2f(k * ${TILES_SIDE} - 1) + 0.5), i32(n / ${TILES_PER_LAYER}u), 0.0).rg; }
  }
  if (x > 0.0) { c = pyr_tri(pair_map, 0, ${TABLES.tint}u, size, (xz - F.lamp_rect.xy + 1.5) / (size * 3.0) * size, x, 0u, 1u, vec4f(c, 0.0, 0.0)).xy; }
  let t = vec3f(c, 1.0 - c.x - c.y);
  return t / max(max(t.r, t.g), t.b);
}
${GROUND_LOOK}
${GROUND_AT}
// Painted light reaching a surface at w with normal n. Street lamps hang about seven metres up and light what
// is below them; light from windows and porches stays low. Both come from above or beside, so ground takes
// more than walls.
// In two parts: what reaches the place, whichever way the surface there faces; and the share a surface facing n takes.
fn lamp_reach(w: vec3f) -> vec3f {
${OFF.has('lampmap') ? '  if (true) { return vec3f(0.0); }' : ''}
  let m = lamps_floor_at(w.xz, 1.0);
  if (all(m.rgb == vec3f(0.0))) { return vec3f(0.0); } // most of the map is dark: no height to look up, no tint
  return lamp_reach_lit(w, m.rgb, lamp_floor_above(w, m.a, w.y - ground_at(w.xz)));
}
// for a caller that has w's height over the ground (w.y - ground_at(w.xz)) already: the ground is looked up once a pixel
fn lamp_reach_over(w: vec3f, above: f32) -> vec3f {
${OFF.has('lampmap') ? '  if (true) { return vec3f(0.0); }' : ''}
  let m = lamps_floor_at(w.xz, 1.0);
  if (all(m.rgb == vec3f(0.0))) { return vec3f(0.0); }
  return lamp_reach_lit(w, m.rgb, lamp_floor_above(w, m.a, above));
}
// A facade's: without the light windows and shop displays lay on the floor before their walls. Each is laid from the
// foot of its own wall, brightest there, and on the wall it comes from it would be its window's light spread over the
// wall round it, a halo; but a window's light leaves it outwards, and none falls back on the wall it is set in.
fn lamp_reach_wall(w: vec3f, above: f32) -> vec3f {
${OFF.has('lampmap') ? '  if (true) { return vec3f(0.0); }' : ''}
  let m = lamps_floor_at(w.xz, 0.0);
  if (all(m.rgb == vec3f(0.0))) { return vec3f(0.0); }
  return lamp_reach_lit(w, m.rgb, lamp_floor_above(w, m.a, above));
}
fn lamp_reach_lit(w: vec3f, m: vec3f, above: f32) -> vec3f {
  var low = vec3f(0.0);
  if (m.g > 0.0 && above < 4.5) { low = low_tint(w.xz) * m.g * 1.25 * (1.0 - smoothstep(1.0, 4.5, above)); }
  var street = vec3f(0.0);
  if (m.r + m.b > 0.0) { street = street_light(m) * lamp_heads_up(w, above); }
  return street + low;
}
// The street lamps' light at w over what the lamp map lays on the floor under it, \`above\` m down: each head in w's bucket
// of their grid (wasm/src/shine.c, shine_heads) whose pool reaches there, weighed as its pool is on the floor, gives its
// own (an inverse square from a lens LAMP_LENS m across, and nothing over the horizontal: the lamps shine down) over its
// own on the floor. 1 on the floor itself. The floodlights and work lamps (lamp_pool_head) are listed the same way, their
// heads apart from their pools' middles. Light in the map from no listed head is cut over 3.5-6.5 m as all of it was.
fn lamp_heads_up(w: vec3f, above: f32) -> f32 {
  if (above <= 0.0) { return 1.0; }
  let b = ${SHINE.heads0}u + light_bucket(w.xz);
  let end = shine_grid[b + 1u];
  var weight = 0.0; var sum = 0.0;
  for (var i = shine_grid[b]; i < end; i++) {
    let L = shine_lights[shine_grid[${SHINE.hindex0}u + i]];
    let mid = L.dir.xy - w.xz; let p2 = dot(mid, mid); // its pool's middle: under a street lamp, out ahead of a floodlight
    if (p2 >= L.at.w * L.at.w) { continue; }
    let t = 1.0 - sqrt(p2) / L.at.w; let k = L.col.x * t * t; // its pool's falloff on the floor, roughly (lamp_cast_pool)
    let to = L.at.xz - w.xz; let d2 = dot(to, to); // and its head
    let h = L.at.y - w.y; let h0 = h + above; // the head over w and over the floor under it
    let r = d2 + h * h; let r0 = d2 + h0 * h0;
    let up = smoothstep(-0.05, 0.2, h * inverseSqrt(r + 1e-4)) / max(smoothstep(-0.05, 0.2, h0 * inverseSqrt(r0 + 1e-4)), 0.05);
    weight += k; sum += k * up * (r0 + ${(LAMP_LENS * LAMP_LENS).toFixed(3)}) / (r + ${(LAMP_LENS * LAMP_LENS).toFixed(3)});
  }
  if (weight > 0.0) { return sum / weight; }
  return 1.0 - smoothstep(3.5, 6.5, above);
}
fn lamp_tilt(n: vec3f) -> f32 { return 0.5 + 0.5 * clamp(n.y * 1.4 + 0.25, 0.0, 1.0); }
fn lamp_light(w: vec3f, n: vec3f) -> vec3f { return lamp_reach(w) * lamp_tilt(n); }
// How much of the moon's light the clouds let through to w, by their map (CLOUD_SHADE): over the plane at sea level
// each texel holds what the clouds along the way to the moon from it let through, and how high the lowest begins. A
// point over the cloud there is in the moon (over the edge of a cloud's floor within 60 m).
${CLOUD_SHADE_AT}
fn cloud_shade(w: vec3f) -> f32 {
${OFF.has('cloudshade') ? '  if (true) { return 1.0; }' : ''}
  let uv = (w.xz - MOON_DIR.xz * (w.y / MOON_DIR.y) - cloud_shade_origin()) * ${(1 / CLOUD_SHADE_SPAN).toExponential(6)};
  if (any(uv <= vec2f(0.0)) || any(uv >= vec2f(1.0))) { return 1.0; }
  let t = textureSampleLevel(moon_map, samp_clamp, uv, 0, 0.0).r;
  if (t >= 1.0) { return 1.0; }
  let base = textureSampleLevel(moon_map, samp_clamp, uv, 1, 0.0).r;
  return mix(t, 1.0, smoothstep(base, base + 60.0, w.y));
}
// The moon's light on a surface facing n: its shadow is looked up only where the surface faces it. floor: the share that
// still arrives in shadow.
fn moon_light(w: vec3f, n: vec3f, floor_in: f32) -> vec3f {
  let facing = max(dot(n, MOON_DIR), 0.0);
  if (facing <= 0.0) { return vec3f(0.0); }
  return moon_now() * (facing * mix(moon_shadow(w, n), 1.0, floor_in));
}
// A 4 x 4 tent of shadow comparisons in four filtered fetches (src/shadows.js): its weights along each axis, [1-f, 1, 1, f],
// are two bilinear pairs, (1-f, 1) and (1, f), each fetch placed so the hardware's own weights split its pair in that ratio.
// Returns the two fetch positions (xy the first, zw the second) and leaves the weights 2-f and 1+f to the caller.
fn tent_at(uv: vec2f, size: f32) -> vec4f {
  let s = uv * size - 0.5; let i = floor(s); let f = s - i;
  return vec4f(i - 0.5 + 1.0 / (2.0 - f), i + 1.5 + f / (1.0 + f)) / size;
}
fn tent_f(uv: vec2f, size: f32) -> vec2f { return fract(uv * size - 0.5); }
// The town-wide maps, the moon's field and the sky the ground sees, are kept in tiles (wasm/include/harstad/tiles.h): a
// look at uv is the dense map's look at s = uv size - 0.5 texels, taken in the slot of floor(s)'s tile, which holds that
// texel and the next either way. The dense map is four times the reach's size (MOON_REACH_CELL), and off it the look is
// clamped to its edge texels, as the sampler clamped it. \`at\` is where in the slot, in texels; \`page\` the tile's entry.
struct MapAt { page: vec4u, at: vec2f }
fn map_at(uv: vec2f) -> MapAt {
  let size = ceil(F.ground_rect.zw * 0.25) * 4.0;
  let s = clamp(uv * size - 0.5, vec2f(-0.5), size - 0.5);
  // the tile of floor(s), in floats: a whole number and a half over the side is never within a hundredth of a whole one
  let t = clamp(vec2i(floor((floor(s) + 0.5) * ${(1 / TILES_SIDE).toPrecision(9)})), vec2i(0), tab_dims(${TABLES.map}u) - 1);
  return MapAt(tab_load(${TABLES.map}u, t), s - vec2f(t * ${TILES_SIDE} - 1) + 0.5);
}
// The moon field's reach (r) and the tops (g, wasm/src/moon.c), MOON_REACH_CELL texels of the field to theirs, in tiles
// (moon_sweep_pairs) as the field is: the dense map's look at uv in the slot of floor(s)'s tile, or the tile's two bytes.
fn pair_look(atlas: texture_2d_array<f32>, first: i32, pages: u32, uv: vec2f) -> vec2f {
  let size = ceil(F.ground_rect.zw * 0.25);
  let s = clamp(uv * size - 0.5, vec2f(-0.5), size - 0.5);
  let t = clamp(vec2i(floor((floor(s) + 0.5) * ${(1 / TILES_SIDE).toPrecision(9)})), vec2i(0), tab_dims(pages) - 1);
  let p = tab_load(pages, t);
  if (p.x == ${TILES_CONSTANT}u) { return unpack4x8unorm(p.y).xy; }
  let n = p.x | (p.y << 16u);
  return textureSampleLevel(atlas, samp_clamp, map_uv(n, s - vec2f(t * ${TILES_SIDE} - 1) + 0.5), first + i32(n / ${TILES_PER_LAYER}u), 0.0).xy;
}
// slot k's texel coordinate \`at\` in its layer, normalised: a layer is a power of two across, so this is the same fraction
// of a texel as s (to the bit once s is 1,024 texels or more into the map; nearer its west and north edges, to 1/16,384)
fn map_uv(k: u32, at: vec2f) -> vec2f {
  return (vec2f(f32(k % ${TILES_ACROSS}u), f32(k / ${TILES_ACROSS}u % ${TILES_ACROSS}u)) * ${TILES_SLOT}.0 + at) * ${(1 / TILES_LAYER).toExponential()};
}
// k, c: a page entry's two channels, its slot's number's low half and its high, or TILES_CONSTANT and its value
fn map_read(atlas: texture_2d_array<f32>, first: i32, k: u32, c: u32, at: vec2f) -> f32 {
  if (k == ${TILES_CONSTANT}u) { return unpack2x16float(c).x; }
  let n = k | (c << 16u);
  return textureSampleLevel(atlas, samp_clamp, map_uv(n, at), first + i32(n / ${TILES_PER_LAYER}u), 0.0).r;
}
${PYR_PAGES}
// A bilinear look at level l (1 on) of a town-wide map size0 texels across and down at q, the place in those texels from
// the map's corner: at s = q 2^-l - 0.5 in that level's, clamped to its edges, in the slot of floor(s)'s tile (as
// map_at), or the tile's value
fn pyr_look(atlas: texture_2d_array<f32>, first: i32, pages: u32, size0: vec2f, q: vec2f, l: u32, chan: u32, kind: u32) -> vec4f {
  let size = pyr_size(size0, l);
  let s = clamp(q * exp2(-f32(l)) - 0.5, vec2f(-0.5), size - 0.5);
  let t = clamp(vec2i(floor((floor(s) + 0.5) * ${(1 / TILES_SIDE).toPrecision(9)})), vec2i(0), vec2i(ceil(size * ${(1 / TILES_SIDE).toPrecision(9)})) - 1);
  let p = tab_load(pages + l, t);
  let k = select(p.x, p.z, chan == 2u);
  if (k == ${TILES_CONSTANT}u) { return pyr_const(p, chan, kind); }
  let n = k | (select(p.y, p.w, chan == 2u) << 16u);
  return textureSampleLevel(atlas, samp_clamp, map_uv(n, s - vec2f(t * ${TILES_SIDE} - 1) + 0.5), first + i32(n / ${TILES_PER_LAYER}u), 0.0);
}
// The level a look at a footprint of foot m takes in a map of texels \`texel\` m across, as a sampler's level of detail
// (0 the first level, sharp; between two levels, both weighed), clamped to the map's levels
fn pyr_lod(pages: u32, foot: f32, texel: f32) -> f32 {
  return clamp(log2(max(foot, 1e-6) / texel), 0.0, f32(tab_levels(pages) - 1u));
}
// The look at level of detail x > 0, from the first level's look v0 (which only x < 1 needs) and the levels over it
fn pyr_tri(atlas: texture_2d_array<f32>, first: i32, pages: u32, size0: vec2f, q: vec2f, x: f32, chan: u32, kind: u32, v0: vec4f) -> vec4f {
  let l = u32(x); let f = x - f32(l);
  // the two levels in one loop, so that the look is written into the shader once
  var sum = select(vec4f(0.0), v0 * (1.0 - f), l == 0u);
  for (var k = max(l, 1u); k <= l + 1u; k++) {
    let w = select(f, 1.0 - f, k == l);
    if (w > 0.0) { sum += pyr_look(atlas, first, pages, size0, q, k, chan, kind) * w; }
  }
  return sum;
}
// How many metres a pixel spans at w, and at ground xz, as far below the eye as the ground under it (the looks that have
// only xz): what the eye sees of a map there is that much of it, filtered
fn pixel_span() -> f32 { return 2.0 * F.cam_right.w / F.cam_up.w; }
fn foot_at(w: vec3f) -> f32 { return distance(w, F.cam_pos.xyz) * pixel_span(); }
fn foot_xz(xz: vec2f) -> f32 { return length(vec3f(xz - F.cam_pos.xz, F.cam_pos.y - F.cam_vel.w)) * pixel_span(); }
// How much of the moon reaches w by the field over the whole map: 0 in shadow, 1 in the open or off the map. \`ease\`
// lowers the bar by so many metres, where the near map has the fine shadows and the field only adds the hills'.
fn moon_far(w: vec3f, n: vec3f, ease: f32) -> f32 { return moon_hills(w, n, ease) * cloud_shade(w); }
// the same, by the hills alone
fn moon_hills(w: vec3f, n: vec3f, ease: f32) -> f32 {
${OFF.has('moonfar') ? '  if (true) { return 1.0; } // no field: the near map alone, as it was' : ''}
  let q = w + n * ${MOON_STANDOFF.toFixed(2)};
  let uv = (q.xz - F.ground_rect.xy) / F.ground_rect.zw;
  if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0))) { return 1.0; }
  // the field at the level of detail the pixel's footprint takes (pyr_lod): each level's cell 2^x m wide, and its
  // penumbra that much softer, as a cell's own height step shows
  let foot = foot_at(q);
  let x = pyr_lod(${TABLES.map}u, foot, ${MOON_CELL.toFixed(1)});
  var field = 0.0;
  if (x < 1.0) { let m = map_at(uv); field = map_read(moon_map, ${MOON_FIRST}, m.page.x, m.page.y, m.at); }
  if (x > 0.0) { let size0 = ceil(F.ground_rect.zw * 0.25) * 4.0; field = pyr_tri(moon_map, ${MOON_FIRST}, ${TABLES.map}u, size0, uv * size0, x, 0u, 0u, vec4f(field)).r; }
  let above = field + ground_at(q.xz);
  let h = q.y - above + ease;
  let soft = ${MOON_SOFT.toFixed(5)} * exp2(x);
  // the penumbra is never wider than its widest (r = 1): well clear of it either way, its width need not be read
  if (abs(h) >= soft + ${(MOON_REACH_FAR * MOON_RADIUS).toFixed(5)}) { return step(0.0, h); }
  let xr = pyr_lod(${TABLES.reach}u, foot, ${(4 * MOON_CELL).toFixed(1)});
  var r = 0.0;
  if (xr < 1.0) { r = pair_look(pair_map, tab_first(${TABLES.reachFirst}u), ${TABLES.reach}u, uv).x; }
  if (xr > 0.0) { let size0 = ceil(F.ground_rect.zw * 0.25); r = pyr_tri(pair_map, tab_first(${TABLES.reachFirst}u), ${TABLES.reach}u, size0, uv * size0, xr, 0u, 1u, vec4f(r)).x; }
  let k = soft + r * r * ${(MOON_REACH_FAR * MOON_RADIUS).toFixed(3)};
  return smoothstep(-k, k, h);
}
// How much of the moon reaches w: 0 in shadow, 1 in the open. Near the car the shadow map's, sharp, with the field's
// over it for what the map leaves out (the hills); beyond it the field's alone, faded in over the map's last metres.
fn moon_shadow(w: vec3f, n: vec3f) -> f32 {
${OFF.has('moonshadow') ? '  if (true) { return 1.0; }' : ''}
  let p = F.shadow_vp * vec4f(w + n * 0.35, 1.0); // stood off the surface a little, so it does not shade itself
  let uv = vec2f(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
  let edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y)) * ${2 * SHADOW_HALF}.0; // m inside the map's square
  let near = select(smoothstep(0.0, ${SHADOW_FADE}.0, edge), 0.0, p.z > 1.0);
  let shade = cloud_shade(w);
  let far = moon_hills(w, n, near);
  if (near <= 0.0 || far <= 0.0) { return far * shade; } // beyond the near map; or in a hill's shadow, which the map can only darken
  let z = p.z - 0.0004;
  var sharp = 0.0;
  if (foot_at(w) > ${(3 * 2 * SHADOW_HALF / 4096).toFixed(3)}) { // a pixel wider than the tent's three texels: the tent is under it, and one filtered fetch is what it gives
    sharp = textureSampleCompareLevel(shadow_map, samp_shadow, uv, z);
  } else {
    let t = tent_at(uv, 4096.0); let f = tent_f(uv, 4096.0); let wa = 2.0 - f; let wb = 1.0 + f;
    sharp = (wa.x * wa.y * textureSampleCompareLevel(shadow_map, samp_shadow, t.xy, z) + wb.x * wa.y * textureSampleCompareLevel(shadow_map, samp_shadow, t.zy, z)
      + wa.x * wb.y * textureSampleCompareLevel(shadow_map, samp_shadow, t.xw, z) + wb.x * wb.y * textureSampleCompareLevel(shadow_map, samp_shadow, t.zw, z)) / 9.0;
  }
  return mix(far, min(sharp, far), near) * shade;
}
// How much of the sky reaches w on a surface facing n, by the map of what the ground sees (SKY_SEEN): 1 in the open, and
// away from the ground; under it in a tunnel, the map is of the ground overhead and nothing of it holds.
fn sky_seen(w: vec3f, n: vec3f) -> f32 { return sky_seen_over(w, n, w.y - ground_at(w.xz)); }
fn sky_seen_over(w: vec3f, n: vec3f, above: f32) -> f32 { // above: w.y - ground_at(w.xz), which the caller has
${OFF.has('sky') ? '  if (true) { return 1.0; }' : ''}
  let near = (1.0 - smoothstep(${SKY_SEEN.high[0].toFixed(2)}, ${SKY_SEEN.high[1].toFixed(2)}, above)) * smoothstep(-1.5, -0.5, above);
  if (near <= 0.0) { return 1.0; }
  let uv = (w.xz + n.xz * ${SKY_SEEN.standoff.toFixed(2)} - F.ground_rect.xy) / F.ground_rect.zw;
  let x = pyr_lod(${TABLES.map}u, foot_at(w), 1.0);
  var seen = 0.0;
  if (x < 1.0) { let m = map_at(uv); seen = map_read(pair_map, tab_first(${TABLES.skyFirst}u), m.page.z, m.page.w, m.at); }
  if (x > 0.0) { let size0 = ceil(F.ground_rect.zw * 0.25) * 4.0; seen = pyr_tri(pair_map, tab_first(${TABLES.skyFirst}u), ${TABLES.map}u, size0, uv * size0, x, 2u, 0u, vec4f(seen)).r; }
  let a = pow(seen, ${SKY_SEEN.deepen.toFixed(2)});
  return mix(1.0, mix(a, min(a * ${SKY_SEEN.side.toFixed(2)}, 1.0), 1.0 - abs(n.y)), near);
}
// The inverse square law (gfx.h): what a lamp of 1 cd gives a face d2 m^2 from it facing it, in the scene's units, 1 /
// (d2 SCENE_CD), from no nearer than LIGHT_NEAR; gone smoothly by \`reach\`, where the lamp's brightest has fallen under
// LUX_SEEN: the window (1 - (d / reach)^4)^2 is (1 - (LUX_SEEN / E)^2)^2 there (gfx_light_reach).
fn lux_at(d2: f32, reach: f32) -> f32 {
  let r = d2 / (reach * reach); let edge = 1.0 - r * r;
  return edge * edge * ${(1 / SCENE_CD).toFixed(6)} / max(d2, ${(LIGHT_NEAR * LIGHT_NEAR).toFixed(4)});
}
// A lamp at \`at\` shining along \`aim\`, light its colour times its candela at its brightest: full within \`core\` (a
// cosine off the aim), gone by \`cone\`.
fn spot(w: vec3f, n: vec3f, at: vec3f, aim: vec3f, light: vec3f, reach: f32, cone: f32, core: f32) -> vec3f {
  let to = at - w; let d2 = dot(to, to);
  if (d2 >= reach * reach) { return vec3f(0.0); }
  let l = to * inverseSqrt(max(d2, 1e-8));
  return light * (smoothstep(cone, core, dot(-l, aim)) * lux_at(d2, reach)) * max(dot(n, l), 0.0);
}
// The inverse square law within a beam whose light that way reaches \`reach\` (lux_at), nothing past it: a headlamp's
// reach falls off its axis with its light (head_across)
fn lux_beam(d2: f32, reach: f32) -> f32 {
  if (d2 >= reach * reach) { return 0.0; }
  return lux_at(d2, reach);
}
// Across a headlamp's beam, of its peak, h rad out from its axis in plan (gfx.h, gfx_head_across): half its light some
// 15 degrees out and a tenth at 30, and a low skirt, the foreground's spill, that fades smoothly to nothing by
// HEAD_SPREAD. A real dipped beam's cut-off is sharp up and down only; across, its pool on the road is a soft fan.
fn head_across(h: f32) -> f32 {
  let q = min(h * h * ${(1 / (HEAD_SPREAD * HEAD_SPREAD)).toFixed(5)}, 1.0); let s = 1.0 - q;
  return (0.85 * exp(-h * h * ${(1 / 0.09).toFixed(4)}) + 0.15 * exp(-h * h * ${(1 / 0.81).toFixed(4)})) * s * s;
}
// A headlamp's light up and down, of its peak, v rad up from the car's level. Dipped: the hot spot a degree under the
// level, the foreground's light under that falling as one over the angle down, so the road is lit evenly out to where
// the hot spot meets it; GFX_HEAD_CUT over the level the cut-off, a quarter of a degree, and over it the glare zone.
// Full: level, some 5 degrees to half either way. Never over 1, so a beam's reach that way is its reach times the root
// of head_across.
fn head_updown(v: f32, full: bool) -> f32 {
  if (full) { return exp(-v * v * ${(1 / (0.105 * 0.105)).toFixed(3)}); }
  let glare = 0.03 / (1.0 + v * v * ${(1 / (0.12 * 0.12)).toFixed(3)});
  let e = (max(v, 0.00995) - 0.00995) * ${(1 / 0.0045).toFixed(3)};
  let over = glare + (1.0 - glare) * exp(-e * e);
  return select(over, 0.0175 / max(-v, 0.0175), v <= 0.00995);
}
// A headlamp's light towards l (the unit way from it) on gfx_head_toward's grid (gfx.h): h across from its axis fwd, v
// up from the car's level; x across (head_across), y up and down (head_updown), the beam their product.
fn head_beam(l: vec3f, fwd: vec3f, up: vec3f, full: bool) -> vec2f {
  let ahead = dot(l, fwd);
  if (ahead <= 0.0) { return vec2f(0.0); }
  let v = dot(l, up);
  let h = atan2(sqrt(max(1.0 - ahead * ahead - v * v, 0.0)), ahead);
  return vec2f(head_across(h), head_updown(v, full));
}
// A car's headlamp at \`at\` (dipped, or its main beam), I cd at its peak, on w facing n: reaching \`reach\` on its
// axis, and less off it as its light is (lux_beam)
fn head_at(w: vec3f, n: vec3f, at: vec3f, fwd: vec3f, up: vec3f, full: bool, reach: f32) -> f32 {
  let to = at - w; let d2 = dot(to, to);
  if (d2 >= reach * reach) { return 0.0; }
  let l = to * inverseSqrt(max(d2, 1e-8));
  let facing = dot(n, l);
  if (facing <= 0.0) { return 0.0; } // a face turned from the lamp takes none: the beam's shape is not asked
  let b = head_beam(-l, fwd, up, full);
  return b.x * b.y * lux_beam(d2, reach * sqrt(b.x)) * facing;
}
// How much of a headlamp reaches w: trees, posts, signs and house corners stand in the beam. The same tent.
fn beam_shadow(w: vec3f, n: vec3f) -> f32 {
  let p = F.beam_vp[0] * vec4f(w + n * 0.05, 1.0); // the one map both lamps share (game.c)
  if (p.w <= 0.0) { return 1.0; }
  let q = p.xyz / p.w;
  let uv = vec2f(q.x * 0.5 + 0.5, 0.5 - q.y * 0.5);
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || q.z > 1.0) { return 1.0; }
  let t = tent_at(uv, 1024.0); let f = tent_f(uv, 1024.0); let wa = 2.0 - f; let wb = 1.0 + f; let z = q.z - 0.0015;
  return (wa.x * wa.y * textureSampleCompareLevel(beam_map, samp_shadow, t.xy, 0, z) + wb.x * wa.y * textureSampleCompareLevel(beam_map, samp_shadow, t.zy, 0, z)
    + wa.x * wb.y * textureSampleCompareLevel(beam_map, samp_shadow, t.xw, 0, z) + wb.x * wb.y * textureSampleCompareLevel(beam_map, samp_shadow, t.zw, 0, z)) / 9.0;
}
// How open the way from a light at \`at\` to w is past the town's buildings (wasm/src/seen.c): 1 in the open, 0 where the
// far wall of the nearest building in the way stands nearer the light than w and w lies under the line over its top.
// The light's sectors start at \`first\`, n of them across \`half\` either side of the way m points (in diamond angle, a
// quarter turn a unit; 2 all round). w between two sectors' middles takes some of each, so a shadow's side is a soft edge
// a sector wide, and its start a metre deep.
fn seen_sector(k: u32, d: f32, rise: f32) -> f32 {
  let v = (shine_grid[${SHINE.sec0}u + (k >> 1u)] >> ((k & 1u) * 16u)) & 0xffffu;
  let far = f32(v & 255u) * ${SEEN.step.toFixed(2)};
  let top = f32(v >> 8u) / 127.5 - 1.0; // the sine of how high the building rises, as the light sees it
  return 1.0 - smoothstep(far, far + 1.0, d) * (1.0 - smoothstep(top - 0.02, top + 0.02, rise));
}
fn seen_by(first: u32, n: u32, m: vec2f, half: f32, at: vec3f, w: vec3f) -> f32 {
${OFF.has('seen') ? '  if (true) { return 1.0; }' : ''}
  let to = w.xz - at.xz; let d = length(to);
  if (d < 0.5) { return 1.0; }
  let u = dot(to, m); let s = m.x * to.y - m.y * to.x; let k = s / (abs(u) + abs(s));
  let p = select(select(-2.0 - k, 2.0 - k, s >= 0.0), k, u >= 0.0); // the diamond angle (seen_diamond)
  let f = (p + half) / (2.0 * half) * f32(n) - 0.5;
  let i = floor(f); let t = f - i;
  let last = i32(n) - 1;
  var j0 = i32(i); var j1 = j0 + 1;
  if (half >= 2.0) { j0 = (j0 + i32(n)) % i32(n); j1 = j1 % i32(n); } else { j0 = clamp(j0, 0, last); j1 = clamp(j1, 0, last); }
  let dy = w.y - at.y; let rise = dy * inverseSqrt(d * d + dy * dy);
  return mix(seen_sector(first + u32(j0), d, rise), seen_sector(first + u32(j1), d, rise), t);
}
fn seen_round(first: u32, at: vec3f, w: vec3f) -> f32 { return seen_by(first, ${SEEN.round}u, vec2f(1.0, 0.0), 2.0, at, w); }
// A lamp shining all round (lamp_at) whose shadow over the buildings starts at sector \`first\`
// which of two lamps (xyz, w: reach) w lies within the reach of: bit 0 the first, bit 1 the second
fn lights_near2(w: vec3f, a: vec4f, b: vec4f) -> u32 {
  let ta = a.xyz - w; let tb = b.xyz - w;
  return u32(dot(ta, ta) < a.w * a.w) | (u32(dot(tb, tb) < b.w * b.w) << 1u);
}
fn lamp_seen(w: vec3f, n: vec3f, at: vec4f, light: vec3f, first: u32) -> vec3f {
  let to = at.xyz - w;
  if (dot(to, to) >= at.w * at.w) { return vec3f(0.0); }
  let c = lamp_at(w, n, at, light);
  if (all(c <= vec3f(0.0))) { return c; }
  return c * seen_round(first, at.xyz, w);
}
// The car's own lamps on whatever is around it. Dipped halogen beams: a soft-edged main beam from each lamp, toed
// out a little and aimed at the road some way ahead, over one broad, faint wash that fills in the ground just in
// front of the bumper; and the tail lamps' red pool behind, which flares under braking and whitens in reverse. On full
// beam (F.tail.w, game.c) each lamp's main filament as well: aimed level down the road, out to FULL_REACH, as the
// traffic's (traffic_lamps), a lamp each.
fn car_beams(w: vec3f, n: vec3f) -> vec3f {
  return car_beams_at(w, n);
}
// (the lamps' own bodies, car_beams_at, car_jets_at and car_tail_at, are not calmed in PRELUDE_CALM and PRELUDE_FAR's
// twins: the air between the eye and what is far still passes the car, and lights up round it in cloud, car_air)
fn car_beams_at(w: vec3f, n: vec3f) -> vec3f {
${OFF.has('beams') ? '  if (true) { return vec3f(0.0); }' : ''}
  let at = F.car_pos.xyz; let fwd = F.car_dir.xyz; let up = F.car_up.xyz;
  let ra = car_reach();
  if (dot(w - at, w - at) > ra * ra) { return vec3f(0.0); }
  let left = cross(up, fwd);
  var dip = 0.0; var main = 0.0;
  for (var i = 0; i < 2; i++) {
    let lamp = at + left * ((f32(i) * 2.0 - 1.0) * 0.55) + up * 0.735 + fwd * 2.33;
    dip += head_at(w, n, lamp, fwd, up, false, ${DIP_REACH.toFixed(1)}); // dipped halogens, as the traffic's
    if (F.tail.w > 0.0) { main += head_at(w, n, lamp, fwd, up, true, ${FULL_REACH.toFixed(1)}); } // and on full beam the main beams
  }
  let beams = ${luminous([1.0, 0.715694, 0.423268])} * (dip * ${I_DIP.toFixed(1)}) + ${luminous([1.0, 0.75, 0.48])} * (main * ${I_FULL.toFixed(1)} * F.tail.w);
  if (beams.r > 0.0) { return beams * beam_shadow(w, n); } // the two lamps share one map: looked up once
  return vec3f(0.0);
}
// No flame burns quite steadily: the gas's turbulence makes it breathe a few per cent, a little differently in each
// jet. The flames (JETS) and their light share this, so what they light flickers with them.
fn jet_flicker(i: u32) -> f32 {
  let t = F.cam_pos.w; let k = f32(i) * 1.73;
  return 1.0 + 0.045 * sin(t * 47.0 + k) * sin(t * 29.0 + k * 2.9) + 0.03 * sin(t * 13.0 + k * 5.3) + 0.015 * sin(t * 83.0 + k * 0.7);
}
// The jets' flames (rocket.h) as the lamps they are: each a little way down its plume, falling off as the square of the
// distance (from no nearer than JET_LIGHT.near, the plume's own girth), gone by JET_LIGHT.reach. A plume is no point:
// what faces away from it still gets some, round its length. Nothing at all unless one burns.
fn car_jets(w: vec3f, n: vec3f) -> vec3f {
  return car_jets_at(w, n);
}
fn car_jets_at(w: vec3f, n: vec3f) -> vec3f {
  if (F.jet.r <= 0.0) { return vec3f(0.0); }
  var sum = 0.0;
  for (var i = 0u; i < 5u; i++) {
    let l = F.jet_at[i];
    let to = l.xyz - w; let d2 = dot(to, to);
    if (l.w <= 0.0 || d2 >= ${(JET_LIGHT.reach * JET_LIGHT.reach).toFixed(1)}) { continue; }
    let d = sqrt(d2); let edge = 1.0 - pow4(d / ${JET_LIGHT.reach.toFixed(1)});
    let aim = F.jet_aim[i]; let out = mix(1.0, smoothstep(-0.35, 0.6, dot(-to / max(d, 1e-4), aim.xyz)), aim.w); // boosted, out of the nozzle's mouth
    sum += l.w * jet_flicker(i) * out * edge * edge / max(d2, ${(JET_LIGHT.near * JET_LIGHT.near).toFixed(3)}) * (${JET_LIGHT.wrap.toFixed(2)} + ${(1 - JET_LIGHT.wrap).toFixed(2)} * max(dot(n, to / max(d, 1e-4)), 0.0));
  }
  return F.jet.rgb * sum;
}
// The tail lamps' light from between them on the tail, F.tail (TAIL), light in F.tail's units (TAIL_CD candela each):
// back and out to the sides, none forward of the lamps, so nothing of the car itself takes it; as far as the pair's
// light, running, braking or reversing, reaches (gfx_light_reach).
fn tail_pool(w: vec3f, n: vec3f, at: vec3f, back: vec3f, light: vec3f) -> vec3f {
  let to = at - w; let d2 = dot(to, to);
  let reach2 = dot(light, vec3f(${(0.2126 * TAIL_CD / LUX_SEEN).toFixed(4)}, ${(0.7152 * TAIL_CD / LUX_SEEN).toFixed(4)}, ${(0.0722 * TAIL_CD / LUX_SEEN).toFixed(4)}));
  if (d2 >= reach2) { return vec3f(0.0); }
  let l = to * inverseSqrt(max(d2, 1e-8));
  // a tail lamp throws its candela back within some 15 degrees of level (ECE R7's field); steeply down, at the road a
  // metre or two behind the car, only its lens's spill, a tenth of it
  let level = mix(1.0, 0.1, smoothstep(0.26, 0.5, l.y));
  return light * (${TAIL_CD.toFixed(3)} * smoothstep(0.0, 0.5, -dot(l, back)) * level * lux_at(d2, sqrt(reach2))) * max(dot(n, l), 0.0);
}
fn car_tail(w: vec3f, n: vec3f) -> vec3f {
  return car_tail_pool(w, n) + car_jets(w, n); // (car_jets as PRELUDE_CALM has it)
}
fn car_tail_at(w: vec3f, n: vec3f) -> vec3f { return car_tail_pool(w, n) + car_jets_at(w, n); }
fn car_tail_pool(w: vec3f, n: vec3f) -> vec3f {
  let back = -F.car_dir.xyz;
  return tail_pool(w, n, F.car_pos.xyz + F.car_up.xyz * 0.85 + back * 2.3, back, F.tail.rgb);
}
// The traffic's lamps on what is round each car: its dipped beams as one spot from the middle of its front, aimed at the
// road ahead, and its tail lamps' red pool behind. The buildings stand in the beams (seen_by).
// On the way it finds how much of the light from above each car takes from w, the snow under it and round its foot: a
// soft patch the shape of its body on what faces up within half a metre of the road, darkest beneath it and gone a
// metre out, so not on the car itself, into traffic_shade, which the lighting reads after it (the player's car has its
// own blob, AV_BLOB). It fades as the car's lamps do.
var<private> traffic_shade: f32 = 1.0;
// The bucket of the light list's grid that w's cell falls in (wasm/include/harstad/shine.h, shine_bucket)
fn light_bucket(xz: vec2f) -> u32 {
  let c = vec2i(floor(xz / ${SHINE.cell.toFixed(1)}));
  return ((bitcast<u32>(c.x) * 73856093u) ^ (bitcast<u32>(c.y) * 19349663u)) & ${SHINE.buckets - 1}u;
}
// How open the way from light L to w is past the buildings, by its sectors, looked up from where they were worked out
fn light_seen(L: Light, half: f32, w: vec3f) -> f32 {
  let n = select(${SEEN.car}u, ${SEEN.round}u, half >= 2.0);
  return seen_by(bitcast<u32>(L.seen.w), n, L.seen_m.xy, half, L.seen.xyz, w);
}
// A car's pair of headlamps as one from the middle of its front, at: dipped halogens, and on full beam its main beams
// aimed level down the road, at the snowbanks, the trees and the signs far ahead
// (head_at for each, with what the two share worked out once: the way to the lamps, the face's turn to them, the light
// across the beams' axis)
fn traffic_beams(w: vec3f, n: vec3f, at: vec3f, fwd: vec3f, full: bool) -> vec3f {
  let to = at - w; let d2 = dot(to, to);
  let dip: f32 = ${DIP_REACH.toFixed(1)}; let reach: f32 = select(dip, ${FULL_REACH.toFixed(1)}, full);
  if (d2 >= reach * reach) { return vec3f(0.0); }
  let l = to * inverseSqrt(max(d2, 1e-8));
  let facing = dot(n, l);
  let ahead = -dot(l, fwd);
  if (facing <= 0.0 || ahead <= 0.0) { return vec3f(0.0); }
  let v = -l.y;
  let across = head_across(atan2(sqrt(max(1.0 - ahead * ahead - v * v, 0.0)), ahead));
  if (d2 >= reach * reach * across) { return vec3f(0.0); } // its light that way under LUX_SEEN (lux_beam), or none
  let k = sqrt(across);
  var beam = ${luminous([1.0, 0.715694, 0.423268])} * (across * head_updown(v, false) * lux_beam(d2, dip * k) * facing * ${(2 * I_DIP).toFixed(1)});
  if (full) { beam += ${luminous([1.0, 0.75, 0.48])} * (across * head_updown(v, true) * lux_at(d2, ${FULL_REACH.toFixed(1)} * k) * facing * ${(2 * I_FULL).toFixed(1)}); }
  return beam;
}
fn traffic_lamps(w: vec3f, n: vec3f) -> vec3f {
  var sum = vec3f(0.0);
  traffic_shade = 1.0;
${OFF.has('traffic') ? '  if (true) { return sum; }' : ''}
  // only the lights whose reach comes into w's bucket of the grid (wasm/src/shine.c): most pixels have none to ask
  let b = light_bucket(w.xz);
  let end = shine_grid[b + 1u];
  for (var i = shine_grid[b]; i < end; i++) {
    let L = shine_lights[shine_grid[${SHINE.index0}u + i]];
    let to = w - L.at.xyz; let d2 = dot(to, to);
    if (d2 > L.at.w * L.at.w) { continue; }
    if (L.col.w > 0.5) { // a lamp shining all round: the emergency services' blue lights, floodlights, a fire, the aircraft's, the ships'
      var light = L.col.rgb;
      if (abs(L.dir.x) > 0.5) { light *= event_aim(w, L.at.xyz, L.dir.x); } // a lamp with a reflector (a floodlight, a landing light) or a lens that fans it out (a light bar)
      let c = lamp_at(w, n, L.at, light);
      if (any(c > vec3f(0.0))) { sum += c * light_seen(L, 2.0, w); }
      continue;
    }
    let dir = L.dir;
    let full = dir.w > 1.5; // on full beam, on a dark road (wasm/src/traffic.c, traffic__beams)
    let under = to.y + 0.7; // above the road it stands on
    if (d2 < 30.0 && under < 1.6${OFF.has('contact') ? ' && false' : ''}) {
      let body = vec2f(dot(to.xz, vec2f(dir.x, dir.y)), dot(to.xz, vec2f(-dir.y, dir.x))); // along it and across
      let q = abs(body) - vec2f(1.95, 0.6);
      let sd = length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0); // m out from its body's footprint, less a rounding
      traffic_shade = min(traffic_shade, 1.0 - 0.8 * (1.0 - smoothstep(-0.4, 1.1, sd)) * (1.0 - smoothstep(0.1, 0.6, under)) * smoothstep(0.4, 0.8, n.y));
    }
    let far = select(${DIP_REACH.toFixed(1)}, ${FULL_REACH.toFixed(1)}, full); // as far as its dipped beams reach, or its main beams (game.c, GAME_FULL_REACH)
    if (d2 > (far + 4.0) * (far + 4.0)) { continue; }
    let fwd = vec3f(dir.x, 0.0, dir.y);
    let along = dot(to, fwd);
    // Only what lies ahead of its lamps, 2.2 m ahead of its middle, is worked out for its beams (traffic_beams: as far
    // out as their light is seen, the soft fan of head_across); behind, only the tail lamps' few metres.
    if (along > 2.2) {
      let beam = traffic_beams(w, n, L.at.xyz + fwd * 2.2, fwd, full);
      if (any(beam > vec3f(0.0))) { sum += beam * light_seen(L, ${SEEN.carHalf}, w); }
    } else if (along < 0.0 && d2 < 14.0 * 14.0) {
      sum += tail_pool(w, n, L.at.xyz - fwd * 2.3, -fwd, vec3f(1.0, 0.01033, 0.003677) * (${TAIL.pool.toFixed(3)} * dir.z));
    }
  }
  return sum;
}
fn car_lamps(w: vec3f, n: vec3f) -> vec3f { return car_beams(w, n) + car_tail(w, n) + traffic_lamps(w, n); }
// A lamp shining all round from at.xyz, light its colour times its candela, as far as at.w (gfx_light_reach): the
// inverse square law, lux_at.
fn lamp_at(w: vec3f, n: vec3f, at: vec4f, light: vec3f) -> vec3f {
  let to = at.xyz - w; let d2 = dot(to, to);
  if (d2 >= at.w * at.w) { return vec3f(0.0); }
  return light * lux_at(d2, at.w) * max(dot(n, to * inverseSqrt(max(d2, 1e-8))), 0.0);
}
// The ones that come and go: the colour the TV tower's lit drum throws on the hilltop and the trees around it, the
// tunnel luminaires' light on the car, the Julehus's two coloured lights on its garden, its walls and the street, the
// Christmas truck's bulbs on the road, the banks and the walls it passes, and the air ambulance's searchlight on the
// helipad and what is round it as it comes in.
fn tower_wash(w: vec3f, n: vec3f) -> vec3f {
${OFF.has('tower') ? '  if (true) { return vec3f(0.0); }' : ''}
  // Each of them is nothing beyond its reach, and nearly every pixel is beyond them all. But this is in every shader that
  // lights a surface, and what a shader holds costs it whether it is run or not (measured: a loop made never to run cost
  // the frame what it did, the same loop out of the source 1.6 ms less; a test round the work saves nothing): the more
  // of such work a shader holds, the slower all of it runs. So there is as little of it here as will do: one copy of a
  // shadowed lamp's work (lamp_seen) for all fourteen of them, and loops that the compiler cannot unroll into more.
  var sum = lamp_at(w, n, F.wash_at, F.wash.rgb) + lamp_at(w, n, F.tunnel_at, F.tunnel.rgb) + plough_lamps(w, n, 0.0);
  if (F.heli_at.w > 0.0) { sum += spot(w, n, F.heli_at.xyz, F.heli_aim.xyz, F.heli.rgb, F.heli_at.w, F.heli_aim.w, F.heli.w); }
  // The lamps that shine all round and are shadowed by the buildings, one loop for them all (each copy of that work in
  // a shader costs it, run or not): the Julehus's two, bits 0 and 1; and the truck's four, from bit 2, all within 62 m of
  // the middle of its trailer's sides (lights 1 and 2) with what they reach. (What happens in town, the blue lights,
  // floodlights and fires, the aircraft's and the ships', is in the light list: traffic_lamps.)
  let mid = 0.5 * (F.truck_at[1].xyz + F.truck_at[2].xyz) - w;
  var near = lights_near2(w, F.jul_at[0], F.jul_at[1])
    | select(0u, (lights_near2(w, F.truck_at[0], F.truck_at[1]) << 2u) | (lights_near2(w, F.truck_at[2], F.truck_at[3]) << 4u), dot(mid, mid) < 62.0 * 62.0);
  while (near != 0u) {
    let k = countTrailingZeros(near); near &= near - 1u;
    var at = F.truck_at[min(k, 5u) - min(k, 2u)]; var light = F.truck[min(k, 5u) - min(k, 2u)].rgb; var first = 4u + k; // which of the buildings' shadows (seenRound)
    if (k < 2u) { at = F.jul_at[k]; light = F.jul[k].rgb; first = 10u + k; }
    sum += lamp_seen(w, n, at, light, ${seenRound(0)} + first * ${SEEN.round}u);
  }
  return sum;
}
// How much of a lamp with a reflector at at reaches w: its aim packed in code (wasm/include/harstad/rescue.h, rescue_light_aim),
// its heading and pitch in 256ths, and the cosine off its aim at which its light is gone; full over the inner part of that.
// A round lamp's aim (wasm/include/harstad/rescue.h): over 0 a reflector's cone (event_cone), under 0 a fan all round
// (rescue_light_fan): a light bar's LEDs behind lenses that lay their light out flat, half of it at the sine of the
// angle over or under the level the code holds (in 255ths), and a broad tail three times as wide that has a seventh of it
fn event_aim(w: vec3f, at: vec3f, code: f32) -> f32 {
  if (code > 0.0) { return event_cone(w, at, code); }
  let s = max(f32(u32(-code + 0.5) - 1u) / 255.0, 1.0 / 255.0);
  let x = normalize(w - at).y / s;
  return 0.85 * exp(-0.6931 * x * x) + 0.15 * exp(-0.0770 * x * x);
}
fn event_cone(w: vec3f, at: vec3f, code: f32) -> f32 {
  let s = u32(code + 0.5) - 1u;
  let yaw = (f32(s & 255u) + 0.5) * (6.2831853 / 256.0) - 3.1415927;
  let pitch = f32((s >> 8u) & 255u) * (3.1415927 / 255.0) - 1.5707963;
  let c = f32(s >> 16u) / 255.0;
  let aim = vec3f(cos(pitch) * sin(yaw), sin(pitch), cos(pitch) * cos(yaw));
  return smoothstep(c, mix(c, 1.0, 0.4), dot(normalize(w - at), aim));
}
// The winter road crews' lights (wasm/src/plough.c). A beacon is a lamp behind an amber dome whose reflector turns, so its
// beam, a narrow fan some 25 degrees across, sweeps round and lights what it falls on as it passes; what misses the
// reflector leaves the dome on the beam's own side, and only a little gets through the reflector's back. Each from its own
// dome (wasm/src/fleet.c, fleet_beacon), turned as the beam is: two on the cab's roof, PLOUGH_BAR m either side of the
// middle of their bar, and one on the hopper's back. And the work lights: on
// the roof, on the blade and the road ahead; aside on the wing; behind on the spreader. What stands round a plough is
// lit as its beams come round, and the buildings stand in them: the cab's lights' shadows and the hopper's (seen_by). all_round 1 for what is lit from every side alike (the snow in the air), 0 for
// a surface facing n. Only what lies within a beacon's reach of a plough asks more than one distance.
fn beacon_beam(to: vec3f, a: f32) -> f32 {
  let flat = to.xz / max(length(to.xz), 1e-3);
  let c = max(dot(flat, vec2f(sin(a), cos(a))), 0.0); let c2 = c * c; let c4 = c2 * c2; let c8 = c4 * c4;
  let up = abs(to.y) / max(length(to), 1e-3);
  return c8 * c8 * (1.0 - smoothstep(0.45, 0.9, up)) + (0.01 + 0.05 * c2) * (1.0 - smoothstep(0.7, 1.0, up)); // the beam, and what the dome spills
}
// One beacon at at, its beam turned to a, on w facing n, as far as reach
fn beacon_at(w: vec3f, n: vec3f, at: vec3f, a: f32, reach: f32, all_round: f32) -> f32 {
  let to = at - w; let d2 = dot(to, to);
  if (d2 >= reach * reach) { return 0.0; }
  return beacon_beam(-to, a) * lux_at(d2, reach) * mix(max(dot(n, to * inverseSqrt(max(d2, 1e-8))), 0.0), 1.0, all_round);
}
const PLOUGH_BAR = 0.78; // m either side of their bar's middle the cab's beacons stand (wasm/src/fleet.c, fleet_plough_lamps)
fn plough_spot(w: vec3f, n: vec3f, at: vec3f, aim: vec3f, light: vec3f, reach: f32, cone: f32, core: f32, all_round: f32) -> vec3f {
  let to = at - w; let d2 = dot(to, to);
  if (d2 >= reach * reach) { return vec3f(0.0); }
  let l = to * inverseSqrt(max(d2, 1e-8));
  return light * (smoothstep(cone, core, dot(-l, aim)) * lux_at(d2, reach)) * mix(max(dot(n, l), 0.0), 1.0, all_round);
}
fn plough_lamps(w: vec3f, n: vec3f, all_round: f32) -> vec3f {
  var sum = vec3f(0.0);
${OFF.has('plough') ? '  if (true) { return sum; }' : ''}
  var near = lights_near2(w, F.plough_at[0], F.plough_at[1]) | ((lights_near2(w, F.plough_at[2], F.plough_at[2]) & 1u) << 2u); // (no plough: its reach is 0)
  while (near != 0u) { // only the ploughs within reach, in their order (tower_wash)
    let k = countTrailingZeros(near); near &= near - 1u;
    let at = F.plough_at[k];
    let turn = F.plough_turn[k]; let back = F.plough_back[k]; let colour = F.plough[k];
    let fwd = vec3f(sin(turn.w), 0.0, cos(turn.w)); let right = vec3f(-fwd.z, 0.0, fwd.x); let up = vec3f(0.0, 1.0, 0.0);
    var cab = colour.rgb * (beacon_at(w, n, at.xyz + right * PLOUGH_BAR, turn.x, ${lightReach(PLOUGH_WORK[2]).toFixed(1)}, all_round) + beacon_at(w, n, at.xyz - right * PLOUGH_BAR, turn.y, ${lightReach(PLOUGH_WORK[2]).toFixed(1)}, all_round));
    var hopper = colour.rgb * beacon_at(w, n, back.xyz, turn.z, ${lightReach(PLOUGH_WORK[2]).toFixed(1)}, all_round);
    let work = ${luminous([1.0, 0.93, 0.82])} * back.w;
    cab += plough_spot(w, n, at.xyz + fwd * 0.21 - up * 0.14, normalize(fwd * 0.93 - up * 0.36), work * ${PLOUGH_WORK[0].toFixed(1)}, ${lightReach(PLOUGH_WORK[0]).toFixed(1)}, 0.7, 0.93, all_round);
    if (colour.w > 0.5) { cab += plough_spot(w, n, at.xyz + right * 1.27 - up * 0.34 - fwd * 0.61, normalize(right * 0.3 - up * 0.56 - fwd * 0.77), work * ${PLOUGH_WORK[1].toFixed(1)}, ${lightReach(PLOUGH_WORK[1]).toFixed(1)}, 0.55, 0.9, all_round); }
    hopper += plough_spot(w, n, back.xyz + right * 0.55 - up * 0.19 - fwd * 0.13, normalize(-fwd * 0.78 - up * 0.62), work * ${PLOUGH_WORK[1].toFixed(1)}, ${lightReach(PLOUGH_WORK[1]).toFixed(1)}, 0.5, 0.88, all_round);
    if (any(cab > vec3f(0.0))) { sum += cab * seen_round(${seenRound(0)} + k * ${SEEN.round}u, at.xyz, w); }
    if (any(hopper > vec3f(0.0))) { sum += hopper * seen_round(${seenRound(3)} + k * ${SEEN.round}u, back.xyz, w); }
  }
  return sum;
}
// What of the sky's light the truck's trailer takes from w, the snow under it: as the cars' patches (traffic_lamps), the
// shape of its body (13.6 by 2.55 m, wasm/include/harstad/fleet.h), found from its side lights, which stand 3 m out from
// its middle either side, 2.3 m over its kingpin; softer, as it stands higher. Its tractor has a car's patch.
fn truck_shade(w: vec3f, n: vec3f) -> f32 {
${OFF.has('contact') ? '  if (true) { return 1.0; }' : ''}
  let side = F.truck_at[1].xyz - F.truck_at[2].xyz;
  let mid = 0.5 * (F.truck_at[1].xyz + F.truck_at[2].xyz);
  let to = w - mid;
  if (F.truck_at[1].w <= 0.0 || dot(to.xz, to.xz) > 100.0) { return 1.0; }
  let across = normalize(side.xz);
  let q = abs(vec2f(dot(to.xz, vec2f(-across.y, across.x)), dot(to.xz, across))) - vec2f(6.4, 0.9);
  let sd = length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0);
  return 1.0 - 0.65 * (1.0 - smoothstep(-0.6, 1.4, sd)) * (1.0 - smoothstep(0.1, 0.6, to.y + 2.3)) * smoothstep(0.4, 0.8, n.y); // (the kingpin is at the road's level)
}
// Whether any light that moves (car_lamps and its shades, tower_wash) can reach what lies within r m of xz, by the game's
// grid of where they shine (game.c, game__lit_mark): every cell the square round xz touches is asked, and where that is
// more than four cells a side, or r no number, the answer is yes. For what is drawn whole from one place, asked once in
// its vertex shader.
fn lights_reach(xz: vec2f, r: f32) -> bool {
  if (!(r >= 0.0 && r < 1e4)) { return true; }
  let lo = vec2u(clamp(floor((xz - r - F.lit_at.xy) * F.lit_at.zw), vec2f(0.0), vec2f(${GAME_LIT_GRID - 1}.0)));
  let hi = vec2u(clamp(floor((xz + r - F.lit_at.xy) * F.lit_at.zw), vec2f(0.0), vec2f(${GAME_LIT_GRID - 1}.0)));
  if (any(hi - lo > vec2u(3u))) { return true; }
  for (var j = lo.y; j <= hi.y; j++) {
    for (var i = lo.x; i <= hi.x; i++) {
      let k = j * ${GAME_LIT_GRID / 32}u + (i >> 5u);
      if (((F.lit_grid[k >> 2u][k & 3u] >> (i & 31u)) & 1u) != 0u) { return true; }
    }
  }
  return false;
}
// false where a shader has found (lights_reach) that none of them reaches its pixel: its lighting (litOver, gated) then asks none
var<private> lit_moving: bool = true;
// Lambert under the sky and the moon and in the car's lamps, plus the lamp map; \`gain\` is the material's share of lamp
// light. For what snow cannot lie on (the car, the tunnel's lining).
fn lit_bare(albedo: vec3f, n: vec3f, w: vec3f, gain: f32) -> vec3f {
  let lamps = car_lamps(w, n); // first: it finds the traffic's shade
  let seen = sky_seen(w, n); let under = traffic_shade * truck_shade(w, n); // the cars stand between the snow and the lamps overhead too
  let open = seen * under;
  let sky = (sky_light(w.xz, n.y) + aurora_light(n) * seen) * open + moon_light(w, n, 0.0) + lamps + tower_wash(w, n);
  return albedo * (sky / PI + lamp_light(w, n) * (gain * 1.6 * mix(1.0, seen, ${SKY_SEEN.lamps.toFixed(2)}) * under));
}

// ---- the snow's surface: wind relief, and the glints on any up-facing snow (src/world.js, snowSurface) ----
const SNOW_WIND = vec2f(0.71, -0.71);  // the way the prevailing south-westerly blows, x east, z south
const SNOW_ACROSS = vec2f(0.71, 0.71);
fn max3(v: vec3f) -> f32 { return max(v.x, max(v.y, v.z)); }
// Wind relief: the slope (dh/dx, dh/dz) of the snow at world xz, from one tiling slope map read at two stretches, long
// drifts lying across the wind and fine sastrugi along it. Mip-mapping levels both out with distance; the sastrugi are
// faded out and not read at all where a wave of them is under RIDGE_PX pixels, where they would be flat anyway, and
// the drifts likewise under DRIFT_PX. dx, dy: how xz changes from one pixel to the next.
fn pixel_angle() -> f32 { return 2.0 * F.cam_right.w / F.cam_up.w; } // rad one pixel spans, up the screen
fn snow_slope(xz: vec2f, dist: f32, dx: vec2f, dy: vec2f) -> vec2f {
  let pa = pixel_angle();
  let relief_far = ${DRIFT[0].toFixed(1)} / (${DRIFT_PX.toFixed(1)} * pa); let ridge_far = ${RIDGE[1].toFixed(2)} / (${RIDGE_PX.toFixed(1)} * pa);
  if (dist > relief_far) { return vec2f(0.0); }
  let w = vec2f(dot(xz, SNOW_WIND), dot(xz, SNOW_ACROSS));
  let wx = vec2f(dot(dx, SNOW_WIND), dot(dx, SNOW_ACROSS)); let wy = vec2f(dot(dy, SNOW_WIND), dot(dy, SNOW_ACROSS));
  var s = (textureSampleGrad(detail_map, samp_repeat, w / ${reach(DRIFT)}, 1, wx / ${reach(DRIFT)}, wy / ${reach(DRIFT)}).rg * 2.0 - 1.0) * ${steep(DRIFT)}
    * (1.0 - smoothstep(relief_far * 0.6, relief_far, dist));
  if (dist < ridge_far) {
    s += (textureSampleGrad(detail_map, samp_repeat, w / ${reach(RIDGE)} + 0.37, 1, wx / ${reach(RIDGE)}, wy / ${reach(RIDGE)}).rg * 2.0 - 1.0) * ${steep(RIDGE)}
      * (1.0 - smoothstep(ridge_far * 0.6, ridge_far, dist));
  }
  return s.x * SNOW_WIND + s.y * SNOW_ACROSS;
}
fn glint_hash(p: vec2f) -> vec3f {
  var q = fract(vec3f(p.x, p.y, p.x) * vec3f(0.1031, 0.1030, 0.0973));
  q += dot(q, q.yzx + 33.33);
  return fract((q.xxy + q.yzz) * q.zyx);
}
// One level of grains: each cell on the ground plane holds one grain, at a random spot, with a facet tilted at random
// from the surface normal n. Drawn as a round dot about two CSS pixels across wherever it lands on screen: solving for
// the pixel offset keeps the dot round at grazing angles. Returns how well the facet mirrors the lamps overhead (hl)
// and the headlamps beside the eye (v) into the eye.
fn glint_level(xz: vec2f, cell: f32, dx: vec2f, dy: vec2f, det: f32, n: vec3f, v: vec3f, hl: vec3f) -> vec2f {
  let c = floor(xz / cell);
  let a = glint_hash(c);
  if (a.z > 0.35) { return vec2f(0.0); } // most cells hold no grain that could catch the light: nothing more to work out
  let b = glint_hash(c + 517.0);
  let e = (c + 0.3 + 0.4 * a.xy) * cell - xz;
  let px = vec2f(e.x * dy.y - e.y * dy.x, e.y * dx.x - e.x * dx.y) / det;
  let m = normalize(n + (b * 2.0 - 1.0) * 1.2);
  return (1.0 - smoothstep(0.4, 1.2, length(px) / F.misc.z)) * vec2f(smoothstep(0.92, 1.0, dot(m, hl)), smoothstep(0.92, 1.0, dot(m, v)));
}
// Glints: snow is loose ice grains, and at any moment a few have a facet turned to mirror a light into the eye. They
// twinkle as the eye moves and stand still when it does. lamp: the painted lamplight here; head: the direct light (the
// headlamps; the moon is too dim to count). Cells stay at least five CSS pixels wide by doubling in size with distance,
// cross-fading between two sizes, so no grain is ever smaller than a pixel. They fade out towards where GLINT_SPAN is a pixel, past which
// none are looked for.
fn snow_glints(w: vec3f, n: vec3f, lamp: vec3f, head: vec3f, dx: vec2f, dy: vec2f, dist: f32) -> vec3f {
  let head_amt = smoothstep(0.08, 0.4, max3(head));
  let det = dx.x * dy.y - dx.y * dy.x;
  let glint_far = ${GLINT_SPAN.toFixed(3)} / pixel_angle();
  if (dist > glint_far || max3(lamp) + head_amt < ${GLINT_MIN} || abs(det) < 1e-12) { return vec3f(0.0); }
  let v = (F.cam_pos.xyz - w) / dist; let hl = normalize(v + vec3f(0.0, 1.0, 0.0));
  let level = max(0.0, log2(166.66667 * F.misc.z * max(length(dx), length(dy))));
  let l0 = floor(level); let t = level - l0; let cell = 0.03 * exp2(l0);
  var g = glint_level(w.xz, cell, dx, dy, det, n, v, hl) * (1.0 - t);
  if (t > 0.02) { g += glint_level(w.xz, cell * 2.0, dx, dy, det, n, v, hl) * t; }
  return (lamp * g.x + head * head_amt * g.y) * 7.0 * (1.0 - smoothstep(glint_far * 0.6, glint_far, dist));
}
// ---- rime (rimfrost): the ice the clear, still night grows on whatever is thin and out in the open ----
// A thin thing (a twig, a rail, a post, a fence's boards, a parked car) has little heat in it, so it radiates down to below
// the air's -2 degrees under the open sky and the air's moisture freezes onto it as fine needles; they grow most on the
// side the air comes from, where it keeps bringing more, and hardly at all where a roof or a crown hides the sky. A
// shader whose surfaces carry rime says how much in rime_on (0 for none, which is everything else) before it lights
// them: their albedo goes part of the way to the rime's (tree.c's RIME), and a few needles' facets mirror a lamp or a
// headlamp into the eye, as the snow's grains do, only finer and fewer. Lit by the real lights only: no rime of its own glows.
var<private> rime_on: f32 = 0.0;
const RIME_ALBEDO = vec3f(0.58, 0.62, 0.71);
const RIME_WHITE = ${RIME.white.toFixed(3)}; // the most of the way to it the albedo goes, on the windward face of what sees all the sky
// How much rime a surface facing n carries, seeing a share seen of the sky: some all round, more facing the wind (it
// blows from the south-west, SNOW_WIND the way it goes), and less underneath.
fn rime_share(n: vec3f, seen: f32) -> f32 {
  let windward = smoothstep(0.0, 0.8, -dot(n.xz, SNOW_WIND));
  return rime_on * seen * (${RIME.round.toFixed(2)} + ${(1 - RIME.round).toFixed(2)} * windward) * (0.55 + 0.45 * saturate(n.y + 0.6));
}
// One level of needles: cells on the plane through w that most nearly faces n (ax, ay along it, q there), one needle at a
// random spot in a few of them, its facet tilted at random. A dot about two CSS pixels across, measured across the eye's
// ray so it stays round where the surface is seen edge-on; foot: metres a pixel spans at w.
fn rime_level(q: vec2f, cell: f32, ax: vec3f, ay: vec3f, v: vec3f, n: vec3f, hl: vec3f, foot: f32) -> vec2f {
  let c = floor(q / cell);
  let a = glint_hash(c + 91.0);
  if (a.z > ${RIME.grains.toFixed(2)}) { return vec2f(0.0); }
  let b = glint_hash(c + 733.0);
  let e = (c + 0.3 + 0.4 * a.xy) * cell - q;
  let e3 = ax * e.x + ay * e.y;
  let px = length(e3 - v * dot(e3, v)) / foot;
  let m = normalize(n + (b * 2.0 - 1.0) * 1.1);
  return (1.0 - smoothstep(0.4, 1.2, px / F.misc.z)) * vec2f(smoothstep(0.92, 1.0, dot(m, hl)), smoothstep(0.92, 1.0, dot(m, v)));
}
// The needles that catch a light. lamp: the painted lamplight there; head: the direct light (headlamps). Cells are
// RIME.cell across where that is five CSS pixels or more, and double with distance, cross-faded, so no needle is ever
// smaller than a pixel; they fade out towards where RIME.span is a pixel and none are looked for past it: there the
// rime is only the whitening.
fn rime_glints(w: vec3f, n: vec3f, lamp: vec3f, head: vec3f, dist: f32) -> vec3f {
  let pa = pixel_angle();
  let far = ${RIME.span.toFixed(3)} / pa;
  let head_amt = smoothstep(0.08, 0.4, max3(head));
  if (dist > far || max3(lamp) + head_amt < ${GLINT_MIN}) { return vec3f(0.0); }
  let a = abs(n);
  var ax = vec3f(1.0, 0.0, 0.0); var ay = vec3f(0.0, 0.0, 1.0); var q = w.xz;
  if (a.x > a.y && a.x >= a.z) { ax = vec3f(0.0, 1.0, 0.0); q = w.yz; }
  else if (a.z > a.y && a.z > a.x) { ay = vec3f(0.0, 1.0, 0.0); q = w.xy; }
  let v = (F.cam_pos.xyz - w) / dist; let hl = normalize(v + vec3f(0.0, 1.0, 0.0));
  let foot = dist * pa;
  let level = max(0.0, log2(foot * F.misc.z * 5.0 / ${RIME.cell.toFixed(3)}));
  let l0 = floor(level); let t = level - l0; let cell = ${RIME.cell.toFixed(3)} * exp2(l0);
  var g = rime_level(q, cell, ax, ay, v, n, hl, foot) * (1.0 - t);
  if (t > 0.02) { g += rime_level(q, cell * 2.0, ax, ay, v, n, hl, foot) * t; }
  return (lamp * g.x + head * head_amt * g.y) * ${RIME.bright.toFixed(1)} * (1.0 - smoothstep(far * 0.6, far, dist));
}
// The same light on something snow may lie on. cover: how much of the fragment is lying snow, or -1 to judge it from
// the colour (pale and bluish, as the snow colours are); either way only what faces up holds snow. relief: the share
// of the wind relief shown. dx, dy: how the world xz changes from one pixel to the next. moon_floor: the share of
// moonlight that still arrives in the moon's shadow (a tree's crown scatters it; anything else takes none).
fn lit_under(albedo: vec3f, n0: vec3f, w: vec3f, gain: f32, cover_in: f32, relief: f32, dx: vec2f, dy: vec2f, moon_floor: f32) -> vec3f {
  let above = w.y - ground_at(w.xz);
  return lit_over(albedo, n0, w, gain, cover_in, relief, dx, dy, moon_floor, lamp_reach_over(w, above), 1.0, above);
}
// The same for a caller that has the painted light at w already (lamp_reach), having a use of its own for it. own: the
// share of the sky the surface's own shape leaves it (a wall's inner corners and eaves, bld.c), over what the ground
// round it takes (sky_seen) and the traffic's shade; the light that comes straight from a lamp or the moon is taken by
// none of them.
fn lit_reached(albedo: vec3f, n0: vec3f, w: vec3f, gain: f32, cover_in: f32, relief: f32, dx: vec2f, dy: vec2f, moon_floor: f32, reach: vec3f, own: f32) -> vec3f {
  return lit_over(albedo, n0, w, gain, cover_in, relief, dx, dy, moon_floor, reach, own, w.y - ground_at(w.xz));
}
${litOver('lit_over', false)}fn lit_snow(albedo: vec3f, n0: vec3f, w: vec3f, gain: f32, cover_in: f32, relief: f32, dx: vec2f, dy: vec2f) -> vec3f {
  return lit_under(albedo, n0, w, gain, cover_in, relief, dx, dy, 0.0);
}
// Call where every pixel of the triangle gets here (it asks its neighbours how far apart they are).
fn lit(albedo: vec3f, n: vec3f, w: vec3f, gain: f32) -> vec3f { return lit_snow(albedo, n, w, gain, -1.0, 1.0, dpdx(w.xz), dpdy(w.xz)); }

fn hash2(p: vec2f) -> f32 { return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453); }
fn hash3(p: vec3f) -> f32 { return fract(sin(dot(p, vec3f(127.1, 311.7, 74.7))) * 43758.5453); }
fn noise(p: vec2f) -> f32 {
  let i = floor(p);
  var f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash2(i), hash2(i + vec2f(1.0, 0.0)), f.x), mix(hash2(i + vec2f(0.0, 1.0)), hash2(i + vec2f(1.0, 1.0)), f.x), f.y);
}
// Five octaves of noise, stopped once the sum is sure to end below lo or above hi: noise is within [0, 1], so the
// octaves still to come add less than 2a. For a caller that only needs to know which side of the band the sum falls.
fn fbm_band(q: vec2f, lo: f32, hi: f32) -> f32 {
  var p = q; var a = 0.5; var s = 0.0;
  for (var i = 0; i < 5; i++) {
    s += a * noise(p); p = p * 2.03 + 17.0; a *= 0.5;
    if (s + 2.0 * a < lo || s > hi) { break; }
  }
  return s;
}
// The aurora's noise runs along one row of the lattice a curtain: its y is the same all over the sky, so each octave
// of it is a line of values, mixed along y once a frame (SKY_MAPS' rows) and then only along x. Twelve rows of five
// octaves, for each of three curtains: its slow wander, its slow rays, its fast folds and its fast rays;
// |p.x| is within 4 wherever a curtain is looked at (5 for what the whole form's drift carries), and a slot holds the
// 1024 lattice points that span at the finest octave. \`state\` is the night's aurora this frame (aurora_state).
struct SkyRows { origin: array<i32, 64>, light: array<vec4f, 4>, state: array<vec4f, 2>, g: array<f32, ${60 * 1024}> }
fn row_noise(slot: u32, x: f32) -> f32 {
  let i = floor(x); var f = x - i;
  f = f * f * (3.0 - 2.0 * f);
  let at = slot * 1024u + u32(clamp(i32(i) - sky_rows.origin[slot], 0, 1022));
  return mix(sky_rows.g[at], sky_rows.g[at + 1u], f);
}
// The aurora's light on a surface facing n, as irradiance (as sky_now() is), from the sky the eye sees this frame
// (AURORA_LIGHT): a + b.n in each colour, the curtains' radiance taken to first order in direction, with the snow under
// them giving some back from below.
fn aurora_light(n: vec3f) -> vec3f {
  let l = sky_rows.light;
  return max(l[0].xyz + vec3f(dot(l[1].xyz, n), dot(l[2].xyz, n), dot(l[3].xyz, n)), vec3f(0.0));
}
// fbm_band along a row: the same sum, from the row's lines
fn fbm_row(row: u32, x0: f32, lo: f32, hi: f32) -> f32 {
  var x = x0; var a = 0.5; var s = 0.0;
  for (var k = 0u; k < 5u; k++) {
    s += a * row_noise(row * 5u + k, x); x = x * 2.03 + 17.0; a *= 0.5;
    if (s + 2.0 * a < lo || s > hi) { break; }
  }
  return s;
}
// The night's aurora at the game's time t, all from the clock and a seed, so any frame of it can be drawn again: x how
// bright its curtains are, y how far they have risen from a low arc in the north (high in the sky at 0.6, the arcs poleward of it lower), z how active
// (fast folds and rays, the N2 rim), w the pulsating patches of its recovery; and [1] x the whole form's drift east or
// west, y the event's seed. Quiet, a faint still arc low in the north. An event: the arc brightens and rises slowly
// (growth); breaks up in seconds into bright curtains that fold, ray and race across the sky and overhead (onset and
// breakup), in bursts; and dies away into blinking patches (recovery).
struct Aurora { a: vec4f, b: vec4f }
fn aurora_event(u: f32, k: f32) -> Aurora {
  let quiet = Aurora(vec4f(0.04, -0.45, 0.0, 0.0), vec4f(0.0, fract(k * 0.618), 0.0, 0.0));
  let g = ${AURORA.grow.toFixed(1)} * (1.0 + hash2(vec2f(k, 1.3)));
  let b = ${AURORA.breakup.toFixed(1)} * (1.0 + 0.5 * hash2(vec2f(k, 2.9)));
  let r = ${AURORA.recover.toFixed(1)} * (1.0 + 0.5 * hash2(vec2f(k, 4.1)));
  let strength = 0.7 + 0.3 * hash2(vec2f(k, 5.7));
  if (u < 0.0 || u > g + b + r) { return quiet; }
  var o = quiet;
  if (u < g) { // growth
    let x = u / g;
    o.a = vec4f(0.04 + 0.36 * smoothstep(0.0, 1.0, x), -0.45 + 0.6 * x, 0.0, 0.0);
  } else if (u < g + b) { // onset and breakup, in bursts
    let x = u - g;
    let act = smoothstep(0.0, 6.0, x) * (1.0 - smoothstep(b - 25.0, b, x)) * (0.72 + 0.28 * sin(x * 0.23 + k * 4.0));
    o.a = vec4f(mix(0.4, strength, smoothstep(0.0, 5.0, x)), mix(0.15, 0.6, smoothstep(0.0, 35.0, x)), act, 0.0);
  } else { // recovery
    let x = (u - g - b) / r;
    o.a = vec4f(mix(strength, 0.04, smoothstep(0.0, 0.5, x)), mix(0.6, -0.45, smoothstep(0.3, 1.0, x)), 0.0, smoothstep(0.0, 0.15, x) * (1.0 - smoothstep(0.6, 1.0, x)));
  }
  let sway = 0.8 * sin(u * 0.045 + k * 6.0); // the form's drift: as fast as the event is active
  o.b.x = sway * smoothstep(g, g + 20.0, u) * (1.0 - smoothstep(g + b, g + b + 0.4 * r, u));
  return o;
}
// What the page's debug desk asks for over that (F.car_up.w, struct game's aurora): an event set off at a moment, a
// storm, or none. An event set off takes the place of any the night has within an event's length of it, each from
// quiet to quiet, so nothing switches mid-way.
fn aurora_state(t: f32, ask: f32) -> Aurora {
  if (ask < -1.5) { return Aurora(vec4f(0.0, -0.45, 0.0, 0.0), vec4f(0.0)); }
  if (ask < 0.0) { return aurora_storm(t); }
  if (ask > 0.0) {
    let at = ask - 1.0;
    if (t >= at && t < at + ${AURORA_LONGEST.toFixed(1)}) { return aurora_event(t - at, -2.0); }
    return aurora_night(t, at);
  }
  return aurora_night(t, -1e9);
}
fn aurora_storm(t: f32) -> Aurora {
  return Aurora(vec4f(0.95, 0.35 + 0.25 * sin(t * 0.013), 0.75 + 0.25 * sin(t * 0.23), 0.0), vec4f(0.8 * sin(t * 0.045), 0.3, 0.0, 0.0));
}
// The night's own events, one in each slot; none that would meet one set off at \`asked\`.
fn aurora_night(t: f32, asked: f32) -> Aurora {
${AURORA_MODE === 0 ? '  return Aurora(vec4f(0.0, -0.45, 0.0, 0.0), vec4f(0.0));' : AURORA_MODE === 2 ? '  return aurora_storm(t);' : `  var k = floor(t / ${AURORA.slot.toFixed(1)});
  var start = k * ${AURORA.slot.toFixed(1)} + ${AURORA.lead.toFixed(1)} + hash2(vec2f(k, 0.7)) * ${AURORA.jitter.toFixed(1)};
  if (t < start) { // the slot's own has not begun: the last slot's may still be dying away
    k -= 1.0;
    start = k * ${AURORA.slot.toFixed(1)} + ${AURORA.lead.toFixed(1)} + hash2(vec2f(k, 0.7)) * ${AURORA.jitter.toFixed(1)};
  }
  if (abs(start - asked) < ${AURORA_LONGEST.toFixed(1)}) { return aurora_event(-1.0, k); }
${AURORA_MODE === 1 ? `  // ?aurora=1 likewise (an event is over within ${AURORA_LONGEST} s)
  if (t >= ${AURORA_AT.toFixed(1)} && t < ${(AURORA_AT + AURORA_LONGEST).toFixed(1)}) { return aurora_event(t - ${AURORA_AT.toFixed(1)}, -1.0); }
  if (abs(start - ${AURORA_AT.toFixed(1)}) < ${AURORA_LONGEST.toFixed(1)}) { return aurora_event(-1.0, k); }
` : ''}  return aurora_event(t - start, k);`}
}
// The night sky along d, in linear light: stars, moon, drifting cloud and aurora. \`fine\` is 0 for a cheaper
// sky without stars or the aurora's rays, which is all a rippled reflection can show.
fn sky(d: vec3f, fine: f32) -> vec3f { return sky_under(d, fine, MOON_DIR); }
// The same with the moon along \`moon\`: the water takes its sky with the moon set below the horizon, and draws the moon's glitter itself.
fn sky_under(d: vec3f, fine: f32, moon: vec3f) -> vec3f { return sky_over(d, fine, moon, sky_noise(d, fine)); }
// The part of the sky that is all noise and changes slowly across the view: the aurora's light (rgb, to be added) and
// how much thin cloud drifts over it (a). The sky the eye sees takes it at a quarter of the pixels (SKY_MAPS).
fn sky_noise(d: vec3f, fine: f32) -> vec4f {
  let time = F.cam_pos.w;
  let h = max(d.y, 0.0);
  // aurora: curtains hanging over the northern sky, with fine vertical rays, as the night's event has them now
  let st = sky_rows.state[0]; let level = st.x; let act = st.z; let xs = sky_rows.state[1].x;
  let p = d.xz / (d.y + 0.22);
  var glow = 0.0; var low_rim = 0.0; var top = 0.0;
  let fold = act * ${AURORA.fold.toFixed(2)};
  let sharp = mix(5.0, 15.0, smoothstep(0.08, 0.4, level)); // a quiet arc is a diffuse band
  let rayed = smoothstep(0.08, 0.4, level);
  for (var i = 0; i < 3; i++) {
    let fi = f32(i);
    let w = select(smoothstep(0.2, 0.6, level), 1.0, i == 0); // the arcs poleward of the first only once it is active
    let c0 = p.y + 0.9 + fi * 0.62 - st.y;
    // The noise is most of this shader's cost and a curtain is a thin band of the sky: skip it where the band
    // cannot reach (the wander below is within +-0.85, the folds within +-fold), and its rays where it has faded.
    if (w <= 0.0 || c0 > 2.4 + fold || c0 < -1.87 - fold || d.y < 0.03) { continue; }
    // past c of 1.5 above or -1.02 below, all of the curtain is under 1e-4: the bounded sum stops there
    var c = c0 + (fbm_row(u32(i) * 4u, (p.x + xs) * 0.55 + fi * 7.3, 0.5 - (1.02 + c0 + fold) / 1.7, 0.5 + (1.5 - c0 + fold) / 1.7) - 0.5) * 1.7;
    if (fold > 0.0) { c += (fbm_row(u32(i) * 4u + 2u, (p.x + xs) * 2.2 + F.cam_pos.w * 0.35 + fi * 3.7, -1.0, 2.0) - 0.5) * 2.0 * fold; }
    let cur = exp(-c * c * sharp);
    let rim = exp(-(c + 0.27) * (c + 0.27) * 60.0); // the lower border
    let red = exp(-(c - 0.45) * (c - 0.45) * 9.0); // the top, high in the thin air
    if (cur + rim + red < 1e-4) { continue; }
    var rays = 0.72;
    if (fine > 0.0) {
      let slow = 0.45 + 0.55 * fbm_row(u32(i) * 4u + 1u, p.x * 7.0 + time * 0.05, -1.0, 2.0);
      var fast = slow;
      if (act > 0.0) { fast = 0.2 + 1.1 * fbm_row(u32(i) * 4u + 3u, p.x * 7.0 + time * 1.4 + fi * 2.3, -1.0, 2.0); }
      rays = mix(0.72, mix(slow, fast, act), rayed);
    }
    glow += cur * rays * w; low_rim += rim * rays * w; top += red * rays * w;
  }
  // pulsating aurora as an event dies away: diffuse patches in a loose band where the arcs were, drawn out east and
  // west with ragged edges, each pulsing on and off over seconds out of step with its neighbours. Faded in with the
  // sky's height as the curtains are, so nothing of it is cut.
  var patches = 0.0;
  if (st.w > 0.0 && d.y > 0.03) {
    let seed = sky_rows.state[1].y * 31.0;
    let band = p.y + 0.9 - st.y; // across the arcs' system
    let q = vec2f(p.x * 0.9 + time * 0.004, band * 2.6) + seed;
    let shape = smoothstep(0.48, 0.68, fbm_band(q, 0.48, 0.68)) * exp(-band * band * 1.5);
    if (shape > 0.0) {
      let a = noise(q * 0.8 + 3.1) * 6.2831853 * 2.0; let b = noise(q * 0.8 + 11.7) * 6.2831853 * 2.0;
      let blink = smoothstep(0.25, 0.85, 0.5 + 0.3 * sin(time * 1.698 + a) + 0.2 * sin(time * 0.861 + b)); // 3.7 s and 7.3 s
      patches = shape * (0.25 + 0.75 * blink) * (0.7 + 0.3 * noise(q * 6.0)) * st.w;
    }
  }
  let am = smoothstep(0.03, 0.3, d.y) * ${(AURORA.bright * MOONLIT).toFixed(5)};
  let aurora = (vec3f(0.10, 0.85, 0.30) * (glow * level + patches * ${AURORA.patch.toFixed(2)})
    + vec3f(0.80, 0.12, 0.55) * low_rim * level * (0.08 + ${AURORA.rim.toFixed(2)} * act) // N2 at the lower border, the faster the brighter
    + vec3f(0.55, 0.05, 0.04) * top * level * (0.10 + 0.25 * act)) * am; // 630 nm oxygen over it
  // thin snow cloud drifting through, faintly lit from below
  let cp = d.xz / (h + 0.14) * 1.2 + vec2f(time * 0.006, time * 0.002);
  var cl = 0.0;
  if (d.y > 0.0) { cl = smoothstep(0.52, 0.86, fbm_band(cp * vec2f(0.8, 1.3), 0.52, 0.86)) * smoothstep(0.0, 0.12, d.y); }
  return vec4f(aurora, cl);
}
// The rest of the sky over that noise: the gradient, the stars, the moon.
fn sky_over(d: vec3f, fine: f32, moon: vec3f, noise: vec4f) -> vec3f {
  let time = F.cam_pos.w;
  let h = max(d.y, 0.0);
  var col = mix(vec3f(0.009, 0.012, 0.022), vec3f(0.003, 0.0045, 0.012), pow(h, 0.5)) * (${NIGHT} * F.sky_now.x);
  col += dusk_sky(d);

  if (fine > 0.0) {
    let sp = d * 240.0;
    let sh = hash3(floor(sp));
    var star = step(0.9962, sh) * (1.0 - smoothstep(0.0, 0.42, length(fract(sp) - 0.5)));
    star *= 0.6 + 0.4 * sin(time * (1.0 + sh * 3.0) + sh * 90.0);
    col += vec3f(0.85, 0.9, 1.0) * star * ((0.22 + fract(sh * 77.0) * 0.8) * ${MOONLIT.toFixed(5)}) * smoothstep(0.0, 0.2, d.y);
  }
  col += noise.rgb;

  let to_moon = max(dot(d, moon), 0.0);
  let bright = F.moon_now.rgb; // its phase's and the air's, and none once it has set
  col += vec3f(0.55, 0.66, 0.9) * bright * ((pow(to_moon, 40.0) * 0.05 + pow(to_moon, 700.0) * 0.5) * ${NIGHT}); // the moon's aureole, in the air as the sky is
  let disc = smoothstep(0.99962, 0.99975, to_moon) * fine;
  if (disc > 0.0) { // the disc lit on the sun's side as its phase has it: where on it the sun, at the phase angle behind it, shines
    let across = d - moon * to_moon; let r = 0.0245; // (the disc's radius, sin 1.4 degrees)
    let sp = F.sun.xyz - moon * dot(F.sun.xyz, moon); let sl = length(sp);
    let e1 = select(vec3f(0.0, 1.0, 0.0), sp / sl, sl > 1e-5);
    let x = dot(across, e1) / r; let rr = min(dot(across, across) / (r * r), 1.0);
    let nz = sqrt(1.0 - rr); let i = F.moon_now.w;
    let lit = smoothstep(-0.04, 0.04, x * sin(i) + nz * cos(i));
    // its light kept whatever the phase: the disc's mean is the moon's brightness, over the share of it lit
    let share = max(0.5 * (1.0 + cos(i)), 0.02);
    col += vec3f(1.0, 0.97, 0.9) * disc * ${(2500 / SCENE_CD).toFixed(2)} * (bright / share * lit + bright * 0.004); // the full moon's disc, some 2,500 cd/m2
  }
  return mix(col, vec3f(0.036, 0.038, 0.048) * (${NIGHT} * F.sky_now.x) + DUSK_LIGHT * (F.sun.w * 0.09), noise.a * 0.72);
}
// The twilight's sky (the sun under the horizon, F.sun): its light (as F.sun.w has it on the ground) from a dome
// brighter towards the horizon, a pink and orange glow low over where the sun is, and the band of pink over the dark
// blue of the earth's own shadow opposite. Its light on the ground over pi is its radiance on the whole.
fn dusk_sky(d: vec3f) -> vec3f {
  let k = F.sun.w;
  if (k <= 0.0) { return vec3f(0.0); }
  let h = max(d.y, 0.0);
  let flat = normalize(vec2f(d.x, d.z) + vec2f(1e-6, 0.0)); let sun = normalize(F.sun.xz + vec2f(1e-6, 0.0));
  let toward = dot(flat, sun) * 0.5 + 0.5; // 1 under the sun, 0 opposite
  let low = exp(-h * 9.0);
  let dome = DUSK_LIGHT * (0.55 + 0.9 * pow4(1.0 - h));
  let glow = vec3f(1.0, 0.55, 0.32) * (2.2 * pow4(toward) * toward * toward * low);
  let belt = vec3f(0.85, 0.55, 0.65) * (0.35 * pow4(1.0 - toward) * exp(-abs(h - 0.09) * 18.0)) - DUSK_LIGHT * (0.3 * pow4(1.0 - toward) * exp(-h * 30.0));
  return max(dome + (glow + belt) * dot(DUSK_LIGHT, vec3f(0.2126, 0.7152, 0.0722)), vec3f(0.0)) * (k / PI);
}
`;
// PRELUDE for what none of the lights that come and go can reach (wasm/include/harstad/game.h, game_washes): tower_wash
// and car_jets are nothing in it. Each of those lights is exactly nothing outside its reach, so what none reaches is
// drawn the same either way; but a shader works out all it holds on every pixel, reached or not (tower_wash's own
// note), and with these two out of every shader the frame took 2.1 ms less of 16.5 in the street (the forest's draws
// 0.7 of that, the road's 0.7). The C marks each draw one of them can reach (GFX_CMD_WASHED), and the glue draws those
// alone with the shader made from PRELUDE, everything else with its twin made from this.
const calmed = (code, fn, is = 'vec3f(0.0)') => {
  const a = code.indexOf(`\nfn ${fn}(`), b = code.indexOf('{\n', a) + 2, e = code.indexOf('\n}\n', b);
  if (a < 0 || b < 2 || e < 0) throw new Error(`PRELUDE_CALM: no ${fn}`);
  return code.slice(0, b) + `  return ${is};` + code.slice(e);
};
// The low lights' switches burning now (sw_on), a switch a thread, once a frame before the frame's passes: each of the
// table's room switches (src/worker.js lists them, with their caps, at load) gets the share of its light burning, its
// rooms' weights lit times its cap, at most 1, written over its cap's word (word 1), where sw_on reads it. The rooms'
// hours (room_on, office_on) are the clock's, the same for every pixel: a pixel used to work them out for each of the
// four texels round it.
export const SW_NOW = /* wgsl */`
@group(1) @binding(0) var<storage, read_write> sw_grid: array<u32>;
@group(1) @binding(1) var<storage, read> sw_list: array<vec2u>; // each room switch: its head's word from LAMP_SW0, its cap's bits
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= arrayLength(&sw_list)) { return; }
  let e = sw_list[g.x]; let b = e.x;
  let head = sw_grid[${LAMP_SW0}u + b]; let n = (head >> 8u) & 255u;
  let one = (head & ${1 << 20}u) != 0u; let wts = sw_grid[${LAMP_SW0}u + b + 2u];
  var sum = 0.0; var all = 0.0; // (the weights' bytes, rounded, sum to about 255: all lit is all of it)
  for (var i = 0u; i < n; i++) {
    let home = sw_grid[${LAMP_SW0}u + select(b + 3u + i * 2u, b + 3u, one)]; let id = sw_grid[${LAMP_SW0}u + select(b + 4u + i * 2u, b + 4u + i, one)];
    let on = select(room_on(home, id), office_on(id), ((head >> (16u + i)) & 1u) != 0u);
    let w = f32((wts >> (8u * i)) & 255u);
    all += w; sum += select(0.0, w, on);
  }
  sw_grid[${LAMP_SW0}u + b + 1u] = bitcast<u32>(min(select(1.0, sum / all, all > 0.0) * bitcast<f32>(e.y), 1.0));
}
`;
// And for the traffic's cars a few pixels long (game.c, GAME_HERD_FAR_PX): the far set, and none of the other cars'
// lamps either (traffic_lamps, which walks the light list's cell for every pixel): at a few pixels a car's paint takes
// nothing of another's beams that a pixel could show
export const PRELUDE_HERD_FAR = () => calmed(PRELUDE_FAR, 'traffic_lamps');
export const PRELUDE_CALM = calmed(calmed(PRELUDE, 'tower_wash'), 'car_jets');
// And for what lies, besides, beyond all that only shows near by (game.h, game_far; GFX_CMD_FAR): further from the eye
// than a glint is looked for on snow or rime (snow_glints and rime_glints give nothing past GLINT_SPAN and RIME.span a
// pixel), further from the car than its lamps reach (car_beams, 85 m; car_tail's pool, 11), and outside the near shadow
// map's square, where moon_shadow is the far field's alone. Straight down from 1500 m, where all the ground is that,
// the frame took 3.8 ms less of 27 with these out of every shader.
export const PRELUDE_FAR = calmed(calmed(calmed(calmed(calmed(PRELUDE_CALM, 'snow_glints'), 'rime_glints'), 'car_beams'), 'car_tail'), 'moon_shadow', 'moon_far(w, n, 0.0)');

/* ---- terrain: no vertex buffer. One strip of cells is drawn once per grid row; heights come from a texture ---- */

// cut: a tunnel portal opens this ground. clip: the town is cut out of this piece of the backdrop. Only the chunks a portal
// reaches, and the backdrop's that the town's edge crosses (wasm/src/terrain.c), take a shader with a discard in it: on a
// tile-based GPU a discard costs every fragment of a shader that has one, whether it fires or not (src/tunnels.js, cuts).
// The terrain's grid: where it lies, its photo's pages and the tunnel portals it opens for. The near one is the snow
// field's too (PACK), which shows untouched snow exactly as the terrain does (ground_snow).
const GRID = /* wgsl */`
struct Grid {
  origin: vec2f,      // world position of grid point (0, 0)
  step: f32,
  clip: f32,          // 1: the backdrop
  size: vec2u,        // grid points
  extent: vec2f,      // width and depth of the grid: the backdrop's one photo covers it
  pages: vec2u,       // the photo's pages across and down (tools/imagery.py, pages); none: one photo over the extent
  page_at: vec2f,     // world position of page (0, 0)'s corner: the tiles' corner (the backdrop: of the tiles' grid)
  page: f32,          // m a page's side (the backdrop: a tile's)
  layer: f32,         // texels along a side of a page's layer
  px: vec2f,          // the photo's texels a metre, across and down
  cut_a: array<vec4f, 4>, // tunnel portals, where the ground is opened: mouth x, z, direction in
  cut_b: array<vec4f, 4>, // side from, side to, top at the mouth, slope
  tile: vec4i,        // the heights' layers (hs_tile_layer): the grid point at the tiles' corner, points a tile's side (the backdrop: one tile)
  chunks: vec4u,      // cells along a chunk's side, chunks across, down: the chunks the levels of detail are by (terrain_lod)
}
`;
// The snow on the ground at w as the terrain shows it: its colour, from the summer photo under it, and how much of it
// lies there (thinner under the trees). ddx, ddy: how w.xz changes from one pixel to the next.
const GROUND_SNOW = /* wgsl */`
// The summer photo at xz: a look into the page it lies in, by where the page's corner falls in its layer (photo_pages:
// the layer, that corner in its texels, and the pool the page is held in). A page holds its neighbours' pixels round
// its edge, so a look there weighs them as one photo of the whole map would: the pages meet without a seam. Only the
// pages the eye can see finely are held (wasm/include/harstad/photo.h): pool L (photos, photos1, photos2) holds levels L
// to 2 of a page, and coarser than level 2 the look is into the coarse photo, the whole map at the pages' level 3
// (tools/imagery.py, coarse). The look's level is the footprint's narrower side in texels, or an eighth of its wider (the
// filter's most anisotropy); from 2 to 3 the page's level 2 gives way to the coarse photo, as a trilinear look between the
// two levels weighs them. A page held in no pool is the coarse photo alone.
fn ground_photo(xz: vec2f, ddx: vec2f, ddy: vec2f) -> vec3f {
  if (G.pages.x == 0u) { return textureSampleGrad(photos, samp_photo, (xz - G.origin) / G.extent, 0, ddx / G.extent, ddy / G.extent).rgb; }
  let at = xz - G.page_at;
  let a = length(ddx * G.px); let b = length(ddy * G.px);
  let p = clamp(vec2i(floor(at / G.page)), vec2i(0), vec2i(G.pages) - 1);
  let e = textureLoad(photo_pages, p, 0);
  let t = select(clamp(log2(max(min(a, b), max(a, b) * 0.125)) - 2.0, 0.0, 1.0), 1.0, e.w > 2.5);
  var fine = vec3f(0.0); var coarse = vec3f(0.0);
  if (t < 1.0) {
    let k = G.px / G.layer;
    let uv = (e.yz + (at - vec2f(p) * G.page) * G.px) / G.layer;
    if (e.w < 0.5) { fine = textureSampleGrad(photos, samp_photo, uv, u32(e.x), ddx * k, ddy * k).rgb; }
    else if (e.w < 1.5) { fine = textureSampleGrad(photos1, samp_photo, uv, u32(e.x), ddx * k, ddy * k).rgb; }
    else { fine = textureSampleGrad(photos2, samp_photo, uv, u32(e.x), ddx * k, ddy * k).rgb; }
  }
  if (t > 0.0) {
    let c = G.px / (8.0 * vec2f(textureDimensions(photo_coarse)));
    coarse = textureSampleGrad(photo_coarse, samp_photo, at * c, ddx * c, ddy * c).rgb;
  }
  return mix(fine, coarse, t);
}
fn ground_snow(w: vec3f, ddx: vec2f, ddy: vec2f) -> vec4f {
  let photo = ground_photo(w.xz, ddx, ddy);

  // The summer aerial photo still shapes the snow: open ground is bright, forest canopy and built-up clutter
  // stay darker, so fields, woods and plots remain readable from above.
  let lum = dot(photo, vec3f(0.30, 0.59, 0.11));
  let canopy = clamp((photo.g - max(photo.r, photo.b)) * 9.0, 0.0, 1.0) * (1.0 - smoothstep(0.05, 0.16, lum));
  var grain = 0.0; // the backdrop shows none
  if (G.clip < 1.0) {
    grain = textureSampleGrad(detail_map, samp_repeat, w.xz * 0.37, 0, ddx * 0.37, ddy * 0.37).r * 0.55
      + textureSampleGrad(detail_map, samp_repeat, w.xz * 0.051, 0, ddx * 0.051, ddy * 0.051).r * 0.45;
  }
  var snow = vec3f(0.84, 0.89, 0.97) * (0.70 + 0.34 * smoothstep(0.0, 0.3, lum));
  snow = mix(snow, vec3f(0.30, 0.35, 0.41), canopy * 0.62);
  snow *= mix(0.86 + 0.28 * grain, 1.0, G.clip);
  // all of it is snow; under the trees it lies thinner and sheltered
  return vec4f(snow, 1.0 - canopy * 0.7);
}
`;

// bare: its depth alone (the clouds' pass, src/worker.js), with nothing where the town is cut out of the backdrop
// ---- the ground far off, shaded once a pixel (src/worker.js, defer) ----
// Far from the eye the ground's triangles are a pixel or less across, and the GPU shades every triangle that holds a
// pixel's middle in a 2 x 2 quad of its own: from high over the town each pixel of ground was lit some 4.5 times over.
// So the ground's draws marked GFX_CMD_FAR are drawn first into a buffer of what lighting needs (terrainGbuf: the
// point, its normal, how it moves from pixel to pixel), lit there once a pixel (terrainDeferred), and the frame's own
// ground then only reads its pixel's light (terrainRead). The same shading, from the same inputs: the point and its
// footprint are those of the fragment that holds the pixel's middle; where none of the far ground does, the frame's
// fragment lights itself as before.
const TERRAIN_TAIL = '  let snow = ground_snow(w, ddx, ddy);\n  return vec4f(finish(lit_snow(snow.rgb, normalize(v.n), w, 1.0, snow.a, 1.0, ddx, ddy), w), 1.0);\n}\n';
const terrainTail = (code, tail) => { if (!code.includes(TERRAIN_TAIL)) throw new Error('terrain tail'); return code.replace(TERRAIN_TAIL, tail); };
// g: 0 the town's ground, 1 the backdrop's (their groups differ). a: the point's bits, and in w its normal's x and z
// in 14 bits each over 1 (the ground's normal faces up), the bit that says it is there, the backdrop's bit, and bit 29
// for what is lit with every light, near (the drifts: DUNE_GBUF), not only those that reach far
const gbufWord = (n, g, near) => `0x80000000u | (${g}u << 30u) | (${near}u << 29u) | gbuf_normal(${n})`;
const GBUF_NORMAL = /* wgsl */`
fn gbuf_normal(n: vec3f) -> u32 { let q = vec2u(round((clamp(n.xz, vec2f(-1.0), vec2f(1.0)) * 0.5 + 0.5) * 16383.0)); return (q.x << 14u) | q.y; }
`;
export const terrainGbuf = (cut, clip, g) => GBUF_NORMAL + terrainTail(terrainLit(cut, clip), `  let n = normalize(v.n);
  return GOut(vec4u(bitcast<vec3u>(w), ${gbufWord('n', g, 0)}), vec4f(ddx, ddy));
}
`).replace('@fragment fn fs(v: V) -> @location(0) vec4f {', 'struct GOut { @location(0) a: vec4u, @location(1) b: vec4f }\n@fragment fn fs(v: V) -> GOut {')
  .split('discard; return vec4f(0.0);').join('discard; return GOut();');
export const terrainDeferred = (g, near = 0) => { const c = terrainLit(false, false); return c.slice(0, c.indexOf('@vertex fn vs(')) + /* wgsl */`
@group(2) @binding(0) var g_a: texture_2d<u32>;
@group(2) @binding(1) var g_b: texture_2d<f32>;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f { return vec4f(f32(i & 1u) * 4.0 - 1.0, f32(i >> 1u) * 4.0 - 1.0, 0.0, 1.0); }
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let p = vec2i(pos.xy);
  let a = textureLoad(g_a, p, 0);
  if ((a.w >> 31u) == 0u || ((a.w >> 30u) & 1u) != ${g}u || ((a.w >> 29u) & 1u) != ${near}u) { discard; return vec4f(0.0); }
  let w = bitcast<vec3f>(a.xyz);
  let d = textureLoad(g_b, p, 0); let ddx = d.xy; let ddy = d.zw;
  let q = vec2f(f32((a.w >> 14u) & 0x3fffu), f32(a.w & 0x3fffu)) * ${(2 / 16383).toExponential(8)} - 1.0;
  let n = normalize(vec3f(q.x, sqrt(max(1.0 - dot(q, q), 0.0)), q.y));
  let snow = ground_snow(w, ddx, ddy);
  return vec4f(finish(lit_snow(snow.rgb, n, w, 1.0, snow.a, 1.0, ddx, ddy), w), 1.0);
}
`; };
export const terrainRead = (cut, clip) => terrainTail(terrainLit(cut, clip), `  let lit = textureLoad(terrain_shaded, vec2i(v.pos.xy), 0);
  if (lit.a > 0.75) { return lit; } // the far ground's (the plain faces' is 0.5: PLAIN_SHADE)
${TERRAIN_TAIL}`).replace('@fragment fn fs(v: V) -> @location(0) vec4f {', '@group(2) @binding(0) var terrain_shaded: texture_2d<f32>;\n@fragment fn fs(v: V) -> @location(0) vec4f {');
export const terrain = (cut, clip, bare = false) => !bare ? terrainLit(cut, clip) : terrainLit(cut, clip).replace(/@fragment fn fs[\s\S]*$/,
  `@fragment fn fs(v: V) {${clip ? ' if (town_here(v.w.xz)) { discard; } ' : ''}}\n`);
const terrainLit = (cut, clip) => GRID + /* wgsl */`
@group(1) @binding(0) var<uniform> G: Grid;
@group(1) @binding(1) var heights: texture_2d_array<f32>; // a layer a tile, with a point of its neighbours' round its edge
@group(1) @binding(2) var photos: texture_2d_array<f32>;
@group(1) @binding(3) var samp_photo: sampler;
@group(1) @binding(4) var photo_pages: texture_2d<f32>;
@group(1) @binding(5) var photos1: texture_2d_array<f32>;
@group(1) @binding(6) var photos2: texture_2d_array<f32>;
@group(1) @binding(7) var photo_coarse: texture_2d<f32>;
@group(1) @binding(8) var height_tiles: texture_2d<i32>; // which layer of heights each tile is, -1 for none
@group(1) @binding(9) var<storage, read> lod: array<f32>; // each chunk's lambda this frame (wasm/include/harstad/terrain.h)
${GROUND_SNOW}
struct V { @builtin(position) pos: vec4f, @location(0) w: vec3f, @location(1) n: vec3f }

// Where grid point i's height is, clamped to the grid: in the layer of a tile that holds it, the texel there, with its
// neighbours round it in the same layer. A point on a seam the tile west or north of it holds too, with the same height.
struct Post { at: vec2i, layer: i32 }
fn post(i: vec2i) -> Post {
  let c = clamp(i, vec2i(0), vec2i(G.size) - 1) - G.tile.xy;
  let n = vec2i(textureDimensions(height_tiles));
  let t = clamp(vec2i(floor(vec2f(c) / f32(G.tile.z))), vec2i(0), n - 1);
  let l = c - t * G.tile.z;
  var o = Post(l + 1, -1);
  for (var q = 0; q < 4 && o.layer < 0; q++) {
    let d = vec2i(q & 1, q >> 1);
    let u = t - d;
    if (any((d > vec2i(0)) & (l != vec2i(0))) || any(u < vec2i(0))) { continue; }
    let k = textureLoad(height_tiles, u, 0).r;
    if (k >= 0) { o = Post(l + d * G.tile.z + 1, k); }
  }
  return o;
}
fn height_in(p: Post, d: vec2i) -> f32 { return textureLoad(heights, p.at + d, max(p.layer, 0), 0).r; }
// the backdrop: whether xz is on a tile the map holds, where the town's ground is drawn instead. The tiles' rectangles are
// in its photo_pages, on the tiles' grid from page_at (wasm/src/wasm.c, hs_held)
fn town_here(xz: vec2f) -> bool {
  let t = vec2i(floor((xz - G.page_at) / G.page));
  if (any(t < vec2i(0)) || any(t >= vec2i(textureDimensions(photo_pages)))) { return false; }
  let r = textureLoad(photo_pages, t, 0);
  return xz.x > r.x && xz.y > r.y && xz.x < r.z && xz.y < r.w;
}

// A post's height, and its slope across and down as the fine grid has it (the normal's x and z, over 2 steps up)
fn post_at(i: vec2i) -> vec3f {
  let p = post(i);
  return vec3f(height_in(p, vec2i(0)), height_in(p, vec2i(-1, 0)) - height_in(p, vec2i(1, 0)), height_in(p, vec2i(0, -1)) - height_in(p, vec2i(0, 1)));
}
// The same at grid point i on level L's surface: on the level's triangle that holds it (its cells split as the fine ones
// are, from the post at (1, 0) to the one at (0, 1)), from the triangle's three posts
fn level_at(i: vec2i, L: u32) -> vec3f {
  let s = 1 << L;
  let c = (i >> vec2u(L)) << vec2u(L);
  let f = vec2f(i - c) / f32(s);
  let low = f.x + f.y <= 1.0;
  let o = post_at(select(c + s, c, low)); let b = post_at(c + vec2i(s, 0)); let d = post_at(c + vec2i(0, s));
  return o + (b - o) * select(1.0 - f.y, f.x, low) + (d - o) * select(1.0 - f.x, f.y, low);
}
// Grid point i's lambda: its chunk's; on a chunk's edge the larger of the chunks either side, so both place it alike
fn lambda(i: vec2i) -> f32 {
  let n = vec2i(G.chunks.yz) - 1; let C = i32(G.chunks.x);
  let c = min(i / C, n);
  let e = select(c, max(c - 1, vec2i(0)), (i > vec2i(0)) & (c * C == i));
  let k = vec4i(c.y * (n.x + 1) + c.x, c.y * (n.x + 1) + e.x, e.y * (n.x + 1) + c.x, e.y * (n.x + 1) + e.x);
  return max(max(lod[k.x], lod[k.y]), max(lod[k.z], lod[k.w]));
}

// ii: the row (of the level's rows) and, over TERRAIN_LEVEL_SHIFT, the level of the mesh drawn. Where the point is at
// level 0 it is the fine post; else between its level's surface and the next one's (a geomorph), whatever mesh draws it
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> V {
  let k = ii >> 28u;
  let i = vec2i(i32(vi >> 1u), i32(((ii & 0x0fffffffu) + (vi & 1u)) << k));
  let lam = lambda(i);
  var h: vec3f;
  if (lam <= 0.0) { h = post_at(i); }
  else {
    let L = u32(lam); let f = lam - f32(L);
    h = level_at(i, L);
    if (f > 0.0) { h = mix(h, level_at(i, L + 1u), f); }
  }
  let w = vec3f(G.origin.x + f32(i.x) * G.step, h.x, G.origin.y + f32(i.y) * G.step);
  var o: V;
  o.pos = F.view_proj * vec4f(w, 1.0);
  o.w = w;
  o.n = normalize(vec3f(h.y, 2.0 * G.step, h.z));
  return o;
}

@fragment fn fs(v: V) -> @location(0) vec4f {
  let w = v.w;
  let ddx = dpdx(w.xz); let ddy = dpdy(w.xz);
${clip ? '  if (town_here(w.xz)) { discard; return vec4f(0.0); } // and nothing more is worked out for it\n' : ''}${cut ? `  for (var i = 0; i < 4; i++) { // inside a portal's shell the tunnel is the ground
    let q = w.xz - G.cut_a[i].xy;
    let along = dot(q, G.cut_a[i].zw); let side = dot(q, vec2f(-G.cut_a[i].w, G.cut_a[i].z));
    if (along > 0.0 && along < 16.0 && side > G.cut_b[i].x && side < G.cut_b[i].y && w.y < G.cut_b[i].z + G.cut_b[i].w * along) { discard; return vec4f(0.0); }
  }
` : ''}  let snow = ground_snow(w, ddx, ddy);
  return vec4f(finish(lit_snow(snow.rgb, normalize(v.n), w, 1.0, snow.a, 1.0, ddx, ddy), w), 1.0);
}
`;

/* ---- static meshes ---- */

// walls: a facade texture whose alpha says where the wall colour may tint, and a second that glows. Layers 0 and 1 are
// timber and block; from 2 up, the centre's facade styles, which are drawn at half the size and so have arrays of their own.
//
// Each window's cell of a facade (3.4 m x 2.8 m, a storey) is a room. Its light lives: most rooms hold a steady lamp,
// each its own strength, but some flicker with a television's cuts or a fire, kitchens are whiter, now and then a light
// is switched off for some minutes or a dark room's on, and some windows have a string of Christmas lights along the
// head. Where the glass is clear (the glow's alpha, src/textures.js) the eye looks into the room: a box behind the wall
// as wide as the cell, a storey tall and a few metres deep, followed from the glass to the wall, floor or ceiling the
// ray meets, past a low piece of furniture and a tall one, with the room's lamp in it (interior mapping). The room is
// shaded to the glow it stands in for, as bright on the whole, so that as a window comes under a few pixels it fades
// into the painted glow it was and only the detail goes. All of it from a hash of the window: the corner the wall's
// quad starts at (its first vertex, flat), and the cell's column and row on it.
// The cabin at night is dark: only what lights it shows. The instruments' backlight and the switches' lamps light only
// what faces them, near by (glow: their share; glow_fall: 1/(1 + r^2 fall), r in m); the sky's light comes in only
// through the glass a surface faces (sky: at most this of the sky's light outside); the street lamps, the moon and a
// tunnel's lamps only along their ways in through the glass (cabin_open). Not the car's own beams, which shine ahead
// of it, nor the lamp map's light, which knows nothing of the roof.
const CABIN_DARK = { glow: 0.35, glow_fall: 120, sky: 0.15 };
// What the cabin is made of (tools/car/car.js, FINISH: a vertex's style): how rough each finish is (the sheen's spread),
// its grain (m across a cell of it, and how far the grain tips the surface: leather's pebble, the dash's fine skin), how
// much light it sends back square on (f0), and how much of what is outside the glass it mirrors. Matt is the carpet and
// the cloth; leather the seats, the wheel and the door cards' inserts; soft-touch the dash, the door cards and the trims;
// the gloss black is the centre stack's insert; hard plastic the knobs, the vents and the stalks; the headliner is cloth.
const CABIN_FINISH = [
  { rough: 0.9, cell: 0, tilt: 0, f0: 0.03, mirror: 0 },            // matt
  { rough: 0.48, cell: 0.0018, tilt: 0.16, f0: 0.04, mirror: 0.5 }, // leather
  { rough: 0.6, cell: 0.0011, tilt: 0.12, f0: 0.04, mirror: 0.35 }, // soft-touch vinyl
  { rough: 0.16, cell: 0, tilt: 0, f0: 0.05, mirror: 1 },           // gloss black
  { rough: 0.4, cell: 0, tilt: 0, f0: 0.04, mirror: 0.5 },          // hard plastic
  { rough: 1, cell: 0, tilt: 0, f0: 0.02, mirror: 0 },              // headliner cloth
];
const ROOM = {
  w: 3.4, h: 2.8,                    // a cell of the facades, m
  detail: [3, 8],                    // a metre of wall this many pixels across: the room no more than its mean, then all of it
  string: 0.4, office_string: 0.08,  // the shares of homes' and offices' windows with a string of lights
  tv: 0.16, fire: 0.12, kitchen: 0.12, off: 0.08, on: 0.06, // the shares of lit rooms that do each, and of dark rooms lit a while
  outside: 0.1,                      // the street lamps' light in a room, of what falls on its window
  hall: 0.035,                       // a dark room's light from the hall, of a lit room's
  bright: 0.6,                       // a lit room's light, of what its glow was painted at: evening lamps, not floodlights
  lift: WINDOW_CD / SCENE_CD / 0.65, // a lit window's light, near and far, of what its painted glow gives (some 0.65 units):
                                     // WINDOW_CD (gfx.h, GFX_CD_WINDOW) through its curtains, its Advent star and string with it
  glass: 6.0,                        // a lit pane drawn as a plain face (a shop's display, GFX_GLASS_LIT), of its colour
  many: [0.6, 1.6],                  // a cell's width in pixels under which a pixel's windows are the lit share's, not its one cell's
  star_room: 0.2,                    // the share of homes' lit windows that are a dark room with only an Advent star alight in it
  frame: 0.12,                       // m round a window's opening its light falls on: its casing, sill and reveal, never the wall
};
// the openings, as 256ths of their cell: x0, y0 (up from the floor), x1, y1 (src/textures.js; BLD__OPENING in
// wasm/src/bld.c), for timber, block, classic, brick, ribbon, fins, panel and glass, the boards (none), office, works and clapboard,
// and the ribbon with its spandrels
const OPENINGS = [[84, 76, 172, 198], [53, 78, 203, 190], [72, 54, 184, 204], [72, 54, 184, 204], [4, 66, 252, 186], [16, 26, 228, 216],
  [58, 60, 198, 200], [4, 4, 252, 252], [0, 0, 0, 0], [0, 0, 0, 0], [28, 82, 228, 186], [68, 100, 188, 170], [84, 76, 172, 198], [4, 66, 252, 186]];
// the layers with rooms behind their glass: all but the boards (8 and 9), which have no windows
const ROOMED = 'v.layer < 8u || v.layer >= 10u';
const f4 = x => x.toFixed(4);
const ROOM_FNS = /* wgsl */`
const ROOM_W = ${f4(ROOM.w)};
const ROOM_H = ${f4(ROOM.h)};
const ROOM_WARM = vec3f(1.0, 0.62, 0.30);
const ROOM_COOL = vec3f(0.78, 0.84, 0.9); // an office's fluorescent tubes
fn room_opening(layer: u32) -> vec4f {
  var o = array<vec4f, ${OPENINGS.length}>(${OPENINGS.map(o => `vec4f(${o.map(v => f4(v / 256)).join(', ')})`).join(', ')});
  return o[min(layer, ${OPENINGS.length - 1}u)];
}
// A television: shots of a few seconds, each its own brightness and cast, cut from one to the next, and a faint flicker
fn room_tv(t: f32, seed: f32) -> vec3f {
  let k = floor(t * 0.55 + 0.6 * sin(t * 0.13 + seed) + seed); // a cut every two seconds or so
  let a = fract(sin(k * 12.9898 + seed) * 43758.5453);
  let b = fract(sin(k * 78.233 + seed) * 24634.6345);
  return mix(vec3f(0.45, 0.7, 1.5), vec3f(1.15, 0.95, 0.8), b) * (0.25 + 0.95 * a) * (1.0 + 0.12 * sin(t * 17.0 + seed) * sin(t * 5.3));
}
// What a room's light does over the game's time t, from its cell's mean glow: the colour the glow is multiplied by
// (about 1 on the whole), and in w whether a dark room has a light switched on. Slow, all of it: a quiet evening.
fn room_live(id: u32, t: f32, cell: vec3f, kind: f32) -> vec4f {
  let seed = room_rand(id, 4u) * 64.0;
  let phase = fract(t / (150.0 + 450.0 * room_rand(id, 5u)) + room_rand(id, 6u)); // a switch's minutes, each room its own
  if (max(cell.r, max(cell.g, cell.b)) < 0.015) { // dark, but for an Advent star or candles; in half, a light left on in the hall
    return vec4f(1.0, 1.0, 1.0, select(select(0.0, ${f4(ROOM.hall)}, room_rand(id, 17u) < 0.5), 1.0, kind < ${f4(ROOM.on)} && phase < 0.4));
  }
  if (cell.b > cell.r * 1.4) { return vec4f(room_tv(t, seed), 0.0); } // a television on in a dark room, painted so
  let c = vec3f(0.8 + 0.4 * room_rand(id, 7u)); // each lamp its own strength
  if (cell.b > cell.r * 0.85) { return vec4f(c, 0.0); } // an office, cool and steady
  if (kind < ${f4(ROOM.tv)}) { return vec4f(c * 0.3 + vec3f(0.35, 0.6, 1.4) * room_tv(t, seed) * 0.85, 0.0); }
  if (kind < ${f4(ROOM.tv + ROOM.fire)}) { // a fire: warmer, and never quite still
    return vec4f(vec3f(1.2, 0.75, 0.42) * (0.8 + 0.16 * sin(t * 2.3 + seed) + 0.11 * sin(t * 5.7 + seed * 1.7) + 0.08 * sin(t * 13.1 + seed * 2.3)), 0.0);
  }
  if (kind < ${f4(ROOM.tv + ROOM.fire + ROOM.kitchen)}) { return vec4f(vec3f(0.9, 1.08, 1.6) * 1.2, 0.0); } // a kitchen: whiter, brighter
  if (kind < ${f4(ROOM.tv + ROOM.fire + ROOM.kitchen + ROOM.off)}) { return vec4f(c * select(1.0, 0.03, phase < 0.3), 0.0); } // off a while
  return vec4f(c, 0.0);
}
// A string of lights hung along the head of the window, just inside the glass: warm white, or coloured bulbs each fading
// slowly in its turn. A bulb is a soft spot whose light is kept as it is spread: under a pixel it spreads at the same
// light, and where the bulbs come within a pixel or so of each other they merge into their mean along the string, so
// far off the string is a faint line that neither sparkles nor goes. q: m across and up the cell; px: m a pixel spans.
fn room_string(id: u32, q: vec2f, op: vec4f, t: f32, px: f32) -> vec3f {
  let x0 = op.x * ROOM_W + 0.05; let x1 = op.z * ROOM_W - 0.05; let top = op.w * ROOM_H - 0.04;
  let re = max(0.032, px * 0.7); // the bulb's radius as drawn: its own, or under a pixel a pixel's, its light kept
  if (q.x < x0 - re * 2.0 || q.x > x1 + re * 2.0 || q.y > top + re * 2.0 || q.y < top - 0.2 - re * 2.0) { return vec3f(0.0); }
  let gap = 0.1; let span = x1 - x0; let n = floor(span / gap);
  let sag = 0.03 + 0.06 * room_rand(id, 9u);
  let a = clamp((q.x - x0) / span, 0.0, 1.0);
  let i = clamp(round((q.x - x0) / gap), 0.0, n); let ab = i * gap / span;
  let b = vec2f(x0 + i * gap, top - sag * 4.0 * ab * (1.0 - ab));
  let e = 0.001 / (re * re); // (0.032 / re)^2: the light kept as the spot spreads
  let multi = room_rand(id, 10u) < 0.65; let seed = room_rand(id, 11u) * 40.0;
  var pal = array<vec3f, 4>(vec3f(1.0, 0.1, 0.04), vec3f(1.0, 0.5, 0.06), vec3f(0.12, 0.8, 0.18), vec3f(0.12, 0.3, 1.0));
  let bulb = select(vec3f(1.0, 0.7, 0.36) * (0.85 + 0.15 * sin(t * 0.5 + i * 2.7 + seed)),
                    pal[u32(i) & 3u] * (0.15 + 0.85 * (0.5 + 0.5 * sin(t * 1.6 + i * 1.9 + seed))), multi);
  let mean = select(vec3f(1.0, 0.7, 0.36), vec3f(0.56, 0.42, 0.34) * 0.675, multi);
  // a crisp disc, its edge a pixel soft: no halo round it
  let d = q - b; let spot = bulb * (e * (1.0 - smoothstep(re - px * 0.5, re + px * 0.5, length(d))));
  let yc = top - sag * 4.0 * a * (1.0 - a);
  let along = smoothstep(x0 - re, x0 + re, q.x) * (1.0 - smoothstep(x0 + n * gap - re, x0 + n * gap + re, q.x));
  let line = mean * (e * 1.5708 * re / gap * (1.0 - smoothstep(re - px * 0.5, re + px * 0.5, abs(q.y - yc))) * along);
  return mix(spot, line, smoothstep(0.25, 0.6, re / gap)) * 2.5;
}
// A paper Advent star hung in the window, eight-pointed, lit from within: warm white, or red. Its light is kept as
// it comes under a pixel (spread, not lost), so far off it is a soft point and never sparkles. q: m across and up
// the cell; px: m a pixel spans.
fn room_star(id: u32, q: vec2f, op: vec4f, px: f32) -> vec3f {
  let c = vec2f((op.x + op.z) * 0.5 * ROOM_W, op.w * ROOM_H - 0.3);
  let r0 = 0.17; let d = q - c; let r = length(d);
  let s = max(px, 0.004);
  if (r > r0 + 2.0 * s) { return vec3f(0.0); }
  let t = abs(fract(atan2(d.y, d.x) * (8.0 / 6.2831853) + 0.5) - 0.5) * 2.0; // 0 at a point, 1 between two
  let rim = r0 * mix(1.0, 0.45, t);
  let cover = smoothstep(rim + s, rim - s, r) * min(1.0, (r0 * r0) / (s * s * 4.0 + r0 * r0 * 0.25) * 0.25 + 0.75);
  let red = room_rand(id, 23u) < 0.35;
  let col = select(vec3f(1.0, 0.78, 0.5), vec3f(1.0, 0.22, 0.1), red);
  return col * cover * (1.1 + 0.6 * (1.0 - r / r0));
}
// What glass gives back: a dielectric mirrors by Fresnel's law (Schlick: 4 % head on, all of it at grazing, glass_fresnel) what
// lies along the reflected ray, and lets through the rest. Upwards that is the sky (sky(), the moon's aureole in it) as far as
// the ground's map says the sky is seen that way (sky_seen), the rest the fronts across the street, taken as a wall of albedo
// 0.3 lit by the lamps (reach, lamp_reach) as the snow is; downwards the snow, lit by the moon, the sky and the lamps. No
// light the scene does not have: what it mirrors is what is there.
fn glass_fresnel(ray: vec3f, n: vec3f) -> f32 { return 0.04 + 0.96 * pow5(1.0 - abs(dot(ray, n))); }
fn glass_sheen(ray: vec3f, n: vec3f, reach: vec3f, w: vec3f) -> vec3f {
  let facing = select(-n, n, dot(ray, n) < 0.0);
  let r = reflect(ray, facing);
  let lamps = reach * 1.6;
  let snow = vec3f(0.85) * ((moon_now() * MOON_DIR.y + sky_up(w.xz)) / PI + lamps);
  let wall = vec3f(0.3) * lamps;
  let seen = sky_seen(w, r);
  let env = select(snow, mix(wall, sky(r, 0.0), seen), r.y > 0.0);
  return env * glass_fresnel(ray, n);
}
fn room_box(q: vec2f, lo: vec2f, hi: vec2f, s: f32) -> f32 { // how much of a pixel s across at q falls in the rectangle
  let a = smoothstep(lo - s, lo + s, q) * (1.0 - smoothstep(hi - s, hi + s, q));
  return a.x * a.y;
}
fn room_lamp(p: vec3f, n: vec3f, at: vec3f) -> f32 { // the room's one light on a surface: a little bounced, the rest from the lamp
  let l = at - p; let r2 = dot(l, l);
  return 0.12 + 3.0 * (0.3 + 0.7 * max(dot(n, l) * inverseSqrt(r2), 0.0)) / (1.0 + r2 * 0.6);
}
// The room behind the glass: o where the ray enters it (x across the cell from its left, y up from the floor, z in from
// the glass, m), d the ray in the same frame. s0: m a pixel spans at the glass; sg: how that grows with every metre in.
// Returns in rgb what multiplies the room's glow (mean about 1), in w what multiplies the dark glass for a dark room.
fn room(o: vec3f, d: vec3f, id: u32, kind: f32, office: bool, s0: f32, sg: f32) -> vec4f {
  let depth = 3.0 + 2.5 * room_rand(id, 12u);
  let dd = select(d, vec3f(1e-5), abs(d) < vec3f(1e-5));
  let inv = 1.0 / dd;
  let tx = (select(0.0, ROOM_W, dd.x > 0.0) - o.x) * inv.x;
  let ty = (select(0.0, ROOM_H, dd.y > 0.0) - o.y) * inv.y;
  let tz = (depth - o.z) * inv.z;
  let t = min(tz, min(tx, ty));
  let p = o + d * t; let s = s0 + sg * t;
  // the paint: white, warm white, pale grey, sage, dusty blue, or pine panelling; oak or pale ash underfoot
  let k = room_rand(id, 13u); let r1 = room_rand(id, 14u); let r2 = room_rand(id, 15u); let r3 = room_rand(id, 16u);
  var walls = array<vec3f, 6>(vec3f(0.82, 0.81, 0.78), vec3f(0.84, 0.76, 0.62), vec3f(0.66, 0.67, 0.68), vec3f(0.58, 0.66, 0.52),
                              vec3f(0.52, 0.6, 0.7), vec3f(0.78, 0.52, 0.3));
  let kitchen = kind >= ${f4(ROOM.tv + ROOM.fire)} && kind < ${f4(ROOM.tv + ROOM.fire + ROOM.kitchen)} && !office;
  var wall = walls[min(u32(k * 6.0), 5u)];
  var flr = select(vec3f(0.5, 0.33, 0.19), vec3f(0.72, 0.58, 0.42), r1 < 0.45);
  if (office) { wall = vec3f(0.7, 0.72, 0.74); flr = vec3f(0.3, 0.32, 0.35); }
  if (kitchen) { wall = vec3f(0.84, 0.84, 0.82); }
  let ceil = vec3f(0.82, 0.81, 0.78);
  // the lamp: hung from the ceiling or a standard lamp at the back; a fire or a television sits low
  var lamp = vec3f(ROOM_W * (0.3 + 0.4 * r2), ROOM_H - 0.45, depth * (0.35 + 0.3 * r3));
  if (r3 > 0.6 && !office) { lamp = vec3f(select(0.45, ROOM_W - 0.45, r2 > 0.5), 1.45, depth - 0.45); }
  if (kind < ${f4(ROOM.tv + ROOM.fire)} && !office) { lamp = vec3f(select(0.6, ROOM_W - 0.6, r1 > 0.5), 0.6, depth - 0.3); }
  var alb: vec3f; var n: vec3f;
  if (t == tz) {
    n = vec3f(0.0, 0.0, -1.0); alb = wall;
    if (kitchen) { // tiles over the worktop, cupboards above
      let joint = (1.0 - smoothstep(0.0, 0.01 + s, min(fract(p.x / 0.15), fract(p.y / 0.15)) * 0.15)) * (1.0 - smoothstep(0.01, 0.03, s));
      alb = mix(alb, vec3f(0.9, 0.9, 0.88) * (1.0 - 0.15 * joint), room_box(p.xy, vec2f(0.0, 0.9), vec2f(ROOM_W, 1.5), s));
      alb = mix(alb, select(vec3f(0.85, 0.83, 0.78), vec3f(0.42, 0.5, 0.45), r1 > 0.6), room_box(p.xy, vec2f(0.0, 1.55), vec2f(ROOM_W, 2.3), s));
    } else if (!office) { // a picture, or a mirror over the sofa
      let c = vec2f(ROOM_W * (0.3 + 0.4 * r1), 1.55 + 0.15 * r3); let h = vec2f(0.3 + 0.25 * r2, 0.22 + 0.15 * r1);
      alb = mix(alb, vec3f(0.12, 0.1, 0.08), room_box(p.xy, c - h - 0.03, c + h + 0.03, s));
      alb = mix(alb, mix(vec3f(0.55, 0.35, 0.25), vec3f(0.3, 0.42, 0.55), r3), room_box(p.xy, c - h, c + h, s));
    }
  } else if (t == tx) {
    n = vec3f(-sign(dd.x), 0.0, 0.0); alb = wall * 0.94;
  } else if (dd.y < 0.0) {
    n = vec3f(0.0, 1.0, 0.0); alb = flr;
    if (!office && !kitchen) { alb = mix(alb, mix(vec3f(0.6, 0.18, 0.12), vec3f(0.75, 0.7, 0.62), r2), room_box(p.xz, vec2f(0.6, depth * 0.3), vec2f(ROOM_W - 0.6, depth * 0.8), s)); }
  } else {
    n = vec3f(0.0, -1.0, 0.0); alb = ceil;
  }
  if (k > 0.83 && !office && !kitchen && n.y == 0.0) { // pine panelling's standing boards
    alb *= 1.0 - 0.18 * (1.0 - smoothstep(0.0, 0.012 + s, fract((p.x + p.z) / 0.12) * 0.12)) * (1.0 - smoothstep(0.02, 0.05, s));
  }
  var col = alb * room_lamp(p, n, lamp);
  if (office && n.y < 0.0) { col += vec3f(2.2) * room_box(fract(p.xz / 1.8) * 1.8, vec2f(0.3, 0.3), vec2f(1.5, 0.9), s); } // ceiling panels
  var dark = dot(alb, vec3f(0.33)) * (0.45 + 0.9 * (1.0 - p.z / depth) * (1.0 - p.z / depth));
  // a low piece along the back (a sofa, a sideboard, desks with their screens), and a tall one at a side (shelves, a cupboard)
  let tf = (depth - 0.45 - 0.5 * r2) * inv.z;
  if (tf < t) {
    let q = o + d * tf; let sq = s0 + sg * tf;
    let x0 = 0.35 + 0.8 * r3; let x1 = x0 + 1.4 + 0.6 * r1;
    let top = select(select(0.85, 0.75, r2 > 0.5), 0.92, kitchen || office);
    var cov = room_box(q.xy, vec2f(select(x0, -1.0, office || kitchen), -1.0), vec2f(select(x1, ROOM_W + 1.0, office || kitchen), top), sq);
    let fc = select(mix(vec3f(0.35, 0.3, 0.27), vec3f(0.52, 0.2, 0.15), r1), vec3f(0.45, 0.44, 0.42), office || kitchen);
    var fcol = fc * room_lamp(q, vec3f(0.0, 0.0, -1.0), lamp);
    if (office) {
      let screen = room_box(vec2f(fract(q.x / 0.9) * 0.9, q.y), vec2f(0.25, top + 0.08), vec2f(0.6, top + 0.38), sq);
      fcol = mix(fcol, vec3f(0.35, 0.6, 1.2), screen); cov = max(cov, screen);
    }
    col = mix(col, fcol, cov); dark = mix(dark, dot(fc, vec3f(0.33)) * 0.6, cov);
  }
  let tt = (depth - 0.3) * inv.z;
  if (tt < t && !kitchen) {
    let q = o + d * tt; let sq = s0 + sg * tt;
    let x0 = select(0.12, ROOM_W - 0.92, r3 > 0.5);
    let cov = room_box(q.xy, vec2f(x0, -1.0), vec2f(x0 + 0.8, 1.95), sq);
    var fc = select(vec3f(0.62, 0.44, 0.28), vec3f(0.8, 0.79, 0.76), r1 > 0.5);
    let shelf = fract(q.y / 0.38);
    let books = mix(vec3f(0.4, 0.15, 0.1), vec3f(0.15, 0.25, 0.4), fract(sin(floor(q.x / 0.045) * 91.7 + floor(q.y / 0.38)) * 4375.85));
    fc = mix(fc, books, select(0.0, (1.0 - smoothstep(0.03, 0.06, sq)) * step(0.12, shelf) * step(shelf, 0.8), r2 < 0.6));
    col = mix(col, fc * room_lamp(q, vec3f(0.0, 0.0, -1.0), lamp), cov); dark = mix(dark, dot(fc, vec3f(0.33)) * 0.6, cov);
  }
  // the lamp's shade, alight
  let sl = dot(lamp - o, d);
  if (sl > 0.0 && sl < t && !office) {
    let sq = s0 + sg * sl;
    col += vec3f(3.0, 2.4, 1.6) * (1.0 - smoothstep(0.1 - sq, 0.13 + sq, length(o + d * sl - lamp))) * 0.6;
  }
  // as bright on the whole as the glow it stands in for: the room's mean colour under its mean light
  let mean = (wall * 0.55 + flr * 0.25 + ceil * 0.2) * select(0.75, 1.1, office);
  return vec4f(col / mean, dark / (dot(mean, vec3f(0.33)) * 0.85));
}
`;
// ---- lamps: the chamber behind each lens ----
// A lamp's lens is a window onto its chamber, traced as the rooms behind the windows are (interior mapping). Under the
// outer glass, an inner lens of flutes: each a thin cylinder lens that bends the ray across it and gathers the light into
// a line. Behind that the reflector, cut into cells: a honeycomb of small chrome bowls in a tail lamp or an indicator, one
// deep bowl to each bulb of a headlamp, the bulb at its focus with a cap over its tip. Lit, each bowl is bright where it
// sends its bulb's light to the eye, which shifts over it as the eye moves, and the ridges between the bowls are darker;
// out, the chrome mirrors what comes in through the lens, tinted by it, and the structure shows as it does on a parked
// car. Nothing round the lens glows: what the lamp throws on the scene is its real light (car_beams, car_tail,
// traffic_lamps). Where its cells come under a few pixels the chamber is its mean, so it never sparkles off far away.
// A tail lamp's cells are small and shallow, its ridges lit nearly as its bowls: one smooth red lens with a fine optic in
// it from a few metres, as the T22's is, not a grid of blocks. Styles: 1 a tail and brake lamp, red; 2 an indicator, amber; 3 a reversing lamp, clear; 4 a headlamp's reflector,
// a bulb in a deep bowl behind clear glass; 5 the high-level brake lamp's fine red cells; 6 a fog lamp, a small headlamp.
const LENS = {
  off: 0.14,   // an unlit lens's level (wasm/src/avensis.c, LAMP_OFF): what is over it is light
  bulb: 6,     // a bulb's glass seen straight through the lens, of its lamp's light
};
const LAMP_FNS = /* wgsl */`
// flute: period, strength, 1 for flutes along u; glow: a lit bowl's light where it does not send its bulb's to the eye, and
// where it does, of its lamp's, and how tightly its facets send it; facet: how far each is turned its own way
struct Lens { cell: vec2f, tint: vec3f, depth: f32, flute: vec3f, bulb: f32, glow: vec3f, facet: f32 }
fn lens_style(s: u32) -> Lens {
  switch s {
    case 2u: { return Lens(vec2f(0.03, 0.03), vec3f(1.0, 0.42, 0.05), 0.03, vec3f(0.0042, 0.5, 1.0), 0.0, vec3f(0.3, 1.7, 5.0), 0.22); }
    case 3u: { return Lens(vec2f(0.03, 0.03), vec3f(0.86, 0.88, 0.9), 0.03, vec3f(0.0042, 0.4, 0.0), 0.0, vec3f(0.3, 1.7, 5.0), 0.22); }
    case 4u: { return Lens(vec2f(0.125, 0.118), vec3f(0.92, 0.94, 0.97), 0.075, vec3f(0.009, 0.06, 0.0), 0.012, vec3f(0.06, 0.65, 10.0), 0.06); }
    case 5u: { return Lens(vec2f(0.02, 0.018), vec3f(0.62, 0.025, 0.02), 0.016, vec3f(0.003, 0.45, 0.0), 0.0, vec3f(0.3, 1.7, 5.0), 0.22); }
    case 6u: { return Lens(vec2f(0.072, 0.072), vec3f(0.95, 0.94, 0.85), 0.04, vec3f(0.005, 0.3, 1.0), 0.007, vec3f(0.06, 0.8, 10.0), 0.08); }
    default: { return Lens(vec2f(0.012, 0.011), vec3f(0.62, 0.025, 0.02), 0.012, vec3f(0.0045, 0.35, 0.0), 0.0, vec3f(0.6, 0.85, 2.5), 0.12); }
  }
}
// The chamber as the eye sees it through the lens at p (m on the lens, its cells laid out from 0), along rd: the ray in
// the lens's frame, x along u, y along v, z in (rd.z > 0). light: what the lamp burns at, its lit lens's colour times its
// level, 0 when out; env: the light that comes in through the lens (what lights a white surface there); foot: metres a
// pixel spans there.
fn lamp_chamber(p: vec2f, rd_in: vec3f, L: Lens, light: vec3f, env: vec3f, foot: f32) -> vec3f {
  let cell = L.cell;
  let mean = light * (L.glow.x + 0.35 * L.glow.y) + env * L.tint * 0.2;
  let seen = 1.0 - smoothstep(0.06, 0.25, foot / min(cell.x, cell.y));
  if (seen <= 0.0${OFF.has('chambers') ? ' || true' : ''}) { return mean; }
  // the outer glass bends the ray in towards its normal, n 1.5
  var rd = normalize(vec3f(rd_in.xy / 1.5, max(rd_in.z, 0.05)));
  let fl = 1.0 - smoothstep(0.25, 0.6, foot / L.flute.x);
  let across = L.flute.z > 0.5;
  let ph = select(p.x, p.y, across) * (6.2831853 / L.flute.x);
  let bend = sin(ph) * L.flute.y * fl * 0.5;
  rd = normalize(rd + select(vec3f(bend, 0.0, 0.0), vec3f(0.0, bend, 0.0), across));
  let gather = 1.0 + 0.5 * L.flute.y * fl * cos(ph);
  // the cell's bowl: a sphere through the cell's corners at the lens, its foot \`depth\` behind it
  // a tail lamp's rows each half a cell along from the last, as moulded honeycombs are; a headlamp's bowls side by side
  let row = floor(p.y / cell.y);
  let q = p - vec2f(select(0.5 * cell.x * (row - 2.0 * floor(row * 0.5)), 0.0, L.bulb > 0.0), 0.0);
  let c = floor(q / cell);
  let o = vec3f(q - (c + 0.5) * cell, 0.0);
  let R = (dot(cell, cell) * 0.25 + L.depth * L.depth) / (2.0 * L.depth);
  let z0 = L.depth - R;
  let oc = o - vec3f(0.0, 0.0, z0);
  let b = dot(oc, rd);
  let t = -b + sqrt(max(b * b - dot(oc, oc) + R * R, 0.0));
  let h = o + rd * t;
  // the ray leaves the cell first where it meets the ridge between two bowls
  let hw = cell * 0.5;
  let tw = min((hw.x - o.x * sign(rd.x)) / max(abs(rd.x), 1e-5), (hw.y - o.y * sign(rd.y)) / max(abs(rd.y), 1e-5));
  var lit: vec3f; var off: vec3f;
  if (tw < t) {
    lit = light * (L.glow.x * 1.2) * gather;
    off = env * L.tint * 0.08;
  } else {
    // the bowl's facets, each turned a little its own way: twelve round it, four out from its middle
    var nh = normalize(vec3f(0.0, 0.0, z0) - h);
    let fac = 1.0 - smoothstep(0.15, 0.4, foot / (R * 0.35));
    let id = vec2f(floor(atan2(h.y, h.x) * 1.9099), floor(length(h.xy) / R * 4.0));
    let j = vec2f(hash2(id + c * 7.31), hash2(id.yx + c * 3.17 + 5.0)) - 0.5;
    nh = normalize(nh + vec3f(j * L.facet * fac, 0.0));
    let bulb = vec3f(0.0, 0.0, z0 + R * 0.5); // at the focus
    let r = reflect(rd, nh);
    let lobe = pow(max(dot(r, normalize(bulb - h)), 0.0), L.glow.z);
    lit = light * (L.glow.x + L.glow.y * lobe) * gather;
    // out, the chrome mirrors the lens and what is past it where it looks back out, the dark chamber where it looks in
    off = env * L.tint * (0.1 + 0.22 * smoothstep(0.2, -0.8, r.z)) * (0.6 + 0.4 * gather);
    if (L.bulb > 0.0) { // a headlamp's bulb, its tip capped, seen through the glass; the bowl round it lit by it
      let tb = dot(bulb - o, rd);
      if (tb > 0.0 && tb < t) {
        let tc = (bulb.z - L.bulb * 0.8 - o.z) / rd.z;
        let hc = o + rd * tc;
        if (length(hc.xy - bulb.xy) < L.bulb * 0.55) { // the cap: chrome, mirroring the lit bowl
          lit = light * 0.8; off = env * 0.4;
        } else {
          let k = smoothstep(L.bulb, L.bulb * 0.3, length(o + rd * tb - bulb));
          lit += light * (${LENS.bulb.toFixed(1)} * k); off += env * (0.5 * k);
        }
      }
    }
  }
  return mix(mean, lit + off, seen);
}
// A lens of the traffic or of a parked car (GFX_LENS_*): a rectangle size.x by size.y, laid out from its first corner,
// its cells fitted to it: a headlamp a bowl or two, the rest a honeycomb of them. q: the point from that corner.
const GFX_LENS_HEAD = 9.0; // wasm/include/harstad/gfx.h: a fleet lens, its colour's bytes its size in cm
const GFX_LENS_LAMP = 14.0; // a lens a mesh draws (a lamp's head, a porch lamp): its colour's hue at GFX_CD_LAMP, as its glow sprite draws it far off (lens_burn)
// The style of a fleet lens's chamber by its shading: a dipped headlamp, a tail lamp, an indicator, a reversing lamp (its
// light, lit, lamp_true)
fn lamp_flat_style(flag: f32) -> u32 { return select(select(select(4u, 1u, flag > 9.5), 2u, flag > 10.5), 3u, flag > 11.5); }
fn lamp_flat(style: u32, size: vec2f, q: vec2f, rd: vec3f, light: vec3f, env: vec3f, foot: f32) -> vec3f {
  var L = lens_style(style);
  if (style == 4u) { L.cell = vec2f(size.x / max(1.0, round(size.x / size.y)), size.y); L.depth = 0.6 * size.y; }
  else { L.cell = size / max(vec2f(1.0), round(size / L.cell)); }
  return lamp_chamber(abs(q), rd, L, light, env, foot);
}
// A traffic lamp's light towards the eye, of its peak: to the eye from it in the car's space (x across, y up, z forward),
// a headlamp's dipped beam or a tail lamp's spread, as gfx_head_toward and gfx_tail_toward have them (wasm/src/gfx.c)
fn lamp_toward(to: vec3f, tail: bool) -> f32 {
  let a = select(to.z, -to.z, tail); let flat = length(vec2f(a, to.x));
  if (flat + abs(to.y) <= 0.0) { return 0.0; }
  let h = atan2(abs(to.x), a); let v = atan2(to.y, flat);
  if (tail) {
    return (1.0 - smoothstep(1.5, 1.9, h)) * (0.1 + 0.9 * exp(-h * h / 0.3025)) * (0.15 + 0.85 * exp(-v * v / 0.0441));
  }
  let across = 0.85 * exp(-h * h / 0.09) + 0.15 * exp(-h * h / 0.81);
  let glare = 0.03 / (1.0 + v * v / 0.0144);
  let e = (v - 0.00995) / 0.0045; let low = min(v + 0.0175, 0.0) / 0.15;
  let up_down = select(glare + (1.0 - glare) * exp(-e * e), exp(-low * low), v <= 0.00995); // GFX_HEAD_CUT, GFX_HEAD_GLARE
  return (1.0 - smoothstep(1.6, 2.0, h)) * max(across * up_down, 0.002); // GFX_HEAD_STRAY
}
// The light a lit traffic lamp's chamber is drawn with (lamp_flat's light) for its lens to burn as it truly does towards
// the eye: so that the chamber's mean is the lens's true radiance, as the glow sprite draws it (GLOW: its disc, and its
// point under a pixel). burn: the lamp's level, a tail lamp's over 1 braking (game.c, TRAFFIC_TAIL_BRAKE)
fn lamp_true(style: u32, to: vec3f, tail: bool, burn: f32) -> vec3f {
  let g = lens_style(style).glow;
  let c = select(${vec3(LENS_TRUE.head)}, ${vec3(LENS_TRUE.tail)} * select(${LENS_TRUE.running.toFixed(2)}, 1.0, burn > 1.5), tail);
  return c * (lamp_toward(to, tail) * min(burn, 1.0) / (g.x + 0.35 * g.y));
}
`;
// ---- the glass of everything else: windows drawn as plain faces, boats' wheelhouses, cars' cabins ----
// What is drawn as a plain face of glass says so in its shading (wasm/include/harstad/gfx.h): GFX_GLASS dark, GFX_GLASS_LIT
// lit from within (its colour the room's light), GFX_CABIN a car's, GFX_CAB the truck's. Behind a window is a room as
// the walls' are (room()), laid out from the face's first corner, which soup_quad draws both its triangles from: the
// window's sill 0.6 m over the room's floor, the room 3.4 m along the wall from 0.9 m short of the window's edge, a
// partition where a long run of glass passes into the next. Behind a wheelhouse's panes a console with its screens,
// a wheel and a seat; behind a car's glass its seats and headrests, the dashboard and the wheel, and in the traffic a
// seat, empty (no people in the game). All of it fades into its mean as a metre of it comes under 8 pixels, gone by 3, as the rooms do.
const CAR_GLASS = [0.011, 0.016, 0.024]; // fleet.c's glass, 0x1c232b
const GLASS = {
  sill: 0.6, edge: 0.9,   // m: a plain window's sill over its room's floor, and its edge in from the room's wall
  outside: 0.1,           // the lamps' light in a room or a cabin, of what falls on its glass
  cabin: 0.07,            // a cabin's mean albedo, which its glass shows far off
  detail: [3, 8],
};
const GLASS_FNS = ROOM_FNS + LAMP_FNS + /* wgsl */`
const GFX_GLASS = 4.0; const GFX_GLASS_LIT = 5.0; const GFX_CABIN = 6.0; const GFX_CAB = 7.0; const GFX_GLASS_SHOP = 13.0;
struct CabinHit { t: f32, n: vec3f, alb: vec3f }
// the ray o + d t into the box lo..hi from outside it, if nearer than what h holds: its albedo and the face it meets
fn cabin_box(o: vec3f, inv: vec3f, lo: vec3f, hi: vec3f, alb: vec3f, h: ptr<function, CabinHit>) {
  let t0 = (lo - o) * inv; let t1 = (hi - o) * inv;
  let tn = min(t0, t1); let tf = max(t0, t1);
  let a = max(tn.x, max(tn.y, tn.z)); let b = min(tf.x, min(tf.y, tf.z));
  if (a <= b && a > 0.0 && a < (*h).t) {
    (*h).t = a; (*h).alb = alb;
    (*h).n = -sign(inv) * select(select(vec3f(0.0, 0.0, 1.0), vec3f(0.0, 1.0, 0.0), a == tn.y), vec3f(1.0, 0.0, 0.0), a == tn.x);
  }
}
// and from inside it: where the ray leaves, and through which face
fn cabin_shell(o: vec3f, inv: vec3f, lo: vec3f, hi: vec3f, alb: vec3f) -> CabinHit {
  let t0 = (lo - o) * inv; let t1 = (hi - o) * inv;
  let tf = max(t0, t1); let b = min(tf.x, min(tf.y, tf.z));
  var h: CabinHit; h.t = b; h.alb = alb;
  h.n = sign(inv) * -select(select(vec3f(0.0, 0.0, 1.0), vec3f(0.0, 1.0, 0.0), b == tf.y), vec3f(1.0, 0.0, 0.0), b == tf.x);
  return h;
}
// a ring (a wheel) in the plane z = at.z facing along z, centre at.xy, radius r: if the ray meets it nearer than h
fn cabin_ring(o: vec3f, d: vec3f, at: vec3f, r: f32, alb: vec3f, h: ptr<function, CabinHit>) {
  let t = (at.z - o.z) / d.z;
  if (t > 0.0 && t < (*h).t && abs(length(o.xy + d.xy * t - at.xy) - r) < 0.025) { (*h).t = t; (*h).alb = alb; (*h).n = vec3f(0.0, 0.0, -sign(d.z)); }
}
// A car's cabin in its own space (wasm/src/fleet.c: x across, the driver's side negative; y up; z forward): the seats and
// their headrests, the rear bench, the dashboard and the wheel, and nobody in it (no people in the game); the truck's is the same raised and widened
// into its cab. Lit by what comes in from outside (light) and faintly by the instruments. Returns its colour.
fn car_cabin(o0: vec3f, d0: vec3f, truck: bool, light: vec3f) -> vec3f {
  var o = o0; var d = d0;
  if (truck) { o = vec3f(o0.x / 1.45, o0.y - 0.85, o0.z - 0.5); d = vec3f(d0.x / 1.45, d0.y, d0.z); }
  let dd = select(d, vec3f(1e-5), abs(d) < vec3f(1e-5)); let inv = 1.0 / dd;
  var h = cabin_shell(o, inv, vec3f(-0.74, 0.3, -2.4), vec3f(0.74, 1.5, 1.3), vec3f(0.05, 0.05, 0.055));
  if (h.n.y < -0.5) { h.alb = vec3f(0.16, 0.155, 0.15); } // the headlining
  let seat = vec3f(0.07, 0.068, 0.072);
  for (var k = 0; k < 2; k++) {
    let c = select(-0.37, 0.37, k == 1);
    cabin_box(o, inv, vec3f(c - 0.25, 0.45, -0.46), vec3f(c + 0.25, 1.08, -0.3), seat, &h);
    cabin_box(o, inv, vec3f(c - 0.12, 1.12, -0.44), vec3f(c + 0.12, 1.3, -0.34), seat, &h);
    cabin_box(o, inv, vec3f(c - 0.25, 0.35, -0.46), vec3f(c + 0.25, 0.55, 0.1), seat, &h);
  }
  if (!truck) {
    cabin_box(o, inv, vec3f(-0.66, 0.45, -1.48), vec3f(0.66, 0.98, -1.32), seat, &h);
    cabin_box(o, inv, vec3f(-0.66, 0.35, -1.32), vec3f(0.66, 0.55, -0.85), seat, &h);
  }
  cabin_box(o, inv, vec3f(-0.74, 0.55, 0.58), vec3f(0.74, 0.93, 1.3), vec3f(0.035), &h); // the dashboard
  cabin_ring(o, dd, vec3f(-0.37, 0.98, 0.45), 0.17, vec3f(0.03), &h);
  let p = o + dd * h.t;
  let dash = vec3f(0.06, 0.09, 0.05) / (1.0 + dot(p - vec3f(-0.37, 0.9, 0.55), p - vec3f(-0.37, 0.9, 0.55)) * 6.0);
  return h.alb * (light * (0.55 + 0.45 * max(h.n.y, 0.0)) + dash);
}
// A wheelhouse behind a boat's pane, in the pane's own frame (x along it, y up from its foot, z in): a console under
// the forward windows with its screens alight, a wheel and a seat, a lamp under the roof when the boat is lit (lamp,
// its colour; zero when not). Returns its colour, under light from outside and the lamp.
fn wheelhouse(o: vec3f, d: vec3f, light: vec3f, lamp: vec3f) -> vec3f {
  let dd = select(d, vec3f(1e-5), abs(d) < vec3f(1e-5)); let inv = 1.0 / dd;
  var h = cabin_shell(o, inv, vec3f(0.0, -1.0, 0.0), vec3f(2.0, 1.2, 1.8), vec3f(0.36, 0.27, 0.18)); // varnished boards
  cabin_box(o, inv, vec3f(0.2, -1.0, 1.25), vec3f(1.8, 0.05, 1.8), vec3f(0.08, 0.085, 0.09), &h);
  cabin_box(o, inv, vec3f(0.7, -1.0, 0.55), vec3f(1.3, -0.5, 0.95), vec3f(0.12, 0.1, 0.09), &h);
  cabin_box(o, inv, vec3f(0.7, -0.5, 0.9), vec3f(1.3, 0.1, 1.0), vec3f(0.12, 0.1, 0.09), &h);
  cabin_ring(o, dd, vec3f(1.0, 0.25, 1.15), 0.18, vec3f(0.05), &h);
  let p = o + dd * h.t;
  let l = vec3f(1.0, 1.0, 0.9) - p; let r2 = dot(l, l);
  var c = h.alb * (light * (0.55 + 0.45 * max(h.n.y, 0.0)) + lamp * (0.3 + 2.0 * max(dot(h.n, l) * inverseSqrt(r2), 0.0) / (1.0 + r2)));
  if (h.t == (1.25 - o.z) * inv.z && p.y > -0.3 && fract(p.x * 1.6) < 0.7) { c += vec3f(0.1, 0.4, 0.3) * 0.25 * min(lamp.r, 1.0); } // its screens, on when it is lit
  return c;
}
// A window drawn as a plain face, at w with normal n, its first corner start: the frame of the room behind it, the
// room's hash, and m across, up and in. along: which way along the wall the window runs from its first corner.
struct GlassAt { o: vec3f, d: vec3f, id: u32 }
fn glass_at(w: vec3f, n: vec3f, start: vec3f, ray: vec3f, cw: f32, ch: f32, edge: f32, sill: f32) -> GlassAt {
  let nh = normalize(vec3f(n.x, 0.0, n.z)); let side = vec2f(-nh.z, nh.x);
  let s = dot(w.xz - start.xz, side); let sg = select(-1.0, 1.0, s >= 0.0);
  let x = abs(s) + edge; let y = w.y - start.y + sill;
  let cell = floor(vec2f(x / cw, y / ch));
  var g: GlassAt;
  g.o = vec3f(x - cell.x * cw, y - cell.y * ch, 0.0);
  g.d = vec3f(dot(ray.xz, side) * sg, ray.y, max(-dot(ray, nh), 0.02));
  g.id = room_hash(bitcast<u32>(start.x) ^ room_hash(bitcast<u32>(start.y) ^ room_hash(bitcast<u32>(start.z)
    ^ room_hash(u32(i32(cell.x) + 4096) ^ (u32(i32(cell.y) + 4096) << 13u)))));
  return g;
}
// A parked car's glass after a night out: rimed over, the windscreen and the rear window more than the side windows, in
// feathers that grow across it; a few cars were driven since the frost came, or scraped, and are clear. p: the point in
// the car's space; id: the car's own number, from where it stands; foot: metres a pixel spans, past which the feathers
// even out to their mean.
fn cabin_frost(n: vec3f, p: vec3f, id: u32, foot: f32) -> f32 {
  let h = room_rand(id, 7u);
  if (h < 0.3) { return 0.0; }
  let q = vec2f(p.x * 0.7 + p.z, p.y + 0.37 * p.x);
  let fern = mix(0.5, noise(q * 7.0) * 0.6 + noise(q * 19.0 + 3.1) * 0.4, 1.0 - smoothstep(0.012, 0.04, foot));
  return mix(0.6, 1.0, smoothstep(0.15, 0.5, abs(n.y))) * smoothstep(0.2, 0.55, fern + (h - 0.3) * 0.6);
}
// What the eye sees in a window drawn as a plain face: shaded, the glass as it was lit (its painted colour under the
// moon and the lamps); for a lit one its colour is the room's light. kind: the face's shading. Returns the colour.
fn plain_glass(w: vec3f, n: vec3f, start: vec3f, col: vec3f, shaded: vec3f, kind: f32) -> vec3f {
  let eye = w - F.cam_pos.xyz; let dist = length(eye); let ray = eye / dist;
  let pa = pixel_angle();
  let reach = lamp_reach(w);
  let detail = smoothstep(${GLASS.detail[0].toFixed(1)}, ${GLASS.detail[1].toFixed(1)}, 1.0 / (dist * pa)) * step(abs(n.y), 0.7);
  let shown = 1.0 - glass_fresnel(ray, n); // what the glass does not mirror it lets through
  let outside = reach * ${GLASS.outside.toFixed(3)} + moon_now() * 0.05 + sky_up(w.xz) * 0.04;
  let sheen = glass_sheen(ray, n, reach, w);
  if (kind == GFX_CABIN) { // a parked car's: its cabin as the traffic's (car_cabin), empty; col is the point in the
    // car's space and start.x its heading, as the vertex shader unpacked them (CAR_GLASS)
    let th = start.x * ${(2 * Math.PI * 255 / 256).toFixed(6)};
    let f = vec3f(sin(th), 0.0, cos(th)); let r = vec3f(-f.z, 0.0, f.x);
    let rd = vec3f(dot(ray, r), ray.y, dot(ray, f));
    let light = reach * 0.6 + moon_now() * 0.3 + sky_up(w.xz) * 0.3;
    var inside = light * ${GLASS.cabin.toFixed(3)};
    let near = smoothstep(${GLASS.detail[0].toFixed(1)}, ${GLASS.detail[1].toFixed(1)}, 1.0 / (dist * pa));
    if (near > 0.0) { inside = mix(inside, car_cabin(col, rd, false, light), near); }
    let clear = shaded * 0.5 + inside * shown + sheen;
    // where the car stands: this point less where it is in the car, which every pane of one car shares
    let at = w - r * col.x - f * col.z;
    let frost = cabin_frost(n, col, room_hash(bitcast<u32>(i32(floor(at.x))) ^ room_hash(bitcast<u32>(i32(floor(at.z))))), dist * pa);
    if (frost <= 0.0${OFF.has('rime') ? ' || true' : ''}) { return clear; }
    // the rime on it lit as the glass was: its colour is CAR_GLASS, dark, so no snow was judged on it and what lit it is shaded over it
    let lit_by = shaded / vec3f(${CAR_GLASS.join(', ')});
    let iced = RIME_ALBEDO * lit_by + rime_glints(w, n, reach * lamp_tilt(n) * 1.6, car_lamps(w, n) / PI, dist);
    return mix(clear, iced, frost);
  }
  let lit = kind == GFX_GLASS_LIT || (kind == GFX_GLASS_SHOP && F.town_now.w > 0.5); // a shop's display while the shops' windows are lit
  // lit: its true light, near and far (ROOM.glass)
  let true_light = ${f4(ROOM.glass)};
  var inside = select(outside, col * true_light, lit);
  var glass = select(shaded, vec3f(0.0), lit);
  if (detail > 0.0) {
    let g = glass_at(w, n, start, ray, ROOM_W, ROOM_H, ${GLASS.edge.toFixed(2)}, ${GLASS.sill.toFixed(2)});
    let kindr = room_rand(g.id, 3u);
    let r = room(g.o, g.d, g.id, kindr, false, dist * pa, pa);
    inside = mix(inside, inside * select(vec3f(r.w), r.rgb, lit), detail);
  }
  return glass + inside * shown + sheen;
}
`;

const MESH_IN = /* wgsl */`
struct VIn { @location(0) pos: vec3f, @location(1) nrm: vec4f, @location(2) col: vec4f }
struct V { @builtin(position) pos: vec4f, @location(0) w: vec3f, @location(1) n: vec3f, @location(2) col: vec3f, @location(3) @interpolate(flat) flag: f32,
  @location(4) @interpolate(flat) start: vec3f } // the face's first corner, for what is behind glass (GLASS_FNS)
`;

// vertex colour, lamp-lit; flag 1 is unlit (lit windows, porch lamps, lenses), 2 unlit and above the haze (lights in the
// sky), 4 to 6 glass with a room or a car's seats behind it (GLASS_FNS). A parked car's glass (6) has no colour of its
// own: its colour's bytes are where it is in the car (wasm/include/harstad/gfx.h, GFX_CABIN_X), its alpha the car's heading.
// glazed: whether it draws glass and lenses (PIPE_GLAZED's faces, gfx.h) or only what one does not see into (PIPE_PLAIN's):
// with what only glass and lenses run in it, every plain pixel shaded several times slower, its registers taken up for it.
const plain = glazed => GLASS_FNS + MESH_IN + /* wgsl */`
@vertex fn vs(i: VIn) -> V {
  var o: V;
  o.pos = F.view_proj * vec4f(i.pos, 1.0);
  o.w = i.pos; o.n = i.nrm.xyz; o.col = unpack(i.col); o.flag = round(i.nrm.w * 127.0); o.start = i.pos;
  if (o.flag > GFX_LENS_HEAD - 0.5 && o.flag < GFX_LENS_HEAD + 3.5) { o.col = i.col.rgb * 2.55; } // a lens's size, m (13: a shop's glass)
  if (o.flag == GFX_CABIN) {
    o.col = (i.col.rgb - vec3f(0.5, 0.0, 0.5)) * vec3f(${(2 * 1.0).toFixed(2)}, ${(2.0).toFixed(2)}, ${(2 * 2.56).toFixed(2)});
    o.start = vec3f(i.col.a, 0.0, 0.0);
  }
  return o;
}
@fragment fn fs(v: V) -> @location(0) vec4f {
  rime_on = select(0.0, ${RIME.plain.toFixed(2)}, v.flag < 0.5); // what is lamp-lit stands out in the night: fences, posts, rails, parked cars
  let lens = v.flag > GFX_LENS_HEAD - 0.5 && v.flag < GFX_LENS_HEAD + 3.5;
  let burns = v.flag > GFX_LENS_LAMP - 0.5 && v.flag < GFX_LENS_LAMP + 0.5;
  // lit only where it is used: not on what is its own light (flag 1, lit windows and porch lamps; 2, lights above the
  // haze; the wires): they were lit and the light thrown away, on every pixel of every lit window in the town
  let dx = dpdx(v.w.xz); let dy = dpdy(v.w.xz);
  var shaded = vec3f(0.0);
  if (v.flag < 0.5 || (v.flag > 2.5 && !burns)) { shaded = lit_snow(select(select(v.col, vec3f(${CAR_GLASS.join(', ')}), v.flag == GFX_CABIN), vec3f(1.0), lens), normalize(v.n), v.w, 1.0, -1.0, 1.0, dx, dy); }
  if (lens${glazed ? '' : ' && false'}) { // a parked car's lamp, out: its chamber, lit by what comes in (shaded, as a white surface takes it), from its first corner
    let n = normalize(v.n); let tu = normalize(cross(vec3f(0.0, 1.0, 0.0), n)); let tv = cross(n, tu);
    let eye = v.w - F.cam_pos.xyz; let dist = length(eye); let d = eye / dist; let q = v.w - v.start;
    let inside = lamp_flat(lamp_flat_style(v.flag), v.col.xy, vec2f(dot(q, tu), dot(q, tv)), vec3f(dot(d, tu), dot(d, tv), -dot(d, n)), vec3f(0.0),
      shaded, dist * pixel_angle());
    return vec4f(finish(inside + glass_sheen(d, n, lamp_reach(v.w), v.w), v.w), 1.0);
  }
  // a lens the mesh draws, its own light and, over it as the brighter, its true radiance, as its sprite had it far off
  if (burns) { return vec4f(max(finish(v.col, v.w), lens_burn(v.col, v.w, ${CD_LAMP.toFixed(1)}, dot(v.col, vec3f(0.2126, 0.7152, 0.0722)) / ${LENS_WHITE.toFixed(1)})), 1.0); }
  if (v.flag > 3.5${glazed ? '' : ' && false'}) { return vec4f(finish(plain_glass(v.w, normalize(v.n), v.start, v.col, shaded, v.flag), v.w), 1.0); }
  if (v.flag > 1.5) { let veil = snow_veil_at(v.w); return vec4f(display(mix(v.col, veil.rgb, veil.a)), 1.0); } // a lit surface, seen through the falling snow
  return vec4f(finish(select(shaded, v.col, v.flag > 0.5), v.w), 1.0);
}
`;
export const PLAIN = plain(true), PLAIN_SOLID = plain(false);
// ---- the plain faces far off, lit once a pixel (src/worker.js, defer), as the far ground is (terrainDeferred) ----
// From over the town the far buildings' faces are a pixel or less, each shaded in a 2 x 2 quad of its own, and with
// 4x MSAA every face that holds a sample of a pixel: some pixels were lit many times over. So PLAIN_SOLID's draws
// marked GFX_CMD_FAR are drawn into the far ground's buffer too (PLAIN_GBUF: the point, the normal, how it moves from
// pixel to pixel, its colour, where its lit faces are), lit there once a pixel (PLAIN_SHADE), and the frame's own draw
// of them takes its pixel's light (PLAIN_READ) where the point lit there is its own, within two pixels' footprint;
// elsewhere (the edges, where another face holds the pixel's middle; the unlit faces) it lights itself as before.
// a unit normal in 30 bits, octahedral (y the pole), 15 bits a side
const OCT = /* wgsl */`
fn oct_pack(n: vec3f) -> u32 {
  var p = n.xz / (abs(n.x) + abs(n.y) + abs(n.z));
  if (n.y < 0.0) { p = (1.0 - abs(p.yx)) * select(vec2f(-1.0), vec2f(1.0), p >= vec2f(0.0)); }
  let q = vec2u(round(clamp(p * 0.5 + 0.5, vec2f(0.0), vec2f(1.0)) * 32767.0));
  return (q.x << 15u) | q.y;
}
fn oct_unpack(u: u32) -> vec3f {
  let p = vec2f(f32((u >> 15u) & 0x7fffu), f32(u & 0x7fffu)) * ${(2 / 32767).toExponential(8)} - 1.0;
  var n = vec3f(p.x, 1.0 - abs(p.x) - abs(p.y), p.y);
  let t = max(-n.y, 0.0);
  n.x += select(t, -t, n.x >= 0.0); n.z += select(t, -t, n.z >= 0.0);
  return normalize(n);
}
`;
const PLAIN_VS = '@vertex fn vs(i: VIn) -> V {';
const PLAIN_FS = `@fragment fn fs(v: V) -> @location(0) vec4f {
  rime_on = select(0.0, ${RIME.plain.toFixed(2)}, v.flag < 0.5);`;
// into the buffer: a lit face's point (a's bits, its normal in a.w, the top bit clear: not the ground's), its footprint
// (b) and its colour (c, a 1 where one is)
export const PLAIN_GBUF = OCT + PLAIN_SOLID.slice(0, PLAIN_SOLID.indexOf('@fragment fn fs(')) + /* wgsl */`
struct GOut { @location(0) a: vec4u, @location(1) b: vec4f, @location(2) c: vec4f }
@fragment fn fs(v: V) -> GOut {
  let dx = dpdx(v.w.xz); let dy = dpdy(v.w.xz);
  if (v.flag > 0.5) { discard; return GOut(); }
  return GOut(vec4u(bitcast<vec3u>(v.w), oct_pack(normalize(v.n))), vec4f(dx, dy), vec4f(v.col, 1.0));
}
`;
// lit once a pixel, where the buffer holds a lit plain face; its light's alpha 0.5 (the far ground's is 1)
export const PLAIN_SHADE = OCT + /* wgsl */`
@group(1) @binding(0) var g_a: texture_2d<u32>;
@group(1) @binding(1) var g_b: texture_2d<f32>;
@group(1) @binding(2) var g_c: texture_2d<f32>;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f { return vec4f(f32(i & 1u) * 4.0 - 1.0, f32(i >> 1u) * 4.0 - 1.0, 0.0, 1.0); }
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let p = vec2i(pos.xy);
  let c = textureLoad(g_c, p, 0);
  if (c.a < 0.5) { discard; return vec4f(0.0); }
  let a = textureLoad(g_a, p, 0); let w = bitcast<vec3f>(a.xyz);
  let d = textureLoad(g_b, p, 0);
  rime_on = ${RIME.plain.toFixed(2)};
  return vec4f(finish(lit_snow(c.rgb, oct_unpack(a.w), w, 1.0, -1.0, 1.0, d.xy, d.zw), w), 0.5);
}
`;
// the frame's far plain faces: the pixel's light where it was lit from this face, else their own
export const PLAIN_READ = (() => {
  const c = PLAIN_SOLID; const a = c.indexOf(PLAIN_FS);
  if (a < 0 || c.indexOf(PLAIN_FS, a + 1) >= 0) { throw new Error('plain read'); }
  const lit = '  let dx = dpdx(v.w.xz); let dy = dpdy(v.w.xz);\n'; // (the plain faces' own, taken before the read's return)
  if (c.split(lit).length !== 2) { throw new Error('plain read: lit'); }
  return c.slice(0, a).replace(PLAIN_VS, `@group(1) @binding(0) var plain_shaded: texture_2d<f32>;
@group(1) @binding(1) var plain_at: texture_2d<u32>;
` + PLAIN_VS) + PLAIN_FS + `
  let pdx = dpdx(v.w); let pdy = dpdy(v.w); let near = 2.0 * (abs(pdx) + abs(pdy));
  if (v.flag < 0.5) {
    let p = vec2i(v.pos.xy); let got = textureLoad(plain_shaded, p, 0);
    if (got.a > 0.25 && got.a < 0.75 && all(abs(bitcast<vec3f>(textureLoad(plain_at, p, 0).xyz) - v.w) <= near)) { return vec4f(got.rgb, 1.0); }
  }
` + c.slice(a + PLAIN_FS.length).replace(lit, '  let dx = pdx.xz; let dy = pdy.xz;\n');
})();


// The 4 x 4 ordered-dither threshold at q (0..3 each), (value + 0.5) / 16, from its bits: the table
// 0 8 2 10 / 12 4 14 6 / 3 11 1 9 / 15 7 13 5, worked out rather than indexed, which would put it in memory
const BAYER4 = /* wgsl */`
fn bayer4(q: vec2i) -> f32 {
  let x = u32(q.x); let y = u32(q.y); let e = x ^ y;
  return (f32(((e & 1u) << 3u) | ((y & 1u) << 2u) | (e & 2u) | ((y >> 1u) & 1u)) + 0.5) / 16.0;
}
`;
// Which of a pixel's `samples` samples to keep for a share s of it: as many as that comes to, an ordered dither between
// the counts, and which ones turned pixel by pixel. With multisampling a see-through or fading surface so covers part of
// each pixel, as an edge does, rather than all of some pixels and none of the rest: no lone dark pixels far off.
const KEEP = /* wgsl */`
fn keep(s: f32, q: vec2i, samples: u32) -> u32 {
  let k = min(u32(saturate(s) * f32(samples) + bayer4(q)), samples);
  var m = ((1u << k) - 1u) << (u32(q.x + q.y) % samples);
  m |= m >> samples;
  return m & ((1u << samples) - 1u);
}
`;
// Fine trim (GFX_ITEM_FINE in the C: gutters, downpipes, boards, a capping's sides, a ledge's snow), GFX_FINE_SIZE
// across at the most: drawn as PLAIN is, but thinned away as it shrinks from 3/4 of a pixel across to half a pixel, where
// the frame stops drawing it, so it fades rather than going all at once. With multisampling a pixel keeps fewer of its
// samples (an ordered dither between the counts), so a thin line dims evenly; without, whole pixels are dropped.
export const FINE_SIZE = 0.18; // GFX_FINE_SIZE
// The wires (PIPE_LINE: light strings' cables, hangers, the lines between posts), WIRE m across: a line a pixel wide
// on the screen, so each pixel of it keeps the share of its samples the wire's true width covers there (with an
// ordered dither between the counts, as fine trim): as wide as it is near by, fainter and fainter far off, as a
// photograph shows a wire, and nothing where it covers under an eighth of a pixel's samples. (Drawn a pixel wide at every
// distance, a wire a kilometre off showed as if it were half a metre thick.)
export const WIRE = 0.012;
export const wire = samples => MESH_IN + BAYER4 + /* wgsl */`
struct WireOut { @location(0) c: vec4f, @builtin(sample_mask) mask: u32 }
@vertex fn vs(i: VIn) -> V {
  var o: V;
  o.pos = F.view_proj * vec4f(i.pos, 1.0);
  o.w = i.pos; o.n = i.nrm.xyz; o.col = unpack(i.col); o.flag = round(i.nrm.w * 127.0); o.start = i.pos;
  return o;
}
@fragment fn fs(v: V) -> WireOut {
  let across = ${WIRE} / (distance(v.w, F.cam_pos.xyz) * pixel_angle()); // of the pixel's width
  let q = vec2i(v.pos.xy) & vec2i(3);
  let k = min(u32(saturate(across) * ${samples}.0 + bayer4(q)), ${samples}u);
  if (k == 0u) { discard; }
  var o: WireOut;
  o.mask = ((1u << k) - 1u) << (u32(q.x + q.y) % ${samples}u);
  o.mask |= o.mask >> ${samples}u;
  o.c = vec4f(finish(v.col, v.w), 1.0);
  return o;
}
`;
export const fine = samples => MESH_IN + BAYER4 + /* wgsl */`
struct FineOut { @location(0) c: vec4f, @builtin(sample_mask) mask: u32 }
@vertex fn vs(i: VIn) -> V {
  var o: V;
  o.pos = F.view_proj * vec4f(i.pos, 1.0);
  o.w = i.pos; o.n = i.nrm.xyz; o.col = unpack(i.col); o.flag = round(i.nrm.w * 127.0);
  return o;
}
@fragment fn fs(v: V) -> FineOut {
  let across = ${FINE_SIZE} / (distance(v.w, F.cam_pos.xyz) * pixel_angle()); // pixels
  let q = vec2i(v.pos.xy) & vec2i(3);
  let k = min(u32(saturate(4.0 * across - 2.0) * ${samples}.0 + bayer4(q)), ${samples}u); // samples kept: all from 3/4 of a pixel
  var o: FineOut;
  o.mask = ((1u << k) - 1u) << (u32(q.x + q.y) % ${samples}u); // which, turned pixel by pixel
  o.mask |= o.mask >> ${samples}u;
  rime_on = select(0.0, ${RIME.fine.toFixed(2)}, v.flag < 0.5); // gutters, boards, a rail's sides, a car's sills: thin, so as cold as the sky makes them
  let shaded = lit(v.col, normalize(v.n), v.w, 1.0);
  if (v.flag > 1.5) { o.c = vec4f(display(v.col), 1.0); } else { o.c = vec4f(finish(select(shaded, v.col, v.flag > 0.5), v.w), 1.0); }
  return o;
}
`;

// What moves (struct gfx_live in the C): the tower's lights, Father Christmas and his team, the traffic and the coastal
// steamers. The mesh is sent once, each part in its own space; a vertex names the slot that places it and the tint that
// colours it, and those are all that change. What is in the sky is neither lit nor hazed: it shows as bright as it is,
// right across town. A car's body or a ship's hull is lit as the town is (flag 3, GFX_LAMPLIT), its lamps and windows
// are their own light (1), and both are hazed.
export const LIVE_SLOTS = 3776, LIVE_TINTS = 64; // GFX_LIVE_SLOTS, GFX_LIVE_TINTS
// herd: the traffic's bodies, a look's body drawn once a car (traffic.h, struct traffic_seen), each instance the car's
// slot and, 16 bits up, its paint's tint, which the body's paint (marked TRAFFIC_PAINTED) takes
export const TRAFFIC_PAINTED = 127;
const live = herd => GLASS_FNS + /* wgsl */`
struct Live { m: array<vec4f, ${LIVE_SLOTS * 3}>, tint: array<vec4f, ${LIVE_TINTS}> }
@group(1) @binding(0) var<storage, read> L: Live;
struct VIn { @location(0) pos: vec3f, @location(1) nrm: vec4f, @location(2) col: vec4f${herd ? ', @location(3) car: u32' : ''} }
struct V { @builtin(position) pos: vec4f, @location(0) col: vec3f, @location(1) w: vec3f, @location(2) @interpolate(flat) flag: f32,
  @location(3) local: vec3f, @location(4) @interpolate(flat) eye: vec3f, // the point and the eye in the part's own space
  @location(5) @interpolate(flat) start: vec4f } // a lens's first corner, in the part's space, and how brightly it burns
@vertex fn vs(i: VIn) -> V {
  var o: V;
  let s = ${herd ? 'i32(i.car & 0xffffu)' : '(i32(round(i.nrm.x * 127.0)) + 128 * i32(round(i.nrm.z * 127.0)))'} * 3; let p = vec4f(i.pos, 1.0);
  let e = F.cam_pos.xyz - vec3f(L.m[s].w, L.m[s + 1].w, L.m[s + 2].w); // the slot's turn is a rotation: its transpose undoes it
  o.local = i.pos; o.eye = L.m[s].xyz * e.x + L.m[s + 1].xyz * e.y + L.m[s + 2].xyz * e.z;
  o.w = vec3f(dot(L.m[s], p), dot(L.m[s + 1], p), dot(L.m[s + 2], p));
  o.pos = F.view_proj * vec4f(o.w, 1.0);
  let tint = L.tint[${herd ? `select(i32(round(i.nrm.y * 127.0)), i32(i.car >> 16u), i32(round(i.nrm.y * 127.0)) == ${TRAFFIC_PAINTED})` : 'i32(round(i.nrm.y * 127.0))'}];
  o.col = select(unpack(i.col), tint.rgb, tint.a > 0.5);
  o.flag = round(i.nrm.w * 127.0);
  if (o.flag > GFX_LENS_HEAD - 0.5 && o.flag < GFX_LENS_HEAD + 3.5) {
    o.col = i.col.rgb * 2.55; // its size, m
    // a tail lamp burns brighter under braking: the car's own, of the traffic (the light list, wasm/src/shine.c), whose middle it is beside;
    // and a car of the traffic not in the list has its lamps out (switched off, traffic.h's dark: parked in a station's bay)
    var burn = 1.0;
    if (o.flag < 10.5) {
      var found = false;
      let b = light_bucket(o.w.xz); let end = shine_grid[b + 1u];
      for (var i = shine_grid[b]; i < end; i++) {
        let c = shine_lights[shine_grid[${SHINE.index0}u + i]];
        if (c.col.w < 0.5 && distance(c.at.xyz, o.w) < 3.5) { burn = select(1.0, c.dir.z, o.flag > 9.5); found = true; break; }
      }
      ${herd ? 'if (!found) { burn = 0.0; }' : ''}
    }
    o.start = vec4f(i.pos, burn);
  }
  return o;
}
@fragment fn fs(v: V, @builtin(front_facing) front: bool) -> @location(0) vec4f {
  let face = cross(dpdx(v.w), dpdy(v.w)); // flat, from the surface itself, turned to the eye; under a pixel it is nothing: then up
  var n = select(vec3f(0.0, 1.0, 0.0), normalize(face), dot(face, face) > 1e-24);
  if (dot(n, F.cam_pos.xyz - v.w) < 0.0) { n = -n; }
  let lens = v.flag > GFX_LENS_HEAD - 0.5;
  let burns = v.flag > GFX_LENS_LAMP - 0.5; // a lens of its glow sprite's kind: it burns at its own radiance and takes no light in
  let dxz = dpdx(v.w.xz); let dyz = dpdy(v.w.xz); // (here, where every pixel of the triangle is: lit_snow is then free to be left out)
  var lit_col = vec3f(0.0);
  if (!burns) { lit_col = lit_snow(select(v.col, vec3f(1.0), lens), n, v.w, 1.0, -1.0, 1.0, dxz, dyz); }
  // a lens the glow sprite is the far point of (GFX_LENS_LAMP, _BEACON, _WORK), after the lit colour's derivatives: its own light and, over it as the brighter, its true
  // radiance as the sprite has it (lens_burn). Lamp and beacon: its tint's luminance over the dark glass's is the share it burns;
  // a work light's glass, white, by how it faces the eye as its point does (plough.c).
  if (burns) {
    let own = finish(v.col, v.w);
    let beacon = v.flag > GFX_LENS_LAMP + 0.5 && v.flag < GFX_LENS_LAMP + 1.5;
    let work = v.flag > GFX_LENS_LAMP + 1.5;
    let tinted = max(dot(v.col, vec3f(0.2126, 0.7152, 0.0722)) - ${LENS_FLOOR.toFixed(2)}, 0.0) / ${LENS_WHITE.toFixed(1)};
    let eye_on = abs(dot(n, normalize(F.cam_pos.xyz - v.w)));
    let share = select(tinted, select(0.06, 0.06 + 0.94 * smoothstep(-0.1, 0.7, eye_on), front), work);
    return vec4f(max(own, lens_burn(v.col, v.w, select(${CD_LAMP.toFixed(1)}, ${CD_BEACON.toFixed(1)}, beacon), share)), 1.0);
  }
  if (lens) { // a lamp, its chamber behind its lens: the headlamps and tail lamps burn, the indicators and reversing lamps are out
    let s = sign(v.eye.z - v.local.z); // the lens faces along z, to the side the eye is on
    let d = normalize(v.local - v.eye); let q = v.local - v.start.xyz;
    let on = select(0.0, v.start.w, v.flag < 10.5);
    let foot = distance(v.w, F.cam_pos.xyz) * pixel_angle();
    let style = lamp_flat_style(v.flag); let rd = vec3f(d.x * s, d.y, -d.z * s);
    // what comes in through the glass and back out, as any surface; and the lamp's own light at its true radiance towards
    // the eye, as the glow sprite draws the lens before the mesh takes over from it (traffic.c, LAMP_OWN0..1), clipped
    // as a lens is (tone_hue, keeping its hue) and over the rest as the brighter, as the sprite is: so that a lens near
    // by is as bright on each pixel as it was as a disc and as a point further off, never dimmer
    let back = finish(lamp_flat(style, v.col.xy, q.xy, rd, vec3f(0.0), lit_col, foot) + glass_sheen(normalize(v.w - F.cam_pos.xyz), n, lamp_reach(v.w), v.w), v.w);
    if (on <= 0.0) { return vec4f(back, 1.0); }
    let light = lamp_true(style, v.eye - v.local, v.flag > 9.5, on);
    let depth = dot(v.w - F.cam_pos.xyz, F.cam_fwd.xyz);
    let own = lamp_flat(style, v.col.xy, q.xy, rd, light, vec3f(0.0), foot) * exp(-F.fog.a * F.fog.a * depth * depth); // through a cloud, as finish()
    return vec4f(max(back, encode(tone_hue(own))), 1.0);
  }
  if (v.flag > 5.5 && v.flag < 7.5) { // a car's glass, or the truck's: its cabin behind it, in its own space, empty
    let eye = v.w - F.cam_pos.xyz; let dist = length(eye); let ray = eye / dist;
    let rd = normalize(v.local - v.eye); // the ray in the car's space
    let reach = lamp_reach(v.w);
    let outside = reach * 0.6 + moon_now() * 0.3 + sky_up(v.w.xz) * 0.3;
    var inside = outside * ${GLASS.cabin.toFixed(3)};
    let detail = smoothstep(${GLASS.detail[0].toFixed(1)}, ${GLASS.detail[1].toFixed(1)}, 1.0 / (dist * pixel_angle()));
    if (detail > 0.0) { inside = mix(inside, car_cabin(v.local, rd, v.flag > 6.5, outside), detail); }
    let shown = 1.0 - glass_fresnel(ray, n); // what the glass does not mirror it lets through
    return vec4f(finish(lit_col * 0.5 + inside * shown + glass_sheen(ray, n, reach, v.w), v.w), 1.0);
  }
  let shaded = finish(lit_col, v.w);
  if (v.flag > 2.5) { return vec4f(shaded, 1.0); }
  if (v.flag > 0.5 && v.flag < 1.5) { return vec4f(finish(v.col, v.w), 1.0); }
  let veil = snow_veil_at(v.w); // a lit surface, seen through the falling snow
  return vec4f(display(mix(v.col, veil.rgb, veil.a)), 1.0);
}
`;
export const LIVE = live(false), LIVE_HERD = live(true);

// The discs that rotors and propellers sweep (GFX_BLUR, wasm/src/plane.c, plane_mesh_disc), drawn over all that is solid,
// blended, as one frame's exposure sees them: each blade, chord c at radius r, covers c / r radians; in the 1/60 s the
// frame is exposed for it sweeps on through the tint's x (its speed times that). A point the blade passes is covered
// for the share of that time the blade lies over it, so a fast rotor is a faint disc with its blades smeared into
// streaks round it, and as it runs down they draw in and come to rest as blades. Where on the disc a corner lies is
// in its colour's first two bytes (u, v from -1 to 1): the tint's w is the disc's radius, z its blades, y their chord.
// Blades turn from v towards u. Nothing glows: the blades are lit as anything else is.
export const LIVE_BLUR = /* wgsl */`
struct Live { m: array<vec4f, ${LIVE_SLOTS * 3}>, tint: array<vec4f, ${LIVE_TINTS}> }
@group(1) @binding(0) var<storage, read> L: Live;
struct VIn { @location(0) pos: vec3f, @location(1) nrm: vec4f, @location(2) col: vec4f }
struct V { @builtin(position) pos: vec4f, @location(0) uv: vec2f, @location(1) w: vec3f, @location(2) @interpolate(flat) blade: vec4f }
@vertex fn vs(i: VIn) -> V {
  var o: V;
  let s = (i32(round(i.nrm.x * 127.0)) + 128 * i32(round(i.nrm.z * 127.0))) * 3; let p = vec4f(i.pos, 1.0);
  o.w = vec3f(dot(L.m[s], p), dot(L.m[s + 1], p), dot(L.m[s + 2], p));
  o.pos = F.view_proj * vec4f(o.w, 1.0);
  o.uv = i.col.xy * 2.0 - 1.0;
  o.blade = L.tint[i32(round(i.nrm.y * 127.0))];
  return o;
}
const BLADE = vec3f(0.035, 0.036, 0.04);
@fragment fn fs(v: V) -> @location(0) vec4f {
  let r = length(v.uv) * v.blade.w;
  if (r < 0.05) { discard; }
  let sweep = max(v.blade.x, 1e-4); let w = v.blade.y / r; let n = max(v.blade.z, 1.0);
  let phi = atan2(v.uv.x, v.uv.y);
  var cover = 0.0;
  for (var k = 0.0; k < n; k += 1.0) {
    // where this blade's leading edge was at the exposure's start, behind it by the sweep; the point is under the blade
    // for the part of the sweep when it lay in [u, u + w]
    let u = fract((phi - k * 6.2831853 / n + sweep) / 6.2831853 + 0.5) * 6.2831853 - 3.1415927;
    cover += max(min(u + w, sweep) - max(u, 0.0), 0.0) / sweep;
  }
  cover = min(cover, 1.0);
  if (cover < 0.004) { discard; }
  let face = cross(dpdx(v.w), dpdy(v.w));
  var nrm = select(vec3f(0.0, 1.0, 0.0), normalize(face), dot(face, face) > 1e-24);
  if (dot(nrm, F.cam_pos.xyz - v.w) < 0.0) { nrm = -nrm; }
  return vec4f(finish(lit(BLADE, nrm, v.w, 1.0), v.w), cover);
}
`;

// What floats: moored boats and floating pontoons ride the same swell the sea's surface shows. A boat moves as one: heave
// and the swell's tilt where it lies, and a slow roll, pitch and heave of its own, as much as the C gave it (struct
// boat_sway: a short boat livelier, a sheltered berth quieter; never more than BOAT_ROLL_MAX), and a slow yaw and surge as
// it tugs at its lines. A pontoon heaves vertex by vertex with the swell under it, so a run of them flexes as one jetty.
// A boat's lit windows and lamps (flag 1) are their own light, as on land.
const TUG = { yaw: [0.25, 23], surge: [1.6, 17] }; // shares of its roll (rad, and m per rad); period s
const per = b => (2 * Math.PI / b[1]).toFixed(5);
const afloat = sea => GLASS_FNS + MESH_IN + sea + /* wgsl */`
fn swell_at(p: vec2f) -> f32 {
  var h = 0.0;
  for (var i = 0; i < 3; i++) { h += S.swell[i].w * sin(dot(S.swell[i].xy, p) + S.swell[i].z); }
  return h;
}
// boat: its centre x, z, heading (rad), length (m); sway: its roll (rad), roll period (s), phase (rad), heave (m); all zero on a pontoon
@vertex fn vs(i: VIn, @location(4) boat: vec4f, @location(5) sway: vec4f) -> V {
  var pos = i.pos; var n = i.nrm.xyz;
  if (boat.w > 0.0) { // small angles, so a turn is applied as v + turn x v
    let c = boat.xy; let t = F.cam_pos.w; let ph = sway.z; let w = 6.2831853 / sway.y;
    let fwd = vec3f(sin(boat.z), 0.0, cos(boat.z)); let beam = vec3f(fwd.z, 0.0, -fwd.x);
    let g = swell_slope(c) / (1.0 + boat.w / 30.0); // a longer hull bridges the swell
    let roll = sway.x * sin(t * w + ph) + dot(g, beam.xz);
    let pitch = sway.x * 0.35 * sin(t * w * 0.79 + ph * 1.7) - dot(g, fwd.xz);
    let yaw = sway.x * ${TUG.yaw[0]} * sin(t * ${per(TUG.yaw)} + ph * 2.3);
    let turn = fwd * roll + beam * pitch + vec3f(0.0, yaw, 0.0);
    let shift = fwd * sway.x * ${TUG.surge[0]} * sin(t * ${per(TUG.surge)} + ph * 0.7) + vec3f(0.0, swell_at(c) + sway.w * sin(t * w * 0.83 + ph * 3.1), 0.0);
    n += cross(turn, n);
    pos += cross(turn, pos - vec3f(boat.x, 0.0, boat.y)) + shift;
  } else { pos.y += swell_at(pos.xz); }
  var o: V;
  o.pos = F.view_proj * vec4f(pos, 1.0);
  o.w = pos; o.n = n; o.col = unpack(i.col); o.flag = round(i.nrm.w * 127.0); o.start = pos;
  return o;
}
@fragment fn fs(v: V) -> @location(0) vec4f {
  let shaded = lit(v.col, normalize(v.n), v.w, 1.0); // before any branch: it takes the surface's slope across the pixel
  if (v.flag > 3.5) { // a boat's pane, its wheelhouse behind it: lit by its lamp when the boat is lit, else by what comes in
    let n = normalize(v.n);
    let eye = v.w - F.cam_pos.xyz; let dist = length(eye); let ray = eye / dist;
    let reach = lamp_reach(v.w);
    let outside = reach * ${GLASS.outside.toFixed(3)} * 3.0 + moon_now() * 0.15 + sky_up(v.w.xz) * 0.12;
    let lit_in = v.flag == GFX_GLASS_LIT;
    let lamp = select(vec3f(0.0), v.col * 2.5, lit_in);
    var inside = select(outside * 0.2, v.col * 0.6, lit_in); // its mean, far off
    let detail = smoothstep(${GLASS.detail[0].toFixed(1)}, ${GLASS.detail[1].toFixed(1)}, 1.0 / (dist * pixel_angle())) * step(abs(n.y), 0.7);
    if (detail > 0.0) {
      let g = glass_at(v.w, n, v.start, ray, 2.0, 100.0, 0.3, 0.0);
      inside = mix(inside, wheelhouse(g.o, g.d, outside, lamp), detail);
    }
    let shown = 1.0 - glass_fresnel(ray, n); // what the glass does not mirror it lets through
    return vec4f(finish(select(shaded * 0.6, vec3f(0.0), lit_in) + inside * shown + glass_sheen(ray, n, reach, v.w), v.w), 1.0);
  }
  return vec4f(finish(select(shaded, v.col, v.flag > 0.5), v.w), 1.0);
}
`;

// instanced models: struct inst is position, turn about y, scale, colour
const INSTANCED = /* wgsl */`
struct IIn { @location(4) at: vec4f, @location(5) scale: vec3f, @location(6) tint: vec4f }
fn place(p: vec3f, i: IIn) -> vec3f {
  let s = sin(i.at.w); let c = cos(i.at.w);
  let q = p * i.scale;
  return i.at.xyz + vec3f(q.x * c + q.z * s, q.y, q.z * c - q.x * s);
}
`;

// trees: one flat shade per face, taken from the surface itself so both faces light alike
export const TREE = MESH_IN + INSTANCED + /* wgsl */`
@vertex fn vs(i: VIn, inst: IIn) -> V {
  var o: V;
  let w = place(i.pos, inst);
  o.pos = F.view_proj * vec4f(w, 1.0);
  o.w = w; o.n = vec3f(0.0, 1.0, 0.0); o.col = unpack(i.col) * unpack(inst.tint); o.flag = 0.0;
  return o;
}
@fragment fn fs(v: V) -> @location(0) vec4f {
  let face = cross(dpdx(v.w), dpdy(v.w));
  var n = select(vec3f(0.0, 1.0, 0.0), normalize(face), dot(face, face) > 1e-24);
  if (dot(n, F.cam_pos.xyz - v.w) < 0.0) { n = -n; }
  rime_on = ${RIME.tree.toFixed(2)};
  return vec4f(finish(lit(v.col, n, v.w, 0.9), v.w), 1.0);
}
`;

// ---- the forest (wasm/src/tree.c, wasm/src/forest.c; src/trees.js) ----
// Every model is a triangle list in one float texture, two texels per vertex, one row per variant and level of
// detail; a tree is eight floats (struct tree_inst) in one buffer, and its vertices are fetched by vertex index. Which
// trees are drawn, and at which level, is listed every frame by the C (forest.h): an instance is one word, the tree
// and its level, and each draw runs over exactly one model's vertices. A tree wholly at one level is drawn by FOREST,
// which discards nothing; one part-way between two is listed for both and drawn by FOREST_FADE, which dithers them
// against each other; the twigs of far trees by FOREST_TWIG; and beyond, a quad each, by FOREST_IMPOSTOR.
const TREE_LEVELS = 4, TREE_FADE = [0.1, 0.085, 0.03]; // as wasm/include/harstad/tree.h: TREE_NEAR_FADE, TREE_MID_FADE, TREE_FAR_FADE
// tree_dev: how far each level's outline strays from the next's (m at scale 1), by level and variant; test.c reads it
const TREE_DEVS = [
  [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.20, 0.20, 0.10, 0.15, 0.10, 0.10, 0.15, 0.10, 0.10, 0.10],
  [0.15, 0.25, 0.30, 0.20, 0.20, 0.30, 0.40, 0.40, 0.10, 0.25, 0.30, 0.10, 0.30, 0.25, 0.15, 0.10],
  [0.35, 0.30, 0.40, 0.35, 0.30, 0.30, 0.35, 0.35, 0.25, 0.35, 0.30, 0.25, 0.30, 0.30, 0.25, 0.25]];
// FOREST_MID_AFTER, FOREST_FAR_AFTER (forest.h): each hand-over begins at least this much further out than the last ends
const TREE_AFTER = [(1 + TREE_FADE[0]) / (1 - TREE_FADE[1]) * 1.005, (1 + TREE_FADE[1]) / (1 - TREE_FADE[2]) * 1.005];
const TREE_MOON_SHADE = 0.45; // snow and twigs scatter moonlight inside a crown: the moon's shadow only takes a tree down to this
const FOREST_IN = /* wgsl */`
struct TIn { @builtin(vertex_index) vi: u32, @location(0) pick: u32 } // pick: the tree, and above bit 24 its level
struct Tree { at: vec4f, form: vec4f } // struct tree_inst: x y z yaw; across, up, lean, variant + shade
struct TreeVertex { w: vec3f, local: vec3f, t0: vec4f, t1: vec4f, flag: vec3f }
fn tree_bytes(v: f32) -> vec3f { return floor(vec3f(v / 65536.0, v / 256.0, v) % 256.0) / 255.0; }
fn tree_turn(p: vec3f, yaw: f32) -> vec3f { let c = cos(yaw); let s = sin(yaw); return vec3f(c * p.x + s * p.z, p.y, c * p.z - s * p.x); }
// This vertex of this tree's model at the given level, placed: scaled, turned, leaning downwind, and pinned to the
// terrain where the model says so.
fn tree_vertex(vi: u32, i: Tree, level: i32) -> TreeVertex {
  var o: TreeVertex;
  let row = i32(i.form.w) * ${TREE_LEVELS} + level;
  o.t0 = textureLoad(tree_verts, vec2i(i32(vi) * 2, row), 0);
  o.t1 = textureLoad(tree_verts, vec2i(i32(vi) * 2 + 1, row), 0);
  o.flag = tree_bytes(o.t1.y);
  o.local = o.t0.xyz * i.form.xyx;
  var p = tree_turn(o.local, i.at.w) + i.at.xyz;
  p += vec3f(0.707, 0.0, -0.707) * (i.form.z * o.local.y); // the prevailing wind is from the south-west
  if (o.flag.y > 0.0) { p.y = mix(p.y, tree_ground(p.xz) + o.local.y, o.flag.y); }
  o.w = p;
  return o;
}
`;
const FOREST_VS = /* wgsl */`
@group(1) @binding(0) var tree_verts: texture_2d<f32>;
@group(1) @binding(1) var<storage, read> trees: array<Tree>;
fn tree_ground(xz: vec2f) -> f32 { return ground_at(xz); }
` + FOREST_IN + /* wgsl */`
struct TV { @builtin(position) pos: vec4f, @location(0) w: vec3f, @location(1) col: vec3f, @location(2) local: vec3f, @location(3) n: vec3f,
  @location(4) k: vec4f, // k: how much of this level shows (negative while it fades in), alpha, mark, glow
  @location(5) lit: f32, // over 0 where a moving light can reach the tree (lights_reach): every corner of a tree says the same
  @location(6) hc: vec4f } // the twig haze, lit at its corners (rgb, a 1): FOREST_HAZE
${litOver('lit_far', true)}
`;
// How much of a tree shows at its level, d metres from the eye: 1 in the level's own range, falling to 0 across the
// hand-over at each end (negative while it fades in, so the two levels take complementary pixels). Each level hands
// over to the next where the most its outline strays from the next's, at the tree's scale, comes to half a pixel, or
// just past the hand-over before (forest_switch, wasm/include/harstad/forest.h).
const wgslArray = a => `array<f32, ${a.length}>(${a.map(v => v.toFixed(2)).join(', ')})`;
const forestSwitch = /* wgsl */`
var<private> tree_dev: array<array<f32, ${TREE_DEVS[0].length}>, 3> = array<array<f32, ${TREE_DEVS[0].length}>, 3>(${TREE_DEVS.map(wgslArray).join(', ')});
fn tree_switch(tree: Tree) -> vec3f {
  let s = F.cam_up.w / F.cam_right.w * max(tree.form.x, tree.form.y); // where a metre comes to half a pixel, at its scale
  let v = u32(tree.form.w);
  let near = s * tree_dev[0][v];
  let mid = max(s * tree_dev[1][v], near * ${TREE_AFTER[0].toFixed(6)});
  return vec3f(near, mid, max(s * tree_dev[2][v], mid * ${TREE_AFTER[1].toFixed(6)}));
}
fn tree_hand(at: f32, fade: f32, d: f32) -> f32 { return smoothstep(at * (1.0 - fade), at * (1.0 + fade), d); }
`;
const forestFade = forestSwitch + /* wgsl */`
fn tree_shows(level: i32, d: f32, tree: Tree) -> f32 {
  let at = tree_switch(tree);
  let near = tree_hand(at.x, ${TREE_FADE[0]}, d);
  let mid = tree_hand(at.y, ${TREE_FADE[1]}, d);
  let far = tree_hand(at.z, ${TREE_FADE[2]}, d);
  if (level == 0) { return 1.0 - near; }
  if (level == 1) { return select(1.0 - mid, -near, near < 1.0); }
  if (level == 2) { return select(1.0 - far, -mid, mid < 1.0); }
  return -far;
}
`;
// kind: 'solid' and 'twig' are trees wholly at their level; 'fade' is one handing over; 'haze' the blended part of
// near and mid trees, which only fades out towards the far level.
const forestVertex = kind => FOREST_VS + (kind === 'fade' ? forestFade : kind === 'haze' ? forestSwitch : '') + /* wgsl */`
@vertex fn vs(i: TIn) -> TV {
  var o: TV;
  let tree = trees[i.pick & 0xffffffu];
  let level = i32(i.pick >> 24u);
  let t = tree_vertex(i.vi, tree, level);
  o.w = t.w; o.local = t.t0.xyz;
  o.n = tree_turn(tree_bytes(t.t1.x) * 2.0 - 1.0, tree.at.w);
  ${kind === 'fade' ? 'let shows = tree_shows(level, distance(tree.at.xyz, F.cam_pos.xyz), tree);'
    : kind === 'haze' ? `let shows = 1.0 - tree_hand(tree_switch(tree).y, ${TREE_FADE[1]}, distance(tree.at.xyz, F.cam_pos.xyz));`
    : 'let shows = 1.0;'}
  o.k = vec4f(shows, t.flag.x, tree_bytes(t.t1.z).x, t.flag.z);
  // no corner lies further from the tree's foot, on the map, than its own place in the model and how far its height leans it
  o.lit = f32(lights_reach(tree.at.xz, length(t.local.xz) + abs(tree.form.z * t.local.y) + 1.0));
  let c = tree_bytes(t.t0.w);
  o.col = c * c * (0.78 + 0.4 * fract(tree.form.w));
  o.pos = F.view_proj * vec4f(t.w, 1.0);
  ${kind === 'fade' || kind === 'haze' ? 'if (shows == 0.0) { o.pos = vec4f(0.0, 0.0, 0.0, 1.0); } // at the very end of a hand-over: nothing to draw' : ''}
  ${kind === 'haze' ? 'o.hc = vec4f(haze_lit(o.w, o.col, o.lit > 0.0), 1.0);' : ''}
  return o;
}
`;
// How much of a pixel of a bare crown's shell is twigs: densest where it is seen face-on and fading out at its
// outline. Fine streaks fanning out from the stem give the grain; far off the texture's mip levels even them out.
const FOREST_TWIG_AT = /* wgsl */`
// The detail map at (angle * k + y * a.x, radius * a.y + y * 0.006), the angle being the stem's, in radians. Its gradients are the
// exact derivatives, from the screen derivatives of the point alone: d atan2(z, x) = (x dz - z dx) / r^2 and d r = (x dx + z dz) / r,
// which have no jump where the angle wraps (the pixel across the seam would else take the wrap for a texture a screen wide, the
// coarsest mip), and cost the one atan2 and the one square root, not a second angle and its derivatives.
fn grain_round(local: vec3f, k: f32, a: vec2f) -> f32 {
  let r2 = max(dot(local.xz, local.xz), 1e-8); let r = sqrt(r2);
  let dx = dpdx(local); let dy = dpdy(local);
  let turn = vec2f(local.x * dx.z - local.z * dx.x, local.x * dy.z - local.z * dy.x) * (k / r2);
  let away = vec2f(local.x * dx.x + local.z * dx.z, local.x * dy.x + local.z * dy.z) / r;
  let uv = vec2f(atan2(local.z, local.x) * k + local.y * a.x, r * a.y + local.y * 0.006);
  return textureSampleGrad(detail_map, samp_repeat, uv, 0, vec2f(turn.x + dx.y * a.x, away.x * a.y + dx.y * 0.006), vec2f(turn.y + dy.y * a.x, away.y * a.y + dy.y * 0.006)).r;
}
fn tree_twigs(v: TV) -> f32 {
  let grain = grain_round(v.local, 0.3183099, vec2f(0.004, 0.012)); // 2 / 2 pi: the angle wraps by a whole number of tiles
  let face = abs(dot(normalize(v.n), normalize(F.cam_pos.xyz - v.w)));
  return (0.25 + 1.5 * smoothstep(0.44, 0.6, grain)) * face * face;
}
`;
// Solid trees: birch lenticels, frost that glints in lamplight, glowing bulbs. One flat shade per face. Between levels
// ('fade'), a dithered hand-over: complementary patterns, so the two levels never both drop a pixel. Far off, the twig
// haze is drawn as the solid trees are, screen-door see-through ('twig', and inside a far tree's run while it fades
// in): no blending. Only those two discard; the rest have their depth settled before they are shaded.
const forest = (kind, samples) => forestVertex(kind) + (kind === 'solid' ? '' : FOREST_TWIG_AT + BAYER4 + KEEP) + /* wgsl */`
struct FO { @location(0) c: vec4f${kind === 'solid' ? '' : ', @builtin(sample_mask) mask: u32'} }
@fragment fn fs(v: TV) -> FO {
  var o: FO;
  let face = cross(dpdx(v.w), dpdy(v.w)); // far off a triangle can be under a pixel, its derivatives nothing: then it faces up
  var n = select(vec3f(0.0, 1.0, 0.0), normalize(face), dot(face, face) > 1e-24);
  let to_eye = normalize(F.cam_pos.xyz - v.w);
  if (dot(n, to_eye) < 0.0) { n = -n; }
  let grain = textureSample(detail_map, samp_repeat, vec2f(atan2(v.local.z, v.local.x) * 0.012, v.local.y * 0.24), 0).r;
  let dx = dpdx(v.w.xz); let dy = dpdy(v.w.xz);
${kind === 'solid' ? '' : /* wgsl */`
  // The samples kept: a hand-over's two levels take complementary ones (the one fading in what the one fading out leaves),
  // and see-through twigs as many as they cover. A pixel left with none is not lit first: discard.
  let q = vec2i(v.pos.xy) & vec2i(3);
  let q2 = (vec2i(v.pos.xy) + vec2i(1, 2)) & vec2i(3);
  let full = ${(1 << samples) - 1}u;
  var mask = full;
  ${kind === 'fade' ? `mask = select(full & ~keep(1.0 + v.k.x, q, ${samples}u), keep(v.k.x, q, ${samples}u), v.k.x >= 0.0);` : ''}
  let twigs = tree_twigs(v);
  if (v.k.y < 0.99) { mask &= keep(v.k.y * select(twigs, 1.0, v.k.z > 0.5), q2, ${samples}u); }
  if (mask == 0u) { discard; return o; }
  o.mask = mask;`}
  let albedo = v.col * (1.0 - 0.8 * v.k.z * step(0.99, v.k.y) * (1.0 - smoothstep(0.37, 0.43, grain))); // dark lenticels in short horizontal dashes (the mark on see-through snow is not one)
  let above = v.w.y - ground_at(v.w.xz);
  let reach = lamp_reach_over(v.w, above);
  rime_on = ${RIME.tree.toFixed(2)}; // on what the bark and the twigs already carry (tree.c, PAINT_RIME)
  lit_moving = v.lit > 0.0;
  var c = lit_far(albedo, n, v.w, 0.9, -1.0, 1.0, dx, dy, ${TREE_MOON_SHADE}, reach, 1.0, above);
  let frost = smoothstep(0.25, 0.55, dot(v.col, vec3f(0.333)));
  let rim = pow(1.0 - abs(dot(n, to_eye)), 3.0);
  c += v.col * (frost * rim * 1.6) * (reach * lamp_tilt(n)) + v.col * (v.k.w * 1.3);
  o.c = vec4f(finish(c, v.w), 1.0);
  return o;
}
`;
export const forestShader = forest; // (kind, samples): 'solid', 'fade' or 'twig'
// Twig haze (near and mid): a soft, blended shell, lit as a whole rather than facet by facet. Levels simply
// cross-fade. The snow at the foot rides along.
// The shell is lit as a whole, facing up, and is faint (a few per cent of what is behind it): so its light is taken at
// its corners and spread over it (haze_lit, in the vertex shader), not worked out a pixel at a time. Against lighting
// every pixel the frame differs by at most a few levels in a hundred thousandth of its pixels; the aerial view's haze
// cost 11.7 ms, now 4.8.
const FOREST_HAZE_LIT = /* wgsl */`
fn haze_lit(w: vec3f, col: vec3f, moving: bool) -> vec3f {
  rime_on = ${RIME.tree.toFixed(2)};
  lit_moving = moving;
  let above = w.y - ground_at(w.xz);
  return finish(lit_far(col, vec3f(0.0, 1.0, 0.0), w, 0.9, -1.0, 1.0, vec2f(0.0), vec2f(0.0), ${TREE_MOON_SHADE}, lamp_reach_over(w, above), 1.0, above), w);
}
`;
export const FOREST_HAZE = FOREST_HAZE_LIT + forestVertex('haze') + FOREST_TWIG_AT + /* wgsl */`
@fragment fn fs(v: TV) -> @location(0) vec4f {
  let twigs = tree_twigs(v);
  var a = v.k.y * abs(v.k.x);
  if (v.k.z <= 0.5) { a *= twigs; }
  if (a < 0.5 / 255.0) { discard; } // what would not move the pixel behind by half a step of its eight bits is not drawn
  return vec4f(v.hc.rgb, saturate(a)); // lit at its corners
}
`;
// ---- the forest's impostors (wasm/include/harstad/tree.h, TREE_IMPOSTOR) ----
// Far off, a tree is one quad facing the eye. At load the glue renders each variant's near model from TREE_IMP_VIEWS a
// side of ways (IMPOSTOR_BAKE): colour into one tile of a texture array's layer; the normal, how far towards the eye each
// texel's surface lies, and its cover into the same tile of another. The quad blends the four views nearest the way the
// tree is seen (so nothing jumps as that way turns), each looked up where the eye's ray meets what that view saw: moved
// along the ray by the depth the view has where it meets the plane through the views' middle. It is lit as the models
// are, from the blend's normal, at the point the ray met; its depth is the quad's own, so the GPU can drop what it hides
// before shading it (writing each pixel's depth would cost more than all the rest).
export const IMP_VIEWS = 24, IMP_TILE = 24, IMP_MIPS = 4; // TREE_IMP_VIEWS, TREE_IMP_TILE; mips down to tiles of 3 texels
const IMP_LOW = -20; // TREE_IMP_LOW, degrees
const IMP_VIEW = /* wgsl */`
// The way view (i, j) looks from, towards the eye (test__imp_view): hemi-octahedral, corners included, the elevations
// spread from IMP_LOW to straight above
fn imp_view(i: u32, j: u32) -> vec3f {
  let n = ${(IMP_VIEWS - 1).toFixed(1)};
  let u = f32(i) / n * 2.0 - 1.0; let v = f32(j) / n * 2.0 - 1.0;
  let x = (u + v) * 0.5; let z = (u - v) * 0.5; let y = 1.0 - abs(x) - abs(z);
  let low = ${(IMP_LOW * Math.PI / 180).toFixed(6)};
  let az = atan2(z, x); let el = low + atan2(y, length(vec2f(x, z))) * (${(Math.PI / 2).toFixed(6)} - low) / ${(Math.PI / 2).toFixed(6)};
  return vec3f(cos(el) * cos(az), sin(el), cos(el) * sin(az));
}
// a unit normal in two numbers, 0 to 1, and back: octahedral
fn imp_oct(n: vec3f) -> vec2f {
  var p = n.xz / (abs(n.x) + abs(n.y) + abs(n.z));
  if (n.y < 0.0) { p = (1.0 - abs(p.yx)) * select(vec2f(-1.0), vec2f(1.0), p >= vec2f(0.0)); }
  return p * 0.5 + 0.5;
}
fn imp_unoct(e: vec2f) -> vec3f {
  let f = e * 2.0 - 1.0;
  var n = vec3f(f.x, 1.0 - abs(f.x) - abs(f.y), f.y);
  let t = max(-n.y, 0.0);
  n.x += select(t, -t, n.x >= 0.0); n.z += select(t, -t, n.z >= 0.0);
  return normalize(n);
}
// its frame: across, then up the picture
fn imp_across(w: vec3f) -> vec3f { return normalize(vec3f(-w.z, 0.0, w.x)); }
// where in the grid of views the way w (towards the eye) lies, 0 to IMP_VIEWS - 1 each way. Its bearing's cosine and
// sine are w's own x and z over their length: no angle is taken and turned back
fn imp_grid(w: vec3f) -> vec2f {
  let low = ${(IMP_LOW * Math.PI / 180).toFixed(6)};
  let el = asin(clamp(w.y, -1.0, 1.0)); let flat = length(w.xz);
  let az = select(vec2f(1.0, 0.0), w.xz / flat, flat > 0.0);
  let eg = (max(low, el) - low) / (${(Math.PI / 2).toFixed(6)} - low) * ${(Math.PI / 2).toFixed(6)};
  let g = vec3f(cos(eg) * az.x, sin(eg), cos(eg) * az.y);
  let s = abs(g.x) + g.y + abs(g.z);
  return vec2f((g.x + g.z) / s + 1.0, (g.x - g.z) / s + 1.0) * 0.5 * ${(IMP_VIEWS - 1).toFixed(1)};
}
`;
// Every view's way and its frame, across and up its picture (imp_view, imp_across and their cross), for the trees'
// shaders to look up. The vertex shader had worked the four views' frames out, two dozen sines, cosines and arctangents
// a corner, and handed them to the pixels as eight vectors a corner: with half a million trees on the hill, what a
// corner hands on is what a far tree costs. Now a corner hands on which four views, and a pixel looks their frames up.
// IMP_VIEWS texels a side for the ways, as many again beside them for the acrosses, and again for the ups; rgba32float.
export function impViews() {
  const n = IMP_VIEWS, out = new Float32Array(3 * n * n * 4), low = +(IMP_LOW * Math.PI / 180).toFixed(6), half = +(Math.PI / 2).toFixed(6);
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const u = i / (n - 1) * 2 - 1, v = j / (n - 1) * 2 - 1, x = (u + v) * 0.5, z = (u - v) * 0.5, y = 1 - Math.abs(x) - Math.abs(z);
    const az = Math.atan2(z, x), el = low + Math.atan2(y, Math.hypot(x, z)) * (half - low) / half;
    const w = [Math.cos(el) * Math.cos(az), Math.sin(el), Math.cos(el) * Math.sin(az)], l = Math.hypot(w[2], w[0]);
    const a = [-w[2] / l, 0, w[0] / l];
    out.set(w, (j * 3 * n + i) * 4);
    out.set(a, (j * 3 * n + n + i) * 4);
    out.set([a[1] * w[2] - a[2] * w[1], a[2] * w[0] - a[0] * w[2], a[0] * w[1] - a[1] * w[0]], (j * 3 * n + 2 * n + i) * 4); // cross(a, w)
  }
  return out;
}
// The bake: every view of one variant's near model at once, an instance a view, each into its tile: colour into one
// texture, and the normal (octahedral), the depth and the cover into the other, so that one look tells both how much is
// there and how far. Its solid part first, then its blended part over it (colour and cover only: where it is all there
// is, the normal and depth stay as the clear leaves them, upward and at the middle, as the haze is lit from above).
export const IMPOSTOR_BAKE = /* wgsl */`
@group(0) @binding(0) var tree_verts: texture_2d<f32>;
@group(0) @binding(1) var<uniform> B: vec4u; // x: the variant
fn tree_bytes(v: f32) -> vec3f { return floor(vec3f(v / 65536.0, v / 256.0, v) % 256.0) / 255.0; }
` + IMP_VIEW + /* wgsl */`
struct BV { @builtin(position) pos: vec4f, @location(0) col: vec3f, @location(1) m: vec3f, @location(2) n: vec3f, @location(3) k: vec3f,
  @location(4) @interpolate(flat) view: u32 }
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) view: u32) -> BV {
  var o: BV;
  let row = i32(B.x) * ${TREE_LEVELS};
  let t0 = textureLoad(tree_verts, vec2i(i32(vi) * 2, row), 0);
  let t1 = textureLoad(tree_verts, vec2i(i32(vi) * 2 + 1, row), 0);
  let fr = textureLoad(tree_verts, vec2i(0, row + ${TREE_LEVELS - 1}), 0); // the views' middle and reach
  let i = view / ${IMP_VIEWS}u; let j = view % ${IMP_VIEWS}u;
  let w = imp_view(i, j); let a = imp_across(w); let up = cross(a, w);
  let p = t0.xyz - vec3f(0.0, fr.x, 0.0);
  let t = vec2f(dot(p, a), dot(p, up)) / fr.y;
  let uv = (vec2f(f32(i), f32(j)) + 0.5 + 0.5 * vec2f(t.x, -t.y)) / ${IMP_VIEWS.toFixed(1)};
  o.pos = vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.5 - 0.5 * dot(p, w) / fr.y, 1.0);
  let c = tree_bytes(t0.w);
  o.col = c * c;
  o.m = p; o.n = tree_bytes(t1.x) * 2.0 - 1.0;
  let flag = tree_bytes(t1.y);
  o.k = vec3f(flag.x, tree_bytes(t1.z).x, 0.5 + 0.5 * dot(p, w) / fr.y); // alpha, mark, depth towards the eye
  o.view = view;
  return o;
}
struct BO { @location(0) c: vec4f, @location(1) n: vec4f }
@fragment fn solid(v: BV) -> BO {
  var o: BO;
  let w = imp_view(v.view / ${IMP_VIEWS}u, v.view % ${IMP_VIEWS}u);
  let face = cross(dpdx(v.m), dpdy(v.m));
  var n = select(w, normalize(face), dot(face, face) > 1e-24);
  if (dot(n, w) < 0.0) { n = -n; }
  o.c = vec4f(v.col, 1.0);
  o.n = vec4f(imp_oct(n), v.k.z, 1.0);
  return o;
}
// the blended part: twig haze as thick as it is seen on average (tree_twigs, its grain evened out), snow at the foot whole
@fragment fn haze(v: BV) -> BO {
  var o: BO;
  let w = imp_view(v.view / ${IMP_VIEWS}u, v.view % ${IMP_VIEWS}u);
  let face = abs(dot(normalize(v.n), w));
  o.c = vec4f(v.col, v.k.x * select(0.85 * face * face, 1.0, v.k.y > 0.5));
  o.n = vec4f(0.0, 0.0, 0.0, o.c.a);
  return o;
}
`;
// Each mip of both textures from the one above: cover averaged, colour, normal and depth weighted by it
export const IMPOSTOR_MIP = /* wgsl */`
@group(0) @binding(0) var col: texture_2d<f32>;
@group(0) @binding(1) var nrm: texture_2d<f32>;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f { return vec4f(f32(i & 1u) * 4.0 - 1.0, f32(i >> 1u) * 4.0 - 1.0, 0.0, 1.0); }
struct BO { @location(0) c: vec4f, @location(1) n: vec4f }
@fragment fn fs(@builtin(position) pos: vec4f) -> BO {
  var o: BO;
  var c = vec4f(0.0); var n = vec4f(0.0);
  for (var k = 0; k < 4; k++) {
    let at = vec2i(pos.xy) * 2 + vec2i(k & 1, k >> 1);
    let a = textureLoad(col, at, 0); let b = textureLoad(nrm, at, 0);
    c += vec4f(a.rgb * b.a, b.a); n += vec4f(b.rgb * b.a, b.a);
  }
  o.c = vec4f(c.rgb / max(c.a, 1e-6), 1.0);
  o.n = vec4f(select(vec3f(0.5), n.rgb / max(n.a, 1e-6), n.a > 1e-6), n.a * 0.25);
  return o;
}
`;
// The trees themselves. A tree is a quad facing the eye in its model's own space (unscaled, unturned, unleaned), so the
// views, baked there, fit it however it is placed.
export const forestImpostor = samples => FOREST_IN.replace(/fn tree_vertex[\s\S]*$/, '') + IMP_VIEW + BAYER4 + KEEP + /* wgsl */`
@group(1) @binding(0) var tree_verts: texture_2d<f32>;
@group(1) @binding(1) var<storage, read> trees: array<Tree>;
@group(1) @binding(2) var imp_col: texture_2d_array<f32>;
@group(1) @binding(3) var imp_nrm: texture_2d_array<f32>;
@group(1) @binding(4) var imp_samp: sampler;
@group(1) @binding(5) var imp_views: texture_2d<f32>; // impViews(): each view's way; its across IMP_VIEWS texels on, its up as many again
` + forestFade + /* wgsl */`
${litOver('lit_far', true)}
const WIND = vec3f(0.707, 0.0, -0.707); // as tree_vertex leans
// tree_turn by the tree's yaw, whose cosine and sine (cs) the vertex shader takes once a corner and hands on: the pixels
// take none. Back the other way the sine changes sign.
fn turn_by(p: vec3f, cs: vec2f) -> vec3f { return vec3f(cs.x * p.x + cs.y * p.z, p.y, cs.x * p.z - cs.y * p.x); }
fn to_model(q: vec3f, i: Tree, cs: vec2f) -> vec3f { return turn_by(q - WIND * (i.form.z * q.y), vec2f(cs.x, -cs.y)) / i.form.xyx; }
fn to_world(l: vec3f, i: Tree, cs: vec2f) -> vec3f { let s = l * i.form.xyx; return turn_by(s, cs) + WIND * (i.form.z * s.y); }
// From the vertex shader: where on the quad, what the eye's ray is, the blend, and which four views: little, as every
// corner of every tree carries it. A pixel looks the views' frames up (imp_views).
struct IV { @builtin(position) pos: vec4f, @location(0) m: vec3f,
  @location(1) @interpolate(flat) k: vec4f, // the blend across and up the grid, the shows of a hand-over, the tree (bits)
  @location(2) @interpolate(flat) views: u32, // the first of the four views' tile, i * IMP_VIEWS + j; bit 16: a moving light can reach the tree
  @location(3) @interpolate(flat) cs: vec2f } // the cosine and sine of its yaw
// the eye from the views' middle, in the model's space: the same for every corner and every pixel of a tree
fn imp_eye(tree: Tree, cs: vec2f, mid: f32) -> vec3f { return to_model(F.cam_pos.xyz - tree.at.xyz, tree, cs) - vec3f(0.0, mid, 0.0); }
// What the pixels take flat (k, views, cs) they take from the first corner of their triangle, and both of a tree's
// triangles begin at its corner 0 (src/worker.js, treeQuad; corner 3 where six are drawn without it): only that corner
// works them out, the blend and the four views, how far the tree's hand-over has come, whether a moving light reaches it.
// The other corners only stand where they belong.
@vertex fn vs(i: TIn) -> IV {
  var o: IV;
  let tree = trees[i.pick & 0xffffffu];
  let fr = textureLoad(tree_verts, vec2i(0, i32(tree.form.w) * ${TREE_LEVELS} + ${TREE_LEVELS - 1}), 0);
  o.cs = vec2f(cos(tree.at.w), sin(tree.at.w));
  let eye = imp_eye(tree, o.cs, fr.x);
  let w = normalize(eye);
  var a = vec3f(w.z, 0.0, -w.x);
  a = select(vec3f(1.0, 0.0, 0.0), normalize(a), dot(a, a) > 1e-8);
  let b = cross(w, a);
  let corner = vec2f(f32(i.vi == 1u || i.vi == 3u || i.vi == 4u), f32(i.vi == 2u || i.vi >= 4u)) * 2.0 - 1.0; // two triangles
  // cut to the upright cylinder round the model as it is seen from here: far narrower than the sphere for a tall tree
  let tall = min(fr.y, fr.w * abs(b.y) + fr.z * sqrt(max(0.0, 1.0 - b.y * b.y)));
  o.m = a * (corner.x * min(fr.y, fr.z)) + b * (corner.y * tall);
  if (i.vi == 0u || i.vi == 3u) {
    let g = imp_grid(w);
    let g0 = min(floor(g), vec2f(${(IMP_VIEWS - 2).toFixed(1)}));
    // (the bits above the level say how much larger a far tree under half a pixel is drawn, for those of its cell that are not)
    o.k = vec4f(g - g0, tree_shows(i32((i.pick >> 24u) & 3u), distance(tree.at.xyz, F.cam_pos.xyz), tree), bitcast<f32>(i.pick & 0x3ffffffu));
    var facing = 1.0;
    for (var k = 0u; k < 4u; k++) {
      let vi = u32(g0.x) + (k & 1u); let vj = u32(g0.y) + (k >> 1u);
      facing = min(facing, dot(w, textureLoad(imp_views, vec2u(vi, vj), 0).xyz));
    }
    // How far from the tree's foot the point a pixel lights (fs: at / cover) can lie. A view's point is the quad's, no
    // further from the middle than its corner, moved along the eye's ray by (z - m.w) / min(d.w, -0.05): z within the reach,
    // and d.w, the ray on the view's way, no nearer nothing than the worst view's on the ray to the middle, less how far a
    // ray to the quad's corner turns from that one (under 2 |m| / |eye|). Then scaled and leant as to_world does.
    let corner_m = length(vec2f(min(fr.y, fr.z), tall));
    let along = max(0.05, facing - 2.0 * corner_m / length(eye));
    let far = (corner_m + (fr.y + corner_m) / along + abs(fr.x)) * max(abs(tree.form.x), abs(tree.form.y)) * (1.0 + abs(tree.form.z));
    o.views = (u32(g0.x) * ${IMP_VIEWS}u + u32(g0.y)) | (u32(lights_reach(tree.at.xz, far)) << 16u);
  }
  // A cell of trees each under half a pixel is drawn as fewer of them, each so much larger from its foot (forest.h,
  // FOREST_THIN_STEP): the quad alone, which shows the tree it is and is lit where that tree stands.
  o.pos = F.view_proj * vec4f(tree.at.xyz + to_world(o.m + vec3f(0.0, fr.x, 0.0), tree, o.cs) * exp2(f32(i.pick >> 26u) * 0.125), 1.0);
  return o;
}
// where a point (model space, from the views' middle) lies in a view's tile, across and up its picture scaled by its reach
fn imp_at(p: vec3f, tile: u32, a: vec3f, u: vec3f) -> vec2f {
  let t = clamp(vec2f(dot(p, a), -dot(p, u)), vec2f(-0.98), vec2f(0.98));
  return (vec2f(f32(tile / ${IMP_VIEWS}u), f32(tile % ${IMP_VIEWS}u)) + 0.5 + 0.5 * t) / ${IMP_VIEWS.toFixed(1)};
}
struct IO { @location(0) c: vec4f, @builtin(sample_mask) mask: u32 }
@fragment fn fs(v: IV) -> IO {
  var o: IO;
  let pick = bitcast<u32>(v.k.w);
  let tree = trees[pick & 0xffffffu];
  let layer = u32(tree.form.w);
  let fr = textureLoad(tree_verts, vec2i(0, i32(layer) * ${TREE_LEVELS} + ${TREE_LEVELS - 1}), 0);
  let d = normalize(v.m - imp_eye(tree, v.cs, fr.x)); // along the eye's ray, in the model's space from the views' middle
  // texels a pixel spans, for the mip
  let lod = clamp(log2(max(length(dpdx(v.m)), length(dpdy(v.m))) / (2.0 * fr.y) * ${IMP_TILE.toFixed(1)}), 0.0, ${(IMP_MIPS - 1).toFixed(1)});
  // each view's frame: its way towards its eye, and across and up its picture over the reach (so a model-space point
  // gives its place in the tile at once)
  var fw: array<vec3f, 4>; var fa: array<vec3f, 4>; var fu: array<vec3f, 4>; var tiles: array<u32, 4>;
  for (var k = 0u; k < 4u; k++) {
    tiles[k] = (v.views & 0xffffu) + (k & 1u) * ${IMP_VIEWS}u + (k >> 1u); // one on across the grid, one on up it
    let t = vec2u(tiles[k] / ${IMP_VIEWS}u, tiles[k] % ${IMP_VIEWS}u);
    fw[k] = textureLoad(imp_views, t, 0).xyz; // its across and up from its way, as impViews() makes them, not two looks more
    let a = normalize(vec3f(-fw[k].z, 0.0, fw[k].x));
    fa[k] = a / fr.y; fu[k] = cross(a, fw[k]) / fr.y;
  }
  // Where the eye's ray meets the middle plane, each view's depth (from a mip coarser, where the parallax cannot reach past
  // what it covers): most of the quad is empty, and is let go after that one look.
  // Where a pixel spans four texels or more, how far the views' depths would move the look-up is under a pixel: then that
  // first look is the one kept.
  let small = lod >= 2.0;
  var e: array<vec4f, 4>; var any = 0.0;
  for (var k = 0u; k < 4u; k++) {
    e[k] = textureSampleLevel(imp_nrm, imp_samp, imp_at(v.m, tiles[k], fa[k], fu[k]), layer, select(min(lod + 1.0, ${(IMP_MIPS - 1).toFixed(1)}), lod, small));
    any += e[k].a;
  }
  if (any < 0.002) { discard; }
  var cover = 0.0; var col = vec3f(0.0); var nrm = vec3f(0.0); var at = vec3f(0.0);
  for (var k = 0u; k < 4u; k++) {
    let wt = select(1.0 - v.k.x, v.k.x, (k & 1u) == 1u) * select(1.0 - v.k.y, v.k.y, (k >> 1u) == 1u);
    let w = fw[k]; // the view's way towards its eye
    let z = (e[k].b * 2.0 - 1.0) * fr.y;
    var p = v.m + d * ((z - dot(v.m, w)) / min(dot(d, w), -0.05)); // where the ray meets what this view saw
    var n = e[k];
    if (small) { if (n.a == 0.0) { continue; } p = v.m; } // (a view that saw nothing here adds nothing: its colour is not read)
    else { n = textureSampleLevel(imp_nrm, imp_samp, imp_at(p, tiles[k], fa[k], fu[k]), layer, lod); }
    let c = textureSampleLevel(imp_col, imp_samp, imp_at(p, tiles[k], fa[k], fu[k]), layer, lod);
    cover += wt * n.a; col += wt * n.a * c.rgb; nrm += wt * n.a * imp_unoct(n.rg); at += wt * n.a * (v.m + d * ((z - dot(v.m, w)) / min(dot(d, w), -0.05)));
  }
  // Cut out by its cover as the see-through twigs are (forest()): as many samples as it covers; and fading in, of those
  // only the ones the far level fading out leaves.
  let q = vec2i(v.pos.xy) & vec2i(3);
  o.mask = keep(cover, (vec2i(v.pos.xy) + vec2i(1, 2)) & vec2i(3), ${samples}u);
  if (v.k.z < 0.0) { o.mask &= ~keep(1.0 + v.k.z, q, ${samples}u); }
  if (cover < 0.004 || o.mask == 0u) { discard; }
  let w = tree.at.xyz + to_world(at / cover + vec3f(0.0, fr.x, 0.0), tree, v.cs);
  var n = normalize(turn_by(select(vec3f(0.0, 1.0, 0.0), normalize(nrm), dot(nrm, nrm) > 1e-12) / tree.form.xyx, v.cs));
  let to_eye = normalize(F.cam_pos.xyz - w);
  let albedo = col / cover * (0.78 + 0.4 * fract(tree.form.w));
  let above = w.y - ground_at(w.xz);
  let reach = lamp_reach_over(w, above);
  // no wind relief or glints on the snow: a kilometre off they are far under a pixel; the rime whitens as the models' does
  rime_on = ${RIME.tree.toFixed(2)};
  lit_moving = (v.views >> 16u) != 0u;
  var c = lit_far(albedo, n, w, 0.9, 0.0, 0.0, vec2f(0.0), vec2f(0.0), ${TREE_MOON_SHADE}, reach, 1.0, above); // (no snow cover: the pixel's span on the ground is not asked for)
  let frost = smoothstep(0.25, 0.55, dot(albedo, vec3f(0.333)));
  let rim = pow3(1.0 - abs(dot(n, to_eye)));
  c += albedo * (frost * rim * 1.6) * (reach * lamp_tilt(n));
  o.c = vec4f(finish(c, w), 1.0);
  return o;
}
`;
// Shadows: each tree casts from exactly one level, the one it is drawn with at the middle of a hand-over (far trees
// cast none); the C lists them for each map by the map's own eye: the camera for the headlamps' maps, where the moon's
// map was last centred for that.
export const FOREST_SHADOW = /* wgsl */`
struct Cast { vp: mat4x4f, ground_rect: vec4f, ground_tiles: vec4f, ground_back: vec4f }
@group(0) @binding(0) var<uniform> C: Cast;
@group(1) @binding(0) var tree_verts: texture_2d<f32>;
@group(1) @binding(1) var ground_map: texture_2d_array<f32>;
@group(1) @binding(2) var samp_clamp: sampler;
@group(1) @binding(3) var<storage, read> trees: array<Tree>;
@group(1) @binding(4) var ground_tiles: texture_2d<i32>;
@group(1) @binding(5) var ground_back: texture_2d_array<f32>;
${GROUND_LOOK}
fn tree_ground(xz: vec2f) -> f32 { return ground_look(ground_map, ground_tiles, ground_back, samp_clamp, C.ground_rect, C.ground_tiles, C.ground_back, xz); }
` + FOREST_IN + /* wgsl */`
@vertex fn thrown(i: TIn) -> @builtin(position) vec4f { return C.vp * vec4f(tree_vertex(i.vi, trees[i.pick & 0xffffffu], i32(i.pick >> 24u)).w, 1.0); }
`;

// lamp posts: unlit, each head's colour is its glow
export const LAMP = MESH_IN + INSTANCED + /* wgsl */`
const GFX_LENS_LAMP = 14.0;
@vertex fn vs(i: VIn, inst: IIn) -> V {
  var o: V;
  let w = place(i.pos, inst);
  o.pos = F.view_proj * vec4f(w, 1.0);
  o.w = w; o.n = vec3f(0.0, 1.0, 0.0); o.col = unpack(i.col) * unpack(inst.tint); o.flag = round(i.nrm.w * 127.0);
  return o;
}
// The head's lens (GFX_LENS_LAMP) is white GFX_LAMP_LENS_WHITE times the lamp's tint, whose luminance is the share of its light
// this lamp burns (road.c: its power); drawn over its own light as the brighter, at its true radiance, as the sprite is far off
@fragment fn fs(v: V) -> @location(0) vec4f {
  let own = finish(v.col, v.w);
  if (v.flag < GFX_LENS_LAMP - 0.5) { return vec4f(own, 1.0); }
  return vec4f(max(own, lens_burn(v.col, v.w, ${CD_LAMP.toFixed(1)}, dot(v.col, vec3f(0.2126, 0.7152, 0.0722)) / ${LENS_WHITE.toFixed(1)})), 1.0);
}
`;

// walls: a facade texture and its glow, and the rooms behind its windows (ROOM_FNS, with the shops' and cabins' glass)
export const WALL = ROOM_FNS + /* wgsl */`
@group(1) @binding(0) var facades: texture_2d_array<f32>;
@group(1) @binding(1) var glows: texture_2d_array<f32>;
@group(1) @binding(2) var samp_wall: sampler;
@group(1) @binding(3) var styles: texture_2d_array<f32>;
@group(1) @binding(4) var style_glows: texture_2d_array<f32>;
struct VIn { @location(0) pos: vec3f, @location(1) nrm: vec4f, @location(2) col: vec4f, @location(3) uv: vec2f }
struct V { @builtin(position) pos: vec4f, @location(0) w: vec3f, @location(1) n: vec3f, @location(2) col: vec3f, @location(3) uv: vec2f,
  @location(4) @interpolate(flat) layer: u32, @location(5) @interpolate(flat) start: vec4f } // start: the quad's first corner and its u
@vertex fn vs(i: VIn) -> V {
  var o: V;
  o.pos = F.view_proj * vec4f(i.pos, 1.0);
  o.w = i.pos; o.n = i.nrm.xyz; o.col = unpack(i.col); o.uv = i.uv; o.layer = u32(round(i.nrm.w * 127.0));
  o.start = vec4f(i.pos, i.uv.x); // both of a quad's triangles are drawn from its first corner (soup_quad), so the quad has one
  return o;
}
@fragment fn fs(v: V) -> @location(0) vec4f {
  let dx = dpdx(v.uv); // taken here, where every pixel passes: the layer picks the array below
  let dy = dpdy(v.uv);
  let cell = floor(v.uv * 4.0); // the cell of the atlas: windows in rows 0 to 2
  var t: vec4f; var glow: vec4f; var blurred: vec3f; var mean: vec3f;
  if (v.layer < 2u) {
    t = textureSampleGrad(facades, samp_wall, v.uv, v.layer, dx, dy);
    glow = textureSampleGrad(glows, samp_wall, v.uv, v.layer, dx, dy);
    blurred = textureSampleLevel(glows, samp_wall, v.uv, v.layer, 4.5).rgb;
    mean = textureSampleLevel(glows, samp_wall, (cell + 0.5) * 0.25, v.layer, 8.0).rgb; // 4 x 4 texels: one a cell
  } else {
    t = textureSampleGrad(styles, samp_wall, v.uv, v.layer - 2u, dx, dy);
    glow = textureSampleGrad(style_glows, samp_wall, v.uv, v.layer - 2u, dx, dy);
    blurred = textureSampleLevel(style_glows, samp_wall, v.uv, v.layer - 2u, 4.5).rgb;
    mean = textureSampleLevel(style_glows, samp_wall, (cell + 0.5) * 0.25, v.layer - 2u, 7.0).rgb;
  }
  let albedo = t.rgb * mix(vec3f(1.0), v.col, t.a); // tint the cladding but leave window frames and glass alone
  let n = normalize(v.n);
  // what of the sky the wall's own shape leaves it, which bld.c gives each corner as the length of its normal
  let above = v.w.y - ground_at(v.w.xz);
  let reach = lamp_reach_wall(v.w, above);
  var c = lit_over(albedo, n, v.w, 0.5, -1.0, 1.0, dpdx(v.w.xz), dpdy(v.w.xz), 0.0, reach, ${OFF.has('sky') ? '1.0' : 'min(length(v.n), 1.0)'}, above);
  var light = glow.rgb * 1.45;
  if ((${ROOMED}) && cell.y - 4.0 * floor(cell.y * 0.25) < 3.0) {
    let id = room_hash(bitcast<u32>(v.start.x) ^ room_hash(bitcast<u32>(v.start.y) ^ room_hash(bitcast<u32>(v.start.z)
      ^ room_hash(u32(i32(cell.x) + 65536) ^ (u32(i32(cell.y) + 256) << 20u) ^ (v.layer << 29u)))));
    let kind = room_rand(id, 3u);
    let office = (mean.b > mean.r * 0.85 && mean.b < mean.r * 1.4) || v.layer >= 10u; // the office, works and clapboard facades' rooms, lit or not
    // the room as it was painted, lit or dark (the evening's), and whether by the clock its lamps are on now: when not,
    // it is a dark room, with what light a dark room has
    let home = room_hash(bitcast<u32>(v.start.x) ^ room_hash(bitcast<u32>(v.start.z)))
      ^ select(0u, room_hash(bitcast<u32>(v.start.y) ^ (u32(i32(cell.y) + 256) << 20u)), v.layer != 0u);
    let painted = max(mean.r, max(mean.g, mean.b)) >= 0.015;
    let on = select(room_on(home, id), office_on(id), office && v.layer != 12u); // (the clapboard's rooms are homes)
    // m of wall a pixel spans. Where a pixel spans more than a cell, the glow's mip is the mean of the cells' painted
    // glass, and the share of them lit is the town's (F.sky_now.z), not the one cell its centre falls in: lit windows
    // far off are their mean light, and do not flicker on and off as the eye moves
    let fm = max(length(dx * vec2f(4.0 * ROOM_W, 4.0 * ROOM_H)), length(dy * vec2f(4.0 * ROOM_W, 4.0 * ROOM_H)));
    let many = smoothstep(${f4(ROOM.many[0])}, ${f4(ROOM.many[1])}, fm / ROOM_W);
    let now = mix(select(0.0, 1.0, on || !painted), F.sky_now.z, many);
    glow = vec4f(glow.rgb * now, glow.a); light *= now;
    let mean_now = mean * now;
    var live = room_live(id, F.cam_pos.w, mean_now, kind);
    // Christmas: some homes' windows are a dark room with only an Advent star alight in it, and some of the dark ones too
    let lit_cell = painted;
    let star_only = !office && lit_cell && room_rand(id, 21u) < ${f4(ROOM.star_room)};
    let star = star_only && string_on(id); // its star on its timer // a dark window keeps only what it was painted with: a star, or candles, never both and a string
    if (star_only) { live = vec4f(live.rgb * ${f4(ROOM.hall)}, 0.0); } // the lamps off; only the star's light in it
    else { live = vec4f(live.rgb * ${f4(ROOM.bright)}, live.w); }
    let eye = v.w - F.cam_pos.xyz; let dist = length(eye); let ray = eye / dist;
    let pa = pixel_angle();
    // the glass reflects more and shows less of the room the flatter it is seen
    let shown = 1.0 - glass_fresnel(ray, n); // what the glass does not mirror it lets through
    var inside = glow.rgb * live.rgb + select(ROOM_WARM, ROOM_COOL, office) * (live.w * glow.a * 0.5); // the room's light, as the glass shows it
    let f = fract(v.uv * 4.0) * vec2f(ROOM_W, ROOM_H); // m across the cell the way u runs, and up from its floor
    let detail = smoothstep(${f4(ROOM.detail[0])}, ${f4(ROOM.detail[1])}, 1.0 / fm) * glow.a;
    // A lit window lights its own casing, sill and reveal a little: its glow, blurred, falling on them. Only on them: the
    // untinted frame (the colour's alpha 0) within ROOM.frame of the opening, as much of the pixel as falls there, so the
    // wall round a window keeps its own light and no halo spreads over it near or far.
    let op = room_opening(v.layer) * vec4f(ROOM_W, ROOM_H, ROOM_W, ROOM_H);
    let rim = room_box(f, op.xy - ${f4(ROOM.frame)}, op.zw + ${f4(ROOM.frame)}, 0.5 * fm) * (1.0 - t.a);
    let reveal = albedo * blurred * (0.3 * rim);
    var dark = 1.0; var near_glass = 1.0;
    if (detail > 0.0) {
      let side = normalize(vec2f(-n.z, n.x));
      let along = select(-side, side, dot(v.w.xz - v.start.xz, side) * (v.uv.x - v.start.w) >= 0.0);
      let r = room(vec3f(f, 0.0), vec3f(dot(ray.xz, along), ray.y, -dot(ray, n)), id, kind, office, dist * pa, pa);
      inside = mix(inside, inside * r.rgb, detail);
      dark = mix(1.0, r.w, detail); near_glass = dark;
    }
    // what comes in from outside, the street's lamps and the moon, and shows a dark room dimly: brightest by the glass
    let outside = (reach * ${f4(ROOM.outside)} + moon_now() * 0.05 + sky_up(v.w.xz) * 0.04) * glow.a * near_glass;
    // A lit window gives its true light (ROOM.lift of what its glow is painted at) at every distance: near by, a warm
    // pane far brighter than the street-lit snow; far off, averaged over the pixel with the wall round it by the mips,
    // so that its light on the pixel is its own times the share of the pixel it covers: a lit window a few kilometres off
    // is a bright point, not the grey of its wall. Its room, its star and its string alike; what the glass reflects is
    // not its light.
    let true_light = ${f4(ROOM.lift)};
    light = ((inside * 1.45 + reveal * (live.rgb * now)) * true_light + outside) * shown + glass_sheen(ray, n, reach, v.w) * glow.a;
    if (star) { light += room_star(id, f, room_opening(v.layer), max(fm, dist * pa)) * (shown * true_light); }
    if (lit_cell && !star_only && string_on(id) && room_rand(id, 8u) < select(${f4(ROOM.string)}, ${f4(ROOM.office_string)}, office)) { // strings on their timers
      light += room_string(id, f, room_opening(v.layer), F.cam_pos.w, max(fm, dist * pa)) * true_light;
    }
    c *= dark;
  }
  // Far off a window is its true light averaged over the pixel with the frame and wall round it (the mip chain does
  // that): no dimmer for being far but as its share of the pixel falls (the air dims it in finish, as it does everything).
  c += light;
  return vec4f(finish(c, v.w), 1.0);
}
`;

// ---- the salted roads' wet wheel tracks (wasm/include/harstad/wet.h) ----
// At about -2 C the salt keeps the wheel tracks of the main roads wet black asphalt, and they mirror the street lamps,
// the cars' lamps, the lit shop windows, the moon and the sky, each light drawn out into a long broken streak as on the
// fjord. A wet track is a film of water over the asphalt's grain: its facets are tilted about the road's own slope by a
// gaussian of RMS slope WET.sigma (calm in a puddle, rougher on the rest of a track), about a mean the grain ripples (two
// tilings of the relief map, WET.ripple), whose variance is kept as the mip chain averages it away. Each light's
// reflection is the share of the facets turned to send its light to the eye, as the sea's glint (glint_over): seen from
// a low eye a tilt of a few degrees along the view spans a long way up the road and across it a short one, so each light
// is drawn out towards the eye. The packed snow of the side streets and the slush between the tracks stay matte: how wet
// a texel is comes from how dark the road texture is there (bare asphalt), so the wetness follows the tracks it paints.
//
// The lights are found tile by tile, WET.tile pixels a side (WET_BIN, in the frame's compute passes): each light's strip
// on the road, cut where it would fall under WET.faint however rough the film (as STREAK_FIT cuts the sea's), is laid
// into the tiles it covers, row by row, the brightest WET.cap at most a tile; a pixel sums only its own tile's. The lights:
// those gathered near the salted roads (wasm/src/wet.c) and the shop windows, then the frame's own, made each frame from
// its uniforms: the traffic's headlamps and tail lamps, the Christmas truck's lights and the car's own lamps.
const WET = {
  tile: 32, cap: 32, bands: 24, // pixels a tile side; lights a tile; bands of brightness the lights are counted in, a doubling each
  sigma: [0.05, 0.09],        // RMS slope of the film's facets: in a puddle, on the rest of a track: damp salted asphalt, not a mirror
  spread: 0.003,              // the most slope variance a pixel's spread adds (as the strips are cut for it)
  clear: 6, low: 1, soft: 2,  // m from a light its own wall is not asked about; m over the line that is let by (kerbs, rails, a
                              // square's clutter beside the road); m more over it a top closes the way by
  wet: [0.55, 0.8], sky: 0.4, // how wet a track is, and in a puddle; how much of the sky the film gives back
  ripple: [0.9, 0.37], amp: [0.045, 0.025], // the grain: two tilings of the relief map, m, and the RMS-ish slope each gives
  margin: 0.05,               // the most tilt the grain and the road's own slope add to a light's strip
  faint: 0.0003,              // linear light below which a strip is cut: a fraction of the least step the display shows (STREAK_FAINT)
  stations: 24,               // along each light's strip, in WET_BIN
  head: 8.0, tail: TAIL.wet, truck: 4.0, // power of a dipped headlamp and of a tail lamp at full strength (times their colours); the truck's (times theirs)
};
export const WET_TILE = WET.tile, WET_CAP = WET.cap, WET_BANDS = WET.bands, WET_STATIONS = WET.stations, WET_MADE = SHINE.cars * 4 + 4 + 4; // the frame's own lights: every car's four (the light list), the truck's, the car's
const WET_IN = /* wgsl */`
struct WetLight { at: vec4f, power: vec4f, aim: vec4f, lens: vec4f } // struct wet_light: where and how wide; power, the ground under it; its axis, cone; the radiance its lens is drawn at
// the slope variance a pixel's film can have: the calmest puddle's, and the roughest track's with all the grain the mips
// average away and the most a pixel's spread adds. The strips are cut for this, and no pixel is given more.
const WET_VR = vec2f(${(WET.sigma[0] ** 2).toExponential(4)}, ${(WET.sigma[1] ** 2 + 0.3 * (WET.amp[0] ** 2 + WET.amp[1] ** 2) + WET.spread).toExponential(4)});
fn wet_fresnel(c: f32) -> f32 { let m = 1.0 - clamp(c, 0.0, 1.0); let m2 = m * m; return 0.02 + 0.98 * m2 * m2 * m; }
fn wet_aim(l: WetLight, d: vec3f) -> f32 { // how much of a light shines along d, away from it
  if (l.aim.w < -1.5) { return 1.0; }
  return saturate((dot(d, l.aim.xyz) - l.aim.w) / (1.0 - l.aim.w));
}
fn wet_tiles_x() -> u32 { return (u32(F.cam_fwd.w) + ${WET.tile - 1}u) / ${WET.tile}u; }
fn wet_tiles_n() -> u32 { return wet_tiles_x() * ((u32(F.cam_up.w) + ${WET.tile - 1}u) / ${WET.tile}u); }
`;
// Each frame: the frame's own lights made from its uniforms, then every light's strip binned into the tiles it can show in.
export const WET_BIN = WET_IN + /* wgsl */`
@group(1) @binding(0) var<storage, read_write> wet_lights: array<WetLight>;
@group(1) @binding(1) var<storage, read_write> wet_tiles: array<atomic<u32>>; // a count a tile, WET.cap slots a tile, then WET.bands a tile
@group(1) @binding(2) var<storage, read_write> wet_live: array<atomic<u32>>; // how many lights are worth laying (the next passes' workgroups), 1, 1, 0, then which
@group(1) @binding(3) var<storage, read_write> wet_vis: array<f32>; // a light's stations: how open the way from it to the road there is
@group(1) @binding(4) var tops: texture_2d_array<f32>; // how far the highest thing in each 4 m square stands over the ground (wasm/src/moon.c), g of the
// (the moon's reach and tops in tiles; their page table TABLES.reach, pair_look)
// How much of the way from a light at a down to the road at b is open: the town's tops (roofs, crowns, decks, the
// ground) sampled every few metres along it, each standing more than WET.low m over the line closing it by as much
// again over WET.soft m. The light's own wall (the first WET.clear m, the tops map's squares being 4 m) and the last
// metre and a half over the road (its banks, kerbs and rails, low and near) are not asked.
fn seen(a: vec3f, b: vec3f) -> f32 {
  let run = length(b.xz - a.xz);
  if (run <= ${WET.clear + 1.5}) { return 1.0; }
  let t0 = ${WET.clear.toFixed(1)} / run; let t1 = 1.0 - 1.5 / run;
  let n = clamp(u32(run * (t1 - t0) / 4.0), 1u, 16u); // a sample a square of the map, and no more than 16 (further apart on a long way)
  var open = 1.0;
  for (var k = 0u; k < n; k++) {
    let q = mix(a, b, mix(t0, t1, (f32(k) + 0.5) / f32(n)));
    let top = ground_at(q.xz) + pair_look(tops, tab_first(${TABLES.reachFirst}u), ${TABLES.reach}u, (q.xz - F.ground_rect.xy) / F.ground_rect.zw).y * ${(255 * 0.5).toFixed(1)};
    open = min(open, saturate(1.0 - (top - q.y - ${WET.low.toFixed(1)}) / ${WET.soft.toFixed(1)}));
  }
  return open;
}
const HALOGEN = vec3f(1.0, 0.715694, 0.423268);
const TAIL_RED = vec3f(1.0, 0.01033, 0.003677);
const HEAD_LENS = vec3f(3.9, 3.7, 3.4); const TAIL_LENS = ${vec3(TAIL.lens)}; // the radiance a lit lens is drawn at in the wet road's mirror
fn made(j: u32) -> WetLight { // the j-th of the frame's own lights; nothing (no power) where there is none
  var o = WetLight(vec4f(0.0, -1.0e4, 0.0, 0.1), vec4f(0.0), vec4f(0.0, 0.0, 0.0, -2.0), vec4f(0.0));
  if (j < ${4 * SHINE.cars}u) { // every car of the traffic list (wasm/src/shine.c): dipped headlamps ahead, tail lamps behind, a pair each either side of its middle, 0.7 m over the road
    if ((j >> 2u) >= F.shine.x) { return o; }
    let L = shine_lights[j >> 2u]; let a = vec4f(L.at.xyz, 1.0); let dir = L.dir;
    let fwd = vec3f(dir.x, 0.0, dir.y); let side = vec3f(dir.y, 0.0, -dir.x) * select(-0.6, 0.6, (j & 1u) == 1u);
    if ((j & 2u) == 0u) {
      o.at = vec4f(a.xyz + fwd * 2.2 + side, 0.09); o.power = vec4f(HALOGEN * (${WET.head.toFixed(2)} * a.w * (1.0 + 0.25 * dir.w)), a.y - 0.7); o.lens = vec4f(HEAD_LENS * a.w, 0.0);
      o.aim = vec4f(normalize(fwd - vec3f(0.0, 0.073, 0.0)), 0.55); // dipped: brightest straight ahead and a little down, gone some 60 degrees off it
    } else {
      o.at = vec4f(a.xyz - fwd * 2.3 + side * 1.05 + vec3f(0.0, 0.1, 0.0), 0.06); o.power = vec4f(TAIL_RED * (${WET.tail.toFixed(3)} * dir.z * a.w), a.y - 0.7); o.lens = vec4f(TAIL_LENS * dir.z * a.w, 0.0);
      o.aim = vec4f(-fwd, 0.0);
    }
    return o;
  }
  if (j < ${4 * SHINE.cars + 4}u) { // the Christmas truck's lights, as it lays them on the water (traffic_streaks): its kingpin is at the road's level
    let a = F.truck_at[j - ${4 * SHINE.cars}u];
    if (a.w <= 0.0) { return o; }
    o.at = vec4f(a.xyz, 1.2); o.power = vec4f(F.truck[j - ${4 * SHINE.cars}u].rgb * ${WET.truck.toFixed(1)}, F.truck_at[1].y - 2.3); o.lens = vec4f(F.truck[j - ${4 * SHINE.cars}u].rgb, 0.0);
    return o;
  }
  // the car's own: its dipped halogens (car_beams), and its tail lamps as bright as they light the road behind it (F.tail)
  let k = j - ${4 * SHINE.cars + 4}u; let fwd = F.car_dir.xyz; let up = F.car_up.xyz; let left = cross(up, fwd);
  let s = select(-1.0, 1.0, (k & 1u) == 1u);
  if (k < 2u) {
    o.at = vec4f(F.car_pos.xyz + left * (s * 0.55) + up * 0.735 + fwd * 2.33, 0.09); o.power = vec4f(HALOGEN * (${WET.head.toFixed(2)} * (1.0 + ${(FULL_LENS - 1).toFixed(1)} * F.tail.w)), F.car_pos.y); o.lens = vec4f(HEAD_LENS * (1.0 + ${(FULL_LENS - 1).toFixed(1)} * F.tail.w), 0.0);
    o.aim = vec4f(normalize(fwd - up * 0.073), 0.55);
  } else {
    o.at = vec4f(F.car_pos.xyz + left * (s * 0.62) + up * 0.85 - fwd * 2.25, 0.06); o.power = vec4f(F.tail.rgb * ${(WET.tail / TAIL.pool).toFixed(4)}, F.car_pos.y); o.lens = vec4f(F.tail.rgb / max(F.tail.r, 1e-4) * TAIL_LENS.r, 0.0);
    o.aim = vec4f(-fwd, 0.0);
  }
  return o;
}
// half the difference in elevation between the light seen from s m out from its foot and the eye seen from there: the
// tilt a facet there needs to send the one to the other (STREAK_FNS)
fn tilt(s: f32, d: f32, hl: f32, hc: f32) -> f32 { return 0.5 * (atan2(hl, s) - atan2(hc, d - s)); }
// The most a facet may be tilted from the one that sends the light to the eye and still send it there at WET.faint or
// more, with k the light that reaches the eye per unit of the gaussian over slope, for any film whose slope variance lies
// between vr.x and vr.y; -1 if none
fn reach_tilt(k: f32, vr: vec2f) -> f32 {
  let need = ${WET.faint} / k; // the value of the gaussian it takes
  if (need * 6.2831853 * vr.x >= 1.0) { return -1.0; }
  let wide = -2.0 * vr.y * log(need * 6.2831853 * vr.y); // out past sqrt(2 vr.y) the roughest film holds the most
  if (wide >= 2.0 * vr.y) { return sqrt(wide); }
  let mid = 1.0 / (2.7182818 * 3.1415927 * need); // in between, the film whose sigma is the tilt over root two
  if (mid >= 2.0 * vr.x) { return sqrt(mid); }
  return sqrt(max(-2.0 * vr.x * log(need * 6.2831853 * vr.x), 0.0));
}
// half the strip's width s m out from the light's foot, for facets tilted k across it (STREAK_FNS, half_width_true)
fn half_width(s: f32, d: f32, hl: f32, hc: f32, k: f32) -> f32 {
  let dc = length(vec2f(d - s, hc)); let dl = length(vec2f(s, hl));
  return k * (hc / dc + hl / dl) * dl * dc / (dl + dc);
}
// Twice over: first (count) every light's strip is counted into the tiles it covers, in bands by how bright the light
// can be (a band each doubling over WET.faint); then (main) each tile takes the lights of its brightest bands as long as
// they fit in its WET.cap slots, and no more. Where the street's lights all lie down the road at once, a far stretch of
// it has hundreds: it keeps the brightest, and the same ones every frame, whatever order the lights come in. Each kept
// light's slot carries its band (the top five bits), and the road fades a band out across a tile next to one that had to
// leave it out (wet_least), so a crowded tile is never a block.
// Then a workgroup a light: a thread a station works out the strip, then the threads share its rows of tiles. A near lamp's
// strip covers thousands of tiles; one thread laying them all, one after another, held up the whole frame.
const S = ${WET.stations};
var<workgroup> one: WetLight;
var<workgroup> y0: array<f32, S>; var<workgroup> y1: array<f32, S>; var<workgroup> x0: array<f32, S>; var<workgroup> x1: array<f32, S>;
var<workgroup> kept: array<u32, S>; // bits: 1 kept; 2 under or behind the eye (at one of the heights of road)
var<workgroup> shine: array<f32, S>;
var<workgroup> away: u32; // the light lays no tile, however open its way (count finds it, main is told by its first station)
// First, a thread a light, the frame's own lights are made, and the lights that could show at all are listed for the
// two passes after, a workgroup each.
@compute @workgroup_size(64) fn cull(@builtin(global_invocation_id) gid: vec3u) {
  let total = arrayLength(&wet_lights); let fixed = total - ${WET_MADE}u; let i = gid.x;
  if (i >= total) { return; }
  var l = wet_lights[i];
  if (i >= fixed) { l = made(i - fixed); wet_lights[i] = l; }
  let shut = i < fixed && l.lens.w > 0.5 && F.town_now.w < 0.5; // a display lit only while the shops are (struct wet_light, shops)
  let lum = select(dot(l.power.rgb, vec3f(0.30, 0.59, 0.11)), 0.0, shut);
  let ground = l.power.w;
  let hc = F.cam_pos.y - ground; let hl = l.at.y - ground;
  let dv = F.cam_pos.xz - l.at.xz; let d = length(dv);
  // from at or under the road the light stands over, none of it shows there
  if (lum <= 0.0 || hc < 0.15 || hl < 0.05 || d < 0.3) { return; }
  // How bright the strip's brightest pixel could be: the light seen at the mirror point, through the calmest film (the
  // gaussian's peak), its Fresnel and the haze, as STREAK_FIT tells the sea's. Most of the town's lights are dim and far:
  // under WET.faint there, they are not worked out any further.
  let u = dv / d;
  let sm = d * hl / (hl + hc); let dm = length(vec2f(sm, hl)); let em = length(vec2f(d - sm, hc)); let vy = hc / em;
  let p = vec3f(l.at.x + u.x * sm, ground, l.at.z + u.y * sm);
  let peak = lum * wet_aim(l, (p - l.at.xyz) / dm) / (dm * dm) * wet_fresnel(vy) / (4.0 * max(vy, 0.01) * 6.2831853 * WET_VR.x)
    * exp(-F.fog.a * F.fog.a * (dm + em) * (dm + em));
  if (peak < ${WET.faint}) { return; }
  atomicStore(&wet_live[4u + atomicAdd(&wet_live[0], 1u)], i);
}
// Between, a thread a tile: the dimmest band whose lights still all fit in the tile, with all those brighter (or, where
// the brightest band alone is more than fit, that band, as many of it as do), left in the band counts' first word
@compute @workgroup_size(64) fn least(@builtin(global_invocation_id) gid: vec3u) {
  let lists = wet_tiles_n();
  if (gid.x >= lists) { return; }
  let at = lists * ${WET.cap + 1}u + gid.x * ${WET.bands}u;
  var held = 0u; var least = ${WET.bands}u;
  for (var b = ${WET.bands - 1}; b >= 0; b--) {
    let n = atomicLoad(&wet_tiles[at + u32(b)]); held += n;
    if (held > ${WET.cap}u) { if (least == ${WET.bands}u && n > 0u) { least = u32(b); } break; }
    least = u32(b);
  }
  atomicStore(&wet_tiles[at], least);
}
// A light's strip s m out from its foot, on the line to the eye's, over road at height g, through a way open by open:
// whether it shows there (1) or runs under or behind the eye (2), how bright it can be, how far across it reaches (m),
// and the pixels it covers
struct Station { kept: u32, shine: f32, reach: f32, box: vec4f }
fn station_s(d: f32, t: u32) -> f32 { return d * (0.5 - 0.5 * cos(3.1415927 * f32(t) / ${WET.stations - 1}.0)); }
// the pixels of the road w m either side of p, a little over for the pixels' own spread; or under or behind the eye
fn station_box(p: vec3f, across: vec3f, w: f32) -> Station {
  var o = Station(2u, 0.0, w, vec4f(0.0));
  let ca = F.view_proj * vec4f(p - across * w, 1.0); let cb = F.view_proj * vec4f(p + across * w, 1.0);
  if (min(ca.w, cb.w) < 0.05) { return o; } // the strip runs off the foot of the screen
  let size = vec2f(F.cam_fwd.w, F.cam_up.w);
  let qa = (vec2f(ca.x, -ca.y) / ca.w * 0.5 + 0.5) * size; let qb = (vec2f(cb.x, -cb.y) / cb.w * 0.5 + 0.5) * size;
  o.kept = 1u; o.box = vec4f(min(qa, qb) - 4.0, max(qa, qb) + 4.0);
  return o;
}
fn station(l: WetLight, lum: f32, s: f32, d: f32, g: f32, open: f32) -> Station {
  var o = Station(0u, 0.0, 0.0, vec4f(0.0));
  let hc = F.cam_pos.y - g; let hl = l.at.y - g;
  if (hc < 0.05 || hl < 0.05) { return o; } // the eye or the light at or under that road: none of it shows from there
  let u = (F.cam_pos.xz - l.at.xz) / d;
  let x = d - s; let dl = length(vec2f(s, hl)); let de = length(vec2f(x, hc)); let vy = hc / de;
  let p = vec3f(l.at.x + u.x * s, g, l.at.z + u.y * s);
  let haze = exp(-F.fog.a * F.fog.a * (dl + de) * (dl + de));
  let r = l.at.w / dl; let tl = tilt(s, d, hl, hc);
  let k = lum * wet_aim(l, (p - l.at.xyz) / dl) / (dl * dl) * wet_fresnel(vy) / (4.0 * max(vy, 0.01)) * haze * open;
  let tk = reach_tilt(k, WET_VR + r * r * 0.25) + ${WET.margin.toFixed(2)};
  if (k <= 0.0 || tk <= ${WET.margin.toFixed(2)} || abs(tl) > tk) { return o; }
  o = station_box(p, vec3f(-u.y, 0.0, u.x), half_width(s, d, hl, hc, sqrt(tk * tk - tl * tl)) + l.at.w);
  o.shine = k / (6.2831853 * (WET_VR.x + r * r * 0.25));
  return o;
}
fn station_union(a: Station, b: Station) -> Station {
  var o = Station(a.kept | b.kept, max(a.shine, b.shine), max(a.reach, b.reach), select(b.box, a.box, (a.kept & 1u) != 0u));
  if ((a.kept & b.kept & 1u) != 0u) { o.box = vec4f(min(a.box.xy, b.box.xy), max(a.box.zw, b.box.zw)); }
  return o;
}
@compute @workgroup_size(64) fn count(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) { visit(wg.x, t, false); }
@compute @workgroup_size(64) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) { visit(wg.x, t, true); }
fn visit(at_in: u32, t: u32, laying: bool) {
  if (t == 0u) { one = wet_lights[atomicLoad(&wet_live[4u + at_in])]; }
  let l = workgroupUniformLoad(&one);
  let i = atomicLoad(&wet_live[4u + at_in]);
  let lum = dot(l.power.rgb, vec3f(0.30, 0.59, 0.11));
  let dv = F.cam_pos.xz - l.at.xz; let d = length(dv);
  let u = dv / d; let across = vec3f(-u.y, 0.0, u.x);
  let size = vec2f(F.cam_fwd.w, F.cam_up.w);
  // Stations along the line from the light's foot to the eye's, closer together at either end, where the tilt turns
  // fastest. A station is kept if a facet there could send the light to the eye at WET.faint or more, through the haze
  // and at the Fresnel of the eye's slant there; the strip is as wide there as the tilt it has to spare allows.
  // Every pixel the light shows in must be in the tiles it is laid in, or it ends at a tile's edge as a block of hard
  // edges. So the road is taken at two heights: the ground the light stands on and the ground at the station (a light
  // up a bank, on a raised square or a hillside, its road a few decimetres or metres under it, has its strip, seen from
  // an eye half a metre over the road, tens of pixels from where its own ground puts it). And a strip runs on past the
  // last station it shows at, to where the tilt to spare runs out before the next, metres on along a long way: the
  // station past each end is laid too, as wide as the one beside it (the strip narrows to its end).
  // Off the screen: as it is counted, every box each station could have, whatever the way's openness (a closed way only
  // narrows a station's own box, or leaves the box of its ends' run-on, which does not ask), is found first. Where none
  // of them, nor the stretches between, nor what runs on under the eye, comes onto the screen, the light lays no tile
  // however open its way, so its ways are not asked (seen(), the costly part) and main is told to pass it by.
  if (!laying) {
    if (t < S) {
      let s = station_s(d, t);
      let p = l.at.xz + u * s;
      let gs = array<f32, 2>(l.power.w, ground_at(p));
      var o = Station(0u, 0.0, 0.0, vec4f(0.0));
      for (var h = 0u; h < 2u; h++) {
        o = station_union(o, station(l, lum, s, d, gs[h], 1.0));
        for (var n = 0u; n < 2u; n++) {
          if ((n == 0u && t == 0u) || (n == 1u && t + 1u >= ${WET.stations}u)) { continue; }
          let sn = station_s(d, select(t + 1u, t - 1u, n == 0u));
          let b = station(l, lum, sn, d, select(gs[h], ground_at(l.at.xz + u * sn), h == 1u), 1.0);
          if (b.kept == 1u) { o = station_union(o, station_box(vec3f(p.x, gs[h], p.y), across, b.reach)); }
        }
      }
      kept[t] = o.kept; x0[t] = o.box.x; y0[t] = o.box.y; x1[t] = o.box.z; y1[t] = o.box.w;
    }
    workgroupBarrier();
    if (t == 0u) {
      var ks = 0u; var lo = vec2f(1.0e9); var hi = vec2f(-1.0e9);
      for (var j = 0; j < S; j++) {
        ks |= kept[j];
        if ((kept[j] & 1u) != 0u) { lo = min(lo, vec2f(x0[j], y0[j])); hi = max(hi, vec2f(x1[j], y1[j])); }
      }
      away = select(0u, 1u, ks == 0u || ((ks & 2u) == 0u && (any(hi < vec2f(0.0)) || any(lo >= size))));
      if (away != 0u) { wet_vis[i * S] = -1.0; }
    }
    if (workgroupUniformLoad(&away) != 0u) { return; }
  } else {
    if (t == 0u) { away = select(0u, 1u, wet_vis[i * S] < 0.0); }
    if (workgroupUniformLoad(&away) != 0u) { return; }
  }
  if (t < S) {
    let s = station_s(d, t);
    let p = l.at.xz + u * s;
    let gs = array<f32, 2>(l.power.w, ground_at(p));
    // how much of the way from the light down to the road here is open: worked out once, as it is counted, and only where
    // the strip could reach were it all open
    var open = 1.0;
    if (laying) { open = wet_vis[i * S + t]; }
    else {
      if ((station(l, lum, s, d, gs[0], 1.0).kept | station(l, lum, s, d, gs[1], 1.0).kept) != 0u) {
        open = seen(l.at.xyz, vec3f(p.x, gs[1] + 0.3, p.y));
      }
      wet_vis[i * S + t] = open;
    }
    var o = Station(0u, 0.0, 0.0, vec4f(0.0));
    for (var h = 0u; h < 2u; h++) {
      var c = station(l, lum, s, d, gs[h], open);
      for (var n = 0u; n < 2u && c.kept == 0u; n++) { // an end: the strip shows at the station beside it, at this height
        if ((n == 0u && t == 0u) || (n == 1u && t + 1u >= ${WET.stations}u)) { continue; }
        let sn = station_s(d, select(t + 1u, t - 1u, n == 0u));
        let b = station(l, lum, sn, d, select(gs[h], ground_at(l.at.xz + u * sn), h == 1u), 1.0);
        if (b.kept == 1u) { c = station_union(c, station_box(vec3f(p.x, gs[h], p.y), across, b.reach)); }
      }
      o = station_union(o, c);
    }
    kept[t] = o.kept; shine[t] = o.shine;
    x0[t] = o.box.x; y0[t] = o.box.y; x1[t] = o.box.z; y1[t] = o.box.w;
  }
  workgroupBarrier();
  var under = false; var top = 1.0e9; var foot = -1.0e9; var last = -1; var bright = 0.0;
  for (var j = 0; j < S; j++) {
    bright = max(bright, shine[j]);
    if ((kept[j] & 2u) != 0u) { under = true; }
    if ((kept[j] & 1u) != 0u) { top = min(top, y0[j]); foot = max(foot, y1[j]); last = j; }
  }
  if (last < 0) { return; }
  if (under) { foot = size.y; } // and on under the eye from the last station it was in front of, as wide again either side
  let wide = x1[last] - x0[last];
  if (foot < 0.0 || top >= size.y) { return; }
  let nx = wet_tiles_x(); let lists = wet_tiles_n(); let ny = lists / nx;
  let band = u32(clamp(log2(bright / ${WET.faint}), 0.0, ${WET.bands - 1}.0)); let bands = lists * ${WET.cap + 1}u;
  // Row by row of tiles, a thread a row, the light goes in the tiles under the stretches between kept stations that cross it
  for (var ty = u32(max(top, 0.0) / ${WET.tile}.0) + t; ty <= min(u32(foot / ${WET.tile}.0), ny - 1u); ty += 64u) {
    let ra = f32(ty * ${WET.tile}u); let rb = ra + ${WET.tile}.0;
    var lo = 1.0e9; var hi = -1.0e9;
    for (var j = 0; j < S; j++) {
      if ((kept[j] & 1u) == 0u) { continue; }
      var a = y0[j]; var b = y1[j]; var xa = x0[j]; var xb = x1[j];
      if (j + 1 < S && (kept[j + 1] & 1u) != 0u) { a = min(a, y0[j + 1]); b = max(b, y1[j + 1]); xa = min(xa, x0[j + 1]); xb = max(xb, x1[j + 1]); }
      if (under && j == last) { b = size.y; xa -= wide; xb += wide; }
      if (b >= ra && a < rb) { lo = min(lo, xa); hi = max(hi, xb); }
    }
    if (hi < 0.0 || lo >= size.x || hi < lo) { continue; }
    for (var tx = u32(max(lo, 0.0) / ${WET.tile}.0); tx <= min(u32(hi / ${WET.tile}.0), nx - 1u); tx++) {
      let tile = ty * nx + tx; let at = bands + tile * ${WET.bands}u;
      if (!laying) { atomicAdd(&wet_tiles[at + band], 1u); continue; }
      if (band < atomicLoad(&wet_tiles[at])) { continue; } // (least() left it there)
      let slot = atomicAdd(&wet_tiles[tile], 1u);
      if (slot < ${WET.cap}u) { atomicStore(&wet_tiles[lists + tile * ${WET.cap}u + slot], i | (band << 27u)); } // with its band (wet_least)
    }
  }
}
`;
// What a wet track gives back at w: the sky, the moon and the lights its tile holds, each a glint off the film. n: the
// road's own normal; v: to the eye, dist away; wet: how much of the pixel is wet; puddle: how much of it is a puddle;
// px: the pixel; dx, dy: how w.xz steps from one pixel to the next. Call after the lighting, which finds traffic_shade.
const WET_FNS = WET_IN + /* wgsl */`
@group(1) @binding(2) var<storage, read> wet_lights: array<WetLight>;
@group(1) @binding(3) var<storage, read> wet_tiles: array<u32>;
@group(1) @binding(4) var sky_map: texture_2d<f32>; // the sky the water mirrors, aurora and all but the moon (the WATER's), lod mips blurred
@group(1) @binding(5) var samp_sky: sampler;
@group(1) @binding(6) var<storage, read> wet_vis: array<f32>; // each light's stations: how open the way from it to the road is (WET_BIN's seen())
const WET_SIGMA2 = vec2f(${(WET.sigma[0] ** 2).toExponential(4)}, ${(WET.sigma[1] ** 2).toExponential(4)});
// the night sky in direction r, and the town's light in the haze over the horizon (SKY's dome)
fn wet_sky(r: vec3f, p: vec2f, lod: f32) -> vec3f {
  let h = max(r.y, 0.0);
  let col = textureSampleLevel(sky_map, samp_sky, vec2f(atan2(r.z, r.x) / 6.2831853 + 0.5, sqrt(asin(min(h, 1.0)) / 1.5707963)), lod).rgb;
  let flat = r.xz / max(length(r.xz), 1e-4);
  return col + town_dome(p, flat) * pow8(1.0 - h) * ${AIR.most.toFixed(3)} * F.misc.w;
}
// The least band of brightness the tiles about px hold (WET_BIN's least()), eased across the frame: at each corner of the
// tiles the greatest of the four that meet there, and between the corners mixed. A tile too crowded to hold all its
// lights keeps the brighter bands, and a light of a band it left out fades out across its neighbour to the edge they
// share (the band under the least by the time it gets there), never to stop at it as a block.
fn wet_least(px: vec2f) -> f32 {
  let nx = i32(wet_tiles_x()); let ny = i32(wet_tiles_n()) / nx; let bands = wet_tiles_n() * ${WET.cap + 1}u;
  var m = array<f32, 9>();
  for (var j = 0; j < 9; j++) {
    let q = vec2i(px / ${WET.tile}.0) + vec2i(j % 3 - 1, j / 3 - 1);
    if (all(q >= vec2i(0)) && q.x < nx && q.y < ny) { m[j] = f32(wet_tiles[bands + u32(q.y * nx + q.x) * ${WET.bands}u]); }
  }
  let c = vec4f(max(max(m[0], m[1]), max(m[3], m[4])), max(max(m[1], m[2]), max(m[4], m[5])),
                max(max(m[3], m[4]), max(m[6], m[7])), max(max(m[4], m[5]), max(m[7], m[8]))); // its tile's corners
  let t = px / ${WET.tile}.0 - floor(px / ${WET.tile}.0);
  return mix(mix(c.x, c.y, t.x), mix(c.z, c.w, t.x), t.y);
}
struct Wet { light: vec3f, sheen: f32 } // what the film gives back; and the share of the light it mirrors, which the asphalt under it no longer takes
fn wet_track(w: vec3f, n: vec3f, v: vec3f, dist: f32, wet: f32, puddle: f32, px: vec2f, dx: vec2f, dy: vec2f) -> Wet {
  // the film's grain: the mean slope over the pixel of two tilings of the relief map, turned apart, and the variance the
  // mips have averaged away of it once the pixel spans a good part of a tile. Calmer in a puddle.
  let pa = pixel_angle();
  var m = vec2f(0.0); var lost = 0.0;
  for (var k = 0; k < 2; k++) {
    let tile = select(${WET.ripple[1].toFixed(3)}, ${WET.ripple[0].toFixed(3)}, k == 0); let amp = select(${WET.amp[1].toFixed(3)}, ${WET.amp[0].toFixed(3)}, k == 0) * (1.0 - 0.6 * puddle);
    let cs = select(vec2f(0.8, 0.6), vec2f(1.0, 0.0), k == 0); let turn = mat2x2f(cs.x, cs.y, -cs.y, cs.x);
    let q = turn * w.xz / tile; let qx = turn * dx / tile; let qy = turn * dy / tile;
    let g = (textureSampleGrad(detail_map, samp_repeat, q, 1, qx, qy).rg - 0.5) * 2.0;
    let fade = smoothstep(0.05, 0.6, sqrt(max(dot(qx, qx), dot(qy, qy)))); // how much of a tile the pixel spans
    m += (g * turn) * (amp * (1.0 - fade));
    lost += amp * amp * 0.3 * fade;
  }
  // slopes in the road's plane: along the view, away from the eye, and across it
  let along = -v + n * dot(v, n); // nothing where the eye is straight over the point: then any way along the road
  let ea = select(vec3f(0.0, 0.0, 1.0), normalize(along), dot(along, along) > 1e-12); let eb = cross(n, ea);
  let m3 = vec3f(m.x, 0.0, m.y);
  let mab = vec2f(dot(m3, ea), dot(m3, eb));
  let vr = mix(WET_SIGMA2.y, WET_SIGMA2.x, puddle) + lost;
  let vy = max(dot(v, n), 0.01);
  let nf = normalize(n - m3); // the film's mean facet over the pixel
  let fr = wet_fresnel(dot(v, nf));
  var r = reflect(-v, nf); r.y = abs(r.y);
  // the sky, faint in a film this rough: what it gives back of the dark sky is spread over the asphalt's grain
  var light = wet_sky(r, w.xz, log2(1.0 + sqrt(vr) * 80.0)) * (fr * ${WET.sky.toFixed(2)} * traffic_shade); // (under a car, its body)
  // over one pixel the eye's ray turns by pa, and the point it meets moves by its footprint, along the view the most:
  // the facet each light needs moves with both, and the glint is spread over that rather than let flicker
  let foot = dist * pa;
  let tile = vec2u(px) / ${WET.tile}u; let at = tile.y * wet_tiles_x() + tile.x;
  let count = min(wet_tiles[at], ${WET.cap}u); let first = wet_tiles_n() + at * ${WET.cap}u;
  let least = wet_least(px);
  for (var i = 0u; i < count; i++) {
    let held = wet_tiles[first + i]; let k = held & ${0x07ffffff}u; let at = wet_lights[k].at; // what it takes to turn a light away first, the rest only for those it keeps
    let to = at.xyz - w; let inv = inverseSqrt(dot(to, to)); let ld = to * inv;
    let hv = v + ld; let hn = dot(hv, n);
    if (hn <= 0.0) { continue; }
    let q = -vec2f(dot(hv, ea), dot(hv, eb)) / hn - mab;
    let hl = inverseSqrt(dot(hv, hv)); let hy = hn * hl;
    let s = (pa + foot * inv * vec2f(1.0 / vy, 1.0)) * hl / (hy * hy);
    // never rougher than the strips were cut for (WET_VR), so that no pixel takes a light its tile was not given
    let vrr = min(vr + at.w * at.w * inv * inv * 0.25 + s * s * 0.25, vec2f(WET_VR.y));
    let e = 0.5 * dot(q * q, 1.0 / vrr);
    if (e > 14.0) { continue; }
    let D = exp(-e) / (6.2831853 * sqrt(vrr.x * vrr.y));
    // how open the way from the light down to the road here is: between the two stations of its strip this pixel lies
    // between (along the line from the light's foot to the eye's, the stations closer together at either end)
    let dv = F.cam_pos.xz - at.xz;
    let f = clamp(dot(w.xz - at.xz, dv) / max(dot(dv, dv), 1e-4), 0.0, 1.0);
    let jf = acos(1.0 - 2.0 * f) * ${((WET.stations - 1) / Math.PI).toFixed(5)}; let j0 = min(u32(jf), ${WET.stations - 2}u);
    let open = mix(wet_vis[k * ${WET.stations}u + j0], wet_vis[k * ${WET.stations}u + j0 + 1u], jf - f32(j0));
    let l = wet_lights[k]; let fl = wet_fresnel(dot(v, hv) * hl);
    // never brighter than the lens itself as drawn, times what the film gives back of it: a mirrored light is always the dimmer
    let c = min(l.power.rgb * (inv * inv * D / (4.0 * vy * hy * hy * hy * hy)), l.lens.rgb)
      * (fl * open * wet_aim(l, -ld) * exp(-F.fog.a * F.fog.a / (inv * inv))); // and the haze between the light and the road
    // and none where its strip was cut: it fades out over the threefold above WET.faint, never to a tile's edge; nor
    // where a crowded tile near by keeps only brighter bands than its own (wet_least)
    light += c * (smoothstep(${WET.faint}, ${3 * WET.faint}, dot(c, vec3f(0.30, 0.59, 0.11))) * saturate(f32(held >> 27u) + 1.0 - least));
  }
  { // and the moon, where the eye looks up the road towards it: whether anything stands between is asked only where its
    // glint would show, and of the field over the whole map alone (moon_far): a rough film blurs what the near map adds
    let hv = v + MOON_DIR; let hn = dot(hv, n);
    let q = -vec2f(dot(hv, ea), dot(hv, eb)) / max(hn, 1e-3) - mab;
    let vrr = vr + ${((0.0245 / 2) ** 2).toExponential(3)};
    let e = 0.5 * dot(q, q) / vrr;
    if (hn > 0.0 && e < 14.0) {
      let hl = inverseSqrt(dot(hv, hv)); let hy = hn * hl;
      let g = exp(-e) / (6.2831853 * vrr) * wet_fresnel(dot(v, hv) * hl) / (4.0 * vy * hy * hy * hy * hy);
      if (g * wet * ${Math.max(...[0xb4, 0xc2, 0xee].map(c => ((c / 255) ** 2.2) * 0.22)).toFixed(4)} > ${WET.faint}) { light += moon_now() * g * moon_far(w, n, 0.0); }
    }
  }
  return Wet(light * wet, fr * ${WET.sky.toFixed(2)} * wet);
}
`;

// The snow on the roads (wasm/include/harstad/winter.h): each road's cells of WINTER_CELL m, the right and the left half
// of each apart. A ribbon's vertex says which road it is and how far along (winter_id: wasm/src/road.c, RW), or which
// end cell, or none. What lies there is worked out here from the night's snowfall (F.grit.z), at every distance alike.
const WINTER_CELL = 8.0;
const WINTER_FNS = /* wgsl */`
@group(1) @binding(7) var<storage, read> winter: array<u32>;
struct Winter { fresh: f32, salt: f32, pack: f32, grit: f32, rut: f32, marks: f32 }
fn winter_id(n: vec3f) -> u32 { let d = vec3u(round(n * 127.0) + 127.0); return d.x + d.y * 255u + d.z * 65025u; }
// what lies on half \`side\` of cell \`k\`, \`u\` of the way through it along the road
fn winter_cell(k: u32, side: u32, u: f32) -> Winter {
  let at = u32(F.grit.w) + k * 8u; let fall = F.grit.z;
  let b = winter[at + 6u];
  var base = bitcast<f32>(winter[at + side]);
  let front = (b >> 16u) & 255u;
  if (front != 0u && (front >> 7u) == side) { // a blade part way through: ahead of it, what lay before
    let f = f32(front & 63u) / 63.0; let back = (front & 64u) != 0u;
    if (select(u > f, u < 1.0 - f, back)) { base = bitcast<f32>(winter[at + 2u]); }
  }
  var o: Winter;
  o.fresh = max(fall - base, 0.0); o.salt = max(base - fall, 0.0); o.pack = bitcast<f32>(winter[at + 4u]);
  o.grit = f32(b & 255u) / 255.0; o.marks = f32(b >> 24u) / 255.0;
  // the ruts, filling as snow falls into them after
  o.rut = f32((b >> 8u) & 255u) / 255.0 * (1.0 - smoothstep(0.0, 0.03, fall - bitcast<f32>(winter[at + 3u])));
  return o;
}
fn winter_mix(a: Winter, b: Winter, t: f32) -> Winter {
  var o: Winter;
  o.fresh = mix(a.fresh, b.fresh, t); o.salt = mix(a.salt, b.salt, t); o.pack = mix(a.pack, b.pack, t);
  o.grit = mix(a.grit, b.grit, t); o.rut = mix(a.rut, b.rut, t); o.marks = mix(a.marks, b.marks, t);
  return o;
}
// at a point of a ribbon: its road's cells either side eased into each other over the last quarter metre each side of
// where they meet; valid 0 where there is none
fn winter_at(id: u32, s: f32, across: f32, valid: ptr<function, f32>) -> Winter {
  var road = id; var end = 0u;
  if (id >= ${1 << 23}u) { road = id - ${1 << 23}u; end = 2u; } else if (id >= ${1 << 22}u) { road = id - ${1 << 22}u; end = 1u; }
  let head = u32(F.grit.w);
  var none: Winter;
  *valid = 0.0;
  if (head == 0u || road + 1u >= head) { return none; }
  let f0 = winter[road]; let f1 = winter[road + 1u];
  if (f1 <= f0) { return none; }
  *valid = 1.0;
  let n = f1 - f0; let side = select(0u, 1u, across > 0.5);
  if (end == 1u) { return winter_cell(f0, side, 0.5); }
  if (end == 2u) { return winter_cell(f1 - 1u, side, 0.5); }
  let q = clamp(s / ${WINTER_CELL.toFixed(1)}, 0.0, f32(n) - 1e-3); let i = u32(q); let u = q - f32(i);
  let here = winter_cell(f0 + i, side, u);
  let edge = 0.25 / ${WINTER_CELL.toFixed(1)};
  if (u < edge && i > 0u) { return winter_mix(winter_cell(f0 + i - 1u, side, 1.0), here, 0.5 + 0.5 * u / edge); }
  if (u > 1.0 - edge && i + 1u < n) { return winter_mix(here, winter_cell(f0 + i + 1u, side, 0.0), 0.5 * (u - 1.0 + edge) / edge); }
  return here;
}
`;

// road ribbons, blended over the terrain; the texture's alpha feathers the verge into the snow. wet: the salted roads'
// tracks mirror what WET_BIN found (the ski trails, drawn with this too, have none). snowy: what lies on them is the
// roads' snow's (WINTER_FNS); the ski trails' is their own
const road = (wet, snowy = wet) => (wet ? WET_FNS : '') + (snowy ? WINTER_FNS : '') + /* wgsl */`
@group(1) @binding(0) var surfaces: texture_2d_array<f32>;
@group(1) @binding(1) var samp_road: sampler;
struct VIn { @location(0) pos: vec3f, @location(1) nrm: vec4f, @location(2) col: vec4f, @location(3) uv: vec2f }
struct V { @builtin(position) pos: vec4f, @location(0) w: vec3f, @location(1) col: vec3f, @location(2) uv: vec2f, @location(3) @interpolate(flat) layer: u32,
  @location(4) @interpolate(flat) id: u32 }
@vertex fn vs(i: VIn) -> V {
  var o: V;
  o.pos = F.view_proj * vec4f(i.pos, 1.0);
  o.w = i.pos; o.col = unpack(i.col); o.uv = i.uv; o.layer = u32(round(i.nrm.w * 127.0));
  o.id = ${snowy ? 'winter_id(i.nrm.xyz)' : '0u'};
  return o;
}
@fragment fn fs(v: V) -> @location(0) vec4f {
  let t = textureSample(surfaces, samp_road, v.uv, v.layer);
  let dx = dpdx(v.w); let dy = dpdy(v.w);
  var albedo = t.rgb;
  var bare = 1.0; // how much of the salted asphalt shows: the wet in its tracks with it
${snowy ? '  let packed = textureSample(surfaces, samp_road, v.uv, 0u).rgb; // the side streets\' packed snow, for a salted road packed over\n' : ''}${snowy ? `  { // what lies on it tonight: packed snow over the asphalt, sand, marks, and the fresh snow over all of it, the ruts cut in that
    var ok = 0.0;
    let r = winter_at(v.id, v.uv.y * 12.0, v.uv.x, &ok);
    if (ok > 0.0) {
      // where the wheels run: two tracks in each half, a car's gauge apart about the middle of its lane
      let lane = fract(v.uv.x * 2.0); let wheel = 1.0 - smoothstep(0.05, 0.11, min(abs(lane - 0.28), abs(lane - 0.72)));
      let pc = select(smoothstep(0.001, 0.008, r.pack) * (1.0 - smoothstep(0.0, 0.004, r.salt) * 0.85), 0.0, v.layer == 0u);
      albedo = mix(albedo, packed, pc);
      albedo = mix(albedo, albedo * vec3f(0.6, 0.53, 0.46), 0.55 * r.grit); // sand: dark and brown
      albedo = mix(albedo, albedo * 0.82, 0.5 * r.marks * wheel);          // what snowy tyres laid on the cleared road
      let cover = smoothstep(0.0, 0.025, r.fresh) * (1.0 - 0.65 * r.rut * wheel);
      albedo = mix(albedo, vec3f(0.84, 0.89, 0.97), cover);
      bare = (1.0 - pc) * (1.0 - cover) * smoothstep(0.0, 0.002, r.salt + select(0.0, 1.0, r.pack <= 0.0 && v.layer == 0u));
    }
  }
` : ''}  // traffic has worn the drifts off the carriageway. A ribbon lies BUILD_ROAD_LIFT over the ground only so as not to
  // fight it: within half a metre of the ground it is the floor the lamp map was painted on, and the street lamps' heads
  // are not asked how much nearer it lies to them (lamp_heads_up); higher, on a bridge's deck, it is lit as high as it is
  let above = v.w.y - ground_at(v.w.xz);
  var c = lit_over(albedo * v.col, vec3f(0.0, 1.0, 0.0), v.w, 1.0, -1.0, 0.3, dx.xz, dy.xz, 0.0, lamp_reach_over(v.w, select(above, 0.0, above < 0.5)), 1.0, above);
${wet && !OFF.has('wet') ? `  if (v.layer != 0u && bare > 0.01) { // a salted road: where its texture is bare asphalt, it is wet; more so in the puddles lying in the tracks
    let p = v.w.xz;
    // puddles some metres across, lying where the noise texture (its texels 0.375..0.625 but for a few) is high
    let pn = textureSampleLevel(detail_map, samp_repeat, p * ${(0.11 / 256).toExponential(4)}, 0, 0.0).r * 0.65 + textureSampleLevel(detail_map, samp_repeat, p * ${(0.43 / 256).toExponential(4)} + 0.31, 0, 0.0).r * 0.35;
    let puddle = smoothstep(0.72, 0.88, (pn - 0.375) * 4.0);
    let wet = (1.0 - smoothstep(0.055, 0.12, dot(t.rgb, vec3f(0.2126, 0.7152, 0.0722)))) * mix(${WET.wet[0]}, ${WET.wet[1]}, puddle) * bare;
    if (wet > 0.01) {
      // the road's own slope, a triangle's (its vertices all say up); where the pixel straddles a fold, up
      let slope = cross(dx, dy); var n = select(vec3f(0.0, 1.0, 0.0), normalize(slope), dot(slope, slope) > 1e-24); n *= sign(n.y);
      if (!(n.y >= 0.85)) { n = vec3f(0.0, 1.0, 0.0); }
      let to = F.cam_pos.xyz - v.w; let dist = length(to);
      let r = wet_track(v.w, n, to / dist, dist, wet, puddle, v.pos.xy, dx.xz, dy.xz);
      c = c * (1.0 - r.sheen) + r.light;
    }
${OFF.has('wettiles') ? `    { let tile = vec2u(v.pos.xy) / ${WET.tile}u; let n = wet_tiles[tile.y * wet_tiles_x() + tile.x]; return vec4f(f32(n) / 64.0, wet * 0.3, f32(n > ${WET.cap}u), 1.0); } // how many lights each tile holds: a 64th of red each; blue where it is full` : ''}
  }
` : ''}  return vec4f(finish(c, v.w), t.a);
}
`;
export const ROAD = road(true), TRAIL = road(false), ROAD_DRY = road(false, true);
// ---- the roads far off, lit once a pixel (src/worker.js, droad) ----
// Far off a road's ribbon is draped over every cell of the ground, its triangles a pixel or less across, and the GPU
// shades each one that holds a pixel's middle in a 2 x 2 quad of its own: from over the town each pixel of road was lit
// some 4.2 times over. So its draws marked GFX_CMD_FAR are drawn first (roadPrep) into a list of what lighting needs
// (RoadRec: the point, the surface's colour and cover, how the point moves from pixel to pixel), with the number of the
// last one drawn at each pixel; each listed at a pixel is lit there once (roadDeferred); and the frame's own far road
// takes the light listed nearest it (roadRead). It has no way to light itself: a shader that kept one, never taken, cost
// 1.45 ms of the 3 saved from over the town, by the registers it holds alone.
export const ROAD_RECS = 1 << 19; // the most listed in a frame (from over the town some 200,000, the hidden among them)
const ROAD_RECS_DECL = /* wgsl */`
struct RoadRec { w: vec3f, key: u32, albedo: vec3f, alpha: f32, dx: vec3f, bare: f32, dy: vec3f, lum: f32, lit: vec4f }
`;
const ROAD_KEY = '(0x80000000u | (v.layer << 24u) | v.id)'; // the road's cell id fits 24 bits (winter_id), the layer 7
const ROAD_LIGHT = '  let above = v.w.y - ground_at(v.w.xz);\n';
const ROAD_LUM = 'dot(t.rgb, vec3f(0.2126, 0.7152, 0.0722))';
const ROAD_FS = '@fragment fn fs(v: V) -> @location(0) vec4f {';
const roadCut = (code, at) => { const i = code.indexOf(at); if (i < 0 || code.indexOf(at, i + 1) >= 0) throw new Error(`road: ${at.slice(0, 40)}`); return i; };
// the list's pass: the colour and cover worked out as the road's own, and the rest of what lighting needs, listed
export const roadPrep = (() => { const c = ROAD; const a = roadCut(c, ROAD_FS), l = roadCut(c, ROAD_LIGHT);
  return c.slice(0, a) + ROAD_RECS_DECL + `struct RoadRecs { count: atomic<u32>, pad: vec3u, r: array<RoadRec> }
@group(2) @binding(0) var<storage, read_write> recs: RoadRecs;
@fragment fn fs(v: V) -> @location(0) u32 {` + c.slice(a + ROAD_FS.length, l) + `  let i = atomicAdd(&recs.count, 1u);
  if (i >= ${ROAD_RECS}u) { return 0u; }
  recs.r[i] = RoadRec(v.w, ${ROAD_KEY}, albedo * v.col, t.a, dx, bare, dy, ${ROAD_LUM}, vec4f(0.0));
  return i + 1u;
}
`; })();
// the lighting's pass: each pixel's listed road lit as the road lights itself, from what was listed
export const roadDeferred = (() => { const c = ROAD; const a = roadCut(c, ROAD_FS), l = roadCut(c, ROAD_LIGHT);
  const tail = c.slice(l), end = '  return vec4f(finish(c, v.w), t.a);\n}\n';
  if (!tail.endsWith(end) || tail.split(ROAD_LUM).length !== 2) throw new Error('road: lighting');
  return c.slice(0, a) + ROAD_RECS_DECL + `struct RoadRecs { count: u32, pad: vec3u, r: array<RoadRec> }
@group(2) @binding(0) var<storage, read_write> recs: RoadRecs;
@group(2) @binding(1) var road_index: texture_2d<u32>;
@vertex fn vs_full(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f { return vec4f(f32(i & 1u) * 4.0 - 1.0, f32(i >> 1u) * 4.0 - 1.0, 0.0, 1.0); }
@fragment fn fs(@builtin(position) pos: vec4f) {
  let i = textureLoad(road_index, vec2i(pos.xy), 0).r;
  if (i == 0u) { return; }
  let r = recs.r[i - 1u];
  var v: V; v.pos = pos; v.w = r.w; v.col = vec3f(1.0); v.layer = (r.key >> 24u) & 127u; v.id = r.key & 0xffffffu;
  let albedo = r.albedo; let bare = r.bare; let dx = r.dx; let dy = r.dy; let t = vec4f(0.0, 0.0, 0.0, r.alpha);
` + tail.slice(0, -end.length).split(ROAD_LUM).join('r.lum') + `  recs.r[i - 1u].lit = vec4f(finish(c, v.w), t.a);
}
`; })();
// The frame's far road: the light listed nearest it, under its own cover. Its own, where its pixel's middle is the
// point listed: the same road and layer, within two pixels' footprint of it, there or a pixel away (a triangle that
// holds samples of a pixel but not its middle is shaded at that middle, off its own face); else the light of whatever
// road is listed within a pixel (one under another, the joins of two: the lower one's light is the top one's, under the
// top one, where the two differ by under 6/255 at 0.2% of the pixels from over the town). A lane of a quad that holds
// none of the pixel's samples has nothing to find: it is there only for the derivatives.
export const roadRead = (() => { const c = ROAD; const a = roadCut(c, ROAD_FS);
  return c.slice(0, a) + ROAD_RECS_DECL + `struct RoadRecs { count: u32, pad: vec3u, r: array<RoadRec> }
@group(2) @binding(0) var<storage, read> recs: RoadRecs;
@group(2) @binding(1) var road_index: texture_2d<u32>;
@fragment fn fs(v: V, @builtin(sample_mask) mask: u32) -> @location(0) vec4f {
  let t = textureSample(surfaces, samp_road, v.uv, v.layer);
  let dx = dpdx(v.w); let dy = dpdy(v.w);
  if (mask == 0u) { return vec4f(0.0); }
  let key = ${ROAD_KEY}; let p = vec2i(v.pos.xy); let near = 2.0 * (abs(dx) + abs(dy));
  for (var k = 0; k < 5; k++) { // its own pixel, then the four beside it
    let i = textureLoad(road_index, p + select(select(vec2i(0, 1 - 2 * (k & 1)), vec2i(1 - 2 * (k & 1), 0), k < 3), vec2i(0), k == 0), 0).r;
    if (i > 0u && recs.r[i - 1u].key == key) { let r = recs.r[i - 1u]; if (all(abs(r.w - v.w) <= near)) { return vec4f(r.lit.rgb, t.a); } }
  }
  for (var k = 0; k < 9; k++) { // any road's, round it
    let i = textureLoad(road_index, p + vec2i(k % 3 - 1, k / 3 - 1), 0).r;
    if (i > 0u) { return vec4f(recs.r[i - 1u].lit.rgb, t.a); }
  }
  return vec4f(0.0);
}
`; })();

// The ground under the roads' opaque middles, drawn into the depth alone before the terrain, so that the terrain is not
// shaded where a road is then laid over all of it: a ribbon is blended over the ground (its texture's alpha feathers the
// verge), and under its middle, where that alpha is 1, the ground's shader ran for nothing; in the street the roads are
// a third of the frame. A ribbon lies BUILD_ROAD_LIFT over the ground, and what is drawn after it (the exhaust's steam,
// low over the road) is tested against the ground as before, not against the ribbon: so each of the ribbon's vertices is
// given the depth of the point where its own line of sight meets the ground as the terrain's mesh has it
// (ground_plane: the triangle of the height map's posts under a point), less a little (ROAD_MASK_OVER of a pixel there,
// between 1 mm and 1 cm: what the two meshes' arithmetic differs by is under it). The triangle between three such
// points lies on the ground or, where the ground bends under it, on a chord of it.
// Where a line does not come down on a tile's ground in three steps (seen from below, or edge on; a deck over water),
// the triangles round that vertex are left out.
const ROAD_MASK_OVER = 0.25;
export const ROAD_MASK = /* wgsl */`
@group(1) @binding(0) var surfaces: texture_2d_array<f32>;
@group(1) @binding(1) var samp_road: sampler;
struct VIn { @location(0) pos: vec3f, @location(1) nrm: vec4f, @location(2) col: vec4f, @location(3) uv: vec2f }
struct V { @builtin(position) pos: vec4f, @location(0) uv: vec3f, @location(1) @interpolate(flat) layer: u32 }
// the ground's triangle under xz: its height there, its slope along x and along z, and whether a tile holds it
fn ground_plane(xz: vec2f) -> vec4f {
  let p = clamp((xz - F.ground_rect.xy) / F.ground_rect.zw * F.ground_tiles.zw - 0.5, vec2f(0.0), F.ground_tiles.zw - 1.0);
  let t = clamp(vec2i(floor((p - F.ground_tiles.xy) / ${TILE_STEPS}.0)), vec2i(0), tab_dims(${TABLES.tiles}u) - 1);
  let layer = bitcast<i32>(tab_load(${TABLES.tiles}u, t).x);
  let l = p - F.ground_tiles.xy - vec2f(t) * ${TILE_STEPS}.0;
  let i = clamp(vec2i(floor(l)), vec2i(0), vec2i(${TILE_STEPS - 1})); let f = l - vec2f(i);
  let k = max(layer, 0);
  let b = textureLoad(ground_map, i + vec2i(1, 0), k, 0).r; let c = textureLoad(ground_map, i + vec2i(0, 1), k, 0).r;
  let low = f.x + f.y <= 1.0; // the terrain's strips cut each cell from its post at (1, 0) to the one at (0, 1)
  let o = textureLoad(ground_map, i + select(vec2i(1, 1), vec2i(0, 0), low), k, 0).r;
  let s = select(vec2f(o - c, o - b), vec2f(b - o, c - o), low);
  let h = o + dot(s, select(f - 1.0, f, low));
  return vec4f(h, s * (F.ground_tiles.z / F.ground_rect.z), f32(layer >= 0));
}
@vertex fn vs(i: VIn) -> V {
  let eye = F.cam_pos.xyz; let dir = i.pos - eye;
  let over = clamp(${ROAD_MASK_OVER.toFixed(2)} * length(dir) * pixel_angle(), 0.001, 0.01);
  var t = 1.0; var ok = 1.0;
  for (var k = 0; k < 4; k++) {
    let x = eye + dir * t;
    let g = ground_plane(x.xz);
    let f = x.y - g.x - over;
    let df = dir.y - dot(g.yz, dir.xz);
    if (g.w == 0.0 || !(df < -1e-4) || (k == 0 && f <= 0.0)) { ok = 0.0; break; }
    if (k == 3) { ok = f32(abs(f) < 0.25 * over); break; }
    t -= f / df;
  }
  if (!(t >= 1.0 && t < 2.0)) { ok = 0.0; }
  if (ok == 0.0) { t = 1.0; }
  // where the road's own vertex is on the screen, to the bit, at the depth of the point on the ground: the triangle
  // covers the samples the ribbon's does, its texture is where the ribbon's is, and its depth, which goes linearly
  // over the screen as any flat triangle's does, is that of the triangle between the points on the ground
  let at = F.view_proj * vec4f(i.pos, 1.0); let far = F.view_proj * vec4f(eye + dir * t, 1.0);
  var o: V;
  o.pos = vec4f(at.xy, far.z / far.w * at.w, at.w);
  o.uv = vec3f(i.uv, ok); o.layer = u32(round(i.nrm.w * 127.0));
  return o;
}
@fragment fn fs(v: V) -> @location(0) vec4f {
  if (textureSample(surfaces, samp_road, v.uv.xy, v.layer).a < 1.0 || v.uv.z < 1.0) { discard; }
  return vec4f(0.0);
}
`;

// paved ground: quay aprons, setts, pavements. A repeating texture under the lamps, tinted by the vertex.
export const PAVING = /* wgsl */`
@group(1) @binding(0) var surfaces: texture_2d_array<f32>;
@group(1) @binding(1) var samp_paving: sampler;
struct VIn { @location(0) pos: vec3f, @location(1) nrm: vec4f, @location(2) col: vec4f, @location(3) uv: vec2f }
struct V { @builtin(position) pos: vec4f, @location(0) w: vec3f, @location(1) n: vec3f, @location(2) col: vec3f, @location(3) uv: vec2f, @location(4) @interpolate(flat) layer: u32 }
@vertex fn vs(i: VIn) -> V {
  var o: V;
  o.pos = F.view_proj * vec4f(i.pos, 1.0);
  o.w = i.pos; o.n = i.nrm.xyz; o.col = unpack(i.col); o.uv = i.uv; o.layer = u32(round(i.nrm.w * 127.0));
  return o;
}
@fragment fn fs(v: V) -> @location(0) vec4f {
  let t = textureSample(surfaces, samp_paving, v.uv, v.layer).rgb;
  return vec4f(finish(lit(t * v.col, normalize(v.n), v.w, 1.0), v.w), 1.0);
}
`;

// The centre's shop windows (wasm/src/sentrum.c): each bay of glass a tile of the window atlas, in three layers
// (src/textures.js, displayAtlas): the shop's back wall and floor, what is for sale and what stands in it, and the
// glass with what hangs on it. Behind the glass the eye's ray is followed into the shop, a box as wide and as tall as
// the bay and 1.5 to 4.5 m deep (interior mapping): the picture's back wall at its back, its floor colour underfoot,
// its edges for the sides, its top for the ceiling with downlights in a lit shop, a low display plinth inside the
// glass, and what is for sale stood up between at its own depth, so goods and mannequins move against the back as the
// eye goes by. Seen straight on it is the picture it was; as a bay comes under a few pixels it fades into that
// picture, which is the mean of it. The bay's size is the ratio of how far along and up the glass a pixel steps to
// how far through its tile, which on a flat quad is the same everywhere on it.
const SHOP = {
  cols: 16, rows: 4, tile: 128,   // the window atlas (src/textures.js)
  depth: [1.5, 3.0],              // m deep: the least, and how much more a shop can be
  plinth: 0.12,                   // the display plinth inside the glass, of the bay's height: what stands in the picture stands on it
  detail: [3, 8],                 // a metre of glass this many pixels across: the shop no more than its picture, then all of it
};
export const PANE = ROOM_FNS + /* wgsl */`
@group(1) @binding(0) var shop: texture_2d_array<f32>;
@group(1) @binding(1) var samp_shop: sampler;
struct VIn { @location(0) pos: vec3f, @location(1) nrm: vec4f, @location(2) col: vec4f, @location(3) uv: vec2f }
struct V { @builtin(position) pos: vec4f, @location(0) w: vec3f, @location(1) n: vec3f, @location(2) col: vec3f, @location(3) uv: vec2f,
  @location(4) @interpolate(flat) start: vec3f } // the bay's first corner: both its triangles are drawn from it (soup_quad)
@vertex fn vs(i: VIn) -> V {
  var o: V;
  o.pos = F.view_proj * vec4f(i.pos, 1.0);
  o.w = i.pos; o.n = i.nrm.xyz; o.col = unpack(i.col); o.uv = i.uv; o.start = i.pos;
  return o;
}
const SHOP_GRID = vec2f(${SHOP.cols}.0, ${SHOP.rows}.0);
// a layer of the shop's tile at q (0..1 across and up it), at mip level lv; kept off the tile's edges
fn shop_at(tile: vec2f, q: vec2f, layer: i32, lv: f32) -> vec4f {
  return textureSampleLevel(shop, samp_shop, (tile + clamp(q, vec2f(0.01), vec2f(0.99))) / SHOP_GRID, layer, lv);
}
@fragment fn fs(v: V) -> @location(0) vec4f {
  let n = normalize(v.n);
  let side = normalize(vec2f(-n.z, n.x));
  let g = v.uv * SHOP_GRID; let tile = floor(g); let q = g - tile; // the tile, and where in it
  let dx = dpdx(v.uv); let dy = dpdy(v.uv);
  let gx = dpdx(g); let gy = dpdy(g);
  let along_x = dot(dpdx(v.w.xz), side); let along_y = dot(dpdy(v.w.xz), side);
  // m of glass a tile spans, across and up; and which way across the tile runs along the wall
  let bw = clamp((abs(along_x) + abs(along_y)) / max(abs(gx.x) + abs(gy.x), 1e-7), 0.3, 8.0);
  let bh = clamp((abs(dpdx(v.w.y)) + abs(dpdy(v.w.y))) / max(abs(gx.y) + abs(gy.y), 1e-7), 0.3, 6.0);
  let along = select(-side, side, along_x * gx.x + along_y * gy.x >= 0.0);
  let glass = textureSampleGrad(shop, samp_shop, v.uv, 2, dx, dy);
  let eye = v.w - F.cam_pos.xyz; let dist = length(eye); let ray = eye / dist;
  let fm = max(length(gx * vec2f(bw, bh)), length(gy * vec2f(bw, bh))); // m of glass a pixel spans
  let detail = smoothstep(${SHOP.detail[0].toFixed(1)}, ${SHOP.detail[1].toFixed(1)}, 1.0 / fm) * (1.0 - glass.a);
  var inside = vec3f(0.0);
  if (detail < 1.0) { // the picture as it was painted
    let stock = textureSampleGrad(shop, samp_shop, v.uv, 1, dx, dy);
    inside = stock.rgb + (1.0 - stock.a) * textureSampleGrad(shop, samp_shop, v.uv, 0, dx, dy).rgb;
  }
  if (detail > 0.0) {
    let id = room_hash(bitcast<u32>(v.start.x) ^ room_hash(bitcast<u32>(v.start.z) ^ room_hash(u32(tile.x + tile.y * 16.0))));
    let depth = ${SHOP.depth[0].toFixed(2)} + ${SHOP.depth[1].toFixed(2)} * room_rand(id, 1u);
    let floor_y = bh * ${SHOP.plinth.toFixed(3)};
    let o = vec3f(q.x * bw, q.y * bh, 0.0);
    let d = vec3f(dot(ray.xz, along), ray.y, max(-dot(ray, n), 0.02));
    let dd = select(d, vec3f(1e-5), abs(d) < vec3f(1e-5)); let inv = 1.0 / dd;
    let pa = pixel_angle();
    var col: vec3f;
    if (o.y < floor_y) { // the plinth's face, just inside the glass
      col = shop_at(tile, q, 0, 0.0).rgb * 0.8;
    } else {
      let tx = (select(0.0, bw, dd.x > 0.0) - o.x) * inv.x;
      let ty = (select(floor_y, bh, dd.y > 0.0) - o.y) * inv.y;
      let tz = depth * inv.z;
      let t = min(tz, min(tx, ty));
      let p = o + d * t;
      let lv = clamp(log2((dist + t) * pa / bw * ${SHOP.tile}.0), 0.0, 4.0); // the picture's texels a pixel spans there
      let c = vec2f(p.x / bw, p.y / bh);
      if (t == tz) { col = shop_at(tile, c, 0, lv).rgb; }
      else if (t == tx) { col = shop_at(tile, vec2f(select(0.02, 0.98, dd.x > 0.0), c.y), 0, lv).rgb * 0.7; }
      else if (dd.y < 0.0) { col = shop_at(tile, vec2f(c.x, 0.1), 0, lv).rgb * (0.65 + 0.3 * p.z / depth); }
      else {
        col = shop_at(tile, vec2f(c.x, 0.97), 0, lv).rgb * 0.75;
        // downlights in a lit shop's ceiling, a metre apart
        let s = (dist + t) * pa;
        let f = fract(p.xz) - 0.5;
        col += vec3f(3.0, 2.6, 2.0) * smoothstep(0.08 + s, 0.08 - s, length(f)) * select(0.0, 1.0, v.col.r < 0.75);
      }
      // what is for sale, stood up at its own depth: drawn as the picture has it, seen straight on
      let ts = depth * (0.35 + 0.4 * room_rand(id, 2u)) * inv.z;
      if (ts < t) {
        let s = o + d * ts;
        let lvs = clamp(log2((dist + ts) * pa / bw * ${SHOP.tile}.0), 0.0, 4.0);
        let cs = vec2f(s.x / bw, s.y / bh);
        var stock = shop_at(tile, cs, 1, lvs);
        stock *= select(0.0, 1.0, all(cs > vec2f(0.0)) && all(cs < vec2f(1.0)) && s.y > floor_y);
        col = stock.rgb + (1.0 - stock.a) * col;
      }
    }
    inside = mix(inside, col, detail);
  }
  // lit while the shops are (from half an hour before they open to half an hour after they shut: F.town_now.w), and a
  // third of the windows' displays all night; the rest left with a night light, a twentieth of it
  let bay = room_hash(bitcast<u32>(v.start.x) ^ room_hash(bitcast<u32>(v.start.z)));
  inside *= select(0.05, 1.0, F.town_now.w > 0.5 || room_rand(bay, 60u) < 0.35);
  // the glass: what hangs on it over the shop, and the street and the sky it gives back, more the flatter it is seen
  let c = (glass.rgb + (1.0 - glass.a) * inside) * v.col;
  return vec4f(finish(c + glass_sheen(ray, n, lamp_reach(v.w), v.w) * (1.0 - glass.a * 0.5), v.w), 1.0);
}
`;

// shop signs: lettered boards that are their own light
export const SIGN = /* wgsl */`
@group(1) @binding(0) var atlas: texture_2d_array<f32>;
@group(1) @binding(1) var samp_sign: sampler;
struct VIn { @location(0) pos: vec3f, @location(1) nrm: vec4f, @location(2) col: vec4f, @location(3) uv: vec2f }
struct V { @builtin(position) pos: vec4f, @location(0) w: vec3f, @location(1) col: vec3f, @location(2) uv: vec2f }
@vertex fn vs(i: VIn) -> V {
  var o: V;
  o.pos = F.view_proj * vec4f(i.pos, 1.0);
  o.w = i.pos; o.col = unpack(i.col); o.uv = i.uv;
  return o;
}
@fragment fn fs(v: V) -> @location(0) vec4f {
  return vec4f(finish(textureSample(atlas, samp_sign, v.uv, 0).rgb * v.col, v.w), 1.0);
}
`;

// the centre's surveyed signs and lit letters: as a sign, but cut out where the atlas is clear and blended at its edges
export const LABEL = /* wgsl */`
@group(1) @binding(0) var atlas: texture_2d_array<f32>;
@group(1) @binding(1) var samp_label: sampler;
struct VIn { @location(0) pos: vec3f, @location(1) nrm: vec4f, @location(2) col: vec4f, @location(3) uv: vec2f }
struct V { @builtin(position) pos: vec4f, @location(0) w: vec3f, @location(1) col: vec3f, @location(2) uv: vec2f }
@vertex fn vs(i: VIn) -> V {
  var o: V;
  o.pos = F.view_proj * vec4f(i.pos, 1.0);
  o.w = i.pos; o.col = unpack(i.col); o.uv = i.uv;
  return o;
}
@fragment fn fs(v: V) -> @location(0) vec4f {
  let t = textureSample(atlas, samp_label, v.uv, 0);
  if (t.a < 0.35) { discard; }
  return vec4f(finish(t.rgb * v.col, v.w), t.a);
}
`;

// zebra crossings: paint lying thinly on the road
export const ZEBRA = MESH_IN + /* wgsl */`
@vertex fn vs(i: VIn) -> V {
  var o: V;
  o.pos = F.view_proj * vec4f(i.pos, 1.0);
  o.w = i.pos; o.n = i.nrm.xyz; o.col = unpack(i.col); o.flag = 0.0;
  return o;
}
@fragment fn fs(v: V) -> @location(0) vec4f { return vec4f(finish(lit(v.col, normalize(v.n), v.w, 1.0), v.w), 0.62); }
`;

// tunnel floor and walls: under the luminaires the surface is its own light, and no street lamp reaches in
export const TUBE = /* wgsl */`
@group(1) @binding(0) var surfaces: texture_2d_array<f32>;
@group(1) @binding(1) var samp_road: sampler;
struct VIn { @location(0) pos: vec3f, @location(1) nrm: vec4f, @location(2) col: vec4f, @location(3) uv: vec2f }
struct V { @builtin(position) pos: vec4f, @location(0) w: vec3f, @location(1) n: vec3f, @location(2) uv: vec2f, @location(3) @interpolate(flat) layer: u32 }
@vertex fn vs(i: VIn) -> V {
  var o: V;
  o.pos = F.view_proj * vec4f(i.pos, 1.0);
  o.w = i.pos; o.n = i.nrm.xyz; o.uv = i.uv; o.layer = u32(round(i.nrm.w * 127.0));
  return o;
}
@fragment fn fs(v: V) -> @location(0) vec4f {
  let t = textureSample(surfaces, samp_road, v.uv, v.layer).rgb;
  return vec4f(finish(lit_bare(t, normalize(v.n), v.w, 0.0) + t * 0.5, v.w), 1.0);
}
`;

// The car. One mesh; each vertex names the moving part it belongs to and its material. Paint, chrome and glass
// mirror the sky as the water does (three.js used a blurred environment map; this asks the sky directly).
// lens: whether it draws the lamps' lenses with their chambers (lamp_chamber). Only the lenses' own triangles are drawn
// with it (src/worker.js, car.runs): with that code in it, the rest of the car shades several times slower, its
// registers taken up for what none of its pixels runs.
const ff = v => (+v).toFixed(4); // a number as WGSL takes a float
const FLIGHT_FNS = /* wgsl */`
// The flight instruments on the cluster board's back (src/textures.js, flightFace): their face as painted, with what moves
// drawn over it from the C's readings (C.attitude, C.air, C.thrust): the attitude ball under the face where it is clear,
// the heading card turned under its lubber line, the needles, the thrust columns and the jets' lamps. In the picture's
// colours, as the cluster.
fn flight_needle(c: vec3f, q: vec2f, a: f32, len: f32, w: f32) -> vec3f {
  let dir = vec2f(cos(a), sin(a));
  let d = distance(q, dir * clamp(dot(q, dir), -10.0, len));
  var o = mix(c, vec3f(1.0, 0.302, 0.102), 0.28 * clamp(w + 3.0 - d, 0.0, 1.0));
  return mix(o, vec3f(1.0, 0.302, 0.102), clamp(w - d, 0.0, 1.0));
}
fn flight(uv: vec2f) -> vec3f {
  let p = uv * vec2f(1024.0, 320.0);
  var face = textureSampleLevel(dials, samp_clamp, uv, 2u, 0.0);
  let pitch = C.attitude.x; let roll = C.attitude.y; let heading = C.attitude.z; let climb = C.attitude.w;
  let alt = C.air.x; let speed = C.air.y * 3.6; let fold = C.air.z;
  { // the heading card, turned: N where the car's nose points is at the top
    let q = p - vec2f(${ff(FLIGHT.hdg[0])}, ${ff(FLIGHT.hdg[1])});
    if (dot(q, q) < ${ff((FLIGHT.hdg[2] - 2) ** 2)}) {
      let cs = cos(heading); let sn = sin(heading);
      let r = vec2f(q.x * cs - q.y * sn, q.x * sn + q.y * cs) + vec2f(${ff(FLIGHT.hdg[0])}, ${ff(FLIGHT.hdg[1])});
      face = textureSampleLevel(dials, samp_clamp, r / vec2f(1024.0, 320.0), 2u, 0.0);
      if (q.y < -${ff(FLIGHT.hdg[2] - 18)} && abs(q.x) < (q.y + ${ff(FLIGHT.hdg[2] - 4)}) * 0.4) { face = vec4f(1.0, 0.24, 0.035, 1.0); } // the lubber line
    }
  }
  var c = encode(face.rgb);
  { // the ball: sky over ground, the horizon and the pitch ladder every 10 degrees, rolled and pitched as the car is
    let q = p - vec2f(${ff(FLIGHT.adi[0])}, ${ff(FLIGHT.adi[1])});
    if (face.a < 0.99 && dot(q, q) < ${ff(FLIGHT.adi[2] ** 2)}) {
      let cs = cos(roll); let sn = sin(roll);
      let r = vec2f(q.x * cs + q.y * sn, -q.x * sn + q.y * cs);
      let k = 240.0; // px of the ball to a radian of pitch
      let h = r.y - pitch * k;
      var ball = select(vec3f(0.36, 0.2, 0.08), vec3f(0.12, 0.32, 0.62), h < 0.0);
      ball *= 0.85 + 0.15 * clamp(1.0 - abs(h) / 90.0, 0.0, 1.0);
      let rung = h / (k * 0.1745); let near = abs(rung - round(rung)) * k * 0.1745;
      let wide = select(14.0, 26.0, abs(round(rung)) % 2.0 > 0.5);
      if (near < 1.3 && abs(r.x) < wide && round(rung) != 0.0) { ball = vec3f(0.95, 0.92, 0.85); }
      if (abs(h) < 1.6) { ball = vec3f(0.98, 0.96, 0.9); }
      let bank = atan2(q.x, -q.y) - roll; // the bank pointer, at the rim, turning with the ball
      if (length(q) > ${ff(FLIGHT.adi[2] - 36)} && length(q) < ${ff(FLIGHT.adi[2] - 22)} && abs(bank) < 0.05) { ball = vec3f(1.0, 0.54, 0.12); }
      c = mix(ball, c, face.a);
    }
  }
  // the needles: airspeed, the altimeter's two (1,000 m and 10,000 m a turn), the rate of climb
  c = flight_needle(c, p - vec2f(${ff(FLIGHT.asi[0])}, ${ff(FLIGHT.asi[1])}), ${(Math.PI * 0.75).toFixed(5)} + clamp(speed, 0.0, ${ff(FLIGHT.asi_most)}) / ${ff(FLIGHT.asi_top)} * ${(Math.PI * 1.5 * 1.111).toFixed(5)}, ${ff(FLIGHT.asi[2] - 14)}, 2.5);
  let qa = p - vec2f(${ff(FLIGHT.alt[0])}, ${ff(FLIGHT.alt[1])});
  c = flight_needle(c, qa, -1.5708 + max(alt, 0.0) / 10000.0 * 6.28319, ${ff(FLIGHT.alt[2] - 44)}, 4.0);
  c = flight_needle(c, qa, -1.5708 + fract(max(alt, 0.0) / 1000.0) * 6.28319, ${ff(FLIGHT.alt[2] - 12)}, 2.5);
  c = flight_needle(c, p - vec2f(${ff(FLIGHT.vsi[0])}, ${ff(FLIGHT.vsi[1])}), 3.14159 - clamp(climb, -10.0, 10.0) / 10.0 * 2.79602, ${ff(FLIGHT.vsi[2] - 12)}, 2.5);
  for (var k = 0; k < 4; k++) { // the hubs
    let at = array<vec2f, 4>(vec2f(${ff(FLIGHT.asi[0])}, ${ff(FLIGHT.asi[1])}), vec2f(${ff(FLIGHT.alt[0])}, ${ff(FLIGHT.alt[1])}), vec2f(${ff(FLIGHT.vsi[0])}, ${ff(FLIGHT.vsi[1])}), vec2f(${ff(FLIGHT.hdg[0])}, ${ff(FLIGHT.hdg[1])}))[k];
    c = mix(c, vec3f(0.03, 0.03, 0.035), clamp(9.5 - distance(p, at), 0.0, 1.0) * select(1.0, 0.0, k == 3));
  }
  { // each jet's thrust in lit segments, and whether its wheel is out as a jet
    let b = p - vec2f(${ff(FLIGHT.bars[0])}, ${ff(FLIGHT.bars[1])});
    if (b.x >= 0.0 && b.x < ${ff(FLIGHT.bars[2])} && b.y >= 0.0 && b.y < ${ff(FLIGHT.bars[3])}) {
      let j = u32(b.x / ${ff(FLIGHT.bars[2] / 4)}); let seg = floor((${ff(FLIGHT.bars[3])} - b.y) / ${ff(FLIGHT.bars[3] / 10)});
      let inside = fract(b.x / ${ff(FLIGHT.bars[2] / 4)}) > 0.1 && fract(b.x / ${ff(FLIGHT.bars[2] / 4)}) < 0.9 && fract((${ff(FLIGHT.bars[3])} - b.y) / ${ff(FLIGHT.bars[3] / 10)}) > 0.12;
      if (inside && seg < C.thrust[j] * 10.0 - 0.01) { c = select(vec3f(1.0, 0.55, 0.12), vec3f(1.0, 0.25, 0.1), seg >= 8.5); }
    }
    let l = p - vec2f(${ff(FLIGHT.lamps[0])}, ${ff(FLIGHT.lamps[1] + 20)});
    let j = clamp(floor(l.x / ${ff(FLIGHT.lamps[2] / 4)}), 0.0, 3.0);
    if (distance(l, vec2f(j * ${ff(FLIGHT.lamps[2] / 4)} + ${ff(FLIGHT.lamps[2] / 8)}, 0.0)) < 7.0) {
      c = select(select(vec3f(0.09, 0.08, 0.07), vec3f(1.0, 0.62, 0.1), fold > 0.02), vec3f(0.3, 1.0, 0.36), fold > 0.995);
    }
  }
  { // the boost: the turbine's thrust in lit segments, blue as its flame; and its lamp, amber while the turbine comes
    // out or goes in, blue as far as the boost is on once it is out
    let b = p - vec2f(${ff(FLIGHT.boost[0])}, ${ff(FLIGHT.boost[1])});
    if (b.x >= 0.0 && b.x < ${ff(FLIGHT.boost[2])} && b.y >= 0.0 && b.y < ${ff(FLIGHT.boost[3])}) {
      let seg = floor((${ff(FLIGHT.boost[3])} - b.y) / ${ff(FLIGHT.boost[3] / 10)});
      let inside = b.x > 4.0 && b.x < ${ff(FLIGHT.boost[2] - 4)} && fract((${ff(FLIGHT.boost[3])} - b.y) / ${ff(FLIGHT.boost[3] / 10)}) > 0.12;
      if (inside && seg < C.boost.y * 10.0 - 0.01) { c = select(vec3f(0.3, 0.62, 1.0), vec3f(0.75, 0.88, 1.0), seg >= 7.5); }
    }
    if (distance(p, vec2f(${ff(FLIGHT.boost_lamp[0])}, ${ff(FLIGHT.boost_lamp[1])})) < 8.0) {
      let moving = C.boost.z > 0.001 && C.boost.z < 0.999;
      c = select(mix(vec3f(0.09, 0.08, 0.07), vec3f(0.35, 0.65, 1.0), C.boost.x), vec3f(1.0, 0.62, 0.1), moving);
    }
  }
  return select(pow((c + 0.055) / 1.055, vec3f(2.4)), c / 12.92, c <= vec3f(0.04045));
}
`;
const car = (lens, flight = false, cabin = false, washed = false) => LAMP_FNS + /* wgsl */`
struct Mat { albedo: vec4f, emissive: vec4f, params: vec4f } // rgb + opacity; kind in a; roughness, metalness, mirror strength, picture or lamp-shy
// How hot the inside of a nozzle runs, as the colour it burns with: warm orange, yellow, yellow-white, blue-white,
// ultra hot blue at 1, and in the turbo on to a white-violet blue at 1.35 (wasm/include/harstad/rocket.h, ROCKET_HEAT,
// whose light this is)
fn jet_heat(h: f32) -> vec3f {
  let k = array<vec4f, 6>(vec4f(0.0, 1.0, 0.45, 0.12), vec4f(0.3, 1.0, 0.62, 0.3), vec4f(0.5, 1.0, 0.86, 0.62), vec4f(0.72, 0.7, 0.82, 1.0), vec4f(1.0, 0.25, 0.45, 1.0), vec4f(1.35, 0.62, 0.55, 1.0));
  var c = k[0].yzw;
  for (var j = 0; j < 5; j++) { c = select(c, mix(k[j].yzw, k[j + 1].yzw, clamp((h - k[j].x) / (k[j + 1].x - k[j].x), 0.0, 1.0)), h >= k[j].x); }
  return c;
}
struct Car { parts: array<mat4x4f, 32>, mats: array<Mat, 16>, lamps: array<vec4f, 3>, glow_at: array<vec4f, 8>, glow: array<vec4f, 8>, needle: vec4f, dash: vec4f,
  street_at: array<vec4f, 4>, street: array<vec4f, 4>, // the nearest street lamps: where, and their colour times strength
  attitude: vec4f, air: vec4f, thrust: vec4f, // the flight instruments (avensis.h): pitch, roll, heading, climb; height, speed, wings out, lit; each wing's jets' thrust
  boost: vec4f, // the boost: how far on, the turbine's thrust as a share of its most, how far it is out
  // lamps that light the car from outside it in flight (avensis.h, AV_SPOTS): each engine's flame at its nozzles' mouth,
  // the tail's white navigation lights, the scan lamps on the flanks: where (w: how big), colour times strength, the way
  // each shines (w: the cosine of its beam's edge, -1 all round)
  spot_at: array<vec4f, 12>, spot: array<vec4f, 12>, spot_aim: array<vec4f, 12>,
  // How many of each list are gone through (avensis_lit): the spots, the street lamps, the navigation lights' glows, the
  // glows in all, each up to its last one lit. Every loop over them runs to its count here and not to the list's length:
  // a loop of a fixed length the GPU's compiler unrolls, a copy of its work for every lamp, and a shader is the slower
  // for all it holds, run or not (with all twelve spots dark, as they are on the road, their two loops alone were half
  // of what the car cost the frame).
  lit: vec4f }
@group(1) @binding(0) var<uniform> C: Car;
@group(1) @binding(1) var plate: texture_2d<f32>;
@group(1) @binding(2) var dials: texture_2d_array<f32>; // the cluster, 1024 x 320: as it stands, and with its lamps lit
@group(1) @binding(3) var panels: texture_2d<f32>;
@group(1) @binding(4) var sky_map: texture_2d<f32>; // the sky the water mirrors, aurora and all but the moon (the WATER's)
@group(1) @binding(5) var samp_sky: sampler;
struct VIn { @location(0) pos: vec3f, @location(1) nrm: vec4f, @location(2) col: vec3f, @location(3) uv: vec2f, @location(4) tag: vec4u }
struct V { @builtin(position) pos: vec4f, @location(0) w: vec3f, @location(1) n: vec3f, @location(2) uv: vec2f, @location(3) @interpolate(flat) mat: u32, @location(4) col: vec3f,
  @location(5) @interpolate(flat) lens: vec2u } // a lens's lamp, and how its chamber is made (lens_style), 0 for a plain lit face
@vertex fn vs(i: VIn) -> V {
  var o: V;
  let m = C.parts[u32(round(i.nrm.w * 127.0))];
  o.mat = i.tag.x;
  var w = (m * vec4f(i.pos, 1.0)).xyz;
  let kind = u32(C.mats[o.mat].emissive.a + 0.5);
  o.pos = F.view_proj * vec4f(w, 1.0);
  o.w = w; o.n = (m * vec4f(i.nrm.xyz, 0.0)).xyz; o.uv = i.uv;
  o.col = i.col;
  o.lens = i.tag.yz;
  if (kind == 1u && i.tag.z == 0u) { o.col *= C.lamps[i.tag.y / 4u][i.tag.y % 4u]; } // a lit face: as bright as its lamp burns
  if (kind == 1u && i.tag.z >= 254u) { // inside a nozzle (rocket.c): as bright as its lamp, and as hot as the boost runs it
    let b = C.boost.x; let h = select(0.2 + 0.7 * b * b, 0.3 + 0.7 * b, i.tag.z == 255u) + 0.35 * C.boost.w; // the liner further out lags the hottest; the turbo hotter still
    // as bright in its brightest channel as the face was made, less as it goes blue: blue light that bright would read white
    let c = jet_heat(h);
    o.col = c * (max(max(i.col.x, i.col.y), i.col.z) / max(max(c.x, c.y), c.z) * mix(1.0, 0.3, b)) * C.lamps[i.tag.y / 4u][i.tag.y % 4u];
    o.lens = vec2u(i.tag.y, 0u);
  }
  return o;
}
// What lights the car: the sky, the moon, its own beams; and, where they are let, the lamps that shine all round (the
// tail lamps' pool on the road, the TV tower, a tunnel's luminaires). The cabin and the wheels are left out of those, but
// not out of the light of its jets, which burn right beside them under its wings, nor of its wings' red and green
// navigation lights (C.glow[4] and [5], wasm/include/harstad/wing.h): those light the wings and the flanks near them.
// How far the car is out under the sky: 0 in a tunnel, coming on over its mouth as the luminaires' light on it goes
// (F.tunnel: TUNNEL_LIGHT at most, wasm/src/game.c). Under the rock nothing of the sky, the street lamps or the town
// over it reaches the car, nor does its paint mirror any of it.
fn car_outside() -> f32 { return 1.0 - saturate(F.tunnel.r / 12.0); }
fn car_nav(w: vec3f, n: vec3f) -> vec3f {
  var c = vec3f(0.0);
  for (var k = 4; k < 4 + i32(C.lit.z); k++) {
    let col = C.glow[k].rgb;
    if (col.r + col.g + col.b <= 0.0) { continue; }
    let d = C.glow_at[k].xyz - w; let r2 = max(dot(d, d), 0.0025);
    c += col * (max(dot(n, d * inverseSqrt(r2)), 0.0) / r2);
  }
  return c;
}
// How much of a spot's light reaches w along the unit l (from w to it): inside its beam, softly to its edge
fn spot_beam(k: i32, l: vec3f) -> f32 {
  let a = C.spot_aim[k];
  if (a.w <= -1.0) { return 1.0; }
  return smoothstep(a.w, a.w + (1.0 - a.w) * 0.35, dot(-l, a.xyz));
}
// The spots' light on the car (AV_SPOTS): each falls off as the square of the distance, from no nearer than its size
fn car_spots(w: vec3f, n: vec3f) -> vec3f {
  var c = vec3f(0.0);
  for (var k = 0; k < i32(C.lit.x); k++) {
    let col = C.spot[k].rgb;
    if (col.r + col.g + col.b <= 0.0) { continue; }
    let d = C.spot_at[k].xyz - w; let s = C.spot_at[k].w; let r2 = max(dot(d, d), s * s); let l = d * inverseSqrt(max(dot(d, d), 1e-8));
    c += col * (spot_beam(k, l) * max(dot(n, l), 0.0) / r2);
  }
  return c;
}
// The glint of a lamp size m across at p on a surface (GGX, widened by the lamp's size as it is seen from here, so that
// what a mirror shows of it is never brighter than the lamp itself): radiance, times the lamp's colour over distance squared.
fn glint(w: vec3f, n: vec3f, view: vec3f, nv: f32, rough: f32, f0: vec3f, p: vec3f, size: f32) -> vec3f {
  let to = p - w; let d2 = max(dot(to, to), size * size); let d = sqrt(d2); let l = to / max(length(to), 1e-4);
  let nl = dot(n, l);
  if (nl <= 0.0) { return vec3f(0.0); }
  let h = normalize(l + view); let nh = max(dot(n, h), 0.0);
  let a = min(max(rough * rough, 0.002) + size / (2.0 * d), 1.0); let a2 = a * a;
  let dn = nh * nh * (a2 - 1.0) + 1.0;
  let fr = f0 + (1.0 - f0) * pow5(1.0 - max(dot(h, view), 0.0));
  return fr * (a2 / (PI * dn * dn)) * nl * 0.25 / (max(nv, 0.05) * d2);
}
fn car_light(w: vec3f, n: vec3f, shy: bool) -> vec3f {
  var sum = (sky_light(w.xz, n.y) + aurora_light(n)) * (car_outside() * sky_seen(w, n)) + moon_light(w, n, 0.0) + car_beams(w, n); // the sky as the ground's map says it is seen, as any surface takes it
  sum += car_jets(w, n) + car_nav(w, n) + car_spots(w, n); // not its own tail lamps', which shine away from it
  // washed: the lights that come and go are looked for (tower_wash). Each is nothing beyond its reach, but a shader that
  // has them in it is the slower for it on every pixel, reached or not (what a shader holds costs it, run or not: a test
  // of a uniform round them changed nothing, taking them out of the source did): most of what the car cost, on a mesh
  // whose triangles are a pixel or two each. So the car has two shaders, and the frame says which (game.c,
  // game_car_washed): this one only where one of those lights can reach the car. A tunnel's luminaires, which travel
  // with the car, are in both.
  if (!shy) { sum += ${washed ? 'tower_wash(w, n)' : 'lamp_at(w, n, F.tunnel_at, F.tunnel.rgb)'}; }
  return sum / PI;
}
// How much of a light lying along l from p, both in the body's space (x to the car's left, y up, z forward), gets into
// the cabin: through the side glass, the screen or the hatch glass, not through the roof, the pillars or the doors. The
// greenhouse as five planes (tools/car/car.js, ROOF): the screen from its foot (z 0.98, y 0.995) to its top (0.12, 1.40),
// the hatch glass from (-1.2, 1.425) to (-2.05, 1.08), the roof at 1.44, and the sides leaning in from 0.85 m out at the
// belt to 0.66 at the cant rail. Where the ray leaves says what it went through. soft: penumbra, m per m of path.
fn cabin_open(p: vec3f, l: vec3f, soft: f32) -> f32 {
  var t = 1e9; var which = -1;
  let N = array<vec4f, 5>(vec4f(0.0, 1.0, 0.471, 1.457), vec4f(0.0, 1.0, -0.406, 1.912), vec4f(0.0, 1.0, 0.0, 1.44),
    vec4f(1.0, 0.452, 0.0, 1.307), vec4f(-1.0, 0.452, 0.0, 1.307));
  for (var k = 0; k < 5; k++) {
    let dn = dot(N[k].xyz, l);
    if (dn > 1e-4) { let tk = (N[k].w - dot(N[k].xyz, p)) / dn; if (tk < t) { t = tk; which = k; } }
  }
  if (which < 0 || which == 2) { return 0.0; } // out through the roof, or not out at all
  let q = p + l * max(t, 0.0);
  let e = 0.008 + soft * max(t, 0.0);
  if (which == 0) { return smoothstep(-e, e, 0.70 - abs(q.x)); } // the screen, between the A-pillars
  if (which == 1) { return smoothstep(-e, e, 0.62 - abs(q.x)) * smoothstep(-e, e, q.y - 1.1); } // the hatch glass, over the tail
  // the side glass: over the belt and under the cant rail's frame, behind the A-pillar and ahead of the C-pillar, but for the B
  let screen = 1.457 - q.y - 0.471 * q.z; // how far behind the screen's plane, measured up
  return smoothstep(-e, e, q.y - 1.02) * smoothstep(-e, e, 1.39 - q.y) * smoothstep(-e, e, screen - 0.07) * smoothstep(-e, e, q.z + 1.38)
    * (1.0 - smoothstep(-e, e, 0.04 - abs(q.z + 0.34)));
}
// A needle over the cluster's picture, as a canvas strokes it (tools/car/car.js, needle): a broad faint line under a bright one,
// round-ended, from 24 px behind the hub to 130 px out, then the black hub over its root. q: from the dial's centre, in the
// picture's pixels; colours as the picture stores them, not linear.
fn dial_needle(c: vec3f, q: vec2f, dir: vec2f) -> vec3f {
  let d = distance(q, dir * clamp(dot(q, dir), -24.0, 130.0));
  var o = mix(c, vec3f(1.0, 0.302, 0.102), 0.28 * clamp(6.0 - d, 0.0, 1.0));
  o = mix(o, vec3f(1.0, 0.302, 0.102), clamp(3.0 - d, 0.0, 1.0));
  return mix(o, vec3f(0.0275, 0.0275, 0.0314), clamp(20.5 - length(q), 0.0, 1.0));
}
// The instrument cluster: its picture never changes but for an LCD's text. A lamp that is lit is taken from the picture's second
// layer, where they all are, and the speedometer's and rev counter's needles are drawn over it from where the C says they point.
fn cluster(uv: vec2f) -> vec3f {
  let p = uv * vec2f(1024.0, 320.0);
  var lit = 0.0;
  if (p.y < 96.0) { lit = select(select(0.0, C.dash.y, abs(p.x - 550.0) < 35.0), C.dash.x, abs(p.x - 470.0) < 35.0); } // the turn arrows and their bloom
  else if (abs(p.x - 499.0) < 13.0 && abs(p.y - 280.0) < 13.0) { lit = C.dash.z; } // the brake warning lamp
  let face = textureSampleLevel(dials, samp_clamp, uv, u32(lit + 0.5), 0.0).rgb;
  let speedo = p - vec2f(330.0, 150.0); let revs = p - vec2f(668.0, 150.0);
  if (min(dot(speedo, speedo), dot(revs, revs)) > 137.0 * 137.0) { return face; }
  let c = dial_needle(dial_needle(encode(face), speedo, C.needle.xy), revs, C.needle.zw);
  return select(pow((c + 0.055) / 1.055, vec3f(2.4)), c / 12.92, c <= vec3f(0.04045));
}
// The cabin (kind 2), lit only by what lights it at night (CABIN_DARK), each surface as its finish has it (CABIN_FINISH,
// by its vertex's style). cabin: the shader for the finishes' own triangles (src/worker.js, car.runs), with the leather's
// and the vinyl's grain and what the glass lets them mirror; the matt and the flight cockpit's own faces go without.
fn cabin_finish(s: u32) -> vec4f { // roughness, grain cell (m), its tilt, f0
  return array<vec4f, ${CABIN_FINISH.length}>(${CABIN_FINISH.map(f => `vec4f(${ff(f.rough)}, ${ff(f.cell)}, ${ff(f.tilt)}, ${ff(f.f0)})`).join(', ')})[min(s, ${CABIN_FINISH.length - 1}u)];
}
${cabin ? `// value noise's gradient over a lattice of cells (q in cells), from a hash of each corner
fn grain_corner(c: vec3i) -> f32 {
  var h = (u32(c.x) * 73856093u) ^ (u32(c.y) * 19349663u) ^ (u32(c.z) * 83492791u);
  h = (h ^ (h >> 16u)) * 0x7feb352du; h = (h ^ (h >> 15u)) * 0x846ca68bu; h = h ^ (h >> 16u);
  return f32(h >> 8u) * (1.0 / 8388608.0) - 1.0;
}
fn grain_noise(q: vec3f) -> vec3f {
  let i = vec3i(floor(q)); let f = fract(q);
  let u = f * f * (3.0 - 2.0 * f); let du = 6.0 * f * (1.0 - f);
  let a = grain_corner(i); let b = grain_corner(i + vec3i(1, 0, 0)); let c = grain_corner(i + vec3i(0, 1, 0)); let d = grain_corner(i + vec3i(1, 1, 0));
  let e = grain_corner(i + vec3i(0, 0, 1)); let g = grain_corner(i + vec3i(1, 0, 1)); let h = grain_corner(i + vec3i(0, 1, 1)); let k = grain_corner(i + vec3i(1, 1, 1));
  let k1 = b - a; let k2 = c - a; let k3 = e - a; let k4 = a - b - c + d; let k5 = a - c - e + h; let k6 = a - b - e + g; let k7 = -a + b + c - d + e - g - h + k;
  return du * vec3f(k1 + k4 * u.y + k6 * u.z + k7 * u.y * u.z, k2 + k5 * u.z + k4 * u.x + k7 * u.z * u.x, k3 + k6 * u.x + k5 * u.y + k7 * u.x * u.y);
}
` : ''}// A light's sheen off the surface, from along l: normalised Blinn-Phong of spread sp and Schlick's Fresnel, as a share of
// the light falling there; capped, so that no reflection is brighter than what it mirrors.
fn cabin_sheen(n: vec3f, view: vec3f, l: vec3f, sp: f32, f0: f32) -> f32 {
  let h = normalize(l + view);
  let f = f0 + (1.0 - f0) * pow5(1.0 - max(dot(l, h), 0.0));
  return f * min((sp + 8.0) * 0.125 * pow(max(dot(n, h), 0.0), sp), 40.0);
}
fn cabin_shade(albedo: vec3f, n0: vec3f, w: vec3f, style: u32) -> vec3f {
  let B = C.parts[1]; let R = mat3x3f(B[0].xyz, B[1].xyz, B[2].xyz);
  let p = (w - B[3].xyz) * R; // in the body's space
  let to_eye = F.cam_pos.xyz - w; let dist = length(to_eye); let view = to_eye / dist;
  let fin = cabin_finish(style);
  var n = n0; var rough = fin.x;
${cabin ? `  if (fin.y > 0.0) { // the grain tips the normal; under a few pixels a cell fades to its mean and widens the sheen instead
    let foot = dist * pixel_angle();
    let near0 = 1.0 - smoothstep(0.25, 0.5, foot / fin.y); let near1 = 1.0 - smoothstep(0.25, 0.5, foot / (fin.y * 0.43));
    if (near0 > 0.0) {
      let q = p / fin.y;
      let g = R * ((grain_noise(q) * near0 + grain_noise(q * 2.33 + 17.0) * (0.45 * near1)) * fin.z);
      n = normalize(n - (g - n * dot(g, n)));
    }
    rough = sqrt(rough * rough + fin.z * fin.z * (0.8 * (1.0 - near0) + 0.16 * (1.0 - near1)));
  }
` : ''}  let a = rough * rough; let sp = max(2.0 / (a * a) - 2.0, 0.5);
  if (C.lamps[2].w > 0.5) { // ?cabinlight=1: lit evenly from all round, and broadly from over the screen to show the sheens
    let key = normalize(R * vec3f(-0.3, 0.8, 0.5));
    return albedo * (vec3f(0.55 + 0.3 * dot(n0, R[1])) + vec3f(0.9) * max(dot(n, key), 0.0)) + vec3f(0.5) * cabin_sheen(n, view, key, sp, fin.w);
  }
  // the instruments' backlight and the switches' lamps: only what faces them, near by; and their sheen
  var glow = vec3f(0.0); var sheen = vec3f(0.0);
  for (var k = 0; k < max(i32(C.lit.w), 3); k++) { // the instruments, the jets' switches, the wings' navigation lights and the floor's well lamps (dark but in flight)
    if (k >= 3 && C.glow[k].r + C.glow[k].g + C.glow[k].b <= 0.0) { continue; }
    let d = C.glow_at[k].xyz - w;
    let r2 = dot(d, d); let l = d * inverseSqrt(r2);
    let e = C.glow[k].rgb * (${f4(CABIN_DARK.glow)} * max(dot(n0, l), 0.0) / (1.0 + r2 * ${f4(CABIN_DARK.glow_fall)}));
    glow += e;${cabin ? ' sheen += e * cabin_sheen(n, view, l, sp, fin.w);' : ''}
  }
  // What comes in through the glass: the nearest street lamps, the moon, a tunnel's luminaires, each only where the roof,
  // the pillars and the doors let it, so that their light and their sheen cross the seats and the dash as the car goes by
  var lit = vec3f(0.0);
  for (var k = 0; k < i32(C.lit.y); k++) {
    let col = C.street[k].rgb;
    if (col.r + col.g + col.b <= 0.0) { continue; }
    let to = C.street_at[k].xyz - w; let d2 = dot(to, to); let d = sqrt(d2); let l = to / d;
    let nl = max(dot(n0, l), 0.0);
    if (nl <= 0.0) { continue; }
    let e = col * (cabin_open(p, l * R, 0.3 / d) * nl / max(d2, 1.0)); // a lamp's head is about 0.3 m across
    lit += e;${cabin ? ' sheen += e * cabin_sheen(n, view, l, sp, fin.w);' : ''}
  }
  // the wings' undersides are their doors' cards: in flight the flames and the scan lamps light them from outside
  lit += car_spots(w, n0);
  let moon = moon_light(w, n0, 0.0) * cabin_open(p, MOON_DIR * R, 0.01);
${cabin ? '  sheen += moon * (cabin_sheen(n, view, MOON_DIR, sp, fin.w) / PI);' : ''}
  var tunnel = vec3f(0.0);
  if (F.tunnel_at.w > 0.0) {
    let l = normalize(F.tunnel_at.xyz - w);
    tunnel = lamp_at(w, n0, F.tunnel_at, F.tunnel.rgb) * cabin_open(p, l * R, 0.08);
${cabin ? '    sheen += tunnel * cabin_sheen(n, view, l, sp, fin.w);' : ''}
  }
  // The sky's light reaches a surface only through the glass it faces: the share of its view that is window, taken
  // from how open the way out is along its normal tilted up (the roof and the doors are dark at night), at the most
  // CABIN_DARK.sky of the sky's light outside. The footwells and the dash's face see next to none of it.
  let sees = cabin_open(p, normalize(n0 * R + vec3f(0.0, 0.6, 0.0)), 0.6);
  // the sky's light here (sky_light), its parts looked up once for both the light and what the gloss mirrors of it: the
  // town's light over the place and the lamps' pools on the ground there
  let sky_g = town_light(w.xz); let sky_m = lamps_at(w.xz);
  let sky_below = town_below(sky_g, street_light(sky_m) + vec3f(sky_m.g * 1.25));
  let ambient = ((sky_from(sky_g, n0.y, sky_below) + aurora_light(n0)) * (${f4(CABIN_DARK.sky)} * sees) + moon) / PI;
  var c = albedo * (ambient + glow + lit + tunnel) + sheen;
${cabin ? `  // what is outside, mirrored where the glass lets it be seen: sharply in the gloss black, as the sky's mean in the rest
  let mirror = array<f32, ${CABIN_FINISH.length}>(${CABIN_FINISH.map(f => ff(f.mirror)).join(', ')})[min(style, ${CABIN_FINISH.length - 1}u)];
  let refl = reflect(-view, n);
  let fres = fin.w + (1.0 - fin.w) * pow5(1.0 - max(dot(n, view), 0.0)) * (1.0 - rough);
  var out = sky_from(sky_g, dot(refl, R[1]), sky_below) / PI;
  if (style == 3u) { // the sky as the water mirrors it, and the moon's aureole, which that leaves out
    let h = max(dot(refl, MOON_DIR), 0.0);
    out = textureSampleLevel(sky_map, samp_sky, vec2f(atan2(refl.z, refl.x) / 6.2831853 + 0.5, sqrt(asin(clamp(refl.y, 0.0, 1.0)) / 1.5707963)), 0.0).rgb
      + vec3f(0.55, 0.66, 0.9) * F.moon_now.rgb * ((pow(h, 40.0) * 0.05 + pow(h, 700.0) * 0.5) * ${NIGHT});
  }
  c += out * (fres * mirror * cabin_open(p, refl * R, 0.02 + rough * 0.6));
` : ''}  return c;
}
${flight ? FLIGHT_FNS : ''}// The light off the car, linear, and its opacity; fs finishes it, once. With finish at each of seven returns, and
// so seven copies of the haze's sixteen steps of the car's lamps (cloud_lamps), compiling the car's pipelines crashed
// Chrome's GPU process on Intel's Vulkan driver (Mesa's ANV, tsp2)
fn car_shade(v: V) -> vec4f {
  let m = C.mats[v.mat];
  let kind = u32(m.emissive.a + 0.5);
  let which = u32(m.params.w + 0.5);
  var picture = vec3f(0.0); // only a material that shows one looks its picture up
  if (kind == 3u || kind == 6u) {
    if (which == 0u) { picture = textureSampleLevel(plate, samp_clamp, v.uv, 0.0).rgb; }
    else if (which == 1u) { picture = cluster(v.uv); }
${flight ? '    else if (which == 3u) { picture = flight(v.uv); }' : ''}
    else { picture = textureSampleLevel(panels, samp_clamp, v.uv, 0.0).rgb; }
  }
  let r = length(v.uv);
  let n = normalize(v.n);
  if (kind == 1u) {
    if (v.lens.y == 0u${lens ? '' : ' || true'}) { return vec4f(v.col, 1.0); }
${lens ? `    // A lens, its chamber behind it (lamp_chamber): u runs along the lens square to the body's up, as tools/car/car.js
    // lays its uv out, v up it. Lit as its lamp burns; out, the chrome in it takes what comes in from round the car.
    let up = C.parts[1][1].xyz;
    let tu = normalize(cross(up, n)); let tv = cross(n, tu);
    let eye = v.w - F.cam_pos.xyz; let dist = length(eye); let d = eye / dist;
    let level = C.lamps[v.lens.x / 4u][v.lens.x % 4u];
    let on = max(level - ${LENS.off}, 0.0) / ${(1 - LENS.off).toFixed(2)};
    // what comes in through it: the sky, the moon and the street lamps, not its own car's beams or tail lamps' pool
    let env = (sky_light(v.w.xz, n.y) * car_outside() + moon_light(v.w, n, 0.0)) / PI + lamp_light(v.w, n) * (1.6 * car_outside());
    let inside = lamp_chamber(v.uv, vec3f(dot(d, tu), dot(d, tv), -dot(d, n)), lens_style(v.lens.y), v.col * on, env, dist * pixel_angle());
    // and the outer glass mirrors the sky and the street round it
    let nv = abs(dot(d, n)); let f = 0.04 + 0.96 * pow5(1.0 - nv);
    return vec4f(inside * (1.0 - f) + sky(reflect(d, n), 0.0) * f, 1.0);
` : ''}  }
  if (kind == 3u) { return vec4f(picture * m.albedo.rgb, 1.0); }
  if (kind == 6u) { return vec4f(picture * (m.albedo.rgb * car_light(v.w, n, false) + m.emissive.rgb), 1.0); } // and its own lamps' light (the rear plate's)
  if (kind == 4u) { // the shadow: darkest under the car, gone at the rim
    let t = clamp((r * 32.0 - 4.0) / 28.0, 0.0, 1.0);
    let a = m.albedo.a * select(mix(0.4, 0.0, (t - 0.6) / 0.4), mix(0.62, 0.4, t / 0.6), t < 0.6);
    return vec4f(m.albedo.rgb, a);
  }
  let albedo = m.albedo.rgb * v.col;
  // street-lamp light at the strength any lamp-lit surface takes it: it lands on the diffuse part of the paint and the trim
  let street_k = 1.6 * car_outside();
  let street = lamp_light(v.w, n) * street_k;
  let street_up = lamp_light(v.w, vec3f(0.0, 1.0, 0.0)) * street_k; // what they lay on the ground about it
  if (kind == 2u) { return vec4f(cabin_shade(albedo, n, v.w, v.lens.y), 1.0); }
  let view = normalize(F.cam_pos.xyz - v.w);
  let nv = max(dot(n, view), 0.001);
  let rough = m.params.x; let metal = m.params.y;
  let f0 = mix(vec3f(0.04), albedo, metal);
  let fresnel = f0 + (1.0 - f0) * pow5(1.0 - nv);
  var c = albedo * (1.0 - metal) * car_light(v.w, n, which == 1u);
  if (v.mat < 2u) { c += albedo * (1.0 - metal) * street; }
  { // The sky mirrored in it, by three.js's split-sum fit: how much comes back depends on roughness and angle, and a
    // rough surface loses none of it, it only blurs. On dark paint at night this is most of what there is to see.
    let r4 = rough * vec4f(-1.0, -0.0275, -0.572, 0.022) + vec4f(1.0, 0.0425, 1.04, -0.04);
    let a004 = min(r4.x * r4.x, exp2(-9.28 * nv)) * r4.x + r4.y;
    let fab = vec2f(-1.04, 1.04) * a004 + r4.zw;
    let once = f0 * fab.x + fab.y;
    let rest = 1.0 - (fab.x + fab.y);
    let mean = f0 + (1.0 - f0) * 0.047619;
    // below the horizon what it mirrors is the snow under it, lit by the moon and the sky: brighter than the dark
    // horizon sky() gives there, and no brighter than that snow
    let rv = reflect(-view, n);
    // and by the street lamps round it, which light that snow as they light everything
    let snow_below = vec3f(0.85) * ((moon_now() * MOON_DIR.y + sky_up(v.w.xz)) / PI + street_up);
    // in a tunnel, its lining, as the luminaires light it (TUBE: half its own colour), not the sky
    // and where the sky is shut out (the fronts across the street, the trees), a wall of albedo 0.3 lit by the lamps, as the snow is
    let env = mix(vec3f(0.24, 0.23, 0.21), mix(mix(vec3f(0.3) * street_up, sky(rv, 0.0), sky_seen(v.w, rv)), snow_below, smoothstep(0.0, -0.2, rv.y)), car_outside());
    c += env * (once + once * mean / (1.0 - rest * mean) * rest) * m.params.z;
  }
  let half_v = normalize(MOON_DIR + view);
  let a2 = max(rough * rough * rough * rough, 1e-4);
  let nh = max(dot(n, half_v), 0.0);
  let dn = nh * nh * (a2 - 1.0) + 1.0;
  let moon_nl = max(dot(n, MOON_DIR), 0.0);
  if (moon_nl > 0.0) { c += moon_now() * fresnel * min(a2 / (PI * dn * dn), 40.0) * moon_nl * moon_shadow(v.w, n) * 0.25 / max(nv, 0.05); } // only where the moon's light falls
  { // the glints of the lamps near it: the street's nearest, the wings' navigation lights, the flames, the scan lamps
    var g = vec3f(0.0);
    for (var k = 0; k < i32(C.lit.y); k++) {
      let col = C.street[k].rgb;
      if (col.r + col.g + col.b > 0.0) { g += col * glint(v.w, n, view, nv, rough, f0, C.street_at[k].xyz, 0.18) * car_outside(); }
    }
    for (var k = 4; k < 4 + i32(C.lit.z); k++) {
      let col = C.glow[k].rgb;
      if (col.r + col.g + col.b > 0.0) { g += col * glint(v.w, n, view, nv, rough, f0, C.glow_at[k].xyz, 0.015); }
    }
    for (var k = 0; k < i32(C.lit.x); k++) {
      let col = C.spot[k].rgb;
      if (col.r + col.g + col.b <= 0.0) { continue; }
      let p = C.spot_at[k].xyz;
      g += col * glint(v.w, n, view, nv, rough, f0, p, C.spot_at[k].w) * spot_beam(k, normalize(p - v.w));
    }
    c += g * m.params.z;
  }
  let to_lamp = F.tunnel_at.xyz - v.w; // in a tunnel the luminaires glint on the paint as they pass overhead
  let dl = length(to_lamp);
  if (dl < F.tunnel_at.w && which != 1u) {
    let l = to_lamp / dl;
    let nl = max(dot(n, normalize(l + view)), 0.0);
    let dm = nl * nl * (a2 - 1.0) + 1.0;
    c += F.tunnel.rgb * lux_at(dl * dl, F.tunnel_at.w) * fresnel * min(a2 / (PI * dm * dm), 40.0) * max(dot(n, l), 0.0) * 0.25 / max(nv, 0.05);
  }
  return vec4f(c, m.albedo.a);
}
// out: colour times opacity, and opacity (glare has none: it only adds)
@fragment fn fs(v: V) -> @location(0) vec4f {
  let s = car_shade(v);
  return vec4f(finish(s.rgb, v.w) * s.a, s.a);
}
`;

// and the flight instruments on the cluster board's back (cockpit.h), likewise only on their own triangles
export const CAR = car(true), CAR_BODY = car(false), CAR_FLIGHT = car(false, true), CAR_CABIN = car(false, false, true);
export const CAR_WASHED = car(false, false, false, true); // the body's, where a light that comes and goes reaches the car

/* ---- sprites: a camera-facing square per instance, sized like a GL point ---- */

const CORNERS = /* wgsl */`
const CORNER = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0));
// the same two triangles from four corners, drawn through the worker's quad index (0 1 2, 0 2 3): four vertices shaded
// a sprite where six were
const QUAD = array<vec2f, 4>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0));
// a point of \`size\` pixels across, centred on clip position p
fn sprite(p: vec4f, size: f32, corner: vec2f) -> vec4f {
  return vec4f(p.xy + corner * size / vec2f(F.cam_fwd.w, F.cam_up.w) * p.w, p.zw);
}
`;

// each lamp's lens, bulb and lit pane, and the point of light it shrinks to far off: the brighter of it and what is behind
export const GLOW = CORNERS + /* wgsl */`
// col: the disc as the screen shows it, and its opacity; pt: the point's colour times its light, display-linear (its flux,
// in pixels' worth of that colour); at: where the point falls, in pixels
struct V { @builtin(position) pos: vec4f, @location(0) uv: vec2f, @location(1) @interpolate(flat) col: vec4f,
  @location(2) @interpolate(flat) pt: vec3f, @location(3) @interpolate(flat) at: vec2f }
// A lamp's lens, a bulb, a lit pane: drawn as itself, a crisp lit disc, never as glare round it (no fake glow, anywhere).
// look: opacity, negative where the size given is the lens's own (the strung bulbs, the traffic's and ships' lamps); where
// positive, the size given is that of the glare the sprites once drew, whose bright core (GLOW_CORE of it) is the lens.
// Height over width (its sign unused: the lights in the sky were once let through the haze whole). An opacity
// over 2 is a light across the fjord (wasm/src/fjord.c), drawn at full opacity.
//
// What a lens truly gives is its gain (wasm/include/harstad/gfx.h, gfx_lens_gain) times the radiance it is drawn at:
// thousands of times moonlit snow for a lamp. Seen as a disc it is that radiance, clipped at white by its brightest
// channel alone, so that where it clips it keeps its hue (sodium orange, a tail lamp's red) and does not go to white. Under a
// pixel (POINT_PX) it becomes the point a camera or the eye sees: its light over the pixel, its radiance times the share
// of the pixel it covers, so a lamp stays a full bright point until it is far enough for that to fall under the
// screen's white, and only then fades, as the inverse square. The point's flux is shared over the four pixels round it
// by their nearness to it (a tent a pixel wide: its shares always sum to one), so that as it moves between pixels it
// neither crawls nor dims, whatever the samples the 4x MSAA takes. The flux is exposed whole, before it is shared, and
// clipped at POINT_MOST pixels' worth of white: a point too bright
// for one pixel fills the pixels it falls between as a camera's sensor would, each to its white, and does not go grey
// split four ways; a faint one is shared as it was. So a headlamp or a lit window far off is a white-hot pixel wherever
// it falls, and what the screen shows of it, summed, wanders by at most POINT_MOST as it moves. A lens is drawn over
// what is behind it as the brighter of the two (BLEND.lens in src/worker.js): it hides what is behind it, and where many
// points fall on one pixel (a Christmas tree's bulbs a kilometre off) they clip at the brightest, not add up to white,
// as a camera's pixel clips what lands on it. Every light alike, on the ground or in the sky: through the air its
// light is dimmed by FAR_AIR of the haze's density and by any snow shower; the air adds nothing of its own over it.
// The air's turbulence makes a point twinkle (scintillation): its brightness and a little its colour wander, smoothly,
// more as its path through the low, dense air is long; a lens over a pixel does not twinkle.
const GLOW_CORE = 0.14;
const POINT_PX = vec2f(0.75, 1.5); // a lens's size in pixels over which its point gives way to its disc
const POINT_MOST = 2.0;            // pixels' worth of white a point's light is clipped at
const TWINKLE = 0.35;              // the most a point's light wanders, over a long path low through the air
const TWINKLE_PATH = 4000.0;       // m of air at sea level that is a long path
const TWINKLE_H = 150.0;           // m: the scale height of the air's turbulence
@vertex fn vs(@builtin(vertex_index) vi: u32, @location(0) at: vec4f, @location(1) col: vec4f, @location(2) look: vec2f, @location(3) gain: f32) -> V {
  var o: V;
  // a mesh's lens (a street lamp's head) drawn a little towards the eye, so that the head round it does not hide its point
  let lens_at = at.xyz - (at.xyz - F.cam_pos.xyz) * select(0.0, 0.6 / max(distance(at.xyz, F.cam_pos.xyz), 1.0), gain < 0.0);
  let p = F.view_proj * vec4f(lens_at, 1.0);
  let to = at.xyz - F.cam_pos.xyz; let dist = length(to);
  let depth = dot(to, F.cam_fwd.xyz);
  // three's size attenuation: size * (drawing-buffer height / 2) / depth, in pixels; then the lens alone
  let px = at.w * F.cam_up.w * 0.5 / max(depth, 0.001) * select(GLOW_CORE, 1.0, look.x < 0.0);
  let tall = abs(look.y);
  let point = 1.0 - smoothstep(POINT_PX.x, POINT_PX.y, px);
  let disc = select(1.0 - point, 0.0, gain < 0.0); // a mesh draws the lens itself (a street lamp's head): only its point here
  let thru = lens_thru(at.xyz); // what the air lets through (lens_thru)
  let lens = unpack(col) * max(abs(gain), 1.0) * thru; // its true radiance, as it reaches the eye
  let k = min(abs(look.x), 1.0);
  // the twinkle: two slow waves, their phases from where the light is (continuous as it moves), the colours a little apart
  let path = 1.0 - exp(-dist * exp(-0.5 * (max(F.cam_pos.y, 0.0) + max(at.y, 0.0)) / TWINKLE_H) / TWINKLE_PATH);
  let ph = 6.2832 * fract(vec2f(dot(at.xyz, vec3f(0.0613, 0.0471, 0.0827)), dot(at.xyz, vec3f(0.0539, 0.0917, 0.0389))));
  let t = F.cam_pos.w + vec3f(-0.04, 0.0, 0.04);
  let wander = 0.6 * sin(t * 7.3 + ph.x) + 0.4 * sin(t * 12.1 + ph.y);
  let twinkle = 1.0 + TWINKLE * path * point * wander;
  // the point's light, exposed whole and clipped at POINT_MOST pixels of white, its hue kept (tone_hue's, stretched)
  let flux = lens * (0.7854 * px * px) * (k * point) * twinkle;
  let fm = max(max(flux.r, flux.g), flux.b);
  o.pt = select(vec3f(0.0), flux / fm * (POINT_MOST * tone(vec3f(fm / POINT_MOST)).g), fm > 0.0);
  o.col = vec4f(encode(tone_hue(lens)), k * disc);
  // the quad: the disc at its size, and the three pixels across the point's tent reaches; none at all where neither shows
  let wide = select(0.0, 3.0, point > 0.0);
  let size = select(vec2f(0.0), max(vec2f(px, px * tall) * step(1e-4, disc), vec2f(wide)), disc + point > 0.0);
  o.pos = sprite(p, 1.0, CORNER[vi] * size);
  o.uv = CORNER[vi] * size / max(vec2f(px, px * tall), vec2f(1e-4));
  o.at = vec2f(p.x / p.w * 0.5 + 0.5, 0.5 - p.y / p.w * 0.5) * vec2f(F.cam_fwd.w, F.cam_up.w);
  return o;
}
@fragment fn fs(v: V) -> @location(0) vec4f {
  // the lens: evenly lit to its rim, the rim antialiased over the last quarter of its radius, nothing beyond it
  let a = (1.0 - smoothstep(0.7, 1.0, length(v.uv))) * v.col.a;
  // the point: its share on this pixel by the pixel's centre's nearness to it
  let d = max(1.0 - abs(v.pos.xy - v.at), vec2f(0.0));
  let p = v.pt * (d.x * d.y);
  return vec4f(v.col.rgb * a + encode(p / max(max(max(p.r, p.g), p.b) / hdr_peak(), 1.0)), 0.0); // a pixel clips at the screen's white (on an HDR one its peak), keeping the hue
}
`;

// Cloud (wasm/src/clouds.c): broken stratocumulus, each cloud CLOUD_PUFFS puffs of droplets heaped on a flat floor.
// A puff's density falls off from its middle as a gaussian (sigma r m across, h m up), and is none under the cloud's
// floor, so that along any ray its density is a gaussian in how far along it is: what it hides, where it is seen
// and what of the moon reaches there are integrals of it in closed form (erf), with nothing marched. Its edge is
// worn ragged by the noise's cells (cloudVolume), drifting slowly through it, at the point the ray passes nearest
// its heart, as finely as the pixel can tell (nothing shimmers). Lit as light is scattered in it, at the point most
// of what is seen of it lies (where the ray has let through half of what the puff will take): the moon's through the
// puff towards it and the cloud's other puffs over it (Beer's law), forward-peaked (two Henyey-Greenstein lobes), with
// the light scattered more than once as octaves that reach deeper and spread wider (Wrenninge 2015), and the thin
// edges dark facing it (the powder effect); the night sky's light from above and the aurora's, as the ground has
// them, and the town's light the air over the cloud gives back down (sky_up's, as much of it as the haze over the
// cloud's height holds: AIR.h); and from below the snow's, moonlit where the clouds let the moon through
// (cloud_shade), and the town's light leaving its ground (the town glow's field: as much as comes back down of it,
// over SKYGLOW_BACK). Its droplets absorb nothing: the octaves are weighed (ms) so that a deck reflects and lets
// through what the delta-Eddington two-stream gives a conservative slab of the phase's mean cosine (0.59) with the
// moon 38 degrees up: a thick deck from above some 0.75 of the light, as the snow (0.85) nearly.
const CLOUD_SHAPE = {
  sigma: 0.07,    // extinction per m at full density: some 40 m to see in the thick of one
  shape: 760,     // m the noise tiles over, for the cells; its cells 190, 95, 48 m
  floor: 20,      // m the deck's floor and roof are inside its box's (clouds.c, FLOOR)
  ground: 0.7,    // the snow under the clouds: open snow and forest
  ms: [0.75, 0.85, 0.4], // the octaves of light scattered more than once: the first's weight, each next's of the last's, and the
                         // share of the optical depth each next takes of the last's (fitted to the two-stream slab, see above)
  through: 0.45,  // of the light from the far side of the deck, the share its diffuse light gives out of this side
  reach: 3.0,     // sigmas out from a puff's middle it is drawn to: what it hides there is taken off all of it
};
export const CLOUD_PUFFS = 32; // wasm/include/harstad/clouds.h
export const CLOUD_DIV = 4; // the clouds' picture is drawn this many times smaller each way than the frame
const CLOUD_MS = `var ka = ${CLOUD_SHAPE.ms[2].toFixed(2)}; var kb = ${CLOUD_SHAPE.ms[0].toFixed(2)} * deep;`;
const CLOUD_MS_NEXT = `ka *= ${CLOUD_SHAPE.ms[2].toFixed(2)}; kb *= ${CLOUD_SHAPE.ms[1].toFixed(2)};`;
// The frame's clouds are drawn at half the frame's size each way, in a pass of their own before the frame's: a puff,
// its ragged edge and the noise's finest cells it shows are two of that pass's pixels or more. Every puff goes into the
// one picture, far to near, each over those behind it (its colour times its cover: laid over, not blended in); and
// beside it goes how near the nearest puff on each pixel is seen. The frame then lays that picture over itself
// (CLOUD_OVER) at that depth, so what stands in front of the clouds hides them to the frame's own sample.
// The clouds' puffs, which the picture (CLOUD) and the moon's shadow of them (CLOUD_SHADE) share
const CLOUD_FIELD = /* wgsl */`
@group(1) @binding(0) var cloud_noise: texture_3d<f32>;
@group(1) @binding(1) var samp_cloud: sampler;
struct Puff { at: vec4f, h: vec4f } // its middle, sigma across; sigma up, density, the cloud's floor, its deck's roof (clouds.h, struct cloud_puff)
struct Box { c: vec4f, h: vec4f }   // its cloud's box: the middle, its seed; half sizes, its turn (struct cloud_box)
@group(1) @binding(2) var<storage, read> puffs: array<Puff>;
@group(1) @binding(3) var<storage, read> boxes: array<Box>;
// the error function (Abramowitz and Stegun 7.1.26, within 1.5e-7) and its inverse (Giles 2010, single precision)
fn erf_(x: f32) -> f32 {
  let a = abs(x); let t = 1.0 / (1.0 + 0.3275911 * a);
  let y = 1.0 - t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))) * exp(-a * a);
  return select(-y, y, x >= 0.0);
}
fn erfinv_(x: f32) -> f32 {
  var w = -log(max((1.0 - x) * (1.0 + x), 1e-30));
  if (w < 5.0) {
    w -= 2.5;
    var p = 2.81022636e-08; p = 3.43273939e-07 + p * w; p = -3.5233877e-06 + p * w; p = -4.39150654e-06 + p * w; p = 0.00021858087 + p * w;
    p = -0.00125372503 + p * w; p = -0.00417768164 + p * w; p = 0.246640727 + p * w; p = 1.50140941 + p * w;
    return p * x;
  }
  w = sqrt(w) - 3.0;
  var p = -0.000200214257; p = 0.000100950558 + p * w; p = 0.00134934322 + p * w; p = -0.00367342844 + p * w; p = 0.00573950773 + p * w;
  p = -0.0076224613 + p * w; p = 0.00943887047 + p * w; p = 1.00167406 + p * w; p = 2.83297682 + p * w;
  return p * x;
}
// A puff along o + d t: its density there is g e^(-(t - tc)^2 / (2 s^2)) times its own, and the integral of that over
// all t is A. Returns tc, s, A (with the extinction), g.
fn puff_line(o: vec3f, d: vec3f, p: Puff) -> vec4f {
  let inv = vec3f(1.0 / p.at.w, 1.0 / p.h.x, 1.0 / p.at.w);
  let a = (o - p.at.xyz) * inv; let b = d * inv;
  let bb = dot(b, b); let ab = dot(a, b);
  let g = exp(-0.5 * max(dot(a, a) - ab * ab / bb, 0.0)); let s = inverseSqrt(bb);
  return vec4f(-ab / bb, s, ${CLOUD_SHAPE.sigma.toFixed(4)} * p.h.y * g * s * ${Math.sqrt(Math.PI / 2).toFixed(7)}, g);
}
// what of a puff is left once the noise's cells have worn its edge, at w, where the ray through it passes as near its
// heart as g: the heart whole, the edge broken into the cells that rise highest there
fn puff_worn(w: vec3f, b: Box, lod: f32, g: f32) -> f32 {
  let off = vec3f(517.0, 133.0, 291.0) * b.c.w + vec3f(F.cam_pos.w * 0.3, 0.0, F.cam_pos.w * 0.12); // and slowly changing
  let s = textureSampleLevel(cloud_noise, samp_cloud, ((w - b.c.xyz) * vec3f(1.0, 2.4, 1.0) + off) * ${(1 / CLOUD_SHAPE.shape).toExponential(4)}, lod);
  let heap = smoothstep(0.3, 0.66, s.g * 0.75 + s.b * 0.25) * mix(0.55, 1.0, smoothstep(0.62, 0.92, s.r));
  return clamp((heap + 1.3 * g - 0.75) * 2.0, 0.0, 1.0);
}
`;
const CLOUD_CORNER = /* wgsl */`
fn quad_corner(vi: u32) -> vec2f { return vec2f(select(-1.0, 1.0, vi == 1u || vi == 2u || vi == 4u), select(-1.0, 1.0, vi == 2u || vi == 4u || vi == 5u)); }
`;
// What each puff takes the same at every pixel, worked out once a frame, a texel a puff (CLOUD_PREP_W of them a row):
// whether it shows at all, the optical depth of its cloud's other puffs from its middle to the moon, what lights it
// besides the moon, the air between it and the eye, and which lamps may reach it.
export const CLOUD_PREP_W = 256;
const PUFF_SHIFT = Math.log2(CLOUD_PUFFS);
export const CLOUD_PREP = /* wgsl */`
${CLOUD_FIELD}
struct Out { @location(0) a: vec4f, @location(1) b: vec4f, @location(2) c: vec4f, @location(3) d: vec4f }
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f { return vec4f(f32(i & 1u) * 4.0 - 1.0, f32(i >> 1u) * 4.0 - 1.0, 0.0, 1.0); }
@fragment fn fs(@builtin(position) at: vec4f) -> Out {
  var out: Out;
  let ii = u32(at.y) * ${CLOUD_PREP_W}u + u32(at.x);
  if (ii >= arrayLength(&puffs)) { return out; } // (past the puffs drawn: never drawn, whatever it says)
  let P = puffs[ii]; let c = P.at.xyz;
  let eye = F.cam_pos.xyz; let to = c - eye;
  let R = ${CLOUD_SHAPE.reach.toFixed(1)} * max(P.at.w, P.h.x);
  // none outside the view (its sphere against each side of it), or where what is nearest of it lies in a cloud's haze
  // deep enough to leave less than 1/400 of it (finish's fog)
  let z = dot(to, F.cam_fwd.xyz);
  let tx = length(F.cam_right.xyz); let ty = length(F.cam_up.xyz);
  let x = dot(to, F.cam_right.xyz) / tx; let y = dot(to, F.cam_up.xyz) / ty;
  let near = max(z - R, 0.0) * F.fog.a;
  if (P.h.y <= 0.0 || z < -R || abs(x) - tx * z > R * sqrt(1.0 + tx * tx) || abs(y) - ty * z > R * sqrt(1.0 + ty * ty) || near * near > 6.0) { return out; }
  // the cloud's other puffs between its middle and the moon
  let first = (ii >> ${PUFF_SHIFT}u) << ${PUFF_SHIFT}u;
  var tau = 0.0;
  for (var j = first; j < first + ${CLOUD_PUFFS}u; j++) {
    if (j == ii) { continue; }
    let l = puff_line(c, MOON_DIR, puffs[j]);
    tau += l.z * (1.0 + erf_(l.x * ${Math.SQRT1_2.toFixed(7)} / l.y));
  }
  // what lights it besides the moon: the night sky and the aurora from above; the snow under it, moonlit, and the town's
  // own light leaving its ground (town_up) from below
  let glow = town_light(c.xz).rgb;
  let above = (sky_now() + aurora_light(vec3f(0.0, 1.0, 0.0)) + glow * exp(-max(c.y, 0.0) / ${AIR.h.toFixed(1)})) * ${(1 / Math.PI).toFixed(6)};
  let floor_at = vec3f(c.x, ground_at(c.xz), c.z);
  let below = (${CLOUD_SHAPE.ground.toFixed(2)} * (moon_now() * (MOON_DIR.y * cloud_shade(floor_at)) + sky_up(c.xz)) + town_up(c.xz).rgb) * ${(1 / Math.PI).toFixed(6)};
  // the air between, linear in what is behind it; and the lamps that may light it: 1 the car's, 2 the list's
  let air0 = aerial(vec3f(0.0), c); let air1 = aerial(vec3f(1.0), c) - air0;
  let box = ii >> ${PUFF_SHIFT}u;
  let lamps = select(0.0, 1.0, length(F.car_pos.xyz - c) - R < car_reach()) +
    select(0.0, 2.0, shine_grid[${SHINE.clouds0 + 1}u + box] > shine_grid[${SHINE.clouds0}u + box] || F.heli_at.w > 0.0);
  out.a = vec4f(above, tau); out.b = vec4f(below, lamps); out.c = vec4f(air0, 1.0); out.d = vec4f(air1, 0.0);
  return out;
}
`;
export const CLOUD = /* wgsl */`
${CLOUD_FIELD}${CLOUD_CORNER}
@group(1) @binding(4) var prep_a: texture_2d<f32>; // CLOUD_PREP's
@group(1) @binding(5) var prep_b: texture_2d<f32>;
@group(1) @binding(6) var prep_c: texture_2d<f32>;
@group(1) @binding(7) var prep_d: texture_2d<f32>;
struct V {
  @builtin(position) pos: vec4f,
  @location(0) w: vec3f,                        // a point along the pixel's ray
  @location(1) @interpolate(flat) id: u32,      // which puff; its cloud's lights are the light list's (wasm/src/shine.c, shine_clouds)
  @location(2) @interpolate(flat) other: f32,   // the optical depth of the cloud's other puffs from its middle to the moon
  @location(3) @interpolate(flat) above: vec3f, // what lights it besides the moon, as radiance: from above,
  @location(4) @interpolate(flat) below: vec3f, // and from below
  @location(5) @interpolate(flat) air0: vec3f,  // the air between the eye and it: what it adds,
  @location(6) @interpolate(flat) air1: vec3f,  // and what it lets through
  @location(7) @interpolate(flat) lamps: u32,   // 1: the car's lamps may reach it; 2: lights of the list may; 4: it fills the view
  @location(8) phase: vec4f,                    // the droplets' phase towards the eye for the light scattered once and the
  @location(9) phase4: f32,                     // octaves after it (for a square, between its corners: it is small in the view)
}
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> V {
  var o: V;
  let at = vec2i(i32(ii % ${CLOUD_PREP_W}u), i32(ii / ${CLOUD_PREP_W}u));
  let c4 = textureLoad(prep_c, at, 0);
  if (c4.w <= 0.0) { o.pos = vec4f(0.0); return o; } // (not in view)
  let a4 = textureLoad(prep_a, at, 0); let b4 = textureLoad(prep_b, at, 0);
  o.id = ii; o.other = a4.w; o.above = a4.xyz; o.below = b4.xyz; o.air0 = c4.xyz; o.air1 = textureLoad(prep_d, at, 0).xyz; o.lamps = u32(b4.w + 0.5);
  let P = puffs[ii]; let c = P.at.xyz;
  let eye = F.cam_pos.xyz; let to = c - eye; let dist = length(to);
  let R = ${CLOUD_SHAPE.reach.toFixed(1)} * max(P.at.w, P.h.x);
  let q = quad_corner(vi);
  if (dist < R * 1.1 + 2.0) { // the eye in it, or nearly: the whole view
    o.pos = vec4f(q, 0.5, 1.0);
    o.w = eye + F.cam_fwd.xyz + F.cam_right.xyz * q.x + F.cam_up.xyz * q.y;
    o.lamps |= 4u;
  } else { // a square through its middle facing the eye, as wide as its sphere looks from there
    let f = to / dist; let side = normalize(cross(f, vec3f(0.0, 1.0, 0.0) + f.zxy * 1e-3)); let up = cross(side, f);
    let r = R * dist * inverseSqrt(dist * dist - R * R);
    o.w = c + (side * q.x + up * q.y) * r;
    o.pos = F.view_proj * vec4f(o.w, 1.0);
  }
  let mu = dot(normalize(o.w - eye), MOON_DIR);
  o.phase = vec4f(cloud_phase(mu, 1.0), cloud_phase(mu, 0.5), cloud_phase(mu, 0.25), cloud_phase(mu, 0.125)); o.phase4 = cloud_phase(mu, 0.0625);
  return o;
}
// A light of the light list as the air at w takes it, facing it: its light, and where that comes from. A car's dipped,
// near and full beams and its tail lamps' pool, and a round lamp's light, as traffic_lamps lays them on a surface.
struct Air { light: vec3f, at: vec3f }
fn list_air(L: Light, w: vec3f) -> Air {
  if (L.col.w > 0.5) {
    var light = L.col.rgb;
    if (abs(L.dir.x) > 0.5) { light *= event_aim(w, L.at.xyz, L.dir.x); }
    return Air(lamp_at(w, normalize(L.at.xyz - w), L.at, light) * light_seen(L, 2.0, w), L.at.xyz);
  }
  let dir = L.dir; let full = dir.w > 1.5;
  let fwd = vec3f(dir.x, 0.0, dir.y); let to = w - L.at.xyz; let d2 = dot(to, to); let along = dot(to, fwd);
  let far = select(${DIP_REACH.toFixed(1)}, ${FULL_REACH.toFixed(1)}, full);
  if (along > 2.2 && d2 < (far + 4.0) * (far + 4.0)) {
    let at = L.at.xyz + fwd * 2.2;
    var beam = traffic_beams(w, normalize(at - w), at, fwd, full);
    if (any(beam > vec3f(0.0))) { beam *= light_seen(L, ${SEEN.carHalf}, w); }
    return Air(beam, at);
  }
  let at = L.at.xyz - fwd * 2.3;
  if (along < 0.0 && d2 < 14.0 * 14.0) { return Air(tail_pool(w, normalize(at - w), at, -fwd, vec3f(1.0, 0.01033, 0.003677) * (${TAIL.pool.toFixed(3)} * dir.z)), at); }
  return Air(vec3f(0.0), at);
}
// A lamp's light at a point of cloud, as its moonlight is (fs): once, and the octaves of more than once, through tau of
// cloud on the way from it, at mu the cosine between the way to the lamp and the eye's ray
fn lamp_scatter(mu: f32, tau: f32, deep: f32) -> f32 {
  var ms = exp(-tau) * cloud_phase(mu, 1.0); ${CLOUD_MS} var k = 0.5;
  for (var n = 1; n < 5; n++) { ms += kb * exp(-tau * ka) * cloud_phase(mu, k); ${CLOUD_MS_NEXT} k *= 0.5; }
  return ms;
}
// How far along the ray the puff has shown q of what it shows (alpha, of optical depth tau between erf's ea and eb, the
// ray's nearest to its heart at tc, k the erf's scale): where the light that comes back along the ray has q of its weight
fn puff_seen(q: f32, alpha: f32, tau: f32, ea: f32, eb: f32, tc: f32, k: f32) -> f32 {
  return max(tc + erfinv_(clamp(ea + (eb - ea) * -log(1.0 - q * alpha) / tau, -0.999999, 0.999999)) / k, 0.0);
}
// the optical depth of puff P between w and \`at\`, worn as it is
fn puff_between(w: vec3f, at: vec3f, P: Puff, worn: f32) -> f32 {
  let len = length(at - w); let l = puff_line(w, (at - w) / max(len, 1e-3), P); let k = ${Math.SQRT1_2.toFixed(7)} / l.y;
  return l.z * worn * (erf_((len - l.x) * k) - erf_(-l.x * k));
}
struct Out { @location(0) c: vec4f, @location(1) near: vec4f }
@fragment fn fs(v: V) -> Out {
  var out: Out;
  let P = puffs[v.id]; let B = boxes[v.id >> ${PUFF_SHIFT}u];
  let eye = F.cam_pos.xyz;
  let wd = normalize(v.w - eye);
  let l = puff_line(eye, wd, P);
  // the stretch of the ray over the floor, ahead of the eye
  var ta = 0.0; var tb = 1e7;
  let fl = P.h.z;
  if (abs(wd.y) > 1e-6) { let tf = (fl - eye.y) / wd.y; if (wd.y > 0.0) { ta = max(ta, tf); } else { tb = min(tb, tf); } }
  else if (eye.y < fl) { discard; }
  if (tb <= ta) { discard; }
  if (2.0 * l.z < 0.002) { discard; } // (all of it along the ray would hide too little to see)
  let k = ${Math.SQRT1_2.toFixed(7)} / l.y;
  let ea = erf_((ta - l.x) * k); let eb = erf_((tb - l.x) * k);
  // (none at the square's edge: the puff ends where its sphere does, its density there taken off all of it)
  var tau = l.z * (eb - ea) * max(1.0 - ${Math.exp(-0.5 * CLOUD_SHAPE.reach ** 2).toExponential(4)} / l.w, 0.0) * ${(1 / (1 - Math.exp(-0.5 * CLOUD_SHAPE.reach ** 2))).toFixed(6)};
  if (tau < 0.002) { discard; }
  // its edge worn, where the ray passes nearest its heart, by the noise's mip the pixel's size there can tell
  let tp = clamp(l.x, ta, tb);
  let px = ${2 * CLOUD_DIV}.0 * F.cam_right.w / F.cam_up.w; // the picture's pixel, across, a metre out
  let lod = max(log2(max(px * tp, 1.0) * ${(64 / CLOUD_SHAPE.shape).toExponential(4)}), 0.0);
  let worn = puff_worn(eye + wd * tp, B, lod, l.w);
  tau *= worn;
  if (tau < 0.002) { discard; }
  // where it is seen: where the ray has let through half of what the puff takes of it
  let alpha = 1.0 - exp(-tau);
  let w = eye + wd * puff_seen(0.5, alpha, tau, ea, eb, l.x, k);
  // the moon there, through the rest of the puff and the cloud's others
  let m = puff_line(w, MOON_DIR, P);
  let tau_m = m.z * worn * (1.0 + erf_(m.x * ${Math.SQRT1_2.toFixed(7)} / m.y)) + v.other;
  let a = (w - P.at.xyz) / vec3f(P.at.w, P.h.x, P.at.w);
  let sig = ${CLOUD_SHAPE.sigma.toFixed(4)} * P.h.y * exp(-0.5 * dot(a, a)) * worn;
  let deep = 1.0 - exp(-sig * 60.0);
  let mu = dot(wd, MOON_DIR);
  var ph = array<f32, 5>(v.phase.x, v.phase.y, v.phase.z, v.phase.w, v.phase4);
  if ((v.lamps & 4u) != 0u) { ph = array<f32, 5>(cloud_phase(mu, 1.0), cloud_phase(mu, 0.5), cloud_phase(mu, 0.25), cloud_phase(mu, 0.125), cloud_phase(mu, 0.0625)); }
  var ms = exp(-tau_m) * ph[0]; ${CLOUD_MS}
  for (var n = 1; n < 5; n++) { ms += kb * exp(-tau_m * ka) * ph[n]; ${CLOUD_MS_NEXT} }
  let powder = mix(1.0, 1.0 - exp(-2.0 * sig * 25.0 - tau_m), 0.5 - 0.5 * mu); // the dark edges are seen with the moon behind the eye
  let up = clamp((w.y - fl) / max(P.h.w - fl, 1.0), 0.0, 1.0);
  var src = moon_now() * (ms * powder) + 0.8 * (v.above * mix(${CLOUD_SHAPE.through.toFixed(2)}, 1.0, up) + v.below * mix(1.0, ${CLOUD_SHAPE.through.toFixed(2)}, up));
  // and the car's lamps, through the puff between there and them, scattered as the moon's is; deep in a cloud's haze
  // the haze takes them over (cloud_lamps, laid over the clouds by CLOUD_OVER): what the eye is in is the haze
  let lamps_here = 1.0 - smoothstep(0.25, 0.6, F.cloud.w);
  if ((v.lamps & 1u) != 0u && lamps_here > 0.0) {
    let cu = F.car_up.xyz; let cf = F.car_dir.xyz;
    let lf = normalize(F.car_pos.xyz + cu * 0.73 + cf * 2.3 - w); let lb = normalize(F.car_pos.xyz + cu * 0.85 - cf * 2.3 - w);
    let ef = car_beams_at(w, lf); let eb2 = car_tail_at(w, lb);
    if (any(ef + eb2 > vec3f(0.0))) {
      let tl = puff_between(w, F.car_pos.xyz, P, worn);
      src += (ef * lamp_scatter(dot(wd, lf), tl, deep) + eb2 * lamp_scatter(dot(wd, lb), tl, deep)) * lamps_here;
    }
  }
  // and the list's, each as the car's, and the air ambulance's searchlight
  if ((v.lamps & 2u) != 0u) {
    let box = v.id >> ${PUFF_SHIFT}u;
    let c0 = shine_grid[${SHINE.clouds0}u + box]; let c1 = shine_grid[${SHINE.clouds0 + 1}u + box];
    let c2 = ${OFF.has('cloudlist') ? 'c0' : 'c1 + select(0u, 1u, F.heli_at.w > 0.0)'}; // (?off=cloudlist: none of them)
    for (var i = c0; i < c2; i++) {
      var e = Air(vec3f(0.0), F.heli_at.xyz);
      if (i < c1) {
        let L = shine_lights[shine_grid[${SHINE.cindex0}u + i]];
        let to = L.at.xyz - w;
        if (dot(to, to) >= L.at.w * L.at.w) { continue; }
        e = list_air(L, w);
      } else { // (as tower_wash has it)
        e.light = spot(w, normalize(F.heli_at.xyz - w), F.heli_at.xyz, F.heli_aim.xyz, F.heli.rgb, F.heli_at.w, F.heli_aim.w, F.heli.w);
      }
      if (all(e.light <= vec3f(0.0))) { continue; }
      src += e.light * lamp_scatter(dot(wd, normalize(e.at - w)), puff_between(w, e.at, P, worn), deep);
    }
  }
  // drawn as it is there, through the air and the haze between (finish's, but for the lamps in a cloud's haze, which
  // the frame lays over the clouds: CLOUD_OVER)
  let depth = dot(w - eye, F.cam_fwd.xyz);
  let col = mix(encode(tone(src * v.air1 + v.air0)), F.fog.rgb, 1.0 - exp(-F.fog.a * F.fog.a * depth * depth));
  out.c = vec4f(col * alpha, alpha);
  let clip = F.view_proj * vec4f(w, 1.0);
  out.near = vec4f(select(0.0, clamp(clip.z / clip.w, 0.0, 1.0), alpha > 0.1));
  return out;
}
`;
// The moon's light let through the clouds, over the plane at sea level along its way (cloud_shade): each puff drawn as
// a square round its shadow there, each texel's line to the moon through it integrated in closed form, its edge worn as
// the picture's are, by the noise as fine as the map's texel can tell. What each lets through multiplies into the first
// layer; the second keeps how high the lowest puff on the way begins. The frame's uniforms alone are bound: the map
// itself is in the frame's group.
const FRAME_DECL = '@group(0) @binding(0) var<uniform> F: Frame;';
export const CLOUD_SHADE = PRELUDE.slice(PRELUDE.indexOf('struct Frame {'), PRELUDE.indexOf(FRAME_DECL) + FRAME_DECL.length) + CLOUD_FIELD + CLOUD_CORNER + /* wgsl */`
const MOON_DIR = ${vec3(MOON)};
${CLOUD_SHADE_AT}
struct V {
  @builtin(position) pos: vec4f,
  @location(0) s: vec2f,                   // the point on the plane
  @location(1) @interpolate(flat) id: u32, // which puff
}
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> V {
  var o: V;
  o.id = ii;
  let P = puffs[ii];
  if (P.h.y <= 0.0) { o.pos = vec4f(0.0); return o; }
  // its shadow's ellipse on the plane: sigma across, and along the moon's way its height's slant added to it
  let along = normalize(MOON_DIR.xz); let q = quad_corner(vi);
  let r = ${CLOUD_SHAPE.reach.toFixed(1)} * vec2f(length(vec2f(P.at.w, P.h.x * length(MOON_DIR.xz) / MOON_DIR.y)), P.at.w);
  o.s = P.at.xz - MOON_DIR.xz * (P.at.y / MOON_DIR.y) + along * (q.x * r.x) + vec2f(-along.y, along.x) * (q.y * r.y);
  let uv = (o.s - cloud_shade_origin()) * ${(1 / CLOUD_SHADE_SPAN).toExponential(6)};
  o.pos = vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.5, 1.0);
  return o;
}
struct Out { @location(0) t: vec4f, @location(1) base: vec4f }
@fragment fn fs(v: V) -> Out {
  var out: Out;
  let P = puffs[v.id]; let B = boxes[v.id >> ${PUFF_SHIFT}u];
  let o = vec3f(v.s.x, 0.0, v.s.y);
  let l = puff_line(o, MOON_DIR, P);
  let tf = P.h.z / MOON_DIR.y; let k = ${Math.SQRT1_2.toFixed(7)} / l.y;
  var tau = l.z * (1.0 - erf_((tf - l.x) * k));
  if (tau < 0.004) { discard; }
  tau *= puff_worn(o + MOON_DIR * max(l.x, tf), B, ${Math.log2(CLOUD_SHADE_SPAN / CLOUD_SHADE_PX * 64 / CLOUD_SHAPE.shape).toFixed(3)}, l.w);
  if (tau < 0.004) { discard; }
  out.t = vec4f(exp(-tau));
  out.base = vec4f(max(P.h.z, (l.x - 2.0 * l.y) * MOON_DIR.y));
  return out;
}
`;
// The clouds' picture laid over the frame: its colour between its pixels (a cubic), and the depth of the nearest cloud of the four
// pixels round (depth runs from 1 at the eye to 0 far off), so that a cloud's edge is hidden by what its cloud is. Only
// where there is cloud: the picture is laid tile by tile, CLOUD_TILE of its pixels a side, and a tile with no cloud in
// it nor within two of its pixels (CLOUD_TILES, a pass of a few thousand pixels) is not drawn at all. A pixel that sets its
// own depth is shaded whatever stands in front of it, and most of a frame has no cloud behind it.
export const CLOUD_TILE = 16;
export const CLOUD_TILES = /* wgsl */`
@group(0) @binding(0) var cloud_low: texture_2d<f32>;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f { return vec4f(f32(i & 1u) * 4.0 - 1.0, f32(i >> 1u) * 4.0 - 1.0, 0.0, 1.0); }
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let most = vec2i(textureDimensions(cloud_low)) - 1;
  var any = 0.0; // (and two of the picture's pixels round it: the laying over takes a pixel's colour from the four by four round it)
  for (var y = -2; y < ${CLOUD_TILE + 2}; y++) { for (var x = -2; x < ${CLOUD_TILE + 2}; x++) {
    any = max(any, textureLoad(cloud_low, clamp(vec2i(pos.xy) * ${CLOUD_TILE} + vec2i(x, y), vec2i(0), most), 0).a);
  } }
  return vec4f(any);
}
`;
export const CLOUD_OVER = /* wgsl */`
@group(1) @binding(0) var cloud_low: texture_2d<f32>;
@group(1) @binding(1) var cloud_near: texture_2d<f32>;
@group(1) @binding(2) var samp_low: sampler;
@group(1) @binding(3) var cloud_tiles: texture_2d<f32>;
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) tile: u32) -> @builtin(position) vec4f {
  let n = vec2i(textureDimensions(cloud_tiles));
  let at = vec2i(i32(tile) % n.x, i32(tile) / n.x);
  if (textureLoad(cloud_tiles, at, 0).r <= 0.0) { return vec4f(0.0); } // (no cloud in it, nor near enough its edge to show)
  let corner = vec2f(f32(vi == 1u || vi == 2u || vi == 4u), f32(vi == 2u || vi == 4u || vi == 5u)); // two triangles
  let uv = (vec2f(at) + corner) * ${CLOUD_TILE}.0 / vec2f(textureDimensions(cloud_low));
  return vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
}
struct Out { @location(0) c: vec4f, @builtin(frag_depth) depth: f32 }
@fragment fn fs(@builtin(position) pos: vec4f) -> Out {
  var out: Out;
  let size = vec2f(textureDimensions(cloud_low));
  let uv = pos.xy / vec2f(F.cam_fwd.w, F.cam_up.w);
  // Between the picture's pixels by the Catmull-Rom cubic over the four by four round the point, as nine looks that
  // each weigh two by two of them: a straight line between two pixels blunts a wisp's edge by a pixel, and a star
  // going behind it shows by how much. The curve can overshoot a little at a hard edge: the cover kept within 0 and 1.
  let st = uv * size - 0.5; let ip = floor(st); let f = st - ip;
  let w0 = f * (-0.5 + f * (1.0 - 0.5 * f)); let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
  let w2 = f * (0.5 + f * (2.0 - 1.5 * f)); let w3 = f * f * (-0.5 + 0.5 * f);
  let w12 = w1 + w2;
  let p0 = (ip - 0.5) / size; let p12 = (ip + 0.5 + w2 / w12) / size; let p3 = (ip + 2.5) / size;
  var c = vec4f(0.0);
  c += (textureSampleLevel(cloud_low, samp_low, vec2f(p0.x, p0.y), 0.0) * w0.x + textureSampleLevel(cloud_low, samp_low, vec2f(p12.x, p0.y), 0.0) * w12.x + textureSampleLevel(cloud_low, samp_low, vec2f(p3.x, p0.y), 0.0) * w3.x) * w0.y;
  c += (textureSampleLevel(cloud_low, samp_low, vec2f(p0.x, p12.y), 0.0) * w0.x + textureSampleLevel(cloud_low, samp_low, vec2f(p12.x, p12.y), 0.0) * w12.x + textureSampleLevel(cloud_low, samp_low, vec2f(p3.x, p12.y), 0.0) * w3.x) * w12.y;
  c += (textureSampleLevel(cloud_low, samp_low, vec2f(p0.x, p3.y), 0.0) * w0.x + textureSampleLevel(cloud_low, samp_low, vec2f(p12.x, p3.y), 0.0) * w12.x + textureSampleLevel(cloud_low, samp_low, vec2f(p3.x, p3.y), 0.0) * w3.x) * w3.y;
  out.c = vec4f(max(c.rgb, vec3f(0.0)), clamp(c.a, 0.0, 1.0));
  if (out.c.a < 0.002) { discard; }
  // in a cloud's haze, the car's lamps in it before the cloud (the puffs leave them to here: once a pixel, not a puff)
  if (F.cloud.w >= 0.002) {
    let q = uv * 2.0 - 1.0;
    let d = normalize(F.cam_fwd.xyz + F.cam_right.xyz * q.x - F.cam_up.xyz * q.y);
    out.c = vec4f(lamps_over(out.c.rgb / out.c.a, cloud_lamps(d, 1e6)) * out.c.a, out.c.a);
  }
  let at = vec2i(ip); let most = vec2i(size) - 1;
  out.depth = max(max(textureLoad(cloud_near, clamp(at, vec2i(0), most), 0).r, textureLoad(cloud_near, clamp(at + vec2i(1, 0), vec2i(0), most), 0).r),
    max(textureLoad(cloud_near, clamp(at + vec2i(0, 1), vec2i(0), most), 0).r, textureLoad(cloud_near, clamp(at + vec2i(1, 1), vec2i(0), most), 0).r));
  return out;
}
`;

export const SHOWERS = 4; // wasm/include/harstad/shower.h
// A snow shower out over the fjord (wasm/src/shower.c, F.showers): a quad through its axis facing the eye, as wide as
// the column and as tall as its cloud's base. Each pixel takes as much of the curtain as the snow its ray crosses
// (shower_tau, the far lights behind dim by the same), ragged with drifting streaks of heavier fall and slow lumps, and
// is the snow as the moon and the haze light it, warmed a little by the town's glow where that reaches. Nothing is drawn
// for a slot with no shower, nor once the eye is in one: then the falling snow round it is the shower.
export const SHOWER = CORNERS + /* wgsl */`
struct V { @builtin(position) pos: vec4f, @location(0) w: vec3f, @location(1) @interpolate(flat) k: u32 }
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) k: u32) -> V {
  var o: V;
  let a = F.showers[k * 2u]; let b = F.showers[k * 2u + 1u];
  let to = a.xy - F.cam_pos.xz; let dist = max(length(to), 1.0);
  let side = vec2f(-to.y, to.x) / dist;
  let q = CORNER[vi]; // -1..1 across and up
  let x = a.xy + side * q.x * a.z * 2.8;
  o.w = vec3f(x.x, mix(-30.0, b.x, q.y * 0.5 + 0.5), x.y);
  o.k = k;
  o.pos = F.view_proj * vec4f(o.w, 1.0);
  if (a.w <= 0.0 || dist < a.z * 0.6) { o.pos = vec4f(0.0); }
  return o;
}
@fragment fn fs(v: V) -> @location(0) vec4f {
  let a = F.showers[v.k * 2u]; let b = F.showers[v.k * 2u + 1u];
  let to = v.w - F.cam_pos.xyz; let dist = length(to); let d = to / dist;
  var tau = shower_tau(F.cam_pos.xyz, d, 1e6) * smoothstep(0.6, 1.4, length(a.xy - F.cam_pos.xz) / a.z);
  if (tau < 0.004) { return vec4f(0.0); }
  // across the column (m from its axis, as the eye sees it) and up it; the streaks lean downwind as they fall
  let c = a.xy - F.cam_pos.xz; let u = (c.x * d.z - c.y * d.x) / max(length(d.xz), 1e-4);
  let side = normalize(vec2f(-c.y, c.x)); let lean = dot(b.zw, side) * 0.12;
  let t = F.cam_pos.w;
  let fall = noise(vec2f((u + (b.x - v.w.y) * lean) / 240.0 + b.y, (v.w.y + t * 1.1) / 900.0));
  let lump = noise(vec2f(u / 900.0 - b.y, v.w.y / 1300.0 - t * 0.0025));
  tau *= mix(0.7, 1.3, fall) * mix(0.5, 1.4, lump);
  // its foot frays into the sea air, its top into the cloud
  let alpha = 1.0 - exp(-tau);
  let glow = town_glow(a.xy) * 0.6;
  let lit = (moon_now() * 0.4 + sky_up(a.xy) * 0.28) * (1.0 - 0.35 * smoothstep(0.3 * b.x, b.x, v.w.y)) + glow * (1.0 - smoothstep(0.0, b.x, v.w.y)); // greyer up into its cloud
  let col = mix(air_lit(F.cam_pos.xyz + to * 0.5, d), lit, exp(-air_tau(F.cam_pos.xyz, to) - F.fog.a * dist * FAR_AIR)); // seen through the clear air as all else is
  let veil = snow_veil_at(v.w); // and through the snow falling here
  return vec4f(display(mix(col, veil.rgb, veil.a)), alpha * 0.92);
}
`;

// Falling snow. Three shells of flakes, each a box that wraps around the camera: a few big soft ones right in
// front of the lens, the main fall around the car, and a deep haze of fine ones so lamp cones a street away are
// full of snow too. Flakes flutter, drift on gusts, glint as they turn, and smear into streaks along their
// motion past the camera, so driving fast pulls the snowfall into lines. As many fall as the weather has snow falling
// (src/worker.js, snowDraw): the counts here are for a centimetre an hour, and room is kept for SNOW_MOST times that.
// Each flake is lit, never glowing of itself: by the moon, the sky and the town's light in the air as the veil beyond
// them is (snow_lit), by the street lamps under them, the car's headlamps and the traffic's. Above the cloud the snow
// falls from there are none.
export const SNOW_MOST = 4;
export const SNOW_LAYERS = [
  { count: 700, box: [16, 12, 16], size: 2.0, gain: 0.22, soft: 0.0 }, // out of focus, close to the eye
  { count: 15000, box: [64, 38, 64], size: 1, gain: 1, soft: 0.25 },
  { count: 26000, box: [230, 70, 230], size: 1.25, gain: 0.8, soft: 0.3 },
];
export const SNOW = CORNERS + /* wgsl */`
const BOXES = array<vec3f, 3>(${SNOW_LAYERS.map(l => `vec3f(${l.box.map(v => v.toFixed(1)).join(', ')})`).join(', ')});
const LAYERS = array<vec3f, 3>(${SNOW_LAYERS.map(l => `vec3f(${l.size.toFixed(2)}, ${l.gain.toFixed(2)}, ${l.soft.toFixed(2)})`).join(', ')}); // size, gain, softness
const SHUTTER = 0.0125; // s of motion each flake is smeared over
struct V { @builtin(position) pos: vec4f, @location(0) q: vec2f, @location(1) tint: vec3f, @location(2) streak: vec2f, @location(3) @interpolate(flat) rs: vec2f }
@vertex fn vs(@builtin(vertex_index) vi: u32, @location(0) flake: vec4f, @location(1) layer: f32) -> V {
  let time = F.cam_pos.w; let seed = flake.w; let cam = F.cam_pos.xyz;
  let box = BOXES[u32(layer)]; let L = LAYERS[u32(layer)];
  var big = fract(seed * 7.31); // large flakes are rarer, fall faster and flutter less
  big *= big;
  let fall = 0.9 + big * 1.5 + seed * 0.4;
  var p = flake.xyz * box;
  let gust = sin(time * 0.21 + flake.x * 5.0 + flake.z * 3.1) + 0.6 * sin(time * 0.53 + flake.z * 7.3);
  p.y -= time * fall;
  p.x += time * 0.7 + gust * 2.2 + sin(time * (1.1 + seed * 1.7) + seed * 31.0) * (0.55 - big * 0.3);
  p.z += time * 0.3 + gust * 1.1 + cos(time * (0.9 + seed * 1.9) + seed * 17.0) * (0.55 - big * 0.3);
  p.y += sin(time * (1.7 + seed * 2.0) + seed * 53.0) * 0.12;
  let w = p - cam + box * 0.5;
  p = cam + (w - box * floor(w / box)) - box * 0.5;
  { // the jets' wash sweeps the flakes near the car out from under it, and down
    let out = p - (F.car_pos.xyz - F.car_up.xyz * 0.2);
    let d = length(out);
    p += (out / max(d, 0.05) * 1.8 - vec3f(0.0, 0.7, 0.0)) * (F.jet.w * (1.0 - smoothstep(0.8, 5.5, d)));
  }
  let clip = F.view_proj * vec4f(p, 1.0);
  let dist = dot(p - cam, F.cam_fwd.xyz);
  let viewport = vec2f(F.cam_fwd.w, F.cam_up.w);
  var size = viewport.y * 0.5 / F.cam_right.w * (0.035 + big * 0.075) * L.x / dist;
  var dim = 1.0;
  if (size < 1.3) { dim = size / 1.3; size = 1.3; } // too small to draw: fade instead of flicker
  size = min(size, 30.0);
  // smear along the flake's motion relative to the camera over one shutter time
  let ahead = F.view_proj * vec4f(p + (vec3f(0.7, -fall, 0.3) - F.cam_vel.xyz) * SHUTTER, 1.0);
  var trail = vec2f(0.0);
  if (ahead.w > 0.05 && clip.w > 0.0) { trail = (ahead.xy / ahead.w - clip.xy / clip.w) * 0.5 * viewport; }
  let len = min(length(trail), size * 1.6 + 14.0);
  let full = size + len;
  dim *= size / (size + len * 0.75); // the same light spread over a longer mark
  var fade = (1.0 - smoothstep(box.x * 0.34, box.x * 0.5, length(p - cam))) * smoothstep(0.35, 2.2, dist);
  // flakes flare up in the headlight beam and under street lamps
  let car = F.car_pos.xyz + vec3f(0.0, 0.8, 0.0);
  let v = p - car;
  let along = dot(v, F.car_dir.xyz); let perp = length(v - F.car_dir.xyz * along);
  let beam = smoothstep(1.8, 5.0, along) * (1.0 - smoothstep(0.0, 1.0, perp / (along * 0.4 + 0.7))) * (1.0 - smoothstep(18.0, 50.0, along));
  let lamp = lamps_at(p.xz);
  let above = p.y - ground_at(p.xz);
  fade *= step(-0.5, above); // none under the ground: it does not snow in a tunnel
  let lamp_glow = (vec3f(1.0, 0.58, 0.24) * lamp.r + vec3f(0.80, 0.92, 1.0) * lamp.b) * (1.0 - smoothstep(4.5, 9.0, above))
    + low_tint(p.xz) * lamp.g * (1.0 - smoothstep(1.5, 4.0, above));
  let glint = 0.72 + 0.4 * sin(time * (2.0 + seed * 6.0) + seed * 40.0); // faces catching the light as it tumbles
  var o: V;
  // The mark is a capsule along the streak, as long as the square is wide and size across: the square is cut to the
  // rectangle round it, turned along the streak. Inside it the pixels are those the square had.
  var along_q = vec2f(1.0, 0.0);
  if (len > 0.01) { along_q = normalize(trail); }
  let q = along_q * (QUAD[vi].x * 0.5) + vec2f(-along_q.y, along_q.x) * (QUAD[vi].y * 0.5 * size / full);
  o.pos = sprite(clip, full, q * 2.0);
  if (dist <= 0.0) { o.pos = vec4f(2.0, 2.0, 2.0, 1.0); }
  o.q = q;
  o.streak = vec2f(0.0);
  if (len > 0.01) { o.streak = along_q * (len / full) * 0.5; }
  o.rs = vec2f(size / full * 0.5, L.z);
  fade *= 1.0 - smoothstep(F.snowing.w - 60.0, F.snowing.w, p.y); // out of the cloud's foot, and none above it
  let traffic = traffic_lamps(p, normalize(cam - p)) * ${(0.9 / Math.PI).toFixed(4)};
  o.tint = (snow_lit(p.xz) * 0.9 + lamp_glow * 0.95 + vec3f(1.0, 0.88, 0.70) * beam * 0.95 + traffic) * fade * dim * glint * L.y;
  return o;
}
@fragment fn fs(v: V) -> @location(0) vec4f {
  let ss = dot(v.streak, v.streak);
  var along = 0.0;
  if (ss > 1e-6) { along = clamp(dot(v.q, v.streak) / ss, -1.0, 1.0); }
  let d = length(v.q - v.streak * along);
  let a = 1.0 - smoothstep(v.rs.x * v.rs.y, v.rs.x, d);
  return vec4f(display(v.tint * a), a);
}
`;

/* ---- the snow the car drives through (snow.h): the field near the car, the flecks it throws, the ribbon behind ---- */

const SNOW_IN = /* wgsl */`
struct Snow { lod: vec4f, eye: vec4f, pack_car: vec4f }
@group(1) @binding(0) var<uniform> S: Snow;
`;

// The field (wasm/include/harstad/snow.h, snow_view): a mesh a block of 32 by 32 cells, every block anything has
// touched anywhere on the map, the window's round the car and the store's beyond it alike, read from the GPU's slots:
// each block's cells at five levels and what they lie on, and the slots of its eight neighbours beside its own, so
// that where two blocks meet both read the same cells and neither knows which kept them. Heaps and the wind's drifts
// stand up for real, ruts are only shaded in, since what is under them cannot be dug into. Untouched drifts are drawn as
// the terrain draws its snow, from its own photo, its own normals and the same wind relief: the field and the ground
// beyond it are one surface. The drifts are the same everywhere (wasm/include/harstad/dune.h): beyond the field's blocks
// DUNE draws them, from the same function of the place. A block's mesh is 16, 8, 4, 2 or 1 quads a side as its distance allows, the
// geometry dropped under half a pixel (snow_level_reach), each vertex morphing into the next level's by its own distance
// so that blocks of two levels meet without a crack; and each pixel reads the cells at the level whose texel is half a
// pixel, between two levels. Snow fallen since a slot was laid fills its marks (SNOW_FILL) as it does the window's.
const BARE_COLOUR = 'vec3f(0.028, 0.032, 0.04)', BARE_SHOWS = '0.88'; // what a jet lays bare, wet asphalt or dark ice; and how fully it covers what is drawn beneath
// SNOW_SLOT_WORDS, SNOW_SLOT_HEAD and the levels' texels, SNOW_KEEP; SNOW_FILL_DEEP / SNOW_FILL, 1 / SNOW_FILL_BARE
export const SNOW_SLOT_WORDS = 4 + 1364 + 1024, SNOW_MESHES = 5;
const SNOW_KEEP = 0.0005, FILL_RATE = 0.15 / 0.03, FILL_BARE = 1 / 0.01;
// DUNE_LIFT, DUNE_TOP, DUNE_CORNER, DUNE_N, DUNE_ALONG, DUNE_ACROSS, DUNE_HIGH, DUNE_RELIEF (wasm/include/harstad/dune.h)
const DUNE = { lift: '0.02', top: 1700, corner: 1.6, n: 2048, along: 11, across: 36, high: 0.45, relief: 256 };
export const DUNE_N = DUNE.n, DUNE_RELIEF = DUNE.relief;
const TERRAIN_AT = /* wgsl */`
// The terrain's own triangles, as its vertex shader stands them (terrainLit): its grid cell split along the same
// diagonal as the mesh's, its heights and normals from the same texture, so the field lies on it at any distance
struct Post { at: vec2i, layer: i32 }
fn post(i: vec2i) -> Post {
  let c = clamp(i, vec2i(0), vec2i(G.size) - 1) - G.tile.xy;
  let n = vec2i(textureDimensions(height_tiles));
  let t = clamp(vec2i(floor(vec2f(c) / f32(G.tile.z))), vec2i(0), n - 1);
  let l = c - t * G.tile.z;
  var o = Post(l + 1, -1);
  for (var q = 0; q < 4 && o.layer < 0; q++) {
    let d = vec2i(q & 1, q >> 1);
    let u = t - d;
    if (any((d > vec2i(0)) & (l != vec2i(0))) || any(u < vec2i(0))) { continue; }
    let k = textureLoad(height_tiles, u, 0).r;
    if (k >= 0) { o = Post(l + d * G.tile.z + 1, k); }
  }
  return o;
}
fn height_in(p: Post, d: vec2i) -> f32 { return textureLoad(heights, p.at + d, max(p.layer, 0), 0).r; }
fn grid_point(i: vec2i) -> vec4f { // height, then normal
  let p = post(i);
  let dx = height_in(p, vec2i(-1, 0)) - height_in(p, vec2i(1, 0)); let dz = height_in(p, vec2i(0, -1)) - height_in(p, vec2i(0, 1));
  return vec4f(height_in(p, vec2i(0)), normalize(vec3f(dx, 2.0 * G.step, dz)));
}
fn terrain(at: vec2f) -> vec4f {
  let g = (at - G.origin) / G.step; let gi = vec2i(floor(g)); let f = g - floor(g);
  let b = grid_point(gi + vec2i(1, 0)); let c = grid_point(gi + vec2i(0, 1));
  if (f.x + f.y <= 1.0) { let a = grid_point(gi); return a + (b - a) * f.x + (c - a) * f.y; }
  let d = grid_point(gi + vec2i(1, 1));
  return d + (c - d) * (1.0 - f.x) + (b - d) * (1.0 - f.y);
}
`;
export const PACK = SNOW_IN + GRID + /* wgsl */`
@group(1) @binding(1) var<storage, read> slots: array<u32>;
@group(1) @binding(2) var<storage, read> blocks: array<u32>; // struct snow_inst: bi, bj, level, the nine slots
@group(1) @binding(3) var height_tiles: texture_2d<i32>;     // the near terrain's, as its own shader reads them
@group(1) @binding(4) var heights: texture_2d_array<f32>;
@group(1) @binding(5) var<uniform> G: Grid;
@group(1) @binding(6) var photos: texture_2d_array<f32>;
@group(1) @binding(7) var samp_photo: sampler;
@group(1) @binding(8) var photo_pages: texture_2d<f32>;
@group(1) @binding(9) var photos1: texture_2d_array<f32>;
@group(1) @binding(10) var photos2: texture_2d_array<f32>;
@group(1) @binding(11) var photo_coarse: texture_2d<f32>;
${GROUND_SNOW}
const NONE = 0xffffffffu;
const SLOT_WORDS = ${SNOW_SLOT_WORDS}u;
struct V { @builtin(position) pos: vec4f, @location(0) w: vec3f, @location(1) n: vec3f, @location(2) drift: f32,
  @location(3) @interpolate(flat) b: u32 }
// the slot of block b's neighbourhood (3 x 3 blocks) holding texel t of a level n texels a side, and t within it
fn slot_of(b: u32, t: vec2i, n: i32) -> vec3i {
  let q = select(select(vec2i(1), vec2i(2), t >= vec2i(n)), vec2i(0), t < vec2i(0));
  return vec3i(t - (q - 1) * n, bitcast<i32>(blocks[b * 12u + 3u + u32(q.y * 3 + q.x)]));
}
// a texel of a level: height (m) and bareness (0..1), the snow fallen since its slot was laid filling it
fn texel(b: u32, level: u32, t: vec2i) -> vec2f {
  let n = i32(32u >> level);
  let at = slot_of(b, t, n);
  if (u32(at.z) == NONE) { return vec2f(0.0); }
  let base = u32(at.z) * SLOT_WORDS;
  let v = slots[base + 4u + (1024u - (1024u >> (2u * level))) * 4u / 3u + u32(at.y * n + at.x)];
  let fallen = max(F.grit.z - bitcast<f32>(slots[base]), 0.0);
  let h = f32(i32(v & 0xffffu) - 32768) * ${SNOW_KEEP};
  return vec2f(sign(h) * max(abs(h) - fallen * ${FILL_RATE.toFixed(4)}, 0.0), max(f32(v >> 16u) / 65535.0 - fallen * ${FILL_BARE.toFixed(4)}, 0.0));
}
// what cell c lies on (m): its lift and its drift
fn bed(b: u32, c: vec2i) -> vec2f {
  let at = slot_of(b, c, 32);
  if (u32(at.z) == NONE) { return vec2f(0.0); }
  let v = slots[u32(at.z) * SLOT_WORDS + ${4 + 1364}u + u32(at.y * 32 + at.x)];
  return vec2f(f32(v & 0xffffu), f32(v >> 16u)) * 0.001;
}
// the cells at a level at p (m from the block's corner), between the four texels round it
fn field(b: u32, p: vec2f, level: u32) -> vec2f {
  let u = p / (0.1 * f32(1u << level)) - 0.5; let t = vec2i(floor(u)); let f = u - floor(u);
  return mix(mix(texel(b, level, t), texel(b, level, t + vec2i(1, 0)), f.x), mix(texel(b, level, t + vec2i(0, 1)), texel(b, level, t + vec2i(1, 1)), f.x), f.y);
}
// how its height slopes at a level, from a texel either side: read only where the field is drawn (its height and
// bareness, field's, say whether it is)
fn slope(b: u32, p: vec2f, level: u32) -> vec2f {
  let s = 0.1 * f32(1u << level);
  let dx = field(b, p - vec2f(s, 0.0), level).x - field(b, p + vec2f(s, 0.0), level).x;
  let dz = field(b, p - vec2f(0.0, s), level).x - field(b, p + vec2f(0.0, s), level).x;
  return vec2f(dx, dz) / (2.0 * s);
}
${TERRAIN_AT}@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) b: u32) -> V {
  let bk = vec2i(bitcast<i32>(blocks[b * 12u]), bitcast<i32>(blocks[b * 12u + 1u])); let level = blocks[b * 12u + 2u];
  let n = 16u >> level; let g = vec2u(vi % (n + 1u), vi / (n + 1u));
  let corner = vec2f(bk) * 3.2; let step = 0.2 * f32(1u << level);
  var p = vec2f(g) * step; // from the block's corner
  if (level < 4u) { // morphing into the next level as it nears where that begins: its odd vertices onto the even ones before them
    let r1 = S.lod[level]; let r0 = select(0.0, S.lod[max(level, 1u) - 1u], level > 0u);
    let d = length(vec3f(corner + p - S.eye.xy, max(F.cam_pos.y - ${DUNE.top.toFixed(1)}, 0.0))); // with the least the eye is above any snow, as snow_view (dune_lift)
    let m = clamp((d - (r1 + r0) * 0.5) / ((r1 - r0) * 0.5), 0.0, 1.0);
    let odd = vec2f((bk * i32(n) + vec2i(g)) & vec2i(1));
    p -= odd * step * m;
  }
  let at = corner + p;
  var o: V;
  o.b = b;
  let heap = field(b, p, 0u).x;
  // what it lies on: the highest of the four cells round it, so a roof keeps its edge; and their drifts, between them
  let u = p / 0.1 - 0.5; let c = vec2i(floor(u)); let f = u - floor(u);
  let b0 = bed(b, c); let b1 = bed(b, c + vec2i(1, 0)); let b2 = bed(b, c + vec2i(0, 1)); let b3 = bed(b, c + vec2i(1, 1));
  let lift = max(max(b0.x, b1.x), max(b2.x, b3.x)); let drift = mix(mix(b0.y, b1.y, f.x), mix(b2.y, b3.y, f.x), f.y);
  o.drift = drift;
  let t = terrain(at);
  o.w = vec3f(at.x, t.x + lift + ${DUNE.lift} + max(drift + heap, 0.0), at.y);
  o.n = t.yzw;
  o.pos = F.view_proj * vec4f(o.w, 1.0);
  return o;
}
@fragment fn fs(v: V) -> @location(0) vec4f {
  let dx = dpdx(v.w.xz); let dy = dpdy(v.w.xz);
  let p = v.w.xz - vec2f(vec2i(bitcast<i32>(blocks[v.b * 12u]), bitcast<i32>(blocks[v.b * 12u + 1u]))) * 3.2;
  // the level whose texel is half a pixel: full detail while a cell is more, between two levels after
  let lv = clamp(log2(max(max(length(dx), length(dy)) / 0.1, 1e-6)) - 1.0, 0.0, 4.0);
  let l0 = u32(lv); let k = lv - f32(l0);
  var q = vec4f(field(v.b, p, l0), 0.0, 0.0);
  if (k > 0.004) { q = vec4f(mix(q.xy, field(v.b, p, min(l0 + 1u, 4u)), k), 0.0, 0.0); }
  // its height; and how bare a jet has blown it (snow_wash): there what the snow lay on shows, level, wet and dark
  let deep = q.x; let bare = q.y * ${BARE_SHOWS};
  let touched = max(smoothstep(0.004, 0.022, abs(deep)) * (1.0 - smoothstep(0.0, 0.004, q.y) * step(0.0, deep)), bare); // (ice the field was lifted to shows only where swept; a dish shows all the way down)
  // a drift stands up out of the ground, so is drawn whether touched or not; where untouched, as the terrain would draw it
  let drift = smoothstep(0.0, 0.01, v.drift);
  let alpha = max(touched, drift);
  if (alpha < 0.01) { return vec4f(0.0); } // blended, so nothing laid is the same as dropped: and none of the rest is worked out
  let as_ground = drift * (1.0 - touched) / alpha; // of what is drawn, how much is the ground's own snow, as the terrain shows it
  let tn = normalize(v.n);
  var albedo: vec3f; var n = tn; var cover: f32;
  if (as_ground > 0.999) { // untouched: the terrain's own look, and nothing of the field's need be read
    let ground = ground_snow(v.w, dx, dy);
    albedo = ground.rgb; cover = ground.a;
  } else {
    q = vec4f(q.xy, slope(v.b, p, l0));
    if (k > 0.004) { q = vec4f(q.xy, mix(q.zw, slope(v.b, p, min(l0 + 1u, 4u)), k)); }
    // 0.55 of the field's true slope, on the terrain's: rut walls are near vertical, and would go black
    let field = normalize(tn / tn.y + vec3f(q.z, 0.0, q.w) * ((1.0 - q.y) * 0.55));
    let grain = textureSampleLevel(detail_map, samp_repeat, v.w.xz * 0.37, 0, 0.0).r * 0.55 + textureSampleLevel(detail_map, samp_repeat, v.w.xz * 0.051, 0, 0.0).r * 0.45;
    // thrown snow is clean and bright; pressed snow lies in its own shadow, darker the deeper it is
    let pressed = mix(vec3f(0.62, 0.67, 0.78), vec3f(0.40, 0.45, 0.57), smoothstep(0.02, 0.12, -deep));
    let laid = mix(mix(pressed, vec3f(0.84, 0.89, 0.97), smoothstep(-0.01, 0.015, deep)) * (0.86 + 0.28 * grain), ${BARE_COLOUR}, q.y);
    let judged = smoothstep(0.42, 0.6, min(laid.r, min(laid.g, laid.b))) * smoothstep(-0.03, 0.02, laid.b - laid.r); // (lit_reached's, for a cover of -1)
    albedo = laid; n = field; cover = judged; // what was laid, alone
    if (as_ground >= 0.001) { // and partly the ground's own snow
      let ground = ground_snow(v.w, dx, dy);
      albedo = mix(laid, ground.rgb, as_ground); n = normalize(mix(field, tn, as_ground)); cover = mix(judged, ground.a, as_ground);
    }
  }
  return vec4f(finish(lit_snow(albedo, n, v.w, 1.0, cover, 1.0, dx, dy), v.w), alpha);
}
`;

// The wind's drifts beyond the field's blocks (wasm/include/harstad/dune.h, dune_view): nodes of 16 x 16 quads whose
// lattice is 0.2 * 2^L m, each vertex snapped to the lattice its own distance allows and morphing onto the next as it
// nears where that begins, so nodes of any lattices meet without a crack, and meet the field's blocks; stood on the
// terrain's own triangles by the drifts' height there, as dune_height has it: how freely the wind lays them, from the
// corners round the eye (dune_open, DUNE.corner m apart, wrapping), times the tile of their shapes (dune_relief, a byte
// each). Let down to nothing over the last quarter of the reach where all their height is half a pixel. Drawn as the
// terrain draws its snow, and blended in by their height, so where none lies nothing is drawn over the terrain.
export const DUNE_SHADER = GRID + /* wgsl */`
@group(1) @binding(0) var<storage, read> dune: array<u32>;   // struct dune_frame (12 words), then struct dune_node (4 each)
@group(1) @binding(1) var dune_open: texture_2d<f32>;
@group(1) @binding(2) var dune_relief: texture_2d<f32>;
@group(1) @binding(3) var height_tiles: texture_2d<i32>;
@group(1) @binding(4) var heights: texture_2d_array<f32>;
@group(1) @binding(5) var<uniform> G: Grid;
@group(1) @binding(6) var photos: texture_2d_array<f32>;
@group(1) @binding(7) var samp_photo: sampler;
@group(1) @binding(8) var photo_pages: texture_2d<f32>;
@group(1) @binding(9) var photos1: texture_2d_array<f32>;
@group(1) @binding(10) var photos2: texture_2d_array<f32>;
@group(1) @binding(11) var photo_coarse: texture_2d<f32>;
${GROUND_SNOW}
${TERRAIN_AT}
struct V { @builtin(position) pos: vec4f, @location(0) w: vec3f, @location(1) n: vec3f, @location(2) drift: f32, @location(3) off: f32 }
fn dune_head(k: u32) -> f32 { return bitcast<f32>(dune[k]); }
fn dune_reach(l: u32) -> f32 { return select(dune_head(1u + l), 0.0, l == 0u); } // where lattice l may first be drawn
// how freely the wind lays drifts at xz, 0 .. 1, between the four corners round it (dune.c, dune_height)
fn dune_free(xz: vec2f) -> f32 {
  let u = xz / ${DUNE.corner}; let c = vec2i(floor(u)); let f = u - floor(u); let m = vec2i(${DUNE.n - 1});
  let a = textureLoad(dune_open, c & m, 0).r; let b = textureLoad(dune_open, (c + vec2i(1, 0)) & m, 0).r;
  let e = textureLoad(dune_open, (c + vec2i(0, 1)) & m, 0).r; let g = textureLoad(dune_open, (c + vec2i(1, 1)) & m, 0).r;
  let lo = a + (b - a) * f.x;
  return lo + ((e + (g - e) * f.x) - lo) * f.y;
}
// the tile at xz, -0.5 .. 0.5, along the wind and across it, between its texels' middles (dune.c, dune__tile)
fn dune_tile(xz: vec2f) -> f32 {
  let u = vec2f((xz.x * 0.71 - xz.y * 0.71) / ${DUNE.along}.0, (xz.x * 0.71 + xz.y * 0.71) / ${DUNE.across}.0) * ${DUNE.relief}.0 - 0.5;
  let p = vec2i(floor(u)); let f = u - floor(u); let m = vec2i(${DUNE.relief - 1});
  let a = textureLoad(dune_relief, p & m, 0).r; let b = textureLoad(dune_relief, (p + vec2i(1, 0)) & m, 0).r;
  let e = textureLoad(dune_relief, (p + vec2i(0, 1)) & m, 0).r; let g = textureLoad(dune_relief, (p + vec2i(1, 1)) & m, 0).r;
  let lo = a + (b - a) * f.x;
  return lo + ((e + (g - e) * f.x) - lo) * f.y - 0.5;
}
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) b: u32) -> V {
  let at = 12u + b * 4u;
  let level = dune[at + 2u];
  let k = vec2i(bitcast<i32>(dune[at]), bitcast<i32>(dune[at + 1u])) + vec2i(i32(vi % 17u), i32(vi / 17u)) * (1 << level); // in 0.2 m
  let eye = vec2f(dune_head(0u), dune_head(1u));
  let d = length(vec3f(vec2f(k) * 0.2 - eye, max(F.cam_pos.y - ${DUNE.top.toFixed(1)}, 0.0))); // with the least the eye is above any drift, as dune_view (dune_lift)
  // the lattice its distance allows, and how far onto the next it has gone over the second half of its reach
  var l = 0u;
  for (var q = 1u; q < 5u; q++) { if (d >= dune_reach(q)) { l = q; } }
  var m = 0.0;
  if (l < 4u) { let r0 = dune_reach(l); let r1 = dune_reach(l + 1u); m = clamp((d - (r1 + r0) * 0.5) / ((r1 - r0) * 0.5), 0.0, 1.0); }
  let p = mix(vec2f(k & vec2i(-(1 << l))), vec2f(k & vec2i(-(2 << l))), m) * 0.2;
  let gone = dune_head(6u);
  let h = dune_free(p) * ${DUNE.high} * (dune_tile(p) + 0.5) * (1.0 - smoothstep(gone * 0.75, gone, d));
  let t = terrain(p);
  var o: V;
  o.w = vec3f(p.x, t.x + ${DUNE.lift} + h, p.y);
  o.n = t.yzw;
  o.drift = h;
  o.pos = F.view_proj * vec4f(o.w, 1.0);
  // off the terrain the GPU holds: where it stands, at sea level, and every pixel of a triangle with a corner there left
  // out. (Thrown out of the view, its triangles with the corners still on the terrain were drawn across the screen.)
  o.off = 0.0;
  if (post(vec2i(floor((p - G.origin) / G.step))).layer < 0) { o.w.y = 0.0; o.pos = F.view_proj * vec4f(o.w, 1.0); o.drift = 0.0; o.off = 1.0; }
  return o;
}
@fragment fn fs(v: V) -> @location(0) vec4f {
  let dx = dpdx(v.w.xz); let dy = dpdy(v.w.xz);
${OFF.has('dunefs') ? '  if (true) { return vec4f(0.0); }' : ''}
  // drawn solid, before the ground, so that the ground under a drift is never shaded: a drift shows the ground's own
  // look (ground_snow), so where it is under half a centimetre deep it is left out and the ground shows instead
  if (v.drift < 0.005 || v.off > 0.0) { discard; }
  let ground = ground_snow(v.w, dx, dy);
  return vec4f(finish(lit_snow(ground.rgb, normalize(v.n), v.w, 1.0, ground.a, 1.0, dx, dy), v.w), 1.0);
}
`;

// The drifts into the far ground's buffer (src/worker.js, defer), all of them, near and far: a drift shows the ground's
// own look (ground_snow, lit_snow on the ground's normal), so the buffer's lighting lights it as the ground, once a
// pixel, with every light (bit 29: terrainDeferred near); its lattice is fine (0.2 m), and lit in the frame itself each
// of its triangles was shaded in quads of its own on every one of the 4x MSAA's samples it holds (from 150 m up over
// the town some 10 ms of 85). The frame's own drifts then take their pixel's light (DUNE_READ).
export const DUNE_GBUF = (() => { const c = DUNE_SHADER; const a = c.indexOf('@fragment fn fs(v: V)');
  if (a < 0) { throw new Error('dune gbuf'); }
  return GBUF_NORMAL + c.slice(0, a) + `struct GOut { @location(0) a: vec4u, @location(1) b: vec4f }
@fragment fn fs(v: V) -> GOut {
  let dx = dpdx(v.w.xz); let dy = dpdy(v.w.xz);
  if (v.drift < 0.005 || v.off > 0.0) { discard; return GOut(); }
  return GOut(vec4u(bitcast<vec3u>(v.w), ${gbufWord('normalize(v.n)', 0, 1)}), vec4f(dx, dy));
}
`; })();

// The drifts where the far ground is lit once a pixel (terrainDeferred): a drift shows the ground's own look, so it
// takes that pixel's light, and lights itself only where the far ground is not under it
const DUNE_LIT = '  let ground = ground_snow(v.w, dx, dy);\n';
export const DUNE_READ = (() => { if (!DUNE_SHADER.includes(DUNE_LIT)) { throw new Error('dune read'); }
  return DUNE_SHADER.replace(DUNE_LIT, '  let shaded = textureLoad(terrain_shaded, vec2i(v.pos.xy), 0);\n  if (shaded.a > 0.75) { return shaded; }\n' + DUNE_LIT)
    .replace('@vertex fn vs(', '@group(2) @binding(0) var terrain_shaded: texture_2d<f32>;\n@vertex fn vs('); })();

// flecks thrown up by the tyres, each one a point that flies its own arc
export const SPRAY = CORNERS + SNOW_IN + /* wgsl */`
struct V { @builtin(position) pos: vec4f, @location(0) uv: vec2f, @location(1) tint: vec3f }
@vertex fn vs(@builtin(vertex_index) vi: u32, @location(0) origin: vec3f, @location(1) flung: vec4f) -> V {
  let time = S.pack_car.z;
  let age = time - flung.w; let kind = fract(flung.w * 61.7 + origin.x * 7.3);
  let life = 1.1 * (0.45 + 0.55 * kind); let t = age / life;
  // powder: the air stops it within a metre or two, then it sinks back
  var p = origin + flung.xyz * (1.0 - exp(-3.0 * age)) / 3.0;
  p.y = max(p.y - 2.2 * age * age, origin.y - 0.12);
  let clip = F.view_proj * vec4f(p, 1.0);
  let dist = dot(p - F.cam_pos.xyz, F.cam_fwd.xyz);
  // lit like the falling snow: by the sky, the street lamps and the headlight beam
  let v = p - (F.car_pos.xyz + vec3f(0.0, 0.8, 0.0));
  let along = dot(v, F.car_dir.xyz); let perp = length(v - F.car_dir.xyz * along);
  let beam = smoothstep(1.8, 5.0, along) * (1.0 - smoothstep(0.0, 1.0, perp / (along * 0.4 + 0.7)));
  let lamp = lamps_at(p.xz);
  var o: V;
  o.pos = sprite(clip, clamp(F.cam_up.w * 0.5 / F.cam_right.w * (0.04 + 0.16 * t) * (0.6 + 0.8 * kind) / dist, 1.0, 28.0), QUAD[vi]);
  if (t <= 0.0 || t >= 1.0 || dist <= 0.0) { o.pos = vec4f(2.0, 2.0, 2.0, 1.0); }
  o.uv = QUAD[vi] * 0.5;
  o.tint = (vec3f(0.50, 0.60, 0.82) * 0.2 + (vec3f(1.0, 0.58, 0.24) * (lamp.r + lamp.g) + vec3f(0.80, 0.92, 1.0) * lamp.b) * 0.6 + vec3f(1.0, 0.93, 0.80) * beam)
    * (1.0 - t) * (1.0 - t) * smoothstep(0.0, 0.08, t) * 0.5;
  return o;
}
@fragment fn fs(v: V) -> @location(0) vec4f {
  let a = 1.0 - smoothstep(0.05, 0.5, length(v.uv));
  return vec4f(display(v.tint * a), a);
}
`;

// ---- the jets of the engines under the wings (wasm/src/rocket.c): a flame from each nozzle, and the snow their wash blows off the ground ----
const SHED = { clump: 512, g: 9.81 }; // SHED_CLUMP and SHED_G (wasm/include/harstad/shed.h)
// What shows of a jet is its hot section, glowing up the nozzle (rocket.c), and the gas leaving it. It is raymarched
// inside a box round it as an amber haze drawn out into threads along its flow, fine across it and long along it,
// carried down it as it goes: warmest leaving the nozzle and lit there by the hot section's glow, as bright there as a
// lit lens a hand across and dimmer than the glowing inside it comes from; it cools fast and has all but gone within a
// metre or so, more when the jet is pushed hard. It hides nothing behind it. Only at the
// most a jet gives does it leave the nozzle under-expanded, and then the faintest shock diamonds stand in it, spaced as
// the root of the pressure ratio over one has them (Prandtl's). The cross-wind of the car's motion bends it
// (rocket_jet.bend); where it comes down on the ground it spreads out flat over it, as an impinging jet does.
// Its light on the scene is real light, its own: nothing is drawn round it.
// JET: the haze's spread (tan of its half-angle, at full thrust); the shock cells' spacing in lip radii at none and full,
// the thrust they show from and are whole at, and how many show at most; the gas's emission (radiance a metre) in the
// haze, a diamond and where it spreads over the ground; how fast its threads are carried down it, m/s; the march's step,
// m, and the most steps a pixel takes; the smallest a flame is drawn, in pixels across its lip; under how many pixels long
// a flame is marched in the frame itself, and over it in a picture this many times smaller each way (JETS_LOW)
const JET = { spread: 0.2, cell: [1.6, 3.6], shock: [0.97, 1.12], cells: 4, flame: 4.0, diamond: 6.0, splash: 2.4,
  flow: 9.0, step: 0.05, steps: 18, px: 1.5, direct: 96, low: 4,
  clear: 0.94, tint: 0.035 };
export const JET_SPREAD = JET.spread; // for the glue's bound on the shimmer (src/worker.js, shimmer)
// Boosted (rocket_jet.boost), the burning is all inside the nozzle, which glows ultra hot (the car shader's jet_heat):
// what leaves it is clear, hot air, seen only as the heat shimmer it makes of what is behind it (JETS_HAZE). Of the
// flame outside, JET.clear of it is gone at full boost, and what is left is the faintest tint, JET.tint, blue near the
// nozzle going to yellow further out. The turbine's flame (rocket_jet.core) only burns boosted.
// what of a plough's curl billows, and how much longer, larger and brighter its grains and billows are than the jets' wash's
const PLOUGH_CURL = { billows: 0.55, life: 1.4, size: 1.8, bright: 1.2 };
const PUFF = { life: 2.6, mist: 0.09, grains: 0.2, drag: 2.4, billows: 0.3, roof: 2.2, flake: 0.7, clump: 1.6 }; // s the longest grain lasts (DRIFT_LIFE in the C); brightness of the billows and of the grains (a roof's flake, a tree's clump); 1/s the air stops them; the share that are billows
const f3 = v => v.toFixed(3);
// one instance a nozzle (struct rocket_jet): where its lip is and the flame's length; which way it points and how hard
// it burns; how the wind bends it and the height of what it comes down on; the lip's radius, its heat, whether it comes
// down on water, its light. A flame big on the screen is marched at a quarter of the frame's width and height (JETS_LOW,
// each into a layer of its own), then laid into the frame through its box's faces, so that the scene still hides it at
// the frame's own sharpness; one small on the screen is marched in the frame itself (JETS), where it is cheap.
const JET_BOX = /* wgsl */`
struct V {
  @builtin(position) pos: vec4f, @location(0) w: vec3f,
  @location(1) @interpolate(flat) at: vec4f, @location(2) @interpolate(flat) dir: vec4f, @location(3) @interpolate(flat) bend: vec4f,
  @location(4) @interpolate(flat) look: vec4f, // lip radius as drawn, how much of its light is kept, inside its box (1) or not, which jet
  @location(5) @interpolate(flat) lo: vec3f, @location(6) @interpolate(flat) hi: vec3f, @location(7) @interpolate(flat) wet: f32,
  @location(8) @interpolate(flat) direct: f32, // 1: marched in the frame itself, small on the screen as it is, or the turbine's
  @location(9) @interpolate(flat) more: vec4f, // how far it burns boosted, 0..1; 1 for the boost turbine's flame
}
// the box's twelve triangles, as corners numbered by their x, y, z bits
const CUBE = array<u32, 36>(0u, 2u, 6u, 0u, 6u, 4u, 1u, 5u, 7u, 1u, 7u, 3u, 0u, 4u, 5u, 0u, 5u, 1u, 2u, 3u, 7u, 2u, 7u, 6u, 0u, 1u, 3u, 0u, 3u, 2u, 4u, 6u, 7u, 4u, 7u, 5u);
const BEND_S = 0.6; // ROCKET_BEND_S
// the axis's way off the straight line, s along it: the cross-wind takes the gas as it slows
fn jet_bent(bend: vec3f, s: f32) -> vec3f { let k = max(s, 0.0); return bend * (k * k / (k + BEND_S)); }
// how far the edge of the burning gas stands from the axis, s along it
fn jet_edge(lip: f32, s: f32, power: f32) -> f32 { return lip + max(s, 0.0) * ${f3(JET.spread)} * (0.55 + 0.45 * min(power, 1.0)); }
// how far along its axis a jet meets the floor under it, if it points down at all
fn jet_far(at: vec3f, dir: vec3f, floor: f32) -> f32 { return select(1e4, (at.y - floor) / max(-dir.y, 1e-3), dir.y < -0.05 && floor > -1e3); }
// the layer of JETS_LOW's small pictures nozzle ii's flame is marched into: the front lift jets' pair, the rear pair,
// each cruise turbine's (wasm/include/harstad/rocket.h, rocket_nozzle_jet and rocket_wing)
fn jet_layer(ii: u32) -> u32 { return select(ii - 2u, ii >> 1u, ii < 4u); }
fn jet_box(vi: u32, ii: u32, at: vec4f, dir: vec4f, bend: vec4f, look: vec4f, more: vec4f) -> V {
  var o: V;
  let flick = jet_flicker(ii);
  let len = at.w * (1.0 + 1.5 * (flick - 1.0));
  // too small on screen to be resolved, the flame is drawn two pixels across and keeps its light: as wide again, a
  // quarter as bright, and so on
  let px = pixel_angle() * distance(F.cam_pos.xyz, at.xyz);
  let widen = max(1.0, ${f3(JET.px)} * px / look.x);
  let lip = look.x * widen;
  // the box round it: round the points of its axis as wide as it is there, and round the ring it spreads in on the ground
  var lo = vec3f(1e9); var hi = vec3f(-1e9);
  for (var k = 0; k < 4; k++) {
    let s = len * f32(k) / 3.0;
    let c = at.xyz + dir.xyz * s + jet_bent(bend.xyz, s); let r = jet_edge(lip, s, dir.w) * 1.35 + 0.02;
    lo = min(lo, c - r); hi = max(hi, c + r);
  }
  let far = jet_far(at.xyz, dir.xyz, bend.w);
  if (far < len + lip) {
    let c = at.xyz + dir.xyz * far + jet_bent(bend.xyz, far); let r = (jet_edge(lip, far, dir.w) * 2.2 + 0.3 * min(dir.w, 1.15)) * 1.5;
    lo = min(lo, vec3f(c.x - r, bend.w, c.z - r)); hi = max(hi, vec3f(c.x + r, bend.w + 0.7, c.z + r));
    lo.y = max(lo.y, bend.w - 0.01);
  }
  let corner = CUBE[vi];
  let w = select(lo, hi, vec3<bool>((corner & 1u) != 0u, (corner & 2u) != 0u, (corner & 4u) != 0u));
  let e = F.cam_pos.xyz;
  let inside = all(e > lo - 0.01) && all(e < hi + 0.01);
  o.pos = F.view_proj * vec4f(w, 1.0);
  if (dir.w <= 0.004) { o.pos = vec4f(0.0); }
  o.w = w; o.at = vec4f(at.xyz, len); o.dir = dir; o.bend = bend; o.lo = lo; o.hi = hi;
  o.look = vec4f(lip, flick / (widen * widen), select(0.0, 1.0, inside), f32(ii)); o.wet = look.z;
  o.more = more;
  // the turbine's flame is marched in the frame itself, near or far: the small pictures have a layer only for each wing's
  o.direct = select(0.0, 1.0, (len + lip) / px < ${JET.direct.toFixed(1)} || more.y > 0.5);
  return o;
}
// Value noise in three dimensions, 0..1, from the frame's 256-texel white noise (textures.js, noiseTexture: 96..160
// out of 255, and now and then a spike) read twice: each whole step in z moves the lattice in the texture by (37, 17)
// texels, so two filtered reads, one slice apart, give the eight corners of a cell. Cheap enough to take twice a step.
fn jet_noise(q: vec3f) -> f32 {
  let i = floor(q); var f = fract(q); f = f * f * (3.0 - 2.0 * f);
  let uv = (i.xy + vec2f(37.0, 17.0) * i.z + f.xy + 0.5) * (1.0 / 256.0);
  let a = textureSampleLevel(detail_map, samp_repeat, uv, 0, 0.0).r;
  let b = textureSampleLevel(detail_map, samp_repeat, uv + vec2f(37.0, 17.0) * (1.0 / 256.0), 0, 0.0).r;
  return clamp((mix(a, b, f.z) - 0.3765) * 3.98, 0.0, 1.0);
}
// burning gas by how hot it is, 0..1: dull red where it has all but cooled, through orange and yellow to white
fn jet_colour(h: f32) -> vec3f {
  let red = vec3f(0.9, 0.08, 0.012); let orange = vec3f(1.0, 0.34, 0.045); let yellow = vec3f(1.0, 0.72, 0.28); let white = vec3f(1.0, 0.93, 0.82);
  return mix(mix(mix(red, orange, smoothstep(0.1, 0.35, h)), yellow, smoothstep(0.35, 0.65, h)), white, smoothstep(0.65, 0.95, h));
}
// the light the flame gives off along the ray through this point of its box, and how much of what is behind it its soot hides
fn jet_march(v: V) -> vec4f {
  let e = F.cam_pos.xyz; let rd = normalize(v.w - e);
  let inv = 1.0 / select(rd, vec3f(1e-6), abs(rd) < vec3f(1e-6));
  let ta3 = (v.lo - e) * inv; let tb3 = (v.hi - e) * inv;
  let near3 = min(ta3, tb3); let far3 = max(ta3, tb3);
  var t0 = max(max(max(near3.x, near3.y), near3.z), F.misc.x);
  var t1 = min(min(far3.x, far3.y), far3.z);
  let floor_y = v.bend.w;
  if (rd.y < 0.0) { t1 = min(t1, (floor_y - e.y) / rd.y); } else if (e.y < floor_y) { return vec4f(0.0); } // nothing burns under the ground
  if (t1 <= t0) { return vec4f(0.0); }
  let at = v.at.xyz; let len = v.at.w; let dir = v.dir.xyz; let power = v.dir.w; let hard = min(power, 1.15);
  let lip = v.look.x; let time = F.cam_pos.w; let seed = v.look.w * 17.3;
  // boosted, the burning is inside the nozzle: what leaves it is all but clear
  let boost = v.more.x; let flame = 1.0 - ${f3(JET.clear)} * boost;
  // the core: how far it runs before the mixing layer has eaten into its axis; the shock cells in it
  let cell = lip * mix(${f3(JET.cell[0])}, ${f3(JET.cell[1])}, min(hard, 1.0));
  let shocks = smoothstep(${f3(JET.shock[0])}, ${f3(JET.shock[1])}, power);
  let far = jet_far(at, dir, floor_y);
  let hit = at + dir * far + jet_bent(v.bend.xyz, far);
  // how hot the gas still is on the axis where it comes down, and how wide it is there
  let impact = select(0.0, pow(max(1.0 - far / len, 0.0), 1.2), far < len);
  let spread = jet_edge(lip, far, power) * 2.2 + 0.3 * hard;
  // March only where there is gas: along this ray's way through a cone round the plume's chord, as wide as the plume
  // and as far again as its bent axis strays from the chord; and through the low drum its flame spreads in over the
  // ground. The cone's chord is found from where the ray passes the axis nearest, as wide as the cone is there and
  // then as wide as it is where that chord ends.
  let tip = dir * len + jet_bent(v.bend.xyz, len); let reach = length(tip); let ax = tip / reach;
  let stray = length(jet_bent(v.bend.xyz, len * 0.5) - jet_bent(v.bend.xyz, len) * 0.5);
  var lo_t = 1e9; var hi_t = -1e9;
  {
    let w = e - at; let ca = dot(rd, ax); let sn2 = max(1.0 - ca * ca, 1e-4);
    let wa = dot(w, ax); let tc = (ca * wa - dot(w, rd)) / sn2; let sc = wa + ca * tc;
    let dc = length(w + rd * tc - ax * sc);
    let r0 = jet_edge(lip, clamp(sc, 0.0, reach), power) * 1.35 + stray;
    let r1 = jet_edge(lip, clamp(sc + sqrt(max(r0 * r0 - dc * dc, 0.0) / sn2) * abs(ca), 0.0, reach), power) * 1.35 + stray;
    if (dc < r1) { let h = sqrt((r1 * r1 - dc * dc) / sn2); lo_t = tc - h; hi_t = tc + h; }
  }
  if (impact > 0.0) {
    let o = e.xz - hit.xz; let dd = dot(rd.xz, rd.xz); let b = dot(o, rd.xz); let rs = spread * 1.45;
    let disc = b * b - dd * (dot(o, o) - rs * rs);
    if (disc > 0.0 && dd > 1e-8) {
      var a0 = (-b - sqrt(disc)) / dd; var a1 = (-b + sqrt(disc)) / dd;
      if (abs(rd.y) > 1e-5) { let y0 = (floor_y - e.y) / rd.y; let y1 = (floor_y + 0.6 - e.y) / rd.y; a0 = max(a0, min(y0, y1)); a1 = min(a1, max(y0, y1)); }
      if (a1 > a0) { lo_t = min(lo_t, a0); hi_t = max(hi_t, a1); }
    }
  }
  t0 = max(t0, lo_t); t1 = min(t1, hi_t);
  if (t1 <= t0) { return vec4f(0.0); }
  // the march: as many steps as the way through needs, from a place in each pixel's step that differs pixel to pixel
  let n = clamp(ceil((t1 - t0) / ${f3(JET.step)}), 3.0, ${JET.steps.toFixed(1)});
  let dt = (t1 - t0) / n;
  var t = t0 + dt * fract(52.9829189 * fract(dot(v.pos.xy, vec2f(0.06711056, 0.00583715))));
  var glow = vec3f(0.0);
  for (var k = 0.0; k < n; k += 1.0) {
    let p = e + rd * t; t += dt;
    let d = p - at; let s = dot(d, dir);
    if (s < -0.01) { continue; }
    let off = d - dir * s - jet_bent(v.bend.xyz, s);
    let r = length(off); let u = s / len;
    let edge = jet_edge(lip, s, power);
    var em = vec3f(0.0);
    if (u < 1.0 && r < edge * 1.35) {
      // the gas drawn out into threads along its flow: fine across it, long along it, carried down it as it goes
      let q = off / (edge * 0.3) + dir * ((s - time * ${f3(JET.flow)}) / (edge * 3.5) + seed);
      let threads = jet_noise(q) * (0.55 + 0.9 * jet_noise(q * 2.1 + vec3f(5.1, 1.7, 3.3)));
      let rr = r / edge;
      // warmest as it leaves the hot section, and cooling fast: it has all but gone by the plume's end
      let along = exp(-u * 3.0) * smoothstep(1.0, 0.55, u);
      let heat = along * exp(-rr * rr * 2.5) * (0.55 + 0.45 * min(hard, 1.0));
      // and near the lip lit besides by the hot section glowing inside the nozzle
      let lit = 1.0 + 1.5 * exp(-s / (lip * 2.0));
      let body = smoothstep(1.25, 0.45, rr) * smoothstep(0.0, 0.03, s);
      em = jet_colour(0.25 + 0.55 * heat) * (heat * threads * threads * ${f3(JET.flame)} * lit * body * flame);
      // boosted, the faintest tint in the clear hot air: blue near the nozzle, yellow further out
      em += mix(vec3f(0.25, 0.45, 1.0), vec3f(1.0, 0.8, 0.4), smoothstep(0.05, 0.5, u)) * (${f3(JET.tint)} * boost * along * exp(-rr * rr * 2.0) * threads * body);
      // pushed to the most it gives, the faintest shock diamonds stand in the gas just out of the nozzle
      if (shocks > 0.0) {
        let x = s / cell + 0.5;
        let i = floor(x); let f = abs(fract(x) - 0.5) * 2.0; // 0 at a diamond's middle, 1 between two
        let rd0 = lip * 0.7 * (1.0 - f) * exp(-i * 0.3);
        let lens = smoothstep(rd0, rd0 * 0.2, r) * (1.0 - f);
        em += jet_colour(0.8) * (${f3(JET.diamond)} * flame * shocks * lens * exp(-i * 0.7) * step(i, ${(JET.cells - 0.5).toFixed(1)}) * step(0.5, i));
      }
    }
    let h = p.y - floor_y; let rho = length((p - hit).xz);
    if (impact > 0.0 && h < 0.6 && rho < spread * 1.45) {
      // where it comes down the gas turns and runs out flat over the ground, a ring of flame thinning as it spreads
      let q = vec3f((p - hit).xz / (spread * 0.35), rho * 3.0 - time * 7.0 + seed);
      let eddy = 0.6 * jet_noise(q) + 0.4 * jet_noise(q * 2.1 + 3.7);
      let sheet = exp(-h / (0.05 + rho * 0.12)) * smoothstep(spread, 0.0, rho * (0.7 + 0.6 * eddy));
      let heat = impact * sheet * eddy * eddy;
      // water takes the heat as it flashes to steam: what spreads over it is fainter still
      em += jet_colour(0.2 + 0.5 * heat) * (heat * ${f3(JET.splash)} * mix(1.0, 0.35, v.wet) * flame);
    }
    glow += em * dt;
  }
  return vec4f(glow * v.look.y, 0.0); // it hides nothing behind it: all it does is glow
}
`;
const JET_IN = '@location(0) at: vec4f, @location(1) dir: vec4f, @location(2) bend: vec4f, @location(3) look: vec4f, @location(4) more: vec4f';
// from outside a box, its faces towards the eye are drawn; from inside, those away from it
export const JET_LOW_DIV = JET.low;
export const JETS_LOW = JET_BOX + /* wgsl */`
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32, ${JET_IN}) -> V {
  var o = jet_box(vi, ii, at, dir, bend, look, more);
  if (o.direct > 0.5) { o.pos = vec4f(0.0); }
  return o;
}
struct Low { @location(0) a: vec4f, @location(1) b: vec4f, @location(2) c: vec4f, @location(3) d: vec4f } // each wing's engines' flames
@fragment fn fs(v: V, @builtin(front_facing) front: bool) -> Low {
  if (front == (v.look.z > 0.5)) { discard; }
  let c = jet_march(v); let k = jet_layer(u32(v.look.w + 0.5)); let none = vec4f(0.0);
  var o: Low;
  o.a = select(none, c, k == 0u); o.b = select(none, c, k == 1u); o.c = select(none, c, k == 2u); o.d = select(none, c, k == 3u);
  return o;
}
`;
export const JETS = JET_BOX + /* wgsl */`
@group(1) @binding(0) var flames: texture_2d_array<f32>; // JETS_LOW's, a layer a jet
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32, ${JET_IN}) -> V { return jet_box(vi, ii, at, dir, bend, look, more); }
// What it gives off is added; and wherever the scene leaves its box showing, the frame's alpha is cleared, for JETS_HAZE
// to know where the hot air is not hidden by what stands in front of it
@fragment fn fs(v: V, @builtin(front_facing) front: bool) -> @location(0) vec4f {
  if (front == (v.look.z > 0.5)) { discard; }
  var c: vec4f;
  if (v.direct > 0.5) { c = jet_march(v); }
  else { c = textureSampleLevel(flames, samp_clamp, v.pos.xy / vec2f(F.cam_fwd.w, F.cam_up.w), i32(jet_layer(u32(v.look.w + 0.5))), 0.0); }
  return vec4f(display(c.rgb), 1.0);
}
`;
// Heat shimmer: the hot air out of a jet bends the light through it, so what is behind it wavers, more the hotter and
// faster the air: most at the lip, fading over HAZE.reach m down the flow and gone by 1.5 m. After the main pass the
// rectangle of the frame round the jets is copied out twice; each jet's box is drawn into one copy reading the other a
// few pixels off where it would, and that copy goes back into the frame: the offsets flow down the jet's axis on the screen, long waves along it and short across, carried
// with the gas. Only where the main pass left the box showing (the frame's alpha, JETS), and never reading from what
// stands in front of the jet. HAZE: how far a ray is bent at the lip, as metres at the jet; how far down the flow it
// fades by e, m; the waves' lengths along the flow and across it, m; how fast they are carried, m/s; the most it moves a
// pixel, px.
const HAZE = { bend: 0.035, reach: 0.5, along: 0.09, across: 0.03, flow: 6.0, most: 24,
  boost_reach: 2.0, boost_flow: 3.0 }; // boosted: so many times further it reaches, and faster it streams
export const JETS_HAZE = JET_BOX + /* wgsl */`
@group(1) @binding(0) var scene: texture_2d<f32>; // the frame as the main pass left it, round the jets, from \`rect\`'s corner
@group(1) @binding(1) var<uniform> rect: vec4f;      // where that corner is in the frame, px; and the copy's size
// drawn into the copy's own frame: the frame's pixels less the corner
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32, ${JET_IN}) -> V {
  var o = jet_box(vi, ii, at, dir, bend, look, more);
  let frame = vec2f(F.cam_fwd.w, F.cam_up.w);
  o.pos = vec4f(o.pos.xy * frame / rect.zw + o.pos.w * (frame / rect.zw - 1.0 - 2.0 * rect.xy / rect.zw) * vec2f(1.0, -1.0), o.pos.zw);
  return o;
}
@fragment fn fs(v: V, @builtin(front_facing) front: bool) -> @location(0) vec4f {
  if (front == (v.look.z > 0.5)) { discard; }
  let here = textureLoad(scene, vec2i(v.pos.xy), 0); let px = v.pos.xy + rect.xy; // in the frame
  if (here.a > 0.5) { discard; } // something stands in front of the jet here
  // where this ray passes the jet's axis nearest, and how far down it that is
  let e = F.cam_pos.xyz; let rd = normalize(v.w - e); let at = v.at.xyz; let dir = v.dir.xyz; let len = v.at.w;
  let w = e - at; let ca = dot(rd, dir); let sn2 = max(1.0 - ca * ca, 1e-4);
  let wa = dot(w, dir); let tc = (ca * wa - dot(w, rd)) / sn2; let s = wa + ca * tc;
  let dc = length(w + rd * tc - dir * s - jet_bent(v.bend.xyz, s));
  let edge = jet_edge(v.look.x, s, v.dir.w);
  // boosted the air leaves faster and hotter, and the shimmer is all there is of the jet: stronger, and further down it
  let boost = v.more.x; let reach = ${f3(HAZE.reach)} * (1.0 + ${f3(HAZE.boost_reach)} * boost);
  let gone = mix(1.5, max(1.5, len * 1.1), boost);
  let hot = exp(-dc * dc / (edge * edge * 0.6)) * exp(-max(s, 0.0) / reach) * smoothstep(-0.01, 0.04, s) * (1.0 - smoothstep(gone * 0.8, gone, s))
    * min(max(v.dir.w, 0.35 * boost) * 1.4, 1.0);
  if (hot < 0.01) { discard; }
  // the axis's way across the screen here, and metres a pixel at the jet
  let p0 = F.view_proj * vec4f(at, 1.0); let p1 = F.view_proj * vec4f(at + dir * 0.3, 1.0);
  var flow = (p1.xy / p1.w - p0.xy / p0.w) * vec2f(F.cam_fwd.w, -F.cam_up.w);
  flow = select(vec2f(0.0, 1.0), normalize(flow), dot(flow, flow) > 1e-6);
  let across = vec2f(-flow.y, flow.x);
  let metres = pixel_angle() * max(tc, 0.05);
  let a = dot(px, flow) * metres; let b = dot(px, across) * metres;
  let q = vec2f((a - F.cam_pos.w * ${f3(HAZE.flow)} * (1.0 + ${f3(HAZE.boost_flow)} * boost)) / ${f3(HAZE.along)}, b / ${f3(HAZE.across)});
  let n = vec2f(noise(q) - 0.5, noise(q * 1.9 + vec2f(3.1, 1.3)) - 0.5);
  var shift = (across * n.x + flow * n.y * 0.5) * (${f3(HAZE.bend)} * 2.0 * hot / metres);
  shift = shift * min(1.0, ${HAZE.most.toFixed(1)} / max(length(shift), 1e-4));
  let size = vec2f(textureDimensions(scene));
  let there = textureSampleLevel(scene, samp_clamp, (v.pos.xy + shift) / size, 0.0);
  return vec4f(select(there.rgb, here.rgb, there.a > 0.5), 1.0); // not from what stands in front of it
}
`;
// blown snow (wasm/src/drift.c): each grain leaves the ground where a jet's wash tore it loose, with the wind's own velocity
// there, and races outward low over the ground; the air stops it within a few metres, it rises a little where the wind
// did, and sinks back. Most are grains that catch the light, the rest soft billows that spread as they go.
// What the jets' wash throws (and the spray they throw off the water) is drawn as powder: each thing thrown is POWDER.n
// small particles, its own share each of the throw, so that the cloud is made of many grains and never of a few soft
// blobs. A grain is mm to cm across (POWDER.grain), now and then a clump that tumbles down fast (POWDER.clumps of them);
// where the wash billows, a haze of finer, fainter particles that spread wider, rise higher and linger (POWDER.haze).
// Out along the ground they roll up and back over themselves in the jets' ring vortex (POWDER.roll, m, and how many
// radians a metre gone), lift and settle again, and drift on with the wind. They are lit, not shining: the sky, the moon,
// the street lamps' light where they are, the headlamps' beam and the jets' own light, all scattered forward more than
// back, so a grain between the eye and a light glows; under the car the sky and the moon are mostly kept off them.
// Fewer of the n are drawn, each more opaque, where a cloud is small on the screen (POWDER.px pixels a particle).
export const POWDER = { n: 24, grain: [0.003, 0.01], clumps: 0.05, clump: [0.02, 0.035], haze: [0.006, 0.02, 0.3], roll: [0.12, 0.45, 2.4], px: 2.5,
  albedo: 0.85, g: 0.5 };
export const BLOWN = CORNERS + /* wgsl */`
struct V { @builtin(position) pos: vec4f, @location(0) uv: vec2f, @location(1) tint: vec3f, @location(2) @interpolate(flat) lit: vec2f } // lit: 1 for powder, lit and laid over; its opacity
fn powder_hash(x: f32) -> f32 { return fract(sin(x * 12.9898) * 43758.5453); }
// Henyey-Greenstein, 1 on average over the sphere: c the cosine between the light's way and the way on to the eye
fn powder_phase(c: f32) -> f32 { let g = ${POWDER.g.toFixed(2)}; return (1.0 - g * g) / pow(1.0 + g * g - 2.0 * g * c, 1.5); }
fn powder(sub: u32, corner: vec2f, left: vec4f, flung: vec4f, kind: f32) -> V {
  var o: V;
  let seed = kind * 917.3 + f32(sub) * 13.7;
  let h = vec4f(powder_hash(seed), powder_hash(seed + 1.7), powder_hash(seed + 3.1), powder_hash(seed + 5.9));
  let haze = step(kind, ${f3(PUFF.billows)}); let lump = step(${(1 - POWDER.clumps).toFixed(3)}, h.y) * (1.0 - haze);
  let age = F.cam_pos.w - left.w;
  let life = ${f3(PUFF.life)} * (0.45 + 0.7 * h.x) * mix(1.0, 1.7, haze) * mix(1.0, 0.5, lump); let t = age / life;
  if (t <= 0.0 || t >= 1.0) { o.pos = vec4f(2.0, 2.0, 2.0, 1.0); return o; } // settled long since, or not yet thrown: nothing more to work out
  // its share of the throw: out along it, a little to either side, and up as the wind there rose
  let speed = length(flung.xz); let radial = select(vec2f(1.0, 0.0), flung.xz / max(speed, 1e-4), speed > 1e-3);
  let side = vec2f(-radial.y, radial.x);
  let out = flung.xz * (0.55 + 0.6 * h.y) + side * (speed * mix(0.3, 0.6, haze) * (h.z - 0.5));
  let v0 = vec3f(out.x, flung.y * (0.4 + 0.9 * h.w) + haze * 0.8 * h.w, out.y);
  let drag = ${f3(PUFF.drag)} * mix(1.0, 1.8, haze) * (0.7 + 0.6 * h.x) * mix(1.0, 0.5, lump);
  let gone = (1.0 - exp(-drag * age)) / drag;
  var p = left.xyz + v0 * gone;
  // rolled up and back over itself in the ring vortex, as far as it has gone
  let roll = mix(${f3(POWDER.roll[0])}, ${f3(POWDER.roll[1])}, h.z) * mix(0.6, 1.5, haze) * (1.0 - lump);
  let th = length(out) * gone / max(length(out), 1e-3) * ${f3(POWDER.roll[2])} + h.w;
  p.y += roll * (1.0 - cos(min(th, 3.1)));
  p.x -= radial.x * roll * 0.5 * sin(min(th, 3.1)); p.z -= radial.y * roll * 0.5 * sin(min(th, 3.1));
  p.y = max(p.y - 0.5 * age * age * mix(mix(0.9, 0.18, haze), 6.0, lump), left.y - 0.02);
  p += vec3f(0.7, 0.0, 0.3) * age * t * 0.6; // slowed, it drifts on with the night's wind
  let radius = mix(mix(mix(${f3(POWDER.grain[0])}, ${f3(POWDER.grain[1])}, h.x * h.x), mix(${f3(POWDER.haze[0])}, ${f3(POWDER.haze[1])}, h.x), haze),
    mix(${f3(POWDER.clump[0])}, ${f3(POWDER.clump[1])}, h.x), lump);
  let dist = dot(p - F.cam_pos.xyz, F.cam_fwd.xyz);
  // as many of the n as the cloud is big on the screen, each the more opaque for those left out
  let scale = F.cam_up.w / F.cam_right.w / max(dist, 0.05);
  let shown = clamp(0.6 * scale / ${f3(POWDER.px)}, 1.0, ${POWDER.n.toFixed(1)});
  var px = scale * radius;
  let cover = min(px * px, 1.0); px = max(px, 1.0); // under a pixel: one pixel, as faint as it covers
  let fade = (1.0 - t) * sqrt(1.0 - t) * smoothstep(0.0, 0.05, t);
  let alpha = mix(0.95, ${f3(POWDER.haze[2])}, haze) * fade * cover * min(${POWDER.n.toFixed(1)} / shown, 4.0);
  o.pos = sprite(F.view_proj * vec4f(p, 1.0), min(px, 60.0), corner);
  if (t <= 0.0 || t >= 1.0 || dist <= 0.05 || f32(sub) >= shown || alpha < 0.002) { o.pos = vec4f(2.0, 2.0, 2.0, 1.0); }
  // lit: the sky and the moon (kept off under the car), the street lamps, the headlamps and the jets, forward scattered
  let eye = normalize(F.cam_pos.xyz - p);
  let rel = p - F.car_pos.xyz; let fwd = F.car_dir.xyz; let lft = normalize(cross(F.car_up.xyz, fwd));
  let under = select(1.0, 0.35, abs(dot(rel, lft)) < 0.95 && abs(dot(rel, fwd)) < 2.35 && dot(rel, F.car_up.xyz) < 0.3);
  let lamp = lamps_at(p.xz);
  let lamps = (vec3f(1.0, 0.58, 0.24) * (lamp.r + lamp.g) + vec3f(0.80, 0.92, 1.0) * lamp.b) * 0.8;
  let v = p - (F.car_pos.xyz + vec3f(0.0, 0.8, 0.0));
  let along = dot(v, fwd); let perp = length(v - fwd * along);
  let beam = smoothstep(1.8, 5.0, along) * (1.0 - smoothstep(0.0, 1.0, perp / (along * 0.4 + 0.7))) * powder_phase(dot(fwd, eye));
  var jets = vec3f(0.0);
  if (F.jet.r > 0.0) {
    for (var i = 0u; i < 5u; i++) {
      let l = F.jet_at[i]; let to = p - l.xyz; let d2 = dot(to, to);
      let aim = F.jet_aim[i]; let out = mix(1.0, smoothstep(-0.35, 0.6, dot(to * inverseSqrt(max(d2, 1e-6)), aim.xyz)), aim.w);
      if (l.w > 0.0) { jets += F.jet.rgb * (l.w * jet_flicker(i) * out / max(d2, ${(JET_LIGHT.near * JET_LIGHT.near).toFixed(3)}) * powder_phase(dot(to * inverseSqrt(max(d2, 1e-6)), eye))); }
    }
  }
  let sky = vec3f(0.50, 0.60, 0.82) * 0.2 * under + moon_now() * powder_phase(dot(-MOON_DIR, eye)) * under * 0.5;
  o.tint = ${f3(POWDER.albedo)} * (sky + lamps * under + vec3f(1.0, 0.93, 0.80) * beam + jets / PI);
  o.uv = corner * 0.5; o.lit = vec2f(1.0, alpha);
  return o;
}
@vertex fn vs(@builtin(vertex_index) vx: u32, @location(0) left: vec4f, @location(1) flung: vec4f) -> V {
  let sub = vx / 6u; let corner = CORNER[vx % 6u];
  // whole metres it may fall, then its own number (drift.h); thrown by a road crew (plough.h), that number negated: sand
  // off the gritter's disc, or less 1 again snow off a plough's wing, neither falling below the road
  let crew = flung.w < 0.0; let grit = select(0.0, 1.0, crew && flung.w > -1.0); let curl = select(0.0, 1.0, flung.w <= -1.0);
  let fw = select(flung.w, fract(-flung.w), crew);
  let whole = floor(fw); let kind = fw - whole;
  // a clump off a tree's bough (shed.h) carries SHED_CLUMP over its metres: it falls as one, as a stone does, loosening a
  // little, and is gone where it lands (the powder and the burst it makes are grains of their own)
  let clump = step(${(SHED.clump - 0.5).toFixed(1)}, whole); let drop = whole - clump * ${SHED.clump.toFixed(1)};
  let origin = left.xyz; let age = F.cam_pos.w - left.w;
  // Snow a gust takes off a roof (it may fall: gust.h) is the same snow as what falls: flakes of its size and its light,
  // that linger as they sink and drift; the jets' wash throws grains and small puffs right under the eye; a plough's wing
  // casts its snow off in a heavy curl, more of it billowing and lingering than the wash throws
  let roof = step(0.5, drop) * (1.0 - clump);
  let flake = roof + clump; // drawn as the falling snow is: one of its flakes, or a clump of them
  if (!crew && flake < 0.5) { return powder(sub, corner, left, flung, kind); }
  let billow = step(kind, mix(${f3(PUFF.billows)}, ${f3(PLOUGH_CURL.billows)}, curl)) * (1.0 - flake) * (1.0 - grit);
  let life = mix(${f3(PUFF.life)} * (0.4 + 0.6 * fract(kind * 7.31)) * mix(1.0, ${f3(PUFF.roof)}, roof), sqrt(2.0 * drop / ${SHED.g.toFixed(2)}) + 0.05, clump)
    * mix(1.0, 0.45, grit) * mix(1.0, ${f3(PLOUGH_CURL.life)}, curl); let t = age / life;
  let drag = mix(${f3(PUFF.drag)} * mix(1.0, 1.5, billow) * mix(1.0, 0.45, roof), 0.3, clump);
  let gone = (1.0 - exp(-drag * age)) / drag;
  var p = origin + flung.xyz * gone;
  p.y = max(p.y - 0.5 * age * age * mix(mix(mix(1.0, 0.35, billow), ${SHED.g.toFixed(2)}, clump), 9.8, grit), origin.y - 0.02 - drop - 0.6 * grit); // sand falls as sand does, to the road
  // slowed, it drifts on with the night's wind
  p += vec3f(0.7, 0.0, 0.3) * age * t * 0.6 * (1.0 - clump);
  let clip = F.view_proj * vec4f(p, 1.0);
  let dist = dot(p - F.cam_pos.xyz, F.cam_fwd.xyz);
  let k3 = fract(kind * 3.7);
  let radius = mix(mix(mix(mix(0.014 + 0.012 * k3, mix(0.14, 0.75, t) * (0.7 + fract(kind * 13.1)), billow), 0.018 + 0.035 * k3 * k3, roof), 0.07 + 0.08 * k3, clump),
    0.006 + 0.004 * k3, grit) * mix(1.0, ${f3(PLOUGH_CURL.size)}, curl);
  // lit like the falling snow, by the sky, the street lamps and the headlight beam: white, as snow and spray are, never tinted by the jets
  let v = p - (F.car_pos.xyz + vec3f(0.0, 0.8, 0.0));
  let along = dot(v, F.car_dir.xyz); let perp = length(v - F.car_dir.xyz * along);
  let beam = smoothstep(1.8, 5.0, along) * (1.0 - smoothstep(0.0, 1.0, perp / (along * 0.4 + 0.7)));
  let lamp = lamps_at(p.xz);
  // a flake off a roof too small to draw fades rather than flickers, as the falling snow does; and the lamps' light
  // reaches it as it reaches the falling snow, from below
  var px = F.cam_up.w / F.cam_right.w * radius / dist;
  let dim = mix(1.0, min(1.0, px / 1.3), flake);
  px = max(px, mix(1.0, 1.3, flake));
  let high = mix(1.0, 1.0 - smoothstep(4.5, 9.0, p.y - ground_at(p.xz)), flake);
  let glint = mix(1.0, 0.72 + 0.4 * sin(F.cam_pos.w * (2.0 + kind * 6.0) + kind * 40.0), roof);
  let fade = mix((1.0 - t) * (1.0 - t) * smoothstep(0.0, 0.06, t), 1.0 - smoothstep(0.8, 1.0, t), clump); // a clump is whole till it lands
  var o: V;
  o.pos = sprite(clip, clamp(px, 1.0, 200.0), corner);
  if (t <= 0.0 || t >= 1.0 || dist <= 0.05 || sub > 0u) { o.pos = vec4f(2.0, 2.0, 2.0, 1.0); } // one sprite each
  o.uv = corner * 0.5; o.lit = vec2f(0.0);
  o.tint = (vec3f(0.50, 0.60, 0.82) * 0.2 + (vec3f(1.0, 0.58, 0.24) * (lamp.r + lamp.g) + vec3f(0.80, 0.92, 1.0) * lamp.b) * mix(0.6, 0.95, roof) * high
    + vec3f(1.0, 0.93, 0.80) * beam + plough_lamps(p, vec3f(0.0, 1.0, 0.0), 1.0) * 0.035)
    * fade * mix(mix(${f3(PUFF.grains)}, ${f3(PUFF.mist)}, billow), mix(${f3(PUFF.flake)}, ${f3(PUFF.clump)}, clump), flake) * dim * glint
    * mix(1.0, 0.3, grit) * mix(vec3f(1.0), vec3f(0.62, 0.52, 0.42), grit) * mix(1.0, ${f3(PLOUGH_CURL.bright)}, curl); // sand is dark, and brown
  return o;
}
// powder is laid over what is behind it as a small crisp grain; the rest is added, a soft spot (premultiplied: alpha 0)
@fragment fn fs(v: V) -> @location(0) vec4f {
  if (v.lit.x > 0.5) { let a = (1.0 - smoothstep(0.32, 0.5, length(v.uv))) * v.lit.y; return vec4f(display(v.tint) * a, a); }
  let a = 1.0 - smoothstep(0.05, 0.5, length(v.uv));
  return vec4f(display(v.tint * a) * a, 0.0);
}
`;

// The lasting marks: a long ribbon in a ring, and after it the traffic's, in runs (snow.h SNOW_RUN). They stay as they
// were laid until something in the world changes them: the snow that falls after fills them (each cross-section keeps
// the night's snowfall when it was laid, mm, in its r g b: the tone is its kind's, here), a plough's blade scrapes them
// off (snow_scrape), and one laid again elsewhere is folded into the roads' snow first (winter_fold).
const TRACK_COVER = 0.03; // m of new snow that fills a rut
const LAID = /* wgsl */`
fn laid_fall(c: vec3f) -> f32 { let b = round(c * 255.0); return (b.x + b.y * 256.0 + b.z * 65536.0) / 1000.0; }
fn covered(c: vec3f) -> f32 { return 1.0 - smoothstep(0.0, ${TRACK_COVER}, F.grit.z - laid_fall(c)); }
`;
export const TRACKS = SNOW_IN + LAID + /* wgsl */`
struct V { @builtin(position) pos: vec4f, @location(0) w: vec3f, @location(1) col: vec4f }
@vertex fn vs(@builtin(vertex_index) vi: u32, @location(0) pos: vec3f, @location(1) col: vec4f) -> V {
  var o: V;
  o.pos = F.view_proj * vec4f(pos, 1.0);
  o.w = pos;
  // its kind across (snow.c, KIND): the edges and ridges white, the rut floors and the belly's trough in shadow
  let k = vi % 10u;
  var tone = vec3f(1.0);
  if (k == 2u || k == 3u || k == 6u || k == 7u) { tone = vec3f(92.0, 105.0, 133.0) / 255.0; }
  if (k == 4u || k == 5u) { tone = vec3f(153.0, 166.0, 194.0) / 255.0; }
  o.col = vec4f(tone, col.a * covered(col.rgb));
  return o;
}
@fragment fn fs(v: V) -> @location(0) vec4f { return vec4f(finish(lit(v.col.rgb, vec3f(0.0, 1.0, 0.0), v.w, 1.0), v.w), v.col.a); }
`;

// the gritted track (wasm/src/plough.c): sand on the road behind the gritter, in a ring of sections, until snow falls
// over it; a section laid again elsewhere leaves its sand on the road's cell (winter_fold_grit)
export const GRIT = SNOW_IN + LAID + /* wgsl */`
struct V { @builtin(position) pos: vec4f, @location(0) w: vec3f, @location(1) col: vec4f }
@vertex fn vs(@builtin(vertex_index) vi: u32, @location(0) pos: vec3f, @location(1) col: vec4f) -> V {
  var o: V;
  o.pos = F.view_proj * vec4f(pos, 1.0);
  o.w = pos;
  o.col = vec4f(vec3f(46.0, 37.0, 29.0) / 255.0, col.a * covered(col.rgb)); // sand and grit, wet with the salt in it
  return o;
}
@fragment fn fs(v: V) -> @location(0) vec4f { return vec4f(finish(lit(v.col.rgb, vec3f(0.0, 1.0, 0.0), v.w, 1.0), v.w), v.col.a); }
`;

/* ---- sky and sea ---- */

// one triangle over the whole view, at the far plane: the depth test skips every pixel the land already covers. The
// aurora and the drifting cloud, most of the sky's cost and smooth across a few pixels, come from SKY_MAPS' texture,
// drawn first at half the width and height; the gradient, stars and moon are worked out here, sharp.
// The sky. It takes its noise from SKY_MAPS at a quarter of its pixels
export const SKY = /* wgsl */`
@group(1) @binding(0) var sky_low: texture_2d<f32>;
struct V { @builtin(position) pos: vec4f, @location(0) ndc: vec2f }
@vertex fn vs(@builtin(vertex_index) vi: u32) -> V {
  var o: V;
  o.ndc = vec2f(f32(vi & 1u) * 4.0 - 1.0, f32(vi >> 1u) * 4.0 - 1.0);
  o.pos = vec4f(o.ndc, 0.0, 1.0);
  return o;
}
@fragment fn fs(v: V) -> @location(0) vec4f {
  let d = normalize(F.cam_fwd.xyz + F.cam_right.xyz * v.ndc.x + F.cam_up.xyz * v.ndc.y);
  let noise = textureSampleLevel(sky_low, samp_clamp, v.pos.xy * 0.5 / vec2f(textureDimensions(sky_low)), 0.0);
  // inside a cloud a veil of its grey closes over the sky, as the haze does over everything else
  // the town's light in the low air over it, a dome over the horizon where the town lies
  let flat = d.xz / max(length(d.xz), 1e-4);
  let dome = town_dome(F.cam_pos.xz, flat) * pow8(1.0 - max(d.y, 0.0)) * ${AIR.most.toFixed(3)} * F.misc.w;
  return vec4f(lamps_over(mix(display(snow_veil(sky_over(d, 1.0, MOON_DIR, noise) + dome, F.cam_pos.xyz, d, 3e4)), display(CLOUD_TINT), F.cloud.w), cloud_lamps(d, 1e9)), 1.0);
}
`;
// What is worked out afresh each frame before the frame is drawn, as dispatches in one compute pass (a render pass
// apiece cost each its setup and flush on a tile-based GPU): the sky the water mirrors, and the sky's noise
// (sky_noise) at half the view's width and height, each texel at the middle of the 2 x 2 pixels it stands for. Only the
// frame's uniforms are read. Below the horizon there is no noise and nothing is worked out.
export const SKY_MAPS = /* wgsl */`
@group(1) @binding(0) var panorama: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var low_out: texture_storage_2d<rgba16float, write>;
@group(1) @binding(2) var<storage, read_write> rows_out: SkyRows; // for rows alone, which has a layout of its own
// The aurora's rows for this frame (row_noise): slot id.y, its lattice points from just left of where p.x of -4 falls.
// Rows 4i are curtain i's wander, 4i + 1 its rays, 4i + 2 its folds and 4i + 3 its fast rays, as sky_noise reads them;
// the wander and the folds drift with the whole form, by up to 1 in p. And the aurora's state this frame.
@compute @workgroup_size(64) fn rows(@builtin(global_invocation_id) id: vec3u) {
  let slot = id.y; let row = slot / 5u; let k = slot % 5u; let fi = f32(row / 4u); let time = F.cam_pos.w;
  var at = vec2f(-4.0 * 7.0 + time * 0.05, fi * 3.1 + time * 0.02);
  switch (row % 4u) {
    case 0u: { at = vec2f(-5.0 * 0.55 + fi * 7.3, time * 0.016 + fi); }
    case 2u: { at = vec2f(-5.0 * 2.2 + time * 0.35 + fi * 3.7, time * 0.12 + fi * 5.1); }
    case 3u: { at = vec2f(-4.0 * 7.0 + time * 1.4 + fi * 2.3, fi * 4.3 + time * 0.5); }
    default: {}
  }
  if (slot == 0u && id.x == 0u) { let a = aurora_state(time, F.car_up.w); rows_out.state[0] = a.a; rows_out.state[1] = a.b; }
  for (var j = 0u; j < k; j++) { at = at * 2.03 + 17.0; }
  let origin = i32(floor(at.x)) - 1;
  if (id.x == 0u) { rows_out.origin[slot] = origin; }
  let iy = floor(at.y); var fy = at.y - iy;
  fy = fy * fy * (3.0 - 2.0 * fy);
  let ix = f32(origin + i32(id.x));
  rows_out.g[slot * 1024u + id.x] = mix(hash2(vec2f(ix, iy)), hash2(vec2f(ix, iy + 1.0)), fy);
}
@compute @workgroup_size(8, 8) fn pano(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= vec2u(${SKY_W}u, ${SKY_H}u))) { return; }
  let uv = (vec2f(id.xy) + 0.5) / vec2f(${SKY_W}.0, ${SKY_H}.0);
  let az = (uv.x - 0.5) * 6.2831853; let el = uv.y * uv.y * 1.5707963;
  textureStore(panorama, id.xy, vec4f(sky_under(vec3f(cos(el) * cos(az), sin(el), cos(el) * sin(az)), 1.0, vec3f(0.0, -1.0, 0.0)), 1.0));
}
@compute @workgroup_size(8, 8) fn low(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= textureDimensions(low_out))) { return; }
  let at = (vec2f(id.xy) + 0.5) * 2.0;
  let ndc = vec2f(at.x / F.cam_fwd.w * 2.0 - 1.0, 1.0 - at.y / F.cam_up.w * 2.0);
  let d = normalize(F.cam_fwd.xyz + F.cam_right.xyz * ndc.x + F.cam_up.xyz * ndc.y);
  var n = vec4f(0.0);
  if (d.y > 0.0) { n = sky_noise(d, 1.0); }
  textureStore(low_out, id.xy, n);
}
`;
// The aurora as a light (aurora_light), worked out in the same pass once its rows are: the sky's noise (sky_noise, as the
// rippled water takes it, without the rays, which average out) over the whole sky above the horizon, 64 ways round by
// 32 rings of equal solid angle, less what drifting cloud hides of it as sky_over has it. The irradiance a surface facing
// n takes from a sky of radiance L is a quarter of the integral of L plus half that of L times the way it comes from, in
// n, to first order (spherical harmonics, l <= 1), which a + b.n holds for each colour. The snow below, lit by it from
// above, gives AURORA_BOUNCE of it back as a uniform lower hemisphere, which adds a half of its radiance times pi to a and
// takes as much from b's y. Into the rows' buffer, which every shader reads: the module's sky_rows is that buffer,
// written here.
const AURORA_WAYS = 64, AURORA_RINGS = 32, AURORA_BOUNCE = 0.75;
export const AURORA_LIGHT = PRELUDE.replace('@group(0) @binding(11) var<storage, read> sky_rows', '@group(1) @binding(2) var<storage, read_write> sky_rows') + /* wgsl */`
var<workgroup> sum: array<array<vec4f, ${AURORA_WAYS}>, 4>; // the integral of L, and of L along x, y and z: each way's, then summed
@compute @workgroup_size(${AURORA_WAYS}) fn main(@builtin(local_invocation_index) i: u32) {
  let az = (f32(i) + 0.5) * ${(2 * Math.PI / AURORA_WAYS).toFixed(7)};
  let way = vec2f(cos(az), sin(az));
  var s0 = vec3f(0.0); var sx = vec3f(0.0); var sy = vec3f(0.0); var sz = vec3f(0.0);
  for (var j = 0u; j < ${AURORA_RINGS}u; j++) {
    let y = (f32(j) + 0.5) / ${AURORA_RINGS}.0;
    let d = vec3f(way.x * sqrt(1.0 - y * y), y, way.y * sqrt(1.0 - y * y));
    let nz = sky_noise(d, 0.0);
    let l = nz.rgb * (1.0 - nz.a * 0.72);
    s0 += l; sx += l * d.x; sy += l * d.y; sz += l * d.z;
  }
  let dw = ${(2 * Math.PI / (AURORA_WAYS * AURORA_RINGS)).toExponential(6)}; // sr a sample
  sum[0][i] = vec4f(s0 * dw, 0.0); sum[1][i] = vec4f(sx * dw, 0.0); sum[2][i] = vec4f(sy * dw, 0.0); sum[3][i] = vec4f(sz * dw, 0.0);
  for (var h = ${AURORA_WAYS / 2}u; h > 0u; h >>= 1u) {
    workgroupBarrier();
    if (i < h) { for (var k = 0; k < 4; k++) { sum[k][i] += sum[k][i + h]; } }
  }
  if (i == 0u) {
    let a = sum[0][0].xyz * 0.25; let bx = sum[1][0].xyz * 0.5; var by = sum[2][0].xyz * 0.5; let bz = sum[3][0].xyz * 0.5;
    let back = max(a + by, vec3f(0.0)) * ${(AURORA_BOUNCE / 2).toFixed(4)}; // what the snow, lit from straight above, gives back
    by -= back;
    sky_rows.light[0] = vec4f(a + back, 0.0);
    sky_rows.light[1] = vec4f(bx.r, by.r, bz.r, 0.0);
    sky_rows.light[2] = vec4f(bx.g, by.g, bz.g, 0.0);
    sky_rows.light[3] = vec4f(bx.b, by.b, bz.b, 0.0);
  }
}
`;
// The ripple tiles in the same pass.
export const RIPPLE_MAPS = /* wgsl */`
struct Waves { k: array<vec4f, ${2 * RIPPLE_WAVES}>, g: array<vec4f, ${2 * RIPPLE_WAVES}> }
@group(0) @binding(0) var<uniform> W: Waves;
@group(0) @binding(1) var chop: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var wavelets: texture_storage_2d<rgba16float, write>;
fn tile(o: i32, at: vec2f) -> vec4f {
  var s = vec2f(0.0);
  for (var i = 0; i < ${RIPPLE_WAVES}; i++) { s += W.g[o + i].xy * cos(dot(W.k[o + i].xy, at) + W.k[o + i].z); }
  return vec4f(s, s * s);
}
@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= vec2u(${RIPPLE_N}u))) { return; }
  let at = vec2f(id.xy) + 0.5;
  textureStore(chop, id.xy, tile(0, at));
  textureStore(wavelets, id.xy, tile(${RIPPLE_WAVES}, at));
}
`;

// ---- the fjord (src/water.js) ----
// A still winter night's sea: near-black water with a soft sheen, the sky, the aurora and the moon in it, the town's
// lights laid over it as long broken streaks, slush and pancake ice gathering in the still corners. Drawn from:
//   sea map   rgba8, 4 m to the texel (wasm/src/sea.c): shore distance, open water, patchiness, made ground
//   horizon   two rgba8 maps over the backdrop: per direction (east, then every 45 degrees round through south),
//             sqrt(tan / HORIZON_TAN) of the highest land seen from sea level
//   ripples   two rgba16float tiles redrawn each frame from sums of travelling waves, with mipmaps: slope x, slope z,
//             and their squares, so that filtering keeps the slope variance a pixel spans (LEAN mapping)
//   sky       rgba16float panorama of the sky without its moon, redrawn each frame, with mipmaps: u azimuth,
//             v sqrt(elevation / 90 degrees)
const f5 = v => v.toFixed(5);
const SDF_LO = -8, SDF_HI = 56, SEA_PX = 4, HORIZON_N = 512, HORIZON_TAN = 0.6, HORIZON_STEPS = 40, HORIZON_REACH = [8, 16000], EARTH_R = 6.371e6;
const RIPPLE_TILE = [43, 6.1], RIPPLE_TURN = 0.61;
const CHOP = [0.004, 0.034], WAVELETS = [0.003, 0.05], CAPILLARY = 0.012; // RMS slope of each tile [glassy basin, open fjord]
const WIND_TO = [0.71, -0.71], GUST = [300, 100, 3.2];
const F0 = 0.02, MOON_DISC = 0.0245, BODY = [0.004, 0.007, 0.008], LAP = [0.7, 1.1], LIGHT_K = 3.2, PANCAKE = [9, 3.6];
const SLOPE_MAX = Math.hypot(CHOP[1], WAVELETS[1] * 1.7, CAPILLARY); // the steepest slope the roughest water shows, gusts and all
// A streak is drawn only where the water under it can throw its light: its strip is cut to STREAK_SIGMAS times the RMS slope
// of the roughest water it crosses (in a sheltered basin a tenth of the open fjord's, so a hundredth of the pixels), and a pixel
// of it whose water is calmer than that is given up at STREAK_SIGMAS_PIXEL before the surface is worked out. What the water
// can do is read from the bound map (SEA_MAKE below). Past 4.5 sigma a gaussian holds 4e-5 of its peak, past 5.5 sigma
// 3e-7: nothing that shows. (At six and eight, every strip ran on to where it never shows, and with the eye low over open
// water the near ends of thousands of them piled up in front of it.)
const STREAK_SIGMAS = 4.5, STREAK_SIGMAS_PIXEL = 5.5, STREAK_SPANS = 8; // spans: sea.h's SEA_STREAK_SPANS
const STREAK_FAINT = '0.0002'; // linear light, before exposure: a fraction of the least step the display shows

// The sea map is kept in pages (wasm/include/harstad/sea.h), as the town-wide maps are (tiles.h), and so are its mips
// and the bound map's levels: level l, max(1, size >> l) texels, has a page table of TILES_SIDE-texel tiles in the rows
// of sea_pages from sea_row(l), each entry its sea slot (low half; the page's kind above) or TILES_CONSTANT and then its
// one colour, and the bound map's slot or TILES_CONSTANT and then its one value (rg8). A slot holds its tile's texels
// and a ring of one more round them, so a bilinear look within the tile is the level's look whole (tiles.h). The
// includer says how large the map is, sea_dims().
const seaPages = (seaPage = `fn sea_page(l: u32, t: vec2u) -> vec4u { return tab_load(${TABLES.sea}u, vec2i(vec2u(t.x, t.y + sea_row(l)))); }`) => /* wgsl */`
fn sea_level(l: u32) -> vec2u { return max(sea_dims() >> vec2u(l), vec2u(1u)); }
fn sea_levels() -> u32 { let d = sea_dims(); return firstLeadingBit(max(d.x, d.y)) + 1u; }
fn sea_tiles(l: u32) -> vec2u { return (sea_level(l) + ${TILES_SIDE - 1}u) / ${TILES_SIDE}u; }
fn sea_row(l: u32) -> u32 { var r = 0u; for (var k = 0u; k < l; k++) { r += sea_tiles(k).y; } return r; }
${seaPage}
// where slot k's texel at (in texels from its corner) lies in its layer, and which layer
fn sea_slot(k: u32, at: vec2u) -> vec2u { return vec2u(k % ${TILES_ACROSS}u, k / ${TILES_ACROSS}u % ${TILES_ACROSS}u) * ${TILES_SLOT}u + at; }
// texel c of level l of the map's tiles, unfiltered: channels 0 (the sea map's, from atlas) or 2 (the bound map's)
fn sea_load(atlas: texture_2d_array<f32>, l: u32, c: vec2u, bound: bool) -> vec4f {
  let t = c / ${TILES_SIDE}u; let p = sea_page(l, t);
  let k = select(p.x & 0xffffu, p.z, bound); let v = select(p.y, p.w, bound);
  if (k == ${TILES_CONSTANT}u) { return unpack4x8unorm(v); }
  return textureLoad(atlas, sea_slot(k, c - t * ${TILES_SIDE}u + 1u), i32(k / ${TILES_PER_LAYER}u), 0);
}
`;
// A bilinear look at level l of the sea map at s, in its texels less a half, as the level whole gives it: clamped to its
// edges as the sampler clamped it, in the slot of floor(s)'s tile (as map_at)
const SEA_LOOK = /* wgsl */`
fn sea_bilinear(l: u32, s_in: vec2f) -> vec4f {
  let s = clamp(s_in, vec2f(-0.5), vec2f(sea_level(l)) - 0.5);
  let t = min(vec2u(max(floor((floor(s) + 0.5) * ${(1 / TILES_SIDE).toPrecision(9)}), vec2f(0.0))), sea_tiles(l) - 1u);
  let p = sea_page(l, t); let k = p.x & 0xffffu;
  if (k == ${TILES_CONSTANT}u) { return unpack4x8unorm(p.y); }
  let at = s - vec2f(t * ${TILES_SIDE}u) + 1.5;
  let uv = (vec2f(f32(k % ${TILES_ACROSS}u), f32(k / ${TILES_ACROSS}u % ${TILES_ACROSS}u)) * ${TILES_SLOT}.0 + at) * ${(1 / TILES_LAYER).toExponential()};
  return textureSampleLevel(sea_map, samp_sea, uv, i32(k / ${TILES_PER_LAYER}u), 0.0);
}
`;
// What the sea and the light streaks share: where a point lies, how rough the water there is, its ice.
const SEA_IN = /* wgsl */`
struct Sea {
  rect: vec4f,         // the sea map: x, z, width, depth
  horizon_rect: vec4f, // the horizon maps
  ripple_var: vec4f,   // each ripple tile's whole-tile slope variance per axis: the chop's, the wavelets'
  swell: array<vec4f, 3>, // wave vector, phase, amplitude
}
@group(1) @binding(0) var<uniform> S: Sea;
@group(1) @binding(1) var sea_map: texture_2d_array<f32>; // the sea map's slots, every level's (SEA_PAGES)
@group(1) @binding(2) var ripple_a: texture_2d<f32>;
@group(1) @binding(3) var ripple_b: texture_2d<f32>;
@group(1) @binding(4) var floes: texture_2d<f32>;
@group(1) @binding(5) var horizon: texture_2d_array<f32>; // the horizon's two maps (HORIZON), a layer each
@group(1) @binding(7) var sky_map: texture_2d<f32>;
@group(1) @binding(8) var samp_sea: sampler;    // clamped, mip-mapped
@group(1) @binding(9) var samp_ripple: sampler; // repeating, mip-mapped, anisotropic
@group(1) @binding(10) var samp_sky: sampler;   // round the horizon, clamped up and down
@group(1) @binding(11) var sea_bound: texture_2d_array<f32>; // what the water near a texel can be, at most (SEA_MAKE): its slots
@group(1) @binding(12) var sea_surf: texture_2d<f32>;  // the water's surface under each pair of pixels, this frame (SEA_SURFACE): mean slope, variance
@group(1) @binding(13) var sea_ice: texture_2d<f32>;   // and how much of it is ice
// (both maps' page tables, every level's: TABLES.sea)
fn sea_dims() -> vec2u { return vec2u(S.rect.zw * ${f5(1 / SEA_PX)} + 0.5); }
${seaPages()}${SEA_LOOK}
// The sea map's look at uv over a footprint of gx by gy, as the sampler took it from the map whole and its mips: at the
// level of detail the longer of the two spans in texels gives, between the levels either side of it
fn sea_look(uv: vec2f, gx: vec2f, gy: vec2f) -> vec4f {
  let size = vec2f(sea_dims());
  let lod = clamp(log2(max(max(length(gx * size), length(gy * size)), 1e-20)), 0.0, f32(sea_levels() - 1u));
  let l = u32(lod); let f = lod - f32(l);
  let a = sea_bilinear(l, uv * vec2f(sea_level(l)) - 0.5);
  if (f <= 0.0) { return a; }
  return mix(a, sea_bilinear(l + 1u, uv * vec2f(sea_level(l + 1u)) - 0.5), f);
}

struct SeaAt { shore: f32, open: f32, clump: f32, made: f32 }
fn sea_at(p: vec2f) -> SeaAt {
  let uv = (p - S.rect.xy) / S.rect.zw;
  var s = sea_look(uv, dpdx(uv), dpdy(uv));
  if (any(uv != clamp(uv, vec2f(0.0), vec2f(1.0)))) { s = vec4f(1.0, 1.0, 0.5, 0.0); }
  return SeaAt(mix(${f5(SDF_LO)}, ${f5(SDF_HI)}, s.r), s.g, s.b, s.a);
}
// cat's-paws: patches of ruffled water drifting downwind, drawn out along it
fn gust(p: vec2f) -> f32 {
  let w = vec2f(${f5(WIND_TO[0])}, ${f5(WIND_TO[1])}); let time = F.cam_pos.w;
  let q = vec2f(dot(p, w) - time * ${f5(GUST[2])}, dot(p, vec2f(-w.y, w.x))) / vec2f(${f5(GUST[0])}, ${f5(GUST[1])});
  return smoothstep(0.4, 0.8, noise(q) * 0.65 + noise(q * 2.3 + vec2f(time * 0.011, 5.2)) * 0.35);
}
// the long swell: slope of the sea surface (the same swell rocks the boats)
fn swell_slope(p: vec2f) -> vec2f {
  var s = vec2f(0.0);
  for (var i = 0; i < 3; i++) { s += S.swell[i].xy * S.swell[i].w * cos(dot(S.swell[i].xy, p) + S.swell[i].z); }
  return s;
}
const TURN = mat2x2f(${f5(Math.cos(RIPPLE_TURN))}, ${f5(Math.sin(RIPPLE_TURN))}, ${f5(-Math.sin(RIPPLE_TURN))}, ${f5(Math.cos(RIPPLE_TURN))});
const TURN2 = vec2f(${f5(Math.cos(RIPPLE_TURN) ** 2)}, ${f5(Math.sin(RIPPLE_TURN) ** 2)});
struct Slope { mean: vec2f, vr: vec2f, ice: f32 }
// Mean slope over the pixel and its variance per axis, from both ripple tiles at amplitudes a.x, a.y.
fn ripple(p: vec2f, dx: vec2f, dy: vec2f, a: vec2f) -> Slope {
  let c = textureSampleGrad(ripple_a, samp_ripple, p / ${f5(RIPPLE_TILE[0])}, dx / ${f5(RIPPLE_TILE[0])}, dy / ${f5(RIPPLE_TILE[0])});
  var o: Slope;
  o.mean = c.xy * a.x;
  o.vr = max(c.zw - c.xy * c.xy, vec2f(0.0)) * a.x * a.x;
  // The wavelets average out once a pixel spans a good part of their tile: then their variance alone stands in.
  let q = TURN * p / ${f5(RIPPLE_TILE[1])}; let qx = TURN * dx / ${f5(RIPPLE_TILE[1])}; let qy = TURN * dy / ${f5(RIPPLE_TILE[1])};
  let whole = vec2f(dot(TURN2, S.ripple_var.zw), dot(TURN2.yx, S.ripple_var.zw)); // the tile's variance, turned to world axes
  let span = max(dot(qx, qx), dot(qy, qy)); let fade = smoothstep(0.04, 0.25, span);
  var bm = vec2f(0.0); var bv = whole;
  if (fade < 1.0) {
    let b = textureSampleGrad(ripple_b, samp_ripple, q, qx, qy);
    let v = max(b.zw - b.xy * b.xy, vec2f(0.0));
    bm = b.xy * TURN * (1.0 - fade);
    bv = mix(vec2f(dot(TURN2, v), dot(TURN2.yx, v)), whole, fade);
  }
  o.mean += bm * a.y;
  o.vr += bv * a.y * a.y;
  return o;
}
// How much of the water is ice, 0..1: a ragged rim of slush and pancake ice where the water lies still, wider in the
// sheltered corners, and brash drifting against the quays. blur: how much shore distance a pixel spans.
fn ice_at(s: SeaAt, p: vec2f, blur: f32) -> f32 {
  let still = 1.0 - s.open;
  let reach = mix(0.6, 8.0, still * still) * (0.15 + 1.2 * s.clump) * (1.0 - 0.8 * s.made) + s.made * s.clump * s.clump * 3.0;
  let w = 1.2 + blur;
  // the ragged edge moves the shore by 3.3 m at most: further out than that there is no ice, further in nothing else
  if (s.shore - 3.3 >= reach + w) { return 0.0; }
  if (s.shore + 3.3 <= reach - w) { return 1.0; }
  let ragged = (noise(p * 0.21) - 0.5) * 5.0 + (noise(p * 0.9 + 7.0) - 0.5) * 1.6;
  return 1.0 - smoothstep(reach - w, reach + w, s.shore + ragged);
}
fn fresnel(c: f32) -> f32 { let m = 1.0 - clamp(c, 0.0, 1.0); let m2 = m * m; return ${f5(F0)} + ${f5(1 - F0)} * m2 * m2 * m; }
// Reflection of a small bright source in direction l off a surface of mean slope m and slope variance vr, seen from
// v: the share of facets turned just so (a gaussian over slope), times their Fresnel, per unit solid angle.
// Call where every pixel gets here.
fn glint(v: vec3f, l: vec3f, m: vec2f, vr_in: vec2f) -> f32 {
  let h = normalize(v + l);
  return glint_over(v, l, m, vr_in, fwidth(-h.xz / h.y - m));
}
// The same, told how much slope the pixel spans (d), for a caller that cannot ask its neighbours where it calls this
fn glint_over(v: vec3f, l: vec3f, m: vec2f, vr_in: vec2f, d: vec2f) -> f32 {
  let h = normalize(v + l);
  let q = -h.xz / h.y - m;
  let vr = vr_in + d * d * 0.25; // a pixel spans this much slope: spread the glint over it rather than let it flicker
  let D = exp(-0.5 * dot(q * q, 1.0 / vr)) / (6.2831853 * sqrt(vr.x * vr.y));
  let hy = h.y * h.y;
  return fresnel(dot(v, h)) * D / (4.0 * max(v.y, 0.01) * hy * hy);
}
// How open the water is to the wind. h0: the horizon texel holding the south-west, the way the wind comes from: land high
// and near upwind leaves its lee calm.
fn openness(s: SeaAt, h0: vec4f) -> f32 {
  let exposure = 1.0 - smoothstep(0.02, 0.14, h0.a * h0.a * ${f5(HORIZON_TAN)});
  return s.open * exposure * smoothstep(2.0, 40.0, s.shore);
}
// The most slope variance water that open can show along an axis: every wave of both tiles running one way, the gusts at
// their hardest, no ice. And the steepest the swell gets.
fn rough_most(open: f32) -> f32 {
  let chop = mix(${f5(CHOP[0])}, ${f5(CHOP[1])}, open); let wavelets = mix(${f5(WAVELETS[0])}, ${f5(WAVELETS[1])}, open) * 1.7;
  return chop * chop + wavelets * wavelets + ${f5(CAPILLARY * CAPILLARY)} * (0.15 + open);
}
fn swell_most() -> f32 {
  return length(S.swell[0].xy) * abs(S.swell[0].w) + length(S.swell[1].xy) * abs(S.swell[1].w) + length(S.swell[2].xy) * abs(S.swell[2].w);
}
// The water's surface at p over the pixel: its mean slope and slope variance per axis, and how much of it is ice.
// shore_blur: fwidth of the shore distance. least: the slope variance below which the caller has no use for the surface
// (a streak's pixel, too far from the facet it needs): then the tiles are not read, and it comes back as all ice.
fn surface(p: vec2f, s: SeaAt, open: f32, dx: vec2f, dy: vec2f, shore_blur: f32, least: f32) -> Slope {
  let ice = ice_at(s, p, shore_blur);
  var gusts = 0.0;
  if (ice < 1.0) { gusts = gust(p); } // under ice no wavelet shows
  let amp = vec2f(mix(${f5(CHOP[0])}, ${f5(CHOP[1])}, open), mix(${f5(WAVELETS[0])}, ${f5(WAVELETS[1])}, open) * (0.3 + 1.4 * gusts)) * (1.0 - ice);
  if (dot(amp, amp) + ${f5(CAPILLARY * CAPILLARY)} * (0.15 + open) * (1.0 - ice) < least) { return Slope(vec2f(0.0), vec2f(1.0), 1.0); }
  var o = ripple(p, dx, dy, amp);
  o.mean += swell_slope(p);
  o.vr += ${f5(CAPILLARY * CAPILLARY)} * (0.15 + open) * (1.0 - ice);
  o.ice = ice;
  return o;
}
// What the water within r m of p can be at most: x how open; y above 0 where any of it is nearer the sea than eight metres
// inland. Off the map, and where the map cannot say, the open fjord.
fn bound_at(p: vec2f, r: f32) -> vec2f {
  let px = ${f5(SEA_PX)};
  let level = i32(ceil(log2(max(r / px, 1.0))));
  if (level >= i32(sea_levels())) { return vec2f(1.0); }
  let c = vec2i(floor((p - S.rect.xy) / (px * exp2(f32(level)))));
  if (any(c < vec2i(1)) || any(c > vec2i(sea_level(u32(level))) - 2)) { return vec2f(1.0); }
  return sea_load(sea_bound, u32(level), vec2u(c), true).gr;
}
// The horizon texel at p, none off the backdrop. Call where every pixel gets here.
fn horizon_texel(layer: i32, p: vec2f) -> vec4f {
  let uv = (p - S.horizon_rect.xy) / S.horizon_rect.zw;
  return textureSample(horizon, samp_sea, uv, layer) * f32(all(uv == clamp(uv, vec2f(0.0), vec2f(1.0))));
}
`;

export const AFLOAT = afloat(SEA_IN);
// The water's surface under each pair of pixels across and down, worked out once a frame for the light streaks: a
// streak's pixel reads it instead of working it out for itself, and where thousands of strips cross the water in front
// of a low eye that was most of the frame. The pixel's point on the sea is where its ray meets sea level; the footprint
// of two pixels filters the ripple tiles (LEAN), so what a pair spans is kept as variance.
export const SEA_SURFACE = SEA_IN + /* wgsl */`
@group(2) @binding(0) var surf_out: texture_storage_2d<rgba16float, write>;
@group(2) @binding(1) var ice_out: texture_storage_2d<r32float, write>;
fn sea_point(px: vec2f) -> vec3f { // a pixel's point on the sea, xz, and 1 if its ray meets it at all
  let ndc = vec2f(px.x / F.cam_fwd.w * 2.0 - 1.0, 1.0 - px.y / F.cam_up.w * 2.0);
  let d = F.cam_fwd.xyz + F.cam_right.xyz * ndc.x + F.cam_up.xyz * ndc.y;
  if (d.y >= -1e-6 || F.cam_pos.y <= 0.0) { return vec3f(0.0); }
  return vec3f(F.cam_pos.xz + d.xz * (-F.cam_pos.y / d.y), 1.0);
}
@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= textureDimensions(surf_out))) { return; }
  let px = (vec2f(id.xy) + 0.5) * 2.0;
  let a = sea_point(px);
  if (a.z == 0.0) { textureStore(surf_out, id.xy, vec4f(0.0, 0.0, 1.0, 1.0)); textureStore(ice_out, id.xy, vec4f(1.0)); return; }
  let bx = sea_point(px + vec2f(2.0, 0.0)); let by = sea_point(px + vec2f(0.0, 2.0));
  let far = vec2f(1.0e3, 0.0);
  let dx = select(far, bx.xy - a.xy, bx.z > 0.0); let dy = select(far.yx, by.xy - a.xy, by.z > 0.0);
  let p = a.xy;
  let uv = (p - S.rect.xy) / S.rect.zw;
  var sm = sea_look(uv, dx / S.rect.zw, dy / S.rect.zw);
  if (any(uv != clamp(uv, vec2f(0.0), vec2f(1.0)))) { sm = vec4f(1.0, 1.0, 0.5, 0.0); }
  let s = SeaAt(mix(${f5(SDF_LO)}, ${f5(SDF_HI)}, sm.r), sm.g, sm.b, sm.a);
  let hv = (p - S.horizon_rect.xy) / S.horizon_rect.zw;
  let ht = textureSampleGrad(horizon, samp_sea, hv, 0, dx / S.horizon_rect.zw, dy / S.horizon_rect.zw) * f32(all(hv == clamp(hv, vec2f(0.0), vec2f(1.0))));
  let sf = surface(p, s, openness(s, ht), dx, dy, length(dx) + length(dy), -1.0);
  textureStore(surf_out, id.xy, vec4f(sf.mean, sf.vr));
  textureStore(ice_out, id.xy, vec4f(sf.ice));
}
`;
// The sea, and the lights' streaks over it, which the frame's streak pass laid at half its size
export const WATER = SEA_IN + /* wgsl */`
@group(1) @binding(15) var sea_glow: texture_2d<f32>; // the lights' streaks as the streak pass laid them, at half size
fn shore_tone() -> vec3f { return haze_now() * 0.8; }
struct V { @builtin(position) pos: vec4f, @location(0) w: vec2f }
// one square at sea level, centred under the eye and reaching almost to the far plane
@vertex fn vs(@builtin(vertex_index) vi: u32) -> V {
  let c = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(1.0, 1.0));
  var o: V;
  o.w = F.cam_pos.xz + c[vi] * (F.misc.y * 0.995);
  o.pos = F.view_proj * vec4f(o.w.x, 0.0, o.w.y, 1.0);
  return o;
}
// The sky in direction r, blurred by lod mip levels.
fn sky_at(r: vec3f, lod: f32) -> vec3f {
  let el = asin(clamp(r.y, 0.0, 1.0)) / 1.5707963;
  return textureSampleLevel(sky_map, samp_sky, vec2f(atan2(r.z, r.x) / 6.2831853 + 0.5, sqrt(el)), lod).rgb;
}
// Tangent of the highest land seen towards azimuth a (radians from east, through south), from a point's two horizon texels.
fn horizon_at(h0: vec4f, h1: vec4f, a: f32) -> f32 {
  let t = a / 0.7853982; let s = t - 8.0 * floor(t / 8.0);
  let w0 = max(1.0 - abs(s - vec4f(0.0, 1.0, 2.0, 3.0)), vec4f(0.0)) + max(1.0 - abs(s - vec4f(8.0, 9.0, 10.0, 11.0)), vec4f(0.0));
  let w1 = max(1.0 - abs(s - vec4f(4.0, 5.0, 6.0, 7.0)), vec4f(0.0));
  return (dot(w0, h0 * h0) + dot(w1, h1 * h1)) * ${f5(HORIZON_TAN)};
}
fn sea_lamps(p: vec2f) -> vec3f {
  let m = lamps_at(p);
  return street_light(m) + vec3f(1.0, 0.58, 0.26) * m.g * 1.25;
}
@fragment fn fs(i: V) -> @location(0) vec4f {
  let time = F.cam_pos.w; let p = i.w;
  let to_eye = F.cam_pos.xyz - vec3f(p.x, 0.0, p.y);
  let dist = length(to_eye);
  let v = to_eye / dist;
  let dx = dpdx(p); let dy = dpdy(p);
  let s = sea_at(p);
  let shore_blur = fwidth(s.shore);
  let h0 = horizon_texel(0, p); let h1 = horizon_texel(1, p);
  let sf = surface(p, s, openness(s, h0), dx, dy, shore_blur, -1.0);
${OFF.has('seabound') ? '  if (true) { return vec4f(bound_at(p, 16.0).x, bound_at(p, 1.0).x, openness(s, h0), 1.0); } // how open the bound map says the water within 16 m can be, within a texel, and how open it is' : ''}
  let mean = sf.mean; let vr = sf.vr; let ice = sf.ice;
  let n = normalize(vec3f(-mean.x, 1.0, -mean.y));
  let fr = fresnel(dot(n, v));

  // the sky, the aurora and the town's glow, hidden behind the far shore where it rises above the reflected ray
  var r = reflect(-v, n);
  r.y = abs(r.y);
  let spread = sqrt(vr.x + vr.y);
  var refl = sky_at(r, log2(1.0 + spread * 80.0));
  if (r.y < 0.4) {
    let tan_r = r.y / max(length(r.xz), 1e-4); let tan_h = horizon_at(h0, h1, atan2(r.z, r.x));
    refl = mix(shore_tone(), refl, smoothstep(-1.0, 1.0, (tan_r - tan_h) / (spread * 2.0 + 0.004)));
  }
  // the moon: its glitter path, sharp where the water is still, and the glow round it; none where the quays, the boats,
  // the town or the hills stand between the water and the moon
  let moon_in = moon_shadow(vec3f(p.x, 0.0, p.y), vec3f(0.0, 1.0, 0.0));
  let shine = (moon_now() * glint(v, MOON_DIR, mean, vr + ${f5((MOON_DISC / 2) ** 2)})
    + vec3f(0.55, 0.66, 0.9) * 0.05 * pow(max(dot(r, MOON_DIR), 0.0), 40.0) * fr) * moon_in;
  // the jets' flames, mirrored in the waves under them as the moon is: each as wide as a plume looks from here
  var flames = vec3f(0.0);
  if (F.jet.r > 0.0) {
    let wp = vec3f(p.x, 0.0, p.y); let slope = fwidth(mean);
    for (var k = 0u; k < 5u; k++) {
      let l = F.jet_at[k]; let to = l.xyz - wp; let d2 = dot(to, to);
      if (l.w <= 0.0 || d2 >= ${(JET_LIGHT.reach * JET_LIGHT.reach).toFixed(1)}) { continue; }
      let wide = ${f5(JET_LIGHT.girth * JET_LIGHT.girth)} / d2;
      flames += F.jet.rgb * (l.w * jet_flicker(k) / max(d2, ${(JET_LIGHT.near * JET_LIGHT.near).toFixed(3)})) * glint_over(v, to * inverseSqrt(d2), mean, vr + vec2f(wide), slope);
    }
  }

  // what the water itself gives back: the dark body lit faintly from above, more under the quay lamps
  let lamps = sea_lamps(p) * 1.6;
  let ambient = sky_light(p, n.y) + aurora_light(n);
  let moon_down = moon_now() * max(MOON_DIR.y, 0.0) * moon_in;
  let body = vec3f(${BODY.map(f5).join(', ')}) * (ambient + moon_down + lamps);

  var col = body + (fr * refl + shine + flames) * (1.0 - ice);
  if (ice > 0.003) {
    // slush and grease ice dull the water; pancakes of it, rimmed with snow, crowd in further; shore-fast ice is snowed
    // over. Two cell patterns laid over each other, turned and scaled apart, so the floes never fall into rows; smaller
    // brash against the quays, drifting slowly along them
    let scale = mix(${f5(PANCAKE[0])}, ${f5(PANCAKE[1])}, s.made);
    let q = p / scale + vec2f(time * 0.01 * s.made, 0.0); let qx = dx / scale; let qy = dy / scale;
    let fa = textureSampleGrad(floes, samp_repeat, q, qx, qy); let fb = textureSampleGrad(floes, samp_repeat, TURN * q * 1.37 + 0.31, TURN * qx * 1.37, TURN * qy * 1.37);
    var floe = fb;
    if (fa.r > fb.r) { floe = fa; }
    // round floes, each its own size, growing with the ice until they jostle; a raised rim of snow round each
    let edge = 1.0 - clamp(0.2 + min(ice, 1.0 - 0.4 * s.made) * 0.75 + (floe.g - 0.5) * 0.3, 0.0, 0.97);
    let plates = smoothstep(edge, edge + 0.06, floe.r);
    let rim = plates * (1.0 - smoothstep(edge + 0.07, edge + 0.2, floe.r));
    var albedo = mix(vec3f(0.05, 0.06, 0.07), mix(vec3f(0.22, 0.25, 0.29), vec3f(0.45, 0.49, 0.56), rim), plates);
    albedo = mix(albedo, vec3f(0.5, 0.54, 0.62), smoothstep(0.85, 0.98, ice) * (1.0 - s.made));
    let ice_lit = albedo * (sky_light(p, 0.5) + aurora_light(vec3f(0.0, 1.0, 0.0)) + moon_down + lamps);
    col = mix(col, ice_lit + fr * refl * 0.25 * (1.0 - plates), min(ice * 1.6, 1.0));
  }
  // the wet line where the water laps at ice, rock and wall, swinging in and out with the swell
  let swing = 0.5 + 0.5 * sin(time * 1.3 + dot(p, vec2f(0.21, 0.17)) + s.clump * 6.0);
  let wet_w = ${f5(LAP[0])} + ${f5(LAP[1] - LAP[0])} * swing; let aa = max(shore_blur, 0.05);
  var lap = smoothstep(wet_w + aa, wet_w - aa, s.shore) * smoothstep(-aa, aa, s.shore);
  let slush = max(fwidth(ice), 0.08); // and along the ice's ragged outer edge, a line of wet slush that comes and goes
  lap = max(lap, (1.0 - smoothstep(0.0, slush * (0.6 + swing), abs(ice - 0.25))) * 0.6);
  col = mix(col, vec3f(0.3, 0.33, 0.38) * (ambient + lamps + moon_down), lap * 0.7);

  // Sea smoke: in the cold a thin mist lies on the open water, a few metres deep, drifting with the wind. Seen slantwise
  // the eye looks through more of it; it takes the moon's light and the town's
  let drift = p * 0.012 + vec2f(time * 0.02, time * 0.008);
  let wisps = noise(drift) * 0.6 + noise(p * 0.037 - vec2f(time * 0.03, -time * 0.011)) * 0.4;
  let through = min(1.0 / max(v.y, 0.05), 12.0);
  let smoke = s.open * (1.0 - ice) * smoothstep(0.35, 0.85, wisps) * min(through * 0.035, ${AIR.smoke.toFixed(2)}) * F.misc.w;
  col = mix(col, haze_now() * 4.0 + moon_now() * 0.08 + town_glow(p) * 1.5, smoke);
  col = aerial(col, vec3f(p.x, 0.0, p.y));
  // Haze over the far water, as over the ground: laid over what the screen shows (F.fog is encoded for the screen, as
  // finish and the terrain lay it). It takes on the sky's colour at the horizon there, as the sky is drawn unhazed: so
  // the sea closes on the sky without a seam.
  let fog = 1.0 - exp(-F.fog.a * F.fog.a * dist * dist);
  var haze = F.fog.rgb;
  if (fog > 0.3) { haze = mix(F.fog.rgb, display(sky_at(normalize(vec3f(-v.x, 0.0, -v.z)), 1.0)), smoothstep(0.3, 1.0, fog)); }
  // and the lights' streaks over it, in the haze as the water is (added after the tone map, as they were drawn before)
  let glow = textureSampleLevel(sea_glow, samp_sea, i.pos.xy / vec2f(F.cam_fwd.w, F.cam_up.w), 0.0).rgb;
  return vec4f(lamps_over(mix(display(col), haze, fog), cloud_lamps(-v, dist)) + display(glow * (1.0 - fog) * (1.0 - snow_veil_at(vec3f(p.x, 0.0, p.y)).a)), 1.0); // (the streaks dimmed by the falling snow as the water is)
}
`;

// Each light's reflection drawn where it can fall: a strip of quad on the sea from the foot of the light towards the
// eye, as long and as wide as the roughest water could throw it. Its fragment shader then works out, per pixel, how
// many of the facets there are turned to send that light to the eye (the same glint as the moon's), so a light's streak
// lengthens and breaks up over open water and shrinks to a still image of it in the calm. Instances: struct sea_streak.
// The strip's shape, made once a frame for every strip and all its vertices (STREAK_FIT below), and how its light falls
const STREAK_FNS = /* wgsl */`
// half the difference in elevation between the light seen from s m out from its foot and the eye seen from there: the
// tilt a facet there needs to send the one to the other
fn tilt(s: f32, d: f32, hl: f32, hc: f32) -> f32 { return 0.5 * (atan2(hl, s) - atan2(hc, d - s)); }
// where between lo and hi the tilt comes down to t, to within (hi - lo) / 2^n: the nearer end of that (down) or the further
fn reach_in(lo_in: f32, hi_in: f32, d: f32, hl: f32, hc: f32, t: f32, n: i32, down: bool) -> f32 {
  var lo = lo_in; var hi = hi_in;
  for (var i = 0; i < n; i++) { let m = 0.5 * (lo + hi); if (tilt(m, d, hl, hc) > t) { lo = m; } else { hi = m; } }
  return select(hi, lo, down);
}
fn reach(d: f32, hl: f32, hc: f32, t: f32) -> f32 {
  var lo = 0.0; var hi = d;
  for (var i = 0; i < 14; i++) { let m = 0.5 * (lo + hi); if (tilt(m, d, hl, hc) > t) { lo = m; } else { hi = m; } }
  return 0.5 * (lo + hi);
}
// half the strip's width s m out, for facets tilted k across it: as it has been drawn (level distances, which fall short
// of it where the eye or the lamp is steeply overhead), and as it is
fn half_width(s: f32, d: f32, hl: f32, hc: f32, k: f32) -> f32 {
  let x = d - s;
  return k * (hc / length(vec2f(x, hc)) + hl / length(vec2f(s, hl))) * s * x / d;
}
fn half_width_true(s: f32, d: f32, hl: f32, hc: f32, k: f32) -> f32 {
  let dc = length(vec2f(d - s, hc)); let dl = length(vec2f(s, hl));
  return k * (hc / dc + hl / dl) * dl * dc / (dl + dc);
}
`;
// Each light's strip cut to the water it lies on, once a frame for all its vertices: where it starts and ends along the
// line from the lamp's foot to the eye, how wide the water it crosses lets it be, each station along it and what each
// span asks, and whether it shows at all. STREAK_FIT_WORDS floats a strip.
export const STREAK_FIT_WORDS = 28;
export const STREAK_FIT = SEA_IN + STREAK_FNS + /* wgsl */`
@group(2) @binding(0) var<storage, read> lights: array<f32>; // struct sea_streak: nine floats a light
@group(2) @binding(1) var<storage, read_write> fits: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= arrayLength(&fits) / ${STREAK_FIT_WORDS}u) { return; }
  let a = gid.x * 9u;
  let lamp = vec4f(lights[a], lights[a + 1u], lights[a + 2u], lights[a + 3u]);
  let tint = vec4f(lights[a + 4u], lights[a + 5u], lights[a + 6u], lights[a + 7u]);
  let hc = max(F.cam_pos.y, 0.2); let hl = lamp.y;
  let dv = F.cam_pos.xz - lamp.xz;
  let d = max(length(dv), 0.01);
  let u = dv / d; let across = vec2f(-u.y, u.x);
  let pixel = 2.0 * F.cam_right.w / F.cam_up.w; // the angle one pixel spans
  // as long and as wide as the roughest water there is could throw it
  let k_most = ${f5(LIGHT_K * SLOPE_MAX)};
  let s0_most = reach(d, hl, hc, k_most); let s1_most = reach(d, hl, hc, -k_most); let sm = d * hl / (hl + hc);
  // Then cut to the water it lies on. Along it in ${STREAK_SPANS} spans, half of them either side of the mirror point, each longer
  // than the last by one ratio going away from the lamp's foot, and from the eye: each asks the bound
  // map how rough the water under it, and under every pixel that reads the maps there, can be, and is kept if a facet
  // within ${STREAK_SIGMAS} sigma of that water's (and of the lamp's own width, seen from there) would do. The strip runs from the
  // first span kept to the last, and is as wide as the roughest of them asks.
  var s0 = s1_most; var s1 = s0_most; var k = 0.0; var t_near = 0.0; var t_far = 0.0;
  var ends: array<f32, ${STREAK_SPANS + 1}>; var asks: array<f32, ${STREAK_SPANS}>; // where each span ends, and the tilt it asks for: 0 if it is not kept
  ends[0] = s0_most;
  let swell = swell_most();
  // The most light any facet in a span could throw at the eye: a gaussian of tilts needing at least tau holds at most
  // 1 / (e pi tau^2) there however rough the water (its variance at tau^2 / 2), and never more than the calmest water's
  // peak. A span that could not make STREAK_FAINT even so is left off: seen from above the open fjord, every lamp's strip
  // ran on towards the eye, through water that would have to stand at a slant no wave makes to show it, and a hundred of
  // them lay over each pixel.
  let lum = dot(tint.rgb, vec3f(0.30, 0.59, 0.11));
  let vr_calm = ${f5(CHOP[0] ** 2 + (WAVELETS[0] * 0.3) ** 2 + 0.15 * CAPILLARY ** 2)};
  var sa = s0_most; var ta = tilt(sa, d, hl, hc);
  for (var j = 1; j <= ${STREAK_SPANS}; j++) {
    let f = f32(j) * ${f5(2 / STREAK_SPANS)};
    var sb = d - (d - s1_most) * pow((d - sm) / (d - s1_most), 2.0 - f);
    if (f < 1.0) { sb = s0_most * pow(sm / s0_most, f); }
    let tb = tilt(sb, d, hl, hc); let mid = 0.5 * (sa + sb);
    let wide = 1.15 * max(half_width(mid, d, hl, hc, k_most), max(half_width(sa, d, hl, hc, k_most), half_width(sb, d, hl, hc, k_most))) + lamp.w;
    let x = d - sa; let foot = 1.5 * pixel * (x * x + hc * hc) / hc; // the most of the sea a pixel there spans
    let b = bound_at(lamp.xz + u * mid, length(vec2f(0.5 * (sb - sa), wide)) + 3.0 * foot);
    let size = lamp.w / length(vec2f(sa, hl));
    let t = ${f5(STREAK_SIGMAS)} * sqrt(rough_most(b.x) + size * size * 0.25) + swell;
    let least = select(min(abs(ta), abs(tb)), 0.0, ta >= 0.0 && tb <= 0.0);
    let tau = max(least - swell, 0.0);
    let near_lamp = min(abs(sa), abs(sb)); let far_eye = max(abs(d - sa), abs(d - sb));
    let dl = length(vec2f(near_lamp, hl)); let de = length(vec2f(far_eye, hc)); let vy_s = hc / de;
    let g = min(${f5(1 / (Math.E * Math.PI))} / max(tau * tau, 1e-12), 1.0 / (6.2831853 * (vr_calm + (lamp.w / dl) * (lamp.w / dl) * 0.25)));
    let can = lum / (dl * dl) * fresnel(vy_s) * g / (4.0 * max(vy_s, 0.01)) * exp(-F.fog.a * F.fog.a * (dl + de) * (dl + de));
    if (b.y > 0.0 && least <= t && can >= ${STREAK_FAINT}) {
      if (s0 > sa) { s0 = sa; t_near = t; }
      s1 = sb; t_far = t;
      k = max(k, t); asks[j - 1] = t;
    }
    ends[j] = sb;
    sa = sb; ta = tb;
  }
  let dry = s1 <= s0;
  var margin = 2.0;
  if (!dry && k < k_most) {
    // the ends drawn in to where the tilt leaves what the end spans allow, and out again by what a pixel blurs
    let a = reach_in(s0, s1, d, hl, hc, t_near, 9, true); let b = reach_in(s0, s1, d, hl, hc, -t_far, 9, false);
    s0 = max(s0, a - 4.5 * pixel * (dot(vec2f(d - a, hc), vec2f(d - a, hc)) / hc)); s1 = min(s1, b + 4.5 * pixel * (dot(vec2f(d - b, hc), vec2f(d - b, hc)) / hc));
    margin = 4.5;
  }
  k = min(k, k_most);
  let fit = vec4f(s0, s1, k, f32(dry));
${OFF.has('streakfit') || OFF.has('streakcheck') ? '  s0 = s0_most; s1 = s1_most; k = k_most; margin = 2.0;' : ''}
${OFF.has('streakfit') || OFF.has('streakcheck') ? `  ends = array<f32, ${STREAK_SPANS + 1}>(s0, s0, 0.5 * (s0 + sm), 0.5 * (s0 + sm), sm, sm, 0.5 * (sm + s1), 0.5 * (sm + s1), s1); // as it has been drawn` : ''}
  // How bright the streak's brightest pixel could be: the light seen in the water at the mirror point, through the
  // calmest water there is (the gaussian's peak, 1 / 2 pi vr), its Fresnel and the haze. A streak that could not make even
  // a fraction of the display's least step anywhere is not drawn: most of the lamps near the water are dim and far, and
  // each was a strip over the sea ending in front of the eye, rasterised and blended for nothing.
  let mx = d - sm; let dm = length(vec2f(sm, hl)); let em = length(vec2f(mx, hc)); let vy = hc / em;
  let vr_least = ${f5(CHOP[0] ** 2 + (WAVELETS[0] * 0.3) ** 2 + 0.15 * CAPILLARY ** 2)} + (lamp.w / dm) * (lamp.w / dm) * 0.25;
  let peak = dot(tint.rgb, vec3f(0.30, 0.59, 0.11)) / (dm * dm) * fresnel(vy) / (6.2831853 * vr_least * 4.0 * max(vy, 0.01))
    * exp(-F.fog.a * F.fog.a * (dm + em) * (dm + em));
  let faint = peak < ${STREAK_FAINT};
  let b = gid.x * ${STREAK_FIT_WORDS}u;
  fits[b] = s0; fits[b + 1u] = s1; fits[b + 2u] = k; fits[b + 3u] = f32(dry) + 2.0 * f32(faint); fits[b + 4u] = margin;
  for (var j = 0; j <= ${STREAK_SPANS}; j++) { fits[b + 5u + u32(j)] = ends[j]; }
  for (var j = 0; j < ${STREAK_SPANS}; j++) { fits[b + 14u + u32(j)] = asks[j]; }
  fits[b + 24u] = fit.x; fits[b + 25u] = fit.y; fits[b + 26u] = fit.z; fits[b + 27u] = fit.w;
}
`;
export const STREAK = SEA_IN + STREAK_FNS + /* wgsl */`
struct V { @builtin(position) pos: vec4f, @location(0) p: vec3f, @location(1) tint: vec3f, @location(2) @interpolate(flat) lamp: vec4f,
  @location(3) @interpolate(flat) fit: vec4f } // fit: what the strip was cut to, for ?off=streakcheck
@group(1) @binding(14) var<storage, read> fits: array<f32>; // each strip as STREAK_FIT cut it this frame
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32, @location(0) lamp: vec4f, @location(1) tint: vec4f, @location(2) lit: f32) -> V {
  // a station at each end of every span along the strip, a vertex either side of each
  let k6 = array<u32, 6>(0u, 1u, 2u, 1u, 3u, 2u);
  let id = (vi / 6u) * 2u + k6[vi % 6u];
  let corner = vec2f(f32(id >> 1u), f32(id & 1u) * 2.0 - 1.0);
  // the strip as STREAK_FIT cut it this frame, once for all its vertices
  let hc = max(F.cam_pos.y, 0.2); let hl = lamp.y;
  let dv = F.cam_pos.xz - lamp.xz;
  let d = max(length(dv), 0.01);
  let u = dv / d; let across = vec2f(-u.y, u.x);
  let pixel = 2.0 * F.cam_right.w / F.cam_up.w; // the angle one pixel spans
  let k_most = ${f5(LIGHT_K * SLOPE_MAX)};
  let b = ii * ${STREAK_FIT_WORDS}u;
  let s0 = fits[b]; let s1 = fits[b + 1u]; let k = fits[b + 2u]; let margin = fits[b + 4u];
  let flags = u32(fits[b + 3u]); let dry = (flags & 1u) != 0u; let faint = (flags & 2u) != 0u;
  let fit = vec4f(fits[b + 24u], fits[b + 25u], fits[b + 26u], fits[b + 27u]);
  let at = i32(corner.x); let s = clamp(fits[b + 5u + u32(at)], s0, s1);
  var w = half_width(s, d, hl, hc, k_most) + lamp.w + 2.0 * pixel * length(vec2f(d - s, hc)); // never wider than the roughest water's strip
  if (k < k_most) {
    // and at each station as wide as the rougher of the spans it stands between asks; laid straight between stations, the
    // edge runs inside the curve it follows, so a little over
    let here = max(fits[b + 14u + u32(max(at - 1, 0))], fits[b + 14u + u32(min(at, ${STREAK_SPANS - 1}))]);
    w = min(w, half_width_true(s, d, hl, hc, 1.15 * here) + lamp.w + margin * pixel * length(vec2f(d - s, hc)));
  }
  let p = lamp.xz + u * s + across * corner.y * w;
  var o: V;
  o.p = vec3f(p.x, 0.0, p.y);
  o.lamp = lamp; o.fit = fit;
  // a beacon keeps its charted rhythm: all but out between its flashes
  o.tint = tint.rgb;
  if (tint.a > 0.0 && F.cam_pos.w - tint.a * floor(F.cam_pos.w / tint.a) > lit) { o.tint *= 0.04; }
  o.pos = F.view_proj * vec4f(o.p, 1.0);
${OFF.has('streakfit') || OFF.has('streakcheck') ? '' : '  if (dry || faint) { o.pos = vec4f(0.0); } // nothing but land under it, or too faint to show'}
  return o;
}
@fragment fn fs(i: V) -> @location(0) vec4f {
${OFF.has('streakflat') ? '  if (true) { return vec4f(1.0 / 255.0, 0.0, 0.0, 1.0); } // where the strips are, and how often over: one step of red each' : ''}
  let p = i.p.xz;
  let to_eye = F.cam_pos.xyz - i.p; let to_lamp = i.lamp.xyz - i.p;
  let dist = length(to_eye); let dl = length(to_lamp);
  let v = to_eye / dist; let l = to_lamp / dl;
  // What needs the neighbouring pixels is asked first, while every pixel is still here, so that a pixel can stop as soon as
  // it knows it shows nothing
  let size = i.lamp.w / dl;
  // The facet that would send the light to the eye from here: where the water at this pixel cannot be within
  // ${STREAK_SIGMAS_PIXEL} sigma of it, however it lies, there is nothing to work out.
  let h = normalize(v + l); let q0 = -h.xz / h.y; let spread = fwidth(q0);
  let need = max(length(q0) - swell_most(), 0.0);
  // the variance the water itself must have, beside the lamp's width and the pixel's spread, for this pixel to show anything
  var least = need * need * ${f5(1 / STREAK_SIGMAS_PIXEL ** 2)} - size * size * 0.25 - max(spread.x, spread.y) * max(spread.x, spread.y) * 0.25;
  // First against the roughest water there is: where the strips pile up in front of a low eye, most of their pixels go no
  // further than this
${OFF.has('streakfit') || OFF.has('streakcheck') ? '' : '  if (rough_most(1.0) < least) { return vec4f(0.0, 0.0, 0.0, 1.0); }'}
  // The water under the pixel, as SEA_SURFACE worked it out once for every strip that crosses it
  let at = vec2i(i.pos.xy); // drawn at half size: a texel of the surface to a pixel
  let surf = textureLoad(sea_surf, min(at, vec2i(textureDimensions(sea_surf)) - 1), 0);
  let sf = Slope(surf.xy, surf.zw, textureLoad(sea_ice, min(at, vec2i(textureDimensions(sea_ice)) - 1), 0).r);
${OFF.has('streakfit') || OFF.has('streakcheck') ? '  let would = false;' : ''}
  // never brighter than the lamp itself, times what the water gives back of it: its tint is its radiance times its area
  let c = min(i.tint / (dl * dl) * glint_over(v, l, sf.mean, sf.vr + size * size * 0.25, spread), i.tint * (fresnel(dot(v, h)) / (3.14159265 * i.lamp.w * i.lamp.w)))
    * (1.0 - sf.ice); // the water hazes it with itself
${OFF.has('streakcheck') ? `  { // only the light the cut strip and the pixel's test would have left out, sixty-four times over
    let dv = F.cam_pos.xz - i.lamp.xz; let d = max(length(dv), 0.01); let u = dv / d; let hc = max(F.cam_pos.y, 0.2);
    let along = dot(p - i.lamp.xz, u); let side = abs(dot(p - i.lamp.xz, vec2f(-u.y, u.x)));
    let pixel = 2.0 * F.cam_right.w / F.cam_up.w;
    let inside = i.fit.w < 0.5 && along >= i.fit.x && along <= i.fit.y
      && side <= half_width_true(along, d, i.lamp.y, hc, i.fit.z / 1.15) + i.lamp.w + 2.0 * pixel * length(vec2f(d - along, hc));
    if (inside && !would) { return vec4f(0.0, 0.0, 0.0, 1.0); }
    return vec4f(c * 64.0, 1.0);
  }` : ''}
  return vec4f(c, 1.0); // linear light: the water adds it where it shows
}
`;

// The sea map's levels above the first, and the bound map's (SEA_PAGES), made a level at a time from the slots of the
// level below: each slot a quad TILES_SLOT pixels square in a layer laid out as the atlas's (src/worker.js, seaPages).
// kind 'mip': a level of the sea map as makeMips made it from the map whole, a bilinear look at each texel's middle in
// the level below; 'bound0': the bound map's first level, from the sea map's; 'bound': a level of it from the one below.
// The bound map: for each texel of the sea map and each level above it, the most the water within a texel of that level
// around it can be, however the sea and horizon maps are filtered there: r the shore distance, g how open (openness()).
// Its first level is made from the maps themselves, over five texels by five; the rest from the level below, four by four.
// U: the map's texels across and down, and the level made.
export const SEA_MAKE = kind => /* wgsl */`
struct B { rect: vec4f, horizon_rect: vec4f }
@group(0) @binding(0) var src: texture_2d_array<f32>;
@group(0) @binding(1) var sea_pages: texture_2d<u32>;
${kind === 'mip' ? '@group(0) @binding(2) var samp_sea: sampler;' : ''}
@group(0) @binding(3) var<uniform> U: vec4u;
@group(0) @binding(4) var<storage, read> slots: array<vec4u>; // each slot made: its number, its tile's column and row
${kind === 'bound0' ? `@group(0) @binding(5) var horizon: texture_2d<f32>;
@group(0) @binding(6) var<uniform> R: B;` : ''}
fn sea_dims() -> vec2u { return U.xy; }
${seaPages('fn sea_page(l: u32, t: vec2u) -> vec4u { return textureLoad(sea_pages, vec2u(t.x, t.y + sea_row(l)), 0); }')}${kind === 'mip' ? SEA_LOOK.replace('textureSampleLevel(sea_map', 'textureSampleLevel(src') : ''}
struct V { @builtin(position) pos: vec4f, @location(0) @interpolate(flat) slot: vec4u }
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> V {
  let corner = array<vec2f, 6>(vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0), vec2f(0.0, 1.0));
  let s = slots[ii];
  let px = (vec2f(sea_slot(s.x, vec2u(0u))) + corner[vi] * ${TILES_SLOT}.0) * ${(1 / TILES_LAYER).toExponential()};
  return V(vec4f(px.x * 2.0 - 1.0, 1.0 - px.y * 2.0, 0.0, 1.0), s);
}
@fragment fn fs(v: V) -> @location(0) vec4f {
  let l = U.z; let size = vec2i(sea_level(l));
  // the level's texel this pixel of the slot holds: its tile's, or of the ring round it, clamped to the level as the sampler clamped it
  let c = clamp(vec2i(v.slot.yz * ${TILES_SIDE}u) - 1 + vec2i(v.pos.xy) - vec2i(sea_slot(v.slot.x, vec2u(0u))), vec2i(0), size - 1);
${kind === 'mip' ? `  return sea_bilinear(l - 1u, (vec2f(c) + 0.5) / vec2f(size) * vec2f(sea_level(l - 1u)) - 0.5);` : kind === 'bound0' ? `  let top = size - 1;
  var most = vec2f(0.0);
  for (var j = -2; j <= 2; j++) { for (var i = -2; i <= 2; i++) { most = max(most, sea_load(src, 0u, vec2u(clamp(c + vec2i(i, j), vec2i(0), top)), false).rg); } }
  // the least the land upwind can shelter anywhere a horizon texel read from here could lie
  let px = R.rect.zw / vec2f(top + 1); let n = vec2f(textureDimensions(horizon)); let ntop = vec2i(n) - 1;
  let lo = (R.rect.xy + (vec2f(c) - 2.0) * px - R.horizon_rect.xy) / R.horizon_rect.zw; let hi = (R.rect.xy + (vec2f(c) + 3.0) * px - R.horizon_rect.xy) / R.horizon_rect.zw;
  var lee = 1.0;
  if (any(lo < vec2f(0.0)) || any(hi > vec2f(1.0))) { lee = 0.0; }
  let h0 = vec2i(floor(lo * n - 0.5)); let h1 = vec2i(floor(hi * n - 0.5)) + 1;
  for (var j = h0.y; j <= h1.y; j++) { for (var i = h0.x; i <= h1.x; i++) { lee = min(lee, textureLoad(horizon, clamp(vec2i(i, j), vec2i(0), ntop), 0).a); } }
  let open = most.g * (1.0 - smoothstep(0.02, 0.14, lee * lee * ${f5(HORIZON_TAN)})) * smoothstep(2.0, 40.0, mix(${f5(SDF_LO)}, ${f5(SDF_HI)}, most.r));
  return vec4f(most.r, min(ceil(open * 255.0) / 255.0, 1.0), 0.0, 1.0);` : `  let top = vec2i(sea_level(l - 1u)) - 1;
  var most = vec2f(0.0);
  for (var j = -1; j <= 2; j++) { for (var i = -1; i <= 2; i++) { most = max(most, sea_load(src, l - 1u, vec2u(clamp(c * 2 + vec2i(i, j), vec2i(0), top)), true).rg); } }
  return vec4f(most, 0.0, 1.0);`}
}
`;

// What the car stirs up when it meets the water (src/water.js): the wake's ribbon, rings spreading from a splash, and
// spray. The buffers are refilled by wasm/src/water.c; the shaders age and fade them. Foam is lit by the sky, the moon
// and the painted lamps.
const WAKE_LIFE = 9, RING_LIFE = 5, SPRAY_LIFE = 1.4, RINGS = 8; // RINGS: WATER_RINGS in the C
const DIMPLE = { radius: 0.62, reach: 3.4, close: 1.1 }; // a jet's dimple: the lip's radius at full wash, how far its ripples run (m), s it takes to close over
const FOAM = /* wgsl */`
fn foam_light(p: vec2f) -> vec3f {
  let m = lamps_at(p);
  return sky_up(p) + moon_now() * 0.6 + (street_light(m) + vec3f(1.0, 0.58, 0.26) * m.g * 1.25) * 1.6;
}
`;
export const WAKE = FOAM + /* wgsl */`
struct V { @builtin(position) pos: vec4f, @location(0) wake: vec4f, @location(1) p: vec2f }
// wake: side -1..1, time laid, speed then, half width (m)
@vertex fn vs(@location(0) pos: vec3f, @location(1) wake: vec4f) -> V { return V(F.view_proj * vec4f(pos, 1.0), wake, pos.xz); }
@fragment fn fs(i: V) -> @location(0) vec4f {
  let age = F.cam_pos.w - i.wake.y; let fade = 1.0 - smoothstep(0.0, ${f5(WAKE_LIFE)}, age);
  let side = abs(i.wake.x); let off = side * i.wake.w;
  // a strip of churned white water as wide as the car, breaking up and fading; the two thin arms of the V
  let churn = (1.0 - smoothstep(0.6, 1.6 + age * 0.4, off)) * (1.0 - smoothstep(0.5, 5.0, age));
  let arms = smoothstep(0.9, 0.97, side) * (1.0 - smoothstep(0.98, 1.0, side)) * 0.35;
  let broken = smoothstep(0.25, 0.75, noise(i.p * 1.7) * 0.6 + noise(i.p * 5.3) * 0.4 + 0.25 - age * 0.06);
  let a = (churn * broken * 0.85 + arms * (0.6 + 0.4 * broken)) * fade * clamp(i.wake.z / 12.0, 0.0, 1.0);
  return vec4f(finish(vec3f(0.42, 0.46, 0.5) * foam_light(i.p), vec3f(i.p.x, 0.0, i.p.y)), a);
}
`;
export const RING = FOAM + /* wgsl */`
struct V { @builtin(position) pos: vec4f, @location(0) ring: vec4f, @location(1) p: vec2f, @location(2) age: f32, @location(3) @interpolate(flat) jet: f32 }
// ring: x, z, time begun, strength (a negative strength repeats every 3 * -strength s). After the first ${RINGS} come the
// dimples the car's jets blow in the water: x, z, when last blown, how hard.
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32, @location(0) ring: vec4f) -> V {
  let c = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(1.0, -1.0), vec2f(-1.0, -1.0), vec2f(-1.0, 1.0), vec2f(1.0, 1.0));
  var age = F.cam_pos.w - ring.z;
  var o: V;
  if (ii >= ${RINGS}u) {
    o.age = age; o.ring = ring; o.jet = 1.0;
    o.p = ring.xy + c[vi] * ${f5(DIMPLE.reach)};
    o.pos = F.view_proj * vec4f(o.p.x, 0.0, o.p.y, 1.0);
    if (age > ${f5(DIMPLE.close)} || ring.w <= 0.0) { o.pos = vec4f(0.0); }
    return o;
  }
  if (ring.w < 0.0) { let per = -ring.w * 3.0; age = age - per * floor(age / per); }
  o.age = age; o.ring = ring; o.jet = 0.0;
  o.p = ring.xy + c[vi] * (1.5 + age * 1.1 + abs(ring.w));
  o.pos = F.view_proj * vec4f(o.p.x, 0.0, o.p.y, 1.0);
  if (age > ${f5(RING_LIFE)}) { o.pos = vec4f(0.0); }
  return o;
}
@fragment fn fs(i: V) -> @location(0) vec4f {
  if (i.jet > 0.5) {
    // under a jet the water is pressed into a smooth dark hollow, its lip thrown up white, and ripples run out from it
    let s = min(i.ring.w * 1.6, 1.0) * (1.0 - smoothstep(0.1, ${f5(DIMPLE.close)}, i.age));
    let t = F.cam_pos.w; let d = length(i.p - i.ring.xy) + (noise(i.p * 2.1 + vec2f(t * 0.6, t * 0.4)) - 0.5) * 0.16; // never quite round
    let lip_at = ${f5(DIMPLE.radius)} * (0.6 + 0.4 * s); let out = d - lip_at;
    let hollow = (1.0 - smoothstep(lip_at * 0.45, lip_at, d)) * 0.55 * s;
    let lip = exp(-out * out / 0.006) * (0.55 + 0.45 * noise(i.p * 11.0 + vec2f(t * 3.1, -t * 2.3)));
    let run = pow(max(cos((out - t * 2.4) * 6.5), 0.0), 3.0) * exp(-max(out, 0.0) * 1.0) * step(0.0, out) * (1.0 - smoothstep(${f5(DIMPLE.reach * 0.6)}, ${f5(DIMPLE.reach)}, d));
    let foam = (lip * 0.8 + run * 0.45) * s;
    let col = mix(vec3f(0.004, 0.006, 0.009), vec3f(0.42, 0.46, 0.52) * foam_light(i.p), foam / max(foam + hollow, 1e-4));
    return vec4f(finish(col, vec3f(i.p.x, 0.0, i.p.y)), min(foam + hollow, 0.9));
  }
  let r = length(i.p - i.ring.xy); let front = 0.8 + i.age * 1.0; let k = r - front;
  // a short train of crests behind the leading one, dying out as it spreads
  let crest = max(cos(k * 6.0), 0.0) * exp(-k * k * 1.2);
  let a = crest * (1.0 - smoothstep(0.0, ${f5(RING_LIFE)}, i.age)) * min(abs(i.ring.w), 1.0) * 0.35 / (1.0 + i.age);
  return vec4f(finish(vec3f(0.4, 0.44, 0.5) * foam_light(i.p), vec3f(i.p.x, 0.0, i.p.y)), a);
}
`;
export const DROPS = FOAM + /* wgsl */`
struct V { @builtin(position) pos: vec4f, @location(0) uv: vec2f, @location(1) p: vec2f, @location(2) fade: f32 }
// droplets thrown up and falling back: each its own birth place, velocity and time; drawn as a round dot a few pixels across
@vertex fn vs(@builtin(vertex_index) vi: u32, @location(0) at: vec3f, @location(1) vel: vec4f) -> V {
  let c = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0));
  let age = F.cam_pos.w - vel.w;
  let p = at + vel.xyz * age - vec3f(0.0, 4.9 * age * age, 0.0);
  var o: V;
  o.fade = (1.0 - smoothstep(0.0, ${f5(SPRAY_LIFE)}, age)) * step(0.0, p.y) * step(0.0, age);
  o.p = p.xz; o.uv = c[vi] * 0.5;
  o.pos = F.view_proj * vec4f(p, 1.0);
  let depth = dot(p - F.cam_pos.xyz, F.cam_fwd.xyz);
  let size = clamp(F.cam_up.w * 0.5 / F.cam_right.w * 0.09 / max(depth, 0.01), 1.0, 24.0); // pixels across
  o.pos = vec4f(o.pos.xy + c[vi] * size / vec2f(F.cam_fwd.w, F.cam_up.w) * o.pos.w, o.pos.zw);
  if (o.fade <= 0.0 || depth <= 0.0) { o.pos = vec4f(0.0); }
  return o;
}
@fragment fn fs(i: V) -> @location(0) vec4f {
  let a = smoothstep(0.5, 0.15, length(i.uv)) * i.fade * 0.5;
  return vec4f(finish(vec3f(0.55, 0.6, 0.66) * foam_light(i.p), vec3f(i.p.x, 0.0, i.p.y)), a);
}
`;

// Chimney smoke (wasm/include/harstad/smoke.h): a plume over every stove burning tonight, SMOKE.puffs puffs each, every
// one of them on the path it takes in still air, worked out from the clock as smoke.c works it out, with the same
// numbers: lifted off the pot, its lift dying as it cools, dragged towards the light wind, the plume wandering across
// it and each puff straying from the plume. Where the air does more than the wind (a gust, the jets, the car's wake)
// the game simulates the plume and sends how far each puff has been pushed off that path, and how it has spread and
// thinned. The game sends the plumes in order, furthest first; within one, its puffs go furthest first too.
// Each puff is a soft ball of smoke: its depth through it, its light through itself from the moon above and the lamps
// below, darker on the side away from each, and what of them it scatters on towards the eye when it is seen against
// them. No glow: what shows is the smoke itself, lit. The exhaust steam of the engines running near the eye
// (SMOKE_STEAM) is the same stuff, puffs the game moves itself, lit the same way: red behind a car by its tail lamps.
const SMOKE = { puffs: 32, cool: 2.4, lift: 0.9, drawLift: 1.1, grip: 0.9, wind: 0.8, windDir: [0.9191, 0.3939], creep: 0.05,
  wander: [0.12, 0.012], stray: [0.05, 0.07], r: [0.17, 0.10, 0.0035],
  dense: 4.0,   // optical depth through the middle of a young puff from a stove drawing its hardest
  steam: 2.5,   // s over which the condensed water in it goes again, and it turns from white to grey
  minPx: 1.5 }; // pixels across a puff is drawn at the least, thinned by its true size against that
const SMOKE_PUFF = CORNERS + /* wgsl */`
struct Stove { at: vec4f, k: vec4f } // the pot's top and the ground under it; draw, phase, rate, and its key's bits
@group(1) @binding(0) var<storage, read> stoves: array<Stove>;
@group(1) @binding(1) var<storage, read> order: array<u32>;   // furthest first: the stove, and its slot plus one from bit 16
@group(1) @binding(2) var<storage, read> pushed: array<vec4f>; // per slot, per puff: its push and the radius gained; its velocity off the path, the share of its smoke left
@group(1) @binding(3) var<storage, read> steam: array<vec4f>;  // the exhaust steam: x, y, z, radius; optical depth, age as a share of its life, its own number
fn smoke_hash(v0: u32) -> u32 { var v = v0; v ^= v >> 16u; v *= 0x7feb352du; v ^= v >> 15u; v *= 0x846ca68bu; return v ^ (v >> 16u); }
fn unit(h: u32, shift: u32) -> f32 { return f32((h >> shift) & 0xffffu) / 65536.0; }
// Henyey-Greenstein's phase, g = 0.55: smoke sends most of what it scatters on the way the light was going
fn hg(c: f32) -> f32 { let g = 0.55; return (1.0 - g * g) / pow(1.0 + g * g - 2.0 * g * c, 1.5) * 0.25; }
struct V {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,          // across the puff, -1..1 each way
  @location(1) tau: f32,           // optical depth through its middle
  @location(2) moon_q: vec3f,      // the way to the moon and up, in the puff's own frame (along, across, to the eye)
  @location(3) up_q: vec3f,
  @location(4) albedo: vec3f,
  @location(5) below: vec3f,       // the street lamps' and windows' light from under it
  @location(6) front: vec3f,       // the cars', the truck's and the tower's on the side the eye sees, and from behind
  @location(7) back: vec3f,
  @location(8) phase: vec2f,       // how much of the moon's and of the light from below it sends on towards the eye
  @location(9) haze: vec4f,        // what the air between adds: its colour and share
  @location(10) fog: vec4f,        // the car's lamps in a cloud's haze (cloud_lamps); w: the haze's share
  @location(11) fade: vec3f,       // height over its source and over the ground (m), and how near the source it is
  @location(12) lump: vec2f,
  @location(13) wisp: f32,         // 1 for exhaust steam: small tufts and curls, broken up finely; 0 for chimney smoke
  @location(14) town: vec4f,       // the town's light given back round it (town_light)
  @location(15) veil: vec4f,       // the falling snow's between, as snow_veil has it: its colour and share
}
// One puff at w, r across, tau thick through its middle, moving at vel (it is drawn out along that by streak
// seconds' worth across the view), lit; its source at source, over the ground at ground.
fn puff(vi: u32, w: vec3f, r_in: f32, tau_in: f32, vel: vec3f, streak: f32, albedo: vec3f, source: vec3f, ground: f32, lump: vec2f) -> V {
  var o: V;
  o.pos = vec4f(0.0, 0.0, 2.0, 1.0); // nothing, unless it shows
  var r = r_in; var tau = tau_in;
  let to = w - F.cam_pos.xyz; let dist = length(to); let view = to / dist;
  let depth = dot(to, F.cam_fwd.xyz);
  if (tau < 0.002 || depth < -r) { return o; }
  // drawn no smaller than SMOKE.minPx across, and thinned by how much smaller it truly is
  let px = r / max(depth, 0.05) * F.cam_up.w / F.cam_right.w;
  if (px < ${f3(SMOKE.minPx)}) { tau *= px * px / ${f3(SMOKE.minPx * SMOKE.minPx)}; r *= ${f3(SMOKE.minPx)} / px; }
  // out of the view to either side, above or below, by more than it reaches drawn out (three radii at the most): it is
  // not lit, which is most of this shader's work, and most of the town's plumes are out of any one view
  let mid = F.view_proj * vec4f(w, 1.0);
  let room = 3.0 * r / F.cam_right.w * max(1.0, F.cam_up.w / F.cam_fwd.w);
  if (depth > 3.0 * r && (abs(mid.x) > mid.w + room || abs(mid.y) > mid.w + room)) { return o; }
  // drawn out along its motion across the view, so a column out of a pot is one and not beads on a string
  let side = vel - view * dot(vel, view); let sl = length(side);
  var ax = normalize(cross(vec3f(0.0, 1.0, 0.0), view) + vec3f(1e-5, 0.0, 0.0));
  if (sl > 0.05) { ax = side / sl; }
  let ay = cross(view, ax);
  let stretch = min(0.5 * sl * streak, 2.0 * r);
  tau *= r / (r + 0.5 * stretch);
  let c = QUAD[vi];
  let corner = w + ax * (c.x * (r + stretch)) + ay * (c.y * r);
  o.pos = F.view_proj * vec4f(corner, 1.0);
  o.uv = c; o.tau = tau; o.albedo = albedo;
  let ez = -view;
  o.moon_q = vec3f(dot(MOON_DIR, ax), dot(MOON_DIR, ay), dot(MOON_DIR, ez));
  o.up_q = vec3f(ax.y, ay.y, ez.y);
  o.phase = vec2f(hg(dot(MOON_DIR, view)), hg(-view.y));
  // from below: the lamps' pools as the snow under it gives them back, falling off as it rises over them; and the
  // lamps themselves, where it is down among them
  let above = w.y - ground;
  let m = glow_look(w.xz, 1u); // (level 1.5 as it was, the nearest mip 1)
  o.below = (street_light(m) * 0.9 + vec3f(1.0, 0.75, 0.45) * m.g * 0.5) / (1.0 + above * above / 64.0) + lamp_reach(w);
  o.town = town_light(w.xz);
  o.front = car_lamps(w, ez) + tower_wash(w, ez);
  o.back = car_lamps(w, view) + tower_wash(w, view);
  o.wisp = 0.0;
  // the air between, as aerial() has it, worked out once a puff
  o.haze = vec4f(air_lit(F.cam_pos.xyz + to * 0.5, to / max(dist, 1e-4)), 1.0 - exp(-air_tau(F.cam_pos.xyz, to)));
  o.fog = vec4f(cloud_lamps(normalize(corner - F.cam_pos.xyz), length(corner - F.cam_pos.xyz)), 1.0 - exp(-F.fog.a * F.fog.a * depth * depth));
  o.veil = snow_veil_at(corner);
  o.fade = vec3f(corner.y - source.y, corner.y - ground, 1.0 - smoothstep(0.6, 1.6, length((w - source).xz) / max(r, 0.3)));
  o.lump = lump;
  return o;
}
@fragment fn fs(i: V) -> @location(0) vec4f {
  let r2 = dot(i.uv, i.uv);
  if (r2 >= 1.0) { discard; }
  // a lumpy ball, its billows turning slowly; steam broken finer into tufts and curls, ragged out to its edge
  let lump = sin(i.uv.x * 3.3 + i.lump.x) * sin(i.uv.y * 2.9 + i.lump.y);
  var prof = max(1.0 - r2 * (1.0 + 0.35 * lump), 0.0);
  if (i.wisp > 0.0) {
    let fine = sin((i.uv.x + i.uv.y) * 7.0 + i.lump.x * 1.7) * sin((i.uv.x - i.uv.y) * 6.1 - i.lump.y * 1.3);
    prof = max(1.0 - r2 - (0.35 * lump + 0.25 * fine) * (0.35 + r2), 0.0);
  }
  // its foot dissolves where it leaves the pot, and nothing of it stands in the ground
  let fade = mix(1.0, smoothstep(-0.1, 0.35, i.fade.x), i.fade.z) * smoothstep(0.0, 0.6, i.fade.y);
  let depth = i.tau * prof * sqrt(prof) * fade;
  let alpha = 1.0 - exp(-depth);
  if (alpha < 0.002) { discard; }
  let n = vec3f(i.uv * 0.85, sqrt(max(1.0 - r2 * 0.72, 0.0)));
  // through itself: the side away from a light lies under more of its own smoke
  let moon_in = exp(-i.tau * 0.6 * (1.0 - dot(n, i.moon_q)));
  let below_in = exp(-i.tau * 0.6 * (1.0 + dot(n, i.up_q)));
  let thin = 1.0 - prof; // its edges, where it is seen against a light
  let amb = sky_from(i.town, dot(n, i.up_q), 0.0) * (0.75 + 0.25 * exp(-i.tau)); // (the pools' light from below is i.below's)
  let moon = moon_now() * (moon_in * max(dot(n, i.moon_q) * 0.6 + 0.4, 0.0) + i.phase.x * (1.0 + thin) * 2.0);
  let below = i.below * (below_in * max(0.4 - dot(n, i.up_q) * 0.6, 0.0) + i.phase.y * (1.0 + thin) * 2.0) * 1.6;
  let cars = i.front * (0.35 + 0.65 * n.z) + i.back * hg(0.9) * (0.5 + thin) * 2.0;
  let col = i.albedo * ((amb + moon + cars) / PI + below);
  return vec4f(lamps_over(mix(display(mix(mix(col, i.haze.rgb, i.haze.a), i.veil.rgb, i.veil.a)), F.fog.rgb, i.fog.w), i.fog.rgb), alpha);
}
`;
export const SMOKE_PUFFS = SMOKE_PUFF + /* wgsl */`
const N = ${SMOKE.puffs}u;
const WIND_DIR = vec3f(${f5(SMOKE.windDir[0])}, 0.0, ${f5(SMOKE.windDir[1])});
const WIND_ACROSS = vec3f(${f5(-SMOKE.windDir[1])}, 0.0, ${f5(SMOKE.windDir[0])});
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> V {
  let entry = order[ii / N]; let st = stoves[entry & 0xffffu]; let slot = entry >> 16u;
  let pot = st.at.xyz; let draw = st.k.x; let rate = st.k.z; let key = bitcast<u32>(st.k.w);
  let time = F.cam_pos.w;
  // which puff: the youngest first, or the oldest first where the plume's far end lies beyond its pot from the eye
  var k = ii % N;
  if (dot(WIND_DIR * 8.0 + vec3f(0.0, 4.0, 0.0), pot - F.cam_pos.xyz) > 0.0) { k = N - 1u - k; }
  let tn = (time * rate + st.k.y) * f32(N); let fl = floor(tn);
  let e = i32(fl) - i32(k);
  let life = (tn - fl + f32(k)) / f32(N); let a = life / rate;
  // its still-air path (smoke.c, smoke__path4)
  let h0 = smoke_hash(bitcast<u32>(e) + key); let h1 = smoke_hash(h0); let h2 = smoke_hash(h1);
  let q = unit(smoke_hash(key ^ 0x5bd1e995u), 0u);
  let w0 = (${f3(SMOKE.lift)} + ${f3(SMOKE.drawLift)} * draw) * (0.85 + 0.3 * unit(h0, 0u));
  let heat = exp(-a / ${f3(SMOKE.cool)}); let grip = exp(-a * ${f3(SMOKE.grip)});
  let up = w0 * ${f3(SMOKE.cool)} * (1.0 - heat) + ${f3(SMOKE.creep)} * a;
  let along = ${f3(SMOKE.wind)} * (a - (1.0 - grip) / ${f3(SMOKE.grip)});
  let tb = time - a;
  let wander = 0.55 * sin(0.53 * tb + 6.2831853 * q) + 0.3 * sin(1.31 * tb + (2.7 + 12.566371 * q));
  let across = wander * a * (${f5(SMOKE.wander[0])} + ${f5(SMOKE.wander[1])} * a);
  let stray = ${f3(SMOKE.stray[0])} + ${f3(SMOKE.stray[1])} * a;
  let sa = 2.0 * stray * (unit(h0, 16u) - 0.5); let su = 1.2 * stray * (unit(h1, 0u) - 0.5); let sc = 2.0 * stray * (unit(h1, 16u) - 0.5);
  var w = pot + WIND_DIR * (along + sa) + WIND_ACROSS * (across + sc) + vec3f(0.0, up + su, 0.0);
  var r = (${f3(SMOKE.r[0])} + a * (${f3(SMOKE.r[1])} + ${f5(SMOKE.r[2])} * a)) * (0.85 + 0.3 * unit(h2, 0u)) * (0.85 + 0.3 * draw);
  var vel = WIND_DIR * (${f3(SMOKE.wind)} * (1.0 - grip)) + vec3f(0.0, w0 * heat + ${f3(SMOKE.creep)}, 0.0);
  var left = 1.0; // the share of its smoke a blast has not torn away
  if (slot > 0u) { // simulated: what the air has done to it off that path
    let at = ((slot - 1u) * N + (bitcast<u32>(e) & (N - 1u))) * 2u;
    let p0 = pushed[at]; let p1 = pushed[at + 1u];
    w += p0.xyz; r += p0.w; vel += p1.xyz; left = p1.w;
  }
  // the same smoke spread wider is thinner through
  let r0 = ${f3(SMOKE.r[0])} * (0.85 + 0.3 * draw);
  let tau = ${f3(SMOKE.dense)} * draw * pow(r0 / r, 0.9) * smoothstep(0.0, 0.03, life) * pow(1.0 - life, 0.7) * left;
  // white while the water in it is still condensed, then the grey-blue of thin wood smoke
  let albedo = mix(vec3f(0.50, 0.52, 0.57), vec3f(0.90, 0.91, 0.93), exp(-a / ${f3(SMOKE.steam)}));
  return puff(vi, w, r, tau, vel, 1.0 / (rate * f32(N)), albedo, pot, st.at.w, vec2f(unit(h2, 16u), unit(h0, 8u)) * 6.2831853 + time * vec2f(0.21, -0.17));
}
`;
// The exhaust steam (smoke.c, smoke_steam): condensed water, its puffs as the game moved them. A tail lamp lights only
// the vapour near it: each lamp as it stands at the back of its car, shining back, falling off as the square of the
// distance; not the pool its light makes on the road a metre behind, which lights the snow and the chimney smoke.
export const SMOKE_STEAM = SMOKE_PUFF + /* wgsl */`
fn tail_lamps(w: vec3f, n: vec3f) -> vec3f {
  let back = -F.car_dir.xyz; let to = F.car_pos.xyz + F.car_up.xyz * 0.8 + back * 2.25 - w; // the player's, the pair as one
  let d2 = max(dot(to, to), 0.04); let l = to * inverseSqrt(d2);
  return F.tail.rgb * (0.12 * max(dot(-l, back), 0.0) * (0.3 + 0.7 * max(dot(n, l), 0.0)) / d2) * (1.0 - smoothstep(4.0, 8.0, sqrt(d2)));
}
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> V {
  let p = steam[ii * 2u]; let k = steam[ii * 2u + 1u];
  let lump = vec2f(k.z * 40.0, k.z * 23.0) + F.cam_pos.w * vec2f(1.1, -0.9);
  // what it is (wasm/include/harstad/smoke.h, smoke_kind): exhaust steam, white; a fire's smoke, the dark grey-brown of
  // burning timber and what is in a house, its steam white again once the water is on it (sent as steam); a hose's water
  let fire = k.w > 0.5 && k.w < 1.5; let spray = k.w > 1.5 && k.w < 2.5; let cloud = k.w > 2.5;
  let albedo = select(select(select(vec3f(0.9, 0.91, 0.93), vec3f(0.17, 0.155, 0.145), fire), vec3f(0.72, 0.77, 0.82), spray), vec3f(0.86, 0.87, 0.89), cloud);
  // no pot to rise from (its source well below it), and soft where it meets the road
  var o = puff(vi, p.xyz, p.w, k.x, vec3f(0.0), 0.0, albedo, p.xyz - vec3f(0.0, 10.0, 0.0), ground_at(p.xz) - 0.3, lump);
  if (o.tau > 0.0) {
    let view = normalize(p.xyz - F.cam_pos.xyz);
    o.front = car_beams(p.xyz, -view) + traffic_lamps(p.xyz, -view) + tail_lamps(p.xyz, -view) + tower_wash(p.xyz, -view);
    o.back = car_beams(p.xyz, view) + traffic_lamps(p.xyz, view) + tail_lamps(p.xyz, view) + tower_wash(p.xyz, view);
  }
  o.wisp = select(1.0, 0.0, fire || cloud);
  return o;
}
`;

// The horizon maps, marched once over the terrain's heights and the backdrop's. first: 0 for the map that holds east
// round through south to the south-west, 4 for the one that holds west round through north to the north-east.
export const HORIZON = /* wgsl */`
struct H { ground_rect: vec4f, rect: vec4f, first: f32, outer_n: f32, ground_tiles: vec4f, ground_back: vec4f }
@group(0) @binding(0) var<uniform> U: H;
@group(0) @binding(1) var ground: texture_2d_array<f32>;
@group(0) @binding(2) var outer: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;
@group(0) @binding(4) var ground_tiles: texture_2d<i32>;
@group(0) @binding(5) var ground_back: texture_2d_array<f32>;
${GROUND_LOOK}
@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f { return vec4f(f32(vi & 1u) * 4.0 - 1.0, f32(vi >> 1u) * 4.0 - 1.0, 0.0, 1.0); }
fn height_at(p: vec2f) -> f32 {
  let g = (p - U.ground_rect.xy) / U.ground_rect.zw;
  if (all(g == clamp(g, vec2f(0.002), vec2f(0.998)))) { return ground_look(ground, ground_tiles, ground_back, samp, U.ground_rect, U.ground_tiles, U.ground_back, p); }
  let o = (p - U.rect.xy) / U.rect.zw * ((U.outer_n - 1.0) / U.outer_n) + 0.5 / U.outer_n;
  if (all(o == clamp(o, vec2f(0.0), vec2f(1.0)))) { return textureSampleLevel(outer, samp, o, 0.0).r; }
  return -10.0;
}
@fragment fn fs(@builtin(position) at: vec4f) -> @location(0) vec4f {
  let p = U.rect.xy + at.xy / ${HORIZON_N}.0 * U.rect.zw;
  var o: vec4f;
  for (var k = 0; k < 4; k++) {
    let a = (U.first + f32(k)) * 0.7853982;
    let d = vec2f(cos(a), sin(a));
    var best = 0.0;
    for (var i = 0; i < ${HORIZON_STEPS}; i++) {
      let s = ${HORIZON_REACH[0].toFixed(1)} * pow(${(HORIZON_REACH[1] / HORIZON_REACH[0]).toFixed(1)}, f32(i) / ${(HORIZON_STEPS - 1).toFixed(1)});
      best = max(best, (height_at(p + d * s) - s * s / ${(2 * EARTH_R).toFixed(1)}) / s);
    }
    o[k] = sqrt(clamp(best / ${HORIZON_TAN.toFixed(2)}, 0.0, 1.0));
  }
  return o;
}
`;
export const HORIZON_SIZE = HORIZON_N;

// Depth only, from the moon: what casts a shadow, drawn into the shadow map. Its own small prelude: the frame's
// bind group holds the map itself, which cannot be read while it is being written.
export const SHADOW = /* wgsl */`
@group(0) @binding(0) var<uniform> vp: mat4x4f;
@vertex fn fixed(@location(0) pos: vec3f) -> @builtin(position) vec4f { return vp * vec4f(pos, 1.0); }
@vertex fn placed(@location(0) pos: vec3f, @location(4) at: vec4f, @location(5) scale: vec3f) -> @builtin(position) vec4f {
  let s = sin(at.w); let c = cos(at.w);
  let q = pos * scale;
  return vp * vec4f(at.xyz + vec3f(q.x * c + q.z * s, q.y, q.z * c - q.x * s), 1.0);
}
`;

// The haze's map in tiles (lamp.h, LAMP_GLOW_*), a layer of slots at a time. Its first level: the lamp map's first mip
// as makeMips would make it from the map whole (a bilinear look at each of its texels' middles), from the map's tiles,
// at the dense texel each of a slot's stands for, clamped to the map. at: the layer, the slots, the level; tiles: each
// slot's tile. No slot: nothing.
const GLOW_IN = /* wgsl */`
struct GlowU { size: vec4f, at: vec4u } // the lamp map's texels across and down, and its first mip's; the layer, the slots, the level
@group(0) @binding(2) var samp_clamp: sampler;
@group(0) @binding(3) var<uniform> U: GlowU;
@group(0) @binding(4) var<storage, read> tiles: array<vec2u>;
@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  return vec4f(f32(vi & 1u) * 4.0 - 1.0, 1.0 - f32(vi >> 1u) * 4.0, 0.0, 1.0);
}
// the slot's tile at level l, the texel p of it in the layer: the dense level's texel p stands for, clamped to the dense
// level's edges, as its texel in the slot; k the slot, or past the last for none
struct GlowAt { k: u32, i: vec2i, q: vec2u }
fn glow_at(pos: vec2f, l: u32) -> GlowAt {
  let side = ${GLOW_SLOT}u >> l; let p = vec2u(pos); let q = p / side;
  let k = U.at.x * ${GLOW_ACROSS * GLOW_ACROSS}u + q.y * ${GLOW_ACROSS}u + q.x;
  if (k >= U.at.y) { return GlowAt(k, vec2i(0), q); }
  let o = vec2i(tiles[k] * (${GLOW_SIDE}u >> l)) - (${GLOW_RIM} >> l);
  let n = max(vec2i(U.size.zw) >> vec2u(l), vec2i(1));
  return GlowAt(k, clamp(o + vec2i(p % side), vec2i(0), n - 1) - o, q);
}
`;
export const LAMP_GLOW = /* wgsl */`
@group(0) @binding(0) var lamp_map: texture_2d_array<f32>;
@group(0) @binding(1) var lamp_pages: texture_2d<u32>;
${GLOW_IN}
${lampLook('vec2i(textureDimensions(lamp_pages))', t => `textureLoad(lamp_pages, ${t}, 0)`)}
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let g = glow_at(pos.xy, 0u);
  if (g.k >= U.at.y) { return vec4f(0.0); }
  let d = vec2f(tiles[g.k] * ${GLOW_SIDE}u) - ${GLOW_RIM}.0 + vec2f(g.i); // the dense first mip's texel
  return vec4f(lamp_look((d + 0.5) / U.size.zw * U.size.xy - 0.5).rgb, 1.0);
}
`;
// Its second level, each texel a bilinear look where makeMips looked for the dense texel it stands for, clamped to the
// dense level's edges (src: the level over it, of the layer)
export const GLOW_MIP = /* wgsl */`
@group(0) @binding(0) var src: texture_2d<f32>;
${GLOW_IN}
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let l = U.at.z;
  let g = glow_at(pos.xy, l);
  if (g.k >= U.at.y) { return vec4f(0.0); }
  // where makeMips looks for the dense texel: at its middle as a share of its level, in the level over it (between the
  // 2 x 2 over it, but where that level is of odd size, a little on), moved into the slot by whole texels
  let o = vec2i(tiles[g.k] * (${GLOW_SIDE}u >> l)) - (${GLOW_RIM} >> l);
  let up = max(vec2i(U.size.zw) >> vec2u(l - 1u), vec2i(1));
  let x = (vec2f(o + g.i) + 0.5) * vec2f(up) / vec2f(max(up >> vec2u(1u), vec2i(1)));
  let c = vec2f(g.q * (${GLOW_SLOT * 2}u >> l)) + x - vec2f(o * 2);
  return textureSampleLevel(src, samp_clamp, c / f32(${GLOW_LAYER}u >> (l - 1u)), 0.0);
}
`;
// The ground photo's pages compressed as the GPU holds them (src/worker.js, photoPaging): BC7's mode 6, every 4 x 4 block
// of a level one line through its colours, its two ends of 7 bits and a bit shared by an end's channels, and each texel
// one of 16 steps along it. A block's line is its colours' principal axis, from the least to the most of them along it;
// then, twice, the ends that best fit the steps the texels took (least squares), kept where they fit better. src: the
// level's bytes (the page's texture seen as rgba8unorm, so the bytes are as decoded, not made linear); blocks: one
// vec4u a block, row by row, U.z to a row.
export const BC7 = /* wgsl */`
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> blocks: array<vec4u>;
@group(0) @binding(2) var<uniform> U: vec4u; // blocks across, down, and to a row of the buffer
const W4 = array<u32, 16>(0u, 4u, 9u, 13u, 17u, 21u, 26u, 30u, 34u, 38u, 43u, 47u, 51u, 55u, 60u, 64u);
// an end as 7 bits a channel and the shared bit that comes nearer it
fn quantise(e: vec3f) -> vec4u {
  var best = vec4u(0u); var err = 1e30;
  for (var p = 0u; p < 2u; p++) {
    let q = clamp(round((e - f32(p)) * 0.5), vec3f(0.0), vec3f(127.0));
    let d = q * 2.0 + f32(p) - e;
    if (dot(d, d) < err) { err = dot(d, d); best = vec4u(vec3u(q), p); }
  }
  return best;
}
fn colour(q: vec4u) -> vec3u { return (q.xyz << vec3u(1u)) | vec3u(q.w); }
// each texel's nearest of the 16 steps from a to b, as the hardware makes them; returns the squared error
fn fit(px: ptr<function, array<vec3f, 16>>, a: vec4u, b: vec4u, idx: ptr<function, array<u32, 16>>) -> f32 {
  var steps: array<vec3f, 16>;
  let ca = colour(a); let cb = colour(b);
  for (var k = 0u; k < 16u; k++) { steps[k] = vec3f(((64u - W4[k]) * ca + W4[k] * cb + vec3u(32u)) >> vec3u(6u)); }
  var total = 0.0;
  for (var i = 0u; i < 16u; i++) {
    var bi = 0u; var be = 1e30;
    for (var k = 0u; k < 16u; k++) { let d = (*px)[i] - steps[k]; let e = dot(d, d); if (e < be) { be = e; bi = k; } }
    (*idx)[i] = bi; total += be;
  }
  return total;
}
fn put(w: ptr<function, array<u32, 4>>, at: ptr<function, u32>, v: u32, n: u32) {
  let i = *at >> 5u; let s = *at & 31u;
  (*w)[i] |= v << s;
  if (s + n > 32u) { (*w)[i + 1u] |= v >> (32u - s); }
  *at += n;
}
@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= U.x || id.y >= U.y) { return; }
  var px: array<vec3f, 16>;
  var mean = vec3f(0.0);
  for (var i = 0u; i < 16u; i++) {
    px[i] = round(textureLoad(src, vec2i(id.xy * 4u + vec2u(i & 3u, i >> 2u)), 0).rgb * 255.0);
    mean += px[i];
  }
  mean /= 16.0;
  var c0 = vec3f(0.0); var c1 = vec3f(0.0); // the covariance: xx yy zz, xy xz yz
  for (var i = 0u; i < 16u; i++) { let d = px[i] - mean; c0 += d * d; c1 += d.xxy * d.yzz; }
  var axis = vec3f(0.57735);
  for (var k = 0; k < 8; k++) {
    let v = vec3f(c0.x * axis.x + c1.x * axis.y + c1.y * axis.z, c1.x * axis.x + c0.y * axis.y + c1.z * axis.z, c1.y * axis.x + c1.z * axis.y + c0.z * axis.z);
    let l = length(v);
    if (l < 1e-6) { break; }
    axis = v / l;
  }
  var lo = 1e30; var hi = -1e30;
  for (var i = 0u; i < 16u; i++) { let t = dot(px[i] - mean, axis); lo = min(lo, t); hi = max(hi, t); }
  var a = quantise(clamp(mean + axis * lo, vec3f(0.0), vec3f(255.0)));
  var b = quantise(clamp(mean + axis * hi, vec3f(0.0), vec3f(255.0)));
  var idx: array<u32, 16>;
  var err = fit(&px, a, b, &idx);
  for (var r = 0; r < 2 && err > 0.0; r++) { // the ends that best fit the steps taken
    var aa = 0.0; var ab = 0.0; var bb = 0.0; var ax = vec3f(0.0); var bx = vec3f(0.0);
    for (var i = 0u; i < 16u; i++) {
      let w = f32(W4[idx[i]]) / 64.0;
      aa += (1.0 - w) * (1.0 - w); ab += (1.0 - w) * w; bb += w * w;
      ax += (1.0 - w) * px[i]; bx += w * px[i];
    }
    let det = aa * bb - ab * ab;
    if (abs(det) < 1e-6) { break; }
    let na = quantise(clamp((ax * bb - bx * ab) / det, vec3f(0.0), vec3f(255.0)));
    let nb = quantise(clamp((bx * aa - ax * ab) / det, vec3f(0.0), vec3f(255.0)));
    var nidx: array<u32, 16>;
    let nerr = fit(&px, na, nb, &nidx);
    if (nerr >= err) { break; }
    a = na; b = nb; idx = nidx; err = nerr;
  }
  if (idx[0] >= 8u) { // the first texel's step must have its top bit clear: the ends swapped
    let t = a; a = b; b = t;
    for (var i = 0u; i < 16u; i++) { idx[i] = 15u - idx[i]; }
  }
  var w = array<u32, 4>(0u, 0u, 0u, 0u); var at = 0u;
  put(&w, &at, 64u, 7u); // mode 6
  put(&w, &at, a.x, 7u); put(&w, &at, b.x, 7u); put(&w, &at, a.y, 7u); put(&w, &at, b.y, 7u); put(&w, &at, a.z, 7u); put(&w, &at, b.z, 7u);
  put(&w, &at, 127u, 7u); put(&w, &at, 127u, 7u); // alpha, unused
  put(&w, &at, a.w, 1u); put(&w, &at, b.w, 1u);
  put(&w, &at, idx[0], 3u);
  for (var i = 1u; i < 16u; i++) { put(&w, &at, idx[i], 4u); }
  blocks[id.y * U.z + id.x] = vec4u(w[0], w[1], w[2], w[3]);
}
`;
export const MIP = /* wgsl */`
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
struct V { @builtin(position) pos: vec4f, @location(0) uv: vec2f }
@vertex fn vs(@builtin(vertex_index) vi: u32) -> V {
  var o: V;
  o.uv = vec2f(f32(vi & 1u) * 2.0, f32(vi >> 1u) * 2.0);
  o.pos = vec4f(o.uv.x * 2.0 - 1.0, 1.0 - o.uv.y * 2.0, 0.0, 1.0);
  return o;
}
@fragment fn fs(v: V) -> @location(0) vec4f { return textureSample(src, samp, v.uv); }
`;
// A mip level from the one above it, in a compute pass: for each of `count` textures, src the level above, dst this one.
// Each texel the mean of the 2 x 2 above it; at an edge of odd size the last row or column is counted twice.
export const halve = (count, fmt) => Array.from({ length: count }, (_, i) => /* wgsl */`
@group(0) @binding(${i * 2}) var src${i}: texture_2d<f32>;
@group(0) @binding(${i * 2 + 1}) var dst${i}: texture_storage_2d<${fmt}, write>;`).join('') + /* wgsl */`
@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= textureDimensions(dst0))) { return; }
  let p = vec2i(id.xy) * 2;
` + Array.from({ length: count }, (_, i) => /* wgsl */`
  {
    let q = min(p + 1, vec2i(textureDimensions(src${i})) - 1);
    textureStore(dst${i}, id.xy, (textureLoad(src${i}, p, 0) + textureLoad(src${i}, vec2i(q.x, p.y), 0) + textureLoad(src${i}, vec2i(p.x, q.y), 0) + textureLoad(src${i}, q, 0)) * 0.25);
  }`).join('') + `
}
`;

// ---- the upscale (?scale=0.77): AMD's FidelityFX Super Resolution 1, its two passes (ffx_fsr1.h, MIT) ----
// EASU: each output pixel from the 12 input pixels round it, a lanczos-2 lobe stretched along the local edge, clamped
// to the 2 x 2 nearest so it rings nowhere. RCAS: sharpened by the cross of its neighbours, held where that would
// overshoot. Both read what the frame wrote, tone mapped and encoded, the space FSR 1 was made for.
export const FSR = /* wgsl */`
struct Up { scale: vec2f, offset: vec2f, sharp: f32 } // input over output pixels; where output pixel 0 falls in the input; RCAS's lobe scale
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<uniform> U: Up;
@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  return vec4f(f32(vi & 1u) * 4.0 - 1.0, f32(vi >> 1u) * 4.0 - 1.0, 0.0, 1.0);
}
fn at(p: vec2i) -> vec3f { return textureLoad(src, clamp(p, vec2i(0), vec2i(textureDimensions(src)) - 1), 0).rgb; }
fn luma(c: vec3f) -> f32 { return c.b * 0.5 + (c.r * 0.5 + c.g); } // luma times 2
// One bilinear corner's share of the edge's direction and how much of an edge it is, from the cross of five lumas
//     a
//   b c d
//     e
fn edge_at(w: f32, a: f32, b: f32, c: f32, d: f32, e: f32) -> vec3f {
  let dx = d - b; let dy = e - a;
  let lx = saturate(abs(dx) / max(max(abs(d - c), abs(c - b)), 1e-8));
  let ly = saturate(abs(dy) / max(max(abs(e - c), abs(c - a)), 1e-8));
  return vec3f(dx, dy, lx * lx + ly * ly) * w;
}
fn tap(off: vec2f, dir: vec2f, len: vec2f, lob: f32, clp: f32, c: vec3f, acc: ptr<function, vec4f>) {
  let v = vec2f(off.x * dir.x + off.y * dir.y, off.y * dir.x - off.x * dir.y) * len;
  let d2 = min(dot(v, v), clp);
  // lanczos 2 without sin, rcp or sqrt: (25/16 (2/5 x^2 - 1)^2 - (25/16 - 1)) (lob x^2 - 1)^2
  var wb = 0.4 * d2 - 1.0; var wa = lob * d2 - 1.0;
  wb *= wb; wa *= wa;
  let w = (1.5625 * wb - 0.5625) * wa;
  *acc += vec4f(c * w, w);
}
@fragment fn easu(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = floor(pos.xy) * U.scale + U.offset;
  let fp = floor(q); let pp = q - fp; let i = vec2i(fp);
  //     b c
  //   e f g h
  //   i j k l
  //     n o
  let b = at(i + vec2i(0, -1)); let c = at(i + vec2i(1, -1));
  let e = at(i + vec2i(-1, 0)); let f = at(i); let g = at(i + vec2i(1, 0)); let h = at(i + vec2i(2, 0));
  let ii = at(i + vec2i(-1, 1)); let j = at(i + vec2i(0, 1)); let k = at(i + vec2i(1, 1)); let l = at(i + vec2i(2, 1));
  let n = at(i + vec2i(0, 2)); let o = at(i + vec2i(1, 2));
  let bL = luma(b); let cL = luma(c); let eL = luma(e); let fL = luma(f); let gL = luma(g); let hL = luma(h);
  let iL = luma(ii); let jL = luma(j); let kL = luma(k); let lL = luma(l); let nL = luma(n); let oL = luma(o);
  let s = edge_at((1.0 - pp.x) * (1.0 - pp.y), bL, eL, fL, gL, jL) + edge_at(pp.x * (1.0 - pp.y), cL, fL, gL, hL, kL)
    + edge_at((1.0 - pp.x) * pp.y, fL, iL, jL, kL, nL) + edge_at(pp.x * pp.y, gL, jL, kL, lL, oL);
  var dir = s.xy;
  let dr = dot(dir, dir);
  let flat = dr < 1.0 / 32768.0;
  dir = select(dir * inverseSqrt(dr), vec2f(1.0, dir.y), flat);
  let len = s.z * 0.5 * s.z * 0.5;
  // stretched from 1 along an axis to sqrt 2 on a diagonal; on an edge the window widens across it
  let stretch = dot(dir, dir) / max(abs(dir.x), abs(dir.y));
  let len2 = vec2f(1.0 + (stretch - 1.0) * len, 1.0 - 0.5 * len);
  let lob = 0.5 + (0.25 - 0.04 - 0.5) * len;
  let clp = 1.0 / lob;
  var acc = vec4f(0.0);
  tap(vec2f(0.0, -1.0) - pp, dir, len2, lob, clp, b, &acc); tap(vec2f(1.0, -1.0) - pp, dir, len2, lob, clp, c, &acc);
  tap(vec2f(-1.0, 1.0) - pp, dir, len2, lob, clp, ii, &acc); tap(vec2f(0.0, 1.0) - pp, dir, len2, lob, clp, j, &acc);
  tap(vec2f(0.0, 0.0) - pp, dir, len2, lob, clp, f, &acc); tap(vec2f(-1.0, 0.0) - pp, dir, len2, lob, clp, e, &acc);
  tap(vec2f(1.0, 1.0) - pp, dir, len2, lob, clp, k, &acc); tap(vec2f(2.0, 1.0) - pp, dir, len2, lob, clp, l, &acc);
  tap(vec2f(2.0, 0.0) - pp, dir, len2, lob, clp, h, &acc); tap(vec2f(1.0, 0.0) - pp, dir, len2, lob, clp, g, &acc);
  tap(vec2f(1.0, 2.0) - pp, dir, len2, lob, clp, o, &acc); tap(vec2f(0.0, 2.0) - pp, dir, len2, lob, clp, n, &acc);
  let lo = min(min(f, g), min(j, k)); let hi = max(max(f, g), max(j, k));
  return vec4f(clamp(acc.rgb / acc.w, lo, hi), 1.0);
}
@fragment fn rcas(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  //   b
  // d e f
  //   h
  let p = vec2i(pos.xy);
  let b = at(p + vec2i(0, -1)); let d = at(p + vec2i(-1, 0)); let e = at(p); let f = at(p + vec2i(1, 0)); let h = at(p + vec2i(0, 1));
  let lo = min(min(b, d), min(f, h)); let hi = max(max(b, d), max(f, h));
  // the most negative lobe that keeps every channel within [0, 1]
  let hit_lo = min(lo, e) / max(4.0 * hi, vec3f(1e-6)); let hit_hi = (1.0 - max(hi, e)) / min(4.0 * lo - 4.0, vec3f(-1e-6));
  let lobes = max(-hit_lo, hit_hi);
  let lobe = max(-(0.25 - 1.0 / 16.0), min(max(lobes.r, max(lobes.g, lobes.b)), 0.0)) * U.sharp;
  return vec4f((lobe * (b + d + f + h) + e) / (4.0 * lobe + 1.0), 1.0);
}
`;
