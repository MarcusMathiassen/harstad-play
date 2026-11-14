// WGSL for every pipeline. One shared prelude (the frame's uniforms, the painted lamp light, the night's two
// real lights, tone mapping, haze), then a vertex and fragment pair per kind of thing.
//
// The look is the three.js game's, reproduced term by term: Lambert under a hemisphere light and the moon,
// plus the lamp map by world position; ACES tone mapping at exposure 1.25; sRGB encoding done by hand into a
// plain (non-sRGB) target, with the haze mixed in after it, as three does. Depth is reversed (1 near, 0 far).

import { snowRelief } from './textures.js';

// For looking into the frame (?off=a,b on the page): the shaders are made with the named parts changed. Nothing is in play.
const OFF = new Set((new URL(import.meta.url).searchParams.get('off') || '').split(','));

const srgb = h => [h >> 16 & 255, h >> 8 & 255, h & 255].map(v => { v /= 255; return v < 0.04045 ? v * 0.0773993808 : Math.pow(v * 0.9478672986 + 0.0521327014, 2.4); });
const vec3 = a => `vec3f(${a.map(v => v.toFixed(6)).join(', ')})`;
const scaled = (h, k) => vec3(srgb(h).map(v => v * k));
const MOON = [0.46, 0.62, 0.64].map(v => v / Math.hypot(0.46, 0.62, 0.64)); // high in the south-east
// The snow's surface (src/world.js, snowSurface): drifts lying across the wind, one relief tile spanning so many metres
// along and across it, and their height scale; and sastrugi, the small sharp ridges the wind carves along itself.
const DRIFT = [11, 36, 0.45], RIDGE = [6, 1.7, 0.022], RELIEF_TOP = snowRelief().slope;
const RIDGE_FAR = 120, RELIEF_FAR = 700, GLINT_FAR = 350; // m from the eye: no sastrugi, no drifts, no glints beyond these
const GLINT_MIN = 0.04; // light below which no glints are looked for (faint window spill, moonlight)
// the ripple tiles' size and how many waves each sums; the size of the sky the water mirrors
export const RIPPLE_N = 128, RIPPLE_WAVES = 28, SKY_W = 256, SKY_H = 64;
const reach = L => `vec2f(${L[0].toFixed(5)}, ${L[1].toFixed(5)})`, steep = L => `vec2f(${(RELIEF_TOP * L[2] / L[0]).toFixed(5)}, ${(RELIEF_TOP * L[2] / L[1]).toFixed(5)})`;

export const PRELUDE = /* wgsl */`
struct Frame {
  view_proj: mat4x4f,
  cam_pos: vec4f,     // w: time
  cam_right: vec4f,   // scaled by tan(fov_x / 2); w: tan(fov_y / 2)
  cam_up: vec4f,      // scaled by tan(fov_y / 2); w: viewport height
  cam_fwd: vec4f,     // w: viewport width
  lamp_rect: vec4f,
  ground_rect: vec4f,
  inner: vec4f,
  car_pos: vec4f,     // w: heading
  car_dir: vec4f,     // w: speed
  car_up: vec4f,
  misc: vec4f,
  cam_vel: vec4f,     // the camera's own motion, m/s
  tail: vec4f,        // the tail lamps' light on the road: colour times strength
  shadow_vp: mat4x4f, // world to the moon's shadow map
  fog: vec4f,         // the haze: its colour as the screen shows it, density (thicker and paler inside a cloud)
  cloud: vec4f,       // the clouds' drift since they were sorted; w: how deep the eye is in one
  wash_at: vec4f,     // the TV tower's coloured light: where from; w: its reach
  wash: vec4f,        // its colour times its strength
  tunnel_at: vec4f,   // the tunnel luminaires' light on the car: where; w: its reach, 0 in the open
  tunnel: vec4f,      // its colour times its strength
  beam_vp: array<mat4x4f, 2>, // world to each headlamp's shadow map
  shadow_eye: vec4f,  // where the moon's shadow map is centred
  jet: vec4f,         // the wheel jets' glow on what is under the car: colour times strength; w: how hard they stir the air
}
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var lamp_map: texture_2d<f32>;
@group(0) @binding(2) var ground_map: texture_2d<f32>;
@group(0) @binding(3) var detail_map: texture_2d<f32>;
@group(0) @binding(4) var samp_clamp: sampler;
@group(0) @binding(5) var samp_repeat: sampler;
@group(0) @binding(6) var shadow_map: texture_depth_2d;
@group(0) @binding(7) var samp_shadow: sampler_comparison;
@group(0) @binding(8) var beam_map: texture_depth_2d_array;
@group(0) @binding(9) var relief_map: texture_2d<f32>;
@group(0) @binding(10) var tint_map: texture_2d<f32>;
@group(0) @binding(11) var<storage, read> sky_rows: SkyRows;

const PI = 3.141592653589793;
const MOON_DIR = ${vec3(MOON)};
const MOON_LIGHT = ${scaled(0xb4c2ee, 0.22)};
const SKY_LIGHT = ${scaled(0xa3a9c8, 0.21)};   // soft and only faintly blue, so lamplight reads warm
const GROUND_LIGHT = ${scaled(0x8f8f9f, 0.21)};
const HAZE_COLOUR = ${scaled(0x12151c, 1)};     // in clear air; F.fog is the haze where the eye is
const HAZE_DENSITY = 0.00016;                   // thin haze only: the far shore and mountains stay in view
const CLOUD_TINT = ${scaled(0x343945, 1)};
const EXPOSURE = 1.25;

// colours arrive as sqrt(c / scale) in rgb and scale * 16 in a (gfx_rgb in the C)
fn unpack(c: vec4f) -> vec3f { return c.rgb * c.rgb * (c.a * 15.9375); }

fn aces(c: vec3f) -> vec3f {
  let v = mat3x3f(0.59719, 0.07600, 0.02840, 0.35458, 0.90834, 0.13383, 0.04823, 0.01566, 0.83777) * (c * (EXPOSURE / 0.6));
  let a = v * (v + 0.0245786) - 0.000090537;
  let b = v * (0.983729 * v + 0.4329510) + 0.238081;
  return saturate(mat3x3f(1.60475, -0.10208, -0.00327, -0.53108, 1.10813, -0.07276, -0.07367, -0.00605, 1.07602) * (a / b));
}
fn encode(c: vec3f) -> vec3f {
  return select(pow(c, vec3f(0.41666)) * 1.055 - 0.055, c * 12.92, c <= vec3f(0.0031308));
}
// tone map, encode, and only then the haze: three's order, which the look was tuned under
fn finish(linear: vec3f, w: vec3f) -> vec3f {
  let depth = dot(w - F.cam_pos.xyz, F.cam_fwd.xyz);
  let fog = 1.0 - exp(-F.fog.a * F.fog.a * depth * depth);
  return mix(encode(aces(linear)), F.fog.rgb, fog);
}
fn display(linear: vec3f) -> vec3f { return encode(aces(linear)); }

// lights thin out with distance, as they do through cold air: full strength nearby, under half far off
fn light_fade(dist: f32) -> f32 { return mix(1.0, 0.42, smoothstep(250.0, 3000.0, dist)); }

fn lamps_at(xz: vec2f) -> vec3f {
  // the map's outermost texels are never painted, so a clamped lookup reads black off the map
  return textureSampleLevel(lamp_map, samp_clamp, clamp((xz - F.lamp_rect.xy) / F.lamp_rect.zw, vec2f(0.0), vec2f(1.0)), 0.0).rgb;
}
// Colour of the low light (the lamp map's green) at world xz, brightest channel 1: the tint (wasm/src/lamp.c, src/tint.js)
// holds its chromaticity, three metres to the texel over the lamp map's rectangle.
fn low_tint(xz: vec2f) -> vec3f {
  let c = textureSampleLevel(tint_map, samp_clamp, (xz - F.lamp_rect.xy + 1.5) / (vec2f(textureDimensions(tint_map)) * 3.0), 0.0).rg;
  let t = vec3f(c, 1.0 - c.x - c.y);
  return t / max(max(t.r, t.g), t.b);
}
fn ground_at(xz: vec2f) -> f32 {
  return textureSampleLevel(ground_map, samp_clamp, (xz - F.ground_rect.xy) / F.ground_rect.zw, 0.0).r;
}
// Painted light reaching a surface at w with normal n. Street lamps hang about seven metres up and light what
// is below them; light from windows and porches stays low. Both come from above or beside, so ground takes
// more than walls.
// In two parts: what reaches the place, whichever way the surface there faces; and the share a surface facing n takes.
fn lamp_reach(w: vec3f) -> vec3f {
  let m = lamps_at(w.xz);
  if (all(m == vec3f(0.0))) { return vec3f(0.0); } // most of the map is dark: no height to look up, no tint
  let above = w.y - ground_at(w.xz);
  var low = vec3f(0.0);
  if (m.g > 0.0 && above < 4.5) { low = low_tint(w.xz) * m.g * 1.25 * (1.0 - smoothstep(1.0, 4.5, above)); }
  return (vec3f(1.0, 0.58, 0.24) * m.r + vec3f(0.80, 0.92, 1.0) * m.b) * (1.0 - smoothstep(4.0, 9.5, above)) + low;
}
fn lamp_tilt(n: vec3f) -> f32 { return 0.5 + 0.5 * clamp(n.y * 1.4 + 0.25, 0.0, 1.0); }
fn lamp_light(w: vec3f, n: vec3f) -> vec3f { return lamp_reach(w) * lamp_tilt(n); }
// The moon's light on a surface facing n: its shadow is looked up only where the surface faces it. floor: the share that
// still arrives in shadow.
fn moon_light(w: vec3f, n: vec3f, floor_in: f32) -> vec3f {
  let facing = max(dot(n, MOON_DIR), 0.0);
  if (facing <= 0.0) { return vec3f(0.0); }
  return MOON_LIGHT * (facing * mix(moon_shadow(w, n), 1.0, floor_in));
}
// A 4 x 4 tent of shadow comparisons in four filtered fetches (src/shadows.js): its weights along each axis, [1-f, 1, 1, f],
// are two bilinear pairs, (1-f, 1) and (1, f), each fetch placed so the hardware's own weights split its pair in that ratio.
// Returns the two fetch positions (xy the first, zw the second) and leaves the weights 2-f and 1+f to the caller.
fn tent_at(uv: vec2f, size: f32) -> vec4f {
  let s = uv * size - 0.5; let i = floor(s); let f = s - i;
  return vec4f(i - 0.5 + 1.0 / (2.0 - f), i + 1.5 + f / (1.0 + f)) / size;
}
fn tent_f(uv: vec2f, size: f32) -> vec2f { return fract(uv * size - 0.5); }
// How much of the moon reaches w: 0 in shadow, 1 in the open or beyond the map's reach.
fn moon_shadow(w: vec3f, n: vec3f) -> f32 {
  let p = F.shadow_vp * vec4f(w + n * 0.35, 1.0); // stood off the surface a little, so it does not shade itself
  let uv = vec2f(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || p.z > 1.0) { return 1.0; }
  let t = tent_at(uv, 4096.0); let f = tent_f(uv, 4096.0); let wa = 2.0 - f; let wb = 1.0 + f; let z = p.z - 0.0004;
  return (wa.x * wa.y * textureSampleCompareLevel(shadow_map, samp_shadow, t.xy, z) + wb.x * wa.y * textureSampleCompareLevel(shadow_map, samp_shadow, t.zy, z)
    + wa.x * wb.y * textureSampleCompareLevel(shadow_map, samp_shadow, t.xw, z) + wb.x * wb.y * textureSampleCompareLevel(shadow_map, samp_shadow, t.zw, z)) / 9.0;
}
// A lamp at \`at\` shining along \`aim\`: three.js's spot light, falling off with distance to nothing at \`reach\`.
fn spot(w: vec3f, n: vec3f, at: vec3f, aim: vec3f, light: vec3f, reach: f32, decay: f32, cone: f32, core: f32) -> vec3f {
  let to = at - w;
  let d = length(to);
  if (d >= reach) { return vec3f(0.0); }
  let l = to / d;
  let edge = 1.0 - pow(d / reach, 4.0);
  return light * smoothstep(cone, core, dot(-l, aim)) * (edge * edge / max(pow(d, decay), 0.01)) * max(dot(n, l), 0.0);
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
// The car's own lamps on whatever is around it. Dipped halogen beams: a soft-edged main beam from each lamp, toed
// out a little and aimed at the road some way ahead, over one broad, faint wash that fills in the ground just in
// front of the bumper; and the tail lamps' red pool behind, which flares under braking and whitens in reverse.
fn car_beams(w: vec3f, n: vec3f) -> vec3f {
  let at = F.car_pos.xyz; let fwd = F.car_dir.xyz; let up = F.car_up.xyz;
  if (dot(w - at, w - at) > 85.0 * 85.0) { return vec3f(0.0); }
  let left = cross(up, fwd);
  let halogen = vec3f(1.0, 0.715694, 0.423268);
  var sum = spot(w, n, at + up * 0.72 + fwd * 2.3, normalize(up * -3.12 + fwd * 11.7), halogen * 22.0, 28.0, 1.4, 0.581683, 1.0);
  var beams = vec3f(0.0);
  for (var i = 0; i < 2; i++) {
    let s = f32(i) * 2.0 - 1.0;
    beams += spot(w, n, at + left * (s * 0.55) + up * 0.735 + fwd * 2.33, normalize(left * (s * 0.75) + up * -1.735 + fwd * 23.67), halogen * 100.0, 80.0, 1.3, 0.921061, 0.9992);
  }
  if (beams.r > 0.0) { sum += beams * beam_shadow(w, n); } // the two lamps share one map: looked up once
  return sum;
}
// the jets' flames (rocket.h) as one warm lamp under the car's middle: nothing at all unless they burn
fn car_jets(w: vec3f, n: vec3f) -> vec3f {
  let to = F.car_pos.xyz + F.car_up.xyz * 0.25 - w;
  let d = length(to);
  if (F.jet.r <= 0.0 || d >= 6.0) { return vec3f(0.0); }
  let edge = 1.0 - pow(d / 6.0, 4.0);
  return F.jet.rgb * (edge * edge / max(pow(d, 1.5), 0.1)) * (0.3 + 0.7 * max(dot(n, to / d), 0.0));
}
fn car_tail(w: vec3f, n: vec3f) -> vec3f {
  let to = F.car_pos.xyz + F.car_up.xyz * 0.6 - F.car_dir.xyz * 3.2 - w;
  let d = length(to);
  if (d >= 11.0) { return car_jets(w, n); }
  let edge = 1.0 - pow(d / 11.0, 4.0);
  return F.tail.rgb * (edge * edge / max(pow(d, 1.5), 0.01)) * max(dot(n, to / d), 0.0) + car_jets(w, n);
}
fn car_lamps(w: vec3f, n: vec3f) -> vec3f { return car_beams(w, n) + car_tail(w, n); }
// A lamp shining all round from at.xyz, as far as at.w: three.js's point light.
fn lamp_at(w: vec3f, n: vec3f, at: vec4f, light: vec3f, decay: f32) -> vec3f {
  let to = at.xyz - w;
  let d2 = dot(to, to);
  if (d2 >= at.w * at.w) { return vec3f(0.0); }
  let d = sqrt(d2);
  let edge = 1.0 - pow(d / at.w, 4.0);
  return light * (edge * edge / max(pow(d, decay), 0.01)) * max(dot(n, to / d), 0.0);
}
// The two that come and go: the colour the TV tower's lit drum throws on the hilltop and the trees around it, and
// the tunnel luminaires' light on the car.
fn tower_wash(w: vec3f, n: vec3f) -> vec3f {
  return lamp_at(w, n, F.wash_at, F.wash.rgb, 1.3) + lamp_at(w, n, F.tunnel_at, F.tunnel.rgb, 1.2);
}
// Lambert under the sky and the moon and in the car's lamps, plus the lamp map; \`gain\` is the material's share of lamp
// light. For what snow cannot lie on (the car, the tunnel's lining).
fn lit_bare(albedo: vec3f, n: vec3f, w: vec3f, gain: f32) -> vec3f {
  let sky = mix(GROUND_LIGHT, SKY_LIGHT, 0.5 * n.y + 0.5) + moon_light(w, n, 0.0) + car_lamps(w, n) + tower_wash(w, n);
  return albedo * (sky / PI + lamp_light(w, n) * (gain * 1.6) * light_fade(distance(w, F.cam_pos.xyz)));
}

// ---- the snow's surface: wind relief, and the glints on any up-facing snow (src/world.js, snowSurface) ----
const SNOW_WIND = vec2f(0.71, -0.71);  // the way the prevailing south-westerly blows, x east, z south
const SNOW_ACROSS = vec2f(0.71, 0.71);
fn max3(v: vec3f) -> f32 { return max(v.x, max(v.y, v.z)); }
// Wind relief: the slope (dh/dx, dh/dz) of the snow at world xz, from one tiling slope map read at two stretches, long
// drifts lying across the wind and fine sastrugi along it. Mip-mapping levels both out with distance; the sastrugi are
// faded out and not read at all beyond RIDGE_FAR (m from the eye), where they would be flat anyway, and the drifts
// likewise beyond RELIEF_FAR. dx, dy: how xz changes from one pixel to the next.
fn snow_slope(xz: vec2f, dist: f32, dx: vec2f, dy: vec2f) -> vec2f {
  if (dist > ${RELIEF_FAR.toFixed(1)}) { return vec2f(0.0); }
  let w = vec2f(dot(xz, SNOW_WIND), dot(xz, SNOW_ACROSS));
  let wx = vec2f(dot(dx, SNOW_WIND), dot(dx, SNOW_ACROSS)); let wy = vec2f(dot(dy, SNOW_WIND), dot(dy, SNOW_ACROSS));
  var s = (textureSampleGrad(relief_map, samp_repeat, w / ${reach(DRIFT)}, wx / ${reach(DRIFT)}, wy / ${reach(DRIFT)}).rg * 2.0 - 1.0) * ${steep(DRIFT)}
    * (1.0 - smoothstep(${(RELIEF_FAR * 0.6).toFixed(1)}, ${RELIEF_FAR.toFixed(1)}, dist));
  if (dist < ${RIDGE_FAR.toFixed(1)}) {
    s += (textureSampleGrad(relief_map, samp_repeat, w / ${reach(RIDGE)} + 0.37, wx / ${reach(RIDGE)}, wy / ${reach(RIDGE)}).rg * 2.0 - 1.0) * ${steep(RIDGE)}
      * (1.0 - smoothstep(${(RIDGE_FAR * 0.6).toFixed(1)}, ${RIDGE_FAR.toFixed(1)}, dist));
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
  let a = glint_hash(c); let b = glint_hash(c + 517.0);
  let e = (c + 0.3 + 0.4 * a.xy) * cell - xz;
  let px = vec2f(e.x * dy.y - e.y * dy.x, e.y * dx.x - e.x * dx.y) / det;
  let m = normalize(n + (b * 2.0 - 1.0) * 1.2);
  return (1.0 - smoothstep(0.4, 1.2, length(px) / F.misc.z)) * step(a.z, 0.35) * vec2f(smoothstep(0.92, 1.0, dot(m, hl)), smoothstep(0.92, 1.0, dot(m, v)));
}
// Glints: snow is loose ice grains, and at any moment a few have a facet turned to mirror a light into the eye. They
// twinkle as the eye moves and stand still when it does. lamp: the painted lamplight here; head: the direct light (the
// headlamps; the moon is too dim to count). Cells stay at least five CSS pixels wide by doubling in size with distance,
// cross-fading between two sizes, so no grain is ever smaller than a pixel. They fade out towards GLINT_FAR, past which
// none are looked for.
fn snow_glints(w: vec3f, n: vec3f, lamp: vec3f, head: vec3f, dx: vec2f, dy: vec2f) -> vec3f {
  let head_amt = smoothstep(0.08, 0.4, max3(head));
  let det = dx.x * dy.y - dx.y * dy.x;
  let dist = distance(F.cam_pos.xyz, w);
  if (dist > ${GLINT_FAR.toFixed(1)} || max3(lamp) + head_amt < ${GLINT_MIN} || abs(det) < 1e-12) { return vec3f(0.0); }
  let v = (F.cam_pos.xyz - w) / dist; let hl = normalize(v + vec3f(0.0, 1.0, 0.0));
  let level = max(0.0, log2(166.66667 * F.misc.z * max(length(dx), length(dy))));
  let l0 = floor(level); let t = level - l0; let cell = 0.03 * exp2(l0);
  var g = glint_level(w.xz, cell, dx, dy, det, n, v, hl) * (1.0 - t);
  if (t > 0.02) { g += glint_level(w.xz, cell * 2.0, dx, dy, det, n, v, hl) * t; }
  return (lamp * g.x + head * head_amt * g.y) * 7.0 * (1.0 - smoothstep(${(GLINT_FAR * 0.6).toFixed(1)}, ${GLINT_FAR.toFixed(1)}, dist));
}
// The same light on something snow may lie on. cover: how much of the fragment is lying snow, or -1 to judge it from
// the colour (pale and bluish, as the snow colours are); either way only what faces up holds snow. relief: the share
// of the wind relief shown. dx, dy: how the world xz changes from one pixel to the next. moon_floor: the share of
// moonlight that still arrives in the moon's shadow (a tree's crown scatters it; anything else takes none).
fn lit_under(albedo: vec3f, n0: vec3f, w: vec3f, gain: f32, cover_in: f32, relief: f32, dx: vec2f, dy: vec2f, moon_floor: f32) -> vec3f {
  return lit_reached(albedo, n0, w, gain, cover_in, relief, dx, dy, moon_floor, lamp_reach(w));
}
// The same for a caller that has the painted light at w already (lamp_reach), having a use of its own for it.
fn lit_reached(albedo: vec3f, n0: vec3f, w: vec3f, gain: f32, cover_in: f32, relief: f32, dx: vec2f, dy: vec2f, moon_floor: f32, reach: vec3f) -> vec3f {
  var cover = cover_in;
  if (cover < 0.0) { cover = smoothstep(0.42, 0.6, min(albedo.r, min(albedo.g, albedo.b))) * smoothstep(-0.03, 0.02, albedo.b - albedo.r); }
  cover *= smoothstep(0.5, 0.75, n0.y);
  var n = n0;
  if (cover > 0.0) { // the wind relief tilts the normal before any light is gathered
    let slope = snow_slope(w.xz, distance(w, F.cam_pos.xyz), dx, dy) * cover * relief;
    n = normalize(n0 - vec3f(slope.x, 0.0, slope.y));
  }
  let direct = moon_light(w, n, moon_floor) + car_lamps(w, n) + tower_wash(w, n);
  let lamp_in = reach * lamp_tilt(n) * (gain * 1.6) * light_fade(distance(w, F.cam_pos.xyz));
  var c = albedo * ((mix(GROUND_LIGHT, SKY_LIGHT, 0.5 * n.y + 0.5) + direct) / PI + lamp_in);
  if (cover > 0.0) { c += snow_glints(w, n, lamp_in, direct / PI, dx, dy) * cover; }
  return c;
}
fn lit_snow(albedo: vec3f, n0: vec3f, w: vec3f, gain: f32, cover_in: f32, relief: f32, dx: vec2f, dy: vec2f) -> vec3f {
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
// of it is a line of values, mixed along y once a frame (SKY_MAPS' rows) and then only along x. Six rows, a curtain's
// wander and its rays each, of five octaves; |p.x| is within 4 wherever a curtain is looked at, and a slot holds the
// 1024 lattice points that span at the finest octave.
struct SkyRows { origin: array<i32, 32>, g: array<f32, ${30 * 1024}> }
fn row_noise(slot: u32, x: f32) -> f32 {
  let i = floor(x); var f = x - i;
  f = f * f * (3.0 - 2.0 * f);
  let at = slot * 1024u + u32(clamp(i32(i) - sky_rows.origin[slot], 0, 1022));
  return mix(sky_rows.g[at], sky_rows.g[at + 1u], f);
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
  // aurora: curtains hanging over the northern sky, with fine vertical rays
  let p = d.xz / (d.y + 0.22);
  var glow = 0.0; var fringe = 0.0;
  for (var i = 0; i < 3; i++) {
    let fi = f32(i);
    let c0 = p.y + 0.9 + fi * 0.62;
    // The noise is most of this shader's cost and a curtain is a thin band of the sky: skip it where the band
    // cannot reach (the wander below is within +-0.85), and its rays where it has faded below 1e-4.
    if (c0 > 1.65 || c0 < -1.85 || d.y < 0.03) { continue; }
    // past c of 0.79 above or -1.02 below, the curtain is under 1e-4: the bounded sum stops there
    let c = c0 + (fbm_row(u32(i) * 2u, p.x * 0.55 + fi * 7.3, 0.5 - (1.02 + c0) / 1.7, 0.5 + (0.79 - c0) / 1.7) - 0.5) * 1.7;
    let cur = exp(-c * c * 15.0);
    let fr = exp(-(c + 0.2) * (c + 0.2) * 14.0);
    if (cur + fr < 1e-4) { continue; }
    var rays = 0.72;
    if (fine > 0.0) { rays = 0.45 + 0.55 * fbm_row(u32(i) * 2u + 1u, p.x * 7.0 + time * 0.05, -1.0, 2.0); }
    glow += cur * rays; fringe += fr * rays;
  }
  let pulse = 0.75 + 0.25 * sin(time * 0.11);
  let am = smoothstep(0.03, 0.3, d.y) * pulse;
  // thin snow cloud drifting through, faintly lit from below
  let cp = d.xz / (h + 0.14) * 1.2 + vec2f(time * 0.006, time * 0.002);
  var cl = 0.0;
  if (d.y > 0.0) { cl = smoothstep(0.52, 0.86, fbm_band(cp * vec2f(0.8, 1.3), 0.52, 0.86)) * smoothstep(0.0, 0.12, d.y); }
  return vec4f((vec3f(0.10, 0.85, 0.42) * glow + vec3f(0.45, 0.18, 0.75) * fringe * 0.55) * am * 0.085, cl);
}
// The rest of the sky over that noise: the gradient, the stars, the moon.
fn sky_over(d: vec3f, fine: f32, moon: vec3f, noise: vec4f) -> vec3f {
  let time = F.cam_pos.w;
  let h = max(d.y, 0.0);
  var col = mix(vec3f(0.009, 0.012, 0.022), vec3f(0.003, 0.0045, 0.012), pow(h, 0.5));
  col += vec3f(0.02, 0.012, 0.006) * pow(1.0 - h, 12.0); // the town's warm glow on the haze: faint, and hugging the horizon

  if (fine > 0.0) {
    let sp = d * 240.0;
    let sh = hash3(floor(sp));
    var star = step(0.9962, sh) * (1.0 - smoothstep(0.0, 0.42, length(fract(sp) - 0.5)));
    star *= 0.6 + 0.4 * sin(time * (1.0 + sh * 3.0) + sh * 90.0);
    col += vec3f(0.85, 0.9, 1.0) * star * (0.22 + fract(sh * 77.0) * 0.8) * smoothstep(0.0, 0.2, d.y);
  }
  col += noise.rgb;

  let to_moon = max(dot(d, moon), 0.0);
  col += vec3f(0.55, 0.66, 0.9) * (pow(to_moon, 40.0) * 0.05 + pow(to_moon, 700.0) * 0.5);
  col += vec3f(1.0, 0.97, 0.9) * smoothstep(0.99962, 0.99975, to_moon) * 2.6 * fine;
  return mix(col, vec3f(0.036, 0.038, 0.048), noise.a * 0.72);
}
`;

/* ---- terrain: no vertex buffer. One strip of cells is drawn once per grid row; heights come from a texture ---- */

// cut: a tunnel portal opens this ground. clip: the town is cut out of this piece of the backdrop. Only the chunks a portal
// reaches, and the backdrop's that the town's edge crosses (wasm/src/terrain.c), take a shader with a discard in it: on a
// tile-based GPU a discard costs every fragment of a shader that has one, whether it fires or not (src/tunnels.js, cuts).
export const terrain = (cut, clip) => /* wgsl */`
struct Grid {
  origin: vec2f,      // world position of grid point (0, 0)
  step: f32,
  clip: f32,          // 1: the backdrop
  size: vec2u,        // grid points
  extent: vec2f,      // width and depth the tile images cover
  tiles: vec2u,
  pad: vec2u,
  ex: array<vec4f, 2>, // tile edges in grid columns, and rows
  ez: array<vec4f, 2>,
  cut_a: array<vec4f, 4>, // tunnel portals, where the ground is opened: mouth x, z, direction in
  cut_b: array<vec4f, 4>, // side from, side to, top at the mouth, slope
}
@group(1) @binding(0) var<uniform> G: Grid;
@group(1) @binding(1) var heights: texture_2d<f32>;
@group(1) @binding(2) var photos: texture_2d_array<f32>;
@group(1) @binding(3) var samp_photo: sampler;

struct V { @builtin(position) pos: vec4f, @location(0) w: vec3f, @location(1) n: vec3f }

fn height(i: vec2i) -> f32 { return textureLoad(heights, clamp(i, vec2i(0), vec2i(G.size) - 1), 0).r; }

@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) row: u32) -> V {
  let i = vec2i(i32(vi >> 1u), i32(row + (vi & 1u)));
  let w = vec3f(G.origin.x + f32(i.x) * G.step, height(i), G.origin.y + f32(i.y) * G.step);
  let dx = height(i - vec2i(1, 0)) - height(i + vec2i(1, 0));
  let dz = height(i - vec2i(0, 1)) - height(i + vec2i(0, 1));
  var o: V;
  o.pos = F.view_proj * vec4f(w, 1.0);
  o.w = w;
  o.n = normalize(vec3f(dx, 2.0 * G.step, dz));
  return o;
}

fn edge(a: array<vec4f, 2>, k: u32) -> f32 { return a[k >> 2u][k & 3u]; }

@fragment fn fs(v: V) -> @location(0) vec4f {
  let w = v.w;
  let ddx = dpdx(w.xz); let ddy = dpdy(w.xz);
${clip ? '  if (w.x > F.inner.x && w.z > F.inner.y && w.x < F.inner.z && w.z < F.inner.w) { discard; return vec4f(0.0); } // and nothing more is worked out for it\n' : ''}${cut ? `  for (var i = 0; i < 4; i++) { // inside a portal's shell the tunnel is the ground
    let q = w.xz - G.cut_a[i].xy;
    let along = dot(q, G.cut_a[i].zw); let side = dot(q, vec2f(-G.cut_a[i].w, G.cut_a[i].z));
    if (along > 0.0 && along < 16.0 && side > G.cut_b[i].x && side < G.cut_b[i].y && w.y < G.cut_b[i].z + G.cut_b[i].w * along) { discard; return vec4f(0.0); }
  }
` : ''}  // which photo tile this is, and where in it: tiles meet at grid lines, each stretched over its own span
  let g = (w.xz - G.origin) / G.step;
  var cx = 0u; var cz = 0u;
  for (var k = 1u; k < G.tiles.x; k++) { if (g.x >= edge(G.ex, k)) { cx = k; } }
  for (var k = 1u; k < G.tiles.y; k++) { if (g.y >= edge(G.ez, k)) { cz = k; } }
  let lo = G.origin + vec2f(edge(G.ex, cx), edge(G.ez, cz)) * G.step;
  let hi = min(G.origin + vec2f(edge(G.ex, cx + 1u), edge(G.ez, cz + 1u)) * G.step, G.origin + G.extent);
  let photo = textureSampleGrad(photos, samp_photo, (w.xz - lo) / (hi - lo), cz * G.tiles.x + cx, ddx / (hi - lo), ddy / (hi - lo)).rgb;

  // The summer aerial photo still shapes the snow: open ground is bright, forest canopy and built-up clutter
  // stay darker, so fields, woods and plots remain readable from above.
  let lum = dot(photo, vec3f(0.30, 0.59, 0.11));
  let canopy = clamp((photo.g - max(photo.r, photo.b)) * 9.0, 0.0, 1.0) * (1.0 - smoothstep(0.05, 0.16, lum));
  var grain = 0.0; // the backdrop shows none
  if (G.clip < 1.0) {
    grain = textureSampleGrad(detail_map, samp_repeat, w.xz * 0.37, ddx * 0.37, ddy * 0.37).r * 0.55
      + textureSampleGrad(detail_map, samp_repeat, w.xz * 0.051, ddx * 0.051, ddy * 0.051).r * 0.45;
  }
  var snow = vec3f(0.84, 0.89, 0.97) * (0.70 + 0.34 * smoothstep(0.0, 0.3, lum));
  snow = mix(snow, vec3f(0.30, 0.35, 0.41), canopy * 0.62);
  snow *= mix(0.86 + 0.28 * grain, 1.0, G.clip);
  // all of it is snow; under the trees it lies thinner and sheltered
  return vec4f(finish(lit_snow(snow, normalize(v.n), w, 1.0, 1.0 - canopy * 0.7, 1.0, ddx, ddy), w), 1.0);
}
`;

/* ---- static meshes ---- */

const MESH_IN = /* wgsl */`
struct VIn { @location(0) pos: vec3f, @location(1) nrm: vec4f, @location(2) col: vec4f }
struct V { @builtin(position) pos: vec4f, @location(0) w: vec3f, @location(1) n: vec3f, @location(2) col: vec3f, @location(3) @interpolate(flat) flag: f32 }
`;

// vertex colour, lamp-lit; flag 1 is unlit (lit windows, porch lamps, lenses), 2 unlit and above the haze (lights in the sky)
export const PLAIN = MESH_IN + /* wgsl */`
@vertex fn vs(i: VIn) -> V {
  var o: V;
  o.pos = F.view_proj * vec4f(i.pos, 1.0);
  o.w = i.pos; o.n = i.nrm.xyz; o.col = unpack(i.col); o.flag = round(i.nrm.w * 127.0);
  return o;
}
@fragment fn fs(v: V) -> @location(0) vec4f {
  let shaded = lit(v.col, normalize(v.n), v.w, 1.0);
  if (v.flag > 1.5) { return vec4f(display(v.col), 1.0); }
  return vec4f(finish(select(shaded, v.col, v.flag > 0.5), v.w), 1.0);
}
`;

// What moves in the sky (struct gfx_live in the C): the tower's lights, Father Christmas and his team. The mesh is sent
// once, each part in its own space; a vertex names the slot that places it and the tint that colours it, and those are
// all that change. None of it is lit or hazed: it shows as bright as it is, right across town.
export const LIVE_SLOTS = 104, LIVE_TINTS = 16; // GFX_LIVE_SLOTS, GFX_LIVE_TINTS
export const LIVE = /* wgsl */`
struct Live { m: array<vec4f, ${LIVE_SLOTS * 3}>, tint: array<vec4f, ${LIVE_TINTS}> }
@group(1) @binding(0) var<uniform> L: Live;
struct VIn { @location(0) pos: vec3f, @location(1) nrm: vec4f, @location(2) col: vec4f }
struct V { @builtin(position) pos: vec4f, @location(0) col: vec3f }
@vertex fn vs(i: VIn) -> V {
  var o: V;
  let s = i32(round(i.nrm.x * 127.0)) * 3; let p = vec4f(i.pos, 1.0);
  o.pos = F.view_proj * vec4f(dot(L.m[s], p), dot(L.m[s + 1], p), dot(L.m[s + 2], p), 1.0);
  let tint = L.tint[i32(round(i.nrm.y * 127.0))];
  o.col = select(unpack(i.col), tint.rgb, tint.a > 0.5);
  return o;
}
@fragment fn fs(v: V) -> @location(0) vec4f { return vec4f(display(v.col), 1.0); }
`;

// What floats (src/water.js): moored boats and floating pontoons ride the same swell the sea's surface shows. A boat
// moves as one: heave and the swell's tilt where it lies, a slow roll and pitch of its own, and a slow yaw and surge as
// it tugs at its lines. A pontoon heaves vertex by vertex with the swell under it, so a run of them flexes as one jetty.
const BOB = { roll: [0.014, 3.4], pitch: [0.006, 5.1], yaw: [0.008, 23], surge: [0.06, 17], heave: [0.012, 2.6] }; // rad or m; period s
const per = b => (2 * Math.PI / b[1]).toFixed(5);
const afloat = sea => MESH_IN + sea + /* wgsl */`
fn swell_at(p: vec2f) -> f32 {
  var h = 0.0;
  for (var i = 0; i < 3; i++) { h += S.swell[i].w * sin(dot(S.swell[i].xy, p) + S.swell[i].z); }
  return h;
}
// boat: its centre x, z, heading (rad), length (m); all zero on a pontoon
@vertex fn vs(i: VIn, @location(4) boat: vec4f) -> V {
  var pos = i.pos; var n = i.nrm.xyz;
  if (boat.w > 0.0) { // small angles, so a turn is applied as v + turn x v
    let c = boat.xy; let t = F.cam_pos.w;
    let ph = fract(sin(dot(c, vec2f(12.9898, 78.233))) * 43758.5453) * 6.2831853;
    let fwd = vec3f(sin(boat.z), 0.0, cos(boat.z)); let beam = vec3f(fwd.z, 0.0, -fwd.x);
    let g = swell_slope(c) / (1.0 + boat.w / 30.0); // a longer hull bridges the swell
    let roll = ${BOB.roll[0]} * sin(t * ${per(BOB.roll)} + ph) + dot(g, beam.xz);
    let pitch = ${BOB.pitch[0]} * sin(t * ${per(BOB.pitch)} + ph * 1.7) - dot(g, fwd.xz);
    let yaw = ${BOB.yaw[0]} * sin(t * ${per(BOB.yaw)} + ph * 2.3);
    let turn = fwd * roll + beam * pitch + vec3f(0.0, yaw, 0.0);
    let shift = fwd * ${BOB.surge[0]} * sin(t * ${per(BOB.surge)} + ph * 0.7) + vec3f(0.0, swell_at(c) + ${BOB.heave[0]} * sin(t * ${per(BOB.heave)} + ph * 3.1), 0.0);
    n += cross(turn, n);
    pos += cross(turn, pos - vec3f(boat.x, 0.0, boat.y)) + shift;
  } else { pos.y += swell_at(pos.xz); }
  var o: V;
  o.pos = F.view_proj * vec4f(pos, 1.0);
  o.w = pos; o.n = n; o.col = unpack(i.col); o.flag = 0.0;
  return o;
}
@fragment fn fs(v: V) -> @location(0) vec4f { return vec4f(finish(lit(v.col, normalize(v.n), v.w, 1.0), v.w), 1.0); }
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
  var n = normalize(cross(dpdx(v.w), dpdy(v.w)));
  if (dot(n, F.cam_pos.xyz - v.w) < 0.0) { n = -n; }
  return vec4f(finish(lit(v.col, n, v.w, 0.9), v.w), 1.0);
}
`;

// ---- the forest (wasm/src/tree.c, wasm/src/forest.c; src/trees.js) ----
// Every model is a triangle list in one float texture, two texels per vertex, one row per variant and level of
// detail; a tree is eight floats (struct tree_inst) in one buffer, and its vertices are fetched by vertex index. Which
// trees are drawn, and at which level, is listed every frame by the C (forest.h): an instance is one word, the tree
// and its level, and each draw runs over exactly one model's vertices. A tree wholly at one level is drawn by FOREST,
// which discards nothing; one part-way between two is listed for both and drawn by FOREST_FADE, which dithers them
// against each other; the twigs of far trees by FOREST_TWIG.
const TREE_NEAR_M = 120, TREE_NEAR_FADE = 12, TREE_MID_M = 700, TREE_MID_FADE = 60; // as wasm/include/harstad/tree.h
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
  let row = i32(i.form.w) * 3 + level;
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
  @location(4) k: vec4f } // k: how much of this level shows (negative while it fades in), alpha, mark, glow
`;
// How much of a tree shows at its level, d metres from the eye: 1 in the level's own range, falling to 0 across the
// hand-over at each end (negative while it fades in, so the two levels take complementary pixels).
const forestFade = /* wgsl */`
fn tree_shows(level: i32, d: f32) -> f32 {
  let near = smoothstep(${(TREE_NEAR_M - TREE_NEAR_FADE).toFixed(1)}, ${(TREE_NEAR_M + TREE_NEAR_FADE).toFixed(1)}, d);
  let far = smoothstep(${(TREE_MID_M - TREE_MID_FADE).toFixed(1)}, ${(TREE_MID_M + TREE_MID_FADE).toFixed(1)}, d);
  if (level == 0) { return 1.0 - near; }
  if (level == 2) { return -far; }
  return select(1.0 - far, -near, near < 1.0);
}
`;
// kind: 'solid' and 'twig' are trees wholly at their level; 'fade' is one handing over; 'haze' the blended part of
// near and mid trees, which only fades out towards the far level.
const forestVertex = kind => FOREST_VS + (kind === 'fade' ? forestFade : '') + /* wgsl */`
@vertex fn vs(i: TIn) -> TV {
  var o: TV;
  let tree = trees[i.pick & 0xffffffu];
  let level = i32(i.pick >> 24u);
  let t = tree_vertex(i.vi, tree, level);
  o.w = t.w; o.local = t.t0.xyz;
  o.n = tree_turn(tree_bytes(t.t1.x) * 2.0 - 1.0, tree.at.w);
  ${kind === 'fade' ? 'let shows = tree_shows(level, distance(tree.at.xyz, F.cam_pos.xyz));'
    : kind === 'haze' ? `let shows = 1.0 - smoothstep(${(TREE_MID_M - TREE_MID_FADE).toFixed(1)}, ${(TREE_MID_M + TREE_MID_FADE).toFixed(1)}, distance(tree.at.xyz, F.cam_pos.xyz));`
    : 'let shows = 1.0;'}
  o.k = vec4f(shows, t.flag.x, tree_bytes(t.t1.z).x, t.flag.z);
  let c = tree_bytes(t.t0.w);
  o.col = c * c * (0.78 + 0.4 * fract(tree.form.w));
  o.pos = F.view_proj * vec4f(t.w, 1.0);
  ${kind === 'fade' || kind === 'haze' ? 'if (shows == 0.0) { o.pos = vec4f(0.0, 0.0, 0.0, 1.0); } // at the very end of a hand-over: nothing to draw' : ''}
  return o;
}
`;
// How much of a pixel of a bare crown's shell is twigs: densest where it is seen face-on and fading out at its
// outline. Fine streaks fanning out from the stem give the grain; far off the texture's mip levels even them out.
const FOREST_TWIG_AT = /* wgsl */`
fn tree_twigs(v: TV) -> f32 {
  let grain = textureSample(detail_map, samp_repeat, vec2f(atan2(v.local.z, v.local.x) * 0.3 + v.local.y * 0.004, length(v.local.xz) * 0.012 + v.local.y * 0.006)).r;
  let face = abs(dot(normalize(v.n), normalize(F.cam_pos.xyz - v.w)));
  return (0.25 + 1.5 * smoothstep(0.44, 0.6, grain)) * face * face;
}
`;
// Solid trees: birch lenticels, frost that glints in lamplight, glowing bulbs. One flat shade per face. Between levels
// ('fade'), a dithered hand-over: complementary patterns, so the two levels never both drop a pixel. Far off, the twig
// haze is drawn as the solid trees are, screen-door see-through ('twig', and inside a far tree's run while it fades
// in): no blending. Only those two discard; the rest have their depth settled before they are shaded.
const forest = kind => forestVertex(kind) + (kind === 'solid' ? '' : FOREST_TWIG_AT) + /* wgsl */`
@fragment fn fs(v: TV) -> @location(0) vec4f {
  var n = normalize(cross(dpdx(v.w), dpdy(v.w)));
  let to_eye = normalize(F.cam_pos.xyz - v.w);
  if (dot(n, to_eye) < 0.0) { n = -n; }
  let grain = textureSample(detail_map, samp_repeat, vec2f(atan2(v.local.z, v.local.x) * 0.012, v.local.y * 0.24)).r;
  let dx = dpdx(v.w.xz); let dy = dpdy(v.w.xz);
${kind === 'solid' ? '' : /* wgsl */`
  var bayer = array<f32, 16>(0.0, 8.0, 2.0, 10.0, 12.0, 4.0, 14.0, 6.0, 3.0, 11.0, 1.0, 9.0, 15.0, 7.0, 13.0, 5.0);
  let q = vec2i(v.pos.xy) & vec2i(3);
  let b = (bayer[q.x + q.y * 4] + 0.5) / 16.0;
  let q2 = (vec2i(v.pos.xy) + vec2i(1, 2)) & vec2i(3);
  let twigs = tree_twigs(v);
  // a pixel that is dropped is not lit first: discard alone would leave the rest to run
  ${kind === 'fade' ? 'if ((v.k.x >= 0.0 && b >= v.k.x) || (v.k.x < 0.0 && b < 1.0 + v.k.x)) { discard; return vec4f(0.0); }' : ''}
  if (v.k.y < 0.99 && v.k.y * twigs < (bayer[q2.x + q2.y * 4] + 0.5) / 16.0) { discard; return vec4f(0.0); }`}
  let albedo = v.col * (1.0 - 0.8 * v.k.z * (1.0 - smoothstep(0.37, 0.43, grain))); // dark lenticels in short horizontal dashes
  let reach = lamp_reach(v.w);
  var c = lit_reached(albedo, n, v.w, 0.9, -1.0, 1.0, dx, dy, ${TREE_MOON_SHADE}, reach);
  let frost = smoothstep(0.25, 0.55, dot(v.col, vec3f(0.333)));
  let rim = pow(1.0 - abs(dot(n, to_eye)), 3.0);
  c += v.col * (frost * rim * 1.6) * (reach * lamp_tilt(n)) + v.col * (v.k.w * 1.3);
  return vec4f(finish(c, v.w), 1.0);
}
`;
export const FOREST = forest('solid'), FOREST_FADE = forest('fade'), FOREST_TWIG = forest('twig');
// Twig haze (near and mid): a soft, blended shell, lit as a whole rather than facet by facet. Levels simply
// cross-fade. The snow at the foot rides along.
export const FOREST_HAZE = forestVertex('haze') + FOREST_TWIG_AT + /* wgsl */`
@fragment fn fs(v: TV) -> @location(0) vec4f {
  let twigs = tree_twigs(v);
  var a = v.k.y * abs(v.k.x);
  if (v.k.z <= 0.5) { a *= twigs; }
  let c = lit_under(v.col, vec3f(0.0, 1.0, 0.0), v.w, 0.9, -1.0, 1.0, dpdx(v.w.xz), dpdy(v.w.xz), ${TREE_MOON_SHADE});
  return vec4f(finish(c, v.w), saturate(a));
}
`;
// Shadows: each tree casts from exactly one level, the one it is drawn with at the middle of a hand-over (far trees
// cast none); the C lists them for each map by the map's own eye: the camera for the headlamps' maps, where the moon's
// map was last centred for that.
export const FOREST_SHADOW = /* wgsl */`
struct Cast { vp: mat4x4f, ground_rect: vec4f }
@group(0) @binding(0) var<uniform> C: Cast;
@group(1) @binding(0) var tree_verts: texture_2d<f32>;
@group(1) @binding(1) var ground_map: texture_2d<f32>;
@group(1) @binding(2) var samp_clamp: sampler;
@group(1) @binding(3) var<storage, read> trees: array<Tree>;
fn tree_ground(xz: vec2f) -> f32 { return textureSampleLevel(ground_map, samp_clamp, (xz - C.ground_rect.xy) / C.ground_rect.zw, 0.0).r; }
` + FOREST_IN + /* wgsl */`
@vertex fn thrown(i: TIn) -> @builtin(position) vec4f { return C.vp * vec4f(tree_vertex(i.vi, trees[i.pick & 0xffffffu], i32(i.pick >> 24u)).w, 1.0); }
`;

// lamp posts: unlit, each head's colour is its glow
export const LAMP = MESH_IN + INSTANCED + /* wgsl */`
@vertex fn vs(i: VIn, inst: IIn) -> V {
  var o: V;
  let w = place(i.pos, inst);
  o.pos = F.view_proj * vec4f(w, 1.0);
  o.w = w; o.n = vec3f(0.0, 1.0, 0.0); o.col = unpack(i.col) * unpack(inst.tint); o.flag = 1.0;
  return o;
}
@fragment fn fs(v: V) -> @location(0) vec4f { return vec4f(finish(v.col, v.w), 1.0); }
`;

// walls: a facade texture whose alpha says where the wall colour may tint, and a second that glows. Layers 0 and 1 are
// timber and block; from 2 up, the centre's facade styles, which are drawn at half the size and so have arrays of their own.
export const WALL = /* wgsl */`
@group(1) @binding(0) var facades: texture_2d_array<f32>;
@group(1) @binding(1) var glows: texture_2d_array<f32>;
@group(1) @binding(2) var samp_wall: sampler;
@group(1) @binding(3) var styles: texture_2d_array<f32>;
@group(1) @binding(4) var style_glows: texture_2d_array<f32>;
struct VIn { @location(0) pos: vec3f, @location(1) nrm: vec4f, @location(2) col: vec4f, @location(3) uv: vec2f }
struct V { @builtin(position) pos: vec4f, @location(0) w: vec3f, @location(1) n: vec3f, @location(2) col: vec3f, @location(3) uv: vec2f, @location(4) @interpolate(flat) layer: u32 }
@vertex fn vs(i: VIn) -> V {
  var o: V;
  o.pos = F.view_proj * vec4f(i.pos, 1.0);
  o.w = i.pos; o.n = i.nrm.xyz; o.col = unpack(i.col); o.uv = i.uv; o.layer = u32(round(i.nrm.w * 127.0));
  return o;
}
@fragment fn fs(v: V) -> @location(0) vec4f {
  let dx = dpdx(v.uv); // taken here, where every pixel passes: the layer picks the array below
  let dy = dpdy(v.uv);
  var t: vec4f; var glow: vec3f; var blurred: vec3f;
  if (v.layer < 2u) {
    t = textureSampleGrad(facades, samp_wall, v.uv, v.layer, dx, dy);
    glow = textureSampleGrad(glows, samp_wall, v.uv, v.layer, dx, dy).rgb;
    blurred = textureSampleLevel(glows, samp_wall, v.uv, v.layer, 4.5).rgb;
  } else {
    t = textureSampleGrad(styles, samp_wall, v.uv, v.layer - 2u, dx, dy);
    glow = textureSampleGrad(style_glows, samp_wall, v.uv, v.layer - 2u, dx, dy).rgb;
    blurred = textureSampleLevel(style_glows, samp_wall, v.uv, v.layer - 2u, 4.5).rgb;
  }
  let albedo = t.rgb * mix(vec3f(1.0), v.col, t.a); // tint the cladding but leave window frames and glass alone
  // a lit window also lights its own frame, sill and the wall just round it: its glow, blurred, falling on them
  let reveal = albedo * blurred * 0.3;
  let c = lit(albedo, normalize(v.n), v.w, 0.5) + (glow * 1.45 + reveal) * light_fade(distance(v.w, F.cam_pos.xyz));
  return vec4f(finish(c, v.w), 1.0);
}
`;

// road ribbons, blended over the terrain; the texture's alpha feathers the verge into the snow
export const ROAD = /* wgsl */`
@group(1) @binding(0) var surfaces: texture_2d_array<f32>;
@group(1) @binding(1) var samp_road: sampler;
struct VIn { @location(0) pos: vec3f, @location(1) nrm: vec4f, @location(2) col: vec4f, @location(3) uv: vec2f }
struct V { @builtin(position) pos: vec4f, @location(0) w: vec3f, @location(1) col: vec3f, @location(2) uv: vec2f, @location(3) @interpolate(flat) layer: u32 }
@vertex fn vs(i: VIn) -> V {
  var o: V;
  o.pos = F.view_proj * vec4f(i.pos, 1.0);
  o.w = i.pos; o.col = unpack(i.col); o.uv = i.uv; o.layer = u32(round(i.nrm.w * 127.0));
  return o;
}
@fragment fn fs(v: V) -> @location(0) vec4f {
  let t = textureSample(surfaces, samp_road, v.uv, v.layer);
  // traffic has worn the drifts off the carriageway
  return vec4f(finish(lit_snow(t.rgb * v.col, vec3f(0.0, 1.0, 0.0), v.w, 1.0, -1.0, 0.3, dpdx(v.w.xz), dpdy(v.w.xz)), v.w), t.a);
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
export const CAR = /* wgsl */`
struct Mat { albedo: vec4f, emissive: vec4f, params: vec4f } // rgb + opacity; kind in a; roughness, metalness, mirror strength, picture or lamp-shy
struct Car { parts: array<mat4x4f, 8>, mats: array<Mat, 16>, lamps: array<vec4f, 3>, glow_at: array<vec4f, 3>, glow: array<vec4f, 3>, needle: vec4f, dash: vec4f }
@group(1) @binding(0) var<uniform> C: Car;
@group(1) @binding(1) var plate: texture_2d<f32>;
@group(1) @binding(2) var dials: texture_2d_array<f32>; // the cluster, 1024 x 320: as it stands, and with its lamps lit
@group(1) @binding(3) var panels: texture_2d<f32>;
struct VIn { @location(0) pos: vec3f, @location(1) nrm: vec4f, @location(2) col: vec3f, @location(3) uv: vec2f, @location(4) tag: vec4u }
struct V { @builtin(position) pos: vec4f, @location(0) w: vec3f, @location(1) n: vec3f, @location(2) uv: vec2f, @location(3) @interpolate(flat) mat: u32, @location(4) col: vec3f }
@vertex fn vs(i: VIn) -> V {
  var o: V;
  let m = C.parts[u32(round(i.nrm.w * 127.0))];
  o.mat = i.tag.x;
  var w = (m * vec4f(i.pos, 1.0)).xyz;
  let kind = u32(C.mats[o.mat].emissive.a + 0.5);
  if (kind == 5u) { // glare: a square that faces the eye
    w += normalize(F.cam_right.xyz) * (i.uv.x * C.mats[o.mat].params.x) + normalize(F.cam_up.xyz) * (i.uv.y * C.mats[o.mat].params.y);
  }
  o.pos = F.view_proj * vec4f(w, 1.0);
  o.w = w; o.n = (m * vec4f(i.nrm.xyz, 0.0)).xyz; o.uv = i.uv;
  o.col = i.col;
  if (kind == 1u) { o.col *= C.lamps[i.tag.y / 4u][i.tag.y % 4u]; } // a lens: as bright as its lamp burns
  return o;
}
// What lights the car: the sky, the moon, its own beams; and, where they are let, the lamps that shine all round (the
// tail lamps' pool on the road, the TV tower, a tunnel's luminaires). The cabin and the wheels are left out of those.
fn car_light(w: vec3f, n: vec3f, shy: bool) -> vec3f {
  var sum = mix(GROUND_LIGHT, SKY_LIGHT, 0.5 * n.y + 0.5) + moon_light(w, n, 0.0) + car_beams(w, n);
  if (!shy) { sum += car_tail(w, n) + tower_wash(w, n); }
  return sum / PI;
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
// out: colour times opacity, and opacity (glare has none: it only adds)
@fragment fn fs(v: V) -> @location(0) vec4f {
  let m = C.mats[v.mat];
  let kind = u32(m.emissive.a + 0.5);
  let which = u32(m.params.w + 0.5);
  var picture = vec3f(0.0); // only a material that shows one looks its picture up
  if (kind == 3u || kind == 6u) {
    if (which == 0u) { picture = textureSampleLevel(plate, samp_clamp, v.uv, 0.0).rgb; }
    else if (which == 1u) { picture = cluster(v.uv); }
    else { picture = textureSampleLevel(panels, samp_clamp, v.uv, 0.0).rgb; }
  }
  let r = length(v.uv);
  let n = normalize(v.n);
  if (kind == 1u) { return vec4f(finish(v.col, v.w), 1.0); }
  if (kind == 3u) { return vec4f(finish(picture * m.albedo.rgb, v.w), 1.0); }
  if (kind == 6u) { return vec4f(finish(picture * m.albedo.rgb * car_light(v.w, n, false), v.w), 1.0); }
  if (kind == 4u) { // the shadow: darkest under the car, gone at the rim
    let t = clamp((r * 32.0 - 4.0) / 28.0, 0.0, 1.0);
    let a = m.albedo.a * select(mix(0.4, 0.0, (t - 0.6) / 0.4), mix(0.62, 0.4, t / 0.6), t < 0.6);
    return vec4f(finish(m.albedo.rgb, v.w) * a, a);
  }
  if (kind == 5u) {
    let a = m.albedo.a * max(select(mix(0.4, 0.0, (r - 0.25) / 0.75), mix(1.0, 0.4, r / 0.25), r < 0.25), 0.0);
    return vec4f(finish(m.albedo.rgb, v.w) * a, 0.0);
  }
  let albedo = m.albedo.rgb * v.col;
  // street-lamp light at the strength any lamp-lit surface takes it: it lands on the diffuse part of the paint and the trim
  let street = lamp_light(v.w, n) * 1.6 * light_fade(distance(v.w, F.cam_pos.xyz));
  if (kind == 2u) { // the cabin: matt, and lit besides by what the instruments throw on it, a few soft points
    var glow = vec3f(0.0);
    for (var k = 0; k < 3; k++) {
      let d = C.glow_at[k].xyz - v.w;
      let r2 = dot(d, d);
      glow += C.glow[k].rgb * (0.25 + 0.75 * max(dot(n, d * inverseSqrt(r2)), 0.0)) / (1.0 + r2 * 40.0);
    }
    // inside, the roof and pillars keep most of the street lamps' light off: a fifth of it reaches the seats and dash through the glass
    return vec4f(finish(albedo * (car_light(v.w, n, true) + glow + street * 0.2), v.w), 1.0);
  }
  let view = normalize(F.cam_pos.xyz - v.w);
  let nv = max(dot(n, view), 0.001);
  let rough = m.params.x; let metal = m.params.y;
  let f0 = mix(vec3f(0.04), albedo, metal);
  let fresnel = f0 + (1.0 - f0) * pow(1.0 - nv, 5.0);
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
    c += sky(reflect(-view, n), 0.0) * (once + once * mean / (1.0 - rest * mean) * rest) * m.params.z;
  }
  let half_v = normalize(MOON_DIR + view);
  let a2 = max(rough * rough * rough * rough, 1e-4);
  let nh = max(dot(n, half_v), 0.0);
  let dn = nh * nh * (a2 - 1.0) + 1.0;
  c += MOON_LIGHT * fresnel * min(a2 / (PI * dn * dn), 40.0) * max(dot(n, MOON_DIR), 0.0) * 0.25 / max(nv, 0.05);
  let to_lamp = F.tunnel_at.xyz - v.w; // in a tunnel the luminaires glint on the paint as they pass overhead
  let dl = length(to_lamp);
  if (dl < F.tunnel_at.w && which != 1u) {
    let l = to_lamp / dl;
    let edge = 1.0 - pow(dl / F.tunnel_at.w, 4.0);
    let nl = max(dot(n, normalize(l + view)), 0.0);
    let dm = nl * nl * (a2 - 1.0) + 1.0;
    c += F.tunnel.rgb * (edge * edge / max(pow(dl, 1.2), 0.01)) * fresnel * min(a2 / (PI * dm * dm), 40.0) * max(dot(n, l), 0.0) * 0.25 / max(nv, 0.05);
  }
  let to_tail = F.car_pos.xyz + F.car_up.xyz * 0.6 - F.car_dir.xyz * 3.2 - v.w; // and its own tail lamps' pool glints red on its tail
  let dt = length(to_tail);
  if (dt < 11.0 && which != 1u) {
    let l = to_tail / dt;
    let edge = 1.0 - pow(dt / 11.0, 4.0);
    let nl = max(dot(n, normalize(l + view)), 0.0);
    let dm = nl * nl * (a2 - 1.0) + 1.0;
    c += F.tail.rgb * (edge * edge / max(pow(dt, 1.5), 0.01)) * fresnel * min(a2 / (PI * dm * dm), 40.0) * max(dot(n, l), 0.0) * 0.25 / max(nv, 0.05);
  }
  return vec4f(finish(c, v.w) * m.albedo.a, m.albedo.a);
}
`;

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

// the glow around each lamp head and porch light, additive
export const GLOW = CORNERS + /* wgsl */`
struct V { @builtin(position) pos: vec4f, @location(0) uv: vec2f, @location(1) @interpolate(flat) col: vec4f, // as the screen shows it, and its opacity: one colour a sprite, worked out once
  @location(2) @interpolate(flat) soft: f32 }
// look: opacity, negative for the softer dot of the centre's strung bulbs; height over width, negative for a light in the sky, which neither thins with distance nor sinks in the haze
@vertex fn vs(@builtin(vertex_index) vi: u32, @location(0) at: vec4f, @location(1) col: vec4f, @location(2) look: vec2f) -> V {
  var o: V;
  let p = F.view_proj * vec4f(at.xyz, 1.0);
  let depth = dot(at.xyz - F.cam_pos.xyz, F.cam_fwd.xyz);
  // three's size attenuation: size * (drawing-buffer height / 2) / depth, in pixels
  o.pos = sprite(p, at.w * F.cam_up.w * 0.5 / max(depth, 0.001), CORNER[vi] * vec2f(1.0, abs(look.y)));
  o.uv = CORNER[vi];
  let aloft = step(look.y, 0.0); let c = unpack(col);
  o.soft = step(look.x, 0.0);
  o.col = vec4f(mix(finish(c, at.xyz), display(c), aloft), abs(look.x) * mix(mix(1.0, 0.42, smoothstep(250.0, 3000.0, depth)), 1.0, aloft));
  return o;
}
@fragment fn fs(v: V) -> @location(0) vec4f {
  // the sprite's radial falloff: 1 at the centre, .75 at .12, .16 at .4, nothing at the rim
  let r = length(v.uv);
  let hard = select(select(mix(0.16, 0.0, (r - 0.4) / 0.6), mix(0.75, 0.16, (r - 0.12) / 0.28), r < 0.4), mix(1.0, 0.75, r / 0.12), r < 0.12);
  // the soft dot: 1 at the centre, .6 at .2, nothing at the rim
  let a = select(hard, select(mix(0.6, 0.0, (r - 0.2) / 0.8), mix(1.0, 0.6, r / 0.2), r < 0.2), v.soft > 0.5);
  return vec4f(v.col.rgb, max(a, 0.0) * v.col.a);
}
`;

// Cloud: each puff a soft square facing the eye, turned by its seed, ragged with two scales of drifting noise.
// Drawn far to near (the C sorts them); the puffs nearest the eye dissolve, and the haze takes over from them.
export const CLOUD = CORNERS + /* wgsl */`
@group(1) @binding(0) var wisps: texture_2d<f32>;
struct V { @builtin(position) pos: vec4f, @location(0) uv: vec2f, @location(1) look: vec2f, @location(2) dist: f32 }
@vertex fn vs(@builtin(vertex_index) vi: u32, @location(0) at: vec4f, @location(1) look: vec2f) -> V {
  var o: V;
  let a = look.x * 6.2832; let c = cos(a); let s = sin(a);
  let q = QUAD[vi] * 0.5;
  let corner = vec2f(q.x * c - q.y * s, q.x * s + q.y * c) * at.w;
  let centre = at.xyz + F.cloud.xyz;
  o.dist = distance(centre, F.cam_pos.xyz);
  o.pos = F.view_proj * vec4f(centre + normalize(F.cam_right.xyz) * corner.x + normalize(F.cam_up.xyz) * corner.y, 1.0);
  o.uv = q + 0.5; o.look = look;
  return o;
}
@fragment fn fs(v: V) -> @location(0) vec4f {
  let time = F.cam_pos.w;
  let p = v.uv * 2.0 - 1.0;
  let body = 1.0 - length(p);
  let wisp = textureSample(wisps, samp_repeat, v.uv * 0.8 + v.look.x * vec2f(3.1, 7.7) + time * 0.002).r;
  let fine = textureSample(wisps, samp_repeat, v.uv * 2.6 + v.look.x * 5.3 - time * 0.004).r;
  var a = smoothstep(0.0, 0.5, body + (wisp - 0.5) * 0.9 + (fine - 0.5) * 0.3 - 0.12);
  a *= 0.62 * smoothstep(12.0, 150.0, v.dist);
  if (a < 0.004) { return vec4f(0.0); }
  // moonlit and cool on top, faintly warmed underneath by the town
  let col = mix(vec3f(0.06, 0.05, 0.046), vec3f(0.075, 0.085, 0.115), smoothstep(0.1, 0.9, v.look.y) * (0.6 + 0.4 * wisp));
  let hz = v.dist * HAZE_DENSITY;
  return vec4f(display(mix(HAZE_COLOUR, col, exp(-hz * hz))), a);
}
`;

// Falling snow. Three shells of flakes, each a box that wraps around the camera: a few big soft ones right in
// front of the lens, the main fall around the car, and a deep haze of fine ones so lamp cones a street away are
// full of snow too. Flakes flutter, drift on gusts, glint as they turn, and smear into streaks along their
// motion past the camera, so driving fast pulls the snowfall into lines.
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
  let lamp = textureSampleLevel(lamp_map, samp_clamp, clamp((p.xz - F.lamp_rect.xy) / F.lamp_rect.zw, vec2f(0.0), vec2f(1.0)), 0.0).rgb;
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
  o.tint = (vec3f(0.50, 0.60, 0.82) * 0.2 + lamp_glow * 0.95 + vec3f(1.0, 0.88, 0.70) * beam * 0.95) * fade * dim * glint * L.y;
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
struct Snow { pack_at: vec4f, lattice_at: vec4f, pack_car: vec4f }
@group(1) @binding(0) var<uniform> S: Snow;
@group(1) @binding(1) var pack: texture_2d<f32>;
@group(1) @binding(2) var lattice: texture_2d<f32>;
@group(1) @binding(3) var samp_pack: sampler;
`;

// One mesh over the field's window, a vertex at every other cell corner, standing on what the snow lies on there: the
// terrain, a deck, a roof (beds). Heaps stand up for real, ruts are only shaded in, since what is under them cannot be
// dug into.
const BARE_COLOUR = 'vec3f(0.028, 0.032, 0.04)', BARE_SHOWS = '0.88'; // what a jet lays bare, wet asphalt or dark ice; and how fully it covers what is drawn beneath
const PACK_ZERO = 150, PACK_UNIT = 204; // SNOW_ZERO, SNOW_UNIT: the field's heights as bytes
export const PACK = SNOW_IN + /* wgsl */`
@group(1) @binding(4) var beds: texture_2d<u32>; // per cell, mm: how far what the snow lies on stands above the terrain
struct V { @builtin(position) pos: vec4f, @location(0) w: vec3f, @location(1) at: vec2f, @location(2) near: f32 }
fn pack_height(p: vec2f) -> f32 { return textureSampleLevel(pack, samp_pack, p, 0.0).r * 255.0 - ${PACK_ZERO}.0; } // in steps of 1/${PACK_UNIT} m
fn bed(c: vec2i) -> f32 { return f32(textureLoad(beds, c & vec2i(511), 0).r) * 0.001; }
// the terrain's own triangles: its grid cell is split along the same diagonal as the mesh's
fn terrain(at: vec2f) -> f32 {
  let g = (at - S.lattice_at.xy) * S.lattice_at.z; let f = fract(g); let gi = vec2i(floor(g));
  let a = textureLoad(lattice, gi, 0).r; let b = textureLoad(lattice, gi + vec2i(1, 0), 0).r;
  let c = textureLoad(lattice, gi + vec2i(0, 1), 0).r; let d = textureLoad(lattice, gi + vec2i(1, 1), 0).r;
  return select(d + (c - d) * (1.0 - f.x) + (b - d) * (1.0 - f.y), a + (b - a) * f.x + (c - a) * f.y, f.x + f.y <= 1.0);
}
@vertex fn vs(@builtin(vertex_index) vi: u32) -> V {
  let p = vec2f(f32(vi & 255u), f32(vi >> 8u));
  let at = S.pack_at.xy + p * 0.2;
  var o: V;
  o.at = S.pack_at.zw + p * (2.0 / 512.0);
  o.near = 1.0 - smoothstep(17.0, 21.0, distance(at, S.pack_car.xy));
  // averaged over the cells around the vertex: a ridge narrower than the mesh would come out as a row of teeth
  let px = vec2f(1.0 / 512.0, -1.0 / 512.0);
  let heap = pack_height(o.at + px.xx) + pack_height(o.at + px.xy) + pack_height(o.at + px.yx) + pack_height(o.at + px.yy);
  // what it lies on: the highest of the four cells round the vertex, so a roof keeps its edge
  let c = vec2i(round(S.pack_at.zw * 512.0)) + vec2i(p) * 2;
  let lift = max(max(bed(c - vec2i(1, 1)), bed(c - vec2i(0, 1))), max(bed(c - vec2i(1, 0)), bed(c)));
  o.w = vec3f(at.x, terrain(at) + lift + 0.02 + max(heap, 0.0) * (0.25 / ${PACK_UNIT}.0) * o.near, at.y);
  o.pos = F.view_proj * vec4f(o.w, 1.0);
  return o;
}
@fragment fn fs(v: V) -> @location(0) vec4f {
  // its height; and how bare a jet has blown it (snow_wash): there what the snow lay on shows, level, wet and dark
  let cell = textureSampleLevel(pack, samp_pack, v.at, 0.0).rg;
  let deep = (cell.r * 255.0 - ${PACK_ZERO}.0) / ${PACK_UNIT}.0; let bare = cell.g * ${BARE_SHOWS};
  let alpha = max(smoothstep(0.004, 0.022, abs(deep)) * (1.0 - smoothstep(0.0, 0.004, cell.g) * step(0.0, deep)), bare) * v.near; // (ice the field was lifted to shows only where swept; a dish shows all the way down)
  let dx = dpdx(v.w.xz); let dy = dpdy(v.w.xz);
  if (alpha < 0.01) { return vec4f(0.0); } // blended, so nothing laid is the same as dropped: and none of the rest is worked out
  let px = vec2f(1.0 / 512.0, 0.0);
  // 0.55 of the field's true slope: rut walls are near vertical, and would go black
  let n = normalize(vec3f((pack_height(v.at - px.xy) - pack_height(v.at + px.xy)) * (1.0 - cell.g), ${(0.2 * PACK_UNIT / 0.55).toFixed(2)}, (pack_height(v.at - px.yx) - pack_height(v.at + px.yx)) * (1.0 - cell.g)));
  let grain = textureSampleLevel(detail_map, samp_repeat, v.w.xz * 0.37, 0.0).r * 0.55 + textureSampleLevel(detail_map, samp_repeat, v.w.xz * 0.051, 0.0).r * 0.45;
  // thrown snow is clean and bright; pressed snow lies in its own shadow, darker the deeper it is
  let pressed = mix(vec3f(0.62, 0.67, 0.78), vec3f(0.40, 0.45, 0.57), smoothstep(0.02, 0.12, -deep));
  let albedo = mix(mix(pressed, vec3f(0.84, 0.89, 0.97), smoothstep(-0.01, 0.015, deep)) * (0.86 + 0.28 * grain), ${BARE_COLOUR}, cell.g);
  return vec4f(finish(lit_snow(albedo, n, v.w, 1.0, -1.0, 1.0, dx, dy), v.w), alpha);
}
`;

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
  let lamp = textureSampleLevel(lamp_map, samp_clamp, clamp((p.xz - F.lamp_rect.xy) / F.lamp_rect.zw, vec2f(0.0), vec2f(1.0)), 0.0).rgb;
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

// ---- the jets in the car's wheels (wasm/src/rocket.c): a flame from each nozzle, and the snow their wash blows off the ground ----
// the flame's brightness; its half width at its widest, m, and the share of that it has at the lip; the radius of the glow seen
// looking up the nozzle, m
const JET = { bright: 1.3, width: 0.105, lip: 0.72, glow: 0.15 };
const PUFF = { life: 2.6, mist: 0.09, grains: 0.2, drag: 2.4, billows: 0.3, roof: 2.2, flake: 0.7 }; // s the longest grain lasts (DRIFT_LIFE in the C); brightness of the billows and of the grains; 1/s the air stops them; the share that are billows
const f3 = v => v.toFixed(3);
// one instance a nozzle: where its lip is and the flame's length; which way it points and how hard it burns
export const JETS = CORNERS + /* wgsl */`
struct V { @builtin(position) pos: vec4f, @location(0) uv: vec2f, @location(1) @interpolate(flat) look: vec4f }
// A flame's colour by how hot it is, 0..1: red where it is thinnest and coolest, through orange and pale yellow to white.
fn flame(h: f32) -> vec3f {
  let red = vec3f(0.85, 0.07, 0.012); let orange = vec3f(1.0, 0.36, 0.05); let yellow = vec3f(1.0, 0.78, 0.32); let white = vec3f(1.0, 0.95, 0.84);
  return mix(mix(mix(red, orange, smoothstep(0.14, 0.38, h)), yellow, smoothstep(0.36, 0.58, h)), white, smoothstep(0.56, 0.82, h));
}
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32, @location(0) at: vec4f, @location(1) aim: vec4f) -> V {
  let c = CORNER[vi % 6u];
  let time = F.cam_pos.w; let seed = f32(ii) * 1.7;
  // no flame burns quite steadily
  let flick = 1.0 + 0.06 * sin(time * 43.0 + seed) * sin(time * 29.0 + seed * 2.9) + 0.03 * sin(time * 71.0 + seed * 5.3);
  let view = normalize(at.xyz - F.cam_pos.xyz);
  let end_on = abs(dot(view, aim.xyz));
  var o: V; var w: vec3f;
  if (vi < 6u) { // the flame: a strip along the jet, turned about it to face the eye
    let across = normalize(cross(aim.xyz, view) + vec3f(1e-5, 0.0, 0.0));
    let len = at.w * flick;
    w = at.xyz + aim.xyz * ((c.y * 0.5 + 0.5) * len) + across * (c.x * ${f3(JET.width)});
    o.look = vec4f(aim.w * (1.0 - pow(end_on, 6.0)), len, 0.0, seed);
  } else { // seen along the jet that strip is edge-on: a small round glow in the nozzle's mouth stands in for it
    w = at.xyz + aim.xyz * 0.04 + (normalize(F.cam_right.xyz) * c.x + normalize(F.cam_up.xyz) * c.y) * ${f3(JET.glow)};
    o.look = vec4f(aim.w * flick * smoothstep(0.45, 0.95, end_on), at.w, 1.0, seed);
  }
  o.uv = c;
  o.pos = F.view_proj * vec4f(w, 1.0);
  if (aim.w <= 0.004) { o.pos = vec4f(0.0); }
  return o;
}
@fragment fn fs(v: V) -> @location(0) vec4f {
  let power = v.look.x; let hard = min(power, 1.2);
  if (v.look.z > 0.5) {
    let r = 1.0 - smoothstep(0.0, 1.0, length(v.uv));
    return vec4f(display(flame(r * r * (0.5 + 0.5 * hard))), min(r * r * power * ${f3(JET.bright)}, 1.0));
  }
  let along = v.uv.y * 0.5 + 0.5; let x = abs(v.uv.x);
  // its outline: full at the lip, swelling a little just below it, then drawn out to a point; it shimmers as it goes
  let shimmer = 1.0 + 0.08 * along * sin(along * 21.0 - F.cam_pos.w * 33.0 + v.look.w);
  let wide = mix(${f3(JET.lip)}, 1.0, smoothstep(0.0, 0.14, along)) * pow(1.0 - along, 0.75) * (0.62 + 0.38 * min(hard, 1.0)) * shimmer;
  let body = clamp(1.0 - x / max(wide, 1e-3), 0.0, 1.0);
  // hottest on its axis at the lip: white there at full thrust, yellow when throttled back; cooling outwards and along it,
  // so its thin edge and its tail are red. At high thrust shock diamonds stand in the core.
  let shock = smoothstep(0.6, 1.0, power) * exp(-along * 2.6) * cos(along * v.look.y * 30.0 - 1.2);
  let heat = pow(body, 1.35) * mix(0.24, 1.0, pow(1.0 - along, 1.3)) * (0.56 + 0.44 * hard) * (1.0 + 0.3 * shock) + 0.16 * max(shock, 0.0) * body * body;
  // and a breath of blue where it leaves the lip
  let lip = (1.0 - smoothstep(0.0, 0.07, along)) * smoothstep(0.0, 0.5, body) * (1.0 - smoothstep(0.5, 0.9, body));
  let col = flame(heat) + vec3f(0.05, 0.2, 0.9) * lip * 0.6;
  let a = smoothstep(0.0, 0.36, heat) * smoothstep(0.0, 0.035, along) * (1.0 - smoothstep(0.7, 1.0, along)) * min(power * 2.5, 1.0);
  return vec4f(display(col), min(a * ${f3(JET.bright)}, 1.0));
}
`;
// blown snow (wasm/src/drift.c): each grain leaves the ground where a jet's wash tore it loose, with the wind's own velocity
// there, and races outward low over the ground; the air stops it within a few metres, it rises a little where the wind
// did, and sinks back. Most are grains that catch the light, the rest soft billows that spread as they go.
export const BLOWN = CORNERS + /* wgsl */`
struct V { @builtin(position) pos: vec4f, @location(0) uv: vec2f, @location(1) tint: vec3f }
@vertex fn vs(@builtin(vertex_index) vi: u32, @location(0) left: vec4f, @location(1) flung: vec4f) -> V {
  let drop = floor(flung.w); let kind = flung.w - drop; // whole metres it may fall, then its own number (drift.h)
  let origin = left.xyz; let age = F.cam_pos.w - left.w;
  // Snow a gust takes off a roof (it may fall: gust.h) is the same snow as what falls: flakes of its size and its light,
  // that linger as they sink and drift; the jets' wash throws grains and small puffs right under the eye
  let roof = step(0.5, drop);
  let billow = step(kind, ${f3(PUFF.billows)}) * (1.0 - roof);
  let life = ${f3(PUFF.life)} * (0.4 + 0.6 * fract(kind * 7.31)) * mix(1.0, ${f3(PUFF.roof)}, roof); let t = age / life;
  let drag = ${f3(PUFF.drag)} * mix(1.0, 1.5, billow) * mix(1.0, 0.45, roof);
  let gone = (1.0 - exp(-drag * age)) / drag;
  var p = origin + flung.xyz * gone;
  p.y = max(p.y - 0.5 * age * age * mix(1.0, 0.35, billow), origin.y - 0.02 - drop);
  // slowed, it drifts on with the night's wind
  p += vec3f(0.7, 0.0, 0.3) * age * t * 0.6;
  let clip = F.view_proj * vec4f(p, 1.0);
  let dist = dot(p - F.cam_pos.xyz, F.cam_fwd.xyz);
  let k3 = fract(kind * 3.7);
  let radius = mix(mix(0.014 + 0.012 * k3, mix(0.14, 0.75, t) * (0.7 + fract(kind * 13.1)), billow), 0.018 + 0.035 * k3 * k3, roof);
  // lit like the falling snow, by the sky, the street lamps and the headlight beam; and from below by the jets' flames
  let v = p - (F.car_pos.xyz + vec3f(0.0, 0.8, 0.0));
  let along = dot(v, F.car_dir.xyz); let perp = length(v - F.car_dir.xyz * along);
  let beam = smoothstep(1.8, 5.0, along) * (1.0 - smoothstep(0.0, 1.0, perp / (along * 0.4 + 0.7)));
  let lamp = textureSampleLevel(lamp_map, samp_clamp, clamp((p.xz - F.lamp_rect.xy) / F.lamp_rect.zw, vec2f(0.0), vec2f(1.0)), 0.0).rgb;
  let near = p - F.car_pos.xyz;
  // a flake off a roof too small to draw fades rather than flickers, as the falling snow does; and the lamps' light
  // reaches it as it reaches the falling snow, from below
  var px = F.cam_up.w / F.cam_right.w * radius / dist;
  let dim = mix(1.0, min(1.0, px / 1.3), roof);
  px = max(px, mix(1.0, 1.3, roof));
  let high = mix(1.0, 1.0 - smoothstep(4.5, 9.0, p.y - ground_at(p.xz)), roof);
  let glint = mix(1.0, 0.72 + 0.4 * sin(F.cam_pos.w * (2.0 + kind * 6.0) + kind * 40.0), roof);
  var o: V;
  o.pos = sprite(clip, clamp(px, 1.0, 200.0), QUAD[vi]);
  if (t <= 0.0 || t >= 1.0 || dist <= 0.05) { o.pos = vec4f(2.0, 2.0, 2.0, 1.0); }
  o.uv = QUAD[vi] * 0.5;
  o.tint = (vec3f(0.50, 0.60, 0.82) * 0.2 + (vec3f(1.0, 0.58, 0.24) * (lamp.r + lamp.g) + vec3f(0.80, 0.92, 1.0) * lamp.b) * mix(0.6, 0.95, roof) * high
    + vec3f(1.0, 0.93, 0.80) * beam + F.jet.rgb * 0.6 / (1.0 + dot(near, near) * 0.5))
    * (1.0 - t) * (1.0 - t) * smoothstep(0.0, 0.06, t) * mix(mix(${f3(PUFF.grains)}, ${f3(PUFF.mist)}, billow), ${f3(PUFF.flake)}, roof) * dim * glint;
  return o;
}
@fragment fn fs(v: V) -> @location(0) vec4f {
  let a = 1.0 - smoothstep(0.05, 0.5, length(v.uv));
  return vec4f(display(v.tint * a), a);
}
`;

// the lasting marks: a long ribbon in a ring; the further back along it, the more has fallen into them
export const TRACKS = SNOW_IN + /* wgsl */`
struct V { @builtin(position) pos: vec4f, @location(0) w: vec3f, @location(1) col: vec4f }
@vertex fn vs(@builtin(vertex_index) vi: u32, @location(0) pos: vec3f, @location(1) col: vec4f) -> V {
  var o: V;
  o.pos = F.view_proj * vec4f(pos, 1.0);
  o.w = pos;
  o.col = vec4f(col.rgb, col.a * mix(1.0, 0.45, fract((S.lattice_at.w - f32(vi / 10u)) / 28000.0)));
  return o;
}
@fragment fn fs(v: V) -> @location(0) vec4f { return vec4f(finish(lit(v.col.rgb, vec3f(0.0, 1.0, 0.0), v.w, 1.0), v.w), v.col.a); }
`;

/* ---- sky and sea ---- */

// one triangle over the whole view, at the far plane: the depth test skips every pixel the land already covers. The
// aurora and the drifting cloud, most of the sky's cost and smooth across a few pixels, come from SKY_MAPS' texture,
// drawn first at half the width and height; the gradient, stars and moon are worked out here, sharp.
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
  return vec4f(mix(display(sky_over(d, 1.0, MOON_DIR, noise)), display(CLOUD_TINT), F.cloud.w), 1.0);
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
// Rows 2i are curtain i's wander, 2i + 1 its rays, as sky_noise reads them.
@compute @workgroup_size(64) fn rows(@builtin(global_invocation_id) id: vec3u) {
  let slot = id.y; let row = slot / 5u; let k = slot % 5u; let fi = f32(row / 2u); let time = F.cam_pos.w;
  var at = vec2f(-4.0 * 7.0 + time * 0.05, fi * 3.1 + time * 0.02);
  if (row % 2u == 0u) { at = vec2f(-4.0 * 0.55 + fi * 7.3, time * 0.016 + fi); }
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
const SDF_LO = -8, SDF_HI = 56, HORIZON_N = 512, HORIZON_TAN = 0.6, HORIZON_STEPS = 40, HORIZON_REACH = [8, 16000], EARTH_R = 6.371e6;
const RIPPLE_TILE = [43, 6.1], RIPPLE_TURN = 0.61;
const CHOP = [0.004, 0.034], WAVELETS = [0.003, 0.05], CAPILLARY = 0.012; // RMS slope of each tile [glassy basin, open fjord]
const WIND_TO = [0.71, -0.71], GUST = [300, 100, 3.2];
const F0 = 0.02, MOON_DISC = 0.0245, BODY = [0.004, 0.007, 0.008], LAP = [0.7, 1.1], LIGHT_K = 3.2, PANCAKE = [9, 3.6];
const SLOPE_MAX = Math.hypot(CHOP[1], WAVELETS[1] * 1.7, CAPILLARY); // the steepest slope the roughest water shows, gusts and all
// A streak is drawn only where the water under it can throw its light: its strip is cut to STREAK_SIGMAS times the RMS slope
// of the roughest water it crosses (in a sheltered basin a tenth of the open fjord's, so a hundredth of the pixels), and a pixel
// of it whose water is calmer than that is given up at STREAK_SIGMAS_PIXEL before the surface is worked out. What the water
// can do is read from the bound map (SEA_BOUND below). Past six sigma a gaussian holds 1.5e-8 of its peak.
const STREAK_SIGMAS = 6, STREAK_SIGMAS_PIXEL = 8, STREAK_SPANS = 8; // spans: sea.h's SEA_STREAK_SPANS

// What the sea and the light streaks share: where a point lies, how rough the water there is, its ice.
const SEA_IN = /* wgsl */`
struct Sea {
  rect: vec4f,         // the sea map: x, z, width, depth
  horizon_rect: vec4f, // the horizon maps
  ripple_var: vec4f,   // each ripple tile's whole-tile slope variance per axis: the chop's, the wavelets'
  swell: array<vec4f, 3>, // wave vector, phase, amplitude
}
@group(1) @binding(0) var<uniform> S: Sea;
@group(1) @binding(1) var sea_map: texture_2d<f32>;
@group(1) @binding(2) var ripple_a: texture_2d<f32>;
@group(1) @binding(3) var ripple_b: texture_2d<f32>;
@group(1) @binding(4) var floes: texture_2d<f32>;
@group(1) @binding(5) var horizon_a: texture_2d<f32>;
@group(1) @binding(6) var horizon_b: texture_2d<f32>;
@group(1) @binding(7) var sky_map: texture_2d<f32>;
@group(1) @binding(8) var samp_sea: sampler;    // clamped, mip-mapped
@group(1) @binding(9) var samp_ripple: sampler; // repeating, mip-mapped, anisotropic
@group(1) @binding(10) var samp_sky: sampler;   // round the horizon, clamped up and down
@group(1) @binding(11) var sea_bound: texture_2d<f32>; // what the water near a texel can be, at most (SEA_BOUND)

struct SeaAt { shore: f32, open: f32, clump: f32, made: f32 }
fn sea_at(p: vec2f) -> SeaAt {
  let uv = (p - S.rect.xy) / S.rect.zw;
  var s = textureSample(sea_map, samp_sea, uv);
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
  let q = -h.xz / h.y - m;
  let d = fwidth(q);
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
  let px = S.rect.z / f32(textureDimensions(sea_bound).x);
  let level = i32(ceil(log2(max(r / px, 1.0))));
  if (level >= i32(textureNumLevels(sea_bound))) { return vec2f(1.0); }
  let c = vec2i(floor((p - S.rect.xy) / (px * exp2(f32(level)))));
  if (any(c < vec2i(1)) || any(c > vec2i(textureDimensions(sea_bound, level)) - 2)) { return vec2f(1.0); }
  return textureLoad(sea_bound, c, level).gr;
}
// The horizon texel at p, none off the backdrop. Call where every pixel gets here.
fn horizon_texel(map: texture_2d<f32>, p: vec2f) -> vec4f {
  let uv = (p - S.horizon_rect.xy) / S.horizon_rect.zw;
  return textureSample(map, samp_sea, uv) * f32(all(uv == clamp(uv, vec2f(0.0), vec2f(1.0))));
}
`;

export const AFLOAT = afloat(SEA_IN);
export const WATER = SEA_IN + /* wgsl */`
const SHORE_TONE = HAZE_COLOUR * 0.8;
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
  return vec3f(1.0, 0.58, 0.24) * m.r + vec3f(0.80, 0.92, 1.0) * m.b + vec3f(1.0, 0.58, 0.26) * m.g * 1.25;
}
@fragment fn fs(i: V) -> @location(0) vec4f {
  let time = F.cam_pos.w; let p = i.w;
  let to_eye = F.cam_pos.xyz - vec3f(p.x, 0.0, p.y);
  let dist = length(to_eye);
  let v = to_eye / dist;
  let dx = dpdx(p); let dy = dpdy(p);
  let s = sea_at(p);
  let shore_blur = fwidth(s.shore);
  let h0 = horizon_texel(horizon_a, p); let h1 = horizon_texel(horizon_b, p);
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
    refl = mix(SHORE_TONE, refl, smoothstep(-1.0, 1.0, (tan_r - tan_h) / (spread * 2.0 + 0.004)));
  }
  // the moon: its glitter path, sharp where the water is still, and the glow round it
  let shine = MOON_LIGHT * glint(v, MOON_DIR, mean, vr + ${f5((MOON_DISC / 2) ** 2)})
    + vec3f(0.55, 0.66, 0.9) * 0.05 * pow(max(dot(r, MOON_DIR), 0.0), 40.0) * fr;

  // what the water itself gives back: the dark body lit faintly from above, more under the quay lamps
  let lamps = sea_lamps(p) * 1.6;
  let ambient = mix(GROUND_LIGHT, SKY_LIGHT, 0.5 + 0.5 * n.y);
  let moon_down = MOON_LIGHT * max(MOON_DIR.y, 0.0);
  let body = vec3f(${BODY.map(f5).join(', ')}) * (ambient + moon_down + lamps);

  var col = body + (fr * refl + shine) * (1.0 - ice);
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
    let ice_lit = albedo * (mix(GROUND_LIGHT, SKY_LIGHT, 0.75) + moon_down + lamps);
    col = mix(col, ice_lit + fr * refl * 0.25 * (1.0 - plates), min(ice * 1.6, 1.0));
  }
  // the wet line where the water laps at ice, rock and wall, swinging in and out with the swell
  let swing = 0.5 + 0.5 * sin(time * 1.3 + dot(p, vec2f(0.21, 0.17)) + s.clump * 6.0);
  let wet_w = ${f5(LAP[0])} + ${f5(LAP[1] - LAP[0])} * swing; let aa = max(shore_blur, 0.05);
  var lap = smoothstep(wet_w + aa, wet_w - aa, s.shore) * smoothstep(-aa, aa, s.shore);
  let slush = max(fwidth(ice), 0.08); // and along the ice's ragged outer edge, a line of wet slush that comes and goes
  lap = max(lap, (1.0 - smoothstep(0.0, slush * (0.6 + swing), abs(ice - 0.25))) * 0.6);
  col = mix(col, vec3f(0.3, 0.33, 0.38) * (ambient + lamps + moon_down), lap * 0.7);

  // Haze over the far water. It takes on the sky's colour at the horizon there, as the sky is drawn unhazed: so the
  // sea closes on the sky without a seam.
  let fog = 1.0 - exp(-F.fog.a * F.fog.a * dist * dist);
  var haze = F.fog.rgb;
  if (fog > 0.3) { haze = mix(F.fog.rgb, sky_at(normalize(vec3f(-v.x, 0.0, -v.z)), 1.0), smoothstep(0.3, 1.0, fog)); }
  return vec4f(display(mix(col, haze, fog)), 1.0);
}
`;

// Each light's reflection drawn where it can fall: a strip of quad on the sea from the foot of the light towards the
// eye, as long and as wide as the roughest water could throw it. Its fragment shader then works out, per pixel, how
// many of the facets there are turned to send that light to the eye (the same glint as the moon's), so a light's streak
// lengthens and breaks up over open water and shrinks to a still image of it in the calm. Instances: struct sea_streak.
export const STREAK = SEA_IN + /* wgsl */`
struct V { @builtin(position) pos: vec4f, @location(0) p: vec3f, @location(1) tint: vec3f, @location(2) @interpolate(flat) lamp: vec4f,
  @location(3) @interpolate(flat) fit: vec4f } // fit: what the strip was cut to, for ?off=streakcheck
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
@vertex fn vs(@builtin(vertex_index) vi: u32, @location(0) lamp: vec4f, @location(1) tint: vec4f, @location(2) lit: f32) -> V {
  // a station at each end of every span along the strip, a vertex either side of each
  let k6 = array<u32, 6>(0u, 1u, 2u, 1u, 3u, 2u);
  let id = (vi / 6u) * 2u + k6[vi % 6u];
  let corner = vec2f(f32(id >> 1u), f32(id & 1u) * 2.0 - 1.0);
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
    if (b.y > 0.0 && least <= t) {
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
  let at = i32(corner.x); let s = clamp(ends[at], s0, s1);
  var w = half_width(s, d, hl, hc, k_most) + lamp.w + 2.0 * pixel * length(vec2f(d - s, hc)); // never wider than the roughest water's strip
  if (k < k_most) {
    // and at each station as wide as the rougher of the spans it stands between asks; laid straight between stations, the
    // edge runs inside the curve it follows, so a little over
    let here = max(asks[max(at - 1, 0)], asks[min(at, ${STREAK_SPANS - 1})]);
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
${OFF.has('streakfit') || OFF.has('streakcheck') ? '' : '  if (dry) { o.pos = vec4f(0.0); } // nothing but land under it'}
  return o;
}
@fragment fn fs(i: V) -> @location(0) vec4f {
${OFF.has('streakflat') ? '  if (true) { return vec4f(0.03, 0.0, 0.0, 1.0); } // where the strips are, and how often over' : ''}
  let p = i.p.xz;
  let to_eye = F.cam_pos.xyz - i.p; let to_lamp = i.lamp.xyz - i.p;
  let dist = length(to_eye); let dl = length(to_lamp);
  let v = to_eye / dist; let l = to_lamp / dl;
  let s = sea_at(p);
  let open = openness(s, horizon_texel(horizon_a, p));
  let dx = dpdx(p); let dy = dpdy(p); let blur = fwidth(s.shore);
  let size = i.lamp.w / dl;
  // The facet that would send the light to the eye from here: where the water at this pixel cannot be within
  // ${STREAK_SIGMAS_PIXEL} sigma of it, however it lies, there is nothing to work out.
  let h = normalize(v + l); let q0 = -h.xz / h.y; let spread = fwidth(q0);
  let need = max(length(q0) - swell_most(), 0.0);
  // the variance the water itself must have, beside the lamp's width and the pixel's spread, for this pixel to show anything
  var least = need * need * ${f5(1 / STREAK_SIGMAS_PIXEL ** 2)} - size * size * 0.25 - max(spread.x, spread.y) * max(spread.x, spread.y) * 0.25;
  var given_up = rough_most(open) < least;
${OFF.has('streakfit') || OFF.has('streakcheck') ? '  let would = given_up; given_up = false; least = -1.0;' : ''}
  var sf = Slope(vec2f(0.0), vec2f(1.0), 1.0);
  if (!given_up) { sf = surface(p, s, open, dx, dy, blur, least); }
  let c = i.tint / (dl * dl) * glint(v, l, sf.mean, sf.vr + size * size * 0.25) * (1.0 - sf.ice) * exp(-F.fog.a * F.fog.a * dist * dist);
${OFF.has('streakcheck') ? `  { // only the light the cut strip and the pixel's test would have left out, sixty-four times over
    let dv = F.cam_pos.xz - i.lamp.xz; let d = max(length(dv), 0.01); let u = dv / d; let hc = max(F.cam_pos.y, 0.2);
    let along = dot(p - i.lamp.xz, u); let side = abs(dot(p - i.lamp.xz, vec2f(-u.y, u.x)));
    let pixel = 2.0 * F.cam_right.w / F.cam_up.w;
    let inside = i.fit.w < 0.5 && along >= i.fit.x && along <= i.fit.y
      && side <= half_width_true(along, d, i.lamp.y, hc, i.fit.z / 1.15) + i.lamp.w + 2.0 * pixel * length(vec2f(d - along, hc));
    if (inside && !would) { return vec4f(0.0, 0.0, 0.0, 1.0); }
    return vec4f(c * 64.0, 1.0);
  }` : ''}
  return vec4f(display(c), 1.0);
}
`;

// The bound map: for each texel of the sea map and each level above it, the most the water within a texel of that level
// around it can be, however the sea and horizon maps are filtered there: r the shore distance, g how open (openness()).
// first: the level made from the maps themselves, over five texels by five; else from the level below, four by four.
export const SEA_BOUND = first => /* wgsl */`
struct B { rect: vec4f, horizon_rect: vec4f }
@group(0) @binding(0) var src: texture_2d<f32>;
${first ? `@group(0) @binding(1) var horizon: texture_2d<f32>;
@group(0) @binding(2) var<uniform> U: B;` : ''}
@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f { return vec4f(f32(vi & 1u) * 4.0 - 1.0, f32(vi >> 1u) * 4.0 - 1.0, 0.0, 1.0); }
@fragment fn fs(@builtin(position) at: vec4f) -> @location(0) vec4f {
  let c = vec2i(at.xy); let top = vec2i(textureDimensions(src)) - 1;
  var most = vec2f(0.0);
${first ? `  for (var j = -2; j <= 2; j++) { for (var i = -2; i <= 2; i++) { most = max(most, textureLoad(src, clamp(c + vec2i(i, j), vec2i(0), top), 0).rg); } }
  // the least the land upwind can shelter anywhere a horizon texel read from here could lie
  let px = U.rect.zw / vec2f(top + 1); let n = vec2f(textureDimensions(horizon)); let ntop = vec2i(n) - 1;
  let lo = (U.rect.xy + (vec2f(c) - 2.0) * px - U.horizon_rect.xy) / U.horizon_rect.zw; let hi = (U.rect.xy + (vec2f(c) + 3.0) * px - U.horizon_rect.xy) / U.horizon_rect.zw;
  var lee = 1.0;
  if (any(lo < vec2f(0.0)) || any(hi > vec2f(1.0))) { lee = 0.0; }
  let h0 = vec2i(floor(lo * n - 0.5)); let h1 = vec2i(floor(hi * n - 0.5)) + 1;
  for (var j = h0.y; j <= h1.y; j++) { for (var i = h0.x; i <= h1.x; i++) { lee = min(lee, textureLoad(horizon, clamp(vec2i(i, j), vec2i(0), ntop), 0).a); } }
  let open = most.g * (1.0 - smoothstep(0.02, 0.14, lee * lee * ${f5(HORIZON_TAN)})) * smoothstep(2.0, 40.0, mix(${f5(SDF_LO)}, ${f5(SDF_HI)}, most.r));
  return vec4f(most.r, min(ceil(open * 255.0) / 255.0, 1.0), 0.0, 1.0);`
  : `  for (var j = -1; j <= 2; j++) { for (var i = -1; i <= 2; i++) { most = max(most, textureLoad(src, clamp(c * 2 + vec2i(i, j), vec2i(0), top), 0).rg); } }
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
  return SKY_LIGHT + MOON_LIGHT * 0.6 + (vec3f(1.0, 0.58, 0.24) * m.r + vec3f(0.80, 0.92, 1.0) * m.b + vec3f(1.0, 0.58, 0.26) * m.g * 1.25) * 1.6;
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
  return vec4f(display(vec3f(0.42, 0.46, 0.5) * foam_light(i.p)), a);
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
    return vec4f(display(col), min(foam + hollow, 0.9));
  }
  let r = length(i.p - i.ring.xy); let front = 0.8 + i.age * 1.0; let k = r - front;
  // a short train of crests behind the leading one, dying out as it spreads
  let crest = max(cos(k * 6.0), 0.0) * exp(-k * k * 1.2);
  let a = crest * (1.0 - smoothstep(0.0, ${f5(RING_LIFE)}, i.age)) * min(abs(i.ring.w), 1.0) * 0.35 / (1.0 + i.age);
  return vec4f(display(vec3f(0.4, 0.44, 0.5) * foam_light(i.p)), a);
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
  return vec4f(display(vec3f(0.55, 0.6, 0.66) * foam_light(i.p)), a);
}
`;

// Chimney smoke (src/smoke.js): some of the houses have the stove going. A thin pale plume climbs a few metres out of
// the pot, then leans over and thins out downwind. Every puff is an instance, moved here from the clock alone.
const SMOKE = { life: 11, rise: 3.6, drift: 0.55, opacity: 0.5 }; // s a puff lasts; m it climbs; m/s the wind carries it; opacity of a young puff
export const SMOKE_PUFFS = /* wgsl */`
struct V { @builtin(position) pos: vec4f, @location(0) at: vec2f, @location(1) w: vec3f, @location(2) alpha: f32, @location(3) dist: f32 }
// seed: where in its life the puff is, how hard the stove draws, one random number
@vertex fn vs(@builtin(vertex_index) vi: u32, @location(0) base: vec3f, @location(1) seed: vec3f) -> V {
  let c = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0));
  let time = F.cam_pos.w;
  let life = fract(time * ${f5(1 / SMOKE.life)} * (0.8 + 0.4 * seed.z) + seed.x); let age = life * ${SMOKE.life.toFixed(1)};
  // straight up while it is hot, then the wind has it; a slow waver so no two plumes lean alike
  let along = ${SMOKE.drift.toFixed(2)} * age * (0.35 + 0.09 * age);
  let sway = sin(time * 0.31 + seed.z * 40.0 + age * 0.5) * 0.09 * age;
  var w = base + vec3f(SNOW_WIND.x * along - SNOW_WIND.y * sway, ${SMOKE.rise.toFixed(1)} * seed.y * (1.0 - exp(-age * 0.3)), SNOW_WIND.y * along + SNOW_WIND.x * sway);
  var o: V;
  o.dist = distance(w, F.cam_pos.xyz);
  o.w = w; o.at = c[vi];
  // face the eye, and spread as it cools
  let size = (0.26 + 0.13 * age) * (0.8 + 0.4 * seed.z);
  let right = normalize(F.cam_right.xyz); let up = normalize(F.cam_up.xyz);
  o.pos = F.view_proj * vec4f(w + (right * c[vi].x + up * c[vi].y) * size, 1.0);
  o.alpha = ${SMOKE.opacity.toFixed(2)} * seed.y * smoothstep(0.0, 0.06, life) * pow(1.0 - life, 1.1) * (1.0 + 0.7 * exp(-age * 0.6)); // densest where it leaves the pot
  return o;
}
@fragment fn fs(i: V) -> @location(0) vec4f {
  let a = i.alpha * smoothstep(1.0, 0.15, length(i.at));
  if (a < 0.003) { return vec4f(0.0); }
  // grey under the moon; only a plume low enough to stand in lamplight takes any colour from it
  var col = vec3f(0.19, 0.2, 0.225) + lamp_light(i.w, vec3f(0.0, 1.0, 0.0)) * 0.35;
  let hz = i.dist * HAZE_DENSITY;
  col = mix(mix(HAZE_COLOUR, CLOUD_TINT, F.cloud.w), col, exp(-hz * hz)); // the haze's colour in linear light: F.fog holds it as the screen shows it
  return vec4f(display(col), a);
}
`;



// The horizon maps, marched once over the terrain's heights and the backdrop's. first: 0 for the map that holds east
// round through south to the south-west, 4 for the one that holds west round through north to the north-east.
export const HORIZON = /* wgsl */`
struct H { ground_rect: vec4f, rect: vec4f, first: f32, outer_n: f32 }
@group(0) @binding(0) var<uniform> U: H;
@group(0) @binding(1) var ground: texture_2d<f32>;
@group(0) @binding(2) var outer: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;
@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f { return vec4f(f32(vi & 1u) * 4.0 - 1.0, f32(vi >> 1u) * 4.0 - 1.0, 0.0, 1.0); }
fn height_at(p: vec2f) -> f32 {
  let g = (p - U.ground_rect.xy) / U.ground_rect.zw;
  if (all(g == clamp(g, vec2f(0.002), vec2f(0.998)))) { return textureSampleLevel(ground, samp, g, 0.0).r; }
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

// mip chain by repeated halving, one layer at a time
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
