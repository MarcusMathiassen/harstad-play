// The render worker: owns the OffscreenCanvas, the WebGPU device and the game's wasm instance.
//
// ---- how the pieces fit ----
//
// main thread (main.js)                 this worker                          build worker (build_worker.js)
//   DOM, keys, HUD, minimap, audio        harstad.wasm #1: world lookups,      harstad.wasm #2: geometry and the
//   canvas.transferControlToOffscreen     car, camera, culling, draw list      lamp map, stage by stage; then gone
//          | init, resize, events            |  WebGPU: every GPU object           |
//          | keys, pad  (shared memory)      |  one render pass per frame          |
//          v                                 v                                     v
//      postMessage ------------------->  onmessage  <------- { stage, bufs, items } (transferred)
//      hud  <---- shared memory, or without cross-origin isolation a message when what it says has changed
//
// Shared block (a SharedArrayBuffer when the page is cross-origin isolated): Uint32 [0] = keys held (KEY_* in
// game.h) and Float32 [1..6] = the gamepad (struct game_pad), written by the main thread; bytes 64..199 = struct game_hud, written here every frame.
//
// wasm cannot call WebGPU. Each frame hs_frame() fills struct frame (game.h): the uniforms, a list of struct
// gfx_cmd grouped by pipeline, where the car's parts are, and which of the arrays that change (the snow the car
// drives through, the clouds, the lights and the sleigh in the sky) must be sent again. This file copies those
// to their buffers and walks the commands. Buffer handles in commands index `buffers`, filled as build stages arrive.
//
// The frame, one pass, 4x MSAA, reversed depth:
//   opaque  buildings and banks, walls, trees, lamp posts, the car; then terrain, backdrop and sea, which
//           most of the time are behind something already drawn
//   blend   roads, crossings, the car's tracks and the trodden snow over the terrain; sky at the far plane;
//           clouds far to near; glow sprites, spray and falling snow, additive
//
// Textures: the photo's pages decode here (createImageBitmap, off the main thread by construction) into pools of array
// textures, as near as the eye needs them (photoPaging), so the terrain is a single draw; the terrain itself has no
// vertex buffer, only a height texture.

import * as WGSL0 from './shaders.js';
import { analyse, entryFor, groupUse, layoutCounts, overLimits } from './wgsl_use.js';
import { facadeTextures, facadeStyles, displayAtlas, roadTexture, noiseTexture, floePattern, snowFlakes, tunnelTextures, plateTexture, dashPainter, pavingTextures, cloudVolume, trailTexture, snowRelief } from './textures.js';

let WGSL = WGSL0; // or, with ?off=a,b on the page, the shaders made with those parts changed: for looking into the frame
const NONE = 0xffffffff;
const PIPE = { ROAD: 0, PLAIN: 1, WALL: 2, TREE: 3, LAMP: 4, GLOW: 5, TUBE: 6, TUBE_FLOOR: 7, PAVING: 8, ZEBRA: 9, SIGN: 10, TRAIL: 11, LINE: 12,
  FOREST: 13, FOREST_FADE: 14, FOREST_TWIG: 15, HAZE: 16, PANE: 17, LABEL: 18, AFLOAT: 19, SMOKE: 20, STREAK: 21, IMPOSTOR: 22, FINE: 23, GLAZED: 24 }; // enum gfx_pipe
const BUF = { VERTEX: 0, INDEX: 1, INSTANCE: 2, LAMP_MAP: 3, SIGN_ATLAS: 4, BANK_VERTEX: 5, BANK_INDEX: 6, TRAIL_PRINTS: 7, TREE_MODELS: 8, TINT_MAP: 9, LABEL_ATLAS: 10, SEA_MAP: 11, FOREST: 12, LAMP_HEADS: 13, MOON_FIELD: 14, MOON_REACH: 15, SKY_MAP: 16, SMOKE: 17, WET_LIGHTS: 18, MAP_PAGES: 19, REACH_PAGES: 20, SKYGLOW: 21, LAMP_PAGES: 22, SEA_PAGES: 23, TINT_PAGES: 24, MOON_STRIP: 25, LAMP_SWITCH: 26 };
const FRAME = { CMD: 36 }, ITEM_BYTES = 64; // struct gfx_cmd, struct gfx_item; // the rest of struct frame's layout comes from hs_layout()
const HUD_BYTES = 136, SHARED_HUD = 64;
const EVENT = { START: 1, CAM_NEXT: 2, CAM_SET: 3, FLY_TOGGLE: 4, BACK_TO_ROAD: 5, INTRO: 6, BIRD: 7, GEARBOX: 16, ASSIST: 17, STILL: 18, BEAM: 19 };
const SET = { AUTO_DRIVE: 0, AUTO_FLY: 1, ORBIT: 2, TOP: 3, CAM: 4, PIXEL_RATIO: 5, TIME: 6, GUSTS: 7, HAZE: 8, TRAFFIC: 9, MEET: 10, TRUCK: 11, WATCH: 12, FOV: 13, PLOUGH: 14, PLOUGH2: 15, GRITTER: 16, LAMPS: 17, ORBIT_AT: 18, ORBIT_RISE: 19, COCKPIT: 20, AUTO_LAND: 21, LOOK_X: 22, LOOK_Y: 23,
  EVENT_X: 24, EVENT_Z: 25, EVENT: 26, EVENTS: 27, UNIT: 28, EVENT_BACK: 29, FOLLOW: 30, CABIN_LIGHT: 31, GLAZE: 32, BOOST_FROM: 33, BOOST_TO: 34, INTRO: 35, INTRO_AT: 36, WING: 37,
  AURORA: 38, HOUR: 39, ORBIT_ON_X: 40, ORBIT_ON_Y: 41, ORBIT_ON_Z: 42, ALOFT: 43, DAY: 44, CLOCK_RATE: 45, SLEIGH: 46, SNOWFALL: 47, FALLEN: 48, MEET_AT: 49, CLOUD: 50, WEATHER: 51, CRASH_X: 52, CRASH: 53 };
const EVENTS = { police: 1, ambulance: 2, stop: 3, crash: 4, fire: 5, chimney: 6, alarm: 7 }; // enum incident_kind (incident.h)
const UNITS = { p1: 0, p2: 1, p3: 2, ambulance: 3, engine: 4, ladder: 5, tow1: 6, tow2: 7 };    // enum rescue_who (rescue.h)
// ?sky=heli: the views of what is in the sky (wasm/include/harstad/game.h, enum game_sky), the clock set just before it happens
const SKY = { heli: 1, lift: 2, jet: 3, dash: 4, depart: 5, over: 6, sleigh: 7 };
let SAMPLES = 4; // 4x multisampling, or none with ?aa=1 (WebGPU has no counts between)
// ?scale=0.77 draws the frame at that share of the canvas's width and height and upscales it with FSR 1 (WGSL.FSR);
// ?sharp= its sharpening in stops, 0 the most, 0.2 by default, off for none
let SCALE = 1, SHARP = 2 ** -0.2;
const up = {}; // the upscale: its pipelines, the frame drawn small, EASU's output when RCAS follows, their bind groups and passes
let renderW = 0, renderH = 0; // the size the frame is drawn at
const WAKE_BYTES = 96 * 28, RINGS_BYTES = 12 * 16, DROPS_BYTES = 384 * 28; // and what the car stirs up: ribbon vertices, the rings and the jets' dimples, droplets
const SNOW_TRACKS = 28000, SNOW_RUN = 32, SNOW_RUNS = 512; // snow.h: the player's ribbon's ring, and the traffic's runs after it
const SEA_BYTES = 96, WAVES_BYTES = 1792; // struct water_frame: the sea's uniforms, then its ripple tiles' waves
const STAMPS = ['maps', 'moon', 'beam0', 'jets', 'gbuf', 'defer', 'road list', 'road light', 'main', 'shimmer', 'easu', 'rcas', 'clouds', 'cloud shade', 'cloud prep', 'present', ...Array.from({ length: 32 }, (_, i) => 'seg' + i)]; // the passes bench() times on the GPU (seg*: ?split, the main pass cut where it draws each part)
const FOREST_WORDS = 6; // the list of trees to draw holds at most this many words a tree (forest.h)
const TREE = { VARIANTS: 16, LEVELS: 4 }; // TREE_VARIANTS, TREE_LEVELS
const SHADOW_BYTES = 112; // a shadow map's uniforms: world to map, and the ground map's rectangle and tiles, which its trees stand on
const TILE_STEPS = 256, TILE_SIDE = TILE_STEPS + 1, TILE_POSTS = TILE_SIDE * TILE_SIDE; // the map's tiles: steps and posts a side, a block's posts (tile.h)

let device, context, format = 'rgba16float', canvasFormat, hdrWanted = false, hdr = false, canvas, hs, shared, keysView, padView, hudView, keys = 0, pad = new Float32Array(6);
let colourTarget, depthTarget, frameBuffer, frameGroup, pipes = {}, groups = {}, snowBuffer, quadIndex, treeQuad;
const winter = {}; // the snow on the roads (winter.h): the cells the shaders read
const snow = {}; // the snow the car drives through: field texture, lattice, spray, ribbon, and the banks it can flatten
const dune = {}, DUNE_NODES_MAX = 8192; // the drifts beyond it (dune.h): the corners round the eye, the tile, the nodes to draw
const shadow = {}; // the moon's shadow map and what draws into it
const sea = { wakeUntil: -1, dropsUntil: -1 }; // the fjord: its uniforms, the ripple tiles and the sky panorama redrawn each frame, the baked maps
const shine = {}; // every light that moves (wasm/src/shine.c): its records and its grid, sent each frame
const wet = {}; // the salted roads' wet tracks: the lights they mirror, and which of them each tile of the frame holds (WGSL.WET_BIN)
const herd = {}, HERD_CARS = 3584; // the traffic's bodies (wasm/include/harstad/traffic.h); TRAFFIC_CARS, struct traffic_seen's inst
// The clouds' shadow from the moon (WGSL.CLOUD_SHADE): two layers, what they let through and where the lowest begins,
// drawn every frame from the boxes as they are (the clouds drift), before anything that is lit by the moon
const cloudShade = {};
function cloudShadeInit(field) { // and after its two layers the moon's field (WGSL moon_map, MOON_FIRST): one texture fewer for the frame
  const px = WGSL.CLOUD_SHADE_PX;
  if (field.width !== px || field.height !== px || field.format !== 'r16float') throw new Error('the moon field\'s layers are not the clouds\' shade\'s');
  cloudShade.map = device.createTexture({ size: [px, px, WGSL.MOON_FIRST + field.depthOrArrayLayers], format: 'r16float',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  const copy = device.createCommandEncoder();
  copy.copyTextureToTexture({ texture: field }, { texture: cloudShade.map, origin: [0, 0, WGSL.MOON_FIRST] }, [px, px, field.depthOrArrayLayers]);
  device.queue.submit([copy.finish()]);
  field.destroy();
  const layer = i => cloudShade.map.createView({ dimension: '2d', baseArrayLayer: i, arrayLayerCount: 1 });
  cloudShade.pass = { colorAttachments: [{ view: layer(0), loadOp: 'clear', storeOp: 'store', clearValue: [1, 1, 1, 1] },
    { view: layer(1), loadOp: 'clear', storeOp: 'store', clearValue: [65504, 65504, 65504, 65504] }] };
  const frame = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: {} }] });
  cloudShade.frame = device.createBindGroup({ layout: frame, entries: [{ binding: 0, resource: { buffer: frameBuffer, offset: 0, size: FRAME.UNIFORMS } }] });
  const module = device.createShaderModule({ code: WGSL.CLOUD_SHADE });
  module.getCompilationInfo().then(info => { for (const e of info.messages) if (e.type === 'error') post({ error: `cloud shade shader, line ${e.lineNum}: ${e.message}` }); });
  const times = { color: { srcFactor: 'dst', dstFactor: 'zero' }, alpha: { srcFactor: 'dst', dstFactor: 'zero' } };
  const least = { color: { operation: 'min', srcFactor: 'one', dstFactor: 'one' }, alpha: { operation: 'min', srcFactor: 'one', dstFactor: 'one' } };
  cloudShade.pipe = device.createRenderPipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [frame, groupLayouts.cloud] }),
    vertex: { module }, fragment: { module, targets: [{ format: 'r16float', blend: times }, { format: 'r16float', blend: least }] },
    primitive: { cullMode: 'back', frontFace: 'ccw' } });
  // (cleared at once: the frame's group reads it before the first clouds are drawn)
  const enc = device.createCommandEncoder(); enc.beginRenderPass(cloudShade.pass).end(); device.queue.submit([enc.finish()]);
}
function cloudShadePass(enc, clouds) {
  const pass = enc.beginRenderPass(stamped(cloudShade.pass, 'cloud shade'));
  if (clouds) { pass.setPipeline(cloudShade.pipe); pass.setBindGroup(0, cloudShade.frame); pass.setBindGroup(1, live.group); pass.draw(6, clouds * WGSL.CLOUD_PUFFS); }
  pass.end();
}
const live = {}; // what moves in the sky: the mesh of the tower's lights and Father Christmas, their sprites, the clouds' boxes
const CLOUDS = 736, CLOUD_BOX = 32; // wasm/include/harstad/clouds.h: the most cloud boxes drawn (CLOUDS_MOST), and the bytes of a struct cloud_box
let car = {}; // the car: its mesh, its uniform block, the instrument cluster's texture and what it last showed
let rowIndex, outerRowIndex, groundLod; // the ground's strips at each level of detail; each chunk's lambda (hs_ground_lod)
const smoke = { count: 0 }; // the chimneys' smoke (smoke.h): the stoves, their plumes' order each frame, the simulated puffs' pushes
const SMOKE = { PUFFS: 32, SLOTS: 128, PUFF: 32, STEAM: 4096 }; // SMOKE_PUFFS, SMOKE_SLOTS, the bytes a puff sends, SMOKE_STEAM
const grit = {}; // the gritted track (plough.h): the ring of its sections, and how far into it has been laid
const cockpit = {}; // the cabin made over to fly (cockpit.h): its mesh, its two uniform blocks, which indices each block draws
const wing = {}; // the doors folded out into wings and the canopy (wing.h): its mesh, its two uniform blocks, which indices each draws, whether it shows
const rocket = {}; // the jets in the engines under the wings (rocket.h): their fans' and nozzles' mesh and uniform block, the flames' instances, the ring of blown-snow puffs
const ROCKET = { JET: 80, FLAMES: 7, LOW: 4, PUFF: 32, PUFFS: 4096 }; // struct rocket_jet, ROCKET_FLAMES (the four lift jets', the two cruise turbines' and the boost turbine's), the wings' layers in the flames' small pictures (WGSL.JETS_LOW), struct drift_grain, DRIFT_GRAINS
ROCKET.LAYOUT = [{ arrayStride: ROCKET.JET, stepMode: 'instance', attributes: [0, 1, 2, 3, 4].map(i => ({ shaderLocation: i, offset: 16 * i, format: 'float32x4' })) }];
const buffers = [];
let snowLayers = [], timing = null, forestList = null, forestAt = 0; // the buffer the forest's list is kept in, and where the C makes it
let skip = {}; // ?skip=terrain,sky,... leaves parts of the frame out, to see what each costs
const marks = { start: performance.now() }, stats = { frames: 0, draws: 0, herd: [0, 0], cpu: 0, split: [0, 0, 0, 0], times: new Float32Array(4096), n: 0, up: {},
  parts: new Float32Array(4096 * 4), pn: 0 }; // each frame's c, upload, encode, submit, for report()'s spikes
const mark = name => { marks[name] = performance.now() - marks.start; };
const post = (m, t) => postMessage(m, t || []);

/* ---- pipelines made a few at a time ---- */
// At most PIPES_AT_ONCE of the pipelines made asynchronously are being compiled at any time, the rest waiting their turn.
// The GPU process holds each compile's working memory while it runs, and the game asks for some 200 at once as it
// starts: on tsp2's Intel iGPU (Chrome on Mesa's ANV) that crashed Chrome's GPU process (SIGTRAP, in a thread-pool
// thread) on every load, and four at a time loads every time. ?pipes=N for another number.
function pipesAFew(device, n) {
  let busy = 0; const waiting = [];
  const free = () => { const next = waiting.shift(); if (next) next(); else busy--; };
  for (const name of ['createRenderPipelineAsync', 'createComputePipelineAsync']) {
    const f = device[name].bind(device);
    device[name] = async d => {
      if (busy < n) busy++; else await new Promise(go => waiting.push(go));
      try { return await f(d); } finally { free(); }
    };
  }
}

/* ---- the device's limits ---- */
// The game asks the device for no more than WebGPU's defaults, which every GPU offers: 16 sampled textures a shader stage,
// 8 storage buffers, 4 storage textures, 4 bind groups. Every pipeline is checked as it is made, against what its layout
// gives each stage (or, for a layout of 'auto', what its shaders use): one over is reported as an error, with what it
// asks. ?limits lists every pipeline's counts on the page (window.__next.limits).
let pipeTable = null;
function watchLimits(device) {
  const codes = new WeakMap(), groupsOf = new WeakMap(), pipeGroups = new WeakMap();
  const wrap = (name, after) => { const f = device[name].bind(device); device[name] = d => after(f(d), d); };
  wrap('createShaderModule', (m, d) => { codes.set(m, d.code); return m; });
  wrap('createBindGroupLayout', (l, d) => { groupsOf.set(l, d.entries); return l; });
  wrap('createPipelineLayout', (l, d) => { pipeGroups.set(l, d.bindGroupLayouts.map(g => groupsOf.get(g) || [])); return l; });
  const KIND = { texture: { texture: {} }, sampler: { sampler: {} }, storageTexture: { storageTexture: {} }, storage: { buffer: { type: 'storage' } }, uniform: { buffer: {} } };
  const check = d => {
    const stages = [['vertex', d.vertex], ['fragment', d.fragment], ['compute', d.compute]].filter(s => s[1] && codes.has(s[1].module))
      .map(([stage, s]) => ({ stage, code: codes.get(s.module), entry: s.entryPoint }));
    const label = stages.map(s => `${s.stage}:${entryFor(s.code, s.stage, s.entry)}`).join(' ') + ' ' + (stages[0]?.code.match(/\bfn\s+\w+/g) || []).slice(-2).join(',');
    let groups;
    if (d.layout === 'auto') { // what the shaders use, a group for each group they name
      const all = new Set(stages.flatMap(s => { const a = analyse(s.code); return [...a.uses(entryFor(s.code, s.stage, s.entry))].map(id => a.bindings.get(id).group); }));
      groups = [...all].sort((a, b) => a - b).map(g => [...groupUse(stages, g)].map(([binding, visibility]) => {
        const s = stages.find(s => [...analyse(s.code).bindings.values()].some(b => b.group === g && b.binding === binding));
        const b = [...analyse(s.code).bindings.values()].find(b => b.group === g && b.binding === binding);
        return { binding, visibility, ...KIND[b.kind] };
      }));
    } else groups = pipeGroups.get(d.layout) || [];
    const over = overLimits(groups);
    if (pipeTable) {
      const used = [0, 1, 2, 3].map(g => [...groupUse(stages, g)].map(([binding, visibility]) => ({ ...(groups[g] || []).find(e => e.binding === binding), visibility })));
      pipeTable.push({ label, counts: layoutCounts(groups), used: layoutCounts(used), over, code: stages.at(-1)?.code });
      clearTimeout(pipeTable.timer); pipeTable.timer = setTimeout(() => post({ limits: pipeTable.map(({ label, counts, used, over }) => ({ label, counts, used, over })) }), 500);
    }
    if (over.length) post({ error: `a pipeline asks more of the device than WebGPU's defaults (${over.join('; ')}): ${label}` });
  };
  for (const name of ['createRenderPipeline', 'createRenderPipelineAsync', 'createComputePipeline', 'createComputePipelineAsync'])
    wrap(name, (p, d) => { try { check(d); } catch (e) { post({ error: `limits: ${e.message}` }); } return p; });
}

// For the gate (tools/limits.mjs): every pipeline makePipelines makes, on a device that only records what it is asked
// for, each with what its layout gives each stage: [{ label, counts, used, over }]
export async function pipelineLimits(recorder) {
  device = recorder; pipeTable = []; watchLimits(device);
  await makePipelines();
  clearTimeout(pipeTable.timer);
  return pipeTable;
}

/* ---- small WebGPU helpers ---- */

// What the textures made hold, made and not yet destroyed, by what they are called (`label`, or 'other'): the GPU's
// memory is a limit the map's growth must stay under, so it is measured (report(), the marks at ready).
const textureBytes = new Map();
const TEXEL = { r8: 1, rg8: 2, rgba8: 4, bgra8: 4, r16: 2, rg16: 4, rgba16: 8, r32: 4, rg32: 8, rgba32: 16, depth16: 2, depth24: 4, depth32: 4, bc7: 1 };
function tallied(dev) {
  const make = dev.createTexture.bind(dev);
  dev.createTexture = d => {
    const t = make(d), [w, h = 1, n = 1] = Array.isArray(d.size) ? d.size : [d.size.width, d.size.height, d.size.depthOrArrayLayers];
    const texel = TEXEL[d.format.match(/^[a-z]+\d+/)[0]] || 4, cube = d.dimension === '3d', label = d.label || 'other';
    let bytes = 0;
    for (let l = 0; l < (d.mipLevelCount || 1); l++) bytes += Math.max(1, w >> l) * Math.max(1, h >> l) * (cube ? Math.max(1, n >> l) : n) * texel;
    bytes *= d.sampleCount || 1;
    textureBytes.set(label, (textureBytes.get(label) || 0) + bytes);
    const destroy = t.destroy.bind(t);
    let gone = false;
    t.destroy = () => { if (!gone) { gone = true; textureBytes.set(label, textureBytes.get(label) - bytes); } destroy(); };
    return t;
  };
}
const texturesMB = () => Object.fromEntries([...textureBytes, ['all', [...textureBytes.values()].reduce((a, b) => a + b, 0)]].map(([k, v]) => [k, +(v / 1048576).toFixed(1)]));

const mips = (w, h) => 1 + Math.floor(Math.log2(Math.max(w, h)));
let mipPipes = new Map(), mipSampler, pyramidPipes = new Map();
function mipPipe(f) {
  if (!mipPipes.has(f)) {
    const module = device.createShaderModule({ code: WGSL.MIP });
    mipPipes.set(f, device.createRenderPipeline({ layout: 'auto', vertex: { module }, fragment: { module, targets: [{ format: f }] } }));
    mipSampler ||= device.createSampler({ minFilter: 'linear', magFilter: 'linear' });
  }
  return mipPipes.get(f);
}
// The haze's lamp map (lamp_glow): the lamp map's first mip and the one under it, in tiles of their own (lamp.h,
// LAMP_GLOW_*), made from the lamp map's tiles as makeMips would make them from the map whole (WGSL.LAMP_GLOW, GLOW_MIP).
// Which tiles have a slot, and which: channel 2 of the tint's page table `tp` (tw by th), at each glow tile's first tint tile.
function lampGlowFrom(tiles, pages, w, h, tp, tw, th) {
  const SIDE = 248, SLOT = 256, LAYER = 1024, PER = (LAYER / SLOT) ** 2, LEVELS = 2;
  const gw = Math.max(1, w >> 1), gh = Math.max(1, h >> 1), nx = Math.ceil(gw / SIDE), ny = Math.ceil(gh / SIDE), page = new Uint16Array(tp.buffer || tp, tp.byteOffset || 0, tw * th * 4);
  const at = [];
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) { const k = page[((j * 4) * tw + i * 4) * 4 + 2]; if (k !== 0xffff) at[k] = [i, j]; }
  const n = at.length, layers = Math.max(1, Math.ceil(n / PER));
  const texture = device.createTexture({ label: 'lamp glow', size: [LAYER, LAYER, layers], format: 'rgba8unorm', mipLevelCount: LEVELS,
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
  const list = gpuBuffer(new Uint32Array(Math.max(2, n * 2)).map((_, q) => at[q >> 1] ? at[q >> 1][q & 1] : 0), GPUBufferUsage.STORAGE);
  const pipe = code => { const module = device.createShaderModule({ code }); return device.createRenderPipeline({ layout: 'auto', vertex: { module }, fragment: { module, targets: [{ format: 'rgba8unorm' }] } }); };
  const first = pipe(WGSL.LAMP_GLOW), down = pipe(WGSL.GLOW_MIP), samp = device.createSampler({ minFilter: 'linear', magFilter: 'linear' });
  const view = (layer, l) => texture.createView({ dimension: '2d', baseMipLevel: l, mipLevelCount: 1, baseArrayLayer: layer, arrayLayerCount: 1 });
  const enc = device.createCommandEncoder();
  for (let layer = 0; layer < layers; layer++) for (let l = 0; l < LEVELS; l++) {
    const p = l ? down : first, u = gpuBuffer(new Uint32Array([...new Uint32Array(new Float32Array([w, h, gw, gh]).buffer), layer, n, l, 0]), GPUBufferUsage.UNIFORM);
    const pass = enc.beginRenderPass({ colorAttachments: [{ view: view(layer, l), loadOp: 'clear', storeOp: 'store' }] });
    pass.setPipeline(p);
    pass.setBindGroup(0, device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: [
      ...(l ? [{ binding: 0, resource: view(layer, l - 1) }] : [{ binding: 0, resource: arrayView(tiles) }, { binding: 1, resource: pages.createView() }]),
      { binding: 2, resource: samp }, { binding: 3, resource: { buffer: u } }, { binding: 4, resource: { buffer: list } }] }));
    pass.draw(3);
    pass.end();
  }
  device.queue.submit([enc.finish()]);
  return texture;
}

function makeMips(texture) {
  const { format: f, mipLevelCount, depthOrArrayLayers: layers } = texture;
  const pipeline = mipPipe(f), enc = device.createCommandEncoder();
  for (let layer = 0; layer < layers; layer++) for (let level = 1; level < mipLevelCount; level++) {
    const view = l => texture.createView({ dimension: '2d', baseMipLevel: l, mipLevelCount: 1, baseArrayLayer: layer, arrayLayerCount: 1 });
    const pass = enc.beginRenderPass({ colorAttachments: [{ view: view(level), loadOp: 'clear', storeOp: 'store' }] });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: view(level - 1) }, { binding: 1, resource: mipSampler }] }));
    pass.draw(3);
    pass.end();
  }
  device.queue.submit([enc.finish()]);
}

// An array texture from same-sized { width, height, data } images (or ImageBitmaps), with mips.
function arrayTexture(images, fmt, label) {
  const { width, height } = images[0];
  const texture = device.createTexture({ label,
    size: [width, height, images.length], format: fmt, mipLevelCount: mips(width, height),
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });
  images.forEach((im, layer) => {
    if (im.data) device.queue.writeTexture({ texture, origin: [0, 0, layer] }, im.data, { bytesPerRow: width * 4 }, [width, height]);
    else device.queue.copyExternalImageToTexture({ source: im }, { texture, origin: [0, 0, layer] }, [width, height]);
  });
  makeMips(texture);
  return texture;
}
const arrayView = t => t.createView({ dimension: '2d-array' });
// A cube texture from cloudVolume()'s levels, each a cube half the last's side.
function volumeTexture({ size, levels }) {
  const texture = device.createTexture({ size: [size, size, size], dimension: '3d', format: 'rgba8unorm', mipLevelCount: levels.length,
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  levels.forEach((data, mipLevel) => { const n = size >> mipLevel; device.queue.writeTexture({ texture, mipLevel }, data, { bytesPerRow: n * 4, rowsPerImage: n }, [n, n, n]); });
  return texture;
}

// Textures drawn into every frame, with their mip chains: `count` of one size (one or two), halved together. Returns the
// textures, their top levels to draw into, and a function that adds the halving to a compute pass: a dispatch a level,
// all in the one pass. (As render passes, a pass a level, the chains were some twenty passes a frame; on a tile-based
// GPU each pass costs its setup and its flush to memory whatever it draws, and these draw almost nothing.) Each texel
// is the mean of the 2 x 2 above it, what a bilinear sample between them gave. Views and bind groups are made once.
let halvePipes = new Map();
function drawnTextures(count, width, height, fmt) {
  const levels = mips(width, height), last = levels - 1;
  const textures = Array.from({ length: count }, () => device.createTexture({ size: [width, height], format: fmt, mipLevelCount: levels,
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING }));
  const key = fmt + count;
  if (!halvePipes.has(key)) {
    const module = device.createShaderModule({ code: WGSL.halve(count, fmt) });
    halvePipes.set(key, device.createComputePipeline({ layout: 'auto', compute: { module } }));
  }
  const pipeline = halvePipes.get(key);
  const views = textures.map(t => Array.from({ length: levels }, (_, l) => t.createView({ baseMipLevel: l, mipLevelCount: 1 })));
  const from = Array.from({ length: last }, (_, l) => device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
    entries: views.flatMap((v, i) => [{ binding: i * 2, resource: v[l] }, { binding: i * 2 + 1, resource: v[l + 1] }]) }));
  const groups = Array.from({ length: last }, (_, l) => [Math.ceil(Math.max(1, width >> (l + 1)) / 8), Math.ceil(Math.max(1, height >> (l + 1)) / 8)]);
  return { textures, tops: views.map(v => v[0]), halve(pass) {
    pass.setPipeline(pipeline);
    for (let l = 0; l < last; l++) { pass.setBindGroup(0, from[l]); pass.dispatchWorkgroups(groups[l][0], groups[l][1]); }
  } };
}

// The forest's impostors (WGSL.IMPOSTOR_BAKE): every variant's near model from every view, a layer a variant, drawn 4x
// multisampled and resolved, so a view's outline is as smooth as the model's on screen; then their mips, cover-weighted.
// Once, at load. Returns the colour texture and the one of normal, depth and cover, as 2d-array views.
function bakeImpostors(models, variants, levels) {
  const size = WGSL.IMP_VIEWS * WGSL.IMP_TILE, mipLevelCount = WGSL.IMP_MIPS, usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT;
  const col = device.createTexture({ size: [size, size, variants], format: 'rgba8unorm-srgb', mipLevelCount, usage });
  const nrm = device.createTexture({ size: [size, size, variants], format: 'rgba8unorm', mipLevelCount, usage });
  const msaa = fmt => device.createTexture({ size: [size, size], format: fmt, sampleCount: 4, usage: GPUTextureUsage.RENDER_ATTACHMENT });
  const colMs = msaa('rgba8unorm-srgb'), nrmMs = msaa('rgba8unorm'), depth = msaa('depth32float');
  const module = device.createShaderModule({ code: WGSL.IMPOSTOR_BAKE });
  const layout = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, texture: { sampleType: 'unfilterable-float' } },
    { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: {} }] });
  const over = { color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' }, alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' } };
  const cover = { color: { srcFactor: 'zero', dstFactor: 'one' }, alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' } };
  const pipe = (entryPoint, blend, write) => device.createRenderPipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
    vertex: { module, entryPoint: 'vs' }, fragment: { module, entryPoint, targets: [{ format: 'rgba8unorm-srgb', blend }, { format: 'rgba8unorm',
      blend: write ? undefined : cover, writeMask: write ? GPUColorWrite.ALL : GPUColorWrite.ALPHA }] },
    primitive: { cullMode: 'none' }, depthStencil: { format: 'depth32float', depthWriteEnabled: write, depthCompare: 'less' }, multisample: { count: 4 } });
  const solid = pipe('solid', undefined, true), haze = pipe('haze', over, false);
  const sizes = new Uint32Array(hs.memory.buffer, hs.hs_tree_sizes(), variants * levels * 2 + 1); // struct tree_sizes: solid, blended, where the solid begins
  const at = sizes[variants * levels * 2], enc = device.createCommandEncoder();
  for (let v = 0; v < variants; v++) {
    const uniform = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(uniform, 0, new Uint32Array([v, 0, 0, 0]));
    const group = device.createBindGroup({ layout, entries: [{ binding: 0, resource: models }, { binding: 1, resource: { buffer: uniform } }] });
    const layer = t => t.createView({ dimension: '2d', baseArrayLayer: v, arrayLayerCount: 1, baseMipLevel: 0, mipLevelCount: 1 });
    const pass = enc.beginRenderPass({ colorAttachments: [
      { view: colMs.createView(), resolveTarget: layer(col), clearValue: [0, 0, 0, 0], loadOp: 'clear', storeOp: 'discard' },
      { view: nrmMs.createView(), resolveTarget: layer(nrm), clearValue: [0.5, 0.5, 0.5, 0], loadOp: 'clear', storeOp: 'discard' }],
      depthStencilAttachment: { view: depth.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'discard' } });
    pass.setBindGroup(0, group);
    const n = WGSL.IMP_VIEWS * WGSL.IMP_VIEWS, solidN = sizes[v * levels], hazeN = sizes[variants * levels + v * levels];
    pass.setPipeline(solid); pass.draw(solidN, n, at);
    if (hazeN) { pass.setPipeline(haze); pass.draw(hazeN, n, at - hazeN); }
    pass.end();
  }
  const mip = device.createShaderModule({ code: WGSL.IMPOSTOR_MIP });
  const mipPipe = device.createRenderPipeline({ layout: 'auto', vertex: { module: mip }, fragment: { module: mip, targets: [{ format: 'rgba8unorm-srgb' }, { format: 'rgba8unorm' }] } });
  for (let v = 0; v < variants; v++)
    for (let l = 1; l < mipLevelCount; l++) {
      const view = (t, level) => t.createView({ dimension: '2d', baseArrayLayer: v, arrayLayerCount: 1, baseMipLevel: level, mipLevelCount: 1 });
      const pass = enc.beginRenderPass({ colorAttachments: [{ view: view(col, l), loadOp: 'clear', clearValue: [0, 0, 0, 0], storeOp: 'store' },
        { view: view(nrm, l), loadOp: 'clear', clearValue: [0.5, 0.5, 0.5, 0], storeOp: 'store' }] });
      pass.setPipeline(mipPipe);
      pass.setBindGroup(0, device.createBindGroup({ layout: mipPipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: view(col, l - 1) }, { binding: 1, resource: view(nrm, l - 1) }] }));
      pass.draw(3);
      pass.end();
    }
  device.queue.submit([enc.finish()]);
  return { col: col.createView({ dimension: '2d-array' }), nrm: nrm.createView({ dimension: '2d-array' }) };
}

function dataTexture(fmt, width, height, data, bytesPerRow) {
  const texture = device.createTexture({ size: [width, height], format: fmt, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  device.queue.writeTexture({ texture }, data, { bytesPerRow }, [width, height]);
  return texture;
}

// A texture of layers w by h, as many as the data fills, `bytes` a texel: the town-wide maps' tiles (wasm/src/tiles.c).
// `data` may be a list of buffers, each whole layers, the atlas's in order: the moon's field and the sky's map come from
// the build a few layers at a time (wasm/src/build.c, build__trees), and are laid into one texture here, each part as it is.
function layerTexture(fmt, width, height, data, bytes, label, usage = 0) {
  const parts = Array.isArray(data) ? data : [data], per = width * height * bytes;
  const layers = parts.reduce((n, d) => n + d.byteLength / per, 0);
  const texture = device.createTexture({ label, size: [width, height, layers], format: fmt, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | usage });
  let at = 0;
  for (const d of parts) {
    const n = d.byteLength / per;
    device.queue.writeTexture({ texture, origin: [0, 0, at] }, d, { bytesPerRow: width * bytes, rowsPerImage: height }, [width, height, n]);
    at += n;
  }
  return texture;
}

// A map's layers that come a few at a time (the moon's field, the sky's): each part sent to the GPU as it comes, in a
// texture of its own, and the parts copied into one texture of them all once the last is in, on the GPU
function layerPart(fmt, b, bytes, label) {
  return { w: b.w, h: b.h, layers: b.data.byteLength / (b.w * b.h * bytes), texture: layerTexture(fmt, b.w, b.h, b.data, bytes, label, GPUTextureUsage.COPY_SRC) };
}
// A map baked by bands (wasm/src/build.c, build_band), its slots numbered from 0 in each: the bands' layers one after
// another into `parts`, each part's first layer as `at`, and each band's rows of tiles into the page table `into`'s
// channels chan and chan + 1, its slots moved on by the layers before it. A band whose slots would then take a number
// with the mark of none in its low half (tiles.h) begins a layer later, the layer between left empty.
function joinBands(bands, parts, into, chan) {
  let base = 0;
  for (const { rows: [ty0, ty1], parts: own, pages } of bands) {
    const from = pages.data, e0 = ty0 * pages.w * 4, e1 = ty1 * pages.w * 4, PER = 256;
    for (let clash = true; clash;) {
      clash = false;
      for (let e = e0; e < e1 && !clash; e += 4) if (from[e + chan] !== 0xffff && ((base * PER + (from[e + chan] | from[e + chan + 1] << 16)) & 0xffff) === 0xffff) clash = true;
      if (clash) base++;
    }
    for (let e = e0; e < e1; e += 4) {
      if (from[e + chan] === 0xffff) { into.data[e + chan] = 0xffff; into.data[e + chan + 1] = from[e + chan + 1]; continue; }
      const n = base * PER + (from[e + chan] | from[e + chan + 1] << 16);
      into.data[e + chan] = n & 0xffff; into.data[e + chan + 1] = n >>> 16;
    }
    for (const p of own) { p.at = base; base += p.layers; parts.push(p); }
  }
}
function joinLayers(fmt, parts, label, room = 0) { // room: the least layers it is to have (the map's levels over the first after them)
  const layers = parts.reduce((n, p) => Math.max(n, (p.at ?? n) + p.layers), 0);
  if (parts.length === 1 && room <= layers && !parts[0].at) return parts[0].texture;
  const texture = device.createTexture({ label, size: [parts[0].w, parts[0].h, Math.max(layers, room)], format: fmt,
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC });
  const enc = device.createCommandEncoder();
  let at = 0;
  for (const p of parts) { at = p.at ?? at; enc.copyTextureToTexture({ texture: p.texture }, { texture, origin: [0, 0, at] }, [p.w, p.h, p.layers]); at += p.layers; }
  device.queue.submit([enc.finish()]);
  parts.forEach(p => p.texture.destroy()); // (freed once the copies are done)
  return texture;
}

// A town-wide map's coarser levels (WGSL PYR_PAGES): from the first level's page table (`pages`, { w, h, data } of
// rgba16uint, its maps' tiles of size[0] by size[1] texels) the table of every level, level l's tiles in mip l of a table
// a power of two across and down. A tile of a level over the first is its value as the tiles under it are, when every
// texel it and its ring are made from lies in tiles all of one and the same value; any other takes a slot after those
// taken already. chans: { chan, kind } of each map in the table. Returns the table's levels and its texture, and for
// each map the slots it then takes in all and each level's new slots ([slot, column, row, 0] each), for pyramidFill.
function pyramid(pages, size, chans) {
  const SIDE = 62, sizes = [size];
  while (Math.max(...sizes.at(-1)) > SIDE) sizes.push(sizes.at(-1).map(n => Math.ceil(n / 2)));
  const L = sizes.length, tiles = sizes.map(s => s.map(n => Math.ceil(n / SIDE)));
  if (tiles[0][0] > pages.w || tiles[0][1] > pages.h) throw new Error('a town-wide map: its page table is smaller than the map');
  const P = [pages.w, pages.h].map(n => 2 ** Math.ceil(Math.log2(Math.max(n, 2 ** (L - 1)))));
  const dims = sizes.map((_, l) => P.map(n => Math.max(1, n >> l))), out = dims.map(([w, h]) => new Uint16Array(w * h * 4));
  const src = new Uint16Array(pages.data.buffer || pages.data, pages.data.byteOffset || 0, pages.w * pages.h * 4);
  for (let j = 0; j < pages.h; j++) out[0].set(src.subarray(j * pages.w * 4, (j + 1) * pages.w * 4), j * P[0] * 4);
  const maps = chans.map(({ chan, kind }) => {
    let n = 0;
    for (let e = 0; e < src.length; e += 4) if (src[e + chan] !== 0xffff) n = Math.max(n, (src[e + chan] | src[e + chan + 1] << 16) + 1);
    const levels = [[]], words = kind === 2 ? 2 : 1;
    for (let l = 1; l < L; l++) {
      const list = [], [sw, sh] = sizes[l], [uw, uh] = sizes[l - 1], under = out[l - 1], uW = dims[l - 1][0], here = out[l], W = dims[l][0];
      for (let j = 0; j < tiles[l][1]; j++) for (let i = 0; i < tiles[l][0]; i++) {
        // the texels under this tile and its ring, and the tiles they lie in
        const x0 = Math.floor(2 * Math.max(i * SIDE - 1, 0) / SIDE), x1 = Math.floor(Math.min(2 * Math.min(i * SIDE + SIDE, sw - 1) + 1, uw - 1) / SIDE);
        const y0 = Math.floor(2 * Math.max(j * SIDE - 1, 0) / SIDE), y1 = Math.floor(Math.min(2 * Math.min(j * SIDE + SIDE, sh - 1) + 1, uh - 1) / SIDE);
        let one = true, v = null;
        for (let y = y0; y <= y1 && one; y++) for (let x = x0; x <= x1 && one; x++) {
          const e = (y * uW + x) * 4 + chan;
          if (under[e] !== 0xffff) one = false;
          else if (v === null) v = [under[e + 1], words > 1 ? under[e + 2] : 0];
          else one = v[0] === under[e + 1] && (words < 2 || v[1] === under[e + 2]);
        }
        const e = (j * W + i) * 4 + chan;
        if (one) { here[e] = 0xffff; here[e + 1] = v[0]; if (words > 1) here[e + 2] = v[1]; continue; }
        if ((n & 0xffff) === 0xffff) n++; // no slot has the mark of none in its low half (tiles.h)
        here[e] = n & 0xffff; here[e + 1] = n >>> 16; list.push(n, i, j, 0); n++;
      }
      levels.push(list);
    }
    return { slots: n, levels };
  });
  const texture = device.createTexture({ size: P, format: 'rgba16uint', mipLevelCount: L, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  out.forEach((d, l) => device.queue.writeTexture({ texture, mipLevel: l }, d, { bytesPerRow: dims[l][0] * 8 }, dims[l]));
  return { texture, size, maps, levels: out.map((data, l) => ({ w: dims[l][0], h: dims[l][1], data })) };
}
// The frame's page tables in one texture (WGSL.TABLES), so that no shader reads more than WebGPU's default sixteen
// textures a stage: each table's levels laid in rows under the first, whose texels say where each lies (its corner, its
// width and height) and how many levels a table has. A table's texel is as its own texture's was: four 16-bit channels
// (a pyramid's), four 32-bit (the sea's), or one 32-bit value, the bits of an i32 or f32. All are made once, at the load.
// tables: { name: [level 0, 1, ...] }, each { w, h, data } with data a Uint16Array or Uint32Array of 1 or 4 values a texel.
function tableAtlas(tables, firsts) { // firsts: { reachFirst: layer, ... }, where a map begins in the texture it shares (joinMaps)
  const T = WGSL.TABLES, levels = [], dir = new Uint32Array(T.entries * 4);
  for (const [k, first] of Object.entries(firsts)) dir[T[k] * 4] = first;
  for (const [name, list] of Object.entries(tables)) {
    if (list.length >= T.levels) throw new Error(`the ${name} table: ${list.length} levels, room for ${T.levels - 1}`);
    list.forEach((t, l) => levels.push({ e: T[name] + l, ...t }));
    dir[(T[name] + T.levels) * 4] = list.length;
  }
  // shelves, tallest first, as wide as the widest table (and the first row)
  const W = Math.max(T.entries, ...levels.map(t => t.w));
  let x = 0, y = 1, shelf = 0;
  for (const t of [...levels].sort((a, b) => b.h - a.h)) {
    if (x + t.w > W) { x = 0; y += shelf; shelf = 0; }
    t.x = x; t.y = y; x += t.w; shelf = Math.max(shelf, t.h);
    dir.set([t.x, t.y, t.w, t.h], t.e * 4);
  }
  const H = y + shelf;
  if (H > device.limits.maxTextureDimension2D) throw new Error(`the page tables: ${W} x ${H} texels, more than a texture holds`);
  const texture = device.createTexture({ label: 'tables', size: [W, H], format: 'rgba32uint', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  device.queue.writeTexture({ texture }, dir, { bytesPerRow: T.entries * 16 }, [T.entries, 1]);
  for (const t of levels) {
    const n = t.w * t.h, per = t.data.length / n, out = new Uint32Array(n * 4);
    if (per === 4) out.set(t.data); else for (let i = 0; i < n; i++) out[i * 4] = t.data[i];
    device.queue.writeTexture({ texture, origin: [t.x, t.y] }, out, { bytesPerRow: t.w * 16 }, [t.w, t.h]);
  }
  return texture;
}
// Maps of one format and layer size, in tiles, one after another in one texture, every mip of each (WGSL pair_map,
// lamp_map), so that a shader reads fewer textures: { texture, firsts }, each map's first layer. Their own are let go.
function joinMaps(maps, label) {
  const { width, height, format } = maps[0], firsts = [];
  let layers = 0;
  for (const m of maps) {
    if (m.width !== width || m.height !== height || m.format !== format) throw new Error(`the ${label} maps differ in size or format`);
    firsts.push(layers); layers += m.depthOrArrayLayers;
  }
  if (layers > device.limits.maxTextureArrayLayers) throw new Error(`the ${label} maps: ${layers} layers, more than a texture holds`);
  const texture = device.createTexture({ label, size: [width, height, layers], format, mipLevelCount: Math.max(...maps.map(m => m.mipLevelCount)),
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  const enc = device.createCommandEncoder();
  maps.forEach((m, i) => { for (let l = 0; l < m.mipLevelCount; l++)
    enc.copyTextureToTexture({ texture: m, mipLevel: l }, { texture, mipLevel: l, origin: [0, 0, firsts[i]] }, [Math.max(1, width >> l), Math.max(1, height >> l), m.depthOrArrayLayers]); });
  device.queue.submit([enc.finish()]);
  maps.forEach(m => m.destroy());
  return { texture, firsts };
}
// A map's bytes a texel widened to two, the second 0
const widen = data => { const a = new Uint8Array(data.buffer || data, data.byteOffset || 0, data.byteLength), out = new Uint8Array(a.length * 2); for (let i = 0; i < a.length; i++) out[i * 2] = a[i]; return out; };
// Make the levels pyramid() laid out: level by level, each slot drawn into a layer laid out as the atlas's from the
// level under it (WGSL PYR_MAKE), then copied into the atlas, which no pass both reads and draws into.
// steps: the atlas's steps (255 for eight bits) to dither each texel's rounding by, so a level keeps the light of the one
// under it where it is faint (the lamp map's far roads), or 0 to round plainly
function pyramidFill(atlas, pyr, m, chan, kind, steps = 0) {
  const pagesTex = pyr.texture, levels = pyr.maps[m].levels;
  const SLOT = 64, LAYER = 1024, ACROSS = LAYER / SLOT, PER = ACROSS * ACROSS;
  const pipe = pyramidPipes.get(atlas.format) || pyramidPipes.set(atlas.format, (m => device.createRenderPipeline({ layout: 'auto', vertex: { module: m }, fragment: { module: m, targets: [{ format: atlas.format }] } }))(
    device.createShaderModule({ code: WGSL.PYR_MAKE }))).get(atlas.format);
  const scratch = device.createTexture({ size: [LAYER, LAYER, 1], format: atlas.format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
  const view = atlas.createView({ dimension: '2d-array' }), pv = pagesTex.createView(), made = [];
  for (let l = 1; l < levels.length; l++) {
    const s = levels[l], n = s.length / 4;
    if (!n) continue;
    const slots = gpuBuffer(new Uint32Array(s), GPUBufferUsage.STORAGE), u = gpuBuffer(new Uint32Array([l, chan, kind, steps, pyr.size[0], pyr.size[1], 0, 0]), GPUBufferUsage.UNIFORM);
    const group = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: view }, { binding: 1, resource: pv },
      { binding: 3, resource: { buffer: u } }, { binding: 4, resource: { buffer: slots } }] });
    const enc = device.createCommandEncoder();
    for (let i = 0; i < n;) { // a pass a layer, a quad a slot; then each run of slots along a row of the layer copied at once
      const z = Math.floor(s[i * 4] / PER);
      let j = i;
      while (j < n && Math.floor(s[j * 4] / PER) === z) j++;
      const pass = enc.beginRenderPass({ colorAttachments: [{ view: scratch.createView(), loadOp: 'clear', storeOp: 'store' }] });
      pass.setPipeline(pipe); pass.setBindGroup(0, group); pass.draw(6, j - i, 0, i); pass.end();
      for (let a = i; a < j;) {
        const k = s[a * 4], row = Math.floor(k / ACROSS);
        let b = a + 1;
        while (b < j && s[b * 4] === s[(b - 1) * 4] + 1 && Math.floor(s[b * 4] / ACROSS) === row) b++;
        const x = k % ACROSS * SLOT, y = row % ACROSS * SLOT;
        enc.copyTextureToTexture({ texture: scratch, origin: [x, y, 0] }, { texture: atlas, origin: [x, y, z] }, [(b - a) * SLOT, SLOT, 1]);
        a = b;
      }
      i = j;
    }
    device.queue.submit([enc.finish()]);
    made.push(slots, u);
  }
  device.queue.onSubmittedWorkDone().then(() => { scratch.destroy(); made.forEach(b => b.destroy()); });
}
// A texture of a map's tiles with room for `slots` slots, its first layers copied from `from`, which is let go
function roomFor(from, slots, label) {
  const layers = Math.max(1, Math.ceil(slots / 256));
  if (layers <= from.depthOrArrayLayers) return from;
  const texture = device.createTexture({ label, size: [from.width, from.height, layers], format: from.format, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC });
  const enc = device.createCommandEncoder();
  enc.copyTextureToTexture({ texture: from }, { texture }, [from.width, from.height, from.depthOrArrayLayers]);
  device.queue.submit([enc.finish()]);
  from.destroy();
  return texture;
}

function gpuBuffer(data, usage) {
  const size = Math.max(4, (data.byteLength + 3) & ~3);
  const buffer = device.createBuffer({ size, usage, mappedAtCreation: true });
  new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data.buffer || data, data.byteOffset || 0, data.byteLength));
  buffer.unmap();
  return buffer;
}

/* ---- pipelines ---- */

const VTX = [{ shaderLocation: 0, offset: 0, format: 'float32x3' }, { shaderLocation: 1, offset: 12, format: 'snorm8x4' }, { shaderLocation: 2, offset: 16, format: 'unorm8x4' }];
const LAYOUT = {
  plain: [{ arrayStride: 20, attributes: VTX }],
  textured: [{ arrayStride: 28, attributes: [...VTX, { shaderLocation: 3, offset: 20, format: 'float32x2' }] }],
  instanced: [{ arrayStride: 20, attributes: VTX }, { arrayStride: 32, stepMode: 'instance', attributes: [
    { shaderLocation: 4, offset: 0, format: 'float32x4' }, { shaderLocation: 5, offset: 16, format: 'float32x3' }, { shaderLocation: 6, offset: 28, format: 'unorm8x4' }] }],
  glow: [{ arrayStride: 32, stepMode: 'instance', attributes: [
    { shaderLocation: 0, offset: 0, format: 'float32x4' }, { shaderLocation: 1, offset: 16, format: 'unorm8x4' }, { shaderLocation: 2, offset: 20, format: 'float32x2' },
    { shaderLocation: 3, offset: 28, format: 'float32' }] }],
  // struct avensis_vtx: position, normal and part, linear colour, texture coordinates, material and lamp
  car: [{ arrayStride: 40, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }, { shaderLocation: 1, offset: 12, format: 'snorm8x4' },
    { shaderLocation: 2, offset: 16, format: 'float32x3' }, { shaderLocation: 3, offset: 28, format: 'float32x2' }, { shaderLocation: 4, offset: 36, format: 'uint8x4' }] }],
  forest: [{ arrayStride: 4, stepMode: 'instance', attributes: [{ shaderLocation: 0, offset: 0, format: 'uint32' }] }], // a word of the forest's list: tree and level
  snow: [{ arrayStride: 20, stepMode: 'instance', attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x4' }, { shaderLocation: 1, offset: 16, format: 'float32' }] }],
};
const BLEND = {
  over: { color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' }, alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' } },
  add: { color: { srcFactor: 'src-alpha', dstFactor: 'one' }, alpha: { srcFactor: 'zero', dstFactor: 'one' } },
  // a lamp's lens over what is behind it: the brighter of the two, so that where many points of light fall on one pixel (a
  // Christmas tree's bulbs far off) it is as bright as the brightest, clipped there as the eye or a camera clips it, and does
  // not add up past white (GLOW); the frame's alpha is kept
  lens: { color: { operation: 'max', srcFactor: 'one', dstFactor: 'one' }, alpha: { operation: 'max', srcFactor: 'one', dstFactor: 'one' } },
  laid: { color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' }, alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' } },
  // the jets' glow added, and the frame's alpha cleared where their boxes show (WGSL.JETS_HAZE reads that)
  jets: { color: { srcFactor: 'one', dstFactor: 'one' }, alpha: { srcFactor: 'zero', dstFactor: 'one-minus-src-alpha' } },
};
const VF = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, FS = GPUShaderStage.FRAGMENT;
let frameLayout;
const groupLayouts = {}, seaKept = {};
// A group's layout entries cut to those a shader (its vertex and fragment entry points) reads, so that a pipeline's
// stages count no binding it never reads against the device's limits
const cutTo = (entries, code, group) => { const use = groupUse([{ stage: 'vertex', code }, { stage: 'fragment', code }], group); return entries.filter(e => use.has(e.binding)); };

async function makePipelines() {
  frameLayout = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: VF, buffer: {} },
    { binding: 1, visibility: VF, texture: { viewDimension: '2d-array' } }, // the lamp map's tiles (16 says where each is), then the haze's (a vertex shader finishes its sprites too)
    { binding: 2, visibility: VF, texture: { viewDimension: '2d-array' } }, // the terrain's heights, a layer a tile (WGSL GROUND_LOOK; 16 says which layer each tile is)
    { binding: 3, visibility: VF, texture: { viewDimension: '2d-array' } }, // the fine noise and the snow's relief, a layer each (VF: the twig haze is lit at its corners, WGSL.FOREST_HAZE)
    { binding: 4, visibility: VF, sampler: {} },
    { binding: 5, visibility: VF, sampler: {} },
    { binding: 6, visibility: VF, texture: { sampleType: 'depth' } },
    { binding: 7, visibility: VF, sampler: { type: 'comparison' } }, // the smoke's vertex shader lights each puff by the headlamps,
    { binding: 8, visibility: VF, texture: { sampleType: 'depth', viewDimension: '2d-array' } }, // whose beams' shadow map it reads
    { binding: 10, visibility: VF, texture: { viewDimension: '2d-array' } }, // the maps of two channels in tiles: the tint's, the moon's reach, the sky the ground sees (wasm/src/moon.c, ao.c)
    { binding: 11, visibility: VF, buffer: { type: 'read-only-storage' } }, // the aurora's rows (WGSL.SKY_MAPS): a cloud's puff takes their light in its vertex shader
    { binding: 16, visibility: VF, texture: { sampleType: 'uint' } }, // every map's page table, the ground's tiles and the backdrop's heights (WGSL.TABLES, tableAtlas)
    { binding: 17, visibility: VF, texture: {} }, // the town's light given back round it (wasm/src/skyglow.c): the haze over it too, which a sprite's vertex finishes
    { binding: 23, visibility: VF, buffer: { type: 'read-only-storage' } }, // every light that moves (WGSL traffic_lamps): the falling snow's, the smoke's and the steam's vertices light by them too
    { binding: 24, visibility: VF, buffer: { type: 'read-only-storage' } }, // and the grid that says which reach where, with their shadows over the buildings; then the low lights' switches (lamp.h)
    { binding: 25, visibility: VF, texture: { viewDimension: '2d-array' } }, // the moon's light through the clouds (WGSL.CLOUD_SHADE, cloudShade), then its field in tiles (wasm/src/moon.c)
  ] });
  const tex = (binding, viewDimension = '2d', sampleType = 'float', visibility = FS) => ({ binding, visibility, texture: { viewDimension, sampleType } });
  const samp = binding => ({ binding, visibility: FS, sampler: {} });
  const photoLayout = b => [tex(b, '2d-array'), samp(b + 1), tex(b + 2, '2d', 'unfilterable-float'), tex(b + 3, '2d-array'), tex(b + 4, '2d-array'), tex(b + 5)]; // photoEntries
  groupLayouts.terrain = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: VF, buffer: {} }, tex(1, '2d-array', 'unfilterable-float', GPUShaderStage.VERTEX), ...photoLayout(2),
    tex(8, '2d', 'sint', GPUShaderStage.VERTEX), { binding: 9, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }] }); // the heights a layer a tile, and which layer each tile is (gridGroup); each chunk's level of detail
  groupLayouts.wall = device.createBindGroupLayout({ entries: [tex(0, '2d-array'), tex(1, '2d-array'), samp(2), tex(3, '2d-array'), tex(4, '2d-array')] });
  groupLayouts.road = device.createBindGroupLayout({ entries: [tex(0, '2d-array'), samp(1)] });
  // the roads, and what their wet tracks mirror: the lights, with the tiles WET_BIN laid them in, and the sky
  groupLayouts.wetRoad = device.createBindGroupLayout({ entries: [tex(0, '2d-array'), samp(1),
    { binding: 2, visibility: FS, buffer: { type: 'read-only-storage' } }, { binding: 3, visibility: FS, buffer: { type: 'read-only-storage' } }, tex(4), samp(5),
    { binding: 6, visibility: FS, buffer: { type: 'read-only-storage' } }, { binding: 7, visibility: FS, buffer: { type: 'read-only-storage' } }] });
  groupLayouts.single = device.createBindGroupLayout({ entries: [tex(0)] });
  // the clouds' noise, tiling every way; their puffs and their boxes (WGSL.CLOUD, WGSL.CLOUD_SHADE)
  const cloudEntries = [tex(0, '3d'), samp(1), { binding: 2, visibility: VF, buffer: { type: 'read-only-storage' } }, { binding: 3, visibility: VF, buffer: { type: 'read-only-storage' } }];
  groupLayouts.cloud = device.createBindGroupLayout({ entries: cloudEntries });
  groupLayouts.cloudDraw = device.createBindGroupLayout({ entries: [...cloudEntries, // and what each puff takes the same everywhere (WGSL.CLOUD_PREP)
    ...[4, 5, 6, 7].map(b => tex(b, '2d', 'unfilterable-float', GPUShaderStage.VERTEX))] });
  groupLayouts.cloudOver = device.createBindGroupLayout({ entries: [tex(0, '2d', 'float', VF), tex(1), samp(2), tex(3, '2d', 'float', GPUShaderStage.VERTEX)] }); // the clouds' own picture and how near they begin (WGSL.CLOUD_OVER)
  groupLayouts.live = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }] }); // the live mesh's pose: more slots than a uniform block holds
  const seaEntries = [{ binding: 0, visibility: VF, buffer: {} }, tex(1, '2d-array'), ...[2, 3, 4].map(b => tex(b)), tex(5, '2d-array'), tex(7), samp(8), samp(9), samp(10),
    tex(11, '2d-array', 'float', VF), tex(12, '2d', 'unfilterable-float'), tex(13, '2d', 'unfilterable-float'),
    { binding: 14, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }];
  // the sea, and the lights' streaks on it as the streak pass laid them (15); that pass itself has the sea without them
  // each of its pipelines' layout holds only what its shaders read (cutTo): the water's, what floats on it, the streaks'
  // (seaKept says which bindings, for their bind groups)
  for (const [name, code, entries] of [['sea', WGSL.WATER, [...seaEntries, tex(15)]], ['seaAfloat', WGSL.AFLOAT, seaEntries], ['seaGlow', WGSL.STREAK, seaEntries]]) {
    const kept = cutTo(entries, WGSL.PRELUDE + code, 1);
    seaKept[name] = kept.map(e => e.binding);
    groupLayouts[name] = device.createBindGroupLayout({ entries: kept });
  }
  // the field's slots and the blocks to draw (WGSL.PACK); the near terrain's heights, grid and photo: it lies on the terrain, and untouched drifts look as it does
  groupLayouts.snow = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: VF, buffer: {} },
    { binding: 1, visibility: VF, buffer: { type: 'read-only-storage' } }, { binding: 2, visibility: VF, buffer: { type: 'read-only-storage' } },
    tex(3, '2d', 'sint', GPUShaderStage.VERTEX), tex(4, '2d-array', 'unfilterable-float', GPUShaderStage.VERTEX),
    { binding: 5, visibility: VF, buffer: {} }, ...photoLayout(6)] });
  // the drifts beyond the field (WGSL.DUNE_SHADER): their head and nodes, the corners round the eye and the tile of their
  // shapes; the near terrain's heights, grid and photo, as the field's
  groupLayouts.dune = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: VF, buffer: { type: 'read-only-storage' } },
    tex(1, '2d', 'float', GPUShaderStage.VERTEX), tex(2, '2d', 'float', GPUShaderStage.VERTEX),
    tex(3, '2d', 'sint', GPUShaderStage.VERTEX), tex(4, '2d-array', 'unfilterable-float', GPUShaderStage.VERTEX),
    { binding: 5, visibility: VF, buffer: {} }, ...photoLayout(6)] });
  groupLayouts.forest = device.createBindGroupLayout({ entries: [tex(0, '2d', 'unfilterable-float', GPUShaderStage.VERTEX),
    { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }] });
  groupLayouts.impostor = device.createBindGroupLayout({ entries: [tex(0, '2d', 'unfilterable-float', VF), { binding: 1, visibility: VF, buffer: { type: 'read-only-storage' } },
    tex(2, '2d-array'), tex(3, '2d-array'), samp(4), tex(5, '2d', 'unfilterable-float', VF)] }); // 5: the views' frames (WGSL.impViews)
  groupLayouts.smoke = device.createBindGroupLayout({ entries: [0, 1, 2, 3].map(binding => ({ binding, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } })) });
  groupLayouts.car = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: VF, buffer: {} }, tex(1), tex(2, '2d-array'), tex(3), tex(4), { binding: 5, visibility: GPUShaderStage.FRAGMENT, sampler: {} }] });
  groupLayouts.jets = device.createBindGroupLayout({ entries: [tex(0, '2d-array')] }); // the flames marched small, a layer a jet
  groupLayouts.haze = device.createBindGroupLayout({ entries: [tex(0), { binding: 1, visibility: VF, buffer: {} }] }); // the frame round the jets, for their heat shimmer, and where

  const make = (code, { buffers: vb = [], group, blend, depthWrite = true, cull = 'back', bias = 0, slope = 0, topology = 'triangle-list', depthOnly = false, calm = false, far = false, pre = WGSL.PRELUDE }) => {
    const one = prelude => {
      const module = device.createShaderModule({ code: prelude + code });
      module.getCompilationInfo().then(info => { for (const m of info.messages) if (m.type === 'error') post({ error: `shader, line ${m.lineNum}: ${m.message}\n${(prelude + code).split('\n')[m.lineNum - 1]}` }); });
      return device.createRenderPipelineAsync({
        layout: device.createPipelineLayout({ bindGroupLayouts: [frameLayout, ...[].concat(group || []).map(g => groupLayouts[g])] }),
        vertex: { module, buffers: vb },
        fragment: { module, targets: [depthOnly ? { format, writeMask: 0 } : { format, blend }] }, // depthOnly: nothing of it reaches the picture
        primitive: { topology, cullMode: cull, frontFace: 'ccw' },
        // reversed depth: nearer is greater. Sky and sea level sit exactly on values the land may share.
        depthStencil: { format: 'depth32float', depthWriteEnabled: depthWrite, depthCompare: 'greater-equal', depthBias: bias, depthBiasSlopeScale: slope },
        multisample: { count: SAMPLES },
      });
    };
    // calm: a twin of it with none of the lights that come and go in it (WGSL.PRELUDE_CALM), for every draw of its that
    // none of them reaches (GFX_CMD_WASHED says which they do): calmOf has it, by the pipeline, once it is made. The
    // game does not wait for the twins: until one is there its pipeline draws everything, the same picture
    const full = one(pre);
    if (calm) Promise.all([full, one(WGSL.PRELUDE_CALM)]).then(([pipe, twin]) => calmOf.set(pipe, twin), e => post({ error: `shader twin: ${e.message}` }));
    // far: and a third, with nothing in it either of what only shows near by (WGSL.PRELUDE_FAR), for its draws marked GFX_CMD_FAR
    if (far) Promise.all([full, one(WGSL.PRELUDE_FAR)]).then(([pipe, twin]) => farOf.set(pipe, twin), e => post({ error: `shader twin: ${e.message}` }));
    return full;
  };
  const made = await Promise.all([
    make(WGSL.terrain(false, false), { group: 'terrain', calm: true, far: true }),
    make(WGSL.terrain(true, false), { group: 'terrain', calm: true, far: true }), // for the chunks a tunnel portal opens
    make(WGSL.terrain(false, false), { group: 'terrain', cull: 'none', calm: true, far: true }), // the backdrop is seen from below too, where it dips under the sea
    make(WGSL.terrain(false, true), { group: 'terrain', cull: 'none', calm: true, far: true }), // and where the town's edge crosses it, the town is cut out
    make(WGSL.PLAIN_SOLID, { buffers: LAYOUT.plain, calm: true, far: true }), // what one does not see into; the glass and the lenses are PIPE.GLAZED's
    make(WGSL.WALL, { buffers: LAYOUT.textured, group: 'wall', calm: true, far: true }),
    make(WGSL.TREE, { buffers: LAYOUT.instanced, cull: 'none', calm: true, far: true }),
    make(WGSL.LAMP, { buffers: LAYOUT.instanced }),
    // the ribbon lies 12 cm over the terrain; the bias keeps it on top where the ground bulges between its samples
    make(WGSL.ROAD, { buffers: LAYOUT.textured, group: 'wetRoad', blend: BLEND.over, depthWrite: false, cull: 'none', bias: 8, slope: 2, calm: true, far: true }),
    make(WGSL.ROAD_MASK, { buffers: LAYOUT.textured, group: 'wetRoad', cull: 'none', depthOnly: true }), // the ground under the roads' opaque middles, before the terrain
    make(WGSL.ROAD_MASK, { buffers: LAYOUT.textured, group: 'road', cull: 'none', depthOnly: true }), // and under the ski trails'
    make(WGSL.WATER, { group: 'sea', cull: 'none' }),
    make(WGSL.SKY, { group: 'single', depthWrite: false, cull: 'none' }),
    make(WGSL.GLOW, { buffers: LAYOUT.glow, blend: BLEND.lens, depthWrite: false, cull: 'none' }),
    make(WGSL.SNOW, { buffers: LAYOUT.snow, blend: BLEND.add, depthWrite: false, cull: 'none' }),
    make(WGSL.TUBE, { buffers: LAYOUT.textured, group: 'road' }),
    make(WGSL.TUBE, { buffers: LAYOUT.textured, group: 'road', bias: 16, slope: 4 }), // the end of the road outside laps under the mouth
    make(WGSL.CAR_BODY, { buffers: LAYOUT.car, group: 'car' }),
    make(WGSL.PAVING, { buffers: LAYOUT.textured, group: 'road', bias: 4, slope: 1, calm: true }),
    make(WGSL.ZEBRA, { buffers: LAYOUT.plain, blend: BLEND.over, depthWrite: false, cull: 'none', bias: 12, slope: 3, calm: true }),
    make(WGSL.SIGN, { buffers: LAYOUT.textured, group: 'road' }),
    make(WGSL.PACK, { group: 'snow', blend: BLEND.over, depthWrite: false, cull: 'none', bias: 12, slope: 3 }),
    make(WGSL.SPRAY, { group: 'snow', buffers: [{ arrayStride: 28, stepMode: 'instance', attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }, { shaderLocation: 1, offset: 12, format: 'float32x4' }] }],
      blend: BLEND.add, depthWrite: false, cull: 'none' }),
    make(WGSL.TRACKS, { group: 'snow', buffers: [{ arrayStride: 16, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }, { shaderLocation: 1, offset: 12, format: 'unorm8x4' }] }],
      blend: BLEND.over, depthWrite: false, cull: 'none', bias: 16, slope: 4 }), // paving laps over road ribbons and terrain it lies a few centimetres above
    // shadow, glass and glare: colour arrives already multiplied by opacity; the shadow needs lifting off the ground it lies on
    make(WGSL.CAR_BODY, { buffers: LAYOUT.car, group: 'car', blend: BLEND.laid, depthWrite: false, bias: 12, slope: 3 }),
    make(WGSL.SHOWER, { blend: BLEND.over, depthWrite: false, cull: 'none' }), // the snow showers over the fjord: no buffers, the frame's uniforms
    make(WGSL.wire(SAMPLES), { buffers: LAYOUT.plain, topology: 'line-list', cull: 'none' }), // wires
    ...['solid', 'fade', 'twig'].map(kind => make(WGSL.forestShader(kind, SAMPLES), { buffers: LAYOUT.forest, group: 'forest', cull: 'none', calm: true, far: true })),
    make(WGSL.FOREST_HAZE, { buffers: LAYOUT.forest, group: 'forest', blend: BLEND.over, depthWrite: false, cull: 'none', calm: true }),
    make(WGSL.LABEL, { buffers: LAYOUT.textured, group: 'road', blend: BLEND.over }),
    make(WGSL.AFLOAT, { buffers: [LAYOUT.plain[0], { arrayStride: 32, attributes: [{ shaderLocation: 4, offset: 0, format: 'float32x4' }, { shaderLocation: 5, offset: 16, format: 'float32x4' }] }], group: 'seaAfloat' }),
    // lights on the water: laid on the sea's own plane, so lifted off it
    make(WGSL.STREAK, { buffers: [{ arrayStride: 36, stepMode: 'instance', attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x4' }, { shaderLocation: 1, offset: 16, format: 'float32x4' },
      { shaderLocation: 2, offset: 32, format: 'float32' }] }], group: 'seaGlow', blend: BLEND.add, depthWrite: false, cull: 'none', bias: 8, slope: 2 }),
    make(WGSL.SMOKE_PUFFS, { group: 'smoke', blend: BLEND.over, depthWrite: false, cull: 'none' }), // no vertex buffer: the stoves, in the order the game sends
    make(WGSL.SMOKE_STEAM, { group: 'smoke', blend: BLEND.over, depthWrite: false, cull: 'none' }), // the exhaust steam the game moves
    // what the car stirs up on the water: the wake's ribbon and the rings lie on the sea, the spray flies over it
    make(WGSL.WAKE, { buffers: [{ arrayStride: 28, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }, { shaderLocation: 1, offset: 12, format: 'float32x4' }] }],
      blend: BLEND.over, depthWrite: false, cull: 'none', bias: 8, slope: 2 }),
    make(WGSL.RING, { buffers: [{ arrayStride: 16, stepMode: 'instance', attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x4' }] }], blend: BLEND.over, depthWrite: false, cull: 'none', bias: 8, slope: 2 }),
    make(WGSL.DROPS, { buffers: [{ arrayStride: 28, stepMode: 'instance', attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }, { shaderLocation: 1, offset: 12, format: 'float32x4' }] }],
      blend: BLEND.over, depthWrite: false, cull: 'none' }),
    make(WGSL.LIVE, { buffers: LAYOUT.plain, group: 'live', cull: 'none' }), // what moves: in the sky, on the roads, on the fjord
    // the engines' flames, each marched through inside a box whose near faces (or far ones, from inside it) are drawn;
    // what they give off is added, what soot they carry dims what is behind; and the snow they blow about
    make(WGSL.JETS, { buffers: ROCKET.LAYOUT, group: 'jets', blend: BLEND.jets, depthWrite: false, cull: 'none' }),
    make(WGSL.BLOWN, { buffers: [{ arrayStride: ROCKET.PUFF, stepMode: 'instance', attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x4' }, { shaderLocation: 1, offset: 16, format: 'float32x4' }] }],
      blend: BLEND.laid, depthWrite: false, cull: 'none' }),
    make(WGSL.forestImpostor(SAMPLES), { buffers: LAYOUT.forest, group: 'impostor', cull: 'none', calm: true, far: true }), // cut out by its cover, sample by sample
    make(WGSL.fine(SAMPLES), { buffers: LAYOUT.plain, calm: true }), // fine trim, thinned away as it shrinks under a pixel
    make(WGSL.PANE, { buffers: LAYOUT.textured, group: 'road' }), // the centre's shop windows: the shop behind the glass, from the window atlas
    make(WGSL.LIVE_BLUR, { buffers: LAYOUT.plain, group: 'live', blend: BLEND.over, depthWrite: false, cull: 'none' }), // rotors and propellers, smeared by the frame's exposure
    // the gritter's sand on the road, as the car's tracks are laid
    make(WGSL.GRIT, { group: 'snow', buffers: [{ arrayStride: 16, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }, { shaderLocation: 1, offset: 12, format: 'unorm8x4' }] }],
      blend: BLEND.over, depthWrite: false, cull: 'none', bias: 16, slope: 4 }),
    make(WGSL.TRAIL, { buffers: LAYOUT.textured, group: 'road', blend: BLEND.over, depthWrite: false, cull: 'none', bias: 8, slope: 2, calm: true }), // ski trails, as the roads are drawn, with a texture of their own
    make(WGSL.ROAD_DRY, { buffers: LAYOUT.textured, group: 'wetRoad', blend: BLEND.over, depthWrite: false, cull: 'none', bias: 8, slope: 2, calm: true }), // and the roads with nothing wet (?skip=wet)
    make(WGSL.CAR, { buffers: LAYOUT.car, group: 'car' }), // the car's lenses, each with its chamber behind it (car.runs)
    make(WGSL.PLAIN, { buffers: LAYOUT.plain, calm: true, far: true }), // the glass and the lenses among the plain faces (and a third for their draws far off: a car's windows a kilometre away were lit with the car's own beams and the near shadow map, on every sample 4x MSAA gave them)
    make(WGSL.CAR_FLIGHT, { buffers: LAYOUT.car, group: 'car' }), // and the flight instruments, on the cluster board's back
    make(WGSL.CAR_CABIN, { buffers: LAYOUT.car, group: 'car' }), // and the cabin's finishes: leather, vinyl, gloss and plastic
    // and the car's body and its blended faces where a light that comes and goes reaches it (FRAME.CAR_WASHED): at all other times
    // the car's shaders look for none of them
    make(WGSL.CAR_WASHED, { buffers: LAYOUT.car, group: 'car' }),
    make(WGSL.CAR_WASHED, { buffers: LAYOUT.car, group: 'car', blend: BLEND.laid, depthWrite: false, bias: 12, slope: 3 }),
    make(WGSL.CLOUD_OVER, { group: 'cloudOver', blend: BLEND.laid, depthWrite: false, cull: 'none' }), // the clouds' picture laid over the frame
    ...cloudPipes(),
    make(WGSL.DUNE_SHADER, { group: 'dune', cull: 'none', bias: 12, slope: 3 }), // the drifts beyond the field, as it lies: solid, drawn before the ground they hide
    make(WGSL.LIVE_HERD, { buffers: [...LAYOUT.plain, { arrayStride: 4, stepMode: 'instance', attributes: [{ shaderLocation: 3, offset: 0, format: 'uint32' }] }], group: 'live', cull: 'none' }),
    make(WGSL.LIVE_HERD, { buffers: [...LAYOUT.plain, { arrayStride: 4, stepMode: 'instance', attributes: [{ shaderLocation: 3, offset: 0, format: 'uint32' }] }], group: 'live', cull: 'none', pre: WGSL.PRELUDE_HERD_FAR() }), // and the cars a few pixels long (struct traffic_seen, near) // the traffic's bodies, a car each
  ]);
  pipes.herdFar = made.pop(); pipes.herd = made.pop(); pipes.dune = made.pop();
  cloudLow.prepPipe = made.pop(); cloudLow.pipe = made.pop(); pipes.cloudOver = made.pop();
  pipes.carBlendWashed = made.pop(); pipes.carWashed = made.pop();
  { // the far ground lit once a pixel (WGSL.terrainDeferred): its buffer's pipelines, the lighting's, and the frame's that read it
    const tex = (binding, sampleType) => ({ binding, visibility: FS, texture: { sampleType } });
    groupLayouts.gbuf = device.createBindGroupLayout({ entries: [tex(0, 'uint'), tex(1, 'unfilterable-float')] });
    groupLayouts.gbuf3 = device.createBindGroupLayout({ entries: [tex(0, 'uint'), tex(1, 'unfilterable-float'), tex(2, 'unfilterable-float')] }); // with the plain faces' colour (a layout of its own: the ground's pipelines are at the sixteen textures a stage)
    groupLayouts.plainRead = device.createBindGroupLayout({ entries: [tex(0, 'unfilterable-float'), tex(1, 'uint')] });
    groupLayouts.shaded = device.createBindGroupLayout({ entries: [tex(0, 'unfilterable-float')] });
    const kinds = [[false, false, 'back'], [true, false, 'back'], [false, false, 'none'], [false, true, 'none']]; // as ground() draws them: the town's, cut; the backdrop's, clipped
    const layout = groups => device.createPipelineLayout({ bindGroupLayouts: [frameLayout, ...groups.map(g => groupLayouts[g])] });
    Promise.all([
      ...kinds.map(([cut, clip, cull], i) => { const module = device.createShaderModule({ code: WGSL.PRELUDE_FAR + WGSL.terrainGbuf(cut, clip, i >> 1) });
        return device.createRenderPipelineAsync({ layout: layout(['terrain']), vertex: { module }, fragment: { module, targets: [{ format: 'rgba32uint' }, { format: 'rgba16float' }, { format: 'rgba16float', writeMask: 0 }] },
          primitive: { topology: 'triangle-list', cullMode: cull, frontFace: 'ccw' }, depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'greater-equal' } }); }),
      ...[0, 1, 2].map(g => { const module = device.createShaderModule({ code: (g < 2 ? WGSL.PRELUDE_FAR : WGSL.PRELUDE) + WGSL.terrainDeferred(g & 1, g >> 1) }); // 2: the drifts', with every light
        return device.createRenderPipelineAsync({ layout: layout(['terrain', 'gbuf']), vertex: { module }, fragment: { module, targets: [{ format: 'rgba16float' }] },
          depthStencil: { format: 'depth32float', depthWriteEnabled: false, depthCompare: 'less' } }); }), // only where the far ground is (its depth over the clear 0)
      ...kinds.map(([cut, clip, cull]) => make(WGSL.terrainRead(cut, clip), { group: ['terrain', 'shaded'], cull, pre: WGSL.PRELUDE_FAR })),
      (() => { const module = device.createShaderModule({ code: WGSL.PRELUDE_FAR + WGSL.DUNE_GBUF }); // the drifts into the buffer
        return device.createRenderPipelineAsync({ layout: layout(['dune']), vertex: { module }, fragment: { module, targets: [{ format: 'rgba32uint' }, { format: 'rgba16float' }, { format: 'rgba16float', writeMask: 0 }] },
          primitive: { topology: 'triangle-list', cullMode: 'none', frontFace: 'ccw' }, depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'greater-equal', depthBias: 12, depthBiasSlopeScale: 3 } }); })(),
    ]).then(p => { defer.prep = p.slice(0, 4); defer.shade = p.slice(4, 7); defer.read = p.slice(7, 11); defer.dunePrep = p[11]; }, e => post({ error: `defer: ${e.message}` }));
    // and the far plain faces the same way (WGSL.PLAIN_GBUF, PLAIN_SHADE, PLAIN_READ)
    { const gm = device.createShaderModule({ code: WGSL.PRELUDE_FAR + WGSL.PLAIN_GBUF }), sm = device.createShaderModule({ code: WGSL.PRELUDE_FAR + WGSL.PLAIN_SHADE });
      for (const m of [gm, sm]) m.getCompilationInfo().then(info => { for (const x of info.messages) if (x.type === 'error') post({ error: `plain defer shader, line ${x.lineNum}: ${x.message}` }); });
      Promise.all([
        device.createRenderPipelineAsync({ layout: layout([]), vertex: { module: gm, buffers: LAYOUT.plain }, fragment: { module: gm, targets: [{ format: 'rgba32uint' }, { format: 'rgba16float' }, { format: 'rgba16float' }] },
          primitive: { topology: 'triangle-list', cullMode: 'back', frontFace: 'ccw' }, depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'greater-equal' } }),
        device.createRenderPipelineAsync({ layout: layout(['gbuf3']), vertex: { module: sm }, fragment: { module: sm, targets: [{ format: 'rgba16float' }] },
          depthStencil: { format: 'depth32float', depthWriteEnabled: false, depthCompare: 'less' } }),
        make(WGSL.PLAIN_READ, { buffers: LAYOUT.plain, group: 'plainRead', pre: WGSL.PRELUDE_FAR }),
      ]).then(([a, b, c]) => { defer.plainPrep = a; defer.plainShade = b; defer.plainRead = c; }, e => post({ error: `plain defer: ${e.message}` })); }
    // the drifts where it is: they take its light (WGSL.DUNE_READ)
    make(WGSL.DUNE_READ, { group: ['dune', 'shaded'], cull: 'none', bias: 12, slope: 3 }).then(p => { pipes.duneRead = p; }, e => post({ error: `dune read: ${e.message}` }));
    // and the far road lit once a pixel (WGSL.roadDeferred): its list's pipeline, the lighting's, and the frame's that reads it
    const store = (binding, type) => ({ binding, visibility: FS, buffer: { type } });
    groupLayouts.roadPrep = device.createBindGroupLayout({ entries: [store(0, 'storage')] });
    groupLayouts.roadShade = device.createBindGroupLayout({ entries: [store(0, 'storage'), tex(1, 'uint')] });
    groupLayouts.roadRead = device.createBindGroupLayout({ entries: [store(0, 'read-only-storage'), tex(1, 'uint')] });
    const prep = device.createShaderModule({ code: WGSL.PRELUDE_FAR + WGSL.roadPrep }), shade = device.createShaderModule({ code: WGSL.PRELUDE_FAR + WGSL.roadDeferred });
    for (const m of [prep, shade]) m.getCompilationInfo().then(info => { for (const x of info.messages) if (x.type === 'error') post({ error: `droad shader, line ${x.lineNum}: ${x.message}` }); });
    Promise.all([
      device.createRenderPipelineAsync({ layout: layout(['wetRoad', 'roadPrep']), vertex: { module: prep, entryPoint: 'vs', buffers: LAYOUT.textured }, fragment: { module: prep, entryPoint: 'fs', targets: [{ format: 'r32uint' }] },
        primitive: { topology: 'triangle-list', cullMode: 'none', frontFace: 'ccw' }, depthStencil: { format: 'depth32float', depthWriteEnabled: false, depthCompare: 'greater-equal', depthBias: 8, depthBiasSlopeScale: 2 } }),
      device.createRenderPipelineAsync({ layout: layout(['wetRoad', 'roadShade']), vertex: { module: shade, entryPoint: 'vs_full' }, fragment: { module: shade, entryPoint: 'fs', targets: [] },
        depthStencil: { format: 'depth32float', depthWriteEnabled: false, depthCompare: 'always' } }),
      make(WGSL.roadRead, { buffers: LAYOUT.textured, group: ['wetRoad', 'roadRead'], blend: BLEND.over, depthWrite: false, cull: 'none', bias: 8, slope: 2, pre: WGSL.PRELUDE_FAR }),
    ]).then(([a, b, c]) => { droad.prep = a; droad.shade = b; droad.read = c; }, e => post({ error: `droad: ${e.message}` }));
  }
  pipes.carCabin = made.pop(); pipes.carFlight = made.pop(); pipes[PIPE.GLAZED] = made.pop(); pipes.carLens = made.pop(); pipes.roadDry = made.pop(); pipes[PIPE.TRAIL] = made.pop();
  { // the moon's shadow map: depth only, three vertex layouts
    const module = device.createShaderModule({ code: WGSL.SHADOW });
    shadow.layout = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: {} }] });
    const pos = stride => ({ arrayStride: stride, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] });
    const inst = { arrayStride: 32, stepMode: 'instance', attributes: [{ shaderLocation: 4, offset: 0, format: 'float32x4' }, { shaderLocation: 5, offset: 16, format: 'float32x3' }] };
    // the forest casts from its model texture, pinned to the ground map: a group of its own beside the map's uniforms
    const forest = device.createShaderModule({ code: WGSL.FOREST_SHADOW }), V = GPUShaderStage.VERTEX;
    shadow.forestLayout = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: V, texture: { sampleType: 'unfilterable-float' } },
      { binding: 1, visibility: V, texture: { viewDimension: '2d-array' } }, { binding: 2, visibility: V, sampler: {} }, { binding: 3, visibility: V, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: V, texture: { sampleType: 'sint' } }, { binding: 5, visibility: V, texture: { viewDimension: '2d-array', sampleType: 'unfilterable-float' } }] });
    const layouts = { [true]: device.createPipelineLayout({ bindGroupLayouts: [shadow.layout, shadow.forestLayout] }), [false]: device.createPipelineLayout({ bindGroupLayouts: [shadow.layout] }) };
    const depth = (entryPoint, buffers, format = 'depth16unorm', bias = {}) => device.createRenderPipeline({
      layout: layouts[buffers === LAYOUT.forest], vertex: { module: buffers === LAYOUT.forest ? forest : module, entryPoint, buffers },
      primitive: { cullMode: 'none' }, depthStencil: { format, depthWriteEnabled: true, depthCompare: 'less', ...bias } });
    const casters = (...more) => ({ [PIPE.PLAIN]: depth('fixed', [pos(20)], ...more), [PIPE.GLAZED]: depth('fixed', [pos(20)], ...more), [PIPE.FINE]: depth('fixed', [pos(20)], ...more), [PIPE.AFLOAT]: depth('fixed', [pos(20)], ...more), [PIPE.WALL]: depth('fixed', [pos(28)], ...more),
      [PIPE.TREE]: depth('placed', [pos(20), inst], ...more), [PIPE.LAMP]: depth('placed', [pos(20), inst], ...more),
      [PIPE.FOREST]: depth('thrown', LAYOUT.forest, ...more) });
    shadow.pipes = casters();
    // the headlamps' shadow map, one for both lamps (game.c), redrawn every frame they move
    shadow.beamPipes = casters('depth32float', { depthBias: 2, depthBiasSlopeScale: 2 });
    shadow.beams = device.createTexture({ size: [1024, 1024, 1], format: 'depth32float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    shadow.beam = [0].map(layer => {
      const uniform = device.createBuffer({ size: SHADOW_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const view = shadow.beams.createView({ dimension: '2d', baseArrayLayer: layer, arrayLayerCount: 1 });
      return { uniform, name: 'beam' + layer, pass: { colorAttachments: [], depthStencilAttachment: { view, depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' } },
        group: device.createBindGroup({ layout: shadow.layout, entries: [{ binding: 0, resource: { buffer: uniform } }] }) };
    });
    shadow.map = device.createTexture({ size: [4096, 4096], format: 'depth16unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    shadow.uniform = device.createBuffer({ size: SHADOW_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    shadow.group = device.createBindGroup({ layout: shadow.layout, entries: [{ binding: 0, resource: { buffer: shadow.uniform } }] });
    shadow.view = shadow.map.createView();
    shadow.pass = { colorAttachments: [], depthStencilAttachment: { view: shadow.view, depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' } };
  }
  [pipes.terrain, pipes.terrainCut, pipes.outer, pipes.outerCut, pipes[PIPE.PLAIN], pipes[PIPE.WALL], pipes[PIPE.TREE], pipes[PIPE.LAMP], pipes[PIPE.ROAD], pipes.roadMask, pipes.trailMask, pipes.water, pipes.sky, pipes[PIPE.GLOW], pipes.snow, pipes[PIPE.TUBE], pipes[PIPE.TUBE_FLOOR], pipes.car, pipes[PIPE.PAVING], pipes[PIPE.ZEBRA], pipes[PIPE.SIGN], pipes.pack, pipes.spray, pipes.tracks, pipes.carBlend, pipes.shower, pipes[PIPE.LINE],
    pipes[PIPE.FOREST], pipes[PIPE.FOREST_FADE], pipes[PIPE.FOREST_TWIG], pipes[PIPE.HAZE], pipes[PIPE.LABEL], pipes[PIPE.AFLOAT], pipes[PIPE.STREAK], pipes[PIPE.SMOKE], pipes.steam, pipes.wake, pipes.ring, pipes.drops, pipes.live, pipes.jets, pipes.blown, pipes[PIPE.IMPOSTOR], pipes[PIPE.FINE], pipes[PIPE.PANE], pipes.liveBlur, pipes.grit] = made;
  if (SCALE < 1) {
    const module = device.createShaderModule({ code: WGSL.FSR });
    const fsr = entryPoint => device.createRenderPipeline({ layout: 'auto', vertex: { module }, fragment: { module, entryPoint, targets: [{ format }] } });
    up.easu = fsr('easu'); if (SHARP) up.rcas = fsr('rcas');
    up.uniform = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }
  { // the flames big on the screen, marched at a quarter of the frame's size each way, each jet into a layer of its own (see frame())
    const module = device.createShaderModule({ code: WGSL.PRELUDE + WGSL.JETS_LOW });
    module.getCompilationInfo().then(info => { for (const m of info.messages) if (m.type === 'error') post({ error: `shader, line ${m.lineNum}: ${m.message}` }); });
    pipes.jetsLow = await device.createRenderPipelineAsync({
      layout: device.createPipelineLayout({ bindGroupLayouts: [frameLayout] }),
      vertex: { module, buffers: ROCKET.LAYOUT },
      fragment: { module, targets: Array.from({ length: ROCKET.LOW }, () => ({ format: 'rgba16float', blend: BLEND.laid })) },
      primitive: { topology: 'triangle-list', cullMode: 'none' } });
  }
  { // the jets' heat shimmer: their boxes drawn over the finished frame, reading a copy of it (see frame())
    const module = device.createShaderModule({ code: WGSL.PRELUDE + WGSL.JETS_HAZE });
    module.getCompilationInfo().then(info => { for (const m of info.messages) if (m.type === 'error') post({ error: `shader, line ${m.lineNum}: ${m.message}` }); });
    pipes.jetsHaze = await device.createRenderPipelineAsync({
      layout: device.createPipelineLayout({ bindGroupLayouts: [frameLayout, groupLayouts.haze] }),
      vertex: { module, buffers: ROCKET.LAYOUT }, fragment: { module, targets: [{ format }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' } });
  }
  { // the lights' streaks, laid at half size into a texture of their own that the water adds where it shows (see frame())
    const module = device.createShaderModule({ code: WGSL.PRELUDE + WGSL.STREAK });
    pipes.streakGlow = await device.createRenderPipelineAsync({
      layout: device.createPipelineLayout({ bindGroupLayouts: [frameLayout, groupLayouts.seaGlow] }),
      vertex: { module, buffers: [{ arrayStride: 36, stepMode: 'instance', attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x4' }, { shaderLocation: 1, offset: 16, format: 'float32x4' },
        { shaderLocation: 2, offset: 32, format: 'float32' }] }] },
      fragment: { module, targets: [{ format: 'rgba16float', blend: BLEND.add }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' } });
  }
}

/* ---- terrain: a strip of cells, drawn once per row of the height grid ---- */

// at level k (wasm/include/harstad/terrain.h), every 2^k-th post: the cells of level k, each over 2^k of the strip's
// columns, its vertices still numbered by the fine column (the shader reads the column from the index)
function rowStrip(n, k = 0) {
  const m = (n - 1) >> k, idx = new Uint16Array(m * 6);
  for (let i = 0, q = 0; i < m; i++) {
    const a = (i << k) * 2, b = ((i + 1) << k) * 2, c = a + 1, d = b + 1; // (i, row), (i + 1, row), (i, row + 1), (i + 1, row + 1)
    idx[q++] = a; idx[q++] = c; idx[q++] = b; idx[q++] = b; idx[q++] = c; idx[q++] = d;
  }
  return gpuBuffer(idx, GPUBufferUsage.INDEX);
}
const LEVELS = 4, LEVEL_SHIFT = 28; // TERRAIN_LEVELS, TERRAIN_LEVEL_SHIFT

// struct Grid (src/shaders.js, GRID); `photo` is the photo's pages (photoPaging), or none for one photo over the extent.
// `heights` holds a layer a tile, `tiles` which layer each tile of `tile` (its corner's grid point, a tile's points a
// side) is. Returns a function that makes the bind group from the photo's pools as they are now.
function gridGroup({ origin, step, clip, size, extent, heights, tiles, tile, photo, sampler, cuts, keep, held, chunks, lod }) {
  const data = new ArrayBuffer(224), f = new Float32Array(data), u = new Uint32Array(data);
  new Int32Array(data, 192, 3).set(tile);
  u.set(chunks, 52); // cells a chunk's side, chunks across and down (terrain_lod)
  f.set([origin[0], origin[1], step, clip]);
  u.set(size, 4); f.set(extent, 6);
  if (photo.meta) { const m = photo.meta; u.set([m.across, m.down], 8); f.set([...m.origin, m.page, m.size, ...m.px], 10); }
  if (held) f.set(held, 10); // the backdrop: the tiles' grid, whose held tiles photo.pages holds the rectangles of
  if (cuts) for (let i = 0; i < 4; i++) { f.set(cuts.subarray(i * 8, i * 8 + 4), 16 + i * 4); f.set(cuts.subarray(i * 8 + 4, i * 8 + 8), 32 + i * 4); }
  const buffer = gpuBuffer(data, GPUBufferUsage.UNIFORM);
  if (keep) keep.grid = buffer; // (the snow field reads the near one too)
  return () => device.createBindGroup({ layout: groupLayouts.terrain, entries: [
    { binding: 0, resource: { buffer } }, { binding: 1, resource: arrayView(heights) }, ...photoEntries(photo, sampler, 2),
    { binding: 8, resource: tiles.createView() }, { binding: 9, resource: { buffer: lod } }] });
}
// The photo's bindings from `b`: the pools (or the backdrop's one photo, in all three), the sampler, the page table, the coarse photo
function photoEntries(photo, sampler, b) {
  const pools = photo.pools ? photo.pools.map(p => arrayView(p.texture)) : [0, 1, 2].map(() => arrayView(photo.texture));
  return [{ binding: b, resource: pools[0] }, { binding: b + 1, resource: sampler }, { binding: b + 2, resource: photo.pages.createView() },
    { binding: b + 3, resource: pools[1] }, { binding: b + 4, resource: pools[2] },
    { binding: b + 5, resource: (photo.coarse || photo.texture).createView({ dimension: '2d', arrayLayerCount: 1 }) }];
}

/* ---- loading ---- */

// How far the load has come, for the page to show (src/main.js, the outline filling in): the bytes come of the data and
// the module, of the terrain's tiles, of the photo tiles, and the steps of the bake done (the build worker's stages, enum build_stage, the photos
// decoded, the world read in, the upload), in fixed shares. Each share counts only what has happened: a file is reckoned by its
// Content-Length once its headers are in, or, sent compressed, only once it is all here.
const BUILD_STAGES = 11, BAKE_STEPS = BUILD_STAGES + 3;
const loading = { data: { files: 4, heard: 0, got: 0, total: 0 }, terrain: { files: 0, heard: 0, got: 0, total: 0 }, photos: { files: 0, heard: 0, got: 0, total: 0 }, baked: 0, shown: 0 };
const share = g => g.files ? g.heard / g.files * (g.total ? Math.min(1, g.got / g.total) : 0) : 0;
function progress(step = 0) {
  loading.baked += step;
  const p = 0.25 * share(loading.data) + 0.15 * share(loading.terrain) + 0.2 * share(loading.photos) + 0.4 * Math.min(1, loading.baked / BAKE_STEPS);
  if (p >= loading.shown + 0.002 || (p > loading.shown && p >= 1)) post({ progress: loading.shown = p });
}
// fetch(url), its body counted into `group` as it streams in
async function counted(url, group) {
  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`could not load ${url}: ${res.status}`);
  const whole = res.headers.get('content-encoding') ? 0 : +res.headers.get('content-length') || 0;
  group.heard++;
  group.total += whole;
  let got = 0;
  const body = res.body.pipeThrough(new TransformStream({
    transform(chunk, out) { got += chunk.byteLength; if (whole) { group.got += chunk.byteLength; progress(); } out.enqueue(chunk); },
    flush() { if (!whole) { group.got += got; group.total += got; progress(); } } }));
  return new Response(body, { headers: res.headers });
}

// The ground's photo in pages, a quarter of a tile each (tools/imagery.py, pages): every page the same size, its
// neighbours' pixels round its edge; and the coarse photo of them all. Every page's file is fetched, nearest `start`
// first, and kept as it came (some 26 MB of JPEG): only those the eye can see finely are decoded onto the GPU, and
// those again whenever it comes near (photoPaging).
async function loadPhotos([x, z]) {
  const get = async url => (await counted(url, loading.photos)).blob();
  const meta = await (await fetch('../assets/ground/pages.json')).json();
  loading.photos.files = meta.pages.length + 2;
  const far = ([p, q]) => Math.hypot(meta.origin[0] + (p + 0.5) * meta.page - x, meta.origin[1] + (q + 0.5) * meta.page - z);
  const order = meta.pages.map((_, k) => k).sort((a, b) => far(meta.pages[a]) - far(meta.pages[b]));
  const pages = [];
  // a page's file is named by where it lies from the tiles' first corner (tools/geo.py, TILE_X0), which never moves as the
  // map grows; the grid of pages.json begins at the westmost and northmost tile the map holds, `first` pages from it
  const [p0, q0] = meta.first || [0, 0];
  for (const k of order) pages[k] = get(`../assets/ground/${meta.pages[k][0] + p0}_${meta.pages[k][1] + q0}.jpg`);
  const bitmap = async url => createImageBitmap(await get(url), { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
  return { meta, pages, coarse: bitmap('../assets/ground/coarse.jpg'), outer: bitmap('../assets/outer.jpg') };
}

// The photo paged (wasm/include/harstad/photo.h): pool L, an array texture, holds levels L to 2 of the pages the C
// says, a layer each; the coarse photo, the map at the pages' level 3, is always held. The page table says, for each
// page, its layer, where its corner falls in the layer, and its pool, or 3 for none (src/shaders.js, ground_photo).
// After every frame the C says what to load, move and let go (ph.step). A page is loaded by decoding its file into a
// texture of its own size and three levels, whose mips are made as makeMips made them for the whole array, and by
// putting the levels its pool holds into its layer: compressed to BC7 where the GPU reads it (WGSL.BC7: a quarter of
// the bytes, each texel within a step or two of 255 of what it was, as near as the photo's JPEG is to the mosaic), as
// they are where it does not.
const PHOTO = { POOLS: 3, LOAD: 0, MOVE: 1, DROP: 2, FLIGHT: 6 }; // enum photo_op (and GROW, SHRINK); loads under way at most
let photoPool = null;
let builders = []; // the build workers, while the load runs (init); a load that fails stops them
async function photoPaging(asked, binds) {
  const { meta, pages, coarse, outer } = await asked, S = meta.size, bc7 = device.features.has('texture-compression-bc');
  const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC;
  const pool = (L, cap) => device.createTexture({ label: 'photo', size: cap ? [S >> L, S >> L, cap] : [4, 4, 1], format: bc7 ? 'bc7-rgba-unorm-srgb' : 'rgba8unorm-srgb',
    mipLevelCount: cap ? PHOTO.POOLS - L : 1, usage });
  const staging = device.createTexture({ label: 'photo', size: [S, S], format: 'rgba8unorm-srgb', mipLevelCount: PHOTO.POOLS,
    usage: usage | GPUTextureUsage.RENDER_ATTACHMENT, viewFormats: ['rgba8unorm'] });
  // the compressing: a pipeline, and for each level a buffer of its blocks (a row of them a multiple of 256 bytes) and its bind group
  const squeeze = bc7 && (() => {
    const module = device.createShaderModule({ code: WGSL.BC7 }), pipeline = device.createComputePipeline({ layout: 'auto', compute: { module } });
    return Array.from({ length: PHOTO.POOLS }, (_, m) => {
      const n = (S >> m) / 4, row = Math.ceil(n * 16 / 256) * 256;
      const buffer = device.createBuffer({ size: row * n, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: staging.createView({ format: 'rgba8unorm', baseMipLevel: m, mipLevelCount: 1 }) }, { binding: 1, resource: { buffer } },
        { binding: 2, resource: { buffer: gpuBuffer(new Uint32Array([n, n, row / 16, 0]), GPUBufferUsage.UNIFORM) } }] });
      return { n, row, buffer, run(enc) { const pass = enc.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(Math.ceil(n / 8), Math.ceil(n / 8)); pass.end(); } };
    });
  })();
  const table = new Float32Array(meta.across * meta.down * 4);
  for (let k = 0; k < table.length; k += 4) table[k + 3] = PHOTO.POOLS;
  const one = await coarse, outerOne = await outer;
  const ph = { meta, pools: [0, 1, 2].map(L => ({ texture: pool(L, 0), cap: 0 })), table, dirty: true, flight: 0, lates: 0, late: 0, used: [], caps: [],
    pages: dataTexture('rgba32float', meta.across, meta.down, table, meta.across * 16),
    coarse: arrayTexture([one], 'rgba8unorm-srgb', 'photo'),
    outer: { texture: arrayTexture([outerOne], 'rgba8unorm-srgb'), pages: null } }; // its pages: the tiles it is cut away under
  one.close(); outerOne.close();
  const entry = k => (meta.pages[k][1] * meta.across + meta.pages[k][0]) * 4;
  const place = (k, L, slot) => { const [, , u, v] = meta.pages[k]; table.set([slot, u, v, L], entry(k)); ph.dirty = true; };
  // pool L made again `cap` layers deep, the layers both have copied over
  const remake = (L, cap) => {
    const was = ph.pools[L], texture = pool(L, cap), keep = Math.min(was.cap, cap);
    if (keep) {
      const enc = device.createCommandEncoder();
      for (let m = 0; m < PHOTO.POOLS - L; m++) enc.copyTextureToTexture({ texture: was.texture, mipLevel: m }, { texture, mipLevel: m }, [S >> (L + m), S >> (L + m), keep]);
      device.queue.submit([enc.finish()]);
    }
    was.texture.destroy();
    ph.pools[L] = { texture, cap };
    binds();
  };
  const load = (k, L, slot) => {
    ph.flight++;
    pages[k].then(blob => createImageBitmap(blob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' })).then(b => {
      device.queue.copyExternalImageToTexture({ source: b }, { texture: staging }, [S, S]);
      b.close();
      makeMips(staging);
      const enc = device.createCommandEncoder(), to = ph.pools[L].texture;
      for (let m = L; m < PHOTO.POOLS; m++) {
        const into = { texture: to, mipLevel: m - L, origin: [0, 0, slot] };
        if (!squeeze) { enc.copyTextureToTexture({ texture: staging, mipLevel: m }, into, [S >> m, S >> m, 1]); continue; }
        const q = squeeze[m];
        q.run(enc);
        enc.copyBufferToTexture({ buffer: q.buffer, bytesPerRow: q.row, rowsPerImage: q.n }, into, [S >> m, S >> m, 1]);
      }
      device.queue.submit([enc.finish()]);
      place(k, L, slot);
      hs.hs_photo_loaded(k);
      ph.flight--;
      if (ph.dirty) ph.send();
    }).catch(e => post({ error: String(e.stack || e) }));
  };
  ph.send = () => { device.queue.writeTexture({ texture: ph.pages }, table, { bytesPerRow: meta.across * 16 }, [meta.across, meta.down]); ph.dirty = false; };
  // what the C asked, at `out` (what hs_photo_step returns): done in order, the loads begun
  ph.run = out => {
    const head = new Uint32Array(hs.memory.buffer, out, 10), [nask, late, lates, at] = head;
    ph.late = late; ph.lates = lates; ph.used = Array.from(head.subarray(4, 7)); ph.caps = Array.from(head.subarray(7, 10));
    const asks = new Uint32Array(hs.memory.buffer, at, nask * 6).slice();
    for (let i = 0; i < nask; i++) {
      const [op, k, L, slot, from, fromSlot] = asks.subarray(i * 6, i * 6 + 6);
      if (op === PHOTO.LOAD) load(k, L, slot);
      else if (op === PHOTO.MOVE) {
        const enc = device.createCommandEncoder();
        for (let m = L; m < PHOTO.POOLS; m++) enc.copyTextureToTexture({ texture: ph.pools[from].texture, mipLevel: m - from, origin: [0, 0, fromSlot] },
          { texture: ph.pools[L].texture, mipLevel: m - L, origin: [0, 0, slot] }, [S >> m, S >> m, 1]);
        device.queue.submit([enc.finish()]);
        place(k, L, slot);
      }
      else if (op === PHOTO.DROP) { table[entry(k) + 3] = PHOTO.POOLS; ph.dirty = true; }
      else remake(L, slot); // PHOTO_GROW, PHOTO_SHRINK
    }
    if (ph.dirty) ph.send();
  };
  ph.step = dt => ph.run(hs.hs_photo_step(dt, PHOTO.FLIGHT - ph.flight));
  // the pages there are (those over the tiles the map holds), a column and a row each, in pages.json's order; a pool as
  // deep as the GPU lets an array texture be
  const at = hs.hs_alloc(meta.pages.length * 4);
  new Uint16Array(hs.memory.buffer, at, meta.pages.length * 2).set(meta.pages.flatMap(([p, q]) => [p, q]));
  if (hs.hs_photo_init(meta.origin[0], meta.origin[1], meta.page, meta.across, meta.down, 1 / Math.max(...meta.px), at, meta.pages.length,
    device.limits.maxTextureArrayLayers)) throw new Error('could not page the photo');
  return ph;
}

// What the car's chase camera needs of the photo before the first frame, every page of it in
async function photoStart(ph, h, aspect) {
  for (;;) {
    ph.run(hs.hs_photo_start(h, aspect, 1 << 20));
    if (!ph.flight) break;
    while (ph.flight) await new Promise(r => setTimeout(r, 4));
  }
}

// The terrain's heights, tile by tile (wasm/include/harstad/tile.h): world.bin lists the tiles, the C orders them nearest
// the start first, and each tile's file is laid into its block of heights in that order (a block a tile, TILE_POSTS
// floats, never a grid over the tiles' bounding box). Every tile is waited for: until a far version of a tile can stand in
// for it nearer than half a pixel (tile_plan), the map loads whole. Returns the blocks in wasm memory, and a copy of them
// for the build worker taken before world_init eases their edges into the backdrop.
async function loadHeights([bin, size], [x, z]) {
  const at = hs.hs_tiles(bin, size, x, z);
  if (!at) throw new Error('world.bin holds no tiles');
  const [n, , , orderAt, tilesAt] = new Uint32Array(hs.memory.buffer, at, 5);
  const order = Array.from(new Uint32Array(hs.memory.buffer, orderAt, n)), tiles = Int32Array.from(new Int32Array(hs.memory.buffer, tilesAt, n * 6));
  const most = 32 + TILE_POSTS * 4, bytes = n * TILE_POSTS * 4, grid = hs.hs_alloc(bytes), scratch = hs.hs_alloc(most);
  const name = k => `${tiles[k * 6]}_${tiles[k * 6 + 1]}`;
  loading.terrain.files = n;
  const files = order.map(k => counted(`../data/tiles/${name(k)}.bin`, loading.terrain).then(r => r.arrayBuffer()));
  for (let q = 0; q < n; q++) {
    const f = await files[q];
    if (f.byteLength > most) throw new Error(`tile ${name(order[q])} is too large`);
    new Uint8Array(hs.memory.buffer, scratch, f.byteLength).set(new Uint8Array(f));
    if (hs.hs_tile_put(order[q], scratch, f.byteLength, grid)) throw new Error(`tile ${name(order[q])} does not fit the map`);
  }
  return { n, at: grid, bytes, copy: hs.memory.buffer.slice(grid, grid + bytes) };
}

async function init({ canvas: c, shared: sab, params, width, height, ratio }) {
  canvas = c;
  canvas.width = width; canvas.height = height;
  shared = sab;
  for (const k of (params.skip || '').split(',')) skip[k] = true;
  hdrWanted = params.hdr === '1';
  if (params.fmt) format = params.fmt; // ?fmt=: the picture's format (for measuring)
  skipDraws = (params.skip || '').includes(':');
  still = +params.still || 0;
  birdAtStart = 'bird' in params ? Math.max(1, +params.bird | 0) : 0; // ?still=3&bird=4000: out to the bird's-eye view on the first frame of play, that many m up over the car (?bird alone: GAME_BIRD_OPEN, the player's own), the same view every time
  filmFrames = Math.max(0, Math.min(36000, +params.film | 0));
  popFrames = Math.max(0, +params.pop | 0);
  birdStart = birdAtStart;
  paceOn = params.pace !== '0';
  clock = +params.clock || 0;
  rate = Math.max(1, Math.min(16, Math.round(+params.rate || 1)));
  if (params.aa === '1') SAMPLES = 1;
  if (params.dplain) dplain = true; // ?dplain: the far plain faces lit once a pixel too (WGSL.PLAIN_GBUF)
  if (params.deferMin) deferMin = +params.deferMin; // ?deferMin=0: the far ground lit once a pixel whenever it is there
  if (params.droadMin) droadMin = +params.droadMin; // ?droadMin=: the far road by its list only past this many of its indices a pixel
  splitting = 'split' in params;
  if (params.scale) SCALE = Math.min(1, Math.max(0.5, +params.scale || 1));
  if ('sharp' in params) SHARP = params.sharp === 'off' ? 0 : 2 ** -Math.max(0, +params.sharp || 0);
  // the shaders made with parts changed (?off=), or the aurora set (?aurora=, from the moment ?clock= starts play)
  if (params.off || params.aurora) WGSL = await import('./shaders.js?' + new URLSearchParams({ off: params.off || '', aurora: params.aurora || '', clock: params.clock || '', auroraat: params.auroraat || '' }));
  if (shared) { keysView = new Uint32Array(shared, 0, 1); padView = new Float32Array(shared, 4, 6); hudView = new Uint8Array(shared, SHARED_HUD, HUD_BYTES); }

  // The build worker is made first, so that its script is loaded and it is waiting by the time the files are here: its
  // script is fetched through this thread, which is then busy reading the world for a while.
  // Three of them, side by side (enum build_maps_mode): one bakes everything but the moon's field and the sky's map, which
  // are most of the bake's time, and one bakes each of those; the maps come back as one build would make them. Each holds
  // the world and its share of the bake (some 0.5 to 0.6 GB at the most), so a machine with few cores or little memory,
  // or ?builds=1, bakes in one as it did (?builds=3 the three anywhere)
  const MAPS = { ALL: 0, NONE: 1, MOON: 2, SKY: 3, PART: 4 }, names = { 0: 'buildWorker', 1: 'buildWorker', 2: 'buildMoon', 3: 'buildSky', 4: 'buildPart' };
  // The sky's map may be baked by bands of its rows in more of them (?skies=N, up to 8): each band is its rows of the map
  // as the whole bake makes them (wasm/src/build.c, build_band).
  // By default each of the three parts of the bake (the sky's map, the moon's field, everything else) is cut in three
  // on twelve cores or more, in two on eight to eleven, and left whole below: split alike, they end together. On the
  // M4 Pro (12 cores) 3, 3, 3 made ready 12.6 s cold where one each made it 19.2 (load 20-30 from other agents); each
  // build holds 0.5 to 0.9 GB while it runs, so the load's peak is some 1.5 to 2 GB higher.
  const cores = navigator.hardwareConcurrency || 4, roomy = cores >= 6 && (navigator.deviceMemory || 8) >= 8;
  const split = cores >= 12 ? 3 : cores >= 8 ? 2 : 1;
  const skies = Math.max(1, Math.min(8, Math.round(+params.skies) || split));
  // And the moon's field by strips of its rays in more (?moons=N, up to 8): strip 0 is the lead, which the others hand
  // their texels of each band of rows to, straight, and which hands the field out as one build would (moon.h,
  // moon_sweep_strip).
  const moons = Math.max(1, Math.min(8, Math.round(+params.moons) || split));
  // And everything but the maps by shares of its units in more (?geoms=N, up to 8): the tiles' roads and buildings and
  // the stages made whole, taken as they are free, the nearest the start first; the first, the lead, takes in what each
  // unit lays on the town's lights in one build's order and builds the lights (build.h, build_unit). The draws are keyed
  // by their units, so they come out in one build's order (game_add_items_keyed).
  const geoms = Math.max(1, Math.min(8, Math.round(+params.geoms) || split));
  const specs = params.builds === '1' || (params.builds !== '3' && !roomy) ? [{ mode: MAPS.ALL }]
    : [{ mode: MAPS.NONE, share: geoms > 1 ? [0, geoms] : null }, ...Array.from({ length: geoms - 1 }, (_, k) => ({ mode: MAPS.PART, share: [k + 1, geoms] })),
      ...Array.from({ length: moons }, (_, k) => ({ mode: MAPS.MOON, strip: moons > 1 ? [k, moons] : null })),
      ...Array.from({ length: skies }, (_, k) => ({ mode: MAPS.SKY, band: skies > 1 ? [k, skies] : null }))];
  const modes = specs.map(s => s.mode), named = specs.map(s => names[s.mode] + (s.band ? s.band[0] : s.strip ? s.strip[0] : s.share?.[0] ? s.share[0] : ''));
  // the stages that come before this side is ready to take them in, in the order they came; then what takes each as it comes
  const waiting = [];
  let take = null;
  builders = modes.map(() => new Worker(new URL('./build_worker.js', import.meta.url), { type: 'module' }));
  const built = Promise.all(builders.map((w, k) => new Promise((resolve, reject) => {
    w.onerror = e => reject(new Error(e.message));
    w.onmessage = ({ data }) => {
      if (data.failed) reject(new Error(data.failed));
      else if (data.lead) marks.buildLead = data.lead; // its world read and handed on
      else if (data.done) { w.terminate(); marks[named[k]] = data.ms; marks[named[k] + 'Memory'] = data.memory; if (data.waited) marks[named[k] + 'Waited'] = data.waited; mark(named[k] + 'Done'); resolve(); }
      else {
        progress(data.share);
        try { take ? take(data) : waiting.push(data); } catch (e) { reject(e); }
      }
    };
  })));
  built.catch(() => {}); // (awaited after the device and the world; an early failure is reported there)
  // everything that can start at once does: the device, the module, the data, the photo tiles
  const gpu = (async () => {
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('WebGPU is not available in this browser');
    // where the browser can time passes on the GPU itself, bench() reports that too: what the frame costs the GPU, apart from
    // whatever else the machine is doing
    const timed = adapter.features.has('timestamp-query');
    // the ground's photo is held compressed where the GPU can read BC7 (photoPaging). The limits are WebGPU's defaults,
    // which every GPU offers (watchLimits checks each pipeline against them), but for the photo's pools' layers: as deep
    // as the adapter allows
    device = await adapter.requestDevice({ requiredFeatures: ['timestamp-query', 'texture-compression-bc'].filter(f => adapter.features.has(f)),
      requiredLimits: { maxTextureArrayLayers: adapter.limits.maxTextureArrayLayers } });
    if ('limits' in params) pipeTable = [];
    watchLimits(device);
    pipesAFew(device, +params.pipes || 4);
    if (timed) timing = { set: device.createQuerySet({ type: 'timestamp', count: STAMPS.length * 2 }),
      resolve: device.createBuffer({ size: STAMPS.length * 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }),
      read: device.createBuffer({ size: STAMPS.length * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }) };
    // An error on the device can leave the canvas handing out textures that cannot be drawn to (its own could not be made,
    // when the GPU is short of memory): the frame would then fail the same way for ever. So after any error the canvas is
    // configured afresh and the frame's targets remade before the next frame; a lost device cannot be mended from here, and
    // the page is asked to start again where the car is.
    tallied(device);
    device.addEventListener('uncapturederror', e => { mend = true; if (errors++ < 8) post({ error: e.error.message }); });
    device.lost.then(info => { if (info.reason !== 'destroyed') post({ lost: info.message || 'the GPU device was lost' }); });
    context = canvas.getContext('webgpu');
    configure();
    // on an HDR canvas the shaders whose light may rise over white (WGSL HDR_ON), made so before any pipeline is
    if (hdr) WGSL = await import('./shaders.js?' + new URLSearchParams({ off: params.off || '', aurora: params.aurora || '', clock: params.clock || '', auroraat: params.auroraat || '', hdr: '1' }));
    mark('device');
    await makePipelines();
    mark('pipelines');
  })();
  const file = async url => (await counted(url, loading.data)).arrayBuffer();
  const [module, bin, outer, carModel] = await Promise.all([
    WebAssembly.compileStreaming(counted('../wasm/build/harstad.wasm', loading.data)),
    file('../data/world.bin'), file('../data/outer.bin'), file('../data/car.bin')]);

  const env = { sin: Math.sin, cos: Math.cos, tan: Math.tan, atan2: Math.atan2, exp: Math.exp, pow: Math.pow };
  hs = (await WebAssembly.instantiate(module, { env })).exports;
  const put = f => { const ptr = hs.hs_alloc(f.byteLength); new Uint8Array(hs.memory.buffer, ptr, f.byteLength).set(new Uint8Array(f)); return [ptr, f.byteLength]; };
  const binAt = put(bin), start = [+params.x, +params.z].map(v => Number.isFinite(v) ? v : 0); // (0, 0): the first map's middle, by Storgata
  const heights = await loadHeights(binAt, start);
  mark('fetched');
  // The first build worker gets the untouched files, and begins as soon as it has them: it reads the world for itself,
  // so the two read it side by side and the bake does not wait for this one. This instance keeps its own copies. The
  // others begin from its memory as its world_init left it (lead), read once for all of them: reading it is some
  // seconds each, and more side by side.
  const outerAt = put(outer);
  // the moon's strips each a port to their lead
  const strips = specs.map(s => s.strip && s.strip[0] ? new MessageChannel() : null), toStrips = strips.filter(c => c).map(c => c.port1);
  // the geometry's shares each a port to their lead, and one count of the units taken between them where memory is shared
  const shares = specs.map(s => s.share && s.share[0] ? new MessageChannel() : null), fromShares = shares.filter(c => c).map(c => c.port1);
  const counter = geoms > 1 && self.crossOriginIsolated ? new Int32Array(new SharedArrayBuffer(4)) : null;
  const lead = builders.slice(1).map((w, k) => {
    const { port1, port2 } = new MessageChannel(), s = specs[k + 1], extra = s.strip ? (s.strip[0] ? { toLead: strips[k + 1].port2 } : { strips: toStrips })
      : s.share ? { share: s.share, start, counter, toShare: shares[k + 1].port2 } : {};
    w.postMessage({ module, from: port2, mode: modes[k + 1], band: s.band, strip: s.strip, ...extra }, [port2, ...(extra.toLead ? [extra.toLead] : extra.strips || []), ...(extra.toShare ? [extra.toShare] : [])]);
    return port1;
  });
  const share = specs[0].share ? { share: specs[0].share, start, counter, fromShares } : {};
  builders[0].postMessage({ module, bin, heights: heights.copy, outer, mode: modes[0], lead, ...share }, [bin, heights.copy, outer, ...lead, ...(share.fromShares || [])]);
  if (hs.hs_world_init(...binAt, heights.at, heights.bytes, ...outerAt) || hs.hs_car_init(...put(carModel)) || hs.hs_game_init()) throw new Error('could not read the world');
  const lay = new Uint32Array(hs.memory.buffer, hs.hs_layout(), 30);
  if (lay[5] !== Object.keys(PIPE).length) throw new Error('the glue and the module disagree about the pipelines');
  if (lay[24] !== (WGSL.LIVE_SLOTS * 3 + WGSL.LIVE_TINTS) * 16) throw new Error('the glue and the module disagree about the live mesh');
  Object.assign(FRAME, { UNIFORMS: lay[0], PIPE_FIRST: lay[1], CAR: lay[2], CAR_BYTES: lay[3], CMDS: lay[4], SHADOW_FRESH: lay[6], SHADOW: lay[7], SHADOW_VP: lay[8], SNOW: lay[9], LIVE: lay[10], NBEAM: lay[11], BEAM: lay[12], BEAM_VP: lay[13], SHADOW_EYE: lay[14], CAM_POS: lay[15], GROUND_RECT: lay[16], NGROUND: lay[17], GROUND: lay[18], GROUND_DRAWS: lay[19], SEA: lay[20], WAVES: lay[21], WATER_FRESH: lay[22], POSE: lay[23], POSE_BYTES: lay[24], FOREST: lay[25], ODO: lay[26], PLOUGH: lay[27], PLOUGH_BYTES: lay[28],
    CAR_WASHED: lay[29] }); // WATER_FRESH is also where the frame's uniform blocks end
  lists.main.first = FRAME.PIPE_FIRST; lists.main.cmds = FRAME.CMDS; lists.main.nground = FRAME.NGROUND; lists.main.ground = FRAME.GROUND;
  mark('world');
  progress(1);

  const info = new DataView(hs.memory.buffer, hs.hs_info(), 96);
  const f32 = o => info.getFloat32(o, true), u32 = o => info.getUint32(o, true);
  const world = { bx: f32(0), bz: f32(4), w: f32(8), d: f32(12), step: f32(16), outerHalf: f32(20), nx: u32(24), nz: u32(28), outerN: u32(32),
    ncx: u32(36), ncz: u32(40), heights: u32(44), outer: u32(48), ex: u32(52), ez: u32(56), roads: u32(60), roadPts: u32(64), strings: u32(68),
    nroads: u32(72), lampW: u32(76), lampH: u32(80), spawn: [f32(84), f32(88), f32(92)] };
  applyParams(params, world);
  // the look tests (tools/look/run.mjs, __next.look): the next view in the same page, set up as its address would set it up,
  // the camera as at the start, and played ?still seconds from there
  lookAgain = p => { hs.hs_cam_reset(); applyParams(p, world); hs.hs_event(EVENT.BEAM, 'beam' in p ? +p.beam | 0 : 2); // (the page sends ?beam at the start)
    still = +p.still || still; played = 0; birdAtStart = 'bird' in p; last = 0; };
  hs.hs_set(SET.PIXEL_RATIO, (wantedRatio || ratio || 1) * SCALE); // what is sized in CSS pixels is sized in the pixels drawn
  const images = loadPhotos(start);
  postMinimap(world);

  await gpu;
  const mem = () => hs.memory.buffer;
  // The terrain's heights a layer a tile, the layer the tile's number (wasm/src/wasm.c, hs_tile_layer): for the
  // shaders' looks at the ground (WGSL GROUND_LOOK) in half floats, the tile's posts; for the terrain's mesh in float32,
  // and a post round them. And which layer each tile of the tiles' grid is.
  const groundTables = {}; // the ground's tiles and the backdrop's heights, for the frame's tables (tableAtlas)
  let seaPagesTable = null, pyramidTables = null; // and the sea's page table and the town-wide maps'
  const { ground: groundTex, heights: heightTex, tiles: tileLayers } = (() => {
    const n = heights.n, A = TILE_SIDE + 2;
    const make = (format, side) => device.createTexture({ size: [side, side, n], format, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    const ground = make('r16float', TILE_SIDE), lifted = make('r32float', A), half = hs.hs_alloc(TILE_POSTS * 2), full = hs.hs_alloc(A * A * 4);
    for (let k = 0; k < n; k++) {
      hs.hs_tile_layer(k, half, full);
      device.queue.writeTexture({ texture: ground, origin: [0, 0, k] }, new Uint8Array(mem(), half, TILE_POSTS * 2), { bytesPerRow: TILE_SIDE * 2 }, [TILE_SIDE, TILE_SIDE]);
      device.queue.writeTexture({ texture: lifted, origin: [0, 0, k] }, new Uint8Array(mem(), full, A * A * 4), { bytesPerRow: A * 4 }, [A, A]);
    }
    const at = hs.hs_held(), [ni, nj] = new Uint32Array(mem(), at, 2), [x0, z0] = new Float32Array(mem(), at + 8, 2);
    world.tileI = Math.round((x0 - world.bx) / world.step); world.tileJ = Math.round((z0 - world.bz) / world.step); // the grid point at the tiles' corner
    const tiles = new Int32Array(mem(), hs.hs_tile_at(), ni * nj).slice();
    groundTables.tiles = { w: ni, h: nj, data: new Uint32Array(tiles.buffer) };
    return { ground, heights: lifted, tiles: dataTexture('r32sint', ni, nj, tiles, ni * 4) };
  })();
  // the backdrop's heights the same way, one layer and a post round it (the edge's again), and its one tile
  const outerTex = (() => {
    const on = world.outerN, A = on + 2, o = new Float32Array(mem(), world.outer, on * on), p = new Float32Array(A * A);
    for (let j = 0; j < A; j++) for (let i = 0; i < A; i++) p[j * A + i] = o[Math.min(Math.max(j - 1, 0), on - 1) * on + Math.min(Math.max(i - 1, 0), on - 1)];
    const texture = device.createTexture({ size: [A, A, 1], format: 'r32float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    device.queue.writeTexture({ texture }, p, { bytesPerRow: A * 4 }, [A, A]);
    groundTables.back = { w: A, h: A, data: new Uint32Array(p.buffer) };
    return texture;
  })();
  const outerTile = dataTexture('r32sint', 1, 1, new Int32Array([0]), 4);
  const detail = arrayTexture([noiseTexture(), snowRelief()], 'rgba8unorm'); // the fine noise and the snow's relief (WGSL detail_map)
  const [timber, timberGlow] = facadeTextures(true), [block, blockGlow] = facadeTextures(false);
  const facades = arrayTexture([timber, block], 'rgba8unorm-srgb'), glows = arrayTexture([timberGlow, blockGlow], 'rgba8unorm-srgb');
  const styled = facadeStyles(); // the centre's facade styles, at their own size
  const styles = arrayTexture(styled.map(s => s[0]), 'rgba8unorm-srgb'), styleGlows = arrayTexture(styled.map(s => s[1]), 'rgba8unorm-srgb');
  const roads = arrayTexture([roadTexture(0), roadTexture(1), roadTexture(2)], 'rgba8unorm-srgb');
  mark('textures');

  const sampler = (o = {}) => device.createSampler({ minFilter: 'linear', magFilter: 'linear', mipmapFilter: 'linear', ...o });
  const repeat = { addressModeU: 'repeat', addressModeV: 'repeat' };
  groups.wall = device.createBindGroup({ layout: groupLayouts.wall, entries: [
    { binding: 0, resource: arrayView(facades) }, { binding: 1, resource: arrayView(glows) }, { binding: 2, resource: sampler({ ...repeat, maxAnisotropy: 8 }) },
    { binding: 3, resource: arrayView(styles) }, { binding: 4, resource: arrayView(styleGlows) }] });
  groups.pane = device.createBindGroup({ layout: groupLayouts.road, entries: [
    { binding: 0, resource: arrayView(arrayTexture(displayAtlas(), 'rgba8unorm-srgb')) }, { binding: 1, resource: sampler({ maxAnisotropy: 8 }) }] });
  wet.road = [{ binding: 0, resource: arrayView(roads) }, { binding: 1, resource: sampler({ addressModeV: 'repeat', maxAnisotropy: 16 }) }]; // with the wet tracks' lights: wetBind()
  wet.skySampler = sampler({ addressModeU: 'repeat' }); // the sky the water mirrors, round the horizon
  groups.tube = device.createBindGroup({ layout: groupLayouts.road, entries: [
    { binding: 0, resource: arrayView(arrayTexture(tunnelTextures(), 'rgba8unorm-srgb')) }, { binding: 1, resource: sampler({ addressModeV: 'repeat', maxAnisotropy: 16 }) }] });
  groups.paving = device.createBindGroup({ layout: groupLayouts.road, entries: [
    { binding: 0, resource: arrayView(arrayTexture(pavingTextures(), 'rgba8unorm-srgb')) }, { binding: 1, resource: sampler({ ...repeat, maxAnisotropy: 8 }) }] });

  const photoBinds = []; // what is bound to the photo's pools, made again when a pool is
  const photo = await photoPaging(images, () => photoBinds.forEach(f => f()));
  photoPool = photo;
  await photoStart(photo, Math.round(canvas.height * SCALE), canvas.width / Math.max(1, canvas.height));
  progress(1);
  mark('tiles');
  const photoSampler = sampler({ maxAnisotropy: 8 });
  { // each chunk's lambda, sent every frame (hs_ground_lod): the town's ground's, then the backdrop's
    const l = new Uint32Array(mem(), hs.hs_ground_lod(), 8);
    groundLod = [0, 4].map(o => { const n = l[o + 2] * l[o + 3];
      return { at: l[o], n, chunks: [l[o + 1], l[o + 2], l[o + 3]], buffer: device.createBuffer({ size: Math.max(16, n * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }) }; });
  }
  const terrainGroup = gridGroup({ origin: [world.bx, world.bz], step: world.step, clip: 0, size: [world.nx, world.nz], extent: [world.w, world.d],
    chunks: groundLod[0].chunks, lod: groundLod[0].buffer,
    heights: heightTex, tiles: tileLayers, tile: [world.tileI, world.tileJ, TILE_STEPS], photo, sampler: photoSampler, cuts: new Float32Array(mem(), hs.hs_tunnel_cuts(), 32), keep: snow });
  photoBinds.push(() => { groups.terrain = terrainGroup(); });
  const on = world.outerN, ostep = 2 * world.outerHalf / (on - 1);
  { // the tiles the map holds, which the backdrop is cut away under (hs_held): a rectangle a tile of the tiles' grid
    const at = hs.hs_held(), [ni, nj] = new Uint32Array(mem(), at, 2), f = new Float32Array(mem(), at, 5 + ni * nj * 4);
    photo.outer.pages = dataTexture('rgba32float', ni, nj, f.slice(5), ni * 16);
    groups.outer = gridGroup({ origin: [-world.outerHalf, -world.outerHalf], step: ostep, clip: 1, size: [on, on], extent: [2 * world.outerHalf, 2 * world.outerHalf],
      chunks: groundLod[1].chunks, lod: groundLod[1].buffer,
      heights: outerTex, tiles: outerTile, tile: [0, 0, 1 << 30], photo: photo.outer, sampler: photoSampler, held: [f[2], f[3], f[4]] })();
  }
  rowIndex = Array.from({ length: LEVELS + 1 }, (_, k) => rowStrip(world.nx, k)); outerRowIndex = Array.from({ length: LEVELS + 1 }, (_, k) => rowStrip(on, k));
  // every uniform block that changes each frame, in one buffer as struct frame lays them out, sent with one write
  frameBuffer = device.createBuffer({ size: FRAME.WATER_FRESH, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }); // (the live mesh's pose in it is read as storage)
  { // the snow: its arrays live in wasm memory and are sent in pieces as the car changes them
    const [stage, stageSlot, flecks, tracks, , , inst] = new Uint32Array(mem(), hs.hs_snow(), 7);
    // the player's ribbon, one strip round its ring; then the traffic's runs (SNOW_RUNS of SNOW_RUN), each a strip of its own, none joined to the next
    const DST = GPUBufferUsage.COPY_DST, ribbon = new Uint32Array((SNOW_TRACKS - 1 + SNOW_RUNS * (SNOW_RUN - 1)) * 9 * 6);
    // a block's mesh at each of its levels, 16 >> level quads a side, one after another (WGSL.PACK; snow_view)
    const mesh = [];
    for (let l = 0; l < WGSL.SNOW_MESHES; l++) {
      const n = 16 >> l, at = mesh.length;
      for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) { const a = j * (n + 1) + i, b = a + 1, c = a + n + 1, d = c + 1; mesh.push(a, c, b, b, c, d); }
      PACK_MESH[l] = [at, n * n * 6];
    }
    const strip = (q, first, n) => { for (let s = first; s < first + n - 1; s++) for (let v = 0; v < 9; v++) { const a = s * 10 + v, b = a + 10; ribbon[q++] = a; ribbon[q++] = a + 1; ribbon[q++] = b; ribbon[q++] = a + 1; ribbon[q++] = b + 1; ribbon[q++] = b; } return q; };
    for (let k = 0, q = strip(0, 0, SNOW_TRACKS); k < SNOW_RUNS; k++) q = strip(q, SNOW_TRACKS + k * SNOW_RUN, SNOW_RUN);
    Object.assign(snow, { stage, stageSlot, inst, flecks, tracks, slotCap: 512,
      slots: device.createBuffer({ size: 512 * WGSL.SNOW_SLOT_WORDS * 4, usage: GPUBufferUsage.STORAGE | DST | GPUBufferUsage.COPY_SRC }),
      blocks: device.createBuffer({ size: SNOW_SLOTS_MAX * 48, usage: GPUBufferUsage.STORAGE | DST }),
      mesh: gpuBuffer(new Uint16Array(mesh), GPUBufferUsage.INDEX), // (2046 indices: a whole number of words)
      spray: device.createBuffer({ size: 1536 * 28, usage: GPUBufferUsage.VERTEX | DST }),
      ribbon: device.createBuffer({ size: (SNOW_TRACKS + SNOW_RUNS * SNOW_RUN) * 10 * 16, usage: GPUBufferUsage.VERTEX | DST }),
      ribbonIndex: gpuBuffer(ribbon, GPUBufferUsage.INDEX), ribbonCount: (SNOW_TRACKS - 1) * 54, runs: 0 });
    device.queue.writeBuffer(snow.spray, 0, new Uint8Array(mem(), flecks, 1536 * 28));
    snow.bind = () => { snow.group = device.createBindGroup({ layout: groupLayouts.snow, entries: [{ binding: 0, resource: { buffer: frameBuffer, offset: FRAME.SNOW, size: 48 } },
      { binding: 1, resource: { buffer: snow.slots } }, { binding: 2, resource: { buffer: snow.blocks } },
      { binding: 3, resource: tileLayers.createView() }, { binding: 4, resource: arrayView(heightTex) },
      { binding: 5, resource: { buffer: snow.grid } }, ...photoEntries(photo, photoSampler, 6)] }); };
    photoBinds.push(() => snow.bind());
    { // the drifts beyond it (wasm/include/harstad/dune.h): the corners' pages sent as they are filled, the nodes each frame
      const [open, fresh, head, relief] = new Uint32Array(mem(), hs.hs_dune(), 4), T = GPUTextureUsage;
      Object.assign(dune, { openAt: open, fresh, head, count: 0,
        open: device.createTexture({ size: [WGSL.DUNE_N, WGSL.DUNE_N], format: 'r8unorm', usage: T.TEXTURE_BINDING | T.COPY_DST }),
        relief: dataTexture('r8unorm', WGSL.DUNE_RELIEF, WGSL.DUNE_RELIEF, new Uint8Array(mem(), relief, WGSL.DUNE_RELIEF ** 2).slice(), WGSL.DUNE_RELIEF),
        buffer: device.createBuffer({ size: 48 + DUNE_NODES_MAX * 16, usage: GPUBufferUsage.STORAGE | DST }) });
      dune.bind = () => { dune.group = device.createBindGroup({ layout: groupLayouts.dune, entries: [{ binding: 0, resource: { buffer: dune.buffer } },
        { binding: 1, resource: dune.open.createView() }, { binding: 2, resource: dune.relief.createView() },
        { binding: 3, resource: tileLayers.createView() }, { binding: 4, resource: arrayView(heightTex) },
        { binding: 5, resource: { buffer: snow.grid } }, ...photoEntries(photo, photoSampler, 6)] }); };
      photoBinds.push(() => dune.bind());
    }
    photoBinds.forEach(f => f());
  }
  { // the gritted track (wasm/src/plough.c): a ring of sections in wasm memory, sent as the gritter lays them
    const [v, sections, across, state] = new Uint32Array(mem(), hs.hs_grit(), 4), index = new Uint32Array((sections - 1) * (across - 1) * 6);
    for (let s = 0, q = 0; s < sections - 1; s++) for (let k = 0; k < across - 1; k++) { const a = s * across + k, b = a + across; index[q++] = a; index[q++] = a + 1; index[q++] = b; index[q++] = a + 1; index[q++] = b + 1; index[q++] = b; }
    Object.assign(grit, { at: v, sections, across, state: state >> 2, laid: 0, count: index.length,
      buffer: device.createBuffer({ size: sections * across * 16, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST }), index: gpuBuffer(index, GPUBufferUsage.INDEX) });
  }
  { // what moves in the sky: its arrays live in wasm memory too. The mesh is sent once; where its parts are rides in the frame's uniforms
    const [v, idx, glow, boxes, nv, ni, glows, puffs] = new Uint32Array(mem(), hs.hs_live(), 8), DST = GPUBufferUsage.COPY_DST;
    const [lightsAt, gridAt, counts, nlights, words] = new Uint32Array(mem(), hs.hs_shine(), 5);
    if (nlights !== WGSL.SHINE_LIGHTS || words !== WGSL.SHINE_GRID_WORDS) throw new Error('the light list: shine.h and WGSL.SHINE differ');
    Object.assign(shine, { lightsAt, gridAt, counts,
      lights: device.createBuffer({ size: nlights * WGSL.SHINE_LIGHT_BYTES, usage: GPUBufferUsage.STORAGE | DST }),
      grid: device.createBuffer({ size: words * 4, usage: GPUBufferUsage.STORAGE | DST }) });
    Object.assign(live, { v, glowAt: glow, boxes, puffs,
      vertex: device.createBuffer({ size: Math.max(20, nv * 20), usage: GPUBufferUsage.VERTEX | DST }),
      index: gpuBuffer(new Uint8Array(mem(), idx, Math.max(4, ni * 4)), GPUBufferUsage.INDEX),
      glow: device.createBuffer({ size: glows * 32, usage: GPUBufferUsage.VERTEX | DST }),
      cloud: device.createBuffer({ size: CLOUDS * CLOUD_BOX, usage: GPUBufferUsage.STORAGE | DST }),
      puff: device.createBuffer({ size: CLOUDS * WGSL.CLOUD_PUFFS * CLOUD_BOX, usage: GPUBufferUsage.STORAGE | DST }), // struct cloud_puff, as big as a box
      pose: device.createBindGroup({ layout: groupLayouts.live, entries: [{ binding: 0, resource: { buffer: frameBuffer, offset: FRAME.POSE, size: FRAME.POSE_BYTES } }] }),
      noise: [{ binding: 0, resource: volumeTexture(cloudVolume()).createView() },
        { binding: 1, resource: device.createSampler({ minFilter: 'linear', magFilter: 'linear', mipmapFilter: 'linear', addressModeU: 'repeat', addressModeV: 'repeat', addressModeW: 'repeat' }) }] });
    live.group = device.createBindGroup({ layout: groupLayouts.cloud, entries: [...live.noise, { binding: 2, resource: { buffer: live.puff } }, { binding: 3, resource: { buffer: live.cloud } }] });
    // the traffic's bodies, a look each (hs_herd), and the cars the view shows of each look
    const [hv, hi, hnv, hni, runs, seen, looks, cars] = new Uint32Array(mem(), hs.hs_herd(), 8); // (cars: TRAFFIC_CARS, the most a view shows)
    Object.assign(herd, { looks, runs, seen, ni: hni,
      vertex: gpuBuffer(new Uint8Array(mem(), hv, Math.max(20, hnv * 20)), GPUBufferUsage.VERTEX),
      index: gpuBuffer(new Uint8Array(mem(), hi, Math.max(4, hni * 4)), GPUBufferUsage.INDEX),
      inst: device.createBuffer({ size: (1 + 2 * looks + cars) * 4, usage: GPUBufferUsage.VERTEX | DST }) });
  }
  { // the fjord (src/water.js): what is redrawn each frame, and the horizon maps, marched once here on the GPU
    const DST = GPUBufferUsage.COPY_DST, N = WGSL.RIPPLE_N, whole = { topology: 'triangle-list' };
    Object.assign(sea, { at: hs.hs_water(),
      ripples: drawnTextures(2, N, N, 'rgba16float'), sky: drawnTextures(1, WGSL.SKY_W, WGSL.SKY_H, 'rgba16float') });
    // the ripple tiles, the water's sky and the view's sky noise, worked out in the compute pass that also halves the mips
    const maps = device.createShaderModule({ code: WGSL.PRELUDE + WGSL.SKY_MAPS }), rippleMaps = device.createShaderModule({ code: WGSL.RIPPLE_MAPS });
    const store = binding => ({ binding, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } });
    const C = GPUShaderStage.COMPUTE, frameEntry = { binding: 0, resource: { buffer: frameBuffer, offset: 0, size: FRAME.UNIFORMS } };
    // the aurora's rows are written first in the pass, then read by the sky's maps here and by the frame's shaders
    skyLow.rows = device.createBuffer({ size: (64 + 16 + 8 + 60 * 1024) * 4, usage: GPUBufferUsage.STORAGE });
    skyLow.uniforms = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: C, buffer: {} }, { binding: 11, visibility: C, buffer: { type: 'read-only-storage' } }] });
    skyLow.outputs = device.createBindGroupLayout({ entries: [store(0), store(1)] });
    const mapsLayout = device.createPipelineLayout({ bindGroupLayouts: [skyLow.uniforms, skyLow.outputs] });
    skyLow.pano = device.createComputePipeline({ layout: mapsLayout, compute: { module: maps, entryPoint: 'pano' } });
    skyLow.pipe = device.createComputePipeline({ layout: mapsLayout, compute: { module: maps, entryPoint: 'low' } });
    skyLow.frame = device.createBindGroup({ layout: skyLow.uniforms, entries: [frameEntry, { binding: 11, resource: { buffer: skyLow.rows } }] });
    const rowsIn = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: C, buffer: {} }] });
    const rowsOut = device.createBindGroupLayout({ entries: [{ binding: 2, visibility: C, buffer: { type: 'storage' } }] });
    const rowsLayout = device.createPipelineLayout({ bindGroupLayouts: [rowsIn, rowsOut] });
    skyLow.rowsPipe = device.createComputePipeline({ layout: rowsLayout, compute: { module: maps, entryPoint: 'rows' } });
    // and from them the aurora's light, into the same buffer (WGSL.AURORA_LIGHT)
    skyLow.lightPipe = device.createComputePipeline({ layout: rowsLayout, compute: { module: device.createShaderModule({ code: WGSL.AURORA_LIGHT }), entryPoint: 'main' } });
    skyLow.rowsGroups = [device.createBindGroup({ layout: rowsIn, entries: [frameEntry] }), device.createBindGroup({ layout: rowsOut, entries: [{ binding: 2, resource: { buffer: skyLow.rows } }] })];
    sea.ripplePipe = device.createComputePipeline({ layout: 'auto', compute: { module: rippleMaps } });
    sea.rippleGroup = device.createBindGroup({ layout: sea.ripplePipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: frameBuffer, offset: FRAME.WAVES, size: WAVES_BYTES } },
      { binding: 1, resource: sea.ripples.tops[0] }, { binding: 2, resource: sea.ripples.tops[1] }] });

    const S = WGSL.HORIZON_SIZE, march = device.createShaderModule({ code: WGSL.HORIZON }), outerHalf = hs.hs_alloc(on * on * 2);
    hs.hs_outer_half(outerHalf);
    const outerSmooth = dataTexture('r16float', on, on, new Uint8Array(mem(), outerHalf, on * on * 2), on * 2);
    const pipe = device.createRenderPipeline({ layout: 'auto', vertex: { module: march }, fragment: { module: march, targets: [{ format: 'rgba8unorm' }] }, primitive: whole });
    const enc = device.createCommandEncoder();
    // the two maps (WGSL.HORIZON), a layer each of one texture
    const horizon = device.createTexture({ size: [S, S, 2], format: 'rgba8unorm', mipLevelCount: mips(S, S), usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT });
    [0, 4].forEach((first, layer) => {
      const u = new Float32Array([world.bx - world.step / 2, world.bz - world.step / 2, world.nx * world.step, world.nz * world.step,
        -world.outerHalf, -world.outerHalf, 2 * world.outerHalf, 2 * world.outerHalf, first, on, 0, 0, world.tileI, world.tileJ, world.nx, world.nz,
        -world.outerHalf, -world.outerHalf, 2 * world.outerHalf / (on - 1), on]);
      const pass = enc.beginRenderPass({ colorAttachments: [{ view: horizon.createView({ dimension: '2d', baseMipLevel: 0, mipLevelCount: 1, baseArrayLayer: layer, arrayLayerCount: 1 }), loadOp: 'clear', storeOp: 'store' }] });
      pass.setPipeline(pipe);
      pass.setBindGroup(0, device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: gpuBuffer(u, GPUBufferUsage.UNIFORM) } },
        { binding: 1, resource: arrayView(groundTex) }, { binding: 2, resource: outerSmooth.createView() }, { binding: 3, resource: device.createSampler({ minFilter: 'linear', magFilter: 'linear' }) },
        { binding: 4, resource: tileLayers.createView() }, { binding: 5, resource: arrayView(outerTex) }] }));
      pass.draw(3); pass.end();
    });
    device.queue.submit([enc.finish()]);
    makeMips(horizon);
    sea.horizon = horizon;
    sea.floes = arrayTexture([floePattern()], 'rgba8unorm');
    const ribbon = new Uint16Array(47 * 6);
    for (let i = 0; i < 47; i++) ribbon.set([2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 1, 2 * i + 3, 2 * i + 2], i * 6);
    Object.assign(sea, { wake: device.createBuffer({ size: WAKE_BYTES, usage: GPUBufferUsage.VERTEX | DST }), wakeIndex: gpuBuffer(ribbon, GPUBufferUsage.INDEX),
      rings: device.createBuffer({ size: RINGS_BYTES, usage: GPUBufferUsage.VERTEX | DST }), drops: device.createBuffer({ size: DROPS_BYTES, usage: GPUBufferUsage.VERTEX | DST }) });
  }
  {
    const [v, nv, idx, solid, blended] = new Uint32Array(mem(), hs.hs_avensis(), 5), plate = plateTexture('ZH 68801');
    // copied to from canvases: the instrument cluster in two layers, as it stands and with its lamps lit, and the lit panels round the cabin
    const picture = (w, h, layers = 1) => device.createTexture({ size: [w, h, layers], format: 'rgba8unorm-srgb', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT });
    // The solid triangles in runs, in their own order, each a lens's (its first vertex a lamp's, with a chamber behind it;
    // the shaders' provoking vertex) or not: the lenses alone are drawn with the shader that has the chambers in it.
    // The order is kept, so where two faces lie at one depth the same one wins as when they were one draw.
    const bytes = new Uint8Array(mem(), v, nv * 40), tri = new Uint32Array(mem(), idx, solid + blended), runs = [];
    const isLens = t => { const b = tri[t] * 40; return bytes[b + 36] === 4 /* AV_LAMP */ && bytes[b + 38] !== 0; };
    const isFlight = t => bytes[tri[t] * 40 + 36] === 13; // AV_FLIGHT: the flight instruments have a shader of their own too,
    const isFinish = t => { const b = tri[t] * 40; return bytes[b + 36] === 5 /* AV_CABIN */ && bytes[b + 38] !== 0; }; // and the cabin's finishes
    for (let t = 0; t < solid; t += 3) {
      const lens = isLens(t), flight = !lens && isFlight(t), cabin = !lens && !flight && isFinish(t), last = runs[runs.length - 1];
      if (last && last.lens === lens && last.flight === flight && last.cabin === cabin) last.count += 3;
      else runs.push({ first: t, count: 3, lens, flight, cabin });
    }
    for (let t = solid; t < solid + blended; t += 3) if (isLens(t)) throw new Error('a lens among the car\'s blended faces: its shader has no chambers');
    car = { solid, blended, runs, dash: dashPainter(),
      vertex: gpuBuffer(new Uint8Array(mem(), v, nv * 40), GPUBufferUsage.VERTEX), index: gpuBuffer(new Uint8Array(mem(), idx, (solid + blended) * 4), GPUBufferUsage.INDEX),
      dials: picture(1024, 320, 3), panels: picture(512, 256) };
    car.dash.layers.forEach((source, layer) => device.queue.copyExternalImageToTexture({ source }, { texture: car.dials, origin: [0, 0, layer] }, [1024, 320]));
    device.queue.copyExternalImageToTexture({ source: car.dash.panels }, { texture: car.panels }, [512, 256]);
    car.plate = dataTexture('rgba8unorm-srgb', 256, 64, plate.data, 256 * 4).createView(); car.dialsView = car.dials.createView({ dimension: '2d-array' });
    car.sky = [{ binding: 4, resource: sea.sky.textures[0].createView() }, { binding: 5, resource: wet.skySampler }]; // the sky the gloss black mirrors (WGSL.CAR_CABIN)
    car.group = device.createBindGroup({ layout: groupLayouts.car, entries: [{ binding: 0, resource: { buffer: frameBuffer, offset: FRAME.CAR, size: FRAME.CAR_BYTES } },
      { binding: 1, resource: car.plate }, { binding: 2, resource: car.dialsView }, { binding: 3, resource: car.panels.createView() }, ...car.sky] });
  }
  { // the jets: their nozzles are drawn by the car's pipeline from a uniform block of their own; the puffs' ring starts long settled
    const [v, nv, idx, ni, pose, jets, puffs, state, drift] = new Uint32Array(mem(), hs.hs_rocket(), 9), DST = GPUBufferUsage.COPY_DST;
    Object.assign(rocket, { pose, jetsAt: jets, puffsAt: puffs, state: state >> 2, drift: drift >> 2, count: ni,
      vertex: gpuBuffer(new Uint8Array(mem(), v, nv * 40), GPUBufferUsage.VERTEX), index: gpuBuffer(new Uint8Array(mem(), idx, ni * 4), GPUBufferUsage.INDEX),
      uniform: device.createBuffer({ size: FRAME.CAR_BYTES, usage: GPUBufferUsage.UNIFORM | DST }),
      jets: device.createBuffer({ size: ROCKET.FLAMES * ROCKET.JET, usage: GPUBufferUsage.VERTEX | DST }),
      puffs: gpuBuffer(new Uint8Array(mem(), puffs, ROCKET.PUFFS * ROCKET.PUFF), GPUBufferUsage.VERTEX | DST) });
    rocket.group = device.createBindGroup({ layout: groupLayouts.car, entries: [{ binding: 0, resource: { buffer: rocket.uniform } },
      { binding: 1, resource: car.plate }, { binding: 2, resource: car.dialsView }, { binding: 3, resource: car.panels.createView() }, ...car.sky] });
  }
  { // the flight cockpit: its mesh is drawn by the car's pipeline from two uniform blocks of its own, each its own run of indices
    const [v, nv, idx, ni, pose, frame] = new Uint32Array(mem(), hs.hs_cockpit(), 6), DST = GPUBufferUsage.COPY_DST;
    const first = new Uint32Array(mem(), frame, 3), solid = new Uint32Array(mem(), frame + 12, 2);
    Object.assign(cockpit, { pose, vertex: gpuBuffer(new Uint8Array(mem(), v, nv * 40), GPUBufferUsage.VERTEX), index: gpuBuffer(new Uint8Array(mem(), idx, ni * 4), GPUBufferUsage.INDEX),
      runs: [0, 1].map(b => ({ first: first[b], solid: solid[b], blended: first[b + 1] - first[b] - solid[b],
        uniform: device.createBuffer({ size: FRAME.CAR_BYTES, usage: GPUBufferUsage.UNIFORM | DST }) })) });
    for (const r of cockpit.runs) r.group = device.createBindGroup({ layout: groupLayouts.car, entries: [{ binding: 0, resource: { buffer: r.uniform } },
      { binding: 1, resource: car.plate }, { binding: 2, resource: car.dialsView }, { binding: 3, resource: car.panels.createView() }, ...car.sky] });
  }
  { // the doors as wings and the canopy: as the cockpit, uniform blocks of its own (wing.h, WING_BLOCKS), each its own run of indices
    const [v, nv, idx, ni, pose, frame] = new Uint32Array(mem(), hs.hs_wing(), 6), DST = GPUBufferUsage.COPY_DST;
    const BLOCKS = 3; // WING_BLOCKS
    const first = new Uint32Array(mem(), frame, BLOCKS + 1), solid = new Uint32Array(mem(), frame + 4 * (BLOCKS + 1), BLOCKS);
    Object.assign(wing, { pose, shown: (frame >> 2) + 2 * BLOCKS + 1, vertex: gpuBuffer(new Uint8Array(mem(), v, nv * 40), GPUBufferUsage.VERTEX), index: gpuBuffer(new Uint8Array(mem(), idx, ni * 4), GPUBufferUsage.INDEX),
      runs: [...Array(BLOCKS).keys()].map(b => ({ first: first[b], solid: solid[b], blended: first[b + 1] - first[b] - solid[b],
        uniform: device.createBuffer({ size: FRAME.CAR_BYTES, usage: GPUBufferUsage.UNIFORM | DST }) })) });
    for (const r of wing.runs) r.group = device.createBindGroup({ layout: groupLayouts.car, entries: [{ binding: 0, resource: { buffer: r.uniform } },
      { binding: 1, resource: car.plate }, { binding: 2, resource: car.dialsView }, { binding: 3, resource: car.panels.createView() }, ...car.sky] });
  }
  snowfall();
  quadIndex = gpuBuffer(new Uint16Array([0, 1, 2, 0, 2, 3]), GPUBufferUsage.INDEX); // a sprite's two triangles (WGSL QUAD)
  // a far tree's quad (WGSL forestImpostor): of its two triangles' six corners, 3 is 1 again and 5 is 2, so by index its vertex
  // shader, which works out the four views' frames, runs four times a tree where it ran six
  treeQuad = gpuBuffer(new Uint16Array([0, 1, 4, 0, 4, 2]), GPUBufferUsage.INDEX); // both triangles from corner 0, whose work the pixels take flat (WGSL.forestImpostor)

  // geometry, as the build workers finish each stage: taken in as it comes (those that came before this point, in the
  // order they came), so its buffers are on the GPU and let go here while the bake goes on, not all at its end. The draw
  // items keep their order: only the build of everything but the maps makes any, and its stages come in order.
  let lampSwitch, lampTex, lampPages, lampPagesAt, reachPagesAt, lampGlow, tintTex, tintPages, tintPagesAt, seaMap, seaPages, treeModels, moonField, moonReach, reachPages, skyMap, mapPages, skyGlow, triangles = 0, bytes = 0;
  const moonParts = [], skyParts = [], skyBands = []; // skyBands[k]: { rows, parts, pages } of band k of the sky's map
  let mapPagesData = null; // the moon's and the sky's page table, each build's channels of it joined (tiles.h: r g the moon's, b a the sky's)
  let handles = 0; // each step's buffers numbered afresh, whichever build made them: the shares number theirs each from 0
  const takeStage = stage => {
    { // its draws' buffers (struct gfx_item: pipe, vbuf, ibuf, inst) by the numbers given here
      const it = new Uint32Array(stage.items), base = handles;
      for (let k = 0; k < it.length; k += 16) for (let f = 1; f <= 3; f++) if (it[k + f] !== NONE) it[k + f] += base - stage.first;
      handles += stage.bufs.length;
      stage.first = base;
    }
    stage.bufs.forEach((b, i) => {
      if (b.kind === BUF.MOON_FIELD || b.kind === BUF.MOON_REACH || b.kind === BUF.REACH_PAGES || b.kind === BUF.MAP_PAGES) marks.moonBytes = (marks.moonBytes || 0) + b.data.byteLength; // the moon's field is no geometry
      else if (b.kind === BUF.SKY_MAP) marks.skyBytes = (marks.skyBytes || 0) + b.data.byteLength; // nor the sky's
      else if (b.kind === BUF.SKYGLOW) marks.glowBytes = b.data.byteLength; // nor the town's light round it
      else bytes += b.data.byteLength;
      if (b.kind === BUF.LAMP_MAP) lampTex = layerTexture('rgba8unorm', b.w, b.h, b.data, 4, 'lamp', GPUTextureUsage.COPY_SRC); // the lamp map's tiles (wasm/src/lamp.c)
      else if (b.kind === BUF.LAMP_PAGES) lampPagesAt = b; // (their tables, with their levels', once all is in)
      else if (b.kind === BUF.TINT_MAP) tintTex = layerTexture('rg8unorm', b.w, b.h, b.data, 2, 'tint', GPUTextureUsage.COPY_SRC); // the tint's tiles (wasm/src/lamp.c)
      else if (b.kind === BUF.TINT_PAGES) tintPagesAt = b;
      else if (b.kind === BUF.LAMP_SWITCH) { lampSwitch = b.data; marks.switchBytes = b.data.byteLength; } // (lamp.h, struct lamp_switches)
      else if (b.kind === BUF.SEA_MAP) seaMap = b;
      else if (b.kind === BUF.SEA_PAGES) seaPages = b;
      else if (b.kind === BUF.MOON_FIELD) moonParts.push(layerPart('r16float', b, 2, 'moon')); // its layers, a few at a time: one texture of them all below
      else if (b.kind === BUF.MOON_REACH) moonReach = layerTexture('rg8unorm', b.w, b.h, b.data, 2, 'moon', GPUTextureUsage.COPY_SRC); // the reach and the tops in tiles (wasm/src/moon.c)
      else if (b.kind === BUF.REACH_PAGES) reachPagesAt = b;
      else if (b.kind === BUF.SKY_MAP) (stage.band ? (skyBands[stage.band.k] ||= { rows: stage.band.rows, parts: [] }).parts : skyParts).push(layerPart('rg8unorm', { ...b, data: widen(b.data) }, 2, 'sky')); // (in r, two channels as the maps it joins: pair_map)
      else if (b.kind === BUF.MAP_PAGES && stage.band) (skyBands[stage.band.k] ||= { rows: stage.band.rows, parts: [] }).pages = { w: b.w, h: b.h, data: new Uint16Array(b.data) };
      else if (b.kind === BUF.SKYGLOW) skyGlow = dataTexture('rgba16float', b.w, b.data.byteLength / (b.w * 8), b.data, b.w * 8); // (the rows under it: the shares that burn by a switch, skyglow.h)
      else if (b.kind === BUF.WET_LIGHTS) { // the lights the wet tracks mirror, and room after them for the frame's own (WGSL.WET_BIN makes them)
        wet.n = b.data.byteLength / 64 + WGSL.WET_MADE;
        wet.lights = device.createBuffer({ size: wet.n * 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(wet.lights, 0, b.data);
      }
      else if (b.kind === BUF.MAP_PAGES) {
        const from = new Uint16Array(b.data), into = mapPagesData ||= { w: b.w, h: b.h, data: new Uint16Array(from.length) }, c0 = stage.mode === MAPS.SKY ? 2 : 0, c1 = stage.mode === MAPS.MOON ? 2 : 4;
        for (let e = 0; e < from.length; e += 4) for (let c = c0; c < c1; c++) into.data[e + c] = from[e + c];
      }
      else if (b.kind === BUF.TREE_MODELS) treeModels = dataTexture('rgba32float', b.w, b.h, b.data, b.w * 16).createView(); // every tree model, two texels per vertex
      else if (b.kind === BUF.LAMP_HEADS) { // every street lamp's head and every floodlight's and work lamp's (struct lamp_head, 40 bytes): the game picks those nearest the car to light its cabin
        const at = hs.hs_alloc(b.data.byteLength);
        new Uint8Array(hs.memory.buffer, at, b.data.byteLength).set(new Uint8Array(b.data.buffer || b.data, b.data.byteOffset || 0, b.data.byteLength));
        if (hs.hs_lamp_heads(at, b.data.byteLength / 40)) throw new Error('the street lamps: bad list');
      }
      else if (b.kind === BUF.SMOKE) { // every stove burning: the shader draws their plumes, the game orders them and moves those the air stirs
        const n = b.data.byteLength / 32, at = hs.hs_alloc(b.data.byteLength);
        new Uint8Array(hs.memory.buffer, at, b.data.byteLength).set(new Uint8Array(b.data.buffer || b.data, b.data.byteOffset || 0, b.data.byteLength));
        if (hs.hs_smoke(at, n)) throw new Error('out of memory for the smoke');
        const DST = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
        smoke.order = device.createBuffer({ size: n * 4, usage: DST });
        smoke.puffs = device.createBuffer({ size: SMOKE.SLOTS * SMOKE.PUFFS * SMOKE.PUFF, usage: DST });
        smoke.steam = device.createBuffer({ size: SMOKE.STEAM * SMOKE.PUFF, usage: DST });
        smoke.group = device.createBindGroup({ layout: groupLayouts.smoke, entries: [{ binding: 0, resource: { buffer: buffers[stage.first + i] = gpuBuffer(b.data, GPUBufferUsage.STORAGE) } },
          { binding: 1, resource: { buffer: smoke.order } }, { binding: 2, resource: { buffer: smoke.puffs } }, { binding: 3, resource: { buffer: smoke.steam } }] });
        smoke.frame = hs.hs_smoke_frame() >> 2; smoke.count = n * SMOKE.PUFFS;
      }
      else if (b.kind === BUF.FOREST) { // every tree: the shaders read them by index, and the game lists each frame which to draw
        const trees = { buffer: gpuBuffer(b.data, GPUBufferUsage.STORAGE) }, n = b.data.byteLength / 32, at = hs.hs_alloc(b.data.byteLength);
        new Uint8Array(hs.memory.buffer, at, b.data.byteLength).set(new Uint8Array(b.data));
        if (hs.hs_forest(at, n, stage.first + i)) throw new Error('out of memory for the forest');
        buffers[stage.first + i] = device.createBuffer({ size: Math.max(4, n * FOREST_WORDS * 4), usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST }); // the list
        groups.forest = device.createBindGroup({ layout: groupLayouts.forest, entries: [{ binding: 0, resource: treeModels }, { binding: 1, resource: trees }] });
        const imp = bakeImpostors(treeModels, TREE.VARIANTS, TREE.LEVELS);
        const views = device.createTexture({ size: [WGSL.IMP_VIEWS * 3, WGSL.IMP_VIEWS], format: 'rgba32float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
        device.queue.writeTexture({ texture: views }, WGSL.impViews(), { bytesPerRow: WGSL.IMP_VIEWS * 3 * 16 }, [WGSL.IMP_VIEWS * 3, WGSL.IMP_VIEWS]);
        groups.impostor = device.createBindGroup({ layout: groupLayouts.impostor, entries: [{ binding: 0, resource: treeModels }, { binding: 1, resource: trees },
          { binding: 2, resource: imp.col }, { binding: 3, resource: imp.nrm }, { binding: 4, resource: device.createSampler({ minFilter: 'linear', magFilter: 'linear', mipmapFilter: 'linear' }) },
          { binding: 5, resource: views.createView() }] });
        shadow.forest = device.createBindGroup({ layout: shadow.forestLayout, entries: [{ binding: 0, resource: treeModels }, { binding: 1, resource: arrayView(groundTex) },
          { binding: 2, resource: device.createSampler({ minFilter: 'linear', magFilter: 'linear' }) }, { binding: 3, resource: trees },
          { binding: 4, resource: tileLayers.createView() }, { binding: 5, resource: arrayView(outerTex) }] });
        forestList = buffers[stage.first + i]; forestAt = hs.hs_forest_list();
      }
      else if (b.kind === BUF.SIGN_ATLAS) groups.sign = device.createBindGroup({ layout: groupLayouts.road, entries: [
        { binding: 0, resource: arrayView(arrayTexture([{ width: b.w, height: b.h, data: new Uint8Array(b.data) }], 'rgba8unorm-srgb')) }, { binding: 1, resource: sampler({ maxAnisotropy: 8 }) }] });
      else if (b.kind === BUF.LABEL_ATLAS) groups.label = device.createBindGroup({ layout: groupLayouts.road, entries: [
        { binding: 0, resource: arrayView(arrayTexture([{ width: b.w, height: b.h, data: new Uint8Array(b.data) }], 'rgba8unorm-srgb')) }, { binding: 1, resource: sampler({ maxAnisotropy: 8 }) }] });
      else if (b.kind === BUF.TRAIL_PRINTS) groups.trail = device.createBindGroup({ layout: groupLayouts.road, entries: [
        { binding: 0, resource: arrayView(arrayTexture([trailTexture(new Float32Array(b.data))], 'rgba8unorm-srgb')) }, { binding: 1, resource: sampler({ addressModeV: 'repeat', maxAnisotropy: 8 }) }] });
      else if (b.kind === BUF.BANK_VERTEX) { // the verge banks stay editable: the car bursts through them
        const at = hs.hs_alloc(b.data.byteLength);
        new Uint8Array(hs.memory.buffer, at, b.data.byteLength).set(new Uint8Array(b.data));
        snow.bankAt = at; snow.bankCount = b.data.byteLength / 20;
        snow.bank = buffers[stage.first + i] = gpuBuffer(b.data, GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST);
      } else if (b.kind === BUF.BANK_INDEX) {
        const at = hs.hs_alloc(b.data.byteLength);
        new Uint8Array(hs.memory.buffer, at, b.data.byteLength).set(new Uint8Array(b.data));
        if (hs.hs_snow_banks(snow.bankAt, snow.bankCount, at, b.data.byteLength / 4)) throw new Error('out of memory for the snowbanks');
        buffers[stage.first + i] = gpuBuffer(b.data, GPUBufferUsage.INDEX);
      } else buffers[stage.first + i] = gpuBuffer(b.data, b.kind === BUF.INDEX ? GPUBufferUsage.INDEX : GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE);
    });
    const n = stage.items.byteLength / ITEM_BYTES, at = n ? hs.hs_items(n, stage.unit || 0) : 1; // drawn in one build's order
    if (!at) throw new Error('too many draw items');
    new Uint8Array(hs.memory.buffer, at, n * ITEM_BYTES).set(new Uint8Array(stage.items));
    { // the lights' streaks: which buffer holds them, and how many (struct gfx_item: pipe, vbuf, ibuf, inst, .., inst_count, first_inst)
      const it = new Uint32Array(stage.items);
      for (let k = 0; k < n; k++) if (it[k * 16] === PIPE.STREAK) { sea.lights = it[k * 16 + 3]; sea.nlights = Math.max(sea.nlights || 0, it[k * 16 + 8] + it[k * 16 + 7]); }
    }
    const key = 'stage' + stage.stage + (stage.mode === MAPS.MOON ? 'moon' : stage.mode === MAPS.SKY ? 'sky' : '');
    marks[key] = (marks[key] || 0) + stage.ms;
  };
  waiting.splice(0).forEach(takeStage);
  take = takeStage;
  mark('caught'); // the stages that had come are in: the rest are taken as they come
  await built;
  mark('built');
  { // the town-wide maps and their coarser levels (pyramid): the moon's field and the sky the ground sees, which share a
    // page table; the moon's reach; the lamp map; its tint
    const ground = [world.nx, world.nz].map(n => Math.ceil(Math.fround(n * world.step) * 0.25)), lamp = [world.lampW, world.lampH];
    joinBands(skyBands, skyParts, mapPagesData ||= skyBands[0] && { ...skyBands[0].pages, data: new Uint16Array(skyBands[0].pages.data.length) }, 2);
    const maps = pyramid(mapPagesData, ground.map(n => n * 4), [{ chan: 0, kind: 0 }, { chan: 2, kind: 0 }]);
    mapPages = maps.texture;
    moonField = joinLayers('r16float', moonParts, 'moon', Math.ceil(maps.maps[0].slots / 256));
    skyMap = joinLayers('rg8unorm', skyParts, 'sky', Math.ceil(maps.maps[1].slots / 256));
    pyramidFill(moonField, maps, 0, 0, 0);
    pyramidFill(skyMap, maps, 1, 2, 0);
    const reach = pyramid(reachPagesAt, ground, [{ chan: 0, kind: 1 }]);
    moonReach = roomFor(moonReach, reach.maps[0].slots, 'moon');
    pyramidFill(moonReach, reach, 0, 0, 1);
    const lampP = pyramid(lampPagesAt, lamp, [{ chan: 0, kind: 2 }]);
    lampPages = lampP.texture;
    lampTex = roomFor(lampTex, lampP.maps[0].slots, 'lamp');
    pyramidFill(lampTex, lampP, 0, 0, 2, 255);
    const tint = pyramid(tintPagesAt, lamp.map(n => Math.ceil(n / 2)), [{ chan: 0, kind: 1 }]);
    tintPages = tint.texture;
    tintTex = roomFor(tintTex, tint.maps[0].slots, 'tint');
    pyramidFill(tintTex, tint, 0, 0, 1);
    pyramidTables = { map: maps.levels, reach: reach.levels, lamp: lampP.levels, tint: tint.levels };
  }
  { // the sea map in pages (wasm/include/harstad/sea.h): its first level's slots as the build baked them, and from them
    // the levels above it and the bound map (WGSL.SEA_MAKE), what the water near each texel can be at most, level by
    // level, for cutting the streaks to their water. Each level's slots are drawn into layers laid out as the atlas's,
    // then copied into it, so that no pass both reads a texture and draws into it.
    const T = new Uint32Array(seaPages.data.buffer || seaPages.data, seaPages.data.byteOffset || 0, seaPages.data.byteLength / 4);
    const rect = new Float32Array(mem(), sea.at, 8).slice(), W = Math.round(rect[2] / 4), H = Math.round(rect[3] / 4), tw = seaPages.w;
    const side = 62, slot = 64, layer = seaMap.w, across = layer / slot, per = across * across, levels = mips(W, H); // tiles.h
    const seaSlots = [], boundSlots = [];
    let nSea = 1, nBound = 1;
    for (let l = 0, row = 0; l < levels; l++) { // each level's slots of either map: the slot, its tile's column and row
      const w = Math.ceil(Math.max(1, W >> l) / side), h = Math.ceil(Math.max(1, H >> l) / side), a = [], b = [];
      for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) {
        const e = ((row + j) * tw + i) * 4, k = T[e] & 0xffff;
        if (k !== 0xffff) { a.push(k, i, j, 0); nSea = Math.max(nSea, k + 1); }
        if (T[e + 2] !== 0xffff) { b.push(T[e + 2], i, j, 0); nBound = Math.max(nBound, T[e + 2] + 1); }
      }
      seaSlots.push(a); boundSlots.push(b); row += h;
    }
    const TU = GPUTextureUsage, BU = GPUBufferUsage;
    const atlas = (format, n, label) => device.createTexture({ label, size: [layer, layer, Math.ceil(n / per)], format, usage: TU.TEXTURE_BINDING | TU.COPY_DST });
    const seaTex = atlas('rgba8unorm', nSea, 'sea'), bound = atlas('rg8unorm', nBound, 'sea bound');
    device.queue.writeTexture({ texture: seaTex }, seaMap.data, { bytesPerRow: layer * 4, rowsPerImage: layer }, [layer, layer, seaMap.data.byteLength / (layer * layer * 4)]);
    const pagesTex = dataTexture('rgba32uint', tw, seaPages.h, seaPages.data, tw * 16);
    const pipe = (kind, format) => { const module = device.createShaderModule({ code: WGSL.SEA_MAKE(kind) }); return device.createRenderPipeline({ layout: 'auto', vertex: { module }, fragment: { module, targets: [{ format }] } }); };
    const make = (p, format, l, slots, src, dst, more) => {
      const n = slots.length / 4;
      if (!n) return;
      const base = Math.floor(slots[0] / per), top = Math.floor(slots[(n - 1) * 4] / per);
      const scratch = device.createTexture({ size: [layer, layer, top - base + 1], format, usage: TU.RENDER_ATTACHMENT | TU.COPY_SRC });
      const group = device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: [{ binding: 0, resource: src.createView({ dimension: '2d-array' }) },
        { binding: 1, resource: pagesTex.createView() }, { binding: 3, resource: { buffer: gpuBuffer(new Uint32Array([W, H, l, 0]), BU.UNIFORM) } },
        { binding: 4, resource: { buffer: gpuBuffer(new Uint32Array(slots), BU.STORAGE) } }, ...more] });
      const enc = device.createCommandEncoder();
      for (let z = base, i = 0; z <= top; z++) { // a pass a layer, a quad a slot
        let j = i;
        while (j < n && Math.floor(slots[j * 4] / per) === z) j++;
        const pass = enc.beginRenderPass({ colorAttachments: [{ view: scratch.createView({ dimension: '2d', baseArrayLayer: z - base, arrayLayerCount: 1 }), loadOp: 'clear', storeOp: 'store' }] });
        pass.setPipeline(p); pass.setBindGroup(0, group); pass.draw(6, j - i, 0, i); pass.end();
        i = j;
      }
      for (let i = 0; i < n; i++) {
        const k = slots[i * 4], x = k % across * slot, y = Math.floor(k / across) % across * slot, z = Math.floor(k / per);
        enc.copyTextureToTexture({ texture: scratch, origin: [x, y, z - base] }, { texture: dst, origin: [x, y, z] }, [slot, slot, 1]);
      }
      device.queue.submit([enc.finish()]);
      scratch.destroy();
    };
    const mip = pipe('mip', 'rgba8unorm'), first = pipe('bound0', 'rg8unorm'), next = pipe('bound', 'rg8unorm');
    const linear = [{ binding: 2, resource: device.createSampler({ minFilter: 'linear', magFilter: 'linear' }) }];
    for (let l = 1; l < levels; l++) make(mip, 'rgba8unorm', l, seaSlots[l], seaTex, seaTex, linear);
    make(first, 'rg8unorm', 0, boundSlots[0], seaTex, bound, [{ binding: 5, resource: sea.horizon.createView({ dimension: '2d', baseMipLevel: 0, mipLevelCount: 1, baseArrayLayer: 0, arrayLayerCount: 1 }) },
      { binding: 6, resource: { buffer: gpuBuffer(rect, BU.UNIFORM) } }]);
    for (let l = 1; l < levels; l++) make(next, 'rg8unorm', l, boundSlots[l], bound, bound, []);
    sea.map = seaTex.createView({ dimension: '2d-array' });
    sea.bound = bound.createView({ dimension: '2d-array' });
    seaPagesTable = { w: tw, h: seaPages.h, data: new Uint32Array(seaPages.data.buffer || seaPages.data, seaPages.data.byteOffset || 0, tw * seaPages.h * 4) };
  }
  lampGlow = lampGlowFrom(lampTex, lampPages, world.lampW, world.lampH, tintPagesAt.data, tintPagesAt.w, tintPagesAt.h);
  // the maps of two channels in one texture, and the lamp map's with the haze's (WGSL pair_map, lamp_map), so that a
  // shader reads no more than sixteen textures
  const pairs = joinMaps([tintTex, moonReach, skyMap], 'pairs'), lamps = joinMaps([lampTex, lampGlow], 'lamps');
  wet.tops = pairs.texture;
  const tables = tableAtlas({ ...pyramidTables, tiles: [groundTables.tiles], back: [groundTables.back], sea: [seaPagesTable] },
    { reachFirst: pairs.firsts[1], skyFirst: pairs.firsts[2], glowFirst: lamps.firsts[1] });
  sea.entries = [{ binding: 0, resource: { buffer: frameBuffer, offset: FRAME.SEA, size: SEA_BYTES } },
    { binding: 1, resource: sea.map },
    { binding: 2, resource: sea.ripples.textures[0].createView() }, { binding: 3, resource: sea.ripples.textures[1].createView() },
    { binding: 4, resource: sea.floes.createView({ dimension: '2d' }) }, { binding: 5, resource: sea.horizon.createView({ dimension: '2d-array' }) },
    { binding: 7, resource: sea.sky.textures[0].createView() }, { binding: 8, resource: sampler() }, { binding: 9, resource: sampler({ ...repeat, maxAnisotropy: 4 }) },
    { binding: 10, resource: sampler({ addressModeU: 'repeat' }) }, { binding: 11, resource: sea.bound }];
  { // the water's surface under each pair of pixels, for the streaks (WGSL.SEA_SURFACE): worked out in the frame's compute pass
    const module = device.createShaderModule({ code: WGSL.PRELUDE + WGSL.SEA_SURFACE });
    sea.surfPipe = device.createComputePipeline({ layout: 'auto', compute: { module } });
    sea.surfFrame = device.createBindGroup({ layout: sea.surfPipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: frameBuffer, offset: 0, size: FRAME.UNIFORMS } }, { binding: 16, resource: tables.createView() }] });
    sea.surfSea = device.createBindGroup({ layout: sea.surfPipe.getBindGroupLayout(1), entries: sea.entries.filter(e => [0, 1, 2, 3, 5, 8, 9].includes(e.binding)) });
  }
  { // each light's strip, cut once a frame for all its vertices (WGSL.STREAK_FIT)
    const n = sea.nlights || 0, module = device.createShaderModule({ code: WGSL.PRELUDE + WGSL.STREAK_FIT });
    sea.fits = device.createBuffer({ size: Math.max(1, n) * WGSL.STREAK_FIT_WORDS * 4, usage: GPUBufferUsage.STORAGE });
    sea.fitPipe = device.createComputePipeline({ layout: 'auto', compute: { module } });
    sea.fitFrame = device.createBindGroup({ layout: sea.fitPipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: frameBuffer, offset: 0, size: FRAME.UNIFORMS } }, { binding: 16, resource: tables.createView() }] });
    sea.fitSea = device.createBindGroup({ layout: sea.fitPipe.getBindGroupLayout(1), entries: sea.entries.filter(e => [0, 11].includes(e.binding)) });
    sea.fitOut = n ? device.createBindGroup({ layout: sea.fitPipe.getBindGroupLayout(2), entries: [{ binding: 0, resource: { buffer: buffers[sea.lights], size: n * 36 } }, { binding: 1, resource: { buffer: sea.fits } }] }) : null;
    sea.fitGroups = Math.ceil(n / 64);
    // and the lights of the ships, which move: sent every frame into a buffer of their own, cut the same way
    const [at, count, cap] = new Uint32Array(mem(), hs.hs_ship_streaks(), 3);
    Object.assign(sea, { movingAt: at, movingCount: count >> 2, movingCap: cap,
      moving: device.createBuffer({ size: cap * 36, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }),
      movingFits: device.createBuffer({ size: cap * WGSL.STREAK_FIT_WORDS * 4, usage: GPUBufferUsage.STORAGE }) });
    sea.movingOut = device.createBindGroup({ layout: sea.fitPipe.getBindGroupLayout(2), entries: [{ binding: 0, resource: { buffer: sea.moving } }, { binding: 1, resource: { buffer: sea.movingFits } }] });
  }
  { // the wet tracks' lights, binned into the frame's tiles each frame (WGSL.WET_BIN)
    const module = device.createShaderModule({ code: WGSL.PRELUDE + WGSL.WET_BIN });
    const C = GPUShaderStage.COMPUTE;
    wet.layout = [device.createBindGroupLayout({ entries: [{ binding: 0, visibility: C, buffer: {} }, { binding: 2, visibility: C, texture: { viewDimension: '2d-array' } }, { binding: 4, visibility: C, sampler: {} },
      { binding: 16, visibility: C, texture: { sampleType: 'uint' } },
      { binding: 23, visibility: C, buffer: { type: 'read-only-storage' } }] }), // the light list's cars, whose lamps it makes
      device.createBindGroupLayout({ entries: [...[0, 1, 2, 3].map(binding => ({ binding, visibility: C, buffer: { type: 'storage' } })), { binding: 4, visibility: C, texture: { viewDimension: '2d-array' } }] })]; // the reach and the tops in tiles (g, what stands how high: whether a light is seen from the road)
    wet.vis = device.createBuffer({ size: wet.n * WGSL.WET_STATIONS * 4, usage: GPUBufferUsage.STORAGE }); // each light's stations: how open the way to the road is
    const layout = device.createPipelineLayout({ bindGroupLayouts: wet.layout });
    wet.culler = device.createComputePipeline({ layout, compute: { module, entryPoint: 'cull' } });
    wet.counter = device.createComputePipeline({ layout, compute: { module, entryPoint: 'count' } });
    wet.least = device.createComputePipeline({ layout, compute: { module, entryPoint: 'least' } });
    // how many lights are worth laying, as the next passes' workgroup count (1 by 1 more), and which
    wet.live = device.createBuffer({ size: (4 + wet.n) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    wet.args = device.createBuffer({ size: 12, usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST }); // the count copied out of it, as no pass may both write and dispatch from one buffer
    wet.none = new Uint32Array([0, 1, 1, 0]);
    wet.pipe = device.createComputePipeline({ layout, compute: { module, entryPoint: 'main' } });
    wet.frame = device.createBindGroup({ layout: wet.layout[0], entries: [{ binding: 0, resource: { buffer: frameBuffer, offset: 0, size: FRAME.UNIFORMS } },
      { binding: 2, resource: arrayView(groundTex) }, { binding: 4, resource: device.createSampler({ minFilter: 'linear', magFilter: 'linear' }) },
      { binding: 16, resource: tables.createView() }, { binding: 23, resource: { buffer: shine.lights } }] });
    wetBind();
  }
  { // the wake follows what is wet: the sea map's first level, its slots and its page table
    const copy = d => { const at = hs.hs_alloc(d.byteLength); if (at) new Uint8Array(hs.memory.buffer, at, d.byteLength).set(new Uint8Array(d.buffer || d, d.byteOffset || 0, d.byteLength)); return at; };
    const atlas = copy(seaMap.data), pages = copy(seaPages.data);
    if (!atlas || !pages) throw new Error('out of memory for the sea map');
    hs.hs_sea_map(atlas, pages);
  }




  if (!cloudShade.map) cloudShadeInit(moonField);
  { // the light grid's words, and after them the low lights' switches (WGSL.LAMP_SW0), in one buffer
    const sw = lampSwitch || new Uint32Array(4), grid = shine.grid;
    shine.grid = device.createBuffer({ size: WGSL.LAMP_SW0 * 4 + sw.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(shine.grid, WGSL.LAMP_SW0 * 4, sw);
    grid.destroy();
    // the room switches in its table (wasm/include/harstad/lights.h), each with its cap, for WGSL.SW_NOW: a switch's
    // words are its head (its kind | its rooms, n, << 8 | LIGHTS_SW_ONE_HOME at bit 20), then for rooms its cap, its
    // weights, and the one home and n rooms or n homes and rooms; a shop's is its head alone
    const list = [];
    for (let at = 1, first = sw[1], end = sw[2]; at < end;) {
      const head = sw[first + at], kind = head & 255, n = (head >> 8) & 255;
      if (kind === 1) { list.push(first + at, sw[first + at + 1]); at += 3 + ((head & (1 << 20)) ? 1 + n : 2 * n); } else at += 1;
    }
    swNow.n = list.length / 2;
    if (swNow.n) {
      const C = GPUShaderStage.COMPUTE, module = device.createShaderModule({ code: WGSL.PRELUDE + WGSL.SW_NOW });
      const layout = device.createPipelineLayout({ bindGroupLayouts: [device.createBindGroupLayout({ entries: [{ binding: 0, visibility: C, buffer: {} }] }),
        device.createBindGroupLayout({ entries: [{ binding: 0, visibility: C, buffer: { type: 'storage' } }, { binding: 1, visibility: C, buffer: { type: 'read-only-storage' } }] })] });
      swNow.pipe = device.createComputePipeline({ layout, compute: { module, entryPoint: 'main' } });
      const listBuf = device.createBuffer({ size: list.length * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(listBuf, 0, new Uint32Array(list));
      swNow.groups = [device.createBindGroup({ layout: swNow.pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: frameBuffer, offset: 0, size: FRAME.UNIFORMS } }] }),
        device.createBindGroup({ layout: swNow.pipe.getBindGroupLayout(1), entries: [{ binding: 0, resource: { buffer: shine.grid } }, { binding: 1, resource: { buffer: listBuf } }] })];
    }
  }
  const frameEntries = [
    { binding: 0, resource: { buffer: frameBuffer, offset: 0, size: FRAME.UNIFORMS } }, { binding: 1, resource: arrayView(lamps.texture) }, { binding: 2, resource: arrayView(groundTex) },
    { binding: 3, resource: detail.createView({ dimension: '2d-array' }) },
    { binding: 4, resource: device.createSampler({ minFilter: 'linear', magFilter: 'linear' }) }, { binding: 5, resource: sampler(repeat) },
    { binding: 6, resource: shadow.view }, { binding: 7, resource: device.createSampler({ compare: 'less', minFilter: 'linear', magFilter: 'linear' }) },
    { binding: 8, resource: shadow.beams.createView({ dimension: '2d-array' }) },
    { binding: 10, resource: arrayView(pairs.texture) }, { binding: 11, resource: { buffer: skyLow.rows } },
    { binding: 16, resource: tables.createView() }, { binding: 17, resource: skyGlow.createView() },
    { binding: 23, resource: { buffer: shine.lights } }, { binding: 24, resource: { buffer: shine.grid } },
    { binding: 25, resource: cloudShade.map.createView({ dimension: '2d-array' }) },
  ];
  frameGroup = device.createBindGroup({ layout: frameLayout, entries: frameEntries });
  await device.queue.onSubmittedWorkDone();
  mark('uploaded');
  progress(1);
  marks.geometryBytes = bytes;
  marks.wasmMemory = hs.memory.buffer.byteLength;
  resize(...(wanted || [width, height]));
  if ('count' in params) countUploads(); // ?count: what is sent to the GPU each frame, by what it is for, in report()
  requestAnimationFrame(tick);
  await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
  mark('ready');
  marks.textures = texturesMB();
  post({ ready: true, marks, ...roadNames(world) });
}

// The falling snow: room for SNOW_MOST times the flakes each layer was tuned with, a layer after another. As many are
// drawn of each as the weather's snowfall has flakes in the air (struct snow_frame's eye[3], of the tuned: snow_view), the
// first so many of a layer's, which lie at random through its box: as the snow thickens more of them fall, none moves
function snowfall() {
  const layers = WGSL.SNOW_LAYERS.map(l => ({ ...l, count: l.count * WGSL.SNOW_MOST }));
  snowBuffer = gpuBuffer(snowFlakes(layers), GPUBufferUsage.VERTEX);
  let at = 0;
  snowLayers = WGSL.SNOW_LAYERS.map(l => { const first = at; at += l.count * WGSL.SNOW_MOST; return [first, l.count]; });
}
function snowDraw(pass, f) {
  const k = Math.min(WGSL.SNOW_MOST, memF[((f + FRAME.SNOW) >> 2) + 7]);
  pass.setPipeline(pipes.snow); pass.setVertexBuffer(0, snowBuffer); pass.setIndexBuffer(quadIndex, 'uint16');
  for (const [first, count] of snowLayers) { const n = Math.round(count * k); if (n) pass.drawIndexed(6, n, 0, 0, first); }
}

// Places to start from by name (?at=): the walk up Gangsåstoppen (tools/gangsas.py), at its car park at the end of
// Toppenveien facing the steps, and on Ramnfløya, its top, on the path up facing the shelter. x, z, heading
const PLACES = { gangsas: [876.5, 109.5, 2.74], gangsastoppen: [803.5, -630.0, 2.79] };

function applyParams(p, world) {
  const num = k => +(p[k] ?? 0) || 0, at = PLACES[p.at] || [];
  hs.hs_car_place(num('x') || at[0] || world.spawn[0], num('z') || at[1] || world.spawn[1], 'h' in p ? num('h') : at[2] ?? world.spawn[2]);
  hs.hs_set(SET.CAM, num('cam'));
  if (num('orbit')) hs.hs_set(SET.ORBIT, num('orbit'));
  // ?round=3.14&rise=0.1: the orbit held still, that far round from the car's nose (rad) and that share of its radius up
  if ('round' in p || 'rise' in p) { hs.hs_set(SET.ORBIT_AT, num('round')); hs.hs_set(SET.ORBIT_RISE, 'rise' in p ? num('rise') : 0.31); }
  // &on=1.36,0.7,-1.8: and round that point on the car (its space: x left, y up, z forward), to look at a part of it close
  if ('on' in p) String(p.on).split(',').map(Number).forEach((v, k) => hs.hs_set(SET.ORBIT_ON_X + k, v || 0));
  if ('top' in p) hs.hs_set(SET.TOP, num('top'));
  if ('weather' in p) hs.hs_set(SET.WEATHER, num('weather')); // 0 clear, 1 light snow, 2 heavy snow (the desk's)
  if ('snowfall' in p) hs.hs_set(SET.SNOWFALL, num('snowfall')); // the weather's snowfall so many times over (winter.h)
  if ('fallen' in p) hs.hs_set(SET.FALLEN, num('fallen'));       // cm of it on every road before the clock starts
  hs.hs_set(SET.AUTO_DRIVE, num('drive'));
  hs.hs_set(SET.AUTO_FLY, num('fly'));
  hs.hs_set(SET.GUSTS, p.gusts === '0' ? 0 : 1);
  hs.hs_set(SET.HAZE, 'haze' in p ? +p.haze : 1);
  hs.hs_set(SET.TRAFFIC, p.traffic === '0' ? 0 : 1);
  if (num('clock')) hs.hs_set(SET.TIME, num('clock'));
  // ?day=24&hour=12: the town's clock (wasm/include/harstad/clock.h) on that day of December and at that hour; ?hours=60: run 60 times as fast
  if ('day' in p) hs.hs_set(SET.DAY, num('day'));
  if ('hour' in p) hs.hs_set(SET.HOUR, num('hour'));
  if ('hours' in p) hs.hs_set(SET.CLOCK_RATE, num('hours'));
  if ('meetat' in p) hs.hs_set(SET.MEET_AT, num('meetat')); // ?meet=12&meetat=300: that far ahead of it
  if ('meet' in p) hs.hs_set(SET.MEET, num('meet')); // ?meet=12: beside the way of the traffic's car 12, facing it as it comes
  if ('cockpit' in p) hs.hs_set(SET.COCKPIT, num('cockpit')); // ?cockpit=0.75: the flight cockpit held that far open (0.5 made over, 1 the windows open)
  if ('wing' in p) hs.hs_set(SET.WING, num('wing')); // ?wing=0.4: the doors held that far out into wings (wing.h), to look at them
  if ('aloft' in p) hs.hs_set(SET.ALOFT, num('aloft') || 120); // ?aloft=150: the car let go 150 m up, flight armed: on an approach, say
  if ('cloud' in p) hs.hs_set(SET.CLOUD, num('cloud')); // ?cloud=3: the car let go in the heart of cloud 3 (clouds.h), flight armed
  if ('land' in p) hs.hs_set(SET.AUTO_LAND, num('land'));     // ?fly=8&land=20: flight switched off again 20 s in
  if ('look' in p) { const [x, y] = p.look.split(',').map(Number); hs.hs_set(SET.LOOK_X, x || 0); hs.hs_set(SET.LOOK_Y, y || 0); } // ?look=0.3,-1: the driver looking that way
  if ('cabinlight' in p) hs.hs_set(SET.CABIN_LIGHT, num('cabinlight')); // ?cabinlight=1: the cabin lit evenly, to judge its shapes by
  if ('truck' in p) hs.hs_set(SET.TRUCK, num('truck') || 70); // ?truck=70: 70 m ahead of the Christmas truck, beside its way, facing it
  // ?sky=heli: beside UNN Harstad's helipad as the air ambulance comes in (lift: as it leaves), ?sky=jet the 737 coming round onto the
  // final for Evenes south of the town, dash the Dash 8 from Andenes over the town, depart the one off to Tromsø, over a jet at cruise, sleigh Father Christmas's first pass of the night;
  // the clock set to just before. &watch: the camera follows it; &fov=4 narrows the camera's field to see it close
  if ('sky' in p) { const t = hs.hs_sky(SKY[p.sky] ?? num('sky')); if (t >= 0) clock = t; }
  if ('watch' in p) hs.hs_set(SET.WATCH, SKY[p.watch] ?? (num('watch') || SKY[p.sky] || num('sky')));
  if ('fov' in p) hs.hs_set(SET.FOV, num('fov'));
  // ?plough=60, ?plough2=60, ?gritter=60: as far ahead of the main road's plough, the town's or the gritter (behind it if negative)
  if ('plough' in p) hs.hs_set(SET.PLOUGH, num('plough') || 60);
  if ('plough2' in p) hs.hs_set(SET.PLOUGH2, num('plough2') || 60);
  if ('gritter' in p) hs.hs_set(SET.GRITTER, num('gritter') || 60);
  // ?event=fire&at=60,760: something happens there now (police, ambulance, stop, crash, fire, chimney, alarm), and the car is put
  // on the road 45 m short of it, facing it (&back=80 further, &back=0 where it is); ?events=0: nothing happens by itself;
  // ?unit=engine: beside an emergency vehicle (p1, p2, p3, ambulance, engine, ladder, tow1, tow2), at its bay or on its way
  if ('events' in p) hs.hs_set(SET.EVENTS, num('events'));
  if ('back' in p) hs.hs_set(SET.EVENT_BACK, num('back'));
  // ?crash=1300,650: two of the traffic's cars meet on the road nearest there, and the town deals with it (game_crash)
  if ('crash' in p) { const [x, z] = String(p.crash).split(',').map(Number); hs.hs_set(SET.CRASH_X, x || 0); hs.hs_set(SET.CRASH, z || 0); }
  if ('event' in p) {
    if (p.at) { const [x, z] = String(p.at).split(',').map(Number); hs.hs_set(SET.EVENT_X, x); hs.hs_set(SET.EVENT_Z, z); }
    hs.hs_set(SET.EVENT, EVENTS[p.event] ?? num('event'));
  }
  if ('unit' in p) hs.hs_set(SET.UNIT, UNITS[p.unit] ?? num('unit'));
  if ('follow' in p) hs.hs_set(SET.FOLLOW, UNITS[p.follow] ?? num('follow')); // ?follow=p1: the camera behind it wherever it goes
  if ('boost' in p) { const [a = 0, b = 1e9] = String(p.boost).split(',').map(Number); hs.hs_set(SET.BOOST_FROM, a || 0); hs.hs_set(SET.BOOST_TO, b); } // ?fly=8&boost=12,30: the boost switched on 12 s in and off at 30 (?boost=12: on from then)
  if ('glaze' in p) hs.hs_set(SET.GLAZE, p.glaze === '' ? 1 : num('glaze')); // ?glaze or ?glaze=0.5: the roads glazed over with ice, that far
  // ?lamps=brake,left,reverse: the car's lamps held lit to look at them (left, right, brake, reverse, fog)
  if (+p.intro > 0) hs.hs_set(SET.INTRO, +p.intro); // the opening zoom from over the whole map, so many seconds (src/main.js)
  if ('introat' in p) hs.hs_set(SET.INTRO_AT, num('introat')); // ?introat=0.4: held that share of the way down, for pictures
  if ('lamps' in p) hs.hs_set(SET.LAMPS, String(p.lamps).split(',').reduce((m, k) => m | ({ brake: 1, left: 2, right: 4, reverse: 8, fog: 16 }[k] || 0), 0));
}

/* ---- what the main thread needs from the world: street names and the minimap ---- */

const ROAD = 24; // sizeof(struct world_road): pt u32, npt u32, w f64, kind u8, flags u8, pad, name u32
const ROAD_KINDS = ['service', 'residential', 'footway', 'cycleway', 'trunk', 'track', 'unclassified', 'tertiary', 'secondary', 'primary',
  'pedestrian', 'trunk_link', 'primary_link', 'living_street']; // as tools/pack.py
function roadNames(world) {
  const v = new DataView(hs.memory.buffer), bytes = new Uint8Array(hs.memory.buffer), dec = new TextDecoder(), names = [], kinds = [];
  for (let i = 0; i < world.nroads; i++) {
    const at = world.strings + v.getUint32(world.roads + i * ROAD + 20, true);
    let end = at;
    while (bytes[end]) end++;
    names.push(dec.decode(bytes.subarray(at, end)));
    kinds.push(ROAD_KINDS[v.getUint8(world.roads + i * ROAD + 16)]);
  }
  return { names, kinds };
}

function postMinimap(world) {
  const MS = 0.4, v = new DataView(hs.memory.buffer), pts = new Float64Array(hs.memory.buffer, 0, hs.memory.buffer.byteLength >> 3);
  const c = new OffscreenCanvas(Math.ceil(world.w * MS), Math.ceil(world.d * MS)), mc = c.getContext('2d');
  mc.lineJoin = 'round';
  for (const pass of [0, 1]) for (let i = 0; i < world.nroads; i++) {
    const r = world.roads + i * ROAD, w = v.getFloat64(r + 8, true), kind = v.getUint8(r + 16), tunnel = v.getUint8(r + 17) & 1;
    const major = w >= 7, drivable = kind !== 2 && kind !== 3;
    if (major !== !!pass) continue;
    mc.strokeStyle = major ? 'rgba(255,255,255,.95)' : drivable ? 'rgba(255,255,255,.5)' : 'rgba(255,255,255,.18)';
    mc.lineWidth = Math.max(1.2, w * MS * (major ? 0.9 : 0.6));
    mc.setLineDash(tunnel ? [3, 4] : []);
    mc.lineCap = tunnel ? 'butt' : 'round';
    mc.beginPath();
    const p = (world.roadPts >> 3) + v.getUint32(r, true) * 2, n = v.getUint32(r + 4, true);
    for (let k = 0; k < n; k++) mc.lineTo((pts[p + k * 2] - world.bx) * MS, (pts[p + k * 2 + 1] - world.bz) * MS);
    mc.stroke();
  }
  const bitmap = c.transferToImageBitmap();
  post({ minimap: bitmap, scale: MS, x: world.bx, z: world.bz }, [bitmap]);
}

/* ---- the frame ---- */

// the bind groups that hold the wet tracks' tiles, made again when they are: the bin's, and the roads'
function wetBind() {
  if (!wet.pipe || !wet.tiles || !wet.lights) return;
  wet.bin = device.createBindGroup({ layout: wet.layout[1], entries: [{ binding: 0, resource: { buffer: wet.lights } }, { binding: 1, resource: { buffer: wet.tiles } }, { binding: 2, resource: { buffer: wet.live } },
    { binding: 3, resource: { buffer: wet.vis } }, { binding: 4, resource: arrayView(wet.tops) }] });
  groups.road = device.createBindGroup({ layout: groupLayouts.wetRoad, entries: [...wet.road, { binding: 2, resource: { buffer: wet.lights } }, { binding: 3, resource: { buffer: wet.tiles } },
    { binding: 4, resource: sea.sky.textures[0].createView() }, { binding: 5, resource: wet.skySampler }, { binding: 6, resource: { buffer: wet.vis } },
    { binding: 7, resource: { buffer: winterBuffer() } }] });
}

// The snow on the roads (wasm/include/harstad/winter.h): its cells, as the shaders read them, sent whole once and then
// a block of WINTER_BLOCK cells at a time as they change; and the windrows it has grown on the verge banks.
function winterBuffer() {
  if (winter.buffer) return winter.buffer;
  const t = new Uint32Array(hs.memory.buffer, hs.hs_winter(), 10);
  winter.at = t[0]; winter.words = t[1];
  winter.buffer = device.createBuffer({ size: Math.max(32, winter.words * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  if (winter.words) device.queue.writeBuffer(winter.buffer, 0, new Uint8Array(hs.memory.buffer), winter.at, winter.words * 4);
  return winter.buffer;
}
function winterSend() {
  if (!winter.buffer) return;
  const t = new Uint32Array(hs.memory.buffer, hs.hs_winter(), 10), m32 = mem32;
  const [at, , dirty, ndirty, bankDirty, nbankDirty, block, words, head] = t;
  const send = (bits, n, put) => { // each run of set bits once, and the bits cleared
    for (let i = 0; i < n; i++) {
      let word = m32[(bits >> 2) + i];
      if (!word) continue;
      m32[(bits >> 2) + i] = 0;
      for (let b = 0; b < 32;) {
        if (!(word >>> b & 1)) { b++; continue; }
        let e = b;
        while (e < 32 && word >>> e & 1) e++;
        put(i * 32 + b, e - b);
        b = e;
      }
    }
  };
  const ncells = (winter.words - head) / words;
  send(dirty, ndirty, (b, n) => {
    const c0 = b * block, c1 = Math.min(ncells, (b + n) * block);
    if (c1 > c0) device.queue.writeBuffer(winter.buffer, (head + c0 * words) * 4, mem8, at + (head + c0 * words) * 4, (c1 - c0) * words * 4);
  });
  if (snow.bank && bankDirty) send(bankDirty, nbankDirty, (b, n) => {
    const v0 = b * block * 6, v1 = Math.min(snow.bankCount, (b + n) * block * 6); // SNOW_BANK_VERTS a section, 20 bytes each
    if (v1 > v0) device.queue.writeBuffer(snow.bank, v0 * 20, mem8, snow.bankAt + v0 * 20, (v1 - v0) * 20);
  });
}
// The canvas: on an HDR screen (the page asks, params.hdr, by matchMedia('(dynamic-range: high)'); ?hdr=0 for none) a
// float canvas with extended tone mapping, which shows extended sRGB (sRGB-encoded, not linear), its highlights over white up to what the
// screen can give; where that is not to be had, the SDR canvas the browser prefers, as before. Everything is drawn in
// float (format) either way, and taken onto the canvas last (WGSL.PRESENT).
function configure() {
  const usage = GPUTextureUsage.RENDER_ATTACHMENT;
  hdr = false;
  if (hdrWanted) {
    try {
      context.configure({ device, format: 'rgba16float', colorSpace: 'srgb', toneMapping: { mode: 'extended' }, alphaMode: 'opaque', usage });
      const c = context.getConfiguration?.();
      hdr = !!(c && c.toneMapping && c.toneMapping.mode === 'extended');
    } catch { hdr = false; }
  }
  canvasFormat = hdr ? 'rgba16float' : navigator.gpu.getPreferredCanvasFormat();
  if (!hdr) context.configure({ device, format: canvasFormat, alphaMode: 'opaque', usage });
  present.pipe = null;
}
// The frame's picture, as the screen shows it (encode(tone())), in float; and the pass that takes it onto the canvas
const present = { pipe: null, grab: null, texture: null, group: null, pass: { colorAttachments: [{ view: null, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }] } };
function presentPipe(target) {
  const module = device.createShaderModule({ code: WGSL.PRESENT });
  return device.createRenderPipeline({ layout: 'auto', vertex: { module }, fragment: { module, entryPoint: hdr && target === canvasFormat ? 'hdr' : 'sdr', targets: [{ format: target }] } });
}
function presentTo(enc, view, grab) {
  const pipe = grab ? (present.grab ||= presentPipe('rgba8unorm')) : (present.pipe ||= presentPipe(canvasFormat));
  present.pass.colorAttachments[0].view = view;
  const p = enc.beginRenderPass(stamped(present.pass, 'present'));
  if (pipe.at !== present.view) { pipe.at = present.view; pipe.group = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: present.view }] }); } // (made again only when the picture is)
  p.setPipeline(pipe); p.setBindGroup(0, pipe.group); p.draw(3); p.end();
}
let wantedRatio = 0; // and the latest pixel ratio, for the same reason
let wanted = null; // the latest size asked for: a resize that comes while the game loads is the size it first draws at
function resize(width, height) {
  wanted = [width, height];
  if (!device) return;
  width = Math.max(1, width); height = Math.max(1, height); // a minimised window says 0, and a texture cannot be
  canvas.width = width; canvas.height = height;
  colourTarget?.destroy(); depthTarget?.destroy(); present.texture?.destroy();
  const cw = width, ch = height;
  present.texture = device.createTexture({ size: [cw, ch], format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST });
  present.view = present.texture.createView(); // the canvas's size; below, width and height are the frame's
  width = renderW = Math.max(1, Math.round(cw * SCALE)); height = renderH = Math.max(1, Math.round(ch * SCALE));
  // The depth, and with ?aa=4 the multisampled colour, are never stored (the colour is resolved onto the canvas, the depth
  // thrown away): where the browser can say so they live only in the GPU's tile memory and are never allocated
  const kept = GPUTextureUsage.RENDER_ATTACHMENT | (splitting ? 0 : GPUTextureUsage.TRANSIENT_ATTACHMENT || 0); // (?split: kept between the pass's cuts)
  colourTarget = SAMPLES > 1 ? device.createTexture({ size: [width, height], format, sampleCount: SAMPLES, usage: kept }) : null;
  depthTarget = device.createTexture({ size: [width, height], format: 'depth32float', sampleCount: SAMPLES, usage: kept });
  if (colourTarget) mainPass.colorAttachments[0].view = colourTarget.createView();
  else { delete mainPass.colorAttachments[0].resolveTarget; mainPass.colorAttachments[0].storeOp = 'store'; } // unsampled: straight onto the canvas
  if (up.easu) { // drawn small, then upscaled onto the canvas: EASU, then RCAS from EASU's output
    up.scene?.destroy(); up.wide?.destroy();
    const usage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
    up.scene = device.createTexture({ size: [width, height], format, usage: usage | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST });
    mainPass.colorAttachments[0][colourTarget ? 'resolveTarget' : 'view'] = up.scene.createView();
    const sx = width / cw, sy = height / ch;
    device.queue.writeBuffer(up.uniform, 0, new Float32Array([sx, sy, 0.5 * sx - 0.5, 0.5 * sy - 0.5, SHARP, 0, 0, 0]));
    const group = (pipe, texture) => device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: texture.createView() }, { binding: 1, resource: { buffer: up.uniform } }] });
    up.easuGroup = group(up.easu, up.scene);
    if (up.rcas) {
      up.wide = device.createTexture({ size: [cw, ch], format, usage });
      up.rcasGroup = group(up.rcas, up.wide);
      up.easuPass = { colorAttachments: [{ view: up.wide.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }] };
      up.rcasPass = { colorAttachments: [{ view: null, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }] };
    } else up.easuPass = { colorAttachments: [{ view: null, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }] };
  }
  mainPass.depthStencilAttachment.view = depthTarget.createView();
  if (sea.surfPipe) { // the water's surface for the streaks, a texel to two pixels each way, and the sea's group that reads it
    sea.surf?.destroy(); sea.ice?.destroy();
    const size = [Math.ceil(width / 2), Math.ceil(height / 2)], use = GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING;
    sea.surf = device.createTexture({ size, format: 'rgba16float', usage: use });
    sea.ice = device.createTexture({ size, format: 'r32float', usage: use });
    sea.surfOut = device.createBindGroup({ layout: sea.surfPipe.getBindGroupLayout(2), entries: [{ binding: 0, resource: sea.surf.createView() }, { binding: 1, resource: sea.ice.createView() }] });
    sea.surfGroups = [Math.ceil(size[0] / 8), Math.ceil(size[1] / 8)];
    sea.glow?.destroy();
    sea.glow = device.createTexture({ size, format: 'rgba16float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    sea.glowPass = { colorAttachments: [{ view: sea.glow.createView(), loadOp: 'clear', clearValue: [0, 0, 0, 0], storeOp: 'store' }] };
    const own = [...sea.entries, { binding: 12, resource: sea.surf.createView() }, { binding: 13, resource: sea.ice.createView() }, { binding: 14, resource: { buffer: sea.fits } }];
    const kept = (name, entries) => device.createBindGroup({ layout: groupLayouts[name], entries: entries.filter(e => seaKept[name].includes(e.binding)) });
    groups.seaGlow = kept('seaGlow', own);
    groups.seaGlowMoving = kept('seaGlow', [...own.slice(0, -1), { binding: 14, resource: { buffer: sea.movingFits } }]);
    groups.sea = kept('sea', [...own, { binding: 15, resource: sea.glow.createView() }]);
    groups.seaAfloat = kept('seaAfloat', own);
  }
  if (skyLow.pipe) { // the sky's noise at half the width and height (SKY_MAPS)
    skyLow.texture?.destroy();
    skyLow.texture = device.createTexture({ size: [Math.ceil(width / 2), Math.ceil(height / 2)], format: 'rgba16float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING });
    skyLow.group = device.createBindGroup({ layout: groupLayouts.single, entries: [{ binding: 0, resource: skyLow.texture.createView() }] });
    skyLow.out = device.createBindGroup({ layout: skyLow.outputs, entries: [{ binding: 0, resource: sea.sky.tops[0] }, { binding: 1, resource: skyLow.texture.createView() }] });
    skyLow.size = [Math.ceil(Math.ceil(width / 2) / 8), Math.ceil(Math.ceil(height / 2) / 8)];
  }
  { // a count of the wet tracks' lights for each tile of the frame, WGSL.WET_CAP slots for them, and how many each band of brightness has (WGSL.WET_BIN)
    const tiles = Math.ceil(width / WGSL.WET_TILE) * Math.ceil(height / WGSL.WET_TILE);
    wet.tiles?.destroy();
    wet.tiles = device.createBuffer({ size: tiles * (1 + WGSL.WET_CAP + WGSL.WET_BANDS) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    wet.counts = tiles * 4; wet.bands = tiles * (1 + WGSL.WET_CAP) * 4; // where the counts by brightness start
    wetBind();
  }
}
const skyLow = { pipe: null, texture: null, group: null, pass: {} };
const swNow = { pipe: null, groups: null, n: 0 }; // the low lights' switches, each frame (WGSL.SW_NOW)
// The far ground lit once a pixel (WGSL.terrainDeferred): its buffer, its light, the passes that make them, at the frame's size
// It is drawn so only where the far ground holds more than this many cells a pixel: below that its buffers and passes (some
// 0.1-0.2 ms) cost about what it saves (from over the town, 0.10 a pixel, it saves 2.7 ms; in the street and the cockpit, 0.04
// a pixel, it saved nothing)
let deferMin = 0.05;
// The far plain faces lit by the buffer too (WGSL.PLAIN_GBUF, ?dplain): off by default, as from 150 m and 600 m up over the
// town it cost 2.6-4.4 ms more than it saved (the faces drawn twice; their shading was never the cost), within one page,
// the least of three each way
let dplain = false;
const defer = { prep: null, shade: null, read: null, size: '', on: false, textures: [] };
// The far road lit once a pixel (WGSL.roadDeferred): its list, the number of the last of it drawn at each pixel, the passes
// (whenever any far road is drawn: forced on in the street, the cockpit and the aerial view it cost nothing measurable)
const droad = { prep: null, shade: null, read: null, on: false, recs: null };
let droadMin = 0;
function deferTargets() {
  for (const t of defer.textures) t.destroy();
  const size = [renderW, renderH], at = GPUTextureUsage.RENDER_ATTACHMENT, read = at | GPUTextureUsage.TEXTURE_BINDING;
  const a = device.createTexture({ size, format: 'rgba32uint', usage: read }), b = device.createTexture({ size, format: 'rgba16float', usage: read }), c = device.createTexture({ size, format: 'rgba16float', usage: read });
  const depth = device.createTexture({ size, format: 'depth32float', usage: at });
  const lit = device.createTexture({ size, format: 'rgba16float', usage: read });
  defer.textures = [a, b, c, depth, lit];
  defer.gbuf = device.createBindGroup({ layout: groupLayouts.gbuf, entries: [{ binding: 0, resource: a.createView() }, { binding: 1, resource: b.createView() }] });
  defer.gbuf3 = device.createBindGroup({ layout: groupLayouts.gbuf3, entries: [{ binding: 0, resource: a.createView() }, { binding: 1, resource: b.createView() }, { binding: 2, resource: c.createView() }] });
  defer.plainGroup = device.createBindGroup({ layout: groupLayouts.plainRead, entries: [{ binding: 0, resource: lit.createView() }, { binding: 1, resource: a.createView() }] });
  defer.shaded = device.createBindGroup({ layout: groupLayouts.shaded, entries: [{ binding: 0, resource: lit.createView() }] });
  defer.prepPass = { colorAttachments: [{ view: a.createView(), loadOp: 'clear', clearValue: [0, 0, 0, 0], storeOp: 'store' }, { view: b.createView(), loadOp: 'clear', clearValue: [0, 0, 0, 0], storeOp: 'store' },
    { view: c.createView(), loadOp: 'clear', clearValue: [0, 0, 0, 0], storeOp: 'store' }],
    depthStencilAttachment: { view: depth.createView(), depthClearValue: 0, depthLoadOp: 'clear', depthStoreOp: 'store' } };
  defer.shadePass = { colorAttachments: [{ view: lit.createView(), loadOp: 'clear', clearValue: [0, 0, 0, 0], storeOp: 'store' }],
    depthStencilAttachment: { view: depth.createView(), depthReadOnly: true } };
  defer.size = renderW + 'x' + renderH;
  const index = device.createTexture({ size, format: 'r32uint', usage: read });
  defer.textures.push(index);
  droad.recs ||= device.createBuffer({ size: 16 + WGSL.ROAD_RECS * 96, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }); // struct RoadRec, 96 bytes
  const recs = { binding: 0, resource: { buffer: droad.recs } }, at1 = { binding: 1, resource: index.createView() };
  droad.prepGroup = device.createBindGroup({ layout: groupLayouts.roadPrep, entries: [recs] });
  droad.shadeGroup = device.createBindGroup({ layout: groupLayouts.roadShade, entries: [recs, at1] });
  droad.readGroup = device.createBindGroup({ layout: groupLayouts.roadRead, entries: [recs, at1] });
  droad.prepPass = { colorAttachments: [{ view: index.createView(), loadOp: 'clear', clearValue: [0, 0, 0, 0], storeOp: 'store' }], depthStencilAttachment: { view: depth.createView(), depthReadOnly: true } };
  droad.prepBare = { colorAttachments: droad.prepPass.colorAttachments, depthStencilAttachment: { view: depth.createView(), depthClearValue: 0, depthLoadOp: 'clear', depthStoreOp: 'store' } }; // without the far ground's buffer
  droad.shadePass = { colorAttachments: [], depthStencilAttachment: { view: depth.createView(), depthReadOnly: true } };
}
// How many indices pipeline p's draws marked GFX_CMD_FAR (and not GFX_CMD_WASHED) hold in the main list
function farIndices(f, p) {
  const first = (f + lists.main.first) >> 2, cmds = (f + lists.main.cmds) >> 2, from = mem32[first + p], to = mem32[first + p + 1];
  let n = 0;
  for (let i = from; i < to; i++) { const c = cmds + i * 9; if (mem32[c] < WASHED && (mem32[c] & FAR)) n += mem32[c + 4] * mem32[c + 7]; }
  return n;
}
// The draws of pipeline p in the main list marked GFX_CMD_FAR (and not GFX_CMD_WASHED), drawn with what is set
function runFar(pass, f, p) {
  const first = (f + lists.main.first) >> 2, cmds = (f + lists.main.cmds) >> 2, from = mem32[first + p], to = mem32[first + p + 1];
  unbound();
  for (let i = from; i < to; i++) { const c = cmds + i * 9; if (mem32[c] < WASHED && (mem32[c] & FAR)) drawCmd(pass, c); }
}

// Nothing below makes an object each frame beyond what WebGPU itself hands out (the encoder, its passes, the canvas's
// texture and its view): views on the module's memory are kept until it grows, pass descriptors are filled in once.
let last = 0, mem8, mem32, memF, hudBytes, hudAt, benching = false, splitting = false, segNames = [];
const hudSent = new Uint8Array(HUD_BYTES), hudTold = { kmh: -1, up: 0, road: 0, gear: 0, cam: 0, flags: 0, at: 0 }; // without shared memory: the HUD as the page last had it
const probe = { on: false, x: 0, y: 0, any: 0, n: 0 }, PROBE_FLOATS = 126; // the desk's: the cursor in the frame, -1..1; sizeof(struct desk_probe) / 4
let birdAtStart = false, lookAgain = null;
let still = 0, played = 0, clock = 0, rate = 1; // ?still=8 plays eight seconds from the start at sixty frames to the second, whatever the clock says, then stops the game's clock: a view that can be drawn again exactly, for comparing pictures;
// ?clock=900 starts the game's clock 900 s on (and the town's with it): ?hour= and ?day= set the town's
const mainPass = { colorAttachments: [{ view: null, resolveTarget: null, loadOp: 'clear', storeOp: 'discard', clearValue: [0, 0, 0, 1] }],
  depthStencilAttachment: { view: null, depthClearValue: 0, depthLoadOp: 'clear', depthStoreOp: 'discard' } };
const submitted = [null];
// Whether any water showed last time it was asked (an occlusion query on the water's square): while none does, the sea's
// preparation (its ripples, the sky it mirrors, the lights' streaks and the surface under the pixels) is not worked out.
// Water coming into view is a frame late to learn of it; that frame shows it without its streaks.
const waterSeen = { set: null, resolve: null, reads: [], shown: true, cleared: false };
function waterQuery() {
  if (!waterSeen.set) {
    waterSeen.set = device.createQuerySet({ type: 'occlusion', count: 1 });
    waterSeen.resolve = device.createBuffer({ size: 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    for (let i = 0; i < 3; i++) { // each read's answer taken by callbacks made once, not a pair of closures a frame
      const r = { buffer: device.createBuffer({ size: 8, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }), busy: false, done: null, fail: null };
      r.done = () => { waterSeen.shown = new BigUint64Array(r.buffer.getMappedRange(), 0, 1)[0] > 0n; r.buffer.unmap(); r.busy = false; };
      r.fail = () => { r.busy = false; };
      waterSeen.reads.push(r);
    }
    mainPass.occlusionQuerySet = waterSeen.set;
  }
}
const shown = { km: 0, minute: 0, reverse: 1, zone: new Date().getTimezoneOffset() }; // what the dash's pictures say, as dashPainter first paints them
const patchFrom = { source: null, origin: [0, 0] }, patchTo = { texture: null, origin: [0, 0, 0] }, patchSize = [0, 0];
function patch(source, texture, rect) { // the part of a canvas that was painted again, to the same place in its texture
  patchFrom.source = source; patchTo.texture = texture;
  patchFrom.origin[0] = patchTo.origin[0] = rect[0]; patchFrom.origin[1] = patchTo.origin[1] = rect[1]; patchSize[0] = rect[2]; patchSize[1] = rect[3];
  device.queue.copyExternalImageToTexture(patchFrom, patchTo, patchSize);
}
// The jets' heat shimmer (WGSL.JETS_HAZE). Only the rectangle the hot air can cover on the screen is worked on: a jet's
// air bends the light from its lip to 1.5 m down its axis and no further (boosted, as far as its flame is long), within 1.66 of the gas's edge of the
// axis (where it is under a hundredth as hot as on it), the axis itself bent by the cross-wind: a box along the jet
// that long and that wide, cut where it passes behind the eye, with a margin as wide as the most a pixel is moved. From
// the cockpit the jets under the floor are below the frame, and nothing is worked on at all. That rectangle is copied
// out of the frame twice, the boxes are drawn into one copy reading the other, and it is copied back: a pass over the
// frame itself would load and store all of it. The copies grow in steps of 256 pixels and never shrink.
const hazeFrom = { texture: null, origin: [0, 0] }, hazeTo = { texture: null, origin: [0, 0] }, hazeSize = [0, 0];
const hazePass = { colorAttachments: [{ view: null, loadOp: 'load', storeOp: 'store' }] }, hazeRect = new Float32Array(4), hazeCorners = new Float32Array(24), hazeBox = new Float64Array(4);
function hazeTake(x, y, w, W, H) { // a place in clip space, in front of the eye: the rectangle (hazeBox) grown to take it in
  const px = Math.min(Math.max((x / w * 0.5 + 0.5) * W, -1e6), 1e6), py = Math.min(Math.max((0.5 - y / w * 0.5) * H, -1e6), 1e6);
  hazeBox[0] = Math.min(hazeBox[0], px); hazeBox[1] = Math.min(hazeBox[1], py); hazeBox[2] = Math.max(hazeBox[2], px); hazeBox[3] = Math.max(hazeBox[3], py);
}
function hazeCopy(src, dst, x, y, w, h, back) {
  hazeFrom.texture = src; hazeTo.texture = dst;
  hazeFrom.origin[0] = back ? 0 : x; hazeFrom.origin[1] = back ? 0 : y; hazeTo.origin[0] = back ? x : 0; hazeTo.origin[1] = back ? y : 0;
  hazeSize[0] = w; hazeSize[1] = h;
  return hazeFrom;
}
function shimmer(enc, f, target) {
  const W = renderW, H = renderH;
  const m = f >> 2, J = rocket.jetsAt >> 2; // the frame's view_proj, read where it lies
  const NEAR = 0.05, c = hazeCorners, box = hazeBox;
  box[0] = W; box[1] = H; box[2] = 0; box[3] = 0;
  for (let i = 0; i < ROCKET.FLAMES; i++) {
    const j = J + i * (ROCKET.JET >> 2), len = memF[j + 3];
    if (!(memF[j + 7] > 0.004)) continue;
    const dx = memF[j + 4], dy = memF[j + 5], dz = memF[j + 6];
    // across the jet: two ways square to it and to each other
    let ux = Math.abs(dy) < 0.9 ? -dz : 0, uy = Math.abs(dy) < 0.9 ? 0 : dz, uz = Math.abs(dy) < 0.9 ? dx : -dy;
    const ul = Math.hypot(ux, uy, uz) || 1; ux /= ul; uy /= ul; uz /= ul;
    const vx = dy * uz - dz * uy, vy = dz * ux - dx * uz, vz = dx * uy - dy * ux;
    // boosted it reaches as far as the flame is long; never further than the flame's own box does, at its longest flicker
    const far = Math.min(memF[j + 16] > 0 ? Math.max(1.5, len * 1.1) : 1.5, len * 1.14 + 0.8);
    const r = 1.66 * (memF[j + 12] + far * WGSL.JET_SPREAD) + Math.hypot(memF[j + 8], memF[j + 9], memF[j + 10]) * 1.1 + 0.05;
    for (let k = 0; k < 8; k++) { // the box's corners in clip space: x, y, w
      const s = k & 4 ? far : -0.05, a = k & 1 ? r : -r, b = k & 2 ? r : -r;
      const x = memF[j] + dx * s + ux * a + vx * b, y = memF[j + 1] + dy * s + uy * a + vy * b, z = memF[j + 2] + dz * s + uz * a + vz * b;
      c[k * 3] = memF[m] * x + memF[m + 4] * y + memF[m + 8] * z + memF[m + 12];
      c[k * 3 + 1] = memF[m + 1] * x + memF[m + 5] * y + memF[m + 9] * z + memF[m + 13];
      c[k * 3 + 2] = memF[m + 3] * x + memF[m + 7] * y + memF[m + 11] * z + memF[m + 15];
    }
    for (let k = 0; k < 8; k++) {
      if (c[k * 3 + 2] >= NEAR) hazeTake(c[k * 3], c[k * 3 + 1], c[k * 3 + 2], W, H);
      for (let bit = 1; bit < 8; bit <<= 1) { // each edge that passes behind the eye, where it does
        const o = k | bit;
        if (k & bit || (c[k * 3 + 2] >= NEAR) === (c[o * 3 + 2] >= NEAR)) continue;
        const t = (NEAR - c[k * 3 + 2]) / (c[o * 3 + 2] - c[k * 3 + 2]);
        hazeTake(c[k * 3] + (c[o * 3] - c[k * 3]) * t, c[k * 3 + 1] + (c[o * 3 + 1] - c[k * 3 + 1]) * t, NEAR, W, H);
      }
    }
  }
  const x0 = Math.max(0, Math.floor(box[0] - 32)), y0 = Math.max(0, Math.floor(box[1] - 32)), x1 = Math.min(W, Math.ceil(box[2] + 32)), y1 = Math.min(H, Math.ceil(box[3] + 32));
  const w = x1 - x0, h = y1 - y0;
  if (w <= 0 || h <= 0) return;
  if (!rocket.hazeIn || rocket.hazeIn.width < w || rocket.hazeIn.height < h) {
    const bw = Math.max(w, rocket.hazeIn?.width || 0) + 255 & ~255, bh = Math.max(h, rocket.hazeIn?.height || 0) + 255 & ~255;
    rocket.hazeIn?.destroy(); rocket.hazeOut?.destroy();
    rocket.hazeIn = device.createTexture({ size: [bw, bh], format, usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING });
    rocket.hazeOut = device.createTexture({ size: [bw, bh], format, usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC | GPUTextureUsage.RENDER_ATTACHMENT });
    rocket.hazeUniform ||= device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    rocket.hazeGroup = device.createBindGroup({ layout: groupLayouts.haze, entries: [{ binding: 0, resource: rocket.hazeIn.createView() }, { binding: 1, resource: { buffer: rocket.hazeUniform } }] });
    hazePass.colorAttachments[0].view = rocket.hazeOut.createView();
  }
  hazeRect[0] = x0; hazeRect[1] = y0; hazeRect[2] = rocket.hazeOut.width; hazeRect[3] = rocket.hazeOut.height;
  device.queue.writeBuffer(rocket.hazeUniform, 0, hazeRect);
  enc.copyTextureToTexture(hazeCopy(target, rocket.hazeIn, x0, y0, w, h, false), hazeTo, hazeSize);
  enc.copyTextureToTexture(hazeCopy(target, rocket.hazeOut, x0, y0, w, h, false), hazeTo, hazeSize);
  const p = enc.beginRenderPass(stamped(hazePass, 'shimmer'));
  p.setPipeline(pipes.jetsHaze); p.setBindGroup(0, frameGroup); p.setBindGroup(1, rocket.hazeGroup); p.setVertexBuffer(0, rocket.jets);
  p.setScissorRect(0, 0, w, h); p.draw(36, ROCKET.FLAMES);
  p.end();
  enc.copyTextureToTexture(hazeCopy(rocket.hazeOut, target, x0, y0, w, h, true), hazeTo, hazeSize);
}
// A pass descriptor with bench()'s time stamps on it, or without: the descriptor itself is kept from frame to frame.
function stamped(desc, name) {
  if (benching && timing) { const i = STAMPS.indexOf(name) * 2; desc.timestampWrites = { querySet: timing.set, beginningOfPassWriteIndex: i, endOfPassWriteIndex: i + 1 }; }
  else if (desc.timestampWrites) delete desc.timestampWrites;
  return desc;
}
// The field's slots laid this frame, and the blocks to draw (struct snow_frame from its `slots` at word s): the slots'
// buffer grown first where the C hands out more than it holds, what it held copied over before anything is laid in it
function packSend(s) {
  const need = mem32[s], staged = mem32[s + 1], ninst = mem32[s + 2], words = WGSL.SNOW_SLOT_WORDS * 4;
  if (need > snow.slotCap) {
    let cap = snow.slotCap;
    while (cap < need) cap *= 2;
    cap = Math.min(cap, SNOW_SLOTS_MAX);
    const grown = device.createBuffer({ size: cap * words, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(snow.slots, 0, grown, 0, snow.slotCap * words);
    device.queue.submit([enc.finish()]);
    snow.slots.destroy();
    snow.slots = grown; snow.slotCap = cap;
    snow.bind();
  }
  for (let k = 0; k < staged; k++) device.queue.writeBuffer(snow.slots, mem32[(snow.stageSlot >> 2) + k] * words, mem8, snow.stage + k * words, words);
  if (ninst) device.queue.writeBuffer(snow.blocks, 0, mem8, snow.inst, ninst * 48);
}
// The buffers a run of commands has set, by handle: vertex slots 0 and 1, then the indices. A draw sets only what the one
// before it did not; whoever sets a buffer some other way, or begins a pass, forgets them first (unbound).
const bound = [-1, -1, -1];
function unbound() { bound[0] = bound[1] = bound[2] = -1; }
function drawCmd(to, c) { // one struct gfx_cmd, at mem32[c]
  const vbuf = mem32[c + 1], ibuf = mem32[c + 2], inst = mem32[c + 3];
  let slot = 0;
  if (vbuf !== NONE) { if (bound[0] !== vbuf) to.setVertexBuffer(0, buffers[bound[0] = vbuf]); slot = 1; }
  if (inst !== NONE && bound[slot] !== inst) to.setVertexBuffer(slot, buffers[bound[slot] = inst]);
  if (ibuf !== NONE) { if (bound[2] !== ibuf) to.setIndexBuffer(buffers[bound[2] = ibuf], 'uint32'); to.drawIndexed(mem32[c + 4], mem32[c + 7], mem32[c + 5], mem32[c + 6] | 0, mem32[c + 8]); }
  else to.draw(mem32[c + 4], mem32[c + 7], mem32[c + 5], mem32[c + 8]);
}
let mend = false, mended = 0, errors = 0; // an error on the device since the canvas was last configured; when that was; how many have been reported
let skipDraws = false; // ?skip=1:0 leaves out pipeline 1's first draw
// A pipeline's twin for what none of the lights that come and go reaches (make's calm): a draw marked GFX_CMD_WASHED, in
// its pipe's top bit (a ground draw's, in its rows'), is drawn by the pipeline itself, every other by the twin.
// ?skip=calm draws everything with the pipelines themselves, as it was.
// And a third for the draws marked GFX_CMD_FAR beside it, the next bit down, where no light marks them (make's far): with
// nothing of what only shows near by. ?skip=far draws those with the twin.
const calmOf = new Map(), farOf = new Map(), WASHED = 0x80000000, FAR = 0x40000000;
// The frame's list of draws (where in struct frame each part lies, from hs_layout)
const lists = { main: { first: 0, cmds: 0, nground: 0, ground: 0 } };
// The ground a list sees, as the C gathered it: the town's, then the backdrop's; in each the plain chunks, then those a shader opens
// The clouds' own pass (WGSL.CLOUD): at half the frame's size each way, unsampled. Every cloud's puffs, far to near, each
// laid over those behind it; beside the picture, how near the nearest puff on each pixel is seen. The frame lays the
// picture over itself at that depth (WGSL.CLOUD_OVER), so what stands before a cloud hides it.
const cloudLow = {};
function cloudPipes() {
  const module = code => { const m = device.createShaderModule({ code: WGSL.PRELUDE + code }); m.getCompilationInfo().then(info => { for (const e of info.messages) if (e.type === 'error') post({ error: `shader, line ${e.lineNum}: ${e.message}` }); }); return m; };
  const layout = g => device.createPipelineLayout({ bindGroupLayouts: [frameLayout, groupLayouts[g]] });
  const cloud = module(WGSL.CLOUD), prep = module(WGSL.CLOUD_PREP);
  const tiles = device.createShaderModule({ code: WGSL.CLOUD_TILES });
  cloudLow.tilesPipe = device.createRenderPipeline({ layout: 'auto', vertex: { module: tiles }, fragment: { module: tiles, targets: [{ format: 'r16float' }] } });
  // what each puff takes the same everywhere, a texel a puff (WGSL.CLOUD_PREP)
  const rows = Math.ceil(CLOUDS * WGSL.CLOUD_PUFFS / WGSL.CLOUD_PREP_W);
  cloudLow.prep = [0, 1, 2, 3].map(() => device.createTexture({ size: [WGSL.CLOUD_PREP_W, rows], format: 'rgba16float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING }));
  cloudLow.prepPass = { colorAttachments: cloudLow.prep.map(t => ({ view: t.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] })) };
  return [device.createRenderPipelineAsync({ layout: layout('cloudDraw'), vertex: { module: cloud },
    fragment: { module: cloud, targets: [{ format: 'rgba16float', blend: BLEND.laid }, { format: 'r16float', blend: BLEND.lens }] }, // the nearer of two puffs is the greater
    primitive: { cullMode: 'none' } }),
    device.createRenderPipelineAsync({ layout: layout('cloud'), vertex: { module: prep }, fragment: { module: prep, targets: cloudLow.prep.map(() => ({ format: 'rgba16float' })) } })];
}
// The picture, how near its puffs are seen, and which of its tiles hold any, at 1/d of the frame each way: one set for
// the size the picture is drawn at (WGSL.CLOUD_DIV), at every depth in a cloud: a coarser one deep in its haze would jump
function cloudTarget(d) {
  const w = Math.ceil(renderW / d), h = Math.ceil(renderH / d), t = cloudLow[d] ||= { size: [0, 0] };
  if (t.size[0] === w && t.size[1] === h) return t;
  t.size = [w, h];
  for (const x of [t.picture, t.near, t.tiles]) x?.destroy();
  const draw = GPUTextureUsage.RENDER_ATTACHMENT, read = GPUTextureUsage.TEXTURE_BINDING;
  t.picture = device.createTexture({ size: [w, h], format: 'rgba16float', usage: draw | read });
  t.near = device.createTexture({ size: [w, h], format: 'r16float', usage: draw | read });
  t.pass = { colorAttachments: [{ view: t.picture.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] },
    { view: t.near.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] }] };
  // which tiles of the picture have any cloud in them (WGSL.CLOUD_TILES), for the laying over to leave the rest out
  const tw = Math.ceil(w / WGSL.CLOUD_TILE), th = Math.ceil(h / WGSL.CLOUD_TILE);
  t.tiles = device.createTexture({ size: [tw, th], format: 'r16float', usage: draw | read });
  t.tileCount = tw * th;
  t.tilesPass = { colorAttachments: [{ view: t.tiles.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] }] };
  t.tilesGroup = device.createBindGroup({ layout: cloudLow.tilesPipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: t.picture.createView() }] });
  t.group = device.createBindGroup({ layout: groupLayouts.cloudOver, entries: [{ binding: 0, resource: t.picture.createView() }, { binding: 1, resource: t.near.createView() },
    { binding: 2, resource: device.createSampler({ minFilter: 'linear', magFilter: 'linear' }) }, { binding: 3, resource: t.tiles.createView() }] }); // (the sampler clamps at the picture's edge)
  return t;
}
// Every puff of every cloud drawn into the picture; returns the picture's set, for the frame to lay over itself.
function cloudPass(enc, f, clouds) {
  const n = clouds * WGSL.CLOUD_PUFFS, t = cloudTarget(WGSL.CLOUD_DIV);
  cloudLow.drawGroup ||= device.createBindGroup({ layout: groupLayouts.cloudDraw, entries: [...live.noise, { binding: 2, resource: { buffer: live.puff } },
    { binding: 3, resource: { buffer: live.cloud } }, ...cloudLow.prep.map((p, k) => ({ binding: 4 + k, resource: p.createView() }))] });
  for (let k = skip.cloudx10 ? 10 : 1; k > 0; k--) { // (?skip=cloudx10: all of it ten times over, to time it on a busy machine)
    const prep = enc.beginRenderPass(stamped(cloudLow.prepPass, 'cloud prep'));
    prep.setBindGroup(0, frameGroup);
    prep.setPipeline(cloudLow.prepPipe); prep.setBindGroup(1, live.group); prep.setScissorRect(0, 0, WGSL.CLOUD_PREP_W, Math.ceil(n / WGSL.CLOUD_PREP_W)); prep.draw(3);
    prep.end();
    const pass = enc.beginRenderPass(stamped(t.pass, 'clouds'));
    pass.setBindGroup(0, frameGroup);
    pass.setPipeline(cloudLow.pipe); pass.setBindGroup(1, cloudLow.drawGroup); pass.draw(6, n);
    pass.end();
  }
  const tiles = enc.beginRenderPass(t.tilesPass);
  tiles.setPipeline(cloudLow.tilesPipe); tiles.setBindGroup(0, t.tilesGroup); tiles.draw(3);
  tiles.end();
  return t;
}

// The field's blocks (WGSL.PACK), each mesh level's in one instanced draw (struct snow_frame: level_first, level_count)
const PACK_MESH = [], SNOW_SLOTS_MAX = 6144; // snow.h
function packDraw(pass, f) {
  const s = ((f + FRAME.SNOW) >> 2) + 15 + 6 + SNOW_RUNS / 32 + 3;
  pass.setIndexBuffer(snow.mesh, 'uint16');
  for (let l = 0; l < WGSL.SNOW_MESHES; l++) {
    const n = mem32[s + WGSL.SNOW_MESHES + l];
    if (n) pass.drawIndexed(PACK_MESH[l][1], n, PACK_MESH[l][0], 0, mem32[s + l]);
  }
}

// The drifts beyond the field's blocks (WGSL.DUNE_SHADER), every node a block's finest mesh, in one instanced draw; and
// the field's group bound again for what follows
// The traffic's bodies: each look's body drawn once for every car of it that the view shows (struct traffic_seen), with
// the live mesh's pose.
function herdDraw(pass) {
  const at = herd.seen >> 2;
  if (!mem32[at] || !pipes.herd) return 0;
  pass.setPipeline(pipes.herd); pass.setBindGroup(1, live.pose); pass.setVertexBuffer(0, herd.vertex); pass.setVertexBuffer(1, herd.inst); pass.setIndexBuffer(herd.index, 'uint32');
  let draws = 0, tris = 0;
  for (let l = 0, r = herd.runs >> 2; l < herd.looks; l++) {
    const n = mem32[at + 2 + 2 * l];
    if (!n) continue;
    // the near ones first, with the full shading; the rest, a few pixels long, with the far (struct traffic_seen, near)
    const near = pipes.herdFar && !skip.herdfar ? Math.min(mem32[at + 1 + 2 * herd.looks + HERD_CARS + l], n) : n, first = mem32[at + 1 + 2 * l];
    if (near) { pass.drawIndexed(mem32[r + 2 * l + 1], near, mem32[r + 2 * l], 0, first); draws++; }
    if (near < n) { pass.setPipeline(pipes.herdFar); pass.drawIndexed(mem32[r + 2 * l + 1], n - near, mem32[r + 2 * l], 0, first + near); pass.setPipeline(pipes.herd); draws++; }
    tris += n * mem32[r + 2 * l + 1] / 3;
  }
  stats.herd = [mem32[at], tris]; // the cars drawn, and their triangles
  return draws;
}
function duneDraw(pass) {
  if (!dune.count || !pipes.dune) return;
  pass.setPipeline(defer.on && pipes.duneRead ? pipes.duneRead : pipes.dune); pass.setBindGroup(1, dune.group); // (with the far ground's light bound, by it)
  pass.setIndexBuffer(snow.mesh, 'uint16'); pass.drawIndexed(PACK_MESH[0][1], dune.count, PACK_MESH[0][0]);
  pass.setBindGroup(1, snow.group);
}

// How many cells of the ground the far draws (GFX_CMD_FAR) hold in this frame's list
function farCells(f, list) {
  let cells = 0;
  for (let q = 0; q < 4; q++) {
    const n = mem32[((f + list.nground) >> 2) + q];
    for (let i = 0, d = (f + list.ground + q * FRAME.GROUND_DRAWS * 16) >> 2; i < n; i++, d += 4) {
      const rows = mem32[d + 3], k = mem32[d + 2] >>> LEVEL_SHIFT;
      if (rows < WASHED && (rows & FAR)) cells += (mem32[d + 1] / 6 >> k) * ((rows & 0x3fffffff) >> k);
    }
  }
  return cells;
}
function ground(pass, f, list, under = null, prep = false) { // under: the pipelines that draw its depth alone, the town's and the backdrop's by kind
  let draws = 0;
  for (let g = 0; g < 2; g++) {
    pass.setBindGroup(1, g ? groups.outer : groups.terrain);
    const strips = g ? outerRowIndex : rowIndex;
    let level = -1;
    for (let kind = 0; kind < 2; kind++) {
      const n = mem32[((f + list.nground) >> 2) + g * 2 + kind];
      if (!n) continue;
      const pipe = under ? under[g][kind] : g ? (kind ? pipes.outerCut : pipes.outer) : (kind ? pipes.terrainCut : pipes.terrain);
      const lit = !under && list === lists.main && defer.on ? (prep ? defer.prep : defer.read)[g * 2 + kind] : null; // the far ground, by the buffer
      const twin = skip.calm ? undefined : calmOf.get(pipe); // what no light that comes and goes reaches, by the twin (calmOf)
      const third = twin && !skip.far ? farOf.get(pipe) : undefined; // and of that, what is far off, by the third (farOf)
      let now = null;
      for (let i = 0, d = (f + list.ground + (g * 2 + kind) * FRAME.GROUND_DRAWS * 16) >> 2; i < n; i++, d += 4) {
        const rows = mem32[d + 3], far = lit && rows < WASHED && (rows & FAR);
        if (prep && !far) continue;
        const want = far ? lit : !twin || rows >= WASHED ? pipe : third && (rows & FAR) ? third : twin;
        if (want !== now) pass.setPipeline(now = want);
        // at level k (terrain_lod_draws), the level's cells: 2^k of the fine ones a side, the rows counted in the level's
        const row = mem32[d + 2], k = skip.lod ? 0 : row >>> LEVEL_SHIFT;
        if (k !== level) pass.setIndexBuffer(strips[level = k], 'uint16');
        if (list === lists.main && !under && !prep) (stats.levels ||= [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])[g * 5 + k] += mem32[d + 1] / 6 * (rows & 0x3fffffff); // the cells drawn at each level, the town's and the backdrop's (for report())
        pass.drawIndexed((mem32[d + 1] / 6 >> k) * 6, (rows & 0x3fffffff) >> k, (mem32[d] / 6 >> k) * 6, 0, (k << LEVEL_SHIFT | (row & 0x0fffffff) >> k) >>> 0);
      }
      draws += n;
    }
  }
  return draws;
}
function run(pass, f, p, pipe = pipes[p], quads = null, list = lists.main, near = false) { // every draw of pipeline p in this frame's list, drawn with `pipe`; returns how many. quads: each draws six corners an instance, by these; near: none marked far (GFX_CMD_FAR) indices
  const first = (f + list.first) >> 2, cmds = (f + list.cmds) >> 2, from = mem32[first + p], to = mem32[first + p + 1];
  if (!stats.frames && list === lists.main) { stats.indices ||= {}; stats.indices[p] = Array.from({ length: to - from }, (_, i) => mem32[cmds + (from + i) * 9 + 4] * mem32[cmds + (from + i) * 9 + 7]); } // what each pipeline draws, for report()
  if (from === to) return 0;
  const twin = skip.calm ? undefined : calmOf.get(pipe); // (none: every draw by the pipeline, marked or not)
  const third = twin && !skip.far ? (droad.on && pipe === pipes[PIPE.ROAD] ? droad.read : defer.plain && pipe === pipes[PIPE.PLAIN] ? defer.plainRead : farOf.get(pipe)) : undefined; // the far road by its list (droad), the far plain faces by the buffer (defer)
  let now = twin ? null : pipe;
  if (!twin) pass.setPipeline(pipe);
  unbound();
  if (quads) pass.setIndexBuffer(quads, 'uint16');
  for (let i = from; i < to; i++) {
    const c = cmds + i * 9;
    if (twin) { const want = mem32[c] >= WASHED ? pipe : third && (mem32[c] & FAR) ? third : twin; if (want !== now) pass.setPipeline(now = want); }
    if (near && (mem32[c] & FAR) && mem32[c] < WASHED) continue;
    if (quads) { const inst = mem32[c + 3]; if (bound[0] !== inst) pass.setVertexBuffer(0, buffers[bound[0] = inst]); pass.drawIndexed(6, mem32[c + 7], 0, 0, mem32[c + 8]); }
    else if (!skipDraws || !skip[p + ':' + (i - from)]) drawCmd(pass, c);
  }
  return to - from;
}
// Frame pacing. A frame shown now after one refresh and now after two judders, however good the average: when the GPU
// cannot finish every frame within one refresh (a 120 Hz screen gives it 8.3 ms), draw on every `pace`-th refresh only,
// evenly. The refresh is the shortest gap between callbacks lately; a drawn frame that came more than `pace` refreshes
// after the last one is a miss. More than a twentieth missed over the last 60 drawn frames: pace one refresh slower.
// Every few seconds without a miss, try one faster; if that misses at once, wait twice as long before trying again.
// ?pace=0 draws every refresh as before; ?still (pictures for comparing) is never paced.
const PACE = { window: 60, miss: 0.05, most: 4, retry: 4000 };
let pace = 1, paceOn = true, refresh = 1000 / 60, refreshSeen = 0, drawnAt = 0, paceMisses = 0, paceN = 0, paceRing = new Uint8Array(PACE.window);
let paceCalm = 0, paceRetry = PACE.retry, paceTried = 0;
function tick(now) {
  requestAnimationFrame(tick);
  if (benching) return;
  if (refreshSeen) { const gap = now - refreshSeen; if (gap > 2 && gap < refresh * 0.97) refresh = gap; else refresh += (Math.min(gap, refresh * 1.5) - refresh) * 0.002; }
  refreshSeen = now;
  if (paceOn && !still && drawnAt) {
    const since = Math.round((now - drawnAt) / refresh);
    if (since < pace) return;
    const missed = since > pace ? 1 : 0;
    paceMisses += missed - paceRing[paceN % PACE.window]; paceRing[paceN++ % PACE.window] = missed;
    if (paceN >= PACE.window && paceMisses > PACE.window * PACE.miss && pace < PACE.most) {
      if (paceTried && now - paceTried < 2000) paceRetry = Math.min(paceRetry * 2, 60000); // the faster pace failed at once
      pace++; paceRing.fill(0); paceMisses = 0; paceN = 0; paceCalm = now; paceTried = 0;
    } else if (missed) paceCalm = now;
    else if (pace > 1 && now - paceCalm > paceRetry) { pace--; paceRing.fill(0); paceMisses = 0; paceN = 0; paceCalm = now; paceTried = now; }
  }
  drawnAt = now;
  try { frame(now); } catch (e) { post({ error: String(e.stack || e) }); throw e; }
}
function frame(now) {
  const t0 = performance.now();
  if (mend && t0 - mended > 500) { // see where the device's errors are listened for
    mend = false; mended = t0;
    configure();
    resize(canvas.width, canvas.height);
    hs.hs_shadows_stale(); // the shadow maps' textures and views outlive the error, but a pass may have been lost: all three are drawn again
  }
  let dt = last ? (now - last) / 1000 : 0;
  if (still && mem32 && (introGo || mem32[(hudAt + 56) >> 2] & 16)) { // counted from when the title is dismissed, or the zoom set going
    if (!played) { hs.hs_set(SET.TIME, clock); if (birdAtStart) hs.hs_event(EVENT.BIRD, birdAtStart); }
    dt = played++ < still * 60 ? 1 / 60 : 0;
  } else if (still) dt = 0; // before it, nothing moves: however many frames the title is up, the play from it is the same
  if (popFrames) skip.water = played <= popFrames;
  if (last) stats.times[stats.n++ & 4095] = now - last;
  last = now;
  if (keysView) { keys = Atomics.load(keysView, 0); pad.set(padView); }
  hs.hs_pad(pad[0], pad[1], pad[2], pad[3], pad[4], pad[5]);
  for (let k = 1; k < rate && dt > 0; k++) hs.hs_frame(keys, dt, renderW, renderH); // ?rate=4: the game four times as fast (each step its own frame's)
  const f = hs.hs_frame(keys, dt, renderW, renderH);
  photoPool.step(dt); // what of the photo this frame's eye needs, and the car's
  const t1 = performance.now();
  if (!mem32 || mem32.buffer !== hs.memory.buffer) {
    mem8 = new Uint8Array(hs.memory.buffer); mem32 = new Uint32Array(hs.memory.buffer); memF = new Float32Array(hs.memory.buffer);
    hudAt = hs.hs_hud(); hudBytes = new Uint8Array(hs.memory.buffer, hudAt, HUD_BYTES);
  }
  if (hudView) hudView.set(hudBytes);
  if (probe.on && (probe.n++ & 3) === 0) { // the debug desk's panel (src/desk.js): what is under the cursor, what goes on in town, ~15 times a second
    const at = hs.hs_probe(probe.x, probe.y, probe.any) >> 2;
    post({ probe: memF.slice(at, at + PROBE_FLOATS) });
  }
  else { // No shared memory: a message, 80 bytes copied from the one buffer kept for it. What the page writes out (speed in whole
    // km/h, metres up, the road, the gear, the camera, the flags) goes the frame it changes; what only slides (the minimap's place
    // and turn, what the sound follows) goes when it has changed, thirty times a second at most. A parked car sends nothing.
    const h = hudAt >> 2, kmh = Math.round(Math.abs(memF[h]) * 3.6), up = Math.round(memF[h + 1]);
    let told = kmh !== hudTold.kmh || up !== hudTold.up || mem32[h + 11] !== hudTold.road || mem32[h + 12] !== hudTold.gear || mem32[h + 13] !== hudTold.cam || mem32[h + 14] !== hudTold.flags;
    if (!told && t0 - hudTold.at >= 33) for (let i = 0; i < HUD_BYTES && !told; i++) told = (i >> 2) !== 15 && hudSent[i] !== hudBytes[i]; // all but the frame's number
    if (told) {
      hudTold.kmh = kmh; hudTold.up = up; hudTold.road = mem32[h + 11]; hudTold.gear = mem32[h + 12]; hudTold.cam = mem32[h + 13]; hudTold.flags = mem32[h + 14]; hudTold.at = t0;
      hudSent.set(hudBytes);
      postMessage(hudSent.buffer);
    }
  }

  const bytes = mem8;
  device.queue.writeBuffer(frameBuffer, 0, bytes, f, FRAME.WATER_FRESH); // the frame's, the car's, the snow's, the sea's and its waves', the live mesh's pose, in one
  for (const l of groundLod) if (l.n) device.queue.writeBuffer(l.buffer, 0, skip.lod ? (l.zero ||= new Uint8Array(l.n * 4)) : bytes, skip.lod ? 0 : l.at, l.n * 4); // each chunk's level of detail (terrain_lod); ?skip=lod: none
  if (skip.plough) device.queue.writeBuffer(frameBuffer, FRAME.PLOUGH, new Uint8Array(FRAME.PLOUGH_BYTES)); // ?skip=plough: the ploughs light nothing
  { // the snow: its uniforms, and whichever of its arrays the car changed this frame (struct snow_frame)
    const s = ((f + FRAME.SNOW) >> 2) + 15, sprayFirst = mem32[s], sprayCount = mem32[s + 1],
      trackFirst = mem32[s + 2], trackCount = mem32[s + 3], bankFirst = mem32[s + 4], bankCount = mem32[s + 5];
    packSend(s + 6 + SNOW_RUNS / 32);
    if (sprayCount) {
      const n = Math.min(sprayCount, 1536 - sprayFirst);
      device.queue.writeBuffer(snow.spray, sprayFirst * 28, bytes, snow.flecks + sprayFirst * 28, n * 28);
      if (n < sprayCount) device.queue.writeBuffer(snow.spray, 0, bytes, snow.flecks, (sprayCount - n) * 28);
    }
    if (trackCount) {
      device.queue.writeBuffer(snow.ribbon, trackFirst * 160, bytes, snow.tracks + trackFirst * 160, trackCount * 160);
      snow.laid = Math.max(snow.laid || 0, trackFirst + trackCount); // nothing beyond the furthest cross-section ever written needs drawing
    }
    for (let k = 0; k < SNOW_RUNS;) { // the traffic's runs rewritten, a bit each: sent whole, those next to each other in one
      if (!(mem32[s + 6 + (k >> 5)] >>> (k & 31) & 1)) { k++; continue; }
      let n = 1;
      while (k + n < SNOW_RUNS && mem32[s + 6 + ((k + n) >> 5)] >>> ((k + n) & 31) & 1) n++;
      const at = (SNOW_TRACKS + k * SNOW_RUN) * 160;
      device.queue.writeBuffer(snow.ribbon, at, bytes, snow.tracks + at, n * SNOW_RUN * 160);
      snow.runs = Math.max(snow.runs, k + n); // nor beyond the furthest run
      k += n;
    }
    if (bankCount && snow.bank) device.queue.writeBuffer(snow.bank, bankFirst * 20, bytes, snow.bankAt + bankFirst * 20, bankCount * 20);
  }
  { // the drifts beyond the field: the corners' pages filled since the last frame, and the nodes to draw (struct dune_frame)
    const h = dune.head >> 2, n = mem32[h + 7], sent = mem32[h + 8], P = 32, N = WGSL.DUNE_N, across = N / P;
    for (let q = 0; q < sent; q++) {
      const k = mem32[(dune.fresh >> 2) + q], x = (k % across) * P, y = Math.floor(k / across) * P;
      device.queue.writeTexture({ texture: dune.open, origin: [x, y] }, bytes, { offset: dune.openAt + y * N + x, bytesPerRow: N }, [P, P]);
    }
    if (n) device.queue.writeBuffer(dune.buffer, 0, bytes, dune.head, 48 + n * 16);
    dune.count = n;
  }
  winterSend(); // the roads' snow cells changed this frame, and the windrows grown
  { // the sections of the gritted track laid this frame (struct plough_frame)
    const first = mem32[grit.state], count = mem32[grit.state + 1], n = Math.min(count, grit.sections - first), row = grit.across * 16;
    if (n) device.queue.writeBuffer(grit.buffer, first * row, bytes, grit.at + first * row, n * row);
    if (count > n) device.queue.writeBuffer(grit.buffer, 0, bytes, grit.at, (count - n) * row);
    if (count) grit.laid = Math.max(grit.laid, first + n, count > n ? count - n : 0);
  }
  // the jets: nothing is sent while the doors are shut and the snow they blew has settled
  const rocketsOut = mem32[rocket.state], blownLive = mem32[rocket.drift], blownFirst = mem32[rocket.drift + 1], blownCount = mem32[rocket.drift + 2]; // struct rocket_frame, struct drift_frame
  for (let b = 0; b < cockpit.runs.length; b++) device.queue.writeBuffer(cockpit.runs[b].uniform, 0, bytes, cockpit.pose + b * FRAME.CAR_BYTES, FRAME.CAR_BYTES); // the flight cockpit's two blocks
  const wingsOut = mem32[wing.shown]; // struct wing_frame's shown: nothing of it to draw with the doors shut
  if (wingsOut) for (let b = 0; b < wing.runs.length; b++) device.queue.writeBuffer(wing.runs[b].uniform, 0, bytes, wing.pose + b * FRAME.CAR_BYTES, FRAME.CAR_BYTES);
  if (rocketsOut) {
    device.queue.writeBuffer(rocket.uniform, 0, bytes, rocket.pose, FRAME.CAR_BYTES);
    device.queue.writeBuffer(rocket.jets, 0, bytes, rocket.jetsAt, ROCKET.FLAMES * ROCKET.JET);
  }
  if (blownCount) {
    const n = Math.min(blownCount, ROCKET.PUFFS - blownFirst);
    device.queue.writeBuffer(rocket.puffs, blownFirst * ROCKET.PUFF, bytes, rocket.puffsAt + blownFirst * ROCKET.PUFF, n * ROCKET.PUFF);
    if (n < blownCount) device.queue.writeBuffer(rocket.puffs, 0, bytes, rocket.puffsAt, (blownCount - n) * ROCKET.PUFF);
  }
  if (smoke.count) { // the plumes furthest first, and the puffs of those the air has stirred (struct smoke_frame)
    const n = mem32[smoke.frame], slots = mem32[smoke.frame + 1], order = mem32[smoke.frame + 2], puffs = mem32[smoke.frame + 3], steam = mem32[smoke.frame + 4], steaming = mem32[smoke.frame + 5];
    device.queue.writeBuffer(smoke.order, 0, bytes, order, n * 4);
    if (slots) device.queue.writeBuffer(smoke.puffs, 0, bytes, puffs, slots * SMOKE.PUFFS * SMOKE.PUFF);
    if ((smoke.steaming = steaming)) device.queue.writeBuffer(smoke.steam, 0, bytes, steam, SMOKE.STEAM * SMOKE.PUFF);
  }
  const lv = (f + FRAME.LIVE) >> 2, liveV = mem32[lv], liveI = mem32[lv + 1], liveGlow = mem32[lv + 2], clouds = mem32[lv + 3], cloudsFresh = mem32[lv + 4];
  const blurFirst = mem32[lv + 6], blurCount = mem32[lv + 7]; // struct live_frame: the rotors' and propellers' discs
  if (mem32[lv + 5]) device.queue.writeBuffer(live.vertex, 0, bytes, live.v, liveV * 20);
  if (liveGlow) device.queue.writeBuffer(live.glow, 0, bytes, live.glowAt, liveGlow * 32);
  stats.liveGlow = liveGlow; stats.shipStreaks = mem32[sea.movingCount];
  if (cloudsFresh) { device.queue.writeBuffer(live.cloud, 0, bytes, live.boxes, clouds * CLOUD_BOX); device.queue.writeBuffer(live.puff, 0, bytes, live.puffs, clouds * WGSL.CLOUD_PUFFS * CLOUD_BOX); }
  { // the light list (struct shine): its records, the grid's offsets and index list, and the sectors worked out this frame
    const c = shine.counts >> 2, n = mem32[c], nindex = mem32[c + 2], lo = mem32[c + 3], hi = mem32[c + 4];
    if (n) device.queue.writeBuffer(shine.lights, 0, bytes, shine.lightsAt, n * WGSL.SHINE_LIGHT_BYTES);
    device.queue.writeBuffer(shine.grid, 0, bytes, shine.gridAt, (WGSL.SHINE_BUCKETS + 1) * 4);
    if (nindex) device.queue.writeBuffer(shine.grid, WGSL.SHINE_INDEX0 * 4, bytes, shine.gridAt + WGSL.SHINE_INDEX0 * 4, nindex * 4);
    if (hi > lo) { const a = (WGSL.SHINE_BUCKETS + 1 + (lo >> 1)) * 4, b = (WGSL.SHINE_BUCKETS + 1 + ((hi + 1) >> 1)) * 4; device.queue.writeBuffer(shine.grid, a, bytes, shine.gridAt + a, b - a); }
    stats.lights = n; stats.lightsWorked = mem32[c + 5]; stats.cloudLights = mem32[c + 8];
    { const c0 = WGSL.SHINE_CLOUDS0 * 4; device.queue.writeBuffer(shine.grid, c0, bytes, shine.gridAt + c0, (WGSL.SHINE_CLOUD_BOXES + 1 + mem32[c + 8]) * 4); } // each cloud's lights (shine_clouds)
    if (mem32[c + 7] !== shine.headsGen) { // the street lamps' heads, laid once (shine_heads): their records after the moving lights', and their grid
      shine.headsGen = mem32[c + 7]; const heads = mem32[c + 6], h0 = WGSL.SHINE_HEADS0 * 4, words = WGSL.SHINE_BUCKETS + 1 + mem32[(shine.gridAt + h0) / 4 + WGSL.SHINE_BUCKETS];
      if (heads) device.queue.writeBuffer(shine.lights, WGSL.SHINE_MOVING * WGSL.SHINE_LIGHT_BYTES, bytes, shine.lightsAt + WGSL.SHINE_MOVING * WGSL.SHINE_LIGHT_BYTES, heads * WGSL.SHINE_LIGHT_BYTES);
      device.queue.writeBuffer(shine.grid, h0, bytes, shine.gridAt + h0, words * 4);
    }
  }
  { const n = mem32[herd.seen >> 2]; if (n) device.queue.writeBuffer(herd.inst, 0, bytes, herd.seen + (1 + 2 * herd.looks) * 4, n * 4); } // the cars the view shows
  { // The dash. Its needles and lamps ride in the car's uniforms; an LCD is painted again when its text changes (a kilometre
    // driven, a minute gone, the selector moved), and only that rectangle is copied
    const km = mem32[(f + FRAME.ODO) >> 2], reverse = (mem32[(hudAt + 48) >> 2] | 0) < 0 ? -1 : 1;
    let minute = still ? 0 : Math.floor(Date.now() / 60000 - shown.zone) % 1440;
    if (minute !== shown.minute) { // once a minute, and across a change of the clocks
      const d = new Date(); shown.zone = d.getTimezoneOffset(); shown.minute = minute = d.getHours() * 60 + d.getMinutes();
      patch(car.dash.layers[0], car.dials, car.dash.clock(minute)); patch(car.dash.panels, car.panels, car.dash.panelClock(minute));
    }
    if (km !== shown.km) patch(car.dash.layers[0], car.dials, car.dash.odo(String(car.dash.odoStart + (shown.km = km))));
    if (reverse !== shown.reverse) patch(car.dash.panels, car.panels, car.dash.gate(shown.reverse = reverse));
  }
  if (mem32[((f + FRAME.FOREST) >> 2) + 1]) { // the trees this frame draws, and those that stand in each shadow map: the words of the list that are not as they were
    const first = mem32[(f + FRAME.FOREST) >> 2] * 4;
    device.queue.writeBuffer(forestList, first, bytes, forestAt + first, mem32[((f + FRAME.FOREST) >> 2) + 1] * 4);
  }
  { // what the car stirs up on the water, when it has: the wake's ribbon, the rings, the droplets (struct water_frame, after the uniforms)
    const fresh = mem32[(f + FRAME.WATER_FRESH) >> 2], at = sea.at + SEA_BYTES + WAVES_BYTES;
    const time = memF[((f + FRAME.CAM_POS) >> 2) + 3]; // the wake and the droplets are drawn only until the last of them has faded (WAKE_LIFE, SPRAY_LIFE in the shaders)
    if (fresh & 1) { device.queue.writeBuffer(sea.wake, 0, bytes, at, WAKE_BYTES); sea.wakeUntil = time + 9.5; }
    if (fresh & 2) { device.queue.writeBuffer(sea.rings, 0, bytes, at + WAKE_BYTES, RINGS_BYTES); sea.ringsLive = true; }
    if (fresh & 4) { device.queue.writeBuffer(sea.drops, 0, bytes, at + WAKE_BYTES + RINGS_BYTES, DROPS_BYTES); sea.dropsUntil = time + 1.5; }
    sea.wakeLive = time < sea.wakeUntil; sea.dropsLive = time < sea.dropsUntil;
  }
  const moving = skip.water ? 0 : mem32[sea.movingCount]; // the ships' lights on the water, this frame's
  if (moving) device.queue.writeBuffer(sea.moving, 0, bytes, sea.movingAt, moving * 36);
  const t2 = performance.now();
  const enc = device.createCommandEncoder();
  waterQuery();
  const water = !skip.water && waterSeen.shown;
  { // One compute pass: the aurora's rows of noise, which what follows and the frame's shaders read; the ripple tiles
    // and the sky the water and the wet roads mirror, worked out afresh and halved down their mip chains; the lights
    // the wet roads mirror, binned into the frame's tiles; and the sky's noise for this view at half its width and height
    const wetRoads = wet.bin && !skip.road && !skip.wet, binning = wetRoads && !skip.wetlights; // ?skip=wet: no tiles, and the roads drawn dry; ?skip=wetlights: no tiles
    if (wet.bin) { enc.clearBuffer(wet.tiles, 0, wet.counts); enc.clearBuffer(wet.tiles, wet.bands); }
    if (binning) { // the lights that could show listed, a pass of its own: the passes after take their number of workgroups from it
      device.queue.writeBuffer(wet.live, 0, wet.none);
      const lp = enc.beginComputePass();
      lp.setBindGroup(0, wet.frame); lp.setBindGroup(1, wet.bin); lp.setPipeline(wet.culler); lp.dispatchWorkgroups(Math.ceil(wet.n / 64));
      lp.end();
      enc.copyBufferToBuffer(wet.live, 0, wet.args, 0, 12);
    }
    const cp = enc.beginComputePass(stamped(skyLow.pass, 'maps'));
    if (swNow.pipe) { cp.setPipeline(swNow.pipe); cp.setBindGroup(0, swNow.groups[0]); cp.setBindGroup(1, swNow.groups[1]); cp.dispatchWorkgroups(Math.ceil(swNow.n / 64)); } // the low lights' switches burning now (WGSL.SW_NOW)
    if (binning) { // then counted, a workgroup a light; each tile's least band found; then laid
      cp.setBindGroup(0, wet.frame); cp.setBindGroup(1, wet.bin);
      cp.setPipeline(wet.counter); cp.dispatchWorkgroupsIndirect(wet.args, 0);
      cp.setPipeline(wet.least); cp.dispatchWorkgroups(Math.ceil(wet.counts / 4 / 64));
      cp.setPipeline(wet.pipe); cp.dispatchWorkgroupsIndirect(wet.args, 0);
    }
    cp.setPipeline(skyLow.rowsPipe); cp.setBindGroup(0, skyLow.rowsGroups[0]); cp.setBindGroup(1, skyLow.rowsGroups[1]); cp.dispatchWorkgroups(1024 / 64, 60);
    cp.setPipeline(skyLow.lightPipe); cp.dispatchWorkgroups(1);
    if (water || wetRoads) { // the sky the water mirrors, and the wet roads
      cp.setPipeline(skyLow.pano); cp.setBindGroup(0, skyLow.frame); cp.setBindGroup(1, skyLow.out); cp.dispatchWorkgroups(WGSL.SKY_W / 8, WGSL.SKY_H / 8);
      sea.sky.halve(cp);
    }
    if (water) {
      cp.setPipeline(sea.ripplePipe); cp.setBindGroup(0, sea.rippleGroup); cp.dispatchWorkgroups(WGSL.RIPPLE_N / 8, WGSL.RIPPLE_N / 8);
      sea.ripples.halve(cp);
      if (!skip.streak && sea.fitOut) { cp.setPipeline(sea.fitPipe); cp.setBindGroup(0, sea.fitFrame); cp.setBindGroup(1, sea.fitSea); cp.setBindGroup(2, sea.fitOut); cp.dispatchWorkgroups(sea.fitGroups); }
      if (!skip.streak && moving) { cp.setPipeline(sea.fitPipe); cp.setBindGroup(0, sea.fitFrame); cp.setBindGroup(1, sea.fitSea); cp.setBindGroup(2, sea.movingOut); cp.dispatchWorkgroups(Math.ceil(sea.movingCap / 64)); }
      if (!skip.streak) { cp.setPipeline(sea.surfPipe); cp.setBindGroup(0, sea.surfFrame); cp.setBindGroup(1, sea.surfSea); cp.setBindGroup(2, sea.surfOut); cp.dispatchWorkgroups(sea.surfGroups[0], sea.surfGroups[1]); }
    }
    if (!skip.sky && !skip.skylow) { cp.setPipeline(skyLow.pipe); cp.setBindGroup(0, skyLow.frame); cp.setBindGroup(1, skyLow.out); cp.dispatchWorkgroups(skyLow.size[0], skyLow.size[1]); }
    cp.end();
  }
  if (!shadow.placed) { // the ground map's rectangle never changes: each map's uniforms take it once
    for (const u of [shadow.uniform, shadow.beam[0].uniform]) device.queue.writeBuffer(u, 64, bytes, f + FRAME.GROUND_RECT, 48); // and its tiles, and the backdrop's
    shadow.placed = true;
  }
  for (let k = skip.cloudx10 ? 10 : 1; k > 0; k--) cloudShadePass(enc, skip.cloud || skip.cloudshade || !live.group ? 0 : clouds);
  if (mem32[(f + FRAME.SHADOW_FRESH) >> 2]) { // the car has gone far enough: redraw the moon's shadows around where it is now
    device.queue.writeBuffer(shadow.uniform, 0, bytes, f + FRAME.SHADOW_VP, 64);
    const sp = enc.beginRenderPass(stamped(shadow.pass, 'moon'));
    sp.setBindGroup(0, shadow.group); sp.setBindGroup(1, shadow.forest);
    const n = skip.shadow ? 0 : mem32[((f + FRAME.SHADOW_FRESH) >> 2) + 1], at = (f + FRAME.SHADOW) >> 2;
    unbound();
    for (let i = 0, last = -1; i < n; i++) {
      const c = at + i * 9;
      if (mem32[c] !== last) sp.setPipeline(shadow.pipes[last = mem32[c]]);
      drawCmd(sp, c);
    }
    sp.end();
    stats.shadowDraws = n;
  }
  stats.beamDraws = 0;
  if (water || (!skip.water && !waterSeen.cleared)) { // the lights' streaks on the sea, at half size: the water adds them where it shows, so what stands on it hides them
    const gp = enc.beginRenderPass(sea.glowPass); // (with no water in view, cleared once, so that none is left over for when it comes back)
    waterSeen.cleared = !water;
    if (water && !skip.streak) { gp.setBindGroup(0, frameGroup); gp.setBindGroup(1, groups.seaGlow); unbound(); run(gp, f, PIPE.STREAK, pipes.streakGlow); }
    if (water && !skip.streak && moving) { gp.setPipeline(pipes.streakGlow); gp.setBindGroup(1, groups.seaGlowMoving); gp.setVertexBuffer(0, sea.moving); gp.draw(48, moving) /* SEA_STREAK_SPANS quads a strip */; unbound(); }
    gp.end();
  }
  const nbeam = (f + FRAME.NBEAM) >> 2;
  if (mem32[nbeam + 2] || skip.beams) { // what stands in the headlamps' beams, when their map would not come out as it already is
    const beam = shadow.beam[0];
    device.queue.writeBuffer(beam.uniform, 0, bytes, f + FRAME.BEAM_VP, 64);
    const sp = enc.beginRenderPass(stamped(beam.pass, beam.name));
    sp.setBindGroup(0, beam.group); sp.setBindGroup(1, shadow.forest);
    unbound();
    const n = skip.beams ? 0 : mem32[nbeam], at = (f + FRAME.BEAM) >> 2; // ?skip=beams clears the map every frame
    for (let i = 0, last = -1; i < n; i++) {
      const c = at + i * 9;
      if (skip.beamtrees && mem32[c] >= PIPE.FOREST && mem32[c] <= PIPE.FOREST_TWIG) continue; // ?skip=beamtrees, to see what the trees cost there
      if (skip.beamsolid && (mem32[c] < PIPE.FOREST || mem32[c] > PIPE.FOREST_TWIG)) continue;
      if (mem32[c] !== last) sp.setPipeline(shadow.beamPipes[last = mem32[c]]);
      drawCmd(sp, c);
    }
    sp.end();
    stats.beamDraws = n;
  }
  if (rocketsOut && !skip.rocket) { // the flames big on the screen, small: the main pass lays them in where the scene leaves them showing
    const w = Math.ceil(renderW / WGSL.JET_LOW_DIV), h = Math.ceil(renderH / WGSL.JET_LOW_DIV);
    if (!rocket.flames || rocket.flames.width !== w || rocket.flames.height !== h) {
      rocket.flames?.destroy();
      rocket.flames = device.createTexture({ size: [w, h, ROCKET.LOW], format: 'rgba16float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
      rocket.flamePass = { colorAttachments: Array.from({ length: ROCKET.LOW }, (_, i) => i).map(i => ({ view: rocket.flames.createView({ dimension: '2d', baseArrayLayer: i, arrayLayerCount: 1 }), loadOp: 'clear', clearValue: [0, 0, 0, 0], storeOp: 'store' })) };
      rocket.flameGroup = device.createBindGroup({ layout: groupLayouts.jets, entries: [{ binding: 0, resource: rocket.flames.createView({ dimension: '2d-array' }) }] });
    }
    const jp = enc.beginRenderPass(stamped(rocket.flamePass, 'jets'));
    jp.setPipeline(pipes.jetsLow); jp.setBindGroup(0, frameGroup); jp.setVertexBuffer(0, rocket.jets); jp.draw(36, ROCKET.FLAMES);
    jp.end();
  }
  const cloudsLow = clouds && !skip.cloud ? cloudPass(enc, f, clouds) : null;
  const screen = present.texture, onScreen = present.view; // (the canvas, or a grab's texture, only at the end: presentTo)
  const canvasView = grabbing ? grabbing.createView() : context.getCurrentTexture().createView(); // (taken here, where it was before: the wait for it stays before the frame's passes)
  if (!up.scene) mainPass.colorAttachments[0][colourTarget ? 'resolveTarget' : 'view'] = onScreen;
  defer.cells = defer.read && !skip.terrain ? farCells(f, lists.main) : 0;
  defer.dunes = !!(dune.count && pipes.dune && defer.dunePrep && !skip.pack && !skip.dune && !skip.ddune); // the drifts lit in it too (WGSL.DUNE_GBUF)
  defer.plain = dplain && defer.plainRead && !skip.plain && !skip.far && !skip.calm && !skip.dplain ? farIndices(f, PIPE.PLAIN) / 3 : 0;
  defer.on = !!(defer.read && !skip.terrain && !skip.defer && (defer.cells > deferMin * renderW * renderH || defer.dunes || defer.plain > deferMin * renderW * renderH));
  if (!defer.on) defer.plain = 0;
  defer.dunes &&= defer.on;
  if (defer.on) {
    if (defer.size !== renderW + 'x' + renderH) deferTargets();
    const gp = enc.beginRenderPass(stamped(defer.prepPass, 'gbuf'));
    gp.setBindGroup(0, frameGroup); ground(gp, f, lists.main, null, true);
    if (defer.dunes) { gp.setPipeline(defer.dunePrep); gp.setBindGroup(1, dune.group); gp.setIndexBuffer(snow.mesh, 'uint16'); gp.drawIndexed(PACK_MESH[0][1], dune.count, PACK_MESH[0][0]); }
    if (defer.plain) { gp.setPipeline(defer.plainPrep); runFar(gp, f, PIPE.PLAIN); }
    gp.end();
    const lp = enc.beginRenderPass(stamped(defer.shadePass, 'defer'));
    lp.setBindGroup(0, frameGroup); lp.setBindGroup(2, defer.gbuf);
    for (let g = 0; g < (defer.dunes ? 3 : 2); g++) { lp.setBindGroup(1, g === 1 ? groups.outer : groups.terrain); lp.setPipeline(defer.shade[g]); lp.draw(3); }
    if (defer.plain) { lp.setBindGroup(1, defer.gbuf3); lp.setPipeline(defer.plainShade); lp.draw(3); }
    lp.end();
  }
  // the far road, by its list, over the far ground's depth where that was drawn (?skip=droad: lit as it is drawn)
  droad.on = !!(droad.read && groups.road && !skip.road && !skip.wet && !skip.droad && !skip.calm && !skip.far && farIndices(f, PIPE.ROAD) > droadMin * renderW * renderH);
  if (droad.on) {
    if (defer.size !== renderW + 'x' + renderH) deferTargets();
    enc.clearBuffer(droad.recs, 0, 16);
    const rp = enc.beginRenderPass(stamped(defer.on ? droad.prepPass : droad.prepBare, 'road list'));
    rp.setBindGroup(0, frameGroup); rp.setBindGroup(1, groups.road); rp.setBindGroup(2, droad.prepGroup); rp.setPipeline(droad.prep);
    runFar(rp, f, PIPE.ROAD); rp.end();
    const sp = enc.beginRenderPass(stamped(droad.shadePass, 'road light'));
    sp.setBindGroup(0, frameGroup); sp.setBindGroup(1, groups.road); sp.setBindGroup(2, droad.shadeGroup); sp.setPipeline(droad.shade); sp.draw(3); sp.end();
  }
  // ?split (for measuring, with bench): the main pass cut into passes at each part (cut), each timed on its own: what a
  // part costs is its pass's time (plus a cut's load and store of the targets)
  const cutting = splitting && benching && timing, held = { groups: [], vertex: [], index: null, pipe: null };
  let segment = 0; segNames = [];
  const open = desc => { const p = enc.beginRenderPass(desc); if (!cutting) return p;
    const bg = p.setBindGroup.bind(p), vb = p.setVertexBuffer.bind(p), ib = p.setIndexBuffer.bind(p), sp = p.setPipeline.bind(p);
    p.setBindGroup = (i, g) => { held.groups[i] = g; bg(i, g); }; p.setVertexBuffer = (i, b, ...r) => { held.vertex[i] = [b, ...r]; vb(i, b, ...r); };
    p.setIndexBuffer = (...a) => { held.index = a; ib(...a); }; p.setPipeline = q => { held.pipe = q; sp(q); };
    held.groups.forEach((g, i) => g && bg(i, g)); held.vertex.forEach((v, i) => v && vb(i, ...v)); if (held.index) ib(...held.index); if (held.pipe) sp(held.pipe);
    return p; };
  const segDesc = (last) => { const c = mainPass.colorAttachments[0], d = mainPass.depthStencilAttachment;
    return { occlusionQuerySet: mainPass.occlusionQuerySet, colorAttachments: [{ view: c.view, resolveTarget: last ? c.resolveTarget : undefined, loadOp: 'load', storeOp: last ? c.storeOp : 'store' }],
      depthStencilAttachment: { view: d.view, depthLoadOp: 'load', depthStoreOp: last ? 'discard' : 'store' } }; };
  const cut = (name, last = false) => { if (!cutting) return; pass.end(); segNames.push(name); pass = open(stamped(segDesc(last), 'seg' + segment++)); };
  let pass = cutting ? (segNames.push('first'), open(stamped({ ...mainPass, colorAttachments: [{ ...mainPass.colorAttachments[0], resolveTarget: undefined, storeOp: 'store' }],
    depthStencilAttachment: { ...mainPass.depthStencilAttachment, depthStoreOp: 'store' } }, 'main'))) : enc.beginRenderPass(stamped(mainPass, 'main'));
  if (skip.count) { // ?skip=count (for measuring): the triangles each pipeline draws in the main pass, by name, in report().tally
    const names = new Map(); for (const [k, v] of Object.entries(pipes)) if (v && typeof v === 'object' && !names.has(v)) names.set(v, isNaN(+k) ? k : Object.keys(PIPE).find(n => PIPE[n] === +k)); for (const [k, v] of calmOf) names.set(v, (names.get(k) || '?') + '~calm'); for (const [k, v] of farOf) names.set(v, (names.get(k) || '?') + '~far');
    let cur = '?'; const tally = self.__tally = {}; const sp = pass.setPipeline.bind(pass), di = pass.drawIndexed.bind(pass), dr = pass.draw.bind(pass);
    pass.setPipeline = q => { cur = names.get(q) || '?'; sp(q); }; pass.drawIndexed = (n, i = 1, ...r) => { tally[cur] = (tally[cur] || 0) + n * i / 3; di(n, i, ...r); }; pass.draw = (n, i = 1, ...r) => { tally[cur] = (tally[cur] || 0) + n * i / 3; dr(n, i, ...r); }; }
  const carPipe = mem32[(f + FRAME.CAR_WASHED) >> 2] ? pipes.carWashed : pipes.car, carBlend = mem32[(f + FRAME.CAR_WASHED) >> 2] ? pipes.carBlendWashed : pipes.carBlend;
  let draws = 0;
  pass.setBindGroup(0, frameGroup);
  if (!skip.plain) { if (defer.plain) pass.setBindGroup(1, defer.plainGroup); draws += run(pass, f, PIPE.PLAIN); cut('glazed'); draws += run(pass, f, PIPE.GLAZED); }
  cut('line'); draws += run(pass, f, PIPE.LINE);
  cut('live');
  if (liveI && !skip.live) { // all but the rotors' discs, which are drawn blended later
    pass.setPipeline(pipes.live); pass.setBindGroup(1, live.pose); pass.setVertexBuffer(0, live.vertex); pass.setIndexBuffer(live.index, 'uint32');
    pass.drawIndexed(blurCount ? blurFirst : liveI); if (blurCount && blurFirst + blurCount < liveI) pass.drawIndexed(liveI - blurFirst - blurCount, 1, blurFirst + blurCount); draws++;
  }
  cut('herd');
  if (!skip.live && !skip.herd) draws += herdDraw(pass);
  cut('car');
  if (!skip.car) {
    pass.setBindGroup(1, car.group); pass.setVertexBuffer(0, car.vertex); pass.setIndexBuffer(car.index, 'uint32');
    for (const r of car.runs) { if (skip.cabin && r.cabin || skip.body && !r.cabin) continue; pass.setPipeline(r.lens ? pipes.carLens : r.flight ? pipes.carFlight : r.cabin ? pipes.carCabin : carPipe); pass.drawIndexed(r.count, 1, r.first); } // ?skip=cabin, ?skip=body: what each costs
    draws += car.runs.length;
  }
  if (rocketsOut && !skip.rocket) { pass.setPipeline(carPipe); pass.setBindGroup(1, rocket.group); pass.setVertexBuffer(0, rocket.vertex); pass.setIndexBuffer(rocket.index, 'uint32'); pass.drawIndexed(rocket.count); draws++; }
  if (!skip.car && !skip.cockpit) { // the flight cockpit's hinges, rails, screws, wells and panes, behind the car's own faces
    pass.setPipeline(carPipe); pass.setVertexBuffer(0, cockpit.vertex); pass.setIndexBuffer(cockpit.index, 'uint32');
    for (const r of cockpit.runs) { pass.setBindGroup(1, r.group); pass.drawIndexed(r.solid, 1, r.first); draws++; }
  }
  if (wingsOut && !skip.car && !skip.wing) { // the wings' hinges, edges, tips and engines, the canopy's frames and bars
    pass.setPipeline(carPipe); pass.setVertexBuffer(0, wing.vertex); pass.setIndexBuffer(wing.index, 'uint32');
    for (const r of wing.runs) if (r.solid) { pass.setBindGroup(1, r.group); pass.drawIndexed(r.solid, 1, r.first); draws++; }
  }
  cut('wall');
  pass.setBindGroup(1, groups.wall); if (!skip.wall) draws += run(pass, f, PIPE.WALL);
  cut('tube');
  pass.setBindGroup(1, groups.tube); draws += run(pass, f, PIPE.TUBE); draws += run(pass, f, PIPE.TUBE_FLOOR);
  cut('tree');
  if (!skip.tree) draws += run(pass, f, PIPE.TREE);
  cut('fine');
  if (!skip.plain && !skip.fine) draws += run(pass, f, PIPE.FINE); // after the walls and kits: it drops samples, so its depth is not settled first
  cut('forest');
  if (!skip.forest) { pass.setBindGroup(1, groups.forest); draws += run(pass, f, PIPE.FOREST); }
  cut('lamp');
  draws += run(pass, f, PIPE.LAMP);
  cut('paving');
  pass.setBindGroup(1, groups.paving); draws += run(pass, f, PIPE.PAVING);
  cut('sign');
  if (groups.sign) { pass.setBindGroup(1, groups.sign); draws += run(pass, f, PIPE.SIGN); }
  cut('pane');
  pass.setBindGroup(1, groups.pane); draws += run(pass, f, PIPE.PANE);
  cut('roadmask');
  if (!skip.roadmask) { // the ground under the roads' and the trails' middles, into the depth: the terrain is not shaded there (WGSL.ROAD_MASK)
    // Near by only (none marked GFX_CMD_FAR): further off a road is a few pixels wide, and drawing it twice cost more than
    // the ground it kept from being shaded saved (bird's-eye view 3.0 ms, flying 2.6; in the street it saves 4.6). The
    // mask changes nothing that is seen, only what is shaded; ?skip=farmask masks the far ones too, as it was.
    const near = !skip.farmask;
    if (!skip.road) { pass.setBindGroup(1, groups.road); draws += run(pass, f, PIPE.ROAD, pipes.roadMask, null, lists.main, near); }
    if (groups.trail) { pass.setBindGroup(1, groups.trail); draws += run(pass, f, PIPE.TRAIL, pipes.trailMask, null, lists.main, near); }
  }
  cut('terrain');
  if (defer.on) pass.setBindGroup(2, defer.shaded);
  if (!skip.pack && !skip.dune) duneDraw(pass); // the drifts, which hide the ground under them
  if (!skip.terrain) draws += ground(pass, f, lists.main); // the ground in view
  cut('water');
  if (!skip.water) { pass.setBindGroup(1, groups.seaAfloat); draws += run(pass, f, PIPE.AFLOAT); pass.setBindGroup(1, groups.sea); pass.setPipeline(pipes.water); pass.beginOcclusionQuery(0); pass.draw(6); pass.endOcclusionQuery(); }
  if (groups.trail) { pass.setBindGroup(1, groups.trail); draws += run(pass, f, PIPE.TRAIL); } // under the roads, where a trail meets one
  if (droad.on) pass.setBindGroup(2, droad.readGroup);
  cut('road');
  pass.setBindGroup(1, groups.road); if (!skip.road) draws += run(pass, f, PIPE.ROAD, skip.wet ? pipes.roadDry : pipes[PIPE.ROAD]);
  draws += run(pass, f, PIPE.ZEBRA);
  cut('twig+fade');
  if (!skip.forest) { pass.setBindGroup(1, groups.forest); draws += run(pass, f, PIPE.FOREST_TWIG); draws += run(pass, f, PIPE.FOREST_FADE); }
  cut('impostor');
  if (!skip.forest && !skip.impostor && groups.impostor) { pass.setBindGroup(1, groups.impostor); draws += run(pass, f, PIPE.IMPOSTOR, pipes[PIPE.IMPOSTOR], skip.quad6 ? null : treeQuad); } // ?skip=quad6: six vertices a tree, as it was
  cut('tracks+grit');
  pass.setBindGroup(1, snow.group);
  if (!skip.tracks) {
    pass.setPipeline(pipes.tracks); pass.setVertexBuffer(0, snow.ribbon); pass.setIndexBuffer(snow.ribbonIndex, 'uint32'); pass.drawIndexed(Math.min(snow.ribbonCount, (snow.laid || 0) * 54));
    if (snow.runs) pass.drawIndexed(snow.runs * (SNOW_RUN - 1) * 54, 1, snow.ribbonCount); // and the traffic's
  }
  if (!skip.grit && grit.laid) { pass.setPipeline(pipes.grit); pass.setVertexBuffer(0, grit.buffer); pass.setIndexBuffer(grit.index, 'uint32'); pass.drawIndexed(Math.min(grit.count, grit.laid * (grit.across - 1) * 6)); draws++; }
  cut('dune+pack');
  if (!skip.pack) { pass.setPipeline(pipes.pack); packDraw(pass, f); }
  draws += 2;
  cut('sky+sea bits+label');
  if (!skip.sky && !skip.skydraw) { pass.setPipeline(pipes.sky); pass.setBindGroup(1, skyLow.group); pass.draw(3); }
  if (!skip.water) {
    if (sea.wakeLive) { pass.setPipeline(pipes.wake); pass.setVertexBuffer(0, sea.wake); pass.setIndexBuffer(sea.wakeIndex, 'uint16'); pass.drawIndexed(47 * 6); draws++; }
    if (sea.ringsLive) { pass.setPipeline(pipes.ring); pass.setVertexBuffer(0, sea.rings); pass.draw(6, 12); draws++; } // never sent, never drawn; a ring that repeats is drawn for good
    if (sea.dropsLive) { pass.setPipeline(pipes.drops); pass.setVertexBuffer(0, sea.drops); pass.draw(6, 384); draws++; }
  }
  if (groups.label) { pass.setBindGroup(1, groups.label); draws += run(pass, f, PIPE.LABEL); } // cut out and blended, so after all that is solid
  cut('shower+cloudover');
  if (!skip.shower) { pass.setPipeline(pipes.shower); pass.draw(6, WGSL.SHOWERS); draws++; } // far off over the fjord, so under the clouds
  if (cloudsLow && skip.cloudover) {} else if (cloudsLow) { pass.setPipeline(pipes.cloudOver); pass.setBindGroup(1, cloudsLow.group); pass.draw(6, cloudsLow.tileCount); draws++; } // the clouds' picture (cloudPass), over all that is behind them
  // The crowns' haze, blended over the land and the sky behind them: it writes no depth, so the clouds' picture, tested against
  // the depth, would lie over it wherever it hangs against the sky (a far cloud in front of a near tree's soft edge). Laid
  // after the clouds it is over them, as the near trees it belongs to are before any cloud.
  cut('haze');
  if (!skip.haze) { pass.setBindGroup(1, groups.forest); draws += run(pass, f, PIPE.HAZE); }
  cut('smoke');
  if (!skip.smoke && smoke.count) { pass.setPipeline(pipes[PIPE.SMOKE]); pass.setBindGroup(1, smoke.group); pass.setIndexBuffer(quadIndex, 'uint16'); pass.drawIndexed(6, smoke.count); draws++; } // over the clouds, which are always further off
  if (!skip.smoke && smoke.steaming) { pass.setPipeline(pipes.steam); pass.setBindGroup(1, smoke.group); pass.setIndexBuffer(quadIndex, 'uint16'); pass.drawIndexed(6, SMOKE.STEAM); draws++; }
  cut('blended car+glow+jets+spray+snow');
  if (blurCount && !skip.live) { pass.setPipeline(pipes.liveBlur); pass.setBindGroup(1, live.pose); pass.setVertexBuffer(0, live.vertex); pass.setIndexBuffer(live.index, 'uint32'); pass.drawIndexed(blurCount, 1, blurFirst); draws++; }
  if (!skip.car && !skip.glass) { pass.setPipeline(carBlend); pass.setBindGroup(1, car.group); pass.setVertexBuffer(0, car.vertex); pass.setIndexBuffer(car.index, 'uint32'); pass.drawIndexed(car.blended, 1, car.solid); draws++; }
  if (!skip.car && !skip.cockpit) { // and its panes
    pass.setPipeline(carBlend); pass.setVertexBuffer(0, cockpit.vertex); pass.setIndexBuffer(cockpit.index, 'uint32');
    for (const r of cockpit.runs) if (r.blended) { pass.setBindGroup(1, r.group); pass.drawIndexed(r.blended, 1, r.first + r.solid); draws++; }
  }
  if (wingsOut && !skip.car && !skip.wing) { // and the canopy's glass
    pass.setPipeline(carBlend); pass.setVertexBuffer(0, wing.vertex); pass.setIndexBuffer(wing.index, 'uint32');
    for (const r of wing.runs) if (r.blended) { pass.setBindGroup(1, r.group); pass.drawIndexed(r.blended, 1, r.first + r.solid); draws++; }
  }
  if (!skip.glow) draws += run(pass, f, PIPE.GLOW);
  if (liveGlow) { pass.setPipeline(pipes[PIPE.GLOW]); pass.setVertexBuffer(0, live.glow); pass.draw(6, liveGlow); draws++; }
  if (rocketsOut && !skip.rocket) { pass.setPipeline(pipes.jets); pass.setBindGroup(1, rocket.flameGroup); pass.setVertexBuffer(0, rocket.jets); pass.draw(36, ROCKET.FLAMES); draws++; }
  if (blownLive && !skip.rocket) { pass.setPipeline(pipes.blown); pass.setVertexBuffer(0, rocket.puffs); pass.draw(6 * WGSL.POWDER.n, ROCKET.PUFFS); /* POWDER.n particles a grain the wash threw, one sprite for the rest */ draws++; }
  if (!skip.spray) { pass.setPipeline(pipes.spray); pass.setBindGroup(1, snow.group); pass.setVertexBuffer(0, snow.spray); pass.setIndexBuffer(quadIndex, 'uint16'); pass.drawIndexed(6, 1536); draws++; }
  if (!skip.snow && snowBuffer) snowDraw(pass, f);
  cut('end', true);
  pass.end();
  if (rocketsOut && !skip.rocket && !skip.shimmer) shimmer(enc, f, up.scene || screen); // ?skip=shimmer leaves it out
  if (up.scene) { // the frame upscaled onto the canvas
    if (!up.rcas) up.easuPass.colorAttachments[0].view = onScreen;
    let p = enc.beginRenderPass(stamped(up.easuPass, 'easu'));
    p.setPipeline(up.easu); p.setBindGroup(0, up.easuGroup); p.draw(3); p.end();
    if (up.rcas) {
      up.rcasPass.colorAttachments[0].view = onScreen;
      p = enc.beginRenderPass(stamped(up.rcasPass, 'rcas'));
      p.setPipeline(up.rcas); p.setBindGroup(0, up.rcasGroup); p.draw(3); p.end();
    }
  }
  let read = null;
  if (!skip.water) for (const r of waterSeen.reads) if (!r.busy) { read = r; break; }
  if (read) { enc.resolveQuerySet(waterSeen.set, 0, 1, waterSeen.resolve, 0); enc.copyBufferToBuffer(waterSeen.resolve, 0, read.buffer, 0, 8); read.busy = true; }
  if (benching && timing) { enc.resolveQuerySet(timing.set, 0, STAMPS.length * 2, timing.resolve, 0); enc.copyBufferToBuffer(timing.resolve, 0, timing.read, 0, STAMPS.length * 16); }
  presentTo(enc, canvasView, !!grabbing);
  if (filmFrames && still && played && !grabbing) filmRecord(enc, screen);
  const t3 = performance.now();
  submitted[0] = enc.finish();
  device.queue.submit(submitted);
  if (read) read.buffer.mapAsync(GPUMapMode.READ).then(read.done, read.fail);
  const t4 = performance.now(); // the frame's time on this thread: the C, what is sent to buffers and textures, the passes' encoding, finish and submit
  stats.frames++; stats.draws = draws + 5; stats.cpu += t4 - t0;
  stats.split[0] += t1 - t0; stats.split[1] += t2 - t1; stats.split[2] += t3 - t2; stats.split[3] += t4 - t3;
  const k = (stats.pn++ & 4095) * 4; // (no strings or objects made a frame: bench() reads the last frame's back from here)
  stats.parts[k] = t1 - t0; stats.parts[k + 1] = t2 - t1; stats.parts[k + 2] = t3 - t2; stats.parts[k + 3] = t4 - t3;
}

// For measurement: every write to a buffer or texture after loading, by what it is written to
function countUploads() {
  const names = new Map([[frameBuffer, 'frame uniforms'], [snow.spray, 'spray'], [snow.ribbon, 'ribbon'], [grit.buffer, 'grit'], [snow.bank, 'banks'], [rocket.uniform, 'rocket'],
    [rocket.jets, 'jets'], [rocket.puffs, 'blown snow'], [live.vertex, 'live mesh'], [live.glow, 'live glow'], [live.cloud, 'clouds'], [forestList, 'forest list'],
    [sea.wake, 'wake'], [sea.rings, 'rings'], [sea.drops, 'drops'], [car.dials, 'dash'], [car.panels, 'dash']]);
  const q = device.queue, wb = q.writeBuffer.bind(q), wt = q.writeTexture.bind(q), ce = q.copyExternalImageToTexture.bind(q);
  const add = (target, bytes) => { const k = names.get(target) || 'other'; const u = stats.up[k] ||= { bytes: 0, calls: 0 }; u.bytes += bytes; u.calls++; };
  q.writeBuffer = (b, o, d, d0 = 0, size) => { add(b, size ?? (d.byteLength - d0 * (d.BYTES_PER_ELEMENT || 1))); return wb(b, o, d, d0, size); };
  q.writeTexture = (dst, d, layout, size) => { add(dst.texture, (layout.bytesPerRow || 0) * (size[1] || size.height || 1)); return wt(dst, d, layout, size); };
  q.copyExternalImageToTexture = (src, dst, size) => { add(dst.texture, size[0] * size[1] * 4); return ce(src, dst, size); };
}

function report() {
  const n = Math.min(stats.n, 4096), t = Array.from(stats.times.subarray(0, n)).sort((a, b) => a - b);
  const q = p => t.length ? +t[Math.min(t.length - 1, Math.floor(p * t.length))].toFixed(2) : 0;
  const pn = Math.min(stats.pn, 4096), part = i => { const a = Array.from({ length: pn }, (_, k) => stats.parts[k * 4 + i]).sort((a, b) => a - b), at = p => pn ? +a[Math.min(pn - 1, Math.floor(p * pn))].toFixed(2) : 0;
    return { p50: at(0.5), p99: at(0.99), max: at(1) }; };
  const spikes = Object.fromEntries(['c', 'upload', 'encode', 'submit'].map((k, i) => [k, part(i)])); stats.pn = 0;
  const out = { spikes, hdr, canvasFormat, frames: stats.frames, draws: stats.draws, cpuPerFrame: +(stats.cpu / Math.max(1, stats.frames)).toFixed(3), p50: q(0.5), p95: q(0.95), p99: q(0.99), max: q(1),
    cpu: Object.fromEntries(['c', 'upload', 'encode', 'submit'].map((k, i) => [k, +(stats.split[i] / Math.max(1, stats.frames)).toFixed(3)])), beamDraws: stats.beamDraws, liveGlow: stats.liveGlow, shipStreaks: stats.shipStreaks, lights: stats.lights, lightsWorked: stats.lightsWorked,
    wasmMemory: hs.memory.buffer.byteLength, textures: texturesMB(), size: [canvas.width, canvas.height], indices: stats.indices, herd: stats.herd, levels: stats.levels && stats.levels.map(x => Math.round(x / Math.max(1, stats.frames))),
    uploads: Object.fromEntries(Object.entries(stats.up).map(([k, u]) => [k, { kbPerFrame: +(u.bytes / 1024 / Math.max(1, stats.frames)).toFixed(1), callsPerFrame: +(u.calls / Math.max(1, stats.frames)).toFixed(2) }])) };
  stats.frames = stats.n = stats.cpu = 0; stats.split.fill(0); stats.up = {}; stats.levels = null;
  out.photo = { late: photoPool.late, lates: photoPool.lates, used: photoPool.used, caps: photoPool.caps, loading: photoPool.flight };
  out.played = played; // frames of play under ?still: the look tests wait for all still * 60 of them (tools/look/run.mjs)
  return out;
}

// For measurement: n frames back to back, each timed from its start until the GPU has finished it. No vsync,
// no compositor: what a frame costs, CPU and GPU together.
async function bench(n) {
  benching = true;
  const t = [], gpu = STAMPS.map(() => []), span = [], was = STAMPS.map(() => 0n), worst = [];
  report();
  for (let i = 0; i < n + 20; i++) {
    const t0 = performance.now();
    frame(t0);
    await device.queue.onSubmittedWorkDone();
    if (i >= 20) t.push(performance.now() - t0);
    if (timing) { // nanoseconds at the start and end of each pass; a pass that was not drawn this frame keeps the stamps it had
      await timing.read.mapAsync(GPUMapMode.READ);
      const ns = new BigInt64Array(timing.read.getMappedRange());
      let from = 0n, to = 0n, drawn = '';
      STAMPS.forEach((_, k) => {
        const a = ns[k * 2], b = ns[k * 2 + 1];
        if (a === was[k]) return;
        was[k] = a; drawn += STAMPS[k] + ' ' + (Number(b - a) / 1e6).toFixed(1) + ',';
        if (i >= 20) gpu[k].push(Number(b - a) / 1e6);
        if (STAMPS[k] === 'present') return; // (timed, but out of the span: it waits on the canvas, as the frame's first pass did before it was there)
        if (!from || a < from) from = a;
        if (b > to) to = b;
      });
      if (i >= 20 && splitting) { const m = STAMPS.indexOf('main') * 2, e0 = ns[m + 1]; (bench.ends ||= []).push(STAMPS.map((n, k) => n.startsWith('seg') && ns[k * 2 + 1] > e0 ? Number(ns[k * 2 + 1] - e0) / 1e6 : null)); }
      if (i >= 20) { span.push(Number(to - from) / 1e6); worst.push([i - 20, +(Number(to - from) / 1e6).toFixed(2), +t.at(-1).toFixed(2), drawn, +stats.parts[((stats.pn - 1) & 4095) * 4].toFixed(2), +stats.parts[((stats.pn - 1) & 4095) * 4 + 1].toFixed(2)]); }
      timing.read.unmap();
    }
  }
  benching = false; last = 0;
  const q = (a, p) => { a.sort((x, y) => x - y); return a.length ? +a[Math.min(a.length - 1, Math.floor(p * a.length))].toFixed(2) : null; };
  const r = report();
  const ends = bench.ends || []; bench.ends = null;
  const segEnds = STAMPS.map((n, k) => n.startsWith('seg') ? q(ends.map(e => e[k]).filter(x => x != null), 0.5) : null);
  return { segEnds, segNames, n, p50: q(t, 0.5), p95: q(t, 0.95), p99: q(t, 0.99), max: q(t, 1), draws: stats.draws, beamDraws: r.beamDraws, lights: stats.lights, cloudLights: stats.cloudLights, lightsWorked: stats.lightsWorked, cpuPerFrame: r.cpuPerFrame, cpu: r.cpu, herd: r.herd, levels: r.levels,
    gpu: { span: q(span, 0.5), smin: q(span, 0), mmin: q(gpu[STAMPS.indexOf('main')], 0), p50: q(gpu[STAMPS.indexOf('main')], 0.5), beams: q(gpu[STAMPS.indexOf('beam0')], 0.5),
      ...Object.fromEntries(STAMPS.map((name, k) => [name, q(gpu[k], 0.5)])) },
    worst: worst.sort((a, b) => b[1] - a[1]).slice(0, 4), spikes: { span: [q(span, 0.95), q(span, 0.99), q(span, 1)], frame: [q(t, 0.95), q(t, 0.99), q(t, 1)],
      ...Object.fromEntries(STAMPS.map((name, k) => [name, gpu[k].length ? [gpu[k].length, q(gpu[k], 1)] : null]).filter(e => e[1])) } };
}

// For comparing play frame by frame (?still=S&film=N): each of the first N frames of play, as it is drawn, hashed a tile
// of FILM_TILE pixels square at a time on the GPU, with the sums of its red, green and blue (0-255 a texel), into a buffer
// kept for the run (some 70 KB a frame at 1440 x 757),
// read back at the end (__next.film). Two runs whose frames hash alike are alike pixel for pixel; a tile that differs is
// where something was not there in time (a pop). The frame's own work is unchanged but for a copy and a small pass.
const FILM_TILE = 16, FILM_WGSL = `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@group(0) @binding(2) var<uniform> p: vec4u; // where this frame's hashes go, tiles across, width, height
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= p.y * ((p.w + ${FILM_TILE - 1}u) / ${FILM_TILE}u)) { return; }
  let t = vec2u(g.x % p.y, g.x / p.y) * ${FILM_TILE}u;
  var h = 2166136261u;
  var sum = vec3u(0u);
  for (var y = 0u; y < ${FILM_TILE}u; y++) {
    for (var x = 0u; x < ${FILM_TILE}u; x++) {
      let q = t + vec2u(x, y);
      if (q.x < p.z && q.y < p.w) {
        let c = textureLoad(src, q, 0);
        h = (h ^ pack4x8unorm(c)) * 16777619u;
        sum += vec3u(round(c.rgb * 255.0));
      }
    }
  }
  let o = (p.x + g.x) * 4u;
  out[o] = h; out[o + 1u] = sum.r; out[o + 2u] = sum.g; out[o + 3u] = sum.b;
}`;
let film = null, filmFrames = 0; // ?film=N
let birdStart = false, introGo = false; // ?bird: out over the car the moment the play starts (the streaming check's bird's-eye); the opening zoom set going
let popFrames = 0; // ?pop=60: a pop made on purpose, to see the streaming check fail: the sea left out of the first 60 frames of play
function filmRecord(enc, screen) {
  if (!film) {
    const w = canvas.width, h = canvas.height, across = Math.ceil(w / FILM_TILE), down = Math.ceil(h / FILM_TILE), n = filmFrames;
    const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code: FILM_WGSL }), entryPoint: 'main' } });
    const tex = device.createTexture({ size: [w, h], format, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    const buf = device.createBuffer({ size: n * across * down * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const uni = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    film = { n, w, h, across, down, tex, buf, uni, pipeline, frames: 0, bad: false,
      group: device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: tex.createView() }, { binding: 1, resource: { buffer: buf } }, { binding: 2, resource: { buffer: uni } }] }) };
  }
  if (played <= film.frames || film.frames >= film.n) return;
  if (canvas.width !== film.w || canvas.height !== film.h || played !== film.frames + 1) { film.bad = true; return; } // resized, or a frame of play not drawn
  enc.copyTextureToTexture({ texture: screen }, { texture: film.tex }, [film.w, film.h]);
  device.queue.writeBuffer(film.uni, 0, new Uint32Array([film.frames * film.across * film.down, film.across, film.w, film.h]));
  const pass = enc.beginComputePass();
  pass.setPipeline(film.pipeline); pass.setBindGroup(0, film.group); pass.dispatchWorkgroups(Math.ceil(film.across * film.down / 64)); pass.end();
  film.frames++;
}
async function filmRead() {
  if (!film) return { frames: 0 };
  const size = film.frames * film.across * film.down * 16, read = device.createBuffer({ size: Math.max(4, size), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  if (size) enc.copyBufferToBuffer(film.buf, 0, read, 0, size);
  device.queue.submit([enc.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const data = read.getMappedRange().slice(0, size);
  read.unmap(); read.destroy();
  return { frames: film.frames, across: film.across, down: film.down, bad: film.bad, data };
}

// For comparing pictures: one frame drawn into a texture of its own, not the canvas's, and its pixels read back.
let grabbing = null;
async function grab() {
  const w = canvas.width, h = canvas.height, row = (w * 4 + 255) & ~255;
  const texture = device.createTexture({ size: [w, h], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST }); // (the SDR picture, whatever the screen)
  const buffer = device.createBuffer({ size: row * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  benching = true; grabbing = texture;
  try { frame(performance.now()); } finally { benching = false; grabbing = null; last = 0; }
  const enc = device.createCommandEncoder();
  enc.copyTextureToBuffer({ texture }, { buffer, bytesPerRow: row }, [w, h]);
  device.queue.submit([enc.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const from = new Uint8Array(buffer.getMappedRange()), out = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y++) out.set(from.subarray(y * row, y * row + w * 4), y * w * 4);
  buffer.unmap(); buffer.destroy(); texture.destroy();
  return out.buffer;
}

// For measuring HDR (?hdr=1): one frame drawn as the screen shows it, the float picture read back before it goes onto the
// canvas, and of each pixel's brightest channel: the most, the 99.9th and 99th percentiles, the share over 1 (over SDR
// white) and over 2, and where the ten brightest are; and given a box ([x0, y0, x1, y1], shares of the frame), in it the
// most and the mean (a lamp's lens, say, measured as it comes nearer)
async function sceneStats(box) {
  const w = canvas.width, h = canvas.height, row = (w * 8 + 255) & ~255;
  const buffer = device.createBuffer({ size: row * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  benching = true;
  try { frame(performance.now()); } finally { benching = false; last = 0; }
  const enc = device.createCommandEncoder();
  enc.copyTextureToBuffer({ texture: present.texture }, { buffer, bytesPerRow: row }, [w, h]);
  device.queue.submit([enc.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const u = new Uint16Array(buffer.getMappedRange()), half = x => { const e = (x >> 10) & 31, m = x & 1023, v = e ? (1 + m / 1024) * 2 ** (e - 15) : m / 1024 * 2 ** -14; return x & 0x8000 ? -v : v; };
  const m = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { const k = y * row / 2 + x * 4; m[y * w + x] = Math.max(half(u[k]), half(u[k + 1]), half(u[k + 2])); }
  buffer.unmap(); buffer.destroy();
  const s = Float32Array.from(m).sort(), q = p => +s[Math.min(s.length - 1, Math.floor(p * s.length))].toFixed(4);
  let over1 = 0, over2 = 0; for (const v of m) { over1 += v > 1; over2 += v > 2; }
  const top = []; const lim = s[Math.max(0, s.length - 10)];
  for (let i = 0; i < m.length && top.length < 10; i++) if (m[i] >= lim) top.push([i % w, Math.floor(i / w), +m[i].toFixed(3)]);
  let inBox = null;
  if (Array.isArray(box)) {
    const [x0, y0, x1, y1] = [box[0] * w, box[1] * h, box[2] * w, box[3] * h].map(Math.round);
    let most = 0, sum = 0, n = 0;
    for (let y = Math.max(y0, 0); y <= Math.min(y1, h - 1); y++) for (let x = Math.max(x0, 0); x <= Math.min(x1, w - 1); x++) { const v = m[y * w + x]; most = Math.max(most, v); sum += v; n++; }
    inBox = { max: +most.toFixed(4), mean: +(sum / Math.max(n, 1)).toFixed(4), n };
  }
  return { hdr, peak: WGSL.HDR.peak, max: q(1), p999: q(0.999), p99: q(0.99), over1: +(over1 / m.length * 100).toFixed(4), over2: +(over2 / m.length * 100).toFixed(4), top, box: inBox };
}

onmessage = ({ data }) => {
  if (data.init) init(data.init).catch(e => { builders.forEach(w => w.terminate()); post({ error: String(e.stack || e) }); });
  else if (data.resize) { resize(...data.resize); if (data.ratio) { wantedRatio = data.ratio; hs?.hs_set(SET.PIXEL_RATIO, data.ratio * SCALE); } }
  else if (data.event) {
    hs?.hs_event(EVENT[data.event], data.arg | 0);
    if (data.event === 'INTRO') introGo = true; // ?still counts the play from the zoom's start, so the zoom is filmed too
    if (data.event === 'START' && birdStart) hs?.hs_event(EVENT.BIRD, birdStart); // in the same step: no frame between them
  }
  else if ('keys' in data) keys = data.keys;
  else if (data.pad) pad.set(data.pad);
  else if (data.setting === 'gusts') hs?.hs_set(SET.GUSTS, data.value ? 1 : 0);
  else if (data.setting === 'haze') hs?.hs_set(SET.HAZE, +data.value);
  else if (data.setting === 'traffic') hs?.hs_set(SET.TRAFFIC, data.value ? 1 : 0);
  else if (data.meet && data.meet[0] in SET) hs?.hs_set(SET[data.meet[0]], +data.meet[1]);
  else if (data.bird) hs?.hs_bird(...data.bird); // the bird's-eye view moved: op, then its numbers (wasm.c, hs_bird)
  else if (data.probe) Object.assign(probe, data.probe);
  else if (data.act && hs) { const at = hs.hs_act(...data.act) >> 2; post({ acted: Array.from(new Float32Array(hs.memory.buffer, at << 2, 5)), act: data.act[0] }); }
  else if ('follow' in data) hs?.hs_follow(data.follow);
  else if ('rate' in data) rate = Math.max(1, Math.min(16, data.rate | 0));
  else if ('skip' in data) { skip = {}; for (const k of data.skip.split(',')) skip[k] = true; skipDraws = data.skip.includes(':'); }
  else if (data.stats) post({ stats: { ...report(), tally: self.__tally } });
  else if (data.film) filmRead().then(r => post({ stats: r }, r.data ? [r.data] : []), e => post({ error: String(e.stack || e), stats: null }));
  else if (data.look) { lookAgain(data.look); post({ stats: { looked: true } }); }
  else if (data.sceneStats) sceneStats(data.sceneStats).then(r => post({ stats: r }), e => post({ error: String(e.stack || e), stats: {} }));
  else if (data.grab) grab().then(b => post({ stats: { width: canvas.width, height: canvas.height, format: 'rgba8unorm', data: b } }, [b]), e => post({ error: String(e.stack || e), stats: null }));
  else if (data.bench) bench(data.bench).then(r => post({ stats: r }), e => post({ error: String(e.stack || e), stats: {} }));
};
