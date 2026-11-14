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
// game.h) and Float32 [1..6] = the gamepad (struct game_pad), written by the main thread; bytes 64..127 = struct game_hud, written here every frame.
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
// Textures: photo tiles decode here (createImageBitmap, off the main thread by construction) into one array
// texture, so the terrain is a single draw; the terrain itself has no vertex buffer, only a height texture.

import * as WGSL0 from './shaders.js';
import { facadeTextures, facadeStyles, displayAtlas, roadTexture, noiseTexture, floePattern, snowFlakes, tunnelTextures, plateTexture, dashPainter, pavingTextures, cloudNoise, trailTexture, snowRelief } from './textures.js';

let WGSL = WGSL0; // or, with ?off=a,b on the page, the shaders made with those parts changed: for looking into the frame
const NONE = 0xffffffff;
const PIPE = { ROAD: 0, PLAIN: 1, WALL: 2, TREE: 3, LAMP: 4, GLOW: 5, TUBE: 6, TUBE_FLOOR: 7, PAVING: 8, ZEBRA: 9, SIGN: 10, TRAIL: 11, LINE: 12,
  FOREST: 13, FOREST_FADE: 14, FOREST_TWIG: 15, HAZE: 16, PANE: 17, LABEL: 18, AFLOAT: 19, SMOKE: 20, STREAK: 21 }; // enum gfx_pipe
const BUF = { VERTEX: 0, INDEX: 1, INSTANCE: 2, LAMP_MAP: 3, SIGN_ATLAS: 4, BANK_VERTEX: 5, BANK_INDEX: 6, TRAIL_PRINTS: 7, TREE_MODELS: 8, TINT_MAP: 9, LABEL_ATLAS: 10, SEA_MAP: 11, FOREST: 12 };
const FRAME = { CMD: 36 }, ITEM_BYTES = 64; // struct gfx_cmd, struct gfx_item; // the rest of struct frame's layout comes from hs_layout()
const HUD_BYTES = 64, SHARED_HUD = 64;
const EVENT = { START: 1, CAM_NEXT: 2, CAM_SET: 3, FLY_TOGGLE: 4, BACK_TO_ROAD: 5 };
const SET = { AUTO_DRIVE: 0, AUTO_FLY: 1, ORBIT: 2, TOP: 3, CAM: 4, PIXEL_RATIO: 5, TIME: 6, GUSTS: 7 };
let SAMPLES = 4; // 4x multisampling, or none with ?aa=1 (WebGPU has no counts between)
const WAKE_BYTES = 96 * 28, RINGS_BYTES = 12 * 16, DROPS_BYTES = 384 * 28; // and what the car stirs up: ribbon vertices, the rings and the jets' dimples, droplets
const SEA_BYTES = 96, WAVES_BYTES = 1792; // struct water_frame: the sea's uniforms, then its ripple tiles' waves
const STAMPS = ['maps', 'moon', 'beam0', 'main']; // the passes bench() times on the GPU
const FOREST_WORDS = 6; // the list of trees to draw holds at most this many words a tree (forest.h)
const SHADOW_BYTES = 80; // a shadow map's uniforms: world to map, and the ground map's rectangle, which its trees stand on

let device, context, format, canvas, hs, shared, keysView, padView, hudView, keys = 0, pad = new Float32Array(6);
let colourTarget, depthTarget, frameBuffer, frameGroup, pipes = {}, groups = {}, snowBuffer, quadIndex;
const snow = {}; // the snow the car drives through: field texture, lattice, spray, ribbon, and the banks it can flatten
const shadow = {}; // the moon's shadow map and what draws into it
const sea = { wakeUntil: -1, dropsUntil: -1 }; // the fjord: its uniforms, the ripple tiles and the sky panorama redrawn each frame, the baked maps
const live = {}; // what moves in the sky: the mesh of the tower's lights and Father Christmas, their sprites, the clouds' puffs
let car = {}; // the car: its mesh, its uniform block, the instrument cluster's texture and what it last showed
let rowIndex, outerRowIndex;
const rocket = {}; // the jets in the wheels (rocket.h): the nozzles' and struts' mesh and uniform block, the flames' instances, the ring of blown-snow puffs
const ROCKET = { JET: 32, PUFF: 32, PUFFS: 4096 }; // struct rocket_jet, struct drift_grain, DRIFT_GRAINS
const buffers = [];
let snowCount = 0, timing = null, forestList = null, forestAt = 0; // the buffer the forest's list is kept in, and where the C makes it
let skip = {}; // ?skip=terrain,sky,... leaves parts of the frame out, to see what each costs
const marks = { start: performance.now() }, stats = { frames: 0, draws: 0, cpu: 0, split: [0, 0, 0, 0], times: new Float32Array(4096), n: 0, up: {} };
const mark = name => { marks[name] = performance.now() - marks.start; };
const post = (m, t) => postMessage(m, t || []);

/* ---- small WebGPU helpers ---- */

const mips = (w, h) => 1 + Math.floor(Math.log2(Math.max(w, h)));
let mipPipes = new Map(), mipSampler;
function mipPipe(f) {
  if (!mipPipes.has(f)) {
    const module = device.createShaderModule({ code: WGSL.MIP });
    mipPipes.set(f, device.createRenderPipeline({ layout: 'auto', vertex: { module }, fragment: { module, targets: [{ format: f }] } }));
    mipSampler ||= device.createSampler({ minFilter: 'linear', magFilter: 'linear' });
  }
  return mipPipes.get(f);
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
function arrayTexture(images, fmt) {
  const { width, height } = images[0];
  const texture = device.createTexture({
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

function dataTexture(fmt, width, height, data, bytesPerRow) {
  const texture = device.createTexture({ size: [width, height], format: fmt, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  device.queue.writeTexture({ texture }, data, { bytesPerRow }, [width, height]);
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
  glow: [{ arrayStride: 28, stepMode: 'instance', attributes: [
    { shaderLocation: 0, offset: 0, format: 'float32x4' }, { shaderLocation: 1, offset: 16, format: 'unorm8x4' }, { shaderLocation: 2, offset: 20, format: 'float32x2' }] }],
  // struct avensis_vtx: position, normal and part, linear colour, texture coordinates, material and lamp
  car: [{ arrayStride: 40, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }, { shaderLocation: 1, offset: 12, format: 'snorm8x4' },
    { shaderLocation: 2, offset: 16, format: 'float32x3' }, { shaderLocation: 3, offset: 28, format: 'float32x2' }, { shaderLocation: 4, offset: 36, format: 'uint8x4' }] }],
  cloud: [{ arrayStride: 24, stepMode: 'instance', attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x4' }, { shaderLocation: 1, offset: 16, format: 'float32x2' }] }],
  forest: [{ arrayStride: 4, stepMode: 'instance', attributes: [{ shaderLocation: 0, offset: 0, format: 'uint32' }] }], // a word of the forest's list: tree and level
  snow: [{ arrayStride: 20, stepMode: 'instance', attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x4' }, { shaderLocation: 1, offset: 16, format: 'float32' }] }],
};
const BLEND = {
  over: { color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' }, alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' } },
  add: { color: { srcFactor: 'src-alpha', dstFactor: 'one' }, alpha: { srcFactor: 'zero', dstFactor: 'one' } },
  laid: { color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' }, alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' } },
};
const VF = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, FS = GPUShaderStage.FRAGMENT;
let frameLayout;
const groupLayouts = {};

async function makePipelines() {
  frameLayout = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: VF, buffer: {} },
    { binding: 1, visibility: VF, texture: {} },
    { binding: 2, visibility: VF, texture: {} },
    { binding: 3, visibility: FS, texture: {} },
    { binding: 4, visibility: VF, sampler: {} },
    { binding: 5, visibility: FS, sampler: {} },
    { binding: 6, visibility: FS, texture: { sampleType: 'depth' } },
    { binding: 7, visibility: FS, sampler: { type: 'comparison' } },
    { binding: 8, visibility: FS, texture: { sampleType: 'depth', viewDimension: '2d-array' } },
    { binding: 9, visibility: FS, texture: {} },
    { binding: 10, visibility: VF, texture: {} },
    { binding: 11, visibility: FS, buffer: { type: 'read-only-storage' } }, // the aurora's rows (WGSL.SKY_MAPS)
  ] });
  const tex = (binding, viewDimension = '2d', sampleType = 'float', visibility = FS) => ({ binding, visibility, texture: { viewDimension, sampleType } });
  const samp = binding => ({ binding, visibility: FS, sampler: {} });
  groupLayouts.terrain = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: VF, buffer: {} }, tex(1, '2d', 'unfilterable-float', GPUShaderStage.VERTEX), tex(2, '2d-array'), samp(3)] });
  groupLayouts.wall = device.createBindGroupLayout({ entries: [tex(0, '2d-array'), tex(1, '2d-array'), samp(2), tex(3, '2d-array'), tex(4, '2d-array')] });
  groupLayouts.road = device.createBindGroupLayout({ entries: [tex(0, '2d-array'), samp(1)] });
  groupLayouts.single = device.createBindGroupLayout({ entries: [tex(0)] });
  groupLayouts.live = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: {} }] });
  groupLayouts.sea = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: VF, buffer: {} }, ...[1, 2, 3, 4, 5, 6, 7].map(b => tex(b)), samp(8), samp(9), samp(10),
    tex(11, '2d', 'float', VF)] });
  groupLayouts.snow = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: VF, buffer: {} }, tex(1, '2d', 'float', VF),
    tex(2, '2d', 'unfilterable-float', GPUShaderStage.VERTEX), { binding: 3, visibility: VF, sampler: {} }, tex(4, '2d', 'uint', GPUShaderStage.VERTEX)] });
  groupLayouts.forest = device.createBindGroupLayout({ entries: [tex(0, '2d', 'unfilterable-float', GPUShaderStage.VERTEX),
    { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }] });
  groupLayouts.car = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: VF, buffer: {} }, tex(1), tex(2, '2d-array'), tex(3)] });

  const make = (code, { buffers: vb = [], group, blend, depthWrite = true, cull = 'back', bias = 0, slope = 0, topology = 'triangle-list' }) => {
    const module = device.createShaderModule({ code: WGSL.PRELUDE + code });
    module.getCompilationInfo().then(info => { for (const m of info.messages) if (m.type === 'error') post({ error: `shader, line ${m.lineNum}: ${m.message}\n${(WGSL.PRELUDE + code).split('\n')[m.lineNum - 1]}` }); });
    return device.createRenderPipelineAsync({
      layout: device.createPipelineLayout({ bindGroupLayouts: group ? [frameLayout, groupLayouts[group]] : [frameLayout] }),
      vertex: { module, buffers: vb },
      fragment: { module, targets: [{ format, blend }] },
      primitive: { topology, cullMode: cull, frontFace: 'ccw' },
      // reversed depth: nearer is greater. Sky and sea level sit exactly on values the land may share.
      depthStencil: { format: 'depth32float', depthWriteEnabled: depthWrite, depthCompare: 'greater-equal', depthBias: bias, depthBiasSlopeScale: slope },
      multisample: { count: SAMPLES },
    });
  };
  const made = await Promise.all([
    make(WGSL.terrain(false, false), { group: 'terrain' }),
    make(WGSL.terrain(true, false), { group: 'terrain' }), // for the chunks a tunnel portal opens
    make(WGSL.terrain(false, false), { group: 'terrain', cull: 'none' }), // the backdrop is seen from below too, where it dips under the sea
    make(WGSL.terrain(false, true), { group: 'terrain', cull: 'none' }), // and where the town's edge crosses it, the town is cut out
    make(WGSL.PLAIN, { buffers: LAYOUT.plain }),
    make(WGSL.WALL, { buffers: LAYOUT.textured, group: 'wall' }),
    make(WGSL.TREE, { buffers: LAYOUT.instanced, cull: 'none' }),
    make(WGSL.LAMP, { buffers: LAYOUT.instanced }),
    // the ribbon lies 12 cm over the terrain; the bias keeps it on top where the ground bulges between its samples
    make(WGSL.ROAD, { buffers: LAYOUT.textured, group: 'road', blend: BLEND.over, depthWrite: false, cull: 'none', bias: 8, slope: 2 }),
    make(WGSL.WATER, { group: 'sea', cull: 'none' }),
    make(WGSL.SKY, { group: 'single', depthWrite: false, cull: 'none' }),
    make(WGSL.GLOW, { buffers: LAYOUT.glow, blend: BLEND.add, depthWrite: false, cull: 'none' }),
    make(WGSL.SNOW, { buffers: LAYOUT.snow, blend: BLEND.add, depthWrite: false, cull: 'none' }),
    make(WGSL.TUBE, { buffers: LAYOUT.textured, group: 'road' }),
    make(WGSL.TUBE, { buffers: LAYOUT.textured, group: 'road', bias: 16, slope: 4 }), // the end of the road outside laps under the mouth
    make(WGSL.CAR, { buffers: LAYOUT.car, group: 'car' }),
    make(WGSL.PAVING, { buffers: LAYOUT.textured, group: 'road', bias: 4, slope: 1 }),
    make(WGSL.ZEBRA, { buffers: LAYOUT.plain, blend: BLEND.over, depthWrite: false, cull: 'none', bias: 12, slope: 3 }),
    make(WGSL.SIGN, { buffers: LAYOUT.textured, group: 'road' }),
    make(WGSL.PACK, { group: 'snow', blend: BLEND.over, depthWrite: false, cull: 'none', bias: 12, slope: 3 }),
    make(WGSL.SPRAY, { group: 'snow', buffers: [{ arrayStride: 28, stepMode: 'instance', attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }, { shaderLocation: 1, offset: 12, format: 'float32x4' }] }],
      blend: BLEND.add, depthWrite: false, cull: 'none' }),
    make(WGSL.TRACKS, { group: 'snow', buffers: [{ arrayStride: 16, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }, { shaderLocation: 1, offset: 12, format: 'unorm8x4' }] }],
      blend: BLEND.over, depthWrite: false, cull: 'none', bias: 16, slope: 4 }), // paving laps over road ribbons and terrain it lies a few centimetres above
    // shadow, glass and glare: colour arrives already multiplied by opacity; the shadow needs lifting off the ground it lies on
    make(WGSL.CAR, { buffers: LAYOUT.car, group: 'car', blend: BLEND.laid, depthWrite: false, bias: 12, slope: 3 }),
    make(WGSL.CLOUD, { buffers: LAYOUT.cloud, group: 'single', blend: BLEND.over, depthWrite: false, cull: 'none' }),
    make(WGSL.PLAIN, { buffers: LAYOUT.plain, topology: 'line-list', cull: 'none' }), // wires
    ...[WGSL.FOREST, WGSL.FOREST_FADE, WGSL.FOREST_TWIG].map(code => make(code, { buffers: LAYOUT.forest, group: 'forest', cull: 'none' })),
    make(WGSL.FOREST_HAZE, { buffers: LAYOUT.forest, group: 'forest', blend: BLEND.over, depthWrite: false, cull: 'none' }),
    make(WGSL.LABEL, { buffers: LAYOUT.textured, group: 'road', blend: BLEND.over }),
    make(WGSL.AFLOAT, { buffers: [LAYOUT.plain[0], { arrayStride: 16, attributes: [{ shaderLocation: 4, offset: 0, format: 'float32x4' }] }], group: 'sea' }),
    // lights on the water: laid on the sea's own plane, so lifted off it
    make(WGSL.STREAK, { buffers: [{ arrayStride: 36, stepMode: 'instance', attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x4' }, { shaderLocation: 1, offset: 16, format: 'float32x4' },
      { shaderLocation: 2, offset: 32, format: 'float32' }] }], group: 'sea', blend: BLEND.add, depthWrite: false, cull: 'none', bias: 8, slope: 2 }),
    make(WGSL.SMOKE_PUFFS, { buffers: [{ arrayStride: 24, stepMode: 'instance', attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }, { shaderLocation: 1, offset: 12, format: 'float32x3' }] }],
      blend: BLEND.over, depthWrite: false, cull: 'none' }),
    // what the car stirs up on the water: the wake's ribbon and the rings lie on the sea, the spray flies over it
    make(WGSL.WAKE, { buffers: [{ arrayStride: 28, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }, { shaderLocation: 1, offset: 12, format: 'float32x4' }] }],
      blend: BLEND.over, depthWrite: false, cull: 'none', bias: 8, slope: 2 }),
    make(WGSL.RING, { buffers: [{ arrayStride: 16, stepMode: 'instance', attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x4' }] }], blend: BLEND.over, depthWrite: false, cull: 'none', bias: 8, slope: 2 }),
    make(WGSL.DROPS, { buffers: [{ arrayStride: 28, stepMode: 'instance', attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }, { shaderLocation: 1, offset: 12, format: 'float32x4' }] }],
      blend: BLEND.over, depthWrite: false, cull: 'none' }),
    make(WGSL.LIVE, { buffers: LAYOUT.plain, group: 'live' }), // what moves in the sky
    // the wheel jets' flames and the snow they blow about
    make(WGSL.JETS, { buffers: [{ arrayStride: ROCKET.JET, stepMode: 'instance', attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x4' }, { shaderLocation: 1, offset: 16, format: 'float32x4' }] }],
      blend: BLEND.add, depthWrite: false, cull: 'none' }),
    make(WGSL.BLOWN, { buffers: [{ arrayStride: ROCKET.PUFF, stepMode: 'instance', attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x4' }, { shaderLocation: 1, offset: 16, format: 'float32x4' }] }],
      blend: BLEND.add, depthWrite: false, cull: 'none' }),
  ]);
  pipes[PIPE.TRAIL] = made[8]; // ski trails are drawn as the roads are, with a texture of their own
  { // the moon's shadow map: depth only, three vertex layouts
    const module = device.createShaderModule({ code: WGSL.SHADOW });
    shadow.layout = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: {} }] });
    const pos = stride => ({ arrayStride: stride, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] });
    const inst = { arrayStride: 32, stepMode: 'instance', attributes: [{ shaderLocation: 4, offset: 0, format: 'float32x4' }, { shaderLocation: 5, offset: 16, format: 'float32x3' }] };
    // the forest casts from its model texture, pinned to the ground map: a group of its own beside the map's uniforms
    const forest = device.createShaderModule({ code: WGSL.FOREST_SHADOW }), V = GPUShaderStage.VERTEX;
    shadow.forestLayout = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: V, texture: { sampleType: 'unfilterable-float' } },
      { binding: 1, visibility: V, texture: {} }, { binding: 2, visibility: V, sampler: {} }, { binding: 3, visibility: V, buffer: { type: 'read-only-storage' } }] });
    const layouts = { [true]: device.createPipelineLayout({ bindGroupLayouts: [shadow.layout, shadow.forestLayout] }), [false]: device.createPipelineLayout({ bindGroupLayouts: [shadow.layout] }) };
    const depth = (entryPoint, buffers, format = 'depth16unorm', bias = {}) => device.createRenderPipeline({
      layout: layouts[buffers === LAYOUT.forest], vertex: { module: buffers === LAYOUT.forest ? forest : module, entryPoint, buffers },
      primitive: { cullMode: 'none' }, depthStencil: { format, depthWriteEnabled: true, depthCompare: 'less', ...bias } });
    const casters = (...more) => ({ [PIPE.PLAIN]: depth('fixed', [pos(20)], ...more), [PIPE.AFLOAT]: depth('fixed', [pos(20)], ...more), [PIPE.WALL]: depth('fixed', [pos(28)], ...more),
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
  [pipes.terrain, pipes.terrainCut, pipes.outer, pipes.outerCut, pipes[PIPE.PLAIN], pipes[PIPE.WALL], pipes[PIPE.TREE], pipes[PIPE.LAMP], pipes[PIPE.ROAD], pipes.water, pipes.sky, pipes[PIPE.GLOW], pipes.snow, pipes[PIPE.TUBE], pipes[PIPE.TUBE_FLOOR], pipes.car, pipes[PIPE.PAVING], pipes[PIPE.ZEBRA], pipes[PIPE.SIGN], pipes.pack, pipes.spray, pipes.tracks, pipes.carBlend, pipes.cloud, pipes[PIPE.LINE],
    pipes[PIPE.FOREST], pipes[PIPE.FOREST_FADE], pipes[PIPE.FOREST_TWIG], pipes[PIPE.HAZE], pipes[PIPE.LABEL], pipes[PIPE.AFLOAT], pipes[PIPE.STREAK], pipes[PIPE.SMOKE], pipes.wake, pipes.ring, pipes.drops, pipes.live, pipes.jets, pipes.blown] = made;
  pipes[PIPE.PANE] = pipes[PIPE.SIGN]; // the centre's shop windows are drawn as signs are, from the window atlas
}

/* ---- terrain: a strip of cells, drawn once per row of the height grid ---- */

function rowStrip(n) {
  const idx = new Uint16Array((n - 1) * 6);
  for (let i = 0, q = 0; i < n - 1; i++) {
    const a = i * 2, b = a + 2, c = a + 1, d = a + 3; // (i, row), (i + 1, row), (i, row + 1), (i + 1, row + 1)
    idx[q++] = a; idx[q++] = c; idx[q++] = b; idx[q++] = b; idx[q++] = c; idx[q++] = d;
  }
  return gpuBuffer(idx, GPUBufferUsage.INDEX);
}

function gridGroup({ origin, step, clip, size, extent, tiles, ex, ez, heights, photos, sampler, cuts }) {
  const data = new ArrayBuffer(240), f = new Float32Array(data), u = new Uint32Array(data);
  f.set([origin[0], origin[1], step, clip]);
  u.set(size, 4); f.set(extent, 6); u.set(tiles, 8);
  f.set(ex, 12); f.set(ez, 20);
  if (cuts) for (let i = 0; i < 4; i++) { f.set(cuts.subarray(i * 8, i * 8 + 4), 28 + i * 4); f.set(cuts.subarray(i * 8 + 4, i * 8 + 8), 44 + i * 4); }
  return device.createBindGroup({ layout: groupLayouts.terrain, entries: [
    { binding: 0, resource: { buffer: gpuBuffer(data, GPUBufferUsage.UNIFORM) } },
    { binding: 1, resource: heights.createView() }, { binding: 2, resource: arrayView(photos) }, { binding: 3, resource: sampler }] });
}

/* ---- loading ---- */

async function loadTiles(ncx, ncz) {
  const get = async url => createImageBitmap(await (await fetch(url)).blob(), { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
  const urls = [];
  for (let cz = 0; cz < ncz; cz++) for (let cx = 0; cx < ncx; cx++) urls.push(`../assets/ground_${cx}_${cz}.jpg`);
  let done = 0;
  const count = p => p.then(b => { post({ progress: ++done / (urls.length + 1) }); return b; });
  const [outer, ...tiles] = await Promise.all([count(get('../assets/outer.jpg')), ...urls.map(u => count(get(u)))]);
  // tiles differ by a pixel or two; an array texture wants one size, so the odd ones are stretched to the largest
  const width = Math.max(...tiles.map(t => t.width)), height = Math.max(...tiles.map(t => t.height));
  const same = await Promise.all(tiles.map(t => t.width === width && t.height === height ? t
    : createImageBitmap(t, { resizeWidth: width, resizeHeight: height, resizeQuality: 'high', colorSpaceConversion: 'none', premultiplyAlpha: 'none' })));
  return { tiles: same, outer };
}

async function init({ canvas: c, shared: sab, params, width, height, ratio }) {
  canvas = c;
  canvas.width = width; canvas.height = height;
  shared = sab;
  for (const k of (params.skip || '').split(',')) skip[k] = true;
  skipDraws = (params.skip || '').includes(':');
  still = +params.still || 0;
  if (params.aa === '1') SAMPLES = 1;
  if (params.off) WGSL = await import('./shaders.js?off=' + params.off);
  if (shared) { keysView = new Uint32Array(shared, 0, 1); padView = new Float32Array(shared, 4, 6); hudView = new Uint8Array(shared, SHARED_HUD, HUD_BYTES); }

  // everything that can start at once does: the device, the module, the data, the photo tiles
  const gpu = (async () => {
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('WebGPU is not available in this browser');
    // where the browser can time passes on the GPU itself, bench() reports that too: what the frame costs the GPU, apart from
    // whatever else the machine is doing
    const timed = adapter.features.has('timestamp-query');
    device = await adapter.requestDevice({ requiredFeatures: timed ? ['timestamp-query'] : [] });
    if (timed) timing = { set: device.createQuerySet({ type: 'timestamp', count: STAMPS.length * 2 }),
      resolve: device.createBuffer({ size: STAMPS.length * 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }),
      read: device.createBuffer({ size: STAMPS.length * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }) };
    // An error on the device can leave the canvas handing out textures that cannot be drawn to (its own could not be made,
    // when the GPU is short of memory): the frame would then fail the same way for ever. So after any error the canvas is
    // configured afresh and the frame's targets remade before the next frame; a lost device cannot be mended from here, and
    // the page is asked to start again where the car is.
    device.addEventListener('uncapturederror', e => { mend = true; if (errors++ < 8) post({ error: e.error.message }); });
    device.lost.then(info => { if (info.reason !== 'destroyed') post({ lost: info.message || 'the GPU device was lost' }); });
    context = canvas.getContext('webgpu');
    format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: 'opaque' });
    mark('device');
    await makePipelines();
    mark('pipelines');
  })();
  const file = async url => (await fetch(url)).arrayBuffer();
  const [module, bin, heights, outer, carModel] = await Promise.all([
    WebAssembly.compileStreaming(fetch('../wasm/build/harstad.wasm')),
    file('../data/world.bin'), file('../data/height.bin'), file('../data/outer.bin'), file('../data/car.bin')]);
  mark('fetched');

  const env = { sin: Math.sin, cos: Math.cos, tan: Math.tan, atan2: Math.atan2, exp: Math.exp, pow: Math.pow };
  hs = (await WebAssembly.instantiate(module, { env })).exports;
  const put = f => { const ptr = hs.hs_alloc(f.byteLength); new Uint8Array(hs.memory.buffer, ptr, f.byteLength).set(new Uint8Array(f)); return [ptr, f.byteLength]; };
  if (hs.hs_world_init(...put(bin), ...put(heights), ...put(outer)) || hs.hs_car_init(...put(carModel)) || hs.hs_game_init()) throw new Error('could not read the world');
  const lay = new Uint32Array(hs.memory.buffer, hs.hs_layout(), 27);
  if (lay[5] !== Object.keys(PIPE).length) throw new Error('the glue and the module disagree about the pipelines');
  if (lay[24] !== (WGSL.LIVE_SLOTS * 3 + WGSL.LIVE_TINTS) * 16) throw new Error('the glue and the module disagree about the live mesh');
  Object.assign(FRAME, { UNIFORMS: lay[0], PIPE_FIRST: lay[1], CAR: lay[2], CAR_BYTES: lay[3], CMDS: lay[4], SHADOW_FRESH: lay[6], SHADOW: lay[7], SHADOW_VP: lay[8], SNOW: lay[9], LIVE: lay[10], NBEAM: lay[11], BEAM: lay[12], BEAM_VP: lay[13], SHADOW_EYE: lay[14], CAM_POS: lay[15], GROUND_RECT: lay[16], NGROUND: lay[17], GROUND: lay[18], GROUND_DRAWS: lay[19], SEA: lay[20], WAVES: lay[21], WATER_FRESH: lay[22], POSE: lay[23], POSE_BYTES: lay[24], FOREST: lay[25], ODO: lay[26] }); // WATER_FRESH is also where the frame's uniform blocks end
  mark('world');

  // the build worker gets the untouched files; this instance keeps its own copies
  const builder = new Worker(new URL('./build_worker.js', import.meta.url), { type: 'module' });
  builder.postMessage({ module, bin, heights, outer }, [bin, heights, outer]);
  const stages = [], built = new Promise((resolve, reject) => {
    builder.onerror = e => reject(new Error(e.message));
    builder.onmessage = ({ data }) => {
      if (data.done) { builder.terminate(); marks.buildWorker = data.ms; marks.buildMemory = data.memory; resolve(); }
      else stages.push(data);
    };
  });

  const info = new DataView(hs.memory.buffer, hs.hs_info(), 96);
  const f32 = o => info.getFloat32(o, true), u32 = o => info.getUint32(o, true);
  const world = { bx: f32(0), bz: f32(4), w: f32(8), d: f32(12), step: f32(16), outerHalf: f32(20), nx: u32(24), nz: u32(28), outerN: u32(32),
    ncx: u32(36), ncz: u32(40), heights: u32(44), outer: u32(48), ex: u32(52), ez: u32(56), roads: u32(60), roadPts: u32(64), strings: u32(68),
    nroads: u32(72), spawn: [f32(84), f32(88), f32(92)] };
  applyParams(params, world);
  hs.hs_set(SET.PIXEL_RATIO, ratio || 1);
  const images = loadTiles(world.ncx, world.ncz);
  postMinimap(world);

  await gpu;
  const mem = () => hs.memory.buffer;
  const half = hs.hs_alloc(world.nx * world.nz * 2);
  hs.hs_heights_half(half);
  const groundTex = dataTexture('r16float', world.nx, world.nz, new Uint8Array(mem(), half, world.nx * world.nz * 2), world.nx * 2);
  const heightTex = dataTexture('r32float', world.nx, world.nz, new Uint8Array(mem(), world.heights, world.nx * world.nz * 4), world.nx * 4);
  const outerTex = dataTexture('r32float', world.outerN, world.outerN, new Uint8Array(mem(), world.outer, world.outerN ** 2 * 4), world.outerN * 4);
  const detail = arrayTexture([noiseTexture()], 'rgba8unorm');
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
    { binding: 0, resource: arrayView(arrayTexture([displayAtlas()], 'rgba8unorm-srgb')) }, { binding: 1, resource: sampler({ maxAnisotropy: 8 }) }] });
  groups.road = device.createBindGroup({ layout: groupLayouts.road, entries: [
    { binding: 0, resource: arrayView(roads) }, { binding: 1, resource: sampler({ addressModeV: 'repeat', maxAnisotropy: 16 }) }] });
  groups.tube = device.createBindGroup({ layout: groupLayouts.road, entries: [
    { binding: 0, resource: arrayView(arrayTexture(tunnelTextures(), 'rgba8unorm-srgb')) }, { binding: 1, resource: sampler({ addressModeV: 'repeat', maxAnisotropy: 16 }) }] });
  groups.paving = device.createBindGroup({ layout: groupLayouts.road, entries: [
    { binding: 0, resource: arrayView(arrayTexture(pavingTextures(), 'rgba8unorm-srgb')) }, { binding: 1, resource: sampler({ ...repeat, maxAnisotropy: 8 }) }] });

  const { tiles, outer: outerPhoto } = await images;
  const photos = arrayTexture(tiles, 'rgba8unorm-srgb'), outerPhotos = arrayTexture([outerPhoto], 'rgba8unorm-srgb');
  for (const t of [...tiles, outerPhoto]) t.close();
  mark('tiles');
  const photoSampler = sampler({ maxAnisotropy: 8 });
  const edges = ptr => { const e = new Float32Array(8); e.set(new Uint32Array(mem(), ptr, ptr === world.ex ? world.ncx + 1 : world.ncz + 1)); return e; };
  groups.terrain = gridGroup({ origin: [world.bx, world.bz], step: world.step, clip: 0, size: [world.nx, world.nz], extent: [world.w, world.d],
    tiles: [world.ncx, world.ncz], ex: edges(world.ex), ez: edges(world.ez), heights: heightTex, photos, sampler: photoSampler,
    cuts: new Float32Array(mem(), hs.hs_tunnel_cuts(), 32) });
  const on = world.outerN, ostep = 2 * world.outerHalf / (on - 1);
  groups.outer = gridGroup({ origin: [-world.outerHalf, -world.outerHalf], step: ostep, clip: 1, size: [on, on], extent: [2 * world.outerHalf, 2 * world.outerHalf],
    tiles: [1, 1], ex: [0, on - 1, 0, 0, 0, 0, 0, 0], ez: [0, on - 1, 0, 0, 0, 0, 0, 0], heights: outerTex, photos: outerPhotos, sampler: photoSampler });
  rowIndex = rowStrip(world.nx); outerRowIndex = rowStrip(on);
  // every uniform block that changes each frame, in one buffer as struct frame lays them out, sent with one write
  frameBuffer = device.createBuffer({ size: FRAME.WATER_FRESH, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  { // the snow: its arrays live in wasm memory and are sent in pieces as the car changes them
    const [bytes, lattice, flecks, tracks, , , beds] = new Uint32Array(mem(), hs.hs_snow(), 7);
    const DST = GPUBufferUsage.COPY_DST, cells = new Uint16Array(255 * 255 * 6), ribbon = new Uint32Array(27999 * 9 * 6);
    for (let j = 0, q = 0; j < 255; j++) for (let i = 0; i < 255; i++) { const a = j * 256 + i, b = a + 1, c = a + 256, d = c + 1; cells[q++] = a; cells[q++] = c; cells[q++] = b; cells[q++] = b; cells[q++] = c; cells[q++] = d; }
    for (let s = 0, q = 0; s < 27999; s++) for (let v = 0; v < 9; v++) { const a = s * 10 + v, b = a + 10; ribbon[q++] = a; ribbon[q++] = a + 1; ribbon[q++] = b; ribbon[q++] = a + 1; ribbon[q++] = b + 1; ribbon[q++] = b; }
    Object.assign(snow, { bytes, lattice, flecks, tracks, beds,
      pack: device.createTexture({ size: [512, 512], format: 'rg8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST }),
      bed: device.createTexture({ size: [512, 512], format: 'r16uint', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST }),
      latticeTex: device.createTexture({ size: [16, 16], format: 'r32float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST }),
      spray: device.createBuffer({ size: 1536 * 28, usage: GPUBufferUsage.VERTEX | DST }),
      ribbon: device.createBuffer({ size: 28000 * 10 * 16, usage: GPUBufferUsage.VERTEX | DST }),
      cells: gpuBuffer(cells, GPUBufferUsage.INDEX), cellCount: cells.length, ribbonIndex: gpuBuffer(ribbon, GPUBufferUsage.INDEX), ribbonCount: ribbon.length });
    device.queue.writeTexture({ texture: snow.pack }, new Uint8Array(mem(), bytes, 512 * 512 * 2), { bytesPerRow: 1024 }, [512, 512]);
    device.queue.writeTexture({ texture: snow.bed }, new Uint8Array(mem(), beds, 512 * 512 * 2), { bytesPerRow: 1024 }, [512, 512]);
    packTo.texture = snow.pack; bedTo.texture = snow.bed; snow.latticeTo = { texture: snow.latticeTex }; snow.latticeFrom = { offset: lattice, bytesPerRow: 64 }; snow.latticeSize = [16, 16];
    device.queue.writeBuffer(snow.spray, 0, new Uint8Array(mem(), flecks, 1536 * 28));
    snow.group = device.createBindGroup({ layout: groupLayouts.snow, entries: [{ binding: 0, resource: { buffer: frameBuffer, offset: FRAME.SNOW, size: 48 } },
      { binding: 1, resource: snow.pack.createView() }, { binding: 2, resource: snow.latticeTex.createView() },
      { binding: 3, resource: device.createSampler({ minFilter: 'linear', magFilter: 'linear', addressModeU: 'repeat', addressModeV: 'repeat' }) },
      { binding: 4, resource: snow.bed.createView() }] });
  }
  { // what moves in the sky: its arrays live in wasm memory too. The mesh is sent once; where its parts are rides in the frame's uniforms
    const [v, idx, glow, puffs, nv, ni] = new Uint32Array(mem(), hs.hs_live(), 6), DST = GPUBufferUsage.COPY_DST;
    Object.assign(live, { v, glowAt: glow, puffs,
      vertex: device.createBuffer({ size: Math.max(20, nv * 20), usage: GPUBufferUsage.VERTEX | DST }),
      index: gpuBuffer(new Uint8Array(mem(), idx, Math.max(4, ni * 4)), GPUBufferUsage.INDEX),
      glow: device.createBuffer({ size: 128 * 28, usage: GPUBufferUsage.VERTEX | DST }),
      cloud: device.createBuffer({ size: 4600 * 24, usage: GPUBufferUsage.VERTEX | DST }),
      pose: device.createBindGroup({ layout: groupLayouts.live, entries: [{ binding: 0, resource: { buffer: frameBuffer, offset: FRAME.POSE, size: FRAME.POSE_BYTES } }] }),
      group: device.createBindGroup({ layout: groupLayouts.single, entries: [{ binding: 0, resource: arrayTexture([cloudNoise()], 'rgba8unorm').createView({ dimension: '2d' }) }] }) });
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
    skyLow.rows = device.createBuffer({ size: (32 + 30 * 1024) * 4, usage: GPUBufferUsage.STORAGE });
    skyLow.uniforms = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: C, buffer: {} }, { binding: 11, visibility: C, buffer: { type: 'read-only-storage' } }] });
    skyLow.outputs = device.createBindGroupLayout({ entries: [store(0), store(1)] });
    const mapsLayout = device.createPipelineLayout({ bindGroupLayouts: [skyLow.uniforms, skyLow.outputs] });
    skyLow.pano = device.createComputePipeline({ layout: mapsLayout, compute: { module: maps, entryPoint: 'pano' } });
    skyLow.pipe = device.createComputePipeline({ layout: mapsLayout, compute: { module: maps, entryPoint: 'low' } });
    skyLow.frame = device.createBindGroup({ layout: skyLow.uniforms, entries: [frameEntry, { binding: 11, resource: { buffer: skyLow.rows } }] });
    const rowsIn = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: C, buffer: {} }] });
    const rowsOut = device.createBindGroupLayout({ entries: [{ binding: 2, visibility: C, buffer: { type: 'storage' } }] });
    skyLow.rowsPipe = device.createComputePipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [rowsIn, rowsOut] }), compute: { module: maps, entryPoint: 'rows' } });
    skyLow.rowsGroups = [device.createBindGroup({ layout: rowsIn, entries: [frameEntry] }), device.createBindGroup({ layout: rowsOut, entries: [{ binding: 2, resource: { buffer: skyLow.rows } }] })];
    sea.ripplePipe = device.createComputePipeline({ layout: 'auto', compute: { module: rippleMaps } });
    sea.rippleGroup = device.createBindGroup({ layout: sea.ripplePipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: frameBuffer, offset: FRAME.WAVES, size: WAVES_BYTES } },
      { binding: 1, resource: sea.ripples.tops[0] }, { binding: 2, resource: sea.ripples.tops[1] }] });

    const S = WGSL.HORIZON_SIZE, march = device.createShaderModule({ code: WGSL.HORIZON }), outerHalf = hs.hs_alloc(on * on * 2);
    hs.hs_outer_half(outerHalf);
    const outerSmooth = dataTexture('r16float', on, on, new Uint8Array(mem(), outerHalf, on * on * 2), on * 2);
    const pipe = device.createRenderPipeline({ layout: 'auto', vertex: { module: march }, fragment: { module: march, targets: [{ format: 'rgba8unorm' }] }, primitive: whole });
    const enc = device.createCommandEncoder();
    sea.horizon = [0, 4].map(first => {
      const texture = device.createTexture({ size: [S, S], format: 'rgba8unorm', mipLevelCount: mips(S, S), usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT });
      const u = new Float32Array([world.bx - world.step / 2, world.bz - world.step / 2, world.nx * world.step, world.nz * world.step,
        -world.outerHalf, -world.outerHalf, 2 * world.outerHalf, 2 * world.outerHalf, first, on, 0, 0]);
      const pass = enc.beginRenderPass({ colorAttachments: [{ view: texture.createView({ baseMipLevel: 0, mipLevelCount: 1 }), loadOp: 'clear', storeOp: 'store' }] });
      pass.setPipeline(pipe);
      pass.setBindGroup(0, device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: gpuBuffer(u, GPUBufferUsage.UNIFORM) } },
        { binding: 1, resource: groundTex.createView() }, { binding: 2, resource: outerSmooth.createView() }, { binding: 3, resource: device.createSampler({ minFilter: 'linear', magFilter: 'linear' }) }] }));
      pass.draw(3); pass.end();
      return texture;
    });
    device.queue.submit([enc.finish()]);
    sea.horizon.forEach(makeMips);
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
    car = { solid, blended, dash: dashPainter(),
      vertex: gpuBuffer(new Uint8Array(mem(), v, nv * 40), GPUBufferUsage.VERTEX), index: gpuBuffer(new Uint8Array(mem(), idx, (solid + blended) * 4), GPUBufferUsage.INDEX),
      dials: picture(1024, 320, 2), panels: picture(512, 256) };
    car.dash.layers.forEach((source, layer) => device.queue.copyExternalImageToTexture({ source }, { texture: car.dials, origin: [0, 0, layer] }, [1024, 320]));
    device.queue.copyExternalImageToTexture({ source: car.dash.panels }, { texture: car.panels }, [512, 256]);
    car.plate = dataTexture('rgba8unorm-srgb', 256, 64, plate.data, 256 * 4).createView(); car.dialsView = car.dials.createView({ dimension: '2d-array' });
    car.group = device.createBindGroup({ layout: groupLayouts.car, entries: [{ binding: 0, resource: { buffer: frameBuffer, offset: FRAME.CAR, size: FRAME.CAR_BYTES } },
      { binding: 1, resource: car.plate }, { binding: 2, resource: car.dialsView }, { binding: 3, resource: car.panels.createView() }] });
  }
  { // the wheel jets: their nozzles and struts are drawn by the car's pipeline from a uniform block of their own; the puffs' ring starts long settled
    const [v, nv, idx, ni, pose, jets, puffs, state, drift] = new Uint32Array(mem(), hs.hs_rocket(), 9), DST = GPUBufferUsage.COPY_DST;
    Object.assign(rocket, { pose, jetsAt: jets, puffsAt: puffs, state: state >> 2, drift: drift >> 2, count: ni,
      vertex: gpuBuffer(new Uint8Array(mem(), v, nv * 40), GPUBufferUsage.VERTEX), index: gpuBuffer(new Uint8Array(mem(), idx, ni * 4), GPUBufferUsage.INDEX),
      uniform: device.createBuffer({ size: FRAME.CAR_BYTES, usage: GPUBufferUsage.UNIFORM | DST }),
      jets: device.createBuffer({ size: 4 * ROCKET.JET, usage: GPUBufferUsage.VERTEX | DST }),
      puffs: gpuBuffer(new Uint8Array(mem(), puffs, ROCKET.PUFFS * ROCKET.PUFF), GPUBufferUsage.VERTEX | DST) });
    rocket.group = device.createBindGroup({ layout: groupLayouts.car, entries: [{ binding: 0, resource: { buffer: rocket.uniform } },
      { binding: 1, resource: car.plate }, { binding: 2, resource: car.dialsView }, { binding: 3, resource: car.panels.createView() }] });
  }
  snowfall(+(params.snow ?? 0));
  quadIndex = gpuBuffer(new Uint16Array([0, 1, 2, 0, 2, 3]), GPUBufferUsage.INDEX); // a sprite's two triangles (WGSL QUAD)

  // geometry, as the build worker finishes each stage
  await built;
  mark('built');
  let lampTex, tintTex, seaMap, treeModels, triangles = 0, bytes = 0;
  for (const stage of stages) {
    stage.bufs.forEach((b, i) => {
      bytes += b.data.byteLength;
      if (b.kind === BUF.LAMP_MAP) lampTex = dataTexture('rgba8unorm', b.w, b.h, b.data, b.w * 4);
      else if (b.kind === BUF.TINT_MAP) tintTex = dataTexture('rg8unorm', b.w, b.h, b.data, b.w * 2);
      else if (b.kind === BUF.SEA_MAP) seaMap = b;
      else if (b.kind === BUF.TREE_MODELS) treeModels = dataTexture('rgba32float', b.w, b.h, b.data, b.w * 16).createView(); // every tree model, two texels per vertex
      else if (b.kind === BUF.FOREST) { // every tree: the shaders read them by index, and the game lists each frame which to draw
        const trees = { buffer: gpuBuffer(b.data, GPUBufferUsage.STORAGE) }, n = b.data.byteLength / 32, at = hs.hs_alloc(b.data.byteLength);
        new Uint8Array(hs.memory.buffer, at, b.data.byteLength).set(new Uint8Array(b.data));
        if (hs.hs_forest(at, n, stage.first + i)) throw new Error('out of memory for the forest');
        buffers[stage.first + i] = device.createBuffer({ size: Math.max(4, n * FOREST_WORDS * 4), usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST }); // the list
        groups.forest = device.createBindGroup({ layout: groupLayouts.forest, entries: [{ binding: 0, resource: treeModels }, { binding: 1, resource: trees }] });
        shadow.forest = device.createBindGroup({ layout: shadow.forestLayout, entries: [{ binding: 0, resource: treeModels }, { binding: 1, resource: groundTex.createView() },
          { binding: 2, resource: device.createSampler({ minFilter: 'linear', magFilter: 'linear' }) }, { binding: 3, resource: trees }] });
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
      } else buffers[stage.first + i] = gpuBuffer(b.data, b.kind === BUF.INDEX ? GPUBufferUsage.INDEX : GPUBufferUsage.VERTEX);
    });
    const n = stage.items.byteLength / ITEM_BYTES, at = n ? hs.hs_items(n) : 1;
    if (!at) throw new Error('too many draw items');
    new Uint8Array(hs.memory.buffer, at, n * ITEM_BYTES).set(new Uint8Array(stage.items));
    marks['stage' + stage.stage] = stage.ms;
  }
  { // the bound map (WGSL.SEA_BOUND): what the water near each texel can be at most, level by level, for cutting the streaks to their water
    const seaTex = arrayTexture([{ width: seaMap.w, height: seaMap.h, data: new Uint8Array(seaMap.data) }], 'rgba8unorm');
    sea.map = seaTex.createView({ dimension: '2d' });
    const levels = mips(seaMap.w, seaMap.h), bound = device.createTexture({ size: [seaMap.w, seaMap.h], format: 'rg8unorm', mipLevelCount: levels, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT });
    const enc = device.createCommandEncoder(), view = l => bound.createView({ baseMipLevel: l, mipLevelCount: 1 });
    const pipe = first => { const module = device.createShaderModule({ code: WGSL.SEA_BOUND(first) }); return device.createRenderPipeline({ layout: 'auto', vertex: { module }, fragment: { module, targets: [{ format: 'rg8unorm' }] } }); };
    const p0 = pipe(true), pn = pipe(false);
    for (let l = 0; l < levels; l++) {
      const pass = enc.beginRenderPass({ colorAttachments: [{ view: view(l), loadOp: 'clear', storeOp: 'store' }] });
      pass.setPipeline(l ? pn : p0);
      pass.setBindGroup(0, device.createBindGroup({ layout: (l ? pn : p0).getBindGroupLayout(0), entries: l ? [{ binding: 0, resource: view(l - 1) }]
        : [{ binding: 0, resource: seaTex.createView({ dimension: '2d', baseMipLevel: 0, mipLevelCount: 1 }) }, { binding: 1, resource: sea.horizon[0].createView({ baseMipLevel: 0, mipLevelCount: 1 }) },
          { binding: 2, resource: { buffer: gpuBuffer(new Float32Array(mem(), sea.at, 8).slice(), GPUBufferUsage.UNIFORM) } }] }));
      pass.draw(3); pass.end();
    }
    device.queue.submit([enc.finish()]);
    sea.bound = bound.createView();
  }
  groups.sea = device.createBindGroup({ layout: groupLayouts.sea, entries: [{ binding: 0, resource: { buffer: frameBuffer, offset: FRAME.SEA, size: SEA_BYTES } },
    { binding: 1, resource: sea.map },
    { binding: 2, resource: sea.ripples.textures[0].createView() }, { binding: 3, resource: sea.ripples.textures[1].createView() },
    { binding: 4, resource: sea.floes.createView({ dimension: '2d' }) }, { binding: 5, resource: sea.horizon[0].createView() }, { binding: 6, resource: sea.horizon[1].createView() },
    { binding: 7, resource: sea.sky.textures[0].createView() }, { binding: 8, resource: sampler() }, { binding: 9, resource: sampler({ ...repeat, maxAnisotropy: 4 }) },
    { binding: 10, resource: sampler({ addressModeU: 'repeat' }) }, { binding: 11, resource: sea.bound }] });
  { const at = hs.hs_alloc(seaMap.data.byteLength); new Uint8Array(hs.memory.buffer, at, seaMap.data.byteLength).set(new Uint8Array(seaMap.data)); hs.hs_sea_map(at); } // the wake follows what is wet
  frameGroup = device.createBindGroup({ layout: frameLayout, entries: [
    { binding: 0, resource: { buffer: frameBuffer, offset: 0, size: FRAME.UNIFORMS } }, { binding: 1, resource: lampTex.createView() }, { binding: 2, resource: groundTex.createView() },
    { binding: 3, resource: detail.createView({ dimension: '2d' }) },
    { binding: 4, resource: device.createSampler({ minFilter: 'linear', magFilter: 'linear' }) }, { binding: 5, resource: sampler(repeat) },
    { binding: 6, resource: shadow.view }, { binding: 7, resource: device.createSampler({ compare: 'less', minFilter: 'linear', magFilter: 'linear' }) },
    { binding: 8, resource: shadow.beams.createView({ dimension: '2d-array' }) },
    { binding: 9, resource: arrayTexture([snowRelief()], 'rgba8unorm').createView({ dimension: '2d' }) },
    { binding: 10, resource: tintTex.createView() }, { binding: 11, resource: { buffer: skyLow.rows } }] });
  await device.queue.onSubmittedWorkDone();
  mark('uploaded');
  marks.geometryBytes = bytes;
  marks.wasmMemory = hs.memory.buffer.byteLength;
  resize(width, height);
  if ('count' in params) countUploads(); // ?count: what is sent to the GPU each frame, by what it is for, in report()
  requestAnimationFrame(tick);
  await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
  mark('ready');
  post({ ready: true, marks, ...roadNames(world) });
}

// The falling snow: `thick` times the flakes the layers were tuned with, 0 .. 8 (the page's setting, or ?snow=; none unless asked)
function snowfall(thick) {
  if (!device) return;
  const k = Math.min(8, Math.max(0, thick || 0)), layers = WGSL.SNOW_LAYERS.map(l => ({ ...l, count: Math.round(l.count * k) }));
  snowBuffer?.destroy();
  snowCount = layers.reduce((n, l) => n + l.count, 0);
  snowBuffer = snowCount ? gpuBuffer(snowFlakes(layers), GPUBufferUsage.VERTEX) : null;
}

function applyParams(p, world) {
  const num = k => +(p[k] ?? 0) || 0;
  hs.hs_car_place(num('x') || world.spawn[0], num('z') || world.spawn[1], 'h' in p ? num('h') : world.spawn[2]);
  hs.hs_set(SET.CAM, num('cam'));
  if (num('orbit')) hs.hs_set(SET.ORBIT, num('orbit'));
  if ('top' in p) hs.hs_set(SET.TOP, num('top'));
  hs.hs_set(SET.AUTO_DRIVE, num('drive'));
  hs.hs_set(SET.AUTO_FLY, num('fly'));
  hs.hs_set(SET.GUSTS, p.gusts === '0' ? 0 : 1);
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

function resize(width, height) {
  if (!device) return;
  canvas.width = width; canvas.height = height;
  colourTarget?.destroy(); depthTarget?.destroy();
  // The depth, and with ?aa=4 the multisampled colour, are never stored (the colour is resolved onto the canvas, the depth
  // thrown away): where the browser can say so they live only in the GPU's tile memory and are never allocated
  const kept = GPUTextureUsage.RENDER_ATTACHMENT | (GPUTextureUsage.TRANSIENT_ATTACHMENT || 0);
  colourTarget = SAMPLES > 1 ? device.createTexture({ size: [width, height], format, sampleCount: SAMPLES, usage: kept }) : null;
  depthTarget = device.createTexture({ size: [width, height], format: 'depth32float', sampleCount: SAMPLES, usage: kept });
  if (colourTarget) mainPass.colorAttachments[0].view = colourTarget.createView();
  else { delete mainPass.colorAttachments[0].resolveTarget; mainPass.colorAttachments[0].storeOp = 'store'; } // unsampled: straight onto the canvas
  mainPass.depthStencilAttachment.view = depthTarget.createView();
  if (skyLow.pipe) { // the sky's noise at half the width and height (SKY_MAPS)
    skyLow.texture?.destroy();
    skyLow.texture = device.createTexture({ size: [Math.ceil(width / 2), Math.ceil(height / 2)], format: 'rgba16float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING });
    skyLow.group = device.createBindGroup({ layout: groupLayouts.single, entries: [{ binding: 0, resource: skyLow.texture.createView() }] });
    skyLow.out = device.createBindGroup({ layout: skyLow.outputs, entries: [{ binding: 0, resource: sea.sky.tops[0] }, { binding: 1, resource: skyLow.texture.createView() }] });
    skyLow.size = [Math.ceil(Math.ceil(width / 2) / 8), Math.ceil(Math.ceil(height / 2) / 8)];
  }
}
const skyLow = { pipe: null, texture: null, group: null, pass: {} };

// Nothing below makes an object each frame beyond what WebGPU itself hands out (the encoder, its passes, the canvas's
// texture and its view): views on the module's memory are kept until it grows, pass descriptors are filled in once.
let last = 0, mem8, mem32, memF, hudBytes, hudAt, benching = false;
const hudSent = new Uint8Array(HUD_BYTES), hudTold = { kmh: -1, up: 0, road: 0, gear: 0, cam: 0, flags: 0, at: 0 }; // without shared memory: the HUD as the page last had it
let still = 0, played = 0; // ?still=8 plays eight seconds from the start at sixty frames to the second, whatever the clock says, then stops the game's clock: a view that can be drawn again exactly, for comparing pictures
const mainPass = { colorAttachments: [{ view: null, resolveTarget: null, loadOp: 'clear', storeOp: 'discard', clearValue: [0, 0, 0, 1] }],
  depthStencilAttachment: { view: null, depthClearValue: 0, depthLoadOp: 'clear', depthStoreOp: 'discard' } };
const submitted = [null];
const shown = { km: 0, minute: 0, reverse: 1, zone: new Date().getTimezoneOffset() }; // what the dash's pictures say, as dashPainter first paints them
const patchFrom = { source: null, origin: [0, 0] }, patchTo = { texture: null, origin: [0, 0, 0] }, patchSize = [0, 0];
function patch(source, texture, rect) { // the part of a canvas that was painted again, to the same place in its texture
  patchFrom.source = source; patchTo.texture = texture;
  patchFrom.origin[0] = patchTo.origin[0] = rect[0]; patchFrom.origin[1] = patchTo.origin[1] = rect[1]; patchSize[0] = rect[2]; patchSize[1] = rect[3];
  device.queue.copyExternalImageToTexture(patchFrom, patchTo, patchSize);
}
// A pass descriptor with bench()'s time stamps on it, or without: the descriptor itself is kept from frame to frame.
function stamped(desc, name) {
  if (benching && timing) { const i = STAMPS.indexOf(name) * 2; desc.timestampWrites = { querySet: timing.set, beginningOfPassWriteIndex: i, endOfPassWriteIndex: i + 1 }; }
  else if (desc.timestampWrites) delete desc.timestampWrites;
  return desc;
}
function band(r, n) { // rows r .. r + n of the snow field
  packTo.origin[0] = 0; packTo.origin[1] = r; packFrom.offset = snow.bytes + r * 1024; packSize[0] = 512; packSize[1] = n;
  device.queue.writeTexture(packTo, mem8, packFrom, packSize);
}
// what the field lies on, and the field itself, over a block of whole rows or whole columns the window has just scrolled in
function page(c, r, w, h) {
  packTo.origin[0] = c; packTo.origin[1] = r; packFrom.offset = snow.bytes + r * 1024 + c * 2; packSize[0] = w; packSize[1] = h;
  device.queue.writeTexture(packTo, mem8, packFrom, packSize);
  bedTo.origin[0] = c; bedTo.origin[1] = r; bedFrom.offset = snow.beds + r * 1024 + c * 2;
  device.queue.writeTexture(bedTo, mem8, bedFrom, packSize);
}
function pages(c, n, rows) { // n columns (or rows) from c, wrapping
  const a = Math.min(n, 512 - c);
  if (rows) { page(0, c, 512, a); if (a < n) page(0, 0, 512, n - a); }
  else { page(c, 0, a, 512); if (a < n) page(0, 0, n - a, 512); }
}
const packTo = { texture: null, origin: [0, 0] }, packFrom = { offset: 0, bytesPerRow: 1024 }, packSize = [512, 0];
const bedTo = { texture: null, origin: [0, 0] }, bedFrom = { offset: 0, bytesPerRow: 1024 };
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
function run(pass, f, p) { // every draw of pipeline p in this frame's list; returns how many
  const first = (f + FRAME.PIPE_FIRST) >> 2, cmds = (f + FRAME.CMDS) >> 2, from = mem32[first + p], to = mem32[first + p + 1];
  if (!stats.frames) { stats.indices ||= {}; stats.indices[p] = Array.from({ length: to - from }, (_, i) => mem32[cmds + (from + i) * 9 + 4] * mem32[cmds + (from + i) * 9 + 7]); } // what each pipeline draws, for report()
  if (from === to) return 0;
  pass.setPipeline(pipes[p]);
  unbound();
  for (let i = from; i < to; i++) if (!skipDraws || !skip[p + ':' + (i - from)]) drawCmd(pass, cmds + i * 9);
  return to - from;
}
function tick(now) {
  requestAnimationFrame(tick);
  if (benching) return;
  try { frame(now); } catch (e) { post({ error: String(e.stack || e) }); throw e; }
}
function frame(now) {
  const t0 = performance.now();
  if (mend && t0 - mended > 500) { // see where the device's errors are listened for
    mend = false; mended = t0;
    context.configure({ device, format, alphaMode: 'opaque' });
    resize(canvas.width, canvas.height);
    hs.hs_shadows_stale(); // the shadow maps' textures and views outlive the error, but a pass may have been lost: all three are drawn again
  }
  let dt = last ? (now - last) / 1000 : 0;
  if (still && mem32 && mem32[(hudAt + 56) >> 2] & 16) { // counted from when the title is dismissed
    if (!played) hs.hs_set(SET.TIME, 0);
    dt = played++ < still * 60 ? 1 / 60 : 0;
  }
  if (last) stats.times[stats.n++ & 4095] = now - last;
  last = now;
  if (keysView) { keys = Atomics.load(keysView, 0); pad.set(padView); }
  hs.hs_pad(pad[0], pad[1], pad[2], pad[3], pad[4], pad[5]);
  const f = hs.hs_frame(keys, dt, canvas.width, canvas.height);
  const t1 = performance.now();
  if (!mem32 || mem32.buffer !== hs.memory.buffer) {
    mem8 = new Uint8Array(hs.memory.buffer); mem32 = new Uint32Array(hs.memory.buffer); memF = new Float32Array(hs.memory.buffer);
    hudAt = hs.hs_hud(); hudBytes = new Uint8Array(hs.memory.buffer, hudAt, HUD_BYTES);
  }
  if (hudView) hudView.set(hudBytes);
  else { // No shared memory: a message, 64 bytes copied from the one buffer kept for it. What the page writes out (speed in whole
    // km/h, metres up, the road, the gear, the camera, the flags) goes the frame it changes; what only slides (the minimap's place
    // and turn, what the sound follows) goes when it has changed, thirty times a second at most. A parked car sends nothing.
    const h = hudAt >> 2, kmh = Math.round(Math.abs(memF[h]) * 3.6), up = Math.round(memF[h + 1]);
    let told = kmh !== hudTold.kmh || up !== hudTold.up || mem32[h + 11] !== hudTold.road || mem32[h + 12] !== hudTold.gear || mem32[h + 13] !== hudTold.cam || mem32[h + 14] !== hudTold.flags;
    if (!told && t0 - hudTold.at >= 33) for (let i = 0; i < HUD_BYTES - 4 && !told; i++) told = hudSent[i] !== hudBytes[i]; // all but the frame's number
    if (told) {
      hudTold.kmh = kmh; hudTold.up = up; hudTold.road = mem32[h + 11]; hudTold.gear = mem32[h + 12]; hudTold.cam = mem32[h + 13]; hudTold.flags = mem32[h + 14]; hudTold.at = t0;
      hudSent.set(hudBytes);
      postMessage(hudSent.buffer);
    }
  }

  const bytes = mem8;
  device.queue.writeBuffer(frameBuffer, 0, bytes, f, FRAME.WATER_FRESH); // the frame's, the car's, the snow's, the sea's and its waves', the live mesh's pose, in one
  { // the snow: its uniforms, and whichever of its arrays the car changed this frame (struct snow_frame)
    const s = ((f + FRAME.SNOW) >> 2) + 12, whole = mem32[s], row = mem32[s + 1], rows = mem32[s + 2], lattice = mem32[s + 3], sprayFirst = mem32[s + 4], sprayCount = mem32[s + 5],
      trackFirst = mem32[s + 6], trackCount = mem32[s + 7], bankFirst = mem32[s + 8], bankCount = mem32[s + 9],
      pageCol = mem32[s + 10], pageCols = mem32[s + 11], pageRow = mem32[s + 12], pageRows = mem32[s + 13];
    if (whole) page(0, 0, 512, 512);
    else {
      if (pageCols) pages(pageCol, pageCols, false);
      if (pageRows) pages(pageRow, pageRows, true);
      if (rows) { const n = Math.min(rows, 512 - row); band(row, n); if (n < rows) band(0, rows - n); }
    }
    if (lattice) device.queue.writeTexture(snow.latticeTo, bytes, snow.latticeFrom, snow.latticeSize);
    if (sprayCount) {
      const n = Math.min(sprayCount, 1536 - sprayFirst);
      device.queue.writeBuffer(snow.spray, sprayFirst * 28, bytes, snow.flecks + sprayFirst * 28, n * 28);
      if (n < sprayCount) device.queue.writeBuffer(snow.spray, 0, bytes, snow.flecks, (sprayCount - n) * 28);
    }
    if (trackCount) {
      device.queue.writeBuffer(snow.ribbon, trackFirst * 160, bytes, snow.tracks + trackFirst * 160, trackCount * 160);
      snow.laid = Math.max(snow.laid || 0, trackFirst + trackCount); // nothing beyond the furthest cross-section ever written needs drawing
    }
    if (bankCount && snow.bank) device.queue.writeBuffer(snow.bank, bankFirst * 20, bytes, snow.bankAt + bankFirst * 20, bankCount * 20);
  }
  // the wheel jets: nothing is sent while the car is on its wheels and the snow they blew has settled
  const rocketsOut = mem32[rocket.state], blownLive = mem32[rocket.drift], blownFirst = mem32[rocket.drift + 1], blownCount = mem32[rocket.drift + 2]; // struct rocket_frame, struct drift_frame
  if (rocketsOut) {
    device.queue.writeBuffer(rocket.uniform, 0, bytes, rocket.pose, FRAME.CAR_BYTES);
    device.queue.writeBuffer(rocket.jets, 0, bytes, rocket.jetsAt, 4 * ROCKET.JET);
  }
  if (blownCount) {
    const n = Math.min(blownCount, ROCKET.PUFFS - blownFirst);
    device.queue.writeBuffer(rocket.puffs, blownFirst * ROCKET.PUFF, bytes, rocket.puffsAt + blownFirst * ROCKET.PUFF, n * ROCKET.PUFF);
    if (n < blownCount) device.queue.writeBuffer(rocket.puffs, 0, bytes, rocket.puffsAt, (blownCount - n) * ROCKET.PUFF);
  }
  const lv = (f + FRAME.LIVE) >> 2, liveV = mem32[lv], liveI = mem32[lv + 1], liveGlow = mem32[lv + 2], puffs = mem32[lv + 3], puffsFresh = mem32[lv + 4];
  if (mem32[lv + 5]) device.queue.writeBuffer(live.vertex, 0, bytes, live.v, liveV * 20);
  if (liveGlow) device.queue.writeBuffer(live.glow, 0, bytes, live.glowAt, liveGlow * 28);
  if (puffsFresh) device.queue.writeBuffer(live.cloud, 0, bytes, live.puffs, puffs * 24);
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
  const t2 = performance.now();
  const enc = device.createCommandEncoder();
  { // One compute pass: the aurora's rows of noise, which what follows and the frame's shaders read; the ripple tiles
    // and the sky the water mirrors, worked out afresh and halved down their mip chains; and the sky's noise for this
    // view at half its width and height
    const cp = enc.beginComputePass(stamped(skyLow.pass, 'maps'));
    cp.setPipeline(skyLow.rowsPipe); cp.setBindGroup(0, skyLow.rowsGroups[0]); cp.setBindGroup(1, skyLow.rowsGroups[1]); cp.dispatchWorkgroups(1024 / 64, 30);
    if (!skip.water) {
      cp.setPipeline(sea.ripplePipe); cp.setBindGroup(0, sea.rippleGroup); cp.dispatchWorkgroups(WGSL.RIPPLE_N / 8, WGSL.RIPPLE_N / 8);
      cp.setPipeline(skyLow.pano); cp.setBindGroup(0, skyLow.frame); cp.setBindGroup(1, skyLow.out); cp.dispatchWorkgroups(WGSL.SKY_W / 8, WGSL.SKY_H / 8);
      sea.ripples.halve(cp); sea.sky.halve(cp);
    }
    if (!skip.sky && !skip.skylow) { cp.setPipeline(skyLow.pipe); cp.setBindGroup(0, skyLow.frame); cp.setBindGroup(1, skyLow.out); cp.dispatchWorkgroups(skyLow.size[0], skyLow.size[1]); }
    cp.end();
  }
  if (!shadow.placed) { // the ground map's rectangle never changes: each map's uniforms take it once
    for (const u of [shadow.uniform, shadow.beam[0].uniform]) device.queue.writeBuffer(u, 64, bytes, f + FRAME.GROUND_RECT, 16);
    shadow.placed = true;
  }
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
  mainPass.colorAttachments[0][colourTarget ? 'resolveTarget' : 'view'] = (grabbing || context.getCurrentTexture()).createView();
  const pass = enc.beginRenderPass(stamped(mainPass, 'main'));
  let draws = 0;
  pass.setBindGroup(0, frameGroup);
  if (!skip.plain) draws += run(pass, f, PIPE.PLAIN);
  draws += run(pass, f, PIPE.LINE);
  if (liveI && !skip.live) { pass.setPipeline(pipes.live); pass.setBindGroup(1, live.pose); pass.setVertexBuffer(0, live.vertex); pass.setIndexBuffer(live.index, 'uint32'); pass.drawIndexed(liveI); draws++; }
  if (!skip.car) { pass.setPipeline(pipes.car); pass.setBindGroup(1, car.group); pass.setVertexBuffer(0, car.vertex); pass.setIndexBuffer(car.index, 'uint32'); pass.drawIndexed(car.solid); draws++; }
  if (rocketsOut && !skip.rocket) { pass.setPipeline(pipes.car); pass.setBindGroup(1, rocket.group); pass.setVertexBuffer(0, rocket.vertex); pass.setIndexBuffer(rocket.index, 'uint32'); pass.drawIndexed(rocket.count); draws++; }
  pass.setBindGroup(1, groups.wall); if (!skip.wall) draws += run(pass, f, PIPE.WALL);
  pass.setBindGroup(1, groups.tube); draws += run(pass, f, PIPE.TUBE); draws += run(pass, f, PIPE.TUBE_FLOOR);
  if (!skip.tree) draws += run(pass, f, PIPE.TREE);
  if (!skip.forest) { pass.setBindGroup(1, groups.forest); draws += run(pass, f, PIPE.FOREST); draws += run(pass, f, PIPE.FOREST_TWIG); draws += run(pass, f, PIPE.FOREST_FADE); }
  draws += run(pass, f, PIPE.LAMP);
  pass.setBindGroup(1, groups.paving); draws += run(pass, f, PIPE.PAVING);
  if (groups.sign) { pass.setBindGroup(1, groups.sign); draws += run(pass, f, PIPE.SIGN); }
  pass.setBindGroup(1, groups.pane); draws += run(pass, f, PIPE.PANE);
  if (!skip.terrain) { // the ground in view, as the C gathered it: the town's, then the backdrop's; in each the plain chunks, then those a shader opens
    for (let g = 0; g < 2; g++) {
      pass.setBindGroup(1, g ? groups.outer : groups.terrain); pass.setIndexBuffer(g ? outerRowIndex : rowIndex, 'uint16');
      for (let kind = 0; kind < 2; kind++) {
        const n = mem32[((f + FRAME.NGROUND) >> 2) + g * 2 + kind];
        if (!n) continue;
        pass.setPipeline(g ? (kind ? pipes.outerCut : pipes.outer) : (kind ? pipes.terrainCut : pipes.terrain));
        for (let i = 0, d = (f + FRAME.GROUND + (g * 2 + kind) * FRAME.GROUND_DRAWS * 16) >> 2; i < n; i++, d += 4) pass.drawIndexed(mem32[d + 1], mem32[d + 3], mem32[d], 0, mem32[d + 2]);
        draws += n;
      }
    }
  }
  if (!skip.water) { pass.setBindGroup(1, groups.sea); draws += run(pass, f, PIPE.AFLOAT); pass.setPipeline(pipes.water); pass.draw(6); }
  if (groups.trail) { pass.setBindGroup(1, groups.trail); draws += run(pass, f, PIPE.TRAIL); } // under the roads, where a trail meets one
  pass.setBindGroup(1, groups.road); if (!skip.road) draws += run(pass, f, PIPE.ROAD);
  draws += run(pass, f, PIPE.ZEBRA);
  pass.setBindGroup(1, snow.group);
  if (!skip.tracks) { pass.setPipeline(pipes.tracks); pass.setVertexBuffer(0, snow.ribbon); pass.setIndexBuffer(snow.ribbonIndex, 'uint32'); pass.drawIndexed(Math.min(snow.ribbonCount, (snow.laid || 0) * 54)); }
  if (!skip.pack) { pass.setPipeline(pipes.pack); pass.setIndexBuffer(snow.cells, 'uint16'); pass.drawIndexed(snow.cellCount); }
  draws += 2;
  if (!skip.sky && !skip.skydraw) { pass.setPipeline(pipes.sky); pass.setBindGroup(1, skyLow.group); pass.draw(3); }
  if (!skip.water && !skip.streak) { pass.setBindGroup(1, groups.sea); draws += run(pass, f, PIPE.STREAK); }
  if (!skip.water) {
    if (sea.wakeLive) { pass.setPipeline(pipes.wake); pass.setVertexBuffer(0, sea.wake); pass.setIndexBuffer(sea.wakeIndex, 'uint16'); pass.drawIndexed(47 * 6); draws++; }
    if (sea.ringsLive) { pass.setPipeline(pipes.ring); pass.setVertexBuffer(0, sea.rings); pass.draw(6, 12); draws++; } // never sent, never drawn; a ring that repeats is drawn for good
    if (sea.dropsLive) { pass.setPipeline(pipes.drops); pass.setVertexBuffer(0, sea.drops); pass.draw(6, 384); draws++; }
  }
  if (!skip.smoke) draws += run(pass, f, PIPE.SMOKE);
  if (groups.label) { pass.setBindGroup(1, groups.label); draws += run(pass, f, PIPE.LABEL); } // cut out and blended, so after all that is solid
  if (!skip.haze) { pass.setBindGroup(1, groups.forest); draws += run(pass, f, PIPE.HAZE); } // blended over the land and the sky behind the crowns
  if (puffs && !skip.cloud) { pass.setPipeline(pipes.cloud); pass.setBindGroup(1, live.group); pass.setVertexBuffer(0, live.cloud); pass.setIndexBuffer(quadIndex, 'uint16'); pass.drawIndexed(6, puffs); draws++; }
  if (!skip.car) { pass.setPipeline(pipes.carBlend); pass.setBindGroup(1, car.group); pass.setVertexBuffer(0, car.vertex); pass.setIndexBuffer(car.index, 'uint32'); pass.drawIndexed(car.blended, 1, car.solid); draws++; }
  if (!skip.glow) draws += run(pass, f, PIPE.GLOW);
  if (liveGlow) { pass.setPipeline(pipes[PIPE.GLOW]); pass.setVertexBuffer(0, live.glow); pass.draw(6, liveGlow); draws++; }
  if (rocketsOut && !skip.rocket) { pass.setPipeline(pipes.jets); pass.setVertexBuffer(0, rocket.jets); pass.draw(12, 4); draws++; }
  if (blownLive && !skip.rocket) { pass.setPipeline(pipes.blown); pass.setVertexBuffer(0, rocket.puffs); pass.setIndexBuffer(quadIndex, 'uint16'); pass.drawIndexed(6, ROCKET.PUFFS); draws++; }
  if (!skip.spray) { pass.setPipeline(pipes.spray); pass.setBindGroup(1, snow.group); pass.setVertexBuffer(0, snow.spray); pass.setIndexBuffer(quadIndex, 'uint16'); pass.drawIndexed(6, 1536); draws++; }
  if (!skip.snow && snowCount) { pass.setPipeline(pipes.snow); pass.setVertexBuffer(0, snowBuffer); pass.setIndexBuffer(quadIndex, 'uint16'); pass.drawIndexed(6, snowCount); }
  pass.end();
  if (benching && timing) { enc.resolveQuerySet(timing.set, 0, STAMPS.length * 2, timing.resolve, 0); enc.copyBufferToBuffer(timing.resolve, 0, timing.read, 0, STAMPS.length * 16); }
  const t3 = performance.now();
  submitted[0] = enc.finish();
  device.queue.submit(submitted);
  const t4 = performance.now(); // the frame's time on this thread: the C, what is sent to buffers and textures, the passes' encoding, finish and submit
  stats.frames++; stats.draws = draws + 5; stats.cpu += t4 - t0;
  stats.split[0] += t1 - t0; stats.split[1] += t2 - t1; stats.split[2] += t3 - t2; stats.split[3] += t4 - t3;
}

// For measurement: every write to a buffer or texture after loading, by what it is written to
function countUploads() {
  const names = new Map([[frameBuffer, 'frame uniforms'], [snow.spray, 'spray'], [snow.ribbon, 'ribbon'], [snow.bank, 'banks'], [rocket.uniform, 'rocket'],
    [rocket.jets, 'jets'], [rocket.puffs, 'blown snow'], [live.vertex, 'live mesh'], [live.glow, 'live glow'], [live.cloud, 'clouds'], [forestList, 'forest list'],
    [sea.wake, 'wake'], [sea.rings, 'rings'], [sea.drops, 'drops'], [snow.pack, 'snow field'], [snow.latticeTex, 'snow lattice'], [car.dials, 'dash'], [car.panels, 'dash']]);
  const q = device.queue, wb = q.writeBuffer.bind(q), wt = q.writeTexture.bind(q), ce = q.copyExternalImageToTexture.bind(q);
  const add = (target, bytes) => { const k = names.get(target) || 'other'; const u = stats.up[k] ||= { bytes: 0, calls: 0 }; u.bytes += bytes; u.calls++; };
  q.writeBuffer = (b, o, d, d0 = 0, size) => { add(b, size ?? (d.byteLength - d0 * (d.BYTES_PER_ELEMENT || 1))); return wb(b, o, d, d0, size); };
  q.writeTexture = (dst, d, layout, size) => { add(dst.texture, (layout.bytesPerRow || 0) * (size[1] || size.height || 1)); return wt(dst, d, layout, size); };
  q.copyExternalImageToTexture = (src, dst, size) => { add(dst.texture, size[0] * size[1] * 4); return ce(src, dst, size); };
}

function report() {
  const n = Math.min(stats.n, 4096), t = Array.from(stats.times.subarray(0, n)).sort((a, b) => a - b);
  const q = p => t.length ? +t[Math.min(t.length - 1, Math.floor(p * t.length))].toFixed(2) : 0;
  const out = { frames: stats.frames, draws: stats.draws, cpuPerFrame: +(stats.cpu / Math.max(1, stats.frames)).toFixed(3), p50: q(0.5), p95: q(0.95), p99: q(0.99), max: q(1),
    cpu: Object.fromEntries(['c', 'upload', 'encode', 'submit'].map((k, i) => [k, +(stats.split[i] / Math.max(1, stats.frames)).toFixed(3)])), beamDraws: stats.beamDraws,
    wasmMemory: hs.memory.buffer.byteLength, size: [canvas.width, canvas.height], indices: stats.indices,
    uploads: Object.fromEntries(Object.entries(stats.up).map(([k, u]) => [k, { kbPerFrame: +(u.bytes / 1024 / Math.max(1, stats.frames)).toFixed(1), callsPerFrame: +(u.calls / Math.max(1, stats.frames)).toFixed(2) }])) };
  stats.frames = stats.n = stats.cpu = 0; stats.split.fill(0); stats.up = {};
  return out;
}

// For measurement: n frames back to back, each timed from its start until the GPU has finished it. No vsync,
// no compositor: what a frame costs, CPU and GPU together.
async function bench(n) {
  benching = true;
  const t = [], gpu = STAMPS.map(() => []), span = [], was = STAMPS.map(() => 0n);
  report();
  for (let i = 0; i < n + 20; i++) {
    const t0 = performance.now();
    frame(t0);
    await device.queue.onSubmittedWorkDone();
    if (i >= 20) t.push(performance.now() - t0);
    if (timing) { // nanoseconds at the start and end of each pass; a pass that was not drawn this frame keeps the stamps it had
      await timing.read.mapAsync(GPUMapMode.READ);
      const ns = new BigInt64Array(timing.read.getMappedRange());
      let from = 0n, to = 0n;
      STAMPS.forEach((_, k) => {
        const a = ns[k * 2], b = ns[k * 2 + 1];
        if (a === was[k]) return;
        was[k] = a;
        if (i >= 20) gpu[k].push(Number(b - a) / 1e6);
        if (!from || a < from) from = a;
        if (b > to) to = b;
      });
      if (i >= 20) span.push(Number(to - from) / 1e6);
      timing.read.unmap();
    }
  }
  benching = false; last = 0;
  const q = (a, p) => { a.sort((x, y) => x - y); return a.length ? +a[Math.min(a.length - 1, Math.floor(p * a.length))].toFixed(2) : null; };
  const r = report();
  return { n, p50: q(t, 0.5), p95: q(t, 0.95), p99: q(t, 0.99), max: q(t, 1), draws: stats.draws, beamDraws: r.beamDraws, cpuPerFrame: r.cpuPerFrame, cpu: r.cpu,
    gpu: { span: q(span, 0.5), p50: q(gpu[STAMPS.indexOf('main')], 0.5), beams: q(gpu[STAMPS.indexOf('beam0')], 0.5),
      ...Object.fromEntries(STAMPS.map((name, k) => [name, q(gpu[k], 0.5)])) } };
}

// For comparing pictures: one frame drawn into a texture of its own, not the canvas's, and its pixels read back.
let grabbing = null;
async function grab() {
  const w = canvas.width, h = canvas.height, row = (w * 4 + 255) & ~255;
  const texture = device.createTexture({ size: [w, h], format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
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

onmessage = ({ data }) => {
  if (data.init) init(data.init).catch(e => post({ error: String(e.stack || e) }));
  else if (data.resize) { resize(...data.resize); if (data.ratio) hs?.hs_set(SET.PIXEL_RATIO, data.ratio); }
  else if (data.event) hs?.hs_event(EVENT[data.event], data.arg | 0);
  else if ('keys' in data) keys = data.keys;
  else if (data.pad) pad.set(data.pad);
  else if (data.setting === 'snow') snowfall(data.value);
  else if (data.setting === 'gusts') hs?.hs_set(SET.GUSTS, data.value ? 1 : 0);
  else if ('skip' in data) { skip = {}; for (const k of data.skip.split(',')) skip[k] = true; skipDraws = data.skip.includes(':'); }
  else if (data.stats) post({ stats: report() });
  else if (data.grab) grab().then(b => post({ stats: { width: canvas.width, height: canvas.height, format, data: b } }, [b]), e => post({ error: String(e.stack || e), stats: null }));
  else if (data.bench) bench(data.bench).then(r => post({ stats: r }), e => post({ error: String(e.stack || e), stats: {} }));
};
