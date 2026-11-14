// The build worker. A second instance of harstad.wasm that exists only while the game loads: it builds all
// static geometry and the lamp-light map, hands each stage to the render worker as transferred ArrayBuffers,
// and is terminated when done, which returns its memory (the peak of the whole load) to the system.
//
// in:  { module, bin, heights, outer }   the compiled module and the three data files (transferred)
// out: { stage, first, bufs: [{ kind, data, w, h }], items, ms }   one per stage, buffers transferred
//      { done: true, ms, memory }                                  then nothing more
//
// Layouts read here are struct gfx_out, struct gfx_buf and struct gfx_item in wasm/include/harstad/gfx.h.

import { measureSigns, signAtlas, measureLabels, labelAtlas, facadeTextures, facadeStyles, cellLight } from './textures.js';

const OUT_BUFS = 20, BUF_SIZE = 20, MAX_BUFS = 64, ITEM_SIZE = 64, MAX_ITEMS = 8192, SIGN_SIZE = 20, MAX_SIGNS = 128, LABEL_SIZE = 28, BUF_SIGN_ATLAS = 4, BUF_LABEL_ATLAS = 10;

onmessage = async ({ data: { module, bin, heights, outer } }) => {
  const t0 = performance.now();
  const env = { sin: Math.sin, cos: Math.cos, tan: Math.tan, atan2: Math.atan2, exp: Math.exp, pow: Math.pow };
  const { exports: hs } = await WebAssembly.instantiate(module, { env });
  const put = file => {
    const ptr = hs.hs_alloc(file.byteLength);
    new Uint8Array(hs.memory.buffer, ptr, file.byteLength).set(new Uint8Array(file));
    return [ptr, file.byteLength];
  };
  if (hs.hs_world_init(...put(bin), ...put(heights), ...put(outer)) || hs.hs_build_init()) throw new Error('build worker: out of memory or bad data');

  const decoder = new TextDecoder();
  const text = ptr => { const b = new Uint8Array(hs.memory.buffer); let e = ptr; while (b[e]) e++; return decoder.decode(b.subarray(ptr, e)); };
  { // shop signs are lettered here, where the fonts are: tell the C side how wide each name comes out
    const at = hs.hs_alloc(512 * 8), n = hs.hs_shop_names(at, 256);
    const names = Array.from(new Uint32Array(hs.memory.buffer, at, n), text);
    new Uint16Array(hs.memory.buffer, hs.hs_shop_widths(), n).set(measureSigns(names));
    // and each text the centre's surveyed signs may bear, in its weight
    const pairs = new Uint32Array(hs.memory.buffer, at, hs.hs_sentrum_texts(at, 512) * 2);
    const texts = Array.from({ length: pairs.length / 2 }, (_, i) => ({ text: text(pairs[i * 2]), bold: pairs[i * 2 + 1] }));
    new Uint16Array(hs.memory.buffer, hs.hs_sentrum_widths(), texts.length).set(measureLabels(texts));
  }

  // what the facades' lit windows give off, timber, block, then each style, for the light they lay on the snow
  [facadeTextures(true, true), facadeTextures(false, true), ...facadeStyles(true)].forEach(([, glow], i) => new Float64Array(hs.memory.buffer, hs.hs_cell_light() + i * 384, 48).set(cellLight(glow)));

  for (;;) {
    const t = performance.now();
    const out = hs.hs_build_step();
    if (!out) throw new Error('build worker: a stage failed');
    const u32 = new Uint32Array(hs.memory.buffer);
    const [nbufs, nitems, first, stage] = u32.subarray(out >> 2, (out >> 2) + 4);
    if (!stage) break;
    const bufs = [];
    for (let i = 0; i < nbufs; i++) {
      const [ptr, size, kind, w, h] = u32.subarray((out + OUT_BUFS + i * BUF_SIZE) >> 2, ((out + OUT_BUFS + i * BUF_SIZE) >> 2) + 5);
      bufs.push({ kind, w, h, data: hs.memory.buffer.slice(ptr, ptr + size) });
    }
    const signsAt = out + OUT_BUFS + MAX_BUFS * BUF_SIZE + MAX_ITEMS * ITEM_SIZE, nsigns = u32[signsAt >> 2];
    if (nsigns) { // letter the atlas the C side has laid out
      const v = new DataView(hs.memory.buffer), signs = [];
      for (let i = 0; i < nsigns; i++) {
        const o = signsAt + 4 + i * SIGN_SIZE;
        signs.push({ name: text(v.getUint32(o, true)), x: v.getFloat32(o + 4, true), y: v.getFloat32(o + 8, true), w: v.getFloat32(o + 12, true), face: v.getUint32(o + 16, true) });
      }
      const atlas = signAtlas(signs);
      bufs.push({ kind: BUF_SIGN_ATLAS, w: atlas.width, h: atlas.height, data: atlas.data.buffer });
    }
    const labelsAt = signsAt + 4 + MAX_SIGNS * SIGN_SIZE, nlabels = u32[labelsAt >> 2];
    if (nlabels) { // and the centre's surveyed signs, an atlas of their own
      const v = new DataView(hs.memory.buffer), labels = [];
      for (let i = 0; i < nlabels; i++) {
        const o = labelsAt + 4 + i * LABEL_SIZE;
        labels.push({ name: text(v.getUint32(o, true)), x: v.getFloat32(o + 4, true), y: v.getFloat32(o + 8, true), w: v.getFloat32(o + 12, true),
          fg: v.getUint32(o + 16, true), bg: v.getUint32(o + 20, true), letters: v.getUint32(o + 24, true) });
      }
      const atlas = labelAtlas(labels);
      bufs.push({ kind: BUF_LABEL_ATLAS, w: atlas.width, h: atlas.height, data: atlas.data.buffer });
    }
    const at = out + OUT_BUFS + MAX_BUFS * BUF_SIZE;
    const items = hs.memory.buffer.slice(at, at + nitems * ITEM_SIZE);
    postMessage({ stage, first, bufs, items, ms: performance.now() - t }, [items, ...bufs.map(b => b.data)]);
  }
  postMessage({ done: true, ms: performance.now() - t0, memory: hs.memory.buffer.byteLength });
};
