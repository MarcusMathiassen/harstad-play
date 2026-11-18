// The build worker. A second instance of harstad.wasm that exists only while the game loads: it builds all
// static geometry and the lamp-light map, hands each stage to the render worker as transferred ArrayBuffers,
// and is terminated when done, which returns its memory (the peak of the whole load) to the system.
//
// in:  { module, bin, heights, outer, mode, lead }   the compiled module, the three data files (transferred), and what of
//      the bake this one does (enum build_maps_mode: the load runs three, one for everything but the moon's field and the
//      sky's map, and one for each of those, side by side). lead: a MessagePort to each other build worker, which it
//      sends its memory as world_init left it (a SharedArrayBuffer where the page is cross-origin isolated) before it
//      goes on, straight to them: the render worker is reading the world itself meanwhile, and would pass it on late
// in:  { module, from, mode, band }   or, instead of the files, the port that memory comes by: this one begins where the
//      lead's world_init ended, without reading the world again (the module has no state outside its memory but the
//      stack pointer, back where it began once world_init has returned)
//      band: [k, n], for band k of n of the sky's map alone (wasm/src/build.c, build_band)
//      strip: [k, n], for strip k of n of the moon's field alone (build_strip); strip 0, the lead, gets `strips`, a
//      MessagePort to each other strip, and hands out the field as one build would; another gets `toLead`, the port it
//      sends its texels of each band of rows to (BUF_MOON_STRIP), and posts nothing here but that it is done
// in:  share: [k, n], for share k of n of everything but the maps (build.h, build_unit), with start: [x, z], the units
//      taken nearest it first, and counter: an Int32Array on a SharedArrayBuffer the shares take units off, or null (then
//      every n-th from the k-th). Share 0, the lead (mode NONE), gets fromShares, a MessagePort from each other share, which
//      sends it { id, ledger } of each unit it builds; the lead takes every ledger in, in the units' order, and builds the
//      lights last. Another (mode PART) gets toShare, its port to the lead.
// out: { stage, first, bufs: [{ kind, data, w, h }], items, share, ms, mode, band, unit }   one per step (a stage, or a tile of one), buffers
//      transferred; band { k, rows }, the band's rows of the page table's tiles; unit, a share's unit (BUILD_UNIT), else 0
//      { done: true, ms, memory }                                  then nothing more
//
// Layouts read here are struct gfx_out, struct gfx_buf and struct gfx_item in wasm/include/harstad/gfx.h.

import { measureSigns, signAtlas, measureLabels, labelAtlas, facadeTextures, facadeStyles, cellLight } from './textures.js';

const OUT_BUFS = 20, BUF_SIZE = 20, MAX_BUFS = 128, ITEM_SIZE = 64, MAX_ITEMS = 65536, SIGN_SIZE = 20, MAX_SIGNS = 128, LABEL_SIZE = 28, BUF_SIGN_ATLAS = 4, BUF_LABEL_ATLAS = 10, BUF_MOON_STRIP = 25;

const MAPS = { ALL: 0, NONE: 1, MOON: 2, SKY: 3, PART: 4 }; // enum build_maps_mode
const unit = (stage, part) => (stage << 16 | part) >>> 0, BUILD_DETAIL = 9, BUILD_LIGHT = 11; // BUILD_UNIT; enum build_stage

// What comes by a port, in the order it comes: next() is the next message, now or when it comes
function queue(...ports) {
  const got = [], wants = [];
  for (const port of ports) port.onmessage = ({ data }) => wants.length ? wants.shift()(data) : got.push(data);
  return { next: () => got.length ? Promise.resolve(got.shift()) : new Promise(resolve => wants.push(resolve)) };
}

// a failure anywhere in the bake is the load's: an async handler's throw would otherwise be lost here
onmessage = e => build(e.data).catch(err => postMessage({ failed: String(err.stack || err) }));

async function build({ module, bin, heights, outer, from, lead, band: cut, strip, strips, toLead, mode = MAPS.ALL, share, start, counter, fromShares, toShare }) {
  const others = strips?.map(p => queue(p)); // the lead of the moon's strips: each other strip's bands, as they come
  const t0 = performance.now();
  const env = { sin: Math.sin, cos: Math.cos, tan: Math.tan, atan2: Math.atan2, exp: Math.exp, pow: Math.pow };
  const { exports: hs } = await WebAssembly.instantiate(module, { env });
  const put = file => {
    const ptr = hs.hs_alloc(file.byteLength);
    new Uint8Array(hs.memory.buffer, ptr, file.byteLength).set(new Uint8Array(file));
    return [ptr, file.byteLength];
  };
  if (from) { // the lead's memory, whole: its data, its heap and the world in it, at the same addresses
    const world = await new Promise(resolve => { from.onmessage = e => resolve(e.data); });
    from.close();
    hs.memory.grow((world.byteLength - hs.memory.buffer.byteLength) / 65536);
    new Uint8Array(hs.memory.buffer).set(new Uint8Array(world));
  } else if (hs.hs_world_init(...put(bin), ...put(heights), ...put(outer))) throw new Error('build worker: out of memory or bad data');
  if (lead?.length) { // one copy for all the others where it can be shared, else one each
    const n = hs.memory.buffer.byteLength, shared = self.crossOriginIsolated ? new SharedArrayBuffer(n) : null;
    if (shared) new Uint8Array(shared).set(new Uint8Array(hs.memory.buffer));
    lead.forEach(port => {
      const copy = shared || hs.memory.buffer.slice(0);
      port.postMessage(copy, shared ? [] : [copy]);
      port.close();
    });
    postMessage({ lead: performance.now() - t0 });
  }
  if (hs.hs_build_init(mode) || (share && mode === MAPS.NONE && hs.hs_build_share())) throw new Error('build worker: out of memory or bad data');
  // cut: [k, n], band k of n of the sky's map (build_band): its rows of the page table's tiles go with each step
  const rows = cut && hs.hs_build_band(...cut), band = cut ? { k: cut[0], rows: [rows & 0xffff, rows >>> 16] } : null;
  if (cut && !rows) throw new Error('build worker: no such band');
  if (strip && hs.hs_build_strip(...strip)) throw new Error('build worker: no such strip');
  let waited = 0;
  const side = mode === MAPS.MOON || mode === MAPS.SKY; // a map alone: no signs, no facades

  const decoder = new TextDecoder();
  const text = ptr => { const b = new Uint8Array(hs.memory.buffer); let e = ptr; while (b[e]) e++; return decoder.decode(b.subarray(ptr, e)); };
  if (!side) { // shop signs are lettered here, where the fonts are: tell the C side how wide each name comes out
    const at = hs.hs_alloc(512 * 8), n = hs.hs_shop_names(at, 256);
    const names = Array.from(new Uint32Array(hs.memory.buffer, at, n), text);
    new Uint16Array(hs.memory.buffer, hs.hs_shop_widths(), n).set(measureSigns(names));
    // and each text the centre's surveyed signs may bear, in its weight
    const pairs = new Uint32Array(hs.memory.buffer, at, hs.hs_sentrum_texts(at, 512) * 2);
    const texts = Array.from({ length: pairs.length / 2 }, (_, i) => ({ text: text(pairs[i * 2]), bold: pairs[i * 2 + 1] }));
    new Uint16Array(hs.memory.buffer, hs.hs_sentrum_widths(), texts.length).set(measureLabels(texts));
  }

  // what the facades' lit windows give off, timber, block, then each style, for the light they lay on the snow
  if (!side) [facadeTextures(true, true), facadeTextures(false, true), ...facadeStyles(true)].forEach(([, glow], i) => new Float64Array(hs.memory.buffer, hs.hs_cell_light() + i * 384, 48).set(cellLight(glow)));

  // one step's output to the render worker: its buffers and items, transferred, and the signs' and labels' atlases lettered here
  const hand = (out, ms, id) => {
    const u32 = new Uint32Array(hs.memory.buffer);
    const [nbufs, nitems, first, stage] = u32.subarray(out >> 2, (out >> 2) + 4);
    const share = new Float32Array(hs.memory.buffer, out + 16, 1)[0]; // of its stage: a tile's, where the stage goes a tile at a time
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
    if (!others || nbufs || nitems || share) postMessage({ stage, first, bufs, items, share, mode, band, unit: id, ms }, [items, ...bufs.map(b => b.data)]);
  };

  if (share) { // a share of the geometry: units as they are taken (wasm/src/build.c, build_unit), the nearest the start first
    const [k, n] = share, cap = 1 << 18, at = hs.hs_alloc(cap * 4);
    const ids = new Uint32Array(hs.memory.buffer, at, hs.hs_build_units(start[0], start[1], at, cap)).slice();
    if (!ids.length) throw new Error('build worker: no units');
    // taken off one count all the shares hold, where memory can be shared; else every n-th, from the k-th
    let mine = k - n;
    const next = counter ? () => Atomics.add(counter, 0, 1) : () => (mine += n);
    const lead = mode === MAPS.NONE, kept = new Map(); // the lead's: every unit's ledger, until its turn
    const one = id => {
      const t = performance.now(), out = hs.hs_build_unit(id);
      if (!out) throw new Error(`build worker: unit ${id >>> 16}.${id & 0xffff} failed`);
      hand(out, performance.now() - t, id);
      const ledger = hs.memory.buffer.slice(hs.hs_ledger(), hs.hs_ledger() + hs.hs_ledger_size());
      if (lead) kept.set(id, ledger);
      else toShare.postMessage({ id, ledger }, [ledger]);
    };
    if (lead) one(unit(BUILD_DETAIL, 0)); // the details look at the sea, which the lead alone bakes
    for (let i; (i = next()) < ids.length;) one(ids[i]);
    if (lead) { // every ledger in, in one build's order, as they come in; then the lights
      const order = [...ids, unit(BUILD_DETAIL, 0)].sort((a, b) => a - b), from = queue(...fromShares);
      let cursor = 0;
      const take = () => {
        for (let l; cursor < order.length && (l = kept.get(order[cursor])); cursor++) {
          const room = hs.hs_ledger_room(l.byteLength);
          if (!room) throw new Error('build worker: no room for a ledger');
          new Uint8Array(hs.memory.buffer, room, l.byteLength).set(new Uint8Array(l));
          if (hs.hs_ledger_put(room, l.byteLength)) throw new Error('build worker: a ledger would not go in');
          kept.delete(order[cursor]);
        }
      };
      const w0 = performance.now();
      take();
      while (cursor < order.length) {
        const { id, ledger } = await from.next();
        kept.set(id, ledger);
        take();
      }
      waited += performance.now() - w0;
      const t = performance.now(), out = hs.hs_build_unit(unit(BUILD_LIGHT, 0));
      if (!out) throw new Error('build worker: the lights failed');
      hand(out, performance.now() - t, unit(BUILD_LIGHT, 0));
      fromShares.forEach(p => p.close());
    }
    toShare?.close();
  } else for (;;) {
    const t = performance.now();
    const out = hs.hs_build_step();
    if (!out) throw new Error('build worker: a stage failed');
    const u32 = new Uint32Array(hs.memory.buffer);
    const [nbufs, , , stage] = u32.subarray(out >> 2, (out >> 2) + 4);
    if (!stage) break;
    if (toLead) { // a strip not the lead: its texels of a band to the lead, and nothing here
      for (let i = 0; i < nbufs; i++) {
        const [ptr, size, kind] = u32.subarray((out + OUT_BUFS + i * BUF_SIZE) >> 2, ((out + OUT_BUFS + i * BUF_SIZE) >> 2) + 3);
        if (kind !== BUF_MOON_STRIP) continue;
        const data = hs.memory.buffer.slice(ptr, ptr + size);
        toLead.postMessage(data, [data]);
      }
      continue;
    }
    hand(out, performance.now() - t, 0);
    // the lead of the moon's strips: the other strips' texels of the band it waits for, laid in before it goes on
    for (let bn; others && (bn = hs.hs_strip_wait()) >= 0;) {
      const w0 = performance.now();
      for (let k = 1; k < others.length + 1; k++) {
        const pack = await others[k - 1].next();
        if (pack.byteLength > hs.hs_strip_room()) throw new Error('build worker: a strip too large');
        new Uint8Array(hs.memory.buffer, hs.hs_strip_stage(), pack.byteLength).set(new Uint8Array(pack));
        if (hs.hs_strip_put(k, pack.byteLength)) throw new Error(`build worker: strip ${k} of band ${bn} would not go in`);
      }
      waited += performance.now() - w0;
    }
  }
  toLead?.close();
  postMessage({ done: true, ms: performance.now() - t0, memory: hs.memory.buffer.byteLength, waited });
}
