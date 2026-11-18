// What a WGSL module's entry points use of its bindings, and what a pipeline built from it asks of the device.
//
// WebGPU counts a pipeline's textures, samplers and buffers per shader stage over every binding its layout gives that
// stage, used or not; a device has 16 sampled textures a stage, 8 storage buffers, 4 storage textures by default, and
// many GPUs (Intel's under Vulkan among them) offer no more. So the frame's bind group, which holds every map any
// shader reads, is laid out for each pipeline with only what that pipeline's shaders read (worker.js, frameFor), and
// this says what that is: from the entry point, every function it calls, and the module-scope variables any of them
// name. A name a function shadows with a local of its own still counts: the count can only be too high, never too low.
//
// Used by the worker as it makes each pipeline, and by the gate (tools/limits.mjs), which builds every pipeline the
// worker builds and checks each against the defaults.

export const DEFAULT_LIMITS = { sampledTextures: 16, samplers: 16, storageBuffers: 8, storageTextures: 4, uniformBuffers: 12, bindGroups: 4 };

const STAGES = { vertex: 1, fragment: 2, compute: 4 }; // GPUShaderStage's bits

// Comments out, nested block comments too, so a name in a comment is not taken for a use
const strip = code => {
  let out = '', i = 0, depth = 0;
  while (i < code.length) {
    if (depth) { if (code.startsWith('*/', i)) { depth--; i += 2; } else if (code.startsWith('/*', i)) { depth++; i += 2; } else i++; continue; }
    if (code.startsWith('//', i)) { const e = code.indexOf('\n', i); i = e < 0 ? code.length : e; continue; }
    if (code.startsWith('/*', i)) { depth++; i += 2; continue; }
    out += code[i++];
  }
  return out;
};

// The kind of a binding by its declaration: what the device's limits count it as
const kindOf = (space, type) => {
  if (space && /^uniform/.test(space)) return 'uniform';
  if (space && /^storage/.test(space)) return 'storage';
  if (/^texture_storage_/.test(type)) return 'storageTexture';
  if (/^texture_/.test(type)) return 'texture';
  if (/^sampler/.test(type)) return 'sampler';
  return 'other';
};

const cache = new Map();

// { bindings: Map name -> { group, binding, kind }, entries: [{ name, stage }], uses(entry) -> Set of binding names }
export function analyse(code) {
  const hit = cache.get(code);
  if (hit) return hit;
  const src = strip(code);
  const bindings = new Map();
  const declRe = /@group\s*\(\s*(\d+)\s*\)\s*@binding\s*\(\s*(\d+)\s*\)\s*var\s*(?:<\s*([^>]*)>)?\s*([A-Za-z_]\w*)\s*:\s*([A-Za-z_]\w*)/g;
  const declRe2 = /@binding\s*\(\s*(\d+)\s*\)\s*@group\s*\(\s*(\d+)\s*\)\s*var\s*(?:<\s*([^>]*)>)?\s*([A-Za-z_]\w*)\s*:\s*([A-Za-z_]\w*)/g;
  for (const m of src.matchAll(declRe)) bindings.set(m[4], { group: +m[1], binding: +m[2], kind: kindOf(m[3], m[5]) });
  for (const m of src.matchAll(declRe2)) bindings.set(m[4], { group: +m[2], binding: +m[1], kind: kindOf(m[3], m[5]) });
  // every function: its attributes, name and body (matched braces)
  const fns = new Map(), entries = [];
  const fnRe = /((?:@\w+(?:\s*\([^)]*\))?\s*)*)\bfn\s+([A-Za-z_]\w*)\s*\(/g;
  for (const m of src.matchAll(fnRe)) {
    let i = src.indexOf('{', m.index + m[0].length);
    // the parameters and return type may hold braces nowhere, so the first brace opens the body
    const start = i; let depth = 0;
    for (; i < src.length; i++) { if (src[i] === '{') depth++; else if (src[i] === '}' && --depth === 0) break; }
    const body = src.slice(m.index + m[0].length, i + 1);
    fns.set(m[2], new Set(body.match(/[A-Za-z_]\w*/g)));
    const stage = /@vertex/.test(m[1]) ? 'vertex' : /@fragment/.test(m[1]) ? 'fragment' : /@compute/.test(m[1]) ? 'compute' : null;
    if (stage) entries.push({ name: m[2], stage });
    void start;
  }
  const memo = new Map();
  const uses = entry => {
    if (memo.has(entry)) return memo.get(entry);
    const seen = new Set([entry]), out = new Set(), todo = [entry];
    while (todo.length) for (const id of fns.get(todo.pop()) || []) {
      if (bindings.has(id)) out.add(id);
      else if (fns.has(id) && !seen.has(id)) { seen.add(id); todo.push(id); }
    }
    memo.set(entry, out);
    return out;
  };
  const r = { bindings, entries, uses };
  cache.set(code, r);
  return r;
}

// The entry point a stage runs: the one named, else the module's only one for that stage
export function entryFor(code, stage, name) {
  const { entries } = analyse(code);
  if (name) return name;
  const all = entries.filter(e => e.stage === stage);
  if (all.length !== 1) throw new Error(`wgsl_use: ${all.length} ${stage} entry points and none named`);
  return all[0].name;
}

// Which bindings of a group the stages read: Map binding -> visibility bits. stages: [{ stage, code, entry }]
export function groupUse(stages, group) {
  const use = new Map();
  for (const { stage, code, entry } of stages) {
    const a = analyse(code), name = entryFor(code, stage, entry);
    for (const id of a.uses(name)) {
      const b = a.bindings.get(id);
      if (b.group === group) use.set(b.binding, (use.get(b.binding) || 0) | STAGES[stage]);
    }
  }
  return use;
}

// What a pipeline layout asks of a stage, from the bind group layouts' entries as given to createBindGroupLayout:
// { vertex: { sampledTextures, samplers, storageBuffers, storageTextures, uniformBuffers }, fragment, compute }
export function layoutCounts(groups) {
  const zero = () => ({ sampledTextures: 0, samplers: 0, storageBuffers: 0, storageTextures: 0, uniformBuffers: 0 });
  const out = { vertex: zero(), fragment: zero(), compute: zero() };
  for (const entries of groups) for (const e of entries) for (const [stage, bit] of Object.entries(STAGES)) {
    if (!(e.visibility & bit)) continue;
    const c = out[stage];
    if (e.texture || e.externalTexture) c.sampledTextures++;
    else if (e.sampler) c.samplers++;
    else if (e.storageTexture) c.storageTextures++;
    else if (e.buffer) { if (e.buffer.type === 'storage' || e.buffer.type === 'read-only-storage') c.storageBuffers++; else c.uniformBuffers++; }
  }
  return out;
}

// What of those is over the defaults: a list of 'fragment: 17 sampledTextures > 16', empty when all fit
export function overLimits(groups, limits = DEFAULT_LIMITS) {
  const over = [];
  if (groups.length > limits.bindGroups) over.push(`${groups.length} bind groups > ${limits.bindGroups}`);
  for (const [stage, c] of Object.entries(layoutCounts(groups)))
    for (const [k, n] of Object.entries(c)) if (n > limits[k]) over.push(`${stage}: ${n} ${k} > ${limits[k]}`);
  return over;
}
