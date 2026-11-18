// Main thread of the C / WebGPU build: the page, the keys, the gamepad, the HUD and the sound. Nothing here renders or
// simulates; the canvas is handed to src/worker.js, which does both (see the diagram at its top).
//
// To the worker:   { init }, { resize }, { event, arg } by postMessage; keys held and the gamepad through the shared block
// From the worker: { progress }, { minimap }, { ready, marks, names, kinds }, { error }; the HUD through the
//                  shared block. When the page is not cross-origin isolated the keys go as { keys } and the gamepad as { pad } messages, and the HUD
//                  comes as a bare ArrayBuffer of struct game_hud, sent only when what it says has changed (worker.js)
import { startAudio } from './audio.js';
import { readOutline, outlineDrawer } from './outline.js';
import { deskInit } from './desk.js';

// the map in outline comes first, ahead of everything the worker fetches: it is drawn while the rest loads
const outlineData = fetch('data/outline.bin').then(r => r.ok ? r.arrayBuffer() : Promise.reject(new Error(r.status))).then(readOutline);
const $ = id => document.getElementById(id);
const params = new URLSearchParams(location.search); // ?auto&x=&z=&h=&at=&cam=&drive=&fly=&orbit=&top= for testing
// The opening: the map in outline as the town loads, then the town itself from straight over it, and a zoom down to the
// car (wasm/src/cam.c, cam__intro), this many seconds. Where the player would rather less moved it is shorter, and goes
// at once from over the town, without the moment's wait. ?intro=0 leaves it out, as do ?auto and ?still, which start as
// the game always has: ready, the title over a slow orbit, a click to drive
const CALM = matchMedia('(prefers-reduced-motion: reduce)').matches;
// ?intro=8.5 asked for plays the opening even under ?auto or ?still (the streaming check films it: tools/stream_check.mjs)
const INTRO = +params.get('intro') > 0 ? +params.get('intro') : params.get('intro') === '0' || params.has('auto') || params.has('still') ? 0 : CALM ? 5 : 8.5;
const INTRO_HOLD = CALM ? 0 : 1200; // ms the whole town is shown from over it, once it has faded in, before the zoom
// E: the boost on or off in flight, and on the road a gear up with the box in the driver's hands; Q a gear down
const KEY = { KeyW: 1, ArrowUp: 1, KeyS: 2, ArrowDown: 2, KeyA: 4, ArrowLeft: 4, KeyD: 8, ArrowRight: 8, Space: 16, ShiftLeft: 32, ShiftRight: 32, KeyE: 64 | 128, KeyQ: 256 };
const CAMS = ['Chase', 'Far chase', 'Cockpit', 'Hood', 'Bumper', 'Wheel', 'Orbit', 'Overhead', 'Trackside'];
const HUD = 64; // offset of struct game_hud in the shared block

// Without WebGPU there is no game: the title says so.
if (!navigator.gpu) $('go').textContent = 'This needs a browser with WebGPU';

const canvas = $('view');
const pixels = () => { const r = Math.min(devicePixelRatio, 2); return [Math.round(innerWidth * r), Math.round(innerHeight * r)]; };
// The player's settings, kept in the browser between visits; a parameter in the address wins over what was kept
// (only what the player changed is kept, so a default changed later still reaches everyone who left it alone)
const SETTINGS = 'harstad.settings.v2';
const DEFAULTS = { gusts: true, haze: 1, traffic: true, aa: 4, scale: 1, sound: true, music: true, hud: true, manual: false, assist: true, beam: 2 }; // (the falling snow is the weather's: no setting)
const settings = { ...DEFAULTS };
try { Object.assign(settings, JSON.parse(localStorage.getItem(SETTINGS) || '{}')); } catch { /* private window, or blocked */ }
if (params.has('traffic')) settings.traffic = params.get('traffic') !== '0';
if (params.has('aa')) settings.aa = +params.get('aa') === 4 ? 4 : 1;
if (params.has('beam')) settings.beam = Math.min(2, Math.max(0, +params.get('beam') || 0)); // ?beam=1: full beam on (0 off, 2 automatic)
if (params.has('scale')) settings.scale = Math.min(1, Math.max(0.5, +params.get('scale') || 1));
const keep = () => { try { localStorage.setItem(SETTINGS, JSON.stringify(Object.fromEntries(Object.entries(settings).filter(([k, v]) => v !== DEFAULTS[k])))); } catch { /* not kept: fine */ } };

const shared = self.crossOriginIsolated ? new SharedArrayBuffer(208) : null;
const block = shared || new ArrayBuffer(208);
const keysOut = new Uint32Array(block, 0, 1), padOut = new Float32Array(block, 4, 6), hudF = new Float32Array(block, HUD, 11), hudI = new Int32Array(block, HUD + 44, 5), hudIn = new Uint8Array(block, HUD, 136), hudB = new Float32Array(block, HUD + 64, 8), hudR = new Float32Array(block, HUD + 96, 4), hudS = new Float32Array(block, HUD + 112, 6); // hudS: each engine's spool (struct game_hud) // hudB: the boost, the turbine, the boost's switch; the wings, and what of them moves; hudR: the road felt, thump, snow, deep, lock (struct game_hud)

const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
worker.onerror = e => { window.__next.error = e.message || 'the render worker failed to load'; $('go').textContent = window.__next.error; };
const offscreen = canvas.transferControlToOffscreen();
const [width, height] = pixels();
worker.postMessage({ init: { canvas: offscreen, shared, width, height, ratio: Math.min(devicePixelRatio, 2),
  params: { ...Object.fromEntries(params), intro: String(INTRO), gusts: settings.gusts ? '1' : '0', haze: String(settings.haze), traffic: settings.traffic ? '1' : '0', aa: String(settings.aa), scale: String(settings.scale),
  hdr: params.get('hdr') ?? (matchMedia('(dynamic-range: high)').matches ? '1' : '0') } } }, [offscreen]); // an HDR screen: highlights over white (worker.js configure)
addEventListener('resize', () => worker.postMessage({ resize: pixels(), ratio: Math.min(devicePixelRatio, 2) }));

/* ---- the opening ---- */

const outline = $('outline');
let loaded = 0, drawOutline = null, skipped = false, zooming = false;
function sizeOutline() {
  const [w, h] = pixels();
  if (outline.width !== w || outline.height !== h) { outline.width = w; outline.height = h; }
  drawOutline?.(loaded);
}
outlineData.then(map => { drawOutline = outlineDrawer(outline, map, Math.min(devicePixelRatio, 2)); sizeOutline(); }, e => console.warn('no outline:', e.message));
sizeOutline();
addEventListener('resize', sizeOutline);
$('title').classList.add('intro');

// The town is ready: the outline fades into it, seen from where it was drawn; over the town a moment, then the zoom down to
// the car. Skipped, all of it hurries: the fade, and the zoom, which comes to the car in a second or so (GAME_INTRO_CATCH)
function reveal() {
  loaded = 1; drawOutline?.(1);
  outline.classList.toggle('quick', skipped);
  outline.classList.add('gone');
  outline.addEventListener('transitionend', () => { outline.style.display = 'none'; }, { once: true });
  if (!INTRO) { $('title').classList.remove('intro'); $('go').textContent = 'Click to drive'; return; }
  $('go').textContent = skipped ? '' : 'Any key to skip';
  if (!params.has('introat') && !params.has('auto')) setTimeout(() => zoom(), skipped ? 500 : 1600 + INTRO_HOLD); // ?introat: held where it is set (worker.js)
}
function zoom() {
  if (zooming) return;
  zooming = true;
  $('title').classList.add('gone');
  send('INTRO', skipped ? 1 : 0);
}
// a key, a click or a button: during the opening, on to the game
function skip() {
  if (!INTRO) return begin();
  if (!skipped) { skipped = true; if (ready) { zooming ? send('INTRO', 1) : zoom(); } }
  sound();
}

let ready = false, started = false, audio = null, names = [], kinds = [], minimap = null, statsWaiting = [];
const t0 = performance.now();
worker.onmessage = ({ data }) => {
  if (data instanceof ArrayBuffer) hudIn.set(new Uint8Array(data)); // into the block the views above were made on, once
  else if (data.progress) { if (!ready) { loaded = data.progress; drawOutline?.(loaded); $('go').textContent = `Loading the town · ${Math.floor(loaded * 100)} %`; } }
  else if (data.minimap) { // the town's roads, drawn by the worker: shown as they come, never painted again
    minimap = data;
    const town = $('town');
    town.width = data.minimap.width; town.height = data.minimap.height;
    town.getContext('bitmaprenderer').transferFromImageBitmap(data.minimap);
  }
  else if (data.ready) {
    ready = true; names = data.names; kinds = data.kinds;
    send('GEARBOX', settings.manual ? 1 : 0); send('ASSIST', settings.assist ? 1 : 0); send('BEAM', settings.beam); send('STILL', CALM ? 1 : 0);
    window.__next.marks = { ...data.marks, total: performance.now() - t0 };
    reveal();
    // ?auto starts at once, as the game does when it loads itself again (anti-aliasing changed, the GPU lost): with its
    // sound, which the browser lets play from the first key or click; ?mute for a run that should make none
    // ?hold=20: the title held that many seconds first, for a run whose play begins with all it streams in (a reference)
    // with ?intro=8.5 the zoom is set going then instead; ?bird=4000: out over the car that many m up as the play starts (worker.js)
    if (params.has('auto')) setTimeout(() => INTRO ? zoom() : begin(!params.has('mute')), (+params.get('hold') || 0) * 1000);
  } else if (data.stats) statsWaiting.shift()?.(data.stats);
  else if (data.limits) window.__next.limits = data.limits; // ?limits: what each pipeline asks of the device (worker.js, watchLimits)
  else if (data.lost) { // the GPU device is gone and with it everything drawn from: start again where the car is, but not over and over
    let last = 0;
    try { last = +sessionStorage.getItem('harstad.lost') || 0; sessionStorage.setItem('harstad.lost', Date.now()); } catch { last = Date.now(); }
    if (Date.now() - last > 30000) {
      for (const [k, v] of [['x', hudF[3]], ['z', hudF[4]], ['h', hudF[2]], ['cam', hudI[2]]]) params.set(k, v);
      if (started) params.set('auto', '');
      location.replace(location.pathname + '?' + params);
    } else { window.__next.error = data.lost; $('go').textContent = data.lost; }
  }
  else if (data.error) { console.error(data.error); (window.__next.errors ||= []).push(data.error); $('go').textContent = data.error.split('\n')[0]; window.__next.error = data.error; }
};
// for tests and measurements: the load timeline, and frame statistics since the last call
window.__next = { marks: null, error: null, skip: parts => worker.postMessage({ skip: parts }), // leave parts of the frame out, to see what each costs
  intro: () => ({ loaded, ready, skipped, zooming, started }), // where the opening is
  stats: () => new Promise(r => { statsWaiting.push(r); worker.postMessage({ stats: true }); }),
  bench: n => new Promise(r => { statsWaiting.push(r); worker.postMessage({ bench: n }); }),
  grab: () => new Promise(r => { statsWaiting.push(r); worker.postMessage({ grab: true }); }),
  sceneStats: box => new Promise(r => { statsWaiting.push(r); worker.postMessage({ sceneStats: box || true }); }), // (?hdr=1) the float picture's brightest values (and in box, [x0, y0, x1, y1] of the frame), worker.js sceneStats // one frame's pixels, drawn apart from the canvas: { width, height, format, data }
  look: p => new Promise(r => { statsWaiting.push(r); worker.postMessage({ look: p }); }), // the next view in this page, as its address would set it up: look({ x, z, h, cam, still, ... }) (tools/look/run.mjs)
  film: () => new Promise(r => { statsWaiting.push(r); worker.postMessage({ film: true }); }), // ?film=N: { frames, across, down, data } (worker.js, filmRecord)
  meet: (what, value) => worker.postMessage({ meet: [what, value] }), // the car put by the traffic again, as ?truck, ?plough, ?plough2, ?gritter, ?meet put it: meet('PLOUGH', -14)
  hud: () => ({ speed: hudF[0], alt: hudF[1], heading: hudF[2], x: hudF[3], z: hudF[4], y: hudF[5], road: hudI[0], cam: hudI[2], flags: hudI[3], frame: hudI[4], boost: hudB[0], turbine: hudB[1], on: hudB[2] }),
  bird: (...a) => worker.postMessage({ bird: a }), // the bird's-eye view moved, as the mouse moves it: bird(3, x, z, 300) across to over (x, z) (wasm.c, hs_bird)
  follow: car => worker.postMessage({ follow: car }), rate: n => worker.postMessage({ rate: n }),
  probe: () => desk.probe }; // what the desk last heard: struct desk_probe

/* ---- input ---- */

let toastTimer = 0, held = 0, keyHeld = 0, padHeld = 0; // held: the keyboard's keys and the gamepad's buttons that stand for keys, together
function toast(text) {
  $('toast').textContent = text;
  $('toast').classList.add('on');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => $('toast').classList.remove('on'), 1800);
}
const send = (event, arg = 0) => worker.postMessage({ event, arg });
function setKeys(k, p = padHeld) {
  keyHeld = k; padHeld = p;
  if ((k | p) === held) return;
  held = k | p;
  if (shared) Atomics.store(keysOut, 0, held); else worker.postMessage({ keys: held });
}
function toggleFlight(how) {
  send('FLY_TOGGLE');
  toast(hudI[3] & 1 ? 'Flight off' : 'Flight on — ' + how);
}
function setCam(i) {
  const cam = (i + CAMS.length) % CAMS.length;
  send('CAM_SET', cam);
  toast(CAMS[cam]);
  if (audio) audio.setCabin(cam === 2);
}

// The sound starts with the first key or click it may (the browser's rule), in the opening or after it
function sound() {
  if (audio || !ready || params.has('mute')) return;
  audio = startAudio(); audio.setCabin(hudI[2] === 2);
  if (!settings.sound) audio.toggle();
  if (!settings.music) audio.toggleMusic();
}
// the game has started: by the title's going, or as the opening zoom comes to the car (the started flag in the HUD)
function play() {
  started = true;
  $('title').classList.add('gone');
  $('hud').classList.toggle('on', settings.hud);
  setTimeout(() => $('help').classList.add('dim'), 14000);
}
function begin(withSound = true) {
  if (started || !ready || INTRO) return;
  send('START');
  if (withSound) sound();
  play();
}
addEventListener('click', e => { if (!started && !e.target.closest('#settings, #gear')) skip(); else sound(); });

addEventListener('keydown', e => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (KEY[e.code]) setKeys(keyHeld | KEY[e.code]);
  if (e.code === 'Space' || e.code.startsWith('Arrow')) e.preventDefault();
  if (!started) { if (INTRO ? e.code !== 'KeyO' && e.code !== 'KeyT' : e.code === 'Enter' || e.code === 'Space' || e.code === 'KeyW') skip(); return; }
  sound();
  if (e.repeat) return;
  if (e.code === 'KeyC') setCam(hudI[2] + (e.shiftKey ? -1 : 1));
  if (/^Digit[1-9]$/.test(e.code)) setCam(+e.code[5] - 1);
  if (e.code === 'KeyE' && hudI[3] & 1) toast(hudB[2] > 0.5 ? 'Boost off: the turbine goes back in' : 'Boost on: the turbine comes out');
  if (e.code === 'KeyF') toggleFlight('Space to climb, Shift to descend, E for the boost');
  if (e.code === 'KeyM') change('sound', !settings.sound, true);
  if (e.code === 'KeyN') change('music', !settings.music, true);
  if (e.code === 'KeyH') change('hud', !settings.hud);
  if (e.code === 'KeyO') $('settings').classList.toggle('on');
  if (e.code === 'Escape') { $('settings').classList.remove('on'); if (!desk.disarm() && birdOut) bird(false); }
  if (e.code === 'KeyR') send('BACK_TO_ROAD');
  if (e.code === 'KeyB') bird(!birdOut);
  if (e.code === 'KeyT') desk.toggle();
  if (e.code === 'KeyG') change('manual', !settings.manual, true);
  if (e.code === 'KeyY') change('assist', !settings.assist, true);
  if (e.code === 'KeyL') nextBeam();
});

/* ---- the bird's-eye view and the debug desk ----
 *
 * B: straight down over the car from a little way up, and back down to it from wherever it went (wasm/src/game.c, game_bird). Out there the mouse wheel or a pinch
 * comes in or goes out towards the cursor, a drag moves the map, a right-drag (or Shift and a drag) turns it round and
 * tips it; the keys move it too (W A S D, Space in, Shift out), and the gamepad (left stick, the triggers, the right
 * stick). T opens the desk (src/desk.js): choose what is to happen and click the map. */

let birdOut = false, drag = null;
// driving, a drag with the mouse looks round as the right stick does, and let go the view comes back (wasm/src/cam.c)
const mouseLook = [0, 0], LOOK_PX = [320, 220];
function bird(out) {
  if (!started || out === birdOut) return;
  birdOut = out;
  send('BIRD', out ? 1 : 0);
  $('bird').classList.toggle('on', out);
  document.body.classList.toggle('birding', out);
  if (out) toast("Bird's-eye");
}
const desk = deskInit({ worker, toast, names: () => names, settings, change: (what, value) => change(what, value) });
const ndc = (x, y) => [x / innerWidth * 2 - 1, 1 - y / innerHeight * 2];
const onMap = e => started && !e.target.closest('#settings, #gear, #desk');
addEventListener('pointerdown', e => {
  if (!onMap(e)) return;
  drag = { x: e.clientX, y: e.clientY, button: e.button, turn: e.button === 2 || e.shiftKey, moved: 0 };
});
addEventListener('pointermove', e => {
  const [x, y] = ndc(e.clientX, e.clientY);
  desk.point(x, y, onMap(e) ? 1 : 0);
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  drag.moved += Math.abs(dx) + Math.abs(dy);
  if (birdOut && drag.moved > 3) {
    if (drag.turn) worker.postMessage({ bird: [2, -dx * 0.006, -dy * 0.004, 0, 0] }); // round what it looks at, and tipped
    else worker.postMessage({ bird: [1, ...ndc(drag.x, drag.y), x, y] }); // the ground under the cursor goes with it
  }
  if (!birdOut && drag.button === 0 && drag.moved > 3) {
    mouseLook[0] = Math.max(-1, Math.min(1, mouseLook[0] + dx / LOOK_PX[0]));
    mouseLook[1] = Math.max(-1, Math.min(1, mouseLook[1] - dy / LOOK_PX[1]));
  }
  drag.x = e.clientX; drag.y = e.clientY;
});
addEventListener('pointerup', e => {
  if (drag && drag.moved <= 4 && drag.button === 0 && onMap(e)) desk.act(...ndc(e.clientX, e.clientY));
  drag = null; mouseLook[0] = mouseLook[1] = 0;
});
addEventListener('wheel', e => {
  if (!birdOut || !onMap(e)) return;
  e.preventDefault();
  const k = Math.exp(e.deltaY * (e.deltaMode === 1 ? 16 : 1) * (e.ctrlKey ? 0.01 : 0.0015)); // a pinch comes as a wheel with Ctrl, in finer steps
  worker.postMessage({ bird: [0, ...ndc(e.clientX, e.clientY), k, 0] });
}, { passive: false });
addEventListener('contextmenu', e => { if (birdOut && onMap(e)) e.preventDefault(); });
/* ---- settings ---- */

// Apply one setting, keep it, and show it in the panel. The gusts, the haze and the traffic change live; anti-aliasing builds the
// pipelines anew, so the game is loaded again where the car is.
function change(what, value, say = false) {
  if (settings[what] === value) return;
  settings[what] = value;
  keep();
  if (what === 'gusts') worker.postMessage({ setting: 'gusts', value });
  if (what === 'haze') worker.postMessage({ setting: 'haze', value });
  if (what === 'traffic') worker.postMessage({ setting: 'traffic', value });
  if (what === 'sound' && audio) audio.toggle();
  if (what === 'music' && audio) audio.toggleMusic();
  if (what === 'hud') $('hud').classList.toggle('on', value);
  if (what === 'manual') { send('GEARBOX', value ? 1 : 0); if (say) toast(value ? 'Manual gearbox — E up · Q down' : 'Automatic gearbox'); }
  if (what === 'assist') { send('ASSIST', value ? 1 : 0); if (say) toast(value ? 'Assists on: traction control, steering help' : 'Assists off'); }
  if (what === 'beam') { send('BEAM', value); if (say) toast(BEAMS[value]); }
  if (say && (what === 'sound' || what === 'music')) toast(`${what === 'sound' ? 'Sound' : 'Music'} ${value ? 'on' : 'off'}`);
  if (what === 'aa' || what === 'scale') {
    for (const [k, v] of [['x', hudF[3]], ['z', hudF[4]], ['h', hudF[2]], ['cam', hudI[2]]]) params.set(k, v);
    params.set(what, value);
    if (started) params.set('auto', '');
    location.replace(location.pathname + '?' + params);
  }
  shown();
}
// full beam: off (dipped), on, or automatic as the traffic's drivers keep theirs (wasm/src/traffic.c, traffic_beam_keep)
const BEAMS = ['Full beam off: dipped', 'Full beam on', 'Full beam automatic: dipped for others and where the road is lit'];
const nextBeam = () => change('beam', [1, 2, 0][settings.beam] ?? 2, true); // automatic, on, off, and round
function shown() {
  $('s-haze').value = settings.haze; $('s-haze-v').textContent = `${settings.haze}×`;
  $('s-gusts').checked = settings.gusts; $('s-traffic').checked = settings.traffic; $('s-aa').value = settings.aa; $('s-scale').value = settings.scale;
  $('s-sound').checked = settings.sound; $('s-music').checked = settings.music; $('s-hud').checked = settings.hud;
  $('s-manual').checked = settings.manual; $('s-assist').checked = settings.assist; $('s-beam').value = settings.beam;
}
shown();
$('gear').addEventListener('click', () => $('settings').classList.toggle('on'));
$('s-haze').addEventListener('input', e => change('haze', +e.target.value));
$('s-gusts').addEventListener('change', e => change('gusts', e.target.checked));
$('s-traffic').addEventListener('change', e => change('traffic', e.target.checked));
$('s-aa').addEventListener('change', e => change('aa', +e.target.value));
$('s-scale').addEventListener('change', e => change('scale', +e.target.value));
$('s-sound').addEventListener('change', e => change('sound', e.target.checked));
$('s-music').addEventListener('change', e => change('music', e.target.checked));
$('s-hud').addEventListener('change', e => change('hud', e.target.checked));
$('s-manual').addEventListener('change', e => change('manual', e.target.checked));
$('s-assist').addEventListener('change', e => change('assist', e.target.checked));
$('s-beam').addEventListener('change', e => change('beam', +e.target.value));
// the panel's controls take keys and clicks of their own: none of them drives the car or starts the game
$('settings').addEventListener('keydown', e => e.stopPropagation());
$('settings').addEventListener('click', e => e.stopPropagation());

// two keys can map to one bit (W and the up arrow), so a release re-reads what is still down
const down = new Set();
addEventListener('keydown', e => down.add(e.code), true);
addEventListener('keyup', e => { down.delete(e.code); let k = 0; for (const c of down) k |= KEY[c] || 0; setKeys(k); });
addEventListener('blur', () => { down.clear(); setKeys(0); });

/* ---- gamepad ----
 *
 * Read once a frame, in the standard layout. What is analogue goes to the game as struct game_pad (game.h); the buttons
 * that do what a key does are held as that key; the rest are the keyboard's one-off commands.
 *
 *   right trigger  throttle             left stick     steer; in flight, pull back to climb
 *   left trigger   brake, then reverse  right stick    look round
 *   A  handbrake; in flight, climb      X  in flight, descend       Y  flight      B  back to road
 *   left stick pressed in  in flight, the boost on or off
 *   LB, RB  camera                      d-pad left, right  steer    Back  hide the HUD      Start, A  begin
 *   right stick pressed in, X (on the road)  a gear up, down, with the box in the driver's hands
 */
const PAD_HELD = [[0, 16], [2, 32], [14, 4], [15, 8], [10, 64], [11, 128], [2, 256]]; // button, KEY bit
const STICK_DEAD = 0.14, TRIGGER_DEAD = 0.04, STEER_CURVE = 1.5; // a worn stick rests off centre; small steering needs more of the stick's travel than large
const pad = new Float32Array(6); // go, stop, steer, climb, look right, look up
let padWas = 0, padLive = false;
// what is left of a stick once its dead zone is taken out, as far along its own direction: no notch at the axes
function stick(x = 0, y = 0) {
  const r = Math.hypot(x, y);
  if (r <= STICK_DEAD) return [0, 0];
  const k = Math.min(1, (r - STICK_DEAD) / (1 - STICK_DEAD)) / r;
  return [x * k, y * k];
}
const trigger = b => { const v = b ? b.value || +b.pressed : 0; return v <= TRIGGER_DEAD ? 0 : Math.min(1, (v - TRIGGER_DEAD) / (1 - TRIGGER_DEAD)); };
let padNow = null; // the gamepad read this frame, for its rumble
function readPad() {
  let p = null;
  try { for (const g of navigator.getGamepads?.() || []) if (g && g.connected && (!p || (g.mapping === 'standard' && p.mapping !== 'standard'))) p = g; } catch { /* not allowed here */ }
  padNow = p;
  if (!p && !padLive && !mouseLook[0] && !mouseLook[1] && !pad[4] && !pad[5]) return;
  padLive = !!p;
  const b = p ? p.buttons : [], a = p ? p.axes : [];
  const [sx, sy] = stick(a[0], a[1]), [lx, ly] = stick(a[2], a[3]);
  const next = [trigger(b[7]), trigger(b[6]), -Math.sign(sx) * Math.abs(sx) ** STEER_CURVE, sy,
    Math.max(-1, Math.min(1, lx + mouseLook[0])), Math.max(-1, Math.min(1, -ly + mouseLook[1]))];
  if (next.some((v, i) => v !== pad[i])) {
    pad.set(next);
    if (shared) padOut.set(pad); else worker.postMessage({ pad: next });
  }
  let now = 0, keys = 0;
  for (let i = 0; i < b.length && i < 17; i++) if (b[i].pressed) now |= 1 << i;
  for (const [i, k] of PAD_HELD) if (now & 1 << i) keys |= k;
  setKeys(keyHeld, keys);
  const hit = now & ~padWas;
  padWas = now;
  if (!hit) return;
  if (!started) { if (INTRO || hit & (1 << 0 | 1 << 9)) skip(); return; }
  sound();
  if (hit & 1 << 5) setCam(hudI[2] + 1);
  if (hit & 1 << 4) setCam(hudI[2] - 1);
  if (hit & 1 << 3) toggleFlight('pull back or A to climb, X to descend, left stick in for the boost');
  if (hit & 1 << 10 && hudI[3] & 1) toast(hudB[2] > 0.5 ? 'Boost off: the turbine goes back in' : 'Boost on: the turbine comes out');
  if (hit & 1 << 12) bird(!birdOut); // d-pad up: the bird's-eye view, and B back down from it
  if (hit & 1 << 1) { if (birdOut) bird(false); else send('BACK_TO_ROAD'); }
  if (hit & 1 << 8) change('hud', !settings.hud);
  if (hit & 1 << 13) nextBeam(); // d-pad down: the full beam
}
addEventListener('gamepadconnected', () => toast('Gamepad — RT drive · LT brake · right stick look'));

/* ---- the gamepad's rumble ----
 * Where the Gamepad API can shake the pad: a kick from the springs when they are hit, a judder while the wheels are
 * locked and dragged along, and a fine buzz from the ruts of deep snow and the setts, rising with speed. In short pulses,
 * renewed while there is something to feel. */
let rumbleAt = 0;
function rumble(dt) {
  const act = padNow?.vibrationActuator;
  if (!act?.playEffect || (rumbleAt -= dt) > 0) return;
  const v = Math.abs(hudF[0]), flags = hudI[3], ground = flags & 2 ? 0 : 1;
  const strong = Math.min(1, hudR[0] * 0.9 + hudR[3] * Math.min(1, v / 8) * 0.35) * ground;
  const weak = Math.min(1, (hudR[2] * 0.3 + hudR[1] * 0.08) * Math.min(1, v / 15) + Math.min(1, hudF[8]) * 0.25) * ground;
  if (strong < 0.03 && weak < 0.03) return;
  rumbleAt = 0.1;
  act.playEffect('dual-rumble', { duration: 140, strongMagnitude: strong, weakMagnitude: weak }).catch(() => {});
}

/* ---- hud and sound, once a frame from whatever the worker last wrote ---- */

const town = $('town');
let shownStreet = '', streetTimer = 0, last = performance.now();
const state = { fwd: { y: 0 }, road: null }, input = {}, roadKind = { k: '' };

const hudShown = { kmh: -1, limit: -1, alt: -1, x: NaN, z: NaN, heading: NaN, revs: -1, gear: '', tc: -1 };
const DASH_RPM = 6600 / 7000; // the car's rpm of 1 (CAR_DIAL_RPM) as a share of the dial's 7,000 r/min // what the page last showed: nothing is written to it, or drawn, that has not changed
function tick(now) {
  requestAnimationFrame(tick);
  const dt = Math.min(0.05, Math.max(0, (now - last) / 1000));
  last = now;
  readPad();
  if (!started && ready && INTRO && hudI[3] & 16) play(); // the zoom has come to the car
  if (!started) return;
  rumble(dt);
  const speed = hudF[0], alt = hudF[1], heading = hudF[2], x = hudF[3], z = hudF[4], road = hudI[0], flags = hudI[3];
  const kmh = Math.round(Math.abs(speed) * 3.6), limit = (flags >>> 8) & 255, up = alt > 1.5 ? Math.round(alt) : 0;
  if (kmh !== hudShown.kmh) $('speed').textContent = hudShown.kmh = kmh;
  if (limit !== hudShown.limit) { hudShown.limit = limit; $('limit').textContent = limit || ''; $('limit').classList.toggle('on', limit > 0); }
  if (up !== hudShown.alt) { hudShown.alt = up; $('alt').textContent = up ? ` · ${up} m` : ''; }
  { // the dial: the revs to a few hundredths of its sweep, the gear (R in reverse, a dash on the jets), the traction control's lamp
    const revs = Math.round(Math.min(1, Math.max(0, hudF[6] * DASH_RPM)) * 400) / 400, g = hudI[1];
    const gear = flags & 1 ? '–' : g < 0 ? 'R' : String(g), tc = flags & 64 ? 1 : 0;
    if (revs !== hudShown.revs) { hudShown.revs = revs; $('revs').setAttribute('stroke-dasharray', `${revs} 1`); }
    if (gear !== hudShown.gear) $('gearbox').textContent = hudShown.gear = gear;
    if (tc !== hudShown.tc) $('tc').classList.toggle('on', !!(hudShown.tc = tc));
  }
  const name = road >= 0 ? names[road] || '' : '';
  if (name !== shownStreet) { // debounce so junctions don't flicker names
    streetTimer += dt;
    if (streetTimer > 0.5) {
      shownStreet = name; streetTimer = 0;
      $('street').style.opacity = name ? 1 : 0;
      if (name) $('street').textContent = name;
    }
  } else streetTimer = 0;

  if (minimap && (x !== hudShown.x || z !== hudShown.z || heading !== hudShown.heading)) {
    hudShown.x = x; hudShown.z = z; hudShown.heading = heading;
    // heading-up, the car under the middle of the 150 px disc, 1.375 px of screen to one of the map: a transform alone, so nothing is painted
    town.style.transform = `translate(75px,75px) rotate(${heading - Math.PI}rad) scale(${0.55 / minimap.scale}) translate(${(minimap.x - x) * minimap.scale}px,${(minimap.z - z) * minimap.scale}px)`;
  }

  if (audio) {
    const fly = flags & 1, kind = road >= 0 ? kinds[road] : null;
    state.speed = speed; state.y = hudF[5]; state.rpm = hudF[6]; state.throttle = hudF[7]; state.slip = hudF[8];
    state.airborne = !!(flags & 2); state.onRoad = !!(flags & 4);
    if (kind) { roadKind.k = kind; state.road = roadKind; } else state.road = null;
    state.gear = hudI[1]; state.thump = hudR[0]; state.snow = hudR[1]; state.deep = hudR[2];
    state.fwd.y = hudF[10]; state.boost = hudB[0]; state.turbine = hudB[1]; state.wing = hudB[3]; state.move = hudB.subarray(4, 8); state.spool = hudS;
    state.siren = ((flags >>> 16) & 255) / 255; state.doppler = 0.8 + ((flags >>> 24) & 255) / 255 * 0.45; // the nearest siren (rescue.h)
    input.throttle = Math.max(-1, Math.min(1, (held & 1 ? 1 : 0) - (held & 2 ? 1 : 0) + pad[0] - pad[1]));
    input.steer = Math.max(-1, Math.min(1, (held & 4 ? 1 : 0) - (held & 8 ? 1 : 0) + pad[2]));
    input.brake = !fly && !!(held & 16);
    audio.update(state, input, dt);
  }
}
requestAnimationFrame(tick);
