// Main thread of the C / WebGPU build: the page, the keys, the gamepad, the HUD and the sound. Nothing here renders or
// simulates; the canvas is handed to src/worker.js, which does both (see the diagram at its top).
//
// To the worker:   { init }, { resize }, { event, arg } by postMessage; keys held and the gamepad through the shared block
// From the worker: { progress }, { minimap }, { ready, marks, names, kinds }, { error }; the HUD through the
//                  shared block. When the page is not cross-origin isolated the keys go as { keys } and the gamepad as { pad } messages, and the HUD
//                  comes as a bare ArrayBuffer of struct game_hud, sent only when what it says has changed (worker.js)
import { startAudio } from './audio.js';

const $ = id => document.getElementById(id);
const params = new URLSearchParams(location.search); // ?auto&x=&z=&h=&cam=&drive=&fly=&orbit=&top= for testing
const KEY = { KeyW: 1, ArrowUp: 1, KeyS: 2, ArrowDown: 2, KeyA: 4, ArrowLeft: 4, KeyD: 8, ArrowRight: 8, Space: 16, ShiftLeft: 32, ShiftRight: 32 };
const CAMS = ['Chase', 'Far chase', 'Cockpit', 'Roof', 'Bumper', 'Wheel', 'Orbit', 'Overhead'];
const HUD = 64; // offset of struct game_hud in the shared block

// Without WebGPU there is no game: the title says so.
if (!navigator.gpu) $('go').textContent = 'This needs a browser with WebGPU';

const canvas = $('view');
const pixels = () => { const r = Math.min(devicePixelRatio, 2); return [Math.round(innerWidth * r), Math.round(innerHeight * r)]; };
// The player's settings, kept in the browser between visits; a parameter in the address wins over what was kept
// (only what the player changed is kept, so a default changed later still reaches everyone who left it alone)
const SETTINGS = 'harstad.settings.v2';
const DEFAULTS = { snow: 0, gusts: true, aa: 4, sound: true, music: true, hud: true }; // the falling snow is off: busy over the whole view
const settings = { ...DEFAULTS };
try { Object.assign(settings, JSON.parse(localStorage.getItem(SETTINGS) || '{}')); } catch { /* private window, or blocked */ }
if (params.has('snow')) settings.snow = +params.get('snow') || 0;
if (params.has('aa')) settings.aa = +params.get('aa') === 4 ? 4 : 1;
const keep = () => { try { localStorage.setItem(SETTINGS, JSON.stringify(Object.fromEntries(Object.entries(settings).filter(([k, v]) => v !== DEFAULTS[k])))); } catch { /* not kept: fine */ } };

const shared = self.crossOriginIsolated ? new SharedArrayBuffer(128) : null;
const block = shared || new ArrayBuffer(128);
const keysOut = new Uint32Array(block, 0, 1), padOut = new Float32Array(block, 4, 6), hudF = new Float32Array(block, HUD, 11), hudI = new Int32Array(block, HUD + 44, 5), hudIn = new Uint8Array(block, HUD, 64);

const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
worker.onerror = e => { window.__next.error = e.message || 'the render worker failed to load'; $('go').textContent = window.__next.error; };
const offscreen = canvas.transferControlToOffscreen();
const [width, height] = pixels();
worker.postMessage({ init: { canvas: offscreen, shared, width, height, ratio: Math.min(devicePixelRatio, 2),
  params: { ...Object.fromEntries(params), snow: String(settings.snow), gusts: settings.gusts ? '1' : '0', aa: String(settings.aa) } } }, [offscreen]);
addEventListener('resize', () => worker.postMessage({ resize: pixels(), ratio: Math.min(devicePixelRatio, 2) }));

let ready = false, started = false, audio = null, names = [], kinds = [], minimap = null, statsWaiting = [];
const t0 = performance.now();
worker.onmessage = ({ data }) => {
  if (data instanceof ArrayBuffer) hudIn.set(new Uint8Array(data)); // into the block the views above were made on, once
  else if (data.progress) { if (!ready) $('go').textContent = `Loading… ${Math.round(data.progress * 100)}%`; }
  else if (data.minimap) { // the town's roads, drawn by the worker: shown as they come, never painted again
    minimap = data;
    const town = $('town');
    town.width = data.minimap.width; town.height = data.minimap.height;
    town.getContext('bitmaprenderer').transferFromImageBitmap(data.minimap);
  }
  else if (data.ready) {
    ready = true; names = data.names; kinds = data.kinds;
    window.__next.marks = { ...data.marks, total: performance.now() - t0 };
    $('go').textContent = 'Click to drive';
    // ?auto starts at once, as the game does when it loads itself again (anti-aliasing changed, the GPU lost): with its
    // sound, which the browser lets play from the first key or click; ?mute for a run that should make none
    if (params.has('auto')) begin(!params.has('mute'));
  } else if (data.stats) statsWaiting.shift()?.(data.stats);
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
  stats: () => new Promise(r => { statsWaiting.push(r); worker.postMessage({ stats: true }); }),
  bench: n => new Promise(r => { statsWaiting.push(r); worker.postMessage({ bench: n }); }),
  grab: () => new Promise(r => { statsWaiting.push(r); worker.postMessage({ grab: true }); }), // one frame's pixels, drawn apart from the canvas: { width, height, format, data }
  hud: () => ({ speed: hudF[0], alt: hudF[1], heading: hudF[2], x: hudF[3], z: hudF[4], y: hudF[5], road: hudI[0], cam: hudI[2], flags: hudI[3], frame: hudI[4] }) };

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

function begin(withSound = true) {
  if (started || !ready) return;
  started = true;
  send('START');
  if (withSound) {
    audio = startAudio(); audio.setCabin(hudI[2] === 2);
    if (!settings.sound) audio.toggle();
    if (!settings.music) audio.toggleMusic();
  }
  $('title').classList.add('gone');
  $('hud').classList.toggle('on', settings.hud);
  setTimeout(() => $('help').classList.add('dim'), 14000);
}
$('title').addEventListener('click', () => begin());

addEventListener('keydown', e => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (KEY[e.code]) setKeys(keyHeld | KEY[e.code]);
  if (e.code === 'Space' || e.code.startsWith('Arrow')) e.preventDefault();
  if (!started) { if (e.code === 'Enter' || e.code === 'Space' || e.code === 'KeyW') begin(); return; }
  if (e.repeat) return;
  if (e.code === 'KeyC') setCam(hudI[2] + (e.shiftKey ? -1 : 1));
  if (/^Digit[1-8]$/.test(e.code)) setCam(+e.code[5] - 1);
  if (e.code === 'KeyF') toggleFlight('Space to climb, Shift to descend');
  if (e.code === 'KeyM') change('sound', !settings.sound, true);
  if (e.code === 'KeyN') change('music', !settings.music, true);
  if (e.code === 'KeyH') change('hud', !settings.hud);
  if (e.code === 'KeyO') $('settings').classList.toggle('on');
  if (e.code === 'Escape') $('settings').classList.remove('on');
  if (e.code === 'KeyR') send('BACK_TO_ROAD');
});
/* ---- settings ---- */

// Apply one setting, keep it, and show it in the panel. Snow and the gusts change live; anti-aliasing builds the
// pipelines anew, so the game is loaded again where the car is.
function change(what, value, say = false) {
  if (settings[what] === value) return;
  settings[what] = value;
  keep();
  if (what === 'snow') worker.postMessage({ setting: 'snow', value });
  if (what === 'gusts') worker.postMessage({ setting: 'gusts', value });
  if (what === 'sound' && audio) audio.toggle();
  if (what === 'music' && audio) audio.toggleMusic();
  if (what === 'hud') $('hud').classList.toggle('on', value);
  if (say && (what === 'sound' || what === 'music')) toast(`${what === 'sound' ? 'Sound' : 'Music'} ${value ? 'on' : 'off'}`);
  if (what === 'aa') {
    for (const [k, v] of [['x', hudF[3]], ['z', hudF[4]], ['h', hudF[2]], ['cam', hudI[2]]]) params.set(k, v);
    params.set('aa', value);
    if (started) params.set('auto', '');
    location.replace(location.pathname + '?' + params);
  }
  shown();
}
function shown() {
  $('s-snow').value = settings.snow; $('s-snow-v').textContent = `${settings.snow}×`;
  $('s-gusts').checked = settings.gusts; $('s-aa').value = settings.aa;
  $('s-sound').checked = settings.sound; $('s-music').checked = settings.music; $('s-hud').checked = settings.hud;
}
shown();
$('gear').addEventListener('click', () => $('settings').classList.toggle('on'));
$('s-snow').addEventListener('input', e => change('snow', +e.target.value));
$('s-gusts').addEventListener('change', e => change('gusts', e.target.checked));
$('s-aa').addEventListener('change', e => change('aa', +e.target.value));
$('s-sound').addEventListener('change', e => change('sound', e.target.checked));
$('s-music').addEventListener('change', e => change('music', e.target.checked));
$('s-hud').addEventListener('change', e => change('hud', e.target.checked));
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
 *   LB, RB  camera                      d-pad left, right  steer    Back  hide the HUD      Start, A  begin
 */
const PAD_HELD = [[0, 16], [2, 32], [14, 4], [15, 8]]; // button, KEY bit
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
function readPad() {
  let p = null;
  try { for (const g of navigator.getGamepads?.() || []) if (g && g.connected && (!p || (g.mapping === 'standard' && p.mapping !== 'standard'))) p = g; } catch { /* not allowed here */ }
  if (!p && !padLive) return;
  padLive = !!p;
  const b = p ? p.buttons : [], a = p ? p.axes : [];
  const [sx, sy] = stick(a[0], a[1]), [lx, ly] = stick(a[2], a[3]);
  const next = [trigger(b[7]), trigger(b[6]), -Math.sign(sx) * Math.abs(sx) ** STEER_CURVE, sy, lx, -ly];
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
  if (!started) { if (hit & (1 << 0 | 1 << 9)) begin(); return; }
  if (hit & 1 << 5) setCam(hudI[2] + 1);
  if (hit & 1 << 4) setCam(hudI[2] - 1);
  if (hit & 1 << 3) toggleFlight('pull back or A to climb, X to descend');
  if (hit & 1 << 1) send('BACK_TO_ROAD');
  if (hit & 1 << 8) change('hud', !settings.hud);
}
addEventListener('gamepadconnected', () => toast('Gamepad — RT drive · LT brake · right stick look'));

/* ---- hud and sound, once a frame from whatever the worker last wrote ---- */

const town = $('town');
let shownStreet = '', streetTimer = 0, last = performance.now();
const state = { fwd: { y: 0 }, road: null }, input = {}, roadKind = { k: '' };

const hudShown = { kmh: -1, limit: -1, alt: -1, x: NaN, z: NaN, heading: NaN }; // what the page last showed: nothing is written to it, or drawn, that has not changed
function tick(now) {
  requestAnimationFrame(tick);
  const dt = Math.min(0.05, Math.max(0, (now - last) / 1000));
  last = now;
  readPad();
  if (!started) return;
  const speed = hudF[0], alt = hudF[1], heading = hudF[2], x = hudF[3], z = hudF[4], road = hudI[0], flags = hudI[3];
  const kmh = Math.round(Math.abs(speed) * 3.6), limit = flags >>> 8, up = alt > 1.5 ? Math.round(alt) : 0;
  if (kmh !== hudShown.kmh) $('speed').textContent = hudShown.kmh = kmh;
  if (limit !== hudShown.limit) { hudShown.limit = limit; $('limit').textContent = limit || ''; $('limit').classList.toggle('on', limit > 0); }
  if (up !== hudShown.alt) { hudShown.alt = up; $('alt').textContent = up ? ` · ${up} m` : ''; }
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
    state.fwd.y = hudF[10];
    input.throttle = Math.max(-1, Math.min(1, (held & 1 ? 1 : 0) - (held & 2 ? 1 : 0) + pad[0] - pad[1]));
    input.steer = Math.max(-1, Math.min(1, (held & 4 ? 1 : 0) - (held & 8 ? 1 : 0) + pad[2]));
    input.brake = !fly && !!(held & 16);
    audio.update(state, input, dt);
  }
}
requestAnimationFrame(tick);
