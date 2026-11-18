// The debug desk: a panel over the game, hidden until its key (`) is pressed. Click a spot to make something happen
// there through the game's own calls (wasm/include/harstad/desk.h), set the town's clock (its hour, its day and how fast it runs), the ice, the aurora and the
// rest, see what is under the cursor, and follow what is going on in town. It talks to the render worker:
//
//   to it:    { act: [kind, nx, ny] }, { follow: car }, { bird: [op, ...] }, { probe: { on, x, y, any } }, { meet: [SET, value] },
//             { setting, value }, { rate }, { skip }
//   from it:  { probe: Float32Array (struct desk_probe) }, { acted: [said, x, z, d, car], act }

// struct desk_probe, as plain f32s
const PROBE = { HIT: 0, ANY: 3, ROAD: 4, ROAD_D: 5, BLD: 6, CLS: 7, USE: 8, BLD_H: 9, LEVELS: 10, SHOP: 11, CAR: 12,
  SEL: 13, KIND: 14, ROLE: 15, TASK: 16, STATE: 17, SPEED: 18, LOCK: 19, SEL_X: 21, SEL_Z: 22, FLAGS: 23, SEL_ROAD: 24, LIMIT: 25,
  HOUR: 26, TIME: 27, EVENTS: 28, GLAZE: 29, INC: 30, STAGE: 31, SINCE: 32, PENDING: 33, INC_AT: 34,
  UNIT: 36, FIRE: 84, PLOUGH: 100, TRUCK: 112, BIRD: 115, FOLLOW: 119, RUN: 120, DAY: 121, RATE: 122, SUN: 123, MOON: 124, LIT: 125 };
// enum desk_act, and what each says
const ACTS = [
  ['police', 'Police call', 'the police to the house nearest the click'],
  ['ambulance', 'Ambulance', 'the ambulance to the house nearest the click'],
  ['stop', 'Traffic stop', 'a patrol near the click pulls over the car ahead'],
  ['crash', 'Crash', 'a driver near the click, close behind another, looks away; the road does the rest'],
  ['fire', 'House fire', 'the house nearest the click catches fire; the fire service turns out'],
  ['chimney', 'Chimney fire', 'the burning stove nearest the click'],
  ['alarm', 'Smoke alarm', 'at the house nearest the click: the engine comes, nothing burns'],
  ['plough', 'Plough run', 'the nearest plough along the road clicked, and back to its round'],
  ['teleport', 'Put the car here', 'on the road nearest the click, facing along it'],
  ['follow', 'Follow a vehicle', 'the view keeps over the one clicked'],
];
const SAID = ['', 'Queued: something else is under way; it follows once that has been dealt with', 'Nothing near enough fits here', 'Missed: that was the sky'];
const INC = ['', 'Police call', 'Ambulance call', 'Traffic stop', 'Crash', 'House fire', 'Chimney fire', 'Smoke alarm'];
const UNITS = ['Police 1', 'Police 2', 'Police 3', 'Ambulance', 'Fire engine', 'Aerial ladder', 'Tow truck 1', 'Tow truck 2'];
const UNIT_STATE = ['at home', 'leaving', 'on its way', 'at the scene', 'on patrol', 'going home', 'backing in'];
const FIRE = ['House fire', 'Chimney fire', 'Smoke alarm'];
const FIRE_PHASE = ['', 'smouldering', 'growing', 'burning', 'water on it', 'dying', 'out', 'done'];
const PLOUGHS = ['Main-road plough', 'Town plough', 'Gritter'];
const KIND = ['saloon', 'estate', 'long estate', 'hatchback', 'van', 'off-roader'];
const ROLE = ['driving where it likes', 'on an errand', '', '', 'the Christmas truck'];
const TASK = ['out', 'home', 'to the shops', 'visiting', 'to the late shift', 'to the quay'];
const TASK_STATE = ['planning', 'driving', 'parked', 'leaving', 'driving where it likes']; // errand.h's errand_state
const LEG = ['', 'turning, forward', 'turning, backing', 'back to its lane', 'backing to its lane', 'towed back'];
const CLS = ['house', 'outbuilding', 'block', 'hall', 'church'];
const USE = ['lived in', 'offices or public', 'works', ''];
const WEEKDAY = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']; // 1 December 2026 a Tuesday: day % 7
const WEATHER = [['As the night has it', -1], ['Clear', 0], ['Light snow', 1], ['Heavy snow', 2]]; // winter.h, SET_WEATHER
const AURORA = [['As the night has it', 0], ['An event now', 1], ['Storm', -1], ['None', -2]];

const html = `
<h2>Desk <span class="k">T to close</span></h2>
<div class="sec">Click on the map</div>
<div class="acts">${ACTS.map(([id, label, what], i) => `<button data-act="${i}" title="${what}">${label}</button>`).join('')}</div>
<div class="arm" id="d-arm">Choose, then click a spot · Esc to stop</div>
<div class="sec">Under the cursor</div>
<div id="d-hover" class="info">—</div>
<div class="sec">Going on</div>
<div id="d-live" class="live"></div>
<div class="sec">The town's clock</div>
<div id="d-sky" class="info">—</div>
<label>Hour <span><input id="d-hour" type="range" min="0" max="24" step="0.25"> <span id="d-hour-v" class="v"></span></span></label>
<label>Day <select id="d-day">${Array.from({ length: 18 }, (_, i) => `<option value="${14 + i}">${WEEKDAY[(14 + i) % 7]} ${14 + i} December</option>`).join('')}</select></label>
<label>Clock speed <select id="d-hours">${[1, 10, 60, 300, 900, 3600].map(v => `<option value="${v}">${v === 1 ? 'real time' : v + '×'}</option>`).join('')}</select></label>
<div class="sec">The game</div>
<label>Speed <select id="d-rate"><option value="1">1×</option><option value="2">2×</option><option value="4">4×</option><option value="8">8×</option></select></label>
<label>Weather <select id="d-weather">${WEATHER.map(([l, v]) => `<option value="${v}">${l}</option>`).join('')}</select></label>
<label>Aurora <select id="d-aurora">${AURORA.map(([l, v]) => `<option value="${v}">${l}</option>`).join('')}</select></label>
<label>Father Christmas <select id="d-sleigh"><option value="0">As the night has him</option><option value="1">A pass now (over the town some 10 min on)</option></select></label>
<label>Glazed roads <span><input id="d-glaze" type="range" min="0" max="1" step="0.05"> <span id="d-glaze-v" class="v"></span></span></label>
<label>Things happen by themselves <input id="d-events" type="checkbox" checked></label>
<label>Traffic <input id="d-traffic" type="checkbox"></label>
<label>Leave out <input id="d-skip" type="text" placeholder="sky,trees,…" spellcheck="false"></label>`;

export function deskInit({ worker, toast, names, settings, change }) {
  const $ = id => document.getElementById(id);
  const panel = $('desk');
  panel.innerHTML = html;
  let open = false, armed = -1, last = null, hover = { x: 0, y: 0, any: 0 };
  const post = m => worker.postMessage(m);
  const probeOn = () => post({ probe: { on: open, x: hover.x, y: hover.y, any: hover.any } });
  const showArm = () => {
    for (const b of panel.querySelectorAll('[data-act]')) b.classList.toggle('on', +b.dataset.act === armed);
    $('d-arm').textContent = armed < 0 ? 'Choose, then click a spot · Esc to stop' : `${ACTS[armed][1]}: click a spot — ${ACTS[armed][2]}`;
    document.body.classList.toggle('aiming', armed >= 0);
  };
  const arm = i => { armed = armed === i ? -1 : i; showArm(); };
  for (const b of panel.querySelectorAll('[data-act]')) b.addEventListener('click', () => arm(+b.dataset.act));
  // the panel's controls take keys and clicks of their own: none of them drives the car or starts the game
  panel.addEventListener('keydown', e => { if (e.code !== 'KeyT' && e.code !== 'Escape') e.stopPropagation(); });
  for (const t of ['click', 'pointerdown', 'wheel']) panel.addEventListener(t, e => e.stopPropagation());

  const shownHour = h => { const m = Math.round((h % 24) * 60); return `${String(Math.floor(m / 60) % 24).padStart(2, '0')}:${String(m % 60).padStart(2, '0')}`; };
  $('d-hour').addEventListener('input', e => { $('d-hour-v').textContent = shownHour(+e.target.value); post({ meet: ['HOUR', +e.target.value] }); });
  $('d-day').addEventListener('change', e => post({ meet: ['DAY', +e.target.value] }));
  $('d-hours').addEventListener('change', e => post({ meet: ['CLOCK_RATE', +e.target.value] }));
  $('d-rate').addEventListener('change', e => post({ rate: +e.target.value }));
  $('d-weather').addEventListener('change', e => { post({ meet: ['WEATHER', +e.target.value] }); toast(`Weather: ${WEATHER.find(a => a[1] === +e.target.value)?.[0] ?? ''}`); });
  $('d-aurora').addEventListener('change', e => { post({ meet: ['AURORA', +e.target.value] }); if (+e.target.value === 1) e.target.value = 0; toast(`Aurora: ${AURORA.find(a => a[1] === +e.target.value)?.[0] ?? ''}`); });
  $('d-sleigh').addEventListener('change', e => { post({ meet: ['SLEIGH', +e.target.value] }); if (+e.target.value === 1) { e.target.value = 0; toast('Father Christmas: on his way in from the north'); } });
  $('d-glaze').addEventListener('input', e => { $('d-glaze-v').textContent = (+e.target.value).toFixed(2); post({ meet: ['GLAZE', +e.target.value] }); });
  $('d-events').addEventListener('change', e => post({ meet: ['EVENTS', e.target.checked ? 1 : 0] }));
  $('d-traffic').addEventListener('change', e => change('traffic', e.target.checked));
  $('d-skip').addEventListener('change', e => post({ skip: e.target.value.replace(/\s+/g, '') }));

  // what is under the cursor and what goes on, as the worker last said
  const road = i => (i >= 0 && names()[i]) || '';
  function hoverText(p) {
    const out = [];
    if (p[PROBE.ANY]) {
      const r = road(p[PROBE.ROAD]);
      out.push(`<b>${r || 'No road near'}</b>${r ? ` · ${p[PROBE.ROAD_D].toFixed(0)} m off` : ''}`);
      out.push(`x ${p[PROBE.HIT].toFixed(1)} · z ${p[PROBE.HIT + 2].toFixed(1)} · ${p[PROBE.HIT + 1].toFixed(1)} m up`);
      if (p[PROBE.BLD] >= 0) out.push(`Building ${p[PROBE.BLD]}: ${CLS[p[PROBE.CLS]] || ''}${USE[p[PROBE.USE]] ? ', ' + USE[p[PROBE.USE]] : ''} · ${p[PROBE.BLD_H].toFixed(1)} m${p[PROBE.LEVELS] ? ` · ${p[PROBE.LEVELS]} floors` : ''}${p[PROBE.SHOP] ? ' · a shop' : ''}`);
    } else out.push('—');
    if (p[PROBE.SEL] >= 0) {
      const role = p[PROBE.ROLE], kind = p[PROBE.KIND], flags = p[PROBE.FLAGS], leg = flags >> 8;
      const what = role === 3 ? PLOUGHS[kind] : role === 2 ? UNITS[p[PROBE.TASK]] : role === 4 ? 'The Christmas truck' : KIND[kind] || 'car';
      const doing = role === 1 ? `${TASK[p[PROBE.TASK]]}, ${TASK_STATE[p[PROBE.STATE]]}` : role === 2 ? UNIT_STATE[p[PROBE.STATE]] : ROLE[role] || 'on its round';
      const bits = [flags & 1 && 'blue lights', flags & 2 && 'hazards', flags & 4 && 'switched off', flags & 8 && 'pulling in', flags & 16 && 'sliding', flags & 32 && 'in its yard', flags & 64 && 'braking', LEG[leg]].filter(Boolean);
      const locked = Math.max(p[PROBE.LOCK], p[PROBE.LOCK + 1]);
      out.push(`<b>${p[PROBE.FOLLOW] >= 0 ? 'Following' : 'Vehicle'} ${p[PROBE.SEL]}</b>: ${what}, ${doing}`);
      out.push(`${(Math.abs(p[PROBE.SPEED]) * 3.6).toFixed(0)} km/t${p[PROBE.LIMIT] ? ` (limit ${Math.round(p[PROBE.LIMIT] * 3.6)})` : ''}${road(p[PROBE.SEL_ROAD]) ? ' · ' + road(p[PROBE.SEL_ROAD]) : ''}${locked > 0.05 ? ` · wheels locked ${Math.round(locked * 100)} %` : ''}${bits.length ? ' · ' + bits.join(', ') : ''}`);
    }
    return out.join('<br>');
  }
  function liveRows(p) {
    const rows = [];
    const row = (text, data) => rows.push(`<div class="row" ${data}>${text}</div>`);
    const at = (x, z) => `data-x="${x.toFixed(1)}" data-z="${z.toFixed(1)}"`;
    const inc = p[PROBE.INC];
    if (inc) row(`<b>${INC[inc]}</b> · ${Math.round(p[PROBE.SINCE])} s`, at(p[PROBE.INC_AT], p[PROBE.INC_AT + 1]));
    if (p[PROBE.PENDING]) row(`Waiting: ${INC[p[PROBE.PENDING]]}`, '');
    for (let k = 0; k < 4; k++) {
      const f = PROBE.FIRE + k * 4;
      if (p[f] >= 0 && p[f + 1] !== 7) row(`<b>${FIRE[p[f]]}</b> · ${FIRE_PHASE[p[f + 1]]}`, at(p[f + 2], p[f + 3]));
    }
    for (let u = 0; u < UNITS.length; u++) {
      const o = PROBE.UNIT + u * 6;
      if (p[o] >= 0 && p[o] !== 0) row(`${UNITS[u]} · ${UNIT_STATE[p[o]]}${p[o + 5] ? ' · blue lights' : ''}`, `data-car="${p[o + 2]}"`);
    }
    for (let k = 0; k < 3; k++) {
      const o = PROBE.PLOUGH + k * 4;
      if (p[o] >= 0) row(`${PLOUGHS[p[o + 1]]}${p[PROBE.RUN] === p[o] ? ' · on a run' : ''}`, `data-car="${p[o]}"`);
    }
    if (p[PROBE.TRUCK] >= 0) row('The Christmas truck', `data-car="${p[PROBE.TRUCK]}"`);
    rows.push(`<div class="clock">${shownHour(p[PROBE.HOUR])} · game ${Math.round(p[PROBE.TIME])} s</div>`);
    return rows.join('');
  }
  let liveWas = '';
  worker.addEventListener('message', ({ data }) => {
    if (data.probe instanceof Float32Array) {
      last = data.probe;
      if (!open) return;
      $('d-hover').innerHTML = hoverText(last);
      const live = liveRows(last);
      if (live !== liveWas) $('d-live').innerHTML = liveWas = live;
      if (document.activeElement !== $('d-hour')) { $('d-hour').value = last[PROBE.HOUR]; $('d-hour-v').textContent = shownHour(last[PROBE.HOUR]); }
      if (document.activeElement !== $('d-day')) $('d-day').value = last[PROBE.DAY];
      if (document.activeElement !== $('d-hours')) $('d-hours').value = last[PROBE.RATE];
      $('d-sky').textContent = `${WEEKDAY[last[PROBE.DAY] % 7]} ${last[PROBE.DAY]} December · sun ${last[PROBE.SUN].toFixed(1)}° · moon ${last[PROBE.MOON].toFixed(1)}°, ${Math.round(last[PROBE.LIT] * 100)} % lit`;
      if (document.activeElement !== $('d-glaze')) { $('d-glaze').value = last[PROBE.GLAZE]; $('d-glaze-v').textContent = last[PROBE.GLAZE].toFixed(2); }
      $('d-events').checked = !!last[PROBE.EVENTS];
    } else if (data.acted) {
      const [said, , , d] = data.acted, name = ACTS[data.act][1];
      toast(said === 0 ? `${name} · ${d > 1 ? `${Math.round(d)} m from the click` : 'here'}` : `${name}: ${SAID[said]}`);
      if (said === 0 && data.act !== 9) { armed = -1; showArm(); } // following stays chosen, to pick another
    }
  });
  // a row of the list: over a vehicle to follow it, or across to where something happens
  $('d-live').addEventListener('click', e => {
    const r = e.target.closest('.row');
    if (!r) return;
    if (r.dataset.car) post({ follow: +r.dataset.car });
    else if (r.dataset.x) post({ bird: [3, +r.dataset.x, +r.dataset.z, 220, 0] });
  });

  function toggle(on = !open) {
    open = on;
    panel.classList.toggle('on', open);
    if (!open && armed >= 0) { armed = -1; showArm(); }
    if (open) $('d-traffic').checked = settings.traffic;
    probeOn();
  }
  return {
    toggle,
    get open() { return open; },
    get armed() { return armed; },
    get probe() { return last; },
    disarm() { if (armed < 0) return false; armed = -1; showArm(); return true; },
    // the cursor over the frame, -1..1 each way; any: over the frame at all
    point(x, y, any) { hover = { x, y, any }; if (open) probeOn(); },
    // a click on the map with an action chosen
    act(x, y) { if (armed < 0) return false; post({ act: [armed, x, y] }); return true; },
  };
}
