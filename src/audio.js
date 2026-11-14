// Generative ambient score plus the car and the night around it. Everything is synthesised; there are no audio files.
// The graph is built on any BaseAudioContext so scenarios can be rendered offline and measured.

const midi = n => 440 * Math.pow(2, (n - 69) / 12);

// D major, slow and open: D(add9) - Bm7 - Gmaj7 - A(sus)
const CHORDS = [
  [38, 50, 57, 62, 64, 66],
  [35, 47, 54, 57, 62, 66],
  [31, 43, 50, 54, 59, 62],
  [33, 45, 52, 57, 59, 64],
];
const BELLS = [62, 64, 66, 69, 71, 74, 76, 78, 81, 83];
const CHORD_SECONDS = 18;

// Layer levels (linear gain). Exported so they can be nudged live or soloed when measuring.
export const MIX = {
  master: 0.9,
  music: 0.92,    // pads, bells and their reverb/delay tails
  sfx: 0.8,       // everything else
  sfxSolo: 1.5,   // lift for the car and world while the music is off, so the room doesn't feel emptier
  engine: 0.08,
  intake: 0.025,  // breath under load
  rasp: 0.06,     // combustion noise riding on each firing pulse, relative to the tone
  rumble: 0.33,   // low tyre roll, all surfaces
  washWet: 0.075, // salted wet asphalt
  washSnow: 0.17, // packed snow
  washDeep: 0.45, // off-road
  crunch: 0.22,   // snow grains at walking pace
  scrub: 0.3,     // tyres sliding
  cabin: 0.2,     // wind around the body at speed
  air: 0.055,     // whoosh while flying
  sheen: 0.005,   // the airy top of the whoosh
  ambience: 0.07, // night wind when parked
  thump: 0.2,     // touchdown
  whump: 0.16,    // take-off
};

/* ---- engine tunables ---- */

const IDLE_RPM = 660, REDLINE_RPM = 3300; // what the synth plays at idle and flat out: kept low, this is a hum not a howl
const CAR_IDLE = 0.12;                    // state.rpm is a fraction of the car's red line and idles here (IDLE in car.js)
// Fallback gearbox when the car doesn't report rpm: speed (m/s) at which each gear would reach the redline.
const GEAR_TOP = [9, 15, 23, 36];
const SHIFT_UP = 0.72, SHIFT_DOWN = 0.36; // as a fraction of GEAR_TOP; the gap is the hysteresis
const STALL_REV = 0.14;                   // revs the converter lets through when pulling away
const REV_LAG = 0.22;                     // s; long enough that a shift glides rather than steps
const AIR_REV = 0.26, AIR_REV_LOAD = 0.2; // off the ground the revs follow the pedal, not the wheels
const ENGINE_AIR = 0.55;                  // engine level while flying, so the air takes over
const UNEVEN = 0.04;                      // strength of the orders between firing pulses: the old-engine lumpiness
const JITTER_CENTS = 12;                  // how far the revs wander from cycle to cycle: a steady pitch is what reads as a synth

const CABIN_OPEN_HZ = 20000, CABIN_SHUT_HZ = 650; // low-pass on the car and world: outside views, and the cockpit

const WET = new Set(['trunk', 'primary', 'secondary', 'tertiary']);
const NOISE_SECONDS = 7; // long enough that the loop is not heard as a pattern

const clamp = (x, lo, hi) => x < lo ? lo : x > hi ? hi : x;
const hash = n => { const s = Math.sin(n * 127.1 + 311.7) * 43758.5453; return s - Math.floor(s); };

export function startAudio() {
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const audio = buildAudio(ctx);
  // begun from a gamepad there has been no gesture a browser counts, and the context waits for the first one
  const wake = () => { if (ctx.state === 'suspended') ctx.resume(); };
  addEventListener('pointerdown', wake); addEventListener('keydown', wake);
  setInterval(audio.schedule, 500);
  return audio;
}

export function buildAudio(ctx) {
  const rate = ctx.sampleRate;
  const gain = (value, to) => { const g = ctx.createGain(); g.gain.value = value; if (to) g.connect(to); return g; };
  const filter = (type, freq, q = 0.707, db = 0) => {
    const f = ctx.createBiquadFilter();
    f.type = type; f.frequency.value = freq; f.Q.value = q; f.gain.value = db;
    return f;
  };
  // StereoPanner is missing on old Safari; centred is an acceptable fallback
  const panner = pan => {
    if (!ctx.createStereoPanner) return ctx.createGain();
    const p = ctx.createStereoPanner();
    p.pan.value = pan;
    return p;
  };
  const lfo = (hz, depth, param) => {
    const o = ctx.createOscillator();
    o.frequency.value = hz;
    o.connect(gain(depth, param));
    o.start();
  };
  // Retarget from wherever the param is right now: safe mid-fade and under rapid toggling.
  const fade = (param, value, tc) => {
    const now = ctx.currentTime;
    if (param.cancelAndHoldAtTime) param.cancelAndHoldAtTime(now);
    else { const held = param.value; param.cancelScheduledValues(now); param.setValueAtTime(held, now); }
    param.setTargetAtTime(value, now, tc);
  };

  const master = gain(0);
  master.gain.setTargetAtTime(MIX.master, ctx.currentTime, 1.3); // nothing starts abruptly
  // a real safety limiter: the default compressor settings would pump the pads under the engine
  const limiter = ctx.createDynamicsCompressor();
  limiter.threshold.value = -9; limiter.knee.value = 6; limiter.ratio.value = 12;
  limiter.attack.value = 0.004; limiter.release.value = 0.2;
  master.connect(limiter).connect(ctx.destination);

  /* ---- music bus ---- */

  // dry, reverb and delay all pass through here so muting the music takes its tails with it
  const musicBus = gain(MIX.music, master);

  // long, dark hall: decaying stereo noise as the impulse response
  const reverb = ctx.createConvolver();
  {
    const len = rate * 6, ir = ctx.createBuffer(2, len, rate);
    for (let ch = 0; ch < 2; ch++) {
      const d = ir.getChannelData(ch);
      let lp = 0;
      for (let i = 0; i < len; i++) {
        const t = i / len;
        lp += ((Math.random() * 2 - 1) - lp) * (0.5 - 0.42 * t); // highs die first
        d[i] = lp * Math.pow(1 - t, 3.2);
      }
    }
    reverb.buffer = ir;
  }
  reverb.connect(gain(0.55, musicBus));

  const music = gain(1, musicBus);
  music.connect(reverb);

  /* ---- pad ---- */

  const padFilter = filter('lowpass', 620, 0.4);
  lfo(0.031, 260, padFilter.frequency);
  padFilter.connect(music);

  function padChord(notes, at) {
    const dur = CHORD_SECONDS, attack = 6, release = 9;
    notes.forEach((n, i) => {
      for (const detune of [-7, 6]) {
        const o = ctx.createOscillator(), g = ctx.createGain();
        const pan = panner((detune < 0 ? -1 : 1) * (0.25 + 0.1 * i));
        o.type = i < 2 ? 'sine' : 'triangle';
        o.frequency.value = midi(n);
        o.detune.value = detune + (Math.random() * 4 - 2);
        const peak = (i === 0 ? 0.075 : 0.04) / (1 + i * 0.12);
        g.gain.setValueAtTime(0, at);
        g.gain.linearRampToValueAtTime(peak, at + attack);
        g.gain.setValueAtTime(peak, at + dur);
        g.gain.linearRampToValueAtTime(0, at + dur + release);
        o.connect(g).connect(pan).connect(padFilter);
        o.start(at); o.stop(at + dur + release + 0.1);
      }
    });
  }

  /* ---- bells ---- */

  const delay = ctx.createDelay(2), feedback = gain(0.46), delayTone = filter('lowpass', 1800);
  delay.delayTime.value = 0.62;
  delay.connect(delayTone).connect(feedback).connect(delay);
  delay.connect(music);

  function bell(note, at, vel) {
    const g = ctx.createGain(), pan = panner(Math.random() * 1.4 - 0.7);
    g.gain.setValueAtTime(0, at);
    g.gain.linearRampToValueAtTime(vel, at + 0.015);
    g.gain.exponentialRampToValueAtTime(0.0004, at + 5);
    for (const [mult, amt] of [[1, 1], [2, 0.22], [3.01, 0.06]]) {
      const o = ctx.createOscillator();
      o.frequency.value = midi(note) * mult;
      o.connect(gain(amt, g));
      o.start(at); o.stop(at + 5.2);
    }
    g.connect(pan);
    pan.connect(music); pan.connect(delay);
  }

  let chord = 0, nextChord = ctx.currentTime + 0.2, nextBell = ctx.currentTime + 7, lastBell = 4;
  // `ahead` is how far past now to commit notes; an offline render passes its whole length
  function schedule(ahead = 2) {
    const now = ctx.currentTime;
    while (nextChord < now + ahead) {
      padChord(CHORDS[chord++ % CHORDS.length], nextChord);
      nextChord += CHORD_SECONDS;
    }
    while (nextBell < now + ahead) {
      // wander stepwise so phrases feel sung rather than random
      lastBell = Math.min(BELLS.length - 1, Math.max(0, lastBell + Math.round(Math.random() * 6 - 3)));
      bell(BELLS[lastBell], nextBell, 0.035 + Math.random() * 0.03);
      if (Math.random() < 0.35) bell(BELLS[Math.max(0, lastBell - 2)], nextBell + 0.62, 0.03);
      nextBell += 2.5 + Math.random() * 6;
    }
  }
  schedule();

  /* ---- shared sources ---- */

  // everything but the music passes through here, so sitting inside the car can close it in
  const cabin = filter('lowpass', CABIN_OPEN_HZ);
  cabin.connect(master);
  const sfx = gain(MIX.sfx, cabin);

  const noiseBuf = ctx.createBuffer(2, rate * NOISE_SECONDS, rate);
  for (let ch = 0; ch < 2; ch++) {
    const d = noiseBuf.getChannelData(ch);
    for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
  }
  // every layer gets its own read position (and some a different rate) so layers never correlate
  const noise = (offset, speed = 1) => {
    const s = ctx.createBufferSource();
    s.buffer = noiseBuf; s.loop = true; s.playbackRate.value = speed;
    s.start(0, offset);
    return s;
  };

  // sparse soft grains: the envelope of snow compacting under a slow tyre
  const grainBuf = ctx.createBuffer(1, rate * 6, rate);
  {
    const d = grainBuf.getChannelData(0);
    for (let t = 0.05; t < 5.85; t += 0.05 + Math.random() * 0.22) {
      const len = Math.floor(rate * (0.025 + Math.random() * 0.055)), at = Math.floor(t * rate);
      const amp = 0.3 + Math.random() * 0.7, attack = len * 0.2;
      for (let i = 0; i < len; i++) d[at + i] += amp * (i < attack ? i / attack : Math.pow(1 - (i - attack) / (len - attack), 2));
    }
  }

  /* ---- engine ---- */

  // One period is a full four-stroke cycle (two crank turns), so harmonic 4 is the firing pulse.
  // Firing orders carry the hum; the weak orders between them are cylinder-to-cylinder unevenness.
  const engineWave = (() => {
    const n = 64, real = new Float32Array(n + 1), imag = new Float32Array(n + 1);
    for (let i = 1; i <= n; i++) {
      const order = i / 4, amp = i % 4 === 0 ? Math.pow(order, -2) : UNEVEN * (0.4 + 0.6 * hash(i)) / Math.pow(Math.max(order, 0.5), 0.8);
      const phase = hash(i + 100) * Math.PI * 2;
      real[i] = amp * Math.cos(phase); imag[i] = amp * Math.sin(phase);
    }
    return ctx.createPeriodicWave(real, imag);
  })();

  // a slow seamless wander in -1..1: no engine holds its revs or its level perfectly still
  const wanderBuf = ctx.createBuffer(1, rate * 8, rate);
  {
    const d = wanderBuf.getChannelData(0), points = 8 * 11, pts = new Float32Array(points + 1);
    for (let i = 0; i < points; i++) pts[i] = Math.random() * 2 - 1;
    pts[points] = pts[0];
    for (let i = 0; i < d.length; i++) {
      const p = i / d.length * points, k = Math.floor(p), f = 0.5 - 0.5 * Math.cos((p - k) * Math.PI);
      d[i] = pts[k] + (pts[k + 1] - pts[k]) * f;
    }
  }
  const wanderAt = speed => {
    const w = ctx.createBufferSource();
    w.buffer = wanderBuf; w.loop = true; w.playbackRate.value = speed;
    w.start();
    return w;
  };

  // A single voice: two copies a few cents apart beat against each other, and that slow phasing
  // is exactly what made the engine sound like it was coming down a pipe.
  const engineMix = gain(1);
  const engineOsc = ctx.createOscillator();
  engineOsc.setPeriodicWave(engineWave);
  engineOsc.frequency.value = IDLE_RPM / 120;
  engineOsc.connect(gain(0.7, engineMix));
  engineOsc.start();
  const engineSub = ctx.createOscillator(); // clean firing-rate sine for weight
  engineSub.frequency.value = IDLE_RPM / 30;
  engineSub.connect(gain(0.45, engineMix)); // most of the weight lives here
  engineSub.start();
  const jitter = gain(JITTER_CENTS);
  wanderAt(3.1).connect(jitter);
  jitter.connect(engineOsc.detune); jitter.connect(engineSub.detune);

  // Combustion is noise as much as pitch: low noise gated by the engine's own waveform gives each
  // firing pulse a rough, breathy edge and breaks up the pure tone.
  const rasp = gain(0, engineMix);
  noise(0.6).connect(filter('lowpass', 300)).connect(rasp);
  engineOsc.connect(gain(MIX.rasp, rasp.gain));

  // gentle cabin colour only: strong resonances turn a moving pitch into a swept, hollow tone
  const engineHP = filter('highpass', 32), bodyLow = filter('peaking', 85, 0.7, 5), bodyMid = filter('peaking', 230, 0.7, 1);
  const engineLP = filter('lowpass', 200), engineTop = filter('lowpass', 480, 0.5);
  const engineAmp = gain(1), engineGain = gain(0, sfx);
  engineMix.connect(engineHP).connect(bodyLow).connect(bodyMid).connect(engineLP).connect(engineTop).connect(engineAmp).connect(engineGain);

  const wanderAmt = gain(0, engineAmp.gain);
  wanderAt(1).connect(wanderAmt);

  const intakeTone = filter('bandpass', 260, 0.8), intakeGain = gain(0, engineAmp);
  noise(0.3).connect(intakeTone).connect(intakeGain);

  /* ---- tyres ---- */

  const rumbleTone = filter('lowpass', 70), rumbleGain = gain(0, sfx);
  noise(1.1, 0.9).connect(rumbleTone).connect(rumbleGain);

  const washTone = filter('bandpass', 300, 0.45), washTop = filter('lowpass', 1100), washGain = gain(0, sfx);
  noise(2.3).connect(washTone).connect(washTop).connect(washGain);

  const crunchAmp = gain(0), crunchGain = gain(0, sfx);
  const grains = ctx.createBufferSource();
  grains.buffer = grainBuf; grains.loop = true;
  grains.connect(crunchAmp.gain);
  grains.start();
  noise(3.7, 0.8).connect(filter('bandpass', 650, 0.7)).connect(filter('lowpass', 1500)).connect(crunchAmp).connect(crunchGain);

  const scrubTone = filter('bandpass', 420, 1), scrubGain = gain(0, sfx);
  noise(4.4, 1.1).connect(scrubTone).connect(filter('lowpass', 1400)).connect(scrubGain);

  /* ---- wind ---- */

  const cabinTone = filter('lowpass', 250), cabinGain = gain(0, sfx);
  noise(5.2).connect(cabinTone).connect(cabinGain);

  // one band per ear, steered apart by banking; built on a merger so it needs no panner
  const airSplit = ctx.createChannelSplitter(2), airMerge = ctx.createChannelMerger(2);
  noise(6.1, 0.93).connect(airSplit);
  const airTone = [], airGain = [];
  for (let ch = 0; ch < 2; ch++) {
    const tone = filter('bandpass', 260, 0.55), gust = gain(1), g = gain(0);
    lfo(ch ? 0.17 : 0.11, 0.22, gust.gain); // each side breathes on its own
    airSplit.connect(tone, ch);
    tone.connect(gust).connect(g).connect(airMerge, 0, ch);
    airTone.push(tone); airGain.push(g);
  }
  airMerge.connect(filter('lowpass', 2200)).connect(sfx);

  const sheenGain = gain(0, sfx);
  noise(0.9, 1.07).connect(filter('bandpass', 1700, 0.6)).connect(filter('lowpass', 3000)).connect(sheenGain);

  /* ---- night ---- */

  // far, muffled wind so parked with the music off is quiet rather than dead
  const nightGust = gain(1);
  lfo(0.05, 0.4, nightGust.gain); lfo(0.083, 0.25, nightGust.gain);
  noise(1.9, 0.7).connect(filter('bandpass', 260, 0.4)).connect(filter('lowpass', 600)).connect(nightGust).connect(gain(MIX.ambience, sfx));

  /* ---- one-shots ---- */

  // filtered noise swell: `rise` and `fall` in seconds
  function puff(at, level, cutoff, rise, fall) {
    const s = ctx.createBufferSource(), g = ctx.createGain();
    s.buffer = noiseBuf;
    g.gain.setValueAtTime(0, at);
    g.gain.linearRampToValueAtTime(level, at + rise);
    g.gain.setTargetAtTime(0, at + rise, fall / 4);
    s.connect(filter('lowpass', cutoff)).connect(g).connect(sfx);
    s.start(at, Math.random() * (NOISE_SECONDS - 3)); s.stop(at + rise + fall * 2);
  }

  // suspension taking the weight, into snow: `hard` 0..1
  function touchdown(hard) {
    const at = ctx.currentTime, o = ctx.createOscillator(), g = ctx.createGain(), level = MIX.thump * (0.35 + 0.65 * hard);
    o.frequency.setValueAtTime(82, at);
    o.frequency.exponentialRampToValueAtTime(40, at + 0.2);
    g.gain.setValueAtTime(0, at);
    g.gain.linearRampToValueAtTime(level, at + 0.012);
    g.gain.setTargetAtTime(0, at + 0.012, 0.09);
    o.connect(g).connect(sfx);
    o.start(at); o.stop(at + 0.7);
    puff(at, level * 0.5, 380, 0.02, 0.35);
  }

  /* ---- per-frame state ---- */

  let muted = false, musicOff = false, primed = false;
  let rpm = 0, load = 0, air = 0, bank = 0, sink = 0, gear = 0, wasAirborne = false, lastY = 0, now = 0;
  // A parameter is told where to go only when that has changed by more than can be heard (a ten-thousandth): aimed again at the
  // same value it would carry on along the same curve, so a car that is parked, or holding its speed, schedules nothing.
  const aimed = new Map();
  const to = (param, value, tc = 0.08) => {
    const was = aimed.get(param);
    if (was !== undefined && Math.abs(value - was) <= 1e-4 * Math.abs(was) + 1e-7) return;
    aimed.set(param, value);
    param.setTargetAtTime(value, now, tc);
  };

  return {
    schedule,
    // state: the car's state object; input: { throttle, steer, pitch, brake }
    update(state, input, dt) {
      now = ctx.currentTime;
      dt = dt > 0 ? Math.min(dt, 0.1) : 1 / 60;
      const speed = state.speed || 0, v = Math.abs(speed), airborne = !!state.airborne;
      const climb = state.fwd ? clamp(state.fwd.y, -1, 1) : 0;
      const y = state.y || 0;

      if (primed && airborne !== wasAirborne) {
        if (airborne) puff(now, MIX.whump, 240, 0.12, 0.7);
        else touchdown(clamp(sink / 6, 0, 1));
      }
      // descent rate just before the wheels touch decides how heavy the landing sounds
      sink = airborne && primed ? sink + (Math.max(0, (lastY - y) / dt) - sink) * Math.min(1, dt / 0.15) : 0;
      wasAirborne = airborne; lastY = y; primed = true;

      // pedal that is actually driving the car: braking against the direction of travel is not load
      const want = input.throttle || 0;
      const pedal = typeof state.throttle === 'number' ? clamp(state.throttle, 0, 1)
        : want > 0 ? (speed > -0.5 ? 1 : 0) : want < 0 ? (speed < 0.5 ? 0.6 : 0) : 0;
      load += (pedal - load) * (1 - Math.exp(-dt / 0.18));
      air += ((airborne ? 1 : 0) - air) * (1 - Math.exp(-dt / 0.35));
      bank += (clamp(input.steer || 0, -1, 1) - bank) * (1 - Math.exp(-dt / 0.5));
      const ground = 1 - air;

      let rev, lag = REV_LAG;
      if (typeof state.rpm === 'number') { rev = clamp((state.rpm - CAR_IDLE) / (1 - CAR_IDLE), 0, 1); lag = 0.05; }
      else if (airborne) rev = AIR_REV + pedal * AIR_REV_LOAD + Math.max(0, climb) * 0.15;
      else {
        if (speed < 0) gear = 0;
        else if (v > SHIFT_UP * GEAR_TOP[gear] && gear < GEAR_TOP.length - 1) gear++;
        else if (v < SHIFT_DOWN * GEAR_TOP[gear] && gear > 0) gear--;
        rev = Math.max(v / GEAR_TOP[gear], pedal * STALL_REV * (1 - Math.min(1, v / 4)));
      }
      rpm += (Math.min(rev, 1) - rpm) * (1 - Math.exp(-dt / lag));

      /* engine */
      const cycle = (IDLE_RPM + rpm * (REDLINE_RPM - IDLE_RPM)) / 120;
      to(engineOsc.frequency, cycle, 0.03);
      to(engineSub.frequency, cycle * 4, 0.03);
      // the low-pass opens with revs and with load: body on the pedal, soft on the overrun
      to(engineLP.frequency, 150 + 200 * rpm + load * (130 + 170 * rpm));
      to(engineGain.gain, MIX.engine * (1 + 0.15 * load + 0.05 * rpm) * (1 - air * (1 - ENGINE_AIR)));
      to(wanderAmt.gain, 0.16 * (1 - rpm) * (1 - rpm) + 0.04);
      to(intakeTone.frequency, 220 + 260 * rpm);
      to(intakeGain.gain, MIX.intake * load * (0.4 + 0.6 * rpm));

      /* tyres */
      const wet = state.onRoad && state.road != null && WET.has(state.road.k), deep = !state.onRoad;
      const roll = Math.min(1, v / 20) * ground;
      const slip = (typeof state.slip === 'number' ? clamp(state.slip, 0, 1) : input.brake ? 0.6 * Math.min(1, v / 6) : 0) * ground;
      to(rumbleTone.frequency, 70 + v * 5, 0.3);
      to(rumbleGain.gain, MIX.rumble * Math.min(1, v / 15) * (deep ? 1.5 : 1) * ground, 0.25);
      to(washTone.frequency, wet ? 600 + v * 35 : deep ? 180 + v * 8 : 300 + v * 14, 0.35);
      to(washTop.frequency, wet ? 2600 : deep ? 520 : 1100, 0.35);
      to(washGain.gain, (wet ? MIX.washWet : deep ? MIX.washDeep : MIX.washSnow) * roll, 0.25);
      // loudest at walking pace, gone by cruising speed; a slide or a braked stop digs in harder
      const creep = v / (v + 1) / (1 + v * v / 64);
      to(grains.playbackRate, 0.35 + v * 0.16, 0.2);
      to(crunchGain.gain, wet ? 0 : MIX.crunch * creep * (deep ? 1.3 : 1) * (1 + 1.5 * slip) * ground, 0.2);
      to(scrubTone.frequency, wet ? 750 : 420, 0.3);
      to(scrubGain.gain, MIX.scrub * Math.pow(slip, 1.5) * Math.min(1, v / 3), 0.12);

      /* wind */
      const q = Math.min(v, 34) / 34;
      to(cabinTone.frequency, 250 + v * 14, 0.3);
      to(cabinGain.gain, MIX.cabin * q * q, 0.3);
      // a dive rushes and rises in pitch, a climb slackens and drops; banking leans the air to the outside ear
      const whoosh = MIX.air * air * (0.2 + 0.8 * Math.pow(q, 1.5)) * (1 - 0.45 * climb);
      const centre = clamp((200 + v * 10) * (1 - 0.5 * climb), 150, 1100);
      for (let ch = 0; ch < 2; ch++) {
        const side = ch ? bank : -bank; // steering left loads the right-hand side
        to(airTone[ch].frequency, centre * (1 + 0.18 * side), 0.3);
        to(airGain[ch].gain, whoosh * (1 + 0.35 * side), 0.3);
      }
      to(sheenGain.gain, MIX.sheen * air * q * q * (1 - 0.5 * climb), 0.3);
    },
    // mutes only the score; returns true when the music is now off
    // Inside the car the engine, tyres and wind arrive through glass and trim: duller, the music untouched.
    setCabin(inside) {
      cabin.frequency.setTargetAtTime(inside ? CABIN_SHUT_HZ : CABIN_OPEN_HZ, ctx.currentTime, inside ? 0.05 : 0.12);
    },
    toggleMusic() {
      musicOff = !musicOff;
      fade(musicBus.gain, musicOff ? 0 : MIX.music, 0.35);
      fade(sfx.gain, MIX.sfx * (musicOff ? MIX.sfxSolo : 1), 0.6);
      return musicOff;
    },
    toggle() {
      muted = !muted;
      fade(master.gain, muted ? 0 : MIX.master, 0.3);
      return muted;
    },
  };
}
