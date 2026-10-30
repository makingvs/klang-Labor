"""Klang-Labor: realistischere Pups- und Rülps-Geräusche (CC0), Version 2.

Ansatz: impulssynchrone Synthese. Jeder einzelne Schwingungsimpuls (Flattern der
Schließmuskel bzw. des oberen Speiseröhren-Schließmuskels) regt eine eigene, leicht
veränderte Resonanz an. Abstände und Stärken der Impulse sind stark unregelmäßig,
Resonanzen wandern über die Zeit, dazu Hauch-/Blubbergeräusche und etwas Raumhall.
"""
import numpy as np, scipy.signal as sg, subprocess, os, sys, base64, json, wave

SR = 22050
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 11)
OUT = 'out'
os.makedirs(OUT, exist_ok=True)


def walk(n, rate, amt):
    """Glatter Zufallsverlauf mit `rate` Stützstellen pro Sekunde."""
    k = max(2, int(n / SR * rate) + 2)
    return np.interp(np.linspace(0, k - 1, n), np.arange(k), rng.normal(0, 1, k)) * amt


def damped(f, bw, length, phase=0.0):
    t = np.arange(length) / SR
    return np.exp(-np.pi * bw * t) * np.sin(2 * np.pi * f * t + phase)


def room(x, size=.06, mix=.12):
    n = int(SR * size * 5)
    t = np.arange(n) / SR
    ir = rng.normal(0, 1, n) * np.exp(-t / size)
    ir = sg.lfilter(*sg.butter(2, 3000 / (SR / 2)), ir)
    ir /= np.sqrt(np.sum(ir ** 2))
    wet = sg.fftconvolve(x, ir)[:len(x)]
    return x * (1 - mix) + wet * mix * 1.4


def finish(x, tail=.12):
    x = np.concatenate([x, np.zeros(int(tail * SR))])
    x = room(x)
    x = sg.lfilter(*sg.butter(2, 40 / (SR / 2), 'high'), x)
    x = np.tanh(x / (np.max(np.abs(x)) + 1e-9) * 1.6)  # leichte Sättigung, "analog"
    x = x / np.max(np.abs(x)) * .92
    f = int(.006 * SR); x[:f] *= np.linspace(0, 1, f); x[-int(.04 * SR):] *= np.linspace(1, 0, int(.04 * SR))
    return x


def pulse_times(dur, f0_curve, jitter, skip=0.0):
    """Impulszeitpunkte mit unregelmäßigen Abständen."""
    times, t = [], rng.uniform(0, .004)
    while t < dur:
        times.append(t)
        f0 = f0_curve(t / dur)
        period = 1 / f0 * (1 + rng.normal(0, jitter))
        if rng.random() < skip:  # Aussetzer / Subharmonische
            period *= rng.choice([1.5, 2.0])
        t += max(period, .0012)
    return np.array(times)


# ---------------------------------------------------------------- Pupse
def fart(kind):
    dur = {'kurz': rng.uniform(.15, .4), 'lang': rng.uniform(.9, 2.2),
           'quietsch': rng.uniform(.25, .8), 'knatter': rng.uniform(.5, 1.3),
           'feucht': rng.uniform(.3, .9)}[kind]
    base = {'kurz': rng.uniform(70, 150), 'lang': rng.uniform(50, 110), 'quietsch': rng.uniform(170, 360),
            'knatter': rng.uniform(25, 55), 'feucht': rng.uniform(45, 100)}[kind]
    shape = rng.choice(['fall', 'rise', 'arch', 'wobble'])
    wob_f, wob_a = rng.uniform(2, 7), rng.uniform(.05, .25)

    def f0(u):
        c = {'fall': 1.25 - .5 * u, 'rise': .8 + .45 * u, 'arch': 1 + .3 * np.sin(np.pi * u),
             'wobble': 1 + wob_a * np.sin(2 * np.pi * wob_f * u * dur)}[shape]
        return base * c

    jit = {'quietsch': .04, 'knatter': .35}.get(kind, rng.uniform(.08, .2))
    times = pulse_times(dur, f0, jit, skip=.15 if kind in ('knatter', 'feucht') else .04)
    n = int((dur + .15) * SR)
    y = np.zeros(n)
    # Körper-/Stoffresonanzen wandern langsam
    r1 = rng.uniform(130, 260) if kind != 'quietsch' else rng.uniform(350, 700)
    r2 = r1 * rng.uniform(2.3, 3.6)
    drift1, drift2 = walk(n, 6, .12), walk(n, 6, .15)
    env_amp = 1 + walk(n, 18, .35)
    for k, t0 in enumerate(times):
        i = int(t0 * SR)
        u = t0 / dur
        a = max(.05, env_amp[min(i, n - 1)]) * rng.lognormal(0, .35)
        if kind == 'knatter':
            a *= rng.uniform(.4, 1.3)
        f1 = r1 * (1 + drift1[i]); f2 = r2 * (1 + drift2[i])
        L = int(.035 * SR)
        flap = (damped(f1 * rng.uniform(.93, 1.07), rng.uniform(70, 160), L, rng.uniform(0, 1))
                + .55 * damped(f2 * rng.uniform(.9, 1.1), rng.uniform(220, 450), L, rng.uniform(0, 3))
                + .25 * damped(rng.uniform(900, 1600), 700, L))
        click = rng.normal(0, 1, int(.002 * SR)) * .5
        flap[:len(click)] += click
        j = min(n, i + L)
        y[i:j] += a * flap[:j - i]
    # Hauch / Luftstrom
    breath = sg.lfilter(*sg.butter(2, [250 / (SR / 2), 1800 / (SR / 2)], 'band'), rng.normal(0, 1, n))
    y += breath * .05 * np.std(y) / (np.std(breath) + 1e-9) * (3 if kind == 'feucht' else 1)
    # Blubberblasen (feucht): kurze, nach oben gleitende Töne
    if kind in ('feucht', 'knatter') or rng.random() < .25:
        for _ in range(rng.integers(3, 14 if kind == 'feucht' else 6)):
            i = rng.integers(0, max(1, int(dur * SR)))
            L = int(rng.uniform(.012, .04) * SR); t = np.arange(L) / SR
            fb = rng.uniform(250, 900)
            bub = np.sin(2 * np.pi * np.cumsum(fb * (1 + 6 * t)) / SR) * np.exp(-t / (L / SR / 3))
            j = min(n, i + L)
            y[i:j] += bub[:j - i] * np.max(np.abs(y)) * rng.uniform(.15, .45)
    # Hüllkurve: Einschwingen, am Ende ausfransen
    tt = np.arange(n) / SR
    e = np.clip(tt / rng.uniform(.004, .02), 0, 1) * np.clip((dur + .02 - tt) / rng.uniform(.04, .2), 0, 1)
    y *= e ** .7
    y = sg.lfilter(*sg.butter(2, rng.uniform(1800, 3500) / (SR / 2)), y)  # Stoff dämpft Höhen
    return finish(y)


# ---------------------------------------------------------------- Rülpser
# Formanten (F1, F2, F3) dunkler, offener Vokale – Rülpser klingen meist nach "öa", "oa", "uä"
VOW = {'o': (480, 850, 2450), 'oe': (500, 1400, 2300), 'a': (720, 1150, 2500),
       'u': (360, 780, 2300), 'ae': (620, 1650, 2500), 'aa': (800, 1250, 2600)}


def burp(kind):
    dur = {'kurz': rng.uniform(.25, .5), 'mittel': rng.uniform(.5, 1.0),
           'lang': rng.uniform(1.0, 2.0), 'hoch': rng.uniform(.3, .8), 'doppel': rng.uniform(.7, 1.3)}[kind]
    base = rng.uniform(55, 110) if kind != 'hoch' else rng.uniform(130, 210)
    drop = rng.uniform(.65, .95)

    def f0(u):
        return base * (1 + (1 - drop) * (1 - u) * 1.5) * (1 - .15 * u ** 3)

    times = pulse_times(dur, f0, rng.uniform(.12, .25), skip=rng.uniform(.03, .15))
    n = int((dur + .1) * SR)
    y = np.zeros(n)
    # Vokal-Verlauf: Start -> Mitte -> Ende
    seq = [VOW[v] for v in rng.choice(list(VOW), 3)]
    fdrift = [walk(n, 5, .06) for _ in range(3)]
    amp_walk = 1 + walk(n, rng.uniform(12, 30), .45)  # Gurgeln
    gaps = []
    if kind == 'doppel':  # zwei Stöße: "bu–rrp"
        g0 = rng.uniform(.25, .45) * dur
        gaps = [(g0, g0 + rng.uniform(.04, .09))]
    for t0 in times:
        if any(a <= t0 < b for a, b in gaps):
            continue
        i = int(t0 * SR); u = t0 / dur
        if u < .5:
            w = u / .5; F = [seq[0][m] * (1 - w) + seq[1][m] * w for m in range(3)]
        else:
            w = (u - .5) / .5; F = [seq[1][m] * (1 - w) + seq[2][m] * w for m in range(3)]
        F = [F[m] * (1 + fdrift[m][i]) for m in range(3)]
        a = max(.05, amp_walk[i]) * rng.lognormal(0, .3)
        L = int(.03 * SR)
        ir = (1.0 * damped(F[0], rng.uniform(90, 160), L, rng.uniform(0, .5))
              + .7 * damped(F[1], rng.uniform(110, 200), L, rng.uniform(0, 3))
              + .25 * damped(F[2], rng.uniform(180, 300), L, rng.uniform(0, 3))
              + .35 * damped(rng.uniform(160, 240), 120, L))  # Brustraum
        ir[:int(.0015 * SR)] += rng.normal(0, .4, int(.0015 * SR))  # knarziger Anschlag
        j = min(n, i + L)
        y[i:j] += a * ir[:j - i]
    # Hauch, durch die gleichen Vokalfarben gefiltert
    asp = rng.normal(0, 1, n)
    asp = sum(sg.lfilter(*sg.iirpeak(f, Q=f / 250, fs=SR), asp) for f in seq[1])
    y += asp / (np.std(asp) + 1e-9) * np.std(y) * rng.uniform(.12, .25)
    tt = np.arange(n) / SR
    e = np.clip(tt / rng.uniform(.008, .03), 0, 1) * np.clip((dur - tt) / rng.uniform(.06, .25), 0, 1)
    y *= e ** .6
    # kleines "Blopp" am Anfang (Luft löst sich)
    L = int(.025 * SR)
    pop = damped(rng.uniform(140, 260), 60, L) * np.max(np.abs(y)) * rng.uniform(.5, 1.0)
    y[:L] += pop
    y = sg.lfilter(*sg.butter(2, 3800 / (SR / 2)), y)
    return finish(y, tail=.15)


def encode(x, path):
    pcm = (x * 32767).astype('<i2').tobytes()
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-f', 's16le', '-ar', str(SR), '-ac', '1', '-i', '-',
                    '-b:a', '48k', path], input=pcm, check=True)


preview = []
for cat, fn, kinds in [('fart', fart, ['kurz'] * 10 + ['lang'] * 8 + ['quietsch'] * 6 + ['knatter'] * 8 + ['feucht'] * 8),
                       ('burp', burp, ['kurz'] * 10 + ['mittel'] * 12 + ['lang'] * 7 + ['hoch'] * 5 + ['doppel'] * 6)]:
    order = list(kinds); rng.shuffle(order)
    items = []
    for i, k in enumerate(order):
        x = fn(k)
        p = f'{OUT}/{cat}_{i:02d}.mp3'
        encode(x, p)
        items.append([base64.b64encode(open(p, 'rb').read()).decode(), 'Klang-Labor (am Computer erzeugt)', 'CC0', '',
                      round(len(x) / SR, 1)])
        with wave.open(f'{OUT}/{cat}_{i:02d}.wav', 'wb') as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((x * 32767).astype('<i2').tobytes())
        if i < 8:
            preview += [x, np.zeros(int(.45 * SR))]
    with open(f'{OUT}/{cat}.js', 'w') as f:
        f.write(f'KL_SOUNDS({json.dumps(cat)},{json.dumps(items, separators=(",", ":"))});\n')
    print(cat, len(items), os.path.getsize(f'{OUT}/{cat}.js'))
encode(np.concatenate(preview), f'{OUT}/hoerprobe.mp3')
