#!/usr/bin/env python3
"""8-bit BGM generator.

Generates chiptune-style background music as WAV files. Every run rolls a
fresh random seed, and every musical layer is decided procedurally from that
seed -- key, mode, tempo, song form, chord progression (functional-harmony
random walk), melody motifs (random rhythm cells + guided random-walk
pitches), bass style, arpeggio/harmony texture, drum pattern and fills, and
the pulse-wave timbres themselves. Two tracks of the same style therefore
never share a fixed pattern.

Channel layout mimics the NES APU:
    pulse 1  -> melody      (square wave, random duty cycle)
    pulse 2  -> harmony     (arpeggio / chord stabs / sustained pad)
    triangle -> bass
    noise    -> drums       (kick / snare / hi-hat)

Usage:
    python bgm_generator.py                        # random style, random seed
    python bgm_generator.py --style battle         # one battle track
    python bgm_generator.py --style boss --seed 42 # reproducible track
    python bgm_generator.py --all                  # one track per style
    python bgm_generator.py --list-styles
"""

import argparse
import math
import os
import random
import wave

import numpy as np

SR = 44100  # sample rate

# ---------------------------------------------------------------------------
# Music theory tables
# ---------------------------------------------------------------------------

MODES = {
    "major":          [0, 2, 4, 5, 7, 9, 11],
    "lydian":         [0, 2, 4, 6, 7, 9, 11],
    "mixolydian":     [0, 2, 4, 5, 7, 9, 10],
    "minor":          [0, 2, 3, 5, 7, 8, 10],
    "dorian":         [0, 2, 3, 5, 7, 9, 10],
    "phrygian":       [0, 1, 3, 5, 7, 8, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
}

MAJOR_LIKE = {"major", "lydian", "mixolydian"}

# Functional-harmony transition graphs (scale degrees 0..6).
# Each entry: degree -> [(next_degree, weight), ...]
MAJOR_GRAPH = {
    0: [(3, 3), (4, 3), (5, 3), (1, 2), (2, 1)],
    1: [(4, 4), (6, 1), (2, 1), (3, 1)],
    2: [(5, 3), (3, 2), (1, 1)],
    3: [(4, 3), (0, 3), (1, 2), (5, 1), (3, 1)],
    4: [(0, 4), (5, 3), (3, 1), (4, 1)],
    5: [(3, 3), (1, 2), (4, 2), (0, 1), (2, 1)],
    6: [(0, 4), (5, 1)],
}
MINOR_GRAPH = {
    0: [(5, 3), (3, 3), (6, 3), (4, 2), (2, 2), (1, 1)],
    1: [(4, 3), (6, 2), (0, 1)],
    2: [(5, 3), (6, 2), (3, 1), (0, 1)],
    3: [(6, 3), (4, 2), (0, 2), (5, 1)],
    4: [(0, 4), (5, 2), (3, 1)],
    5: [(6, 3), (2, 2), (4, 2), (0, 1), (3, 1)],
    6: [(0, 4), (2, 1), (5, 1)],
}

# Song forms: each letter is a section with its own progression + motif.
FORMS = [
    ["A", "A", "B", "A"],
    ["A", "B", "A", "B"],
    ["A", "B", "A", "C"],
    ["A", "A", "B", "B"],
    ["A", "B", "C", "A"],
]

# Melody rhythm cells: subdivisions of one beat, in 16th-note units
# (each cell sums to 4). Picked by note-density weighting per style.
RHYTHM_CELLS = [
    ([4],          0.15),  # quarter
    ([2, 2],       0.45),  # two eighths
    ([3, 1],       0.60),  # dotted eighth + sixteenth
    ([1, 3],       0.60),  # sixteenth + dotted eighth (syncopated)
    ([2, 1, 1],    0.75),
    ([1, 1, 2],    0.75),
    ([1, 2, 1],    0.80),  # syncopated
    ([1, 1, 1, 1], 0.95),  # four sixteenths
]

# ---------------------------------------------------------------------------
# Style profiles.  These define *distributions*, not fixed patterns --
# every concrete choice is rolled per track from the ranges/pools below.
# ---------------------------------------------------------------------------

STYLES = {
    "adventure": dict(
        desc="bright overworld / exploration theme",
        tempo=(126, 168),
        modes=[("major", 3), ("lydian", 2), ("mixolydian", 2)],
        density=(0.55, 0.80),
        rest_prob=(0.04, 0.10),
        leap_prob=(0.12, 0.22),
        bass_styles=["root8", "octave", "arp", "walk"],
        harmony_styles=["arp16", "arp8", "offbeat", "pad"],
        drum_energy=(0.55, 0.85),
        section_bars=[4, 8],
        melody_duties=[0.50, 0.25],
        vibrato=(0.06, 0.15),
        echo=(0.0, 0.20),
        register=(69, 76),
    ),
    "battle": dict(
        desc="fast, driving combat theme",
        tempo=(160, 200),
        modes=[("minor", 3), ("dorian", 2), ("phrygian", 2)],
        density=(0.75, 0.95),
        rest_prob=(0.02, 0.07),
        leap_prob=(0.18, 0.30),
        bass_styles=["octave", "root8", "gallop", "arp"],
        harmony_styles=["arp16", "offbeat", "arp8"],
        drum_energy=(0.80, 1.00),
        section_bars=[4, 8],
        melody_duties=[0.25, 0.125, 0.50],
        vibrato=(0.0, 0.08),
        echo=(0.0, 0.10),
        register=(71, 79),
    ),
    "boss": dict(
        desc="heavy, menacing boss fight",
        tempo=(140, 180),
        modes=[("harmonic_minor", 3), ("phrygian", 3), ("minor", 1)],
        density=(0.70, 0.92),
        rest_prob=(0.03, 0.08),
        leap_prob=(0.22, 0.35),
        bass_styles=["gallop", "octave", "pedal", "root8"],
        harmony_styles=["arp16", "offbeat", "tremolo"],
        drum_energy=(0.85, 1.00),
        section_bars=[4, 8],
        melody_duties=[0.125, 0.25],
        vibrato=(0.08, 0.18),
        echo=(0.05, 0.20),
        register=(67, 76),
    ),
    "village": dict(
        desc="warm, relaxed safe-area theme",
        tempo=(84, 116),
        modes=[("major", 3), ("mixolydian", 2), ("dorian", 1)],
        density=(0.35, 0.60),
        rest_prob=(0.10, 0.20),
        leap_prob=(0.08, 0.16),
        bass_styles=["pulse4", "arp", "walk", "root8"],
        harmony_styles=["arp8", "pad", "offbeat"],
        drum_energy=(0.0, 0.40),
        section_bars=[4, 8],
        melody_duties=[0.50, 0.25],
        vibrato=(0.10, 0.22),
        echo=(0.10, 0.30),
        register=(69, 76),
    ),
    "dungeon": dict(
        desc="dark, tense underground theme",
        tempo=(92, 126),
        modes=[("minor", 3), ("phrygian", 3), ("dorian", 1)],
        density=(0.35, 0.60),
        rest_prob=(0.12, 0.25),
        leap_prob=(0.15, 0.28),
        bass_styles=["pedal", "pulse4", "octave", "arp"],
        harmony_styles=["arp8", "pad", "tremolo"],
        drum_energy=(0.15, 0.50),
        section_bars=[4, 8],
        melody_duties=[0.125, 0.25],
        vibrato=(0.12, 0.25),
        echo=(0.25, 0.45),
        register=(64, 72),
    ),
    "ending": dict(
        desc="triumphant / bittersweet finale",
        tempo=(96, 132),
        modes=[("major", 3), ("lydian", 2), ("mixolydian", 1)],
        density=(0.40, 0.65),
        rest_prob=(0.06, 0.14),
        leap_prob=(0.10, 0.20),
        bass_styles=["walk", "arp", "pulse4", "root8"],
        harmony_styles=["pad", "arp8", "arp16"],
        drum_energy=(0.20, 0.55),
        section_bars=[4, 8],
        melody_duties=[0.50, 0.25],
        vibrato=(0.12, 0.25),
        echo=(0.15, 0.35),
        register=(69, 77),
    ),
    "kitchen_cozy": dict(
        desc="relaxed kitchen / restaurant background (悠閒廚房)",
        tempo=(96, 118),
        modes=[("mixolydian", 3), ("major", 2), ("dorian", 1)],
        density=(0.35, 0.55),
        rest_prob=(0.12, 0.22),
        leap_prob=(0.08, 0.16),
        bass_styles=["pulse4", "root8", "walk", "arp"],
        harmony_styles=["offbeat", "pad", "arp8"],
        drum_energy=(0.15, 0.40),
        section_bars=[4, 8],
        melody_duties=[0.50, 0.25],
        vibrato=(0.10, 0.20),
        echo=(0.10, 0.25),
        register=(69, 76),
    ),
    "kitchen_busy": dict(
        desc="bustling, upbeat kitchen / dinner-rush background (忙碌廚房)",
        tempo=(126, 156),
        modes=[("lydian", 3), ("mixolydian", 2), ("major", 2)],
        density=(0.60, 0.85),
        rest_prob=(0.05, 0.12),
        leap_prob=(0.14, 0.24),
        bass_styles=["gallop", "arp", "octave", "root8"],
        harmony_styles=["arp16", "arp8", "offbeat"],
        drum_energy=(0.45, 0.70),
        section_bars=[4, 8],
        melody_duties=[0.25, 0.50, 0.125],
        vibrato=(0.05, 0.14),
        echo=(0.05, 0.18),
        register=(71, 79),
    ),
}


def _weighted(rng, pairs):
    items = [p[0] for p in pairs]
    weights = [p[1] for p in pairs]
    return rng.choices(items, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

class Track:
    """A composed piece: note events per channel, plus render parameters."""

    def __init__(self):
        self.melody = []   # (start_beat, dur_beats, midi, velocity)
        self.harmony = []
        self.bass = []
        self.drums = []    # (start_beat, kind, velocity)  kind: kick/snare/hat/ohat
        self.total_beats = 0
        self.tempo = 120
        self.melody_duty = 0.5
        self.harmony_duty = 0.25
        self.vibrato = 0.1
        self.echo = 0.0
        self.meta = {}


def chord_pitches(mode_ivs, key, degree, octave=0):
    """Triad built by stacking scale thirds on the given degree."""
    out = []
    for step in (0, 2, 4):
        d = degree + step
        out.append(key + mode_ivs[d % 7] + 12 * (d // 7) + 12 * octave)
    return out


def make_progression(rng, graph, n_bars, cadence):
    """Random walk on the functional-harmony graph, ending with a cadence."""
    prog = [0] if rng.random() < 0.7 else [_weighted(rng, graph[0])]
    while len(prog) < n_bars - 2:
        prog.append(_weighted(rng, graph[prog[-1]]))
    if cadence == "full":     # dominant-ish -> tonic
        pre = rng.choice([4, 6, 3])
        prog += [pre, 0]
    elif cadence == "half":   # end away from tonic to pull into next section
        prog += [rng.choice([3, 5]), rng.choice([4, 6])]
    else:                     # deceptive
        prog += [4, rng.choice([5, 3])]
    return prog[:n_bars]


def make_bar_rhythm(rng, density, long_note_prob):
    """Rhythm for one 4-beat bar as a list of durations in 16ths."""
    cells = []
    for _ in range(4):
        # pick a cell whose density rank is close to the wanted density
        scored = [(cell, w) for cell, w in RHYTHM_CELLS
                  if abs(w - density) < 0.45]
        cell = rng.choice(scored or RHYTHM_CELLS)[0]
        cells.append(list(cell))
    # occasionally merge two beats into a long note
    if rng.random() < long_note_prob:
        i = rng.randrange(3)
        cells[i] = [4 + cells[i + 1][0] if cells[i] == [4] else 8]
        if cells[i] == [8]:
            cells[i + 1] = cells[i + 1][1:] if len(cells[i + 1]) > 1 else []
            cells[i], cells[i + 1] = [8], []
    out = []
    for c in cells:
        out.extend(c)
    return out


class MelodyBrain:
    """Guided random walk over the scale, biased toward chord tones on
    strong beats.  One instance per section keeps a coherent 'personality'
    while every concrete note stays random."""

    def __init__(self, rng, mode_ivs, key, center, leap_prob):
        self.rng = rng
        self.key = key
        self.leap_prob = leap_prob
        # scale laid out over several octaves, in midi numbers
        self.scale = sorted({key + iv + 12 * o for iv in mode_ivs
                             for o in range(-2, 4)})
        self.lo = center - 10
        self.hi = center + 12
        self.idx = self._nearest(center)
        self.step_bias = rng.choice([-1, 1])  # preferred melodic direction

    def _nearest(self, midi):
        return min(range(len(self.scale)), key=lambda i: abs(self.scale[i] - midi))

    def _clamp(self):
        while self.scale[self.idx] < self.lo:
            self.idx += 1
        while self.scale[self.idx] > self.hi:
            self.idx -= 1

    def next_pitch(self, chord, strong):
        rng = self.rng
        if strong and rng.random() < 0.75:
            # snap to the nearest chord tone (any octave)
            pcs = {p % 12 for p in chord}
            cands = [i for i in range(len(self.scale))
                     if self.scale[i] % 12 in pcs
                     and self.lo <= self.scale[i] <= self.hi]
            if cands:
                self.idx = min(cands, key=lambda i: abs(i - self.idx))
        elif rng.random() < self.leap_prob:
            self.idx += rng.choice([-4, -3, 3, 4, 5])
            self.step_bias = -self.step_bias
        else:
            step = rng.choices([-2, -1, 1, 2], weights=[1, 4, 4, 1])[0]
            if rng.random() < 0.6:
                step = abs(step) * self.step_bias
            self.idx += step
            if rng.random() < 0.25:
                self.step_bias = -self.step_bias
        self.idx = max(2, min(len(self.scale) - 3, self.idx))
        self._clamp()
        return self.scale[self.idx]


def compose_section_melody(rng, track, start_beat, prog, mode_ivs, key,
                           params, motif_rhythms):
    """Melody for one section.  Bars reuse this section's motif rhythms with
    random mutation, so the section hangs together but is never a loop."""
    brain = MelodyBrain(rng, mode_ivs, key, params["center"], params["leap"])
    for bar, degree in enumerate(prog):
        chord = chord_pitches(mode_ivs, key, degree)
        # motif reuse: bar rhythm comes from the section's small rhythm pool
        if bar < len(motif_rhythms) and rng.random() < 0.65:
            rhythm = list(motif_rhythms[bar % len(motif_rhythms)])
        else:
            rhythm = make_bar_rhythm(rng, params["density"], params["long"])
        pos = 0  # in 16ths
        for dur in rhythm:
            beat = start_beat + bar * 4 + pos / 4.0
            strong = pos % 4 == 0
            if rng.random() < params["rest"] and not (bar == 0 and pos == 0):
                pos += dur
                continue
            pitch = brain.next_pitch(chord, strong)
            vel = 1.0 if strong else rng.uniform(0.75, 0.95)
            track.melody.append((beat, dur / 4.0 * 0.92, pitch, vel))
            pos += dur
        # cadence bar: often land long on a chord tone
        if bar == len(prog) - 1 and rng.random() < 0.7:
            beat = start_beat + bar * 4 + 2
            pitch = brain.next_pitch(chord, True)
            track.melody = [n for n in track.melody if n[0] < beat]
            track.melody.append((beat, 1.8, pitch, 1.0))


def compose_bass(rng, track, start_beat, prog, mode_ivs, key, style, next_root):
    base_oct = -2
    roots = [chord_pitches(mode_ivs, key, d, base_oct)[0] for d in prog]
    for bar, degree in enumerate(prog):
        chord = chord_pitches(mode_ivs, key, degree, base_oct)
        root, third, fifth = chord
        b0 = start_beat + bar * 4
        nxt = roots[bar + 1] if bar + 1 < len(roots) else (next_root or roots[0])
        if style == "root8":
            for i in range(8):
                p = root if i % 4 != 3 or rng.random() < 0.6 else fifth
                track.bass.append((b0 + i * 0.5, 0.45, p, 0.9))
        elif style == "octave":
            for i in range(8):
                track.bass.append((b0 + i * 0.5, 0.45,
                                   root + (12 if i % 2 else 0), 0.9))
        elif style == "arp":
            pat = rng.choice([[root, fifth, root + 12, fifth],
                              [root, third, fifth, third],
                              [root, fifth, third, fifth]])
            for i in range(8):
                track.bass.append((b0 + i * 0.5, 0.45, pat[i % 4], 0.9))
        elif style == "walk":
            steps = [root, rng.choice([third, fifth]), fifth,
                     nxt + rng.choice([-2, -1, 1, 2])]
            for i, p in enumerate(steps):
                track.bass.append((b0 + i, 0.9, p, 0.9))
        elif style == "pulse4":
            for i in range(4):
                p = root if i != 3 or rng.random() < 0.7 else fifth
                track.bass.append((b0 + i, 0.9, p, 0.9))
        elif style == "gallop":  # x.xx x.xx driving rhythm
            for i in range(4):
                track.bass.append((b0 + i, 0.4, root, 1.0))
                track.bass.append((b0 + i + 0.5, 0.2, root, 0.8))
                track.bass.append((b0 + i + 0.75, 0.2, root, 0.8))
        elif style == "pedal":
            track.bass.append((b0, 3.4, root, 0.9))
            track.bass.append((b0 + 3.5, 0.45, root + 12, 0.8))
        # approach fill into the next bar
        if rng.random() < 0.30:
            appr = nxt + rng.choice([-2, -1, 1, 2])
            track.bass = [n for n in track.bass if n[0] < b0 + 3.5]
            track.bass.append((b0 + 3.5, 0.4, appr, 0.85))


def compose_harmony(rng, track, start_beat, prog, mode_ivs, key, style):
    base_oct = -1
    for bar, degree in enumerate(prog):
        chord = chord_pitches(mode_ivs, key, degree, base_oct)
        b0 = start_beat + bar * 4
        if style == "arp16":
            order = rng.choice(["up", "down", "updown", "random"])
            seq = {"up": chord + [chord[0] + 12],
                   "down": [chord[0] + 12] + chord[::-1],
                   "updown": chord + [chord[0] + 12, chord[2], chord[1]],
                   "random": rng.sample(chord * 2, 4)}[order]
            for i in range(16):
                track.harmony.append((b0 + i * 0.25, 0.22,
                                      seq[i % len(seq)], 0.7))
        elif style == "arp8":
            seq = rng.choice([chord, chord[::-1],
                              [chord[0], chord[2], chord[1], chord[2]]])
            for i in range(8):
                track.harmony.append((b0 + i * 0.5, 0.45,
                                      seq[i % len(seq)], 0.7))
        elif style == "offbeat":
            for i in range(4):
                for p in chord:
                    track.harmony.append((b0 + i + 0.5, 0.4, p, 0.65))
        elif style == "pad":
            for p in chord:
                track.harmony.append((b0, 3.9, p, 0.45))
        elif style == "tremolo":
            for i in range(8):
                p = chord[0] if i % 2 == 0 else chord[2]
                track.harmony.append((b0 + i * 0.5, 0.45, p, 0.6))


def compose_drums(rng, track, start_beat, n_bars, energy, is_last_section):
    if energy < 0.05:
        return
    # roll a base pattern for this section (16 slots of 16ths per bar)
    kick = {0}
    if rng.random() < energy:
        kick.add(rng.choice([7, 8, 10]))
    if rng.random() < energy * 0.6:
        kick.add(rng.choice([3, 6, 14]))
    snare = set()
    if energy > 0.35:
        snare = {4, 12}
        if rng.random() < energy * 0.4:
            snare.add(rng.choice([7, 11, 15]))
    hat_div = 2 if energy > rng.uniform(0.55, 0.8) else 4  # 16ths vs 8ths
    hat_drop = rng.uniform(0.0, 0.35) * (1.2 - energy)
    open_slot = rng.choice([2, 6, 10, 14]) if rng.random() < 0.5 else None

    for bar in range(n_bars):
        b0 = start_beat + bar * 4
        fill = bar == n_bars - 1 and rng.random() < (0.75 if is_last_section else 0.45)
        for s in range(16):
            beat = b0 + s / 4.0
            if fill and s >= 8:
                # randomized snare/kick fill over the back half of the bar
                if rng.random() < 0.55 + energy * 0.3:
                    kind = rng.choice(["snare", "snare", "kick", "hat"])
                    track.drums.append((beat, kind, rng.uniform(0.6, 1.0)))
                continue
            if s in kick:
                track.drums.append((beat, "kick", 1.0))
            if s in snare:
                track.drums.append((beat, "snare", 0.95))
            if s % hat_div == 0 and rng.random() > hat_drop:
                kind = "ohat" if s == open_slot else "hat"
                vel = 0.5 if s % 4 else 0.7
                track.drums.append((beat, kind, vel))


def compose(style_name, seed, key=None, tempo=None, section_bars=None):
    """Compose a track.  key (0-11), tempo (bpm) and section_bars (4/8) are
    optional overrides; anything left as None is rolled from the style's
    distributions."""
    rng = random.Random(seed)
    prof = STYLES[style_name]
    track = Track()

    mode_name = _weighted(rng, prof["modes"])
    mode_ivs = MODES[mode_name]
    # root somewhere around C3..B3
    key = (key % 12 if key is not None else rng.randrange(12)) + 48
    graph = MAJOR_GRAPH if mode_name in MAJOR_LIKE else MINOR_GRAPH

    track.tempo = tempo if tempo else rng.uniform(*prof["tempo"])
    track.melody_duty = rng.choice(prof["melody_duties"])
    track.harmony_duty = rng.choice([0.25, 0.125, 0.5])
    track.vibrato = rng.uniform(*prof["vibrato"])
    track.echo = rng.uniform(*prof["echo"])

    form = rng.choice(FORMS)
    section_bars = section_bars or rng.choice(prof["section_bars"])
    density = rng.uniform(*prof["density"])
    energy = rng.uniform(*prof["drum_energy"])

    # per-letter section blueprints (progression + melody personality + motif)
    blueprints = {}
    letters = sorted(set(form))
    for i, letter in enumerate(letters):
        cadence = "full" if letter == "A" else rng.choice(["half", "deceptive"])
        prog = make_progression(rng, graph, section_bars, cadence)
        sec_density = min(0.98, max(0.2, density + rng.uniform(-0.1, 0.15) * i))
        params = dict(
            density=sec_density,
            rest=rng.uniform(*prof["rest_prob"]),
            leap=rng.uniform(*prof["leap_prob"]),
            long=rng.uniform(0.15, 0.45),
            center=rng.randrange(*prof["register"]) + (i * rng.choice([0, 3, 5])),
        )
        motifs = [make_bar_rhythm(rng, sec_density, params["long"])
                  for _ in range(rng.choice([1, 2]))]
        bass_style = rng.choice(prof["bass_styles"])
        harm_style = rng.choice(prof["harmony_styles"])
        blueprints[letter] = (prog, params, motifs, bass_style, harm_style)

    beat = 0
    for si, letter in enumerate(form):
        prog, params, motifs, bass_style, harm_style = blueprints[letter]
        nxt_letter = form[(si + 1) % len(form)]
        nxt_root = chord_pitches(mode_ivs, key, blueprints[nxt_letter][0][0], -2)[0]
        compose_section_melody(rng, track, beat, prog, mode_ivs, key,
                               params, motifs)
        compose_bass(rng, track, beat, prog, mode_ivs, key, bass_style, nxt_root)
        compose_harmony(rng, track, beat, prog, mode_ivs, key, harm_style)
        compose_drums(rng, track, beat, len(prog), energy,
                      is_last_section=si == len(form) - 1)
        beat += len(prog) * 4

    track.total_beats = beat
    track.meta = dict(style=style_name, seed=seed, key=key, mode=mode_name,
                      tempo=round(track.tempo), form="".join(form),
                      bars=beat // 4)
    return track


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

def midi_to_freq(m):
    return 440.0 * 2.0 ** ((m - 69) / 12.0)


def render_tone(n, freq, wave_kind, duty, vibrato_semi, vib_rate=5.5):
    t = np.arange(n) / SR
    if vibrato_semi > 0.001:
        # vibrato fades in after ~0.15 s
        depth = vibrato_semi * np.clip((t - 0.15) / 0.2, 0, 1)
        f = freq * 2.0 ** (depth * np.sin(2 * np.pi * vib_rate * t) / 12.0)
    else:
        f = np.full(n, freq)
    phase = np.cumsum(f) / SR
    frac = phase % 1.0
    if wave_kind == "square":
        sig = np.where(frac < duty, 1.0, -1.0)
    else:  # triangle (quantized to 16 steps, NES-style)
        tri = 4.0 * np.abs(frac - 0.5) - 1.0
        sig = np.round((tri + 1) * 7.5) / 7.5 - 1.0
    return sig


def envelope(n, attack=0.004, release=0.012, decay_to=0.75, decay_time=0.35):
    t = np.arange(n) / SR
    env = np.minimum(t / attack, 1.0)
    env *= decay_to + (1 - decay_to) * np.exp(-t / decay_time)
    rel = int(release * SR)
    if rel and n > rel:
        env[-rel:] *= np.linspace(1, 0, rel)
    return env


def render_channel(events, n_total, spb, wave_kind, duty, vibrato, decay_to=0.75):
    buf = np.zeros(n_total)
    for start_beat, dur_beats, midi, vel in events:
        i0 = int(start_beat * spb)
        n = max(int(dur_beats * spb), 32)
        if i0 >= n_total:
            continue
        n = min(n, n_total - i0)
        vib = vibrato if dur_beats >= 0.9 else 0.0
        sig = render_tone(n, midi_to_freq(midi), wave_kind, duty, vib)
        buf[i0:i0 + n] += sig * envelope(n, decay_to=decay_to) * vel
    return buf


def render_drums(events, n_total, spb, seed):
    nrng = np.random.default_rng(seed)
    buf = np.zeros(n_total)

    def add(i0, sig):
        n = min(len(sig), n_total - i0)
        if n > 0:
            buf[i0:i0 + n] += sig[:n]

    for start_beat, kind, vel in events:
        i0 = int(start_beat * spb)
        if i0 >= n_total:
            continue
        if kind == "kick":
            n = int(0.11 * SR)
            t = np.arange(n) / SR
            f = 140 * np.exp(-t * 28) + 42
            sig = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 22)
            add(i0, sig * vel * 1.1)
        elif kind == "snare":
            n = int(0.14 * SR)
            t = np.arange(n) / SR
            noise = nrng.uniform(-1, 1, n) * np.exp(-t * 26)
            tone = 0.35 * np.sin(2 * np.pi * 185 * t) * np.exp(-t * 32)
            add(i0, (noise + tone) * vel * 0.8)
        else:  # hat / ohat
            dur = 0.035 if kind == "hat" else 0.14
            n = int(dur * SR)
            t = np.arange(n) / SR
            noise = nrng.uniform(-1, 1, n)
            noise = np.diff(noise, prepend=0)  # crude high-pass
            add(i0, noise * np.exp(-t * (90 if kind == "hat" else 24)) * vel * 0.5)
    return buf


def render(track, loops=1):
    spb = 60.0 / track.tempo * SR  # samples per beat
    n_total = int(track.total_beats * spb)

    melody = render_channel(track.melody, n_total, spb, "square",
                            track.melody_duty, track.vibrato, decay_to=0.65)
    harmony = render_channel(track.harmony, n_total, spb, "square",
                             track.harmony_duty, 0.0, decay_to=0.55)
    bass = render_channel(track.bass, n_total, spb, "triangle",
                          0.5, 0.0, decay_to=0.9)
    drums = render_drums(track.drums, n_total, spb, track.meta["seed"])

    mix = melody * 0.30 + harmony * 0.15 + bass * 0.32 + drums * 0.26

    if track.echo > 0.01:
        d = int(spb * 0.75)  # dotted-eighth delay
        wet = np.zeros_like(mix)
        wet[d:] = mix[:-d] * track.echo
        wet[2 * d:] += mix[:-2 * d] * track.echo * 0.45
        mix = mix + wet

    if loops > 1:
        mix = np.tile(mix, loops)

    # short fade at the very end so the file doesn't click
    fade = int(0.05 * SR)
    mix[-fade:] *= np.linspace(1, 0, fade)

    peak = np.max(np.abs(mix)) or 1.0
    return (mix / peak * 0.88 * 32767).astype(np.int16)


def write_wav(path, samples):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(samples.tobytes())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def generate(style, seed, out, loops):
    track = compose(style, seed)
    samples = render(track, loops=loops)
    write_wav(out, samples)
    m = track.meta
    secs = len(samples) / SR
    print(f"[{m['style']}] seed={m['seed']}  key={m['key'] % 12}"
          f" ({m['mode']})  tempo={m['tempo']}bpm  form={m['form']}"
          f"  bars={m['bars']}  ->  {out}  ({secs:.1f}s)")


def main():
    ap = argparse.ArgumentParser(description="Procedural 8-bit BGM generator")
    ap.add_argument("--style", choices=sorted(STYLES) + ["random"],
                    default="random", help="music style (default: random)")
    ap.add_argument("--seed", type=int, default=None,
                    help="random seed (default: random; printed for reproducing)")
    ap.add_argument("--out", default=None, help="output wav path")
    ap.add_argument("--loops", type=int, default=1,
                    help="repeat the piece N times in the file")
    ap.add_argument("--all", action="store_true",
                    help="generate one track for every style")
    ap.add_argument("--list-styles", action="store_true")
    args = ap.parse_args()

    if args.list_styles:
        for name in sorted(STYLES):
            print(f"{name:10s} {STYLES[name]['desc']}")
        return

    if args.all:
        for name in sorted(STYLES):
            seed = args.seed if args.seed is not None else random.randrange(2 ** 31)
            out = args.out or f"{name}_{seed}.wav"
            if args.out and len(STYLES) > 1:
                root, ext = os.path.splitext(args.out)
                out = f"{root}_{name}{ext}"
            generate(name, seed, out, args.loops)
        return

    style = args.style
    if style == "random":
        style = random.choice(sorted(STYLES))
    seed = args.seed if args.seed is not None else random.randrange(2 ** 31)
    out = args.out or f"{style}_{seed}.wav"
    generate(style, seed, out, args.loops)


if __name__ == "__main__":
    main()
