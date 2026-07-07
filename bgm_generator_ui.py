# -*- coding: utf-8 -*-
"""bgm_generator_ui.py — 8-bit BGM 產生器 GUI 版

作曲引擎來自 bgm_generator.py(每次產出都不同:隨機調性/曲式/和弦進行/
旋律動機/伴奏型態/鼓組),GUI 提供:

    四聲道鋼琴捲簾 · 混音器(音量/靜音) · 試聽播放 · WAV / MIDI / 分軌匯出

依賴:
    pip install PyQt6 numpy sounddevice
    (sounddevice 缺少時仍可產生與匯出,只是無法試聽)

執行:
    python bgm_generator_ui.py
"""

import random
import struct
import sys
import time

import numpy as np

import bgm_generator as bg

SR = bg.SR

# GUI 顯示名稱 -> 引擎風格鍵
UI_STYLES = [
    ("冒險/探索", "adventure"),
    ("戰鬥", "battle"),
    ("Boss 戰", "boss"),
    ("村莊/日常", "village"),
    ("地下城/神祕", "dungeon"),
    ("結局", "ending"),
    ("廚房悠閒", "kitchen_cozy"),
    ("忙碌廚房", "kitchen_busy"),
]
MODE_LABELS = {
    "major": "大調", "minor": "小調", "lydian": "利地亞",
    "mixolydian": "米索利地亞", "dorian": "多利安",
    "phrygian": "弗里吉亞", "harmonic_minor": "和聲小調",
}
KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# 各軌混音增益預設值(對應 UI 拉桿 0-100)
BASE_GAIN = {"P1": 0.30, "P2": 0.15, "TRI": 0.32, "NOI": 0.26}


# ============================================================
# 引擎輸出 -> GUI 資料結構
# ============================================================

class Note:
    __slots__ = ("start", "length", "pitch", "drum")

    def __init__(self, start, length, pitch=0, drum=None):
        self.start = start      # 16 分音符網格(可為小數)
        self.length = length
        self.pitch = pitch
        self.drum = drum


DRUM_CHAR = {"kick": "K", "snare": "S", "hat": "h", "ohat": "h"}


class Song:
    """把引擎的 Track(拍為單位)轉成 GUI 用的 16 分音符網格。"""

    def __init__(self, track, style_label):
        self.track = track
        self.style = style_label
        m = track.meta
        self.seed = m["seed"]
        self.key = m["key"] % 12
        self.bpm = m["tempo"]
        self.bars = m["bars"]
        self.mode_label = MODE_LABELS.get(m["mode"], m["mode"])
        self.form = m["form"]
        self.total_steps = track.total_beats * 4
        self.duration_sec = track.total_beats * 60.0 / track.tempo
        self.tracks = {
            "P1": [Note(b * 4, d * 4, p) for b, d, p, _ in track.melody],
            "P2": [Note(b * 4, d * 4, p) for b, d, p, _ in track.harmony],
            "TRI": [Note(b * 4, d * 4, p) for b, d, p, _ in track.bass],
            "NOI": [Note(b * 4, 1, drum=DRUM_CHAR[k]) for b, k, _ in track.drums],
        }

    @property
    def key_name(self):
        return KEY_NAMES[self.key]


# ============================================================
# 合成(分軌 + 混音,支援動態音量/靜音)
# ============================================================

def render_stems(track):
    spb = 60.0 / track.tempo * SR
    n_total = int(track.total_beats * spb)
    return {
        "P1": bg.render_channel(track.melody, n_total, spb, "square",
                                track.melody_duty, track.vibrato, decay_to=0.65),
        "P2": bg.render_channel(track.harmony, n_total, spb, "square",
                                track.harmony_duty, 0.0, decay_to=0.55),
        "TRI": bg.render_channel(track.bass, n_total, spb, "triangle",
                                 0.5, 0.0, decay_to=0.9),
        "NOI": bg.render_drums(track.drums, n_total, spb, track.meta["seed"]),
    }


def mix_stems(track, stems, gains=None, mutes=None):
    gains = gains or {}
    mutes = mutes or set()
    mix = np.zeros(len(next(iter(stems.values()))))
    for name, buf in stems.items():
        if name in mutes:
            continue
        mix = mix + buf * gains.get(name, BASE_GAIN[name])

    if track.echo > 0.01:
        spb = 60.0 / track.tempo * SR
        d = int(spb * 0.75)
        wet = np.zeros_like(mix)
        wet[d:] = mix[:-d] * track.echo
        wet[2 * d:] += mix[:-2 * d] * track.echo * 0.45
        mix = mix + wet

    peak = np.max(np.abs(mix)) or 1.0
    return mix / peak * 0.88


def to_pcm(data):
    return (np.clip(data, -1, 1) * 32767).astype(np.int16)


def write_wav(path, data):
    import wave
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes(to_pcm(data).tobytes())


# ============================================================
# MIDI 匯出
# ============================================================

TICKS_PER_QUARTER = 480
MIDI_PROGRAM = {"P1": 80, "P2": 80, "TRI": 38}   # 方波 Lead / 合成貝斯
DRUM_MIDI = {"kick": 36, "snare": 38, "hat": 42, "ohat": 46}


def _varlen(v):
    out = [v & 0x7F]
    v >>= 7
    while v:
        out.append((v & 0x7F) | 0x80)
        v >>= 7
    return bytes(reversed(out))


def write_midi(path, track):
    tpq = TICKS_PER_QUARTER

    def track_chunk(events):
        events.sort(key=lambda e: e[0])
        data = bytearray()
        last = 0
        for tick, msg in events:
            data += _varlen(tick - last) + msg
            last = tick
        data += _varlen(0) + b"\xff\x2f\x00"
        return b"MTrk" + struct.pack(">I", len(data)) + bytes(data)

    chunks = []
    tempo = int(60_000_000 / track.tempo)
    chunks.append(track_chunk([(0, b"\xff\x51\x03" + struct.pack(">I", tempo)[1:])]))

    ch_map = {"P1": 0, "P2": 1, "TRI": 2}
    for name, events in (("P1", track.melody), ("P2", track.harmony),
                         ("TRI", track.bass)):
        ch = ch_map[name]
        ev = [(0, bytes([0xC0 | ch, MIDI_PROGRAM[name]]))]
        for beat, dur, pitch, vel in events:
            p = max(0, min(127, int(pitch)))
            t0 = int(round(beat * tpq))
            t1 = max(t0 + 1, int(round((beat + dur) * tpq)))
            ev.append((t0, bytes([0x90 | ch, p, max(1, int(vel * 110))])))
            ev.append((t1, bytes([0x80 | ch, p, 0])))
        chunks.append(track_chunk(ev))

    ev = []
    for beat, kind, vel in track.drums:
        p = DRUM_MIDI[kind]
        t0 = int(round(beat * tpq))
        ev.append((t0, bytes([0x99, p, max(1, int(vel * 110))])))
        ev.append((t0 + tpq // 4, bytes([0x89, p, 0])))
    chunks.append(track_chunk(ev))

    header = b"MThd" + struct.pack(">IHHH", 6, 1, len(chunks), tpq)
    with open(path, "wb") as f:
        f.write(header + b"".join(chunks))


# ============================================================
# GUI 層
# ============================================================

def run_gui():
    from PyQt6.QtCore import Qt, QTimer, QRectF
    from PyQt6.QtGui import QPainter, QColor, QFont
    from PyQt6.QtWidgets import (
        QApplication, QWidget, QMainWindow, QComboBox, QSpinBox,
        QPushButton, QCheckBox, QLabel, QHBoxLayout, QVBoxLayout,
        QGridLayout, QFileDialog, QMessageBox, QFrame, QSlider,
        QAbstractSpinBox
    )

    try:
        import sounddevice as sd
        HAS_AUDIO = True
    except Exception:
        sd = None
        HAS_AUDIO = False

    COLORS = {
        "P1": QColor("#F06292"),
        "P2": QColor("#E0B040"),
        "TRI": QColor("#4DD0E1"),
        "NOI": QColor("#B39DDB"),
    }
    BG = QColor("#141410")
    GRID = QColor("#2A2A22")
    LANE_LABEL = QColor("#888878")
    PLAYHEAD = QColor("#F5F0D8")

    class PianoRoll(QFrame):
        def __init__(self):
            super().__init__()
            self.song = None
            self.play_frac = -1.0
            self.setMinimumHeight(260)
            self.setStyleSheet("border: 2px solid #3A332C; border-radius: 8px;")

        def set_song(self, song):
            self.song = song
            self.update()

        def set_playhead(self, frac):
            self.play_frac = frac
            self.update()

        def paintEvent(self, _):
            p = QPainter(self)
            p.fillRect(self.rect().adjusted(2, 2, -2, -2), BG)
            if not self.song:
                p.setPen(LANE_LABEL)
                p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "尚未產生樂曲")
                return

            W, H = self.width() - 4, self.height() - 4
            lanes = ["P1", "P2", "TRI", "NOI"]
            lane_h = H / 4
            total = self.song.total_steps

            p.translate(2, 2)

            p.setPen(GRID)
            for bar in range(self.song.bars + 1):
                x = bar * 16 / total * W
                p.drawLine(int(x), 0, int(x), int(H))
            for i in range(1, 4):
                p.drawLine(0, int(i * lane_h), int(W), int(i * lane_h))

            f = QFont("monospace", 8)
            p.setFont(f)
            for li, name in enumerate(lanes):
                y0 = li * lane_h
                p.setPen(LANE_LABEL)
                p.drawText(4, int(y0 + 12), name)
                notes = self.song.tracks[name]
                if not notes:
                    continue

                if name == "NOI":
                    for n in notes:
                        x = n.start / total * W
                        hgt = {"K": 0.8, "S": 0.6, "h": 0.35}[n.drum]
                        p.fillRect(QRectF(x, y0 + lane_h * (1 - hgt) - 4,
                                          max(W / total, 2), lane_h * hgt - 6),
                                   COLORS[name])
                else:
                    lo = min(n.pitch for n in notes) - 2
                    hi = max(n.pitch for n in notes) + 2
                    span = max(hi - lo, 1)
                    for n in notes:
                        x = n.start / total * W
                        w = max(n.length / total * W - 1, 2)
                        ny = y0 + (1 - (n.pitch - lo) / span) * (lane_h - 14) + 7
                        p.fillRect(QRectF(x, ny - 3, w, 6), COLORS[name])

            if self.play_frac >= 0:
                x = int(self.play_frac * W)
                p.setPen(PLAYHEAD)
                p.drawLine(x, 0, x, int(H))

    class Main(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("8-BIT 音樂產生器")
            self.song = None
            self.stems = None
            self.mix = None
            self.play_t0 = None

            self.setStyleSheet("""
                QMainWindow { background-color: #EBE5CE; }
                QLabel { font-weight: bold; color: #3A332C; }
                QComboBox, QSpinBox {
                    border: 2px solid #3A332C;
                    border-radius: 4px;
                    padding: 4px;
                    background-color: white;
                    font-weight: bold;
                    color: #3A332C;
                }
                QComboBox QAbstractItemView {
                    background-color: white;
                    color: #3A332C;
                    selection-background-color: #DFD7BD;
                    selection-color: #3A332C;
                }
                QCheckBox { font-weight: bold; color: #3A332C; }
                QPushButton {
                    border: 2px solid #3A332C;
                    border-radius: 6px;
                    font-weight: bold;
                    padding: 6px 12px;
                    background-color: white;
                    color: #3A332C;
                }
                QPushButton:pressed { background-color: #dcdcdc; }

                QPushButton#btn_red { background-color: #9E3C36; color: white; }
                QPushButton#btn_red:pressed { background-color: #822F2A; }

                QPushButton#btn_yellow { background-color: #DDAA44; color: #3A332C; }
                QPushButton#btn_yellow:pressed { background-color: #C39234; }

                QFrame#mixer_frame { border: 2px solid #3A332C; border-radius: 8px; background-color: #EBE5CE; }
                QFrame#track_frame { border: 2px solid #3A332C; border-radius: 6px; background-color: #DFD7BD; }

                QLabel#status_bar {
                    background-color: #8FAD7D;
                    border: 2px solid #3A332C;
                    border-radius: 6px;
                    padding: 6px;
                }

                QLabel#top_title {
                    background-color: #9E3C36;
                    color: white;
                    border-bottom: 2px solid #3A332C;
                    padding: 10px;
                    font-size: 14px;
                }
            """)

            cw = QWidget()
            self.setCentralWidget(cw)
            root = QVBoxLayout(cw)
            root.setContentsMargins(15, 0, 15, 15)

            # ---- 頂部標題列 ----
            top_lbl = QLabel("8-BIT 音樂產生器  2A03 四聲道  ·  每次產生都不同  ·  SEED 可重現  ·  MIDI / WAV 匯出")
            top_lbl.setObjectName("top_title")
            root.addWidget(top_lbl)
            root.addSpacing(10)

            # ---- 參數列 ----
            param_layout = QHBoxLayout()

            self.style_cb = QComboBox()
            self.style_cb.addItems([lbl for lbl, _ in UI_STYLES])
            self.key_cb = QComboBox()
            self.key_cb.addItems(["隨機"] + KEY_NAMES)
            self.bars_cb = QComboBox()
            self.bars_cb.addItems(["隨機", "16", "32"])
            self.bpm_sp = QSpinBox()
            self.bpm_sp.setRange(0, 240)
            self.bpm_sp.setSpecialValueText("隨機")
            self.bpm_sp.setValue(0)
            self.bpm_sp.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
            self.seed_sp = QSpinBox()
            self.seed_sp.setRange(0, 999999)
            self.seed_sp.setValue(random.randrange(1000000))
            # 隱藏種子的上下按鈕
            self.seed_sp.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
            self.lock_seed_ck = QCheckBox("鎖定種子")
            self.lock_seed_ck.setToolTip("勾選後每次產生都用同一個種子(可重現);"
                                         "不勾選則每次產生自動換新種子")

            self.btn_rand_seed = QPushButton("?")
            self.btn_rand_seed.setFixedSize(36, 36)
            self.btn_rand_seed.setStyleSheet("padding: 0px; font-size: 18px;")
            self.btn_rand_seed.clicked.connect(
                lambda: self.seed_sp.setValue(random.randrange(1000000)))

            def add_param_col(label_text, widget, extra_widget=None):
                v = QVBoxLayout()
                v.addWidget(QLabel(label_text))
                h = QHBoxLayout()
                h.addWidget(widget)
                if extra_widget:
                    h.addWidget(extra_widget)
                v.addLayout(h)
                param_layout.addLayout(v)

            add_param_col("風格", self.style_cb)
            add_param_col("調性", self.key_cb)
            add_param_col("小節數", self.bars_cb)
            add_param_col("BPM", self.bpm_sp)
            add_param_col("種子", self.seed_sp, self.btn_rand_seed)
            v = QVBoxLayout()
            v.addWidget(QLabel(""))
            v.addWidget(self.lock_seed_ck)
            param_layout.addLayout(v)
            param_layout.addStretch()
            root.addLayout(param_layout)

            # ---- 控制列 ----
            ctrl_layout = QHBoxLayout()
            self.gen_btn = QPushButton("產生樂曲")
            self.gen_btn.setObjectName("btn_red")
            self.gen_btn.clicked.connect(self.generate)

            self.play_btn = QPushButton("播放")
            self.play_btn.clicked.connect(self.play)
            self.stop_btn = QPushButton("停止")
            self.stop_btn.clicked.connect(self.stop)
            self.loop_play_ck = QCheckBox("循環播放")
            self.loop_play_ck.setChecked(True)

            for w in (self.gen_btn, self.play_btn, self.stop_btn, self.loop_play_ck):
                ctrl_layout.addWidget(w)

            ctrl_layout.addStretch()
            space_hint = QLabel("空白鍵 = 播放 / 停止")
            space_hint.setStyleSheet("color: #666; font-size: 12px;")
            ctrl_layout.addWidget(space_hint)
            root.addLayout(ctrl_layout)

            # ---- 混音器 ----
            mixer_frame = QFrame()
            mixer_frame.setObjectName("mixer_frame")
            mixer_layout = QGridLayout(mixer_frame)

            self.sliders = {}
            self.mutes = {}
            tracks_info = [
                ("P1", "PULSE 1 主旋律", COLORS["P1"].name()),
                ("P2", "PULSE 2 和聲", COLORS["P2"].name()),
                ("TRI", "TRIANGLE 貝斯", COLORS["TRI"].name()),
                ("NOI", "NOISE 鼓組", COLORS["NOI"].name()),
            ]

            for i, (name, label_text, hex_color) in enumerate(tracks_info):
                row, col = divmod(i, 2)
                t_frame = QFrame()
                t_frame.setObjectName("track_frame")
                t_layout = QHBoxLayout(t_frame)
                t_layout.setContentsMargins(10, 5, 10, 5)

                cbox = QLabel()
                cbox.setFixedSize(14, 14)
                cbox.setStyleSheet(f"background-color: {hex_color}; border: 2px solid #3A332C;")

                lbl = QLabel(label_text)
                lbl.setFixedWidth(120)

                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setRange(0, 100)
                slider.setValue(int(BASE_GAIN[name] * 100))
                slider.setStyleSheet("""
                    QSlider::groove:horizontal { border: 1px solid #999; height: 8px; background: #FFF; border-radius: 4px; }
                    QSlider::handle:horizontal { background: #9E3C36; border: 2px solid #3A332C; width: 14px; margin: -5px 0; border-radius: 7px; }
                """)

                mute_ck = QCheckBox("靜音")

                t_layout.addWidget(cbox)
                t_layout.addWidget(lbl)
                t_layout.addWidget(slider)
                t_layout.addWidget(mute_ck)

                mixer_layout.addWidget(t_frame, row, col)

                self.sliders[name] = slider
                self.mutes[name] = mute_ck

            root.addWidget(mixer_frame)

            # ---- Piano roll ----
            self.roll = PianoRoll()
            root.addWidget(self.roll, stretch=1)

            # ---- 狀態列 ----
            self.status = QLabel("就緒")
            self.status.setObjectName("status_bar")
            root.addWidget(self.status)

            # ---- 匯出列 ----
            export_layout = QHBoxLayout()
            for text, fn in [
                ("下載 MIDI", self.export_midi),
                ("下載混音 WAV", self.export_wav),
                ("下載四軌分軌 WAV", self.export_stems),
            ]:
                b = QPushButton(text)
                b.setObjectName("btn_yellow")
                b.clicked.connect(fn)
                export_layout.addWidget(b)
            export_layout.addStretch()
            root.addLayout(export_layout)

            self.timer = QTimer(self)
            self.timer.setInterval(30)
            self.timer.timeout.connect(self._tick)

            if not HAS_AUDIO:
                self.play_btn.setEnabled(False)
                self.play_btn.setToolTip("未安裝 sounddevice,無法試聽")

        # ---- UI 動態音量獲取 ----
        def _get_gains(self):
            return {name: self.sliders[name].value() / 100.0
                    for name in ("P1", "P2", "TRI", "NOI")}

        def _muted_set(self):
            return {n for n, ck in self.mutes.items() if ck.isChecked()}

        def _remix(self):
            self.mix = mix_stems(self.song.track, self.stems,
                                 self._get_gains(), self._muted_set())

        # ---- 行為 ----
        def generate(self):
            was_playing = self.play_t0 is not None
            self.stop()

            style_label, style_key = UI_STYLES[self.style_cb.currentIndex()]
            if not self.lock_seed_ck.isChecked():
                self.seed_sp.setValue(random.randrange(1000000))
            seed = self.seed_sp.value()

            key_sel = self.key_cb.currentIndex()
            key = None if key_sel == 0 else key_sel - 1
            tempo = self.bpm_sp.value() or None
            bars_sel = self.bars_cb.currentText()
            section_bars = None if bars_sel == "隨機" else int(bars_sel) // 4

            track = bg.compose(style_key, seed, key=key, tempo=tempo,
                               section_bars=section_bars)
            self.song = Song(track, style_label)
            self.stems = render_stems(track)
            self._remix()
            self.roll.set_song(self.song)

            s = self.song
            self.status.setText(
                f"{style_label} － {s.key_name} {s.mode_label} · {s.bpm} BPM · "
                f"曲式 {s.form} · {s.bars} 小節 · 種子 {s.seed}"
            )
            if was_playing:
                self.play()

        def play(self):
            if self.song is None:
                self.generate()
            self._remix()  # 播放時抓取最新拉桿數值
            sd.stop()
            sd.play(self.mix.astype(np.float32), SR,
                    loop=self.loop_play_ck.isChecked())
            self.play_t0 = time.monotonic()
            self.timer.start()

        def stop(self):
            if HAS_AUDIO:
                sd.stop()
            self.timer.stop()
            self.play_t0 = None
            self.roll.set_playhead(-1)

        def _tick(self):
            if self.play_t0 is None or self.song is None:
                return
            el = time.monotonic() - self.play_t0
            dur = self.song.duration_sec
            if self.loop_play_ck.isChecked():
                self.roll.set_playhead((el % dur) / dur)
            elif el >= dur:
                self.stop()
            else:
                self.roll.set_playhead(el / dur)

        def keyPressEvent(self, ev):
            if ev.key() == Qt.Key.Key_Space:
                if self.play_t0 is None:
                    self.play()
                else:
                    self.stop()
            else:
                super().keyPressEvent(ev)

        # ---- 匯出 ----
        def _need_song(self):
            if self.song is None:
                QMessageBox.information(self, "提示", "請先產生樂曲")
                return False
            return True

        def _stem_name(self):
            s = self.song
            return f"{s.style}_{s.key_name}_{s.bpm}bpm_seed{s.seed}"

        def export_midi(self):
            if not self._need_song():
                return
            path, _ = QFileDialog.getSaveFileName(
                self, "匯出 MIDI", self._stem_name() + ".mid", "MIDI (*.mid)")
            if path:
                write_midi(path, self.song.track)
                self.status.setText(f"已匯出 {path}")

        def export_wav(self):
            if not self._need_song():
                return
            path, _ = QFileDialog.getSaveFileName(
                self, "匯出混音 WAV", self._stem_name() + ".wav", "WAV (*.wav)")
            if path:
                self._remix()
                write_wav(path, self.mix)
                self.status.setText(f"已匯出 {path}")

        def export_stems(self):
            if not self._need_song():
                return
            path, _ = QFileDialog.getSaveFileName(
                self, "匯出分軌(選基底檔名)", self._stem_name() + ".wav", "WAV (*.wav)")
            if not path:
                return
            base = path[:-4] if path.lower().endswith(".wav") else path
            gains = self._get_gains()
            for name, data in self.stems.items():
                write_wav(f"{base}_{name}.wav", data * gains[name])
            self.status.setText(f"已匯出 4 軌:{base}_P1/P2/TRI/NOI.wav")

    app = QApplication(sys.argv)
    win = Main()
    win.resize(1100, 750)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_gui()
