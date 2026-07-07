# 8-bit BGM Generator

用 Python 程序化生成 8-bit / chiptune 風格背景音樂，直接輸出 WAV 檔。
每次執行都會擲一個新的隨機 seed，**同一個風格每次產出的音樂都明顯不同**
（實測不同 seed 之間旋律音符重疊率僅約 2~4%）。

## 安裝

```bash
pip install numpy                      # 命令列版只需要 numpy
pip install PyQt6 sounddevice          # GUI 版需要;sounddevice 缺少時仍可產生與匯出,只是無法試聽
```

## GUI 版

```bash
python bgm_generator_ui.py
```

![GUI](docs/screenshot.png)

- **參數列**：風格／調性／小節數／BPM 都可設「隨機」或指定；種子預設每次「產生樂曲」自動換新（勾「鎖定種子」即可重現同一首）
- **混音器**：四軌（PULSE1 主旋律／PULSE2 和聲／TRIANGLE 貝斯／NOISE 鼓組）各自音量拉桿＋靜音
- **鋼琴捲簾**：四聲道視覺化，播放時有 playhead，空白鍵＝播放/停止
- **匯出**：MIDI／混音 WAV／四軌分軌 WAV

## 命令列版

```bash
python bgm_generator.py                          # 隨機風格、隨機 seed
python bgm_generator.py --style battle           # 產生一首戰鬥曲
python bgm_generator.py --style boss --seed 42   # 指定 seed,可重現同一首
python bgm_generator.py --all                    # 每個風格各產生一首
python bgm_generator.py --style village --loops 2 --out town.wav
python bgm_generator.py --list-styles            # 列出所有風格
```

每次產生都會印出 seed，想重現同一首曲子時把 seed 帶回去即可。

## 風格

| 風格 | 說明 |
|---|---|
| `adventure` | 明亮輕快的大地圖／探索主題 |
| `battle` | 快節奏、有推進感的戰鬥曲 |
| `boss` | 沉重威壓感的 Boss 戰 |
| `village` | 溫和放鬆的村莊／安全區 |
| `dungeon` | 陰暗緊張的地下城 |
| `ending` | 壯闊／感傷的結局曲 |
| `kitchen_cozy` | 溫馨悠閒的廚房／餐廳背景（悠閒廚房） |
| `kitchen_busy` | 忙碌歡快的廚房／出餐尖峰背景（忙碌廚房） |

## 為什麼每次都不一樣?

風格檔案裡定義的是**機率分布與素材池**，不是固定樂句。每一首曲子由 seed
驅動、逐層隨機決定：

1. **調性與音階** — 12 個調 × 每風格 2~3 種調式（major / lydian / dorian /
   phrygian / harmonic minor …）隨機挑選。
2. **速度與曲式** — tempo 在風格區間內隨機；曲式從 AABA / ABAB / ABAC /
   AABB / ABCA 中抽選，段落長度 4 或 8 小節。
3. **和弦進行** — 不是查表，而是在功能和聲轉移圖上隨機遊走生成，段尾再接
   隨機挑選的終止式（正格／半終止／假終止）。
4. **旋律** — 每段先隨機生成節奏動機（由 16 分音符節奏細胞組合），音高用
   偏好級進、強拍靠向和弦音的引導式隨機漫步產生；段內動機會重用並隨機
   變形，兼顧「像一首歌」與「不重複」。
5. **伴奏與鼓** — 貝斯型態（八分根音／八度跳躍／琶音／walking／gallop／
   持續低音）、和聲聲部（16 分琶音／後半拍和弦／pad／tremolo）、鼓組
   pattern 與過門全部按風格權重隨機生成。
6. **音色** — 旋律方波的 duty cycle（12.5% / 25% / 50%）、顫音深度、
   echo 量也都是每首隨機。

## 聲部配置（仿 NES APU）

- Pulse 1（方波）— 主旋律
- Pulse 2（方波）— 和聲／琶音
- Triangle（16 階量化三角波）— 貝斯
- Noise — 鼓組（kick / snare / hi-hat）
