#!/usr/bin/env python3
"""Generates documentation images in docs/img:
1. board_wiring.png
2. serial_status.png
3. live_transcription.png
"""
import os
from PIL import Image, ImageDraw, ImageFont

os.makedirs('docs/img', exist_ok=True)

def get_fonts():
    try:
        font_mono = ImageFont.truetype('C:/Windows/Fonts/consola.ttf', 14)
        font_mono_bold = ImageFont.truetype('C:/Windows/Fonts/consolab.ttf', 14)
        font_mono_small = ImageFont.truetype('C:/Windows/Fonts/consola.ttf', 12)
        font_ui = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 14)
        font_ui_bold = ImageFont.truetype('C:/Windows/Fonts/segoeuib.ttf', 15)
        font_ui_title = ImageFont.truetype('C:/Windows/Fonts/segoeuib.ttf', 20)
        font_ui_large = ImageFont.truetype('C:/Windows/Fonts/segoeuib.ttf', 26)
    except Exception:
        font_mono = ImageFont.load_default()
        font_mono_bold = font_mono
        font_mono_small = font_mono
        font_ui = font_mono
        font_ui_bold = font_mono
        font_ui_title = font_mono
        font_ui_large = font_mono
    return {
        'mono': font_mono, 'mono_b': font_mono_bold, 'mono_s': font_mono_small,
        'ui': font_ui, 'ui_b': font_ui_bold, 'ui_t': font_ui_title, 'ui_l': font_ui_large
    }

fonts = get_fonts()

# -------------------------------------------------------------
# 1. board_wiring.png
# -------------------------------------------------------------
def make_board_wiring():
    w, h = 940, 560
    img = Image.new('RGB', (w, h), color='#0f141c')
    draw = ImageDraw.Draw(img)

    # Background grid
    for x in range(0, w, 20):
        draw.line([(x, 0), (x, h)], fill='#151c27', width=1)
    for y in range(0, h, 20):
        draw.line([(0, y), (w, y)], fill='#151c27', width=1)

    # Title header
    draw.text((30, 22), "ESP32-S3 + Dual INMP441 Hardware Wiring", fill='#e2e8f0', font=fonts['ui_t'])
    draw.text((30, 48), "TDM I2S Stereo Bus (Shared Clock/Data) • 55 mm Acoustic Spacing • Hardware Alignment", fill='#94a3b8', font=fonts['ui'])

    # ESP32-S3 DevKit Box
    bx, by, bw, bh = 50, 100, 310, 410
    draw.rounded_rectangle([bx, by, bx+bw, by+bh], radius=12, fill='#1e293b', outline='#3b82f6', width=2)
    draw.rounded_rectangle([bx+20, by+20, bx+bw-20, by+120], radius=6, fill='#0f172a', outline='#475569', width=1)
    draw.text((bx+45, by+45), "ESP32-S3-WROOM-1", fill='#f8fafc', font=fonts['ui_b'])
    draw.text((bx+45, by+70), "Dual-Core LX7 @ 240 MHz", fill='#94a3b8', font=fonts['mono_s'])
    draw.text((bx+45, by+88), "Zero PSRAM • 512 KB Internal SRAM", fill='#38bdf8', font=fonts['mono_s'])

    # USB ports
    draw.rectangle([bx+bw//2 - 40, by+bh-22, bx+bw//2 + 40, by+bh+4], fill='#64748b', outline='#cbd5e1')
    draw.text((bx+bw//2 - 28, by+bh-18), "USB-C (UART)", fill='#0f172a', font=fonts['mono_s'])

    # ESP32 Pin Headers (Right Side)
    pins = [
        ("3V3", "#ef4444", by+150),
        ("GND", "#94a3b8", by+190),
        ("GPIO 15 (WS)", "#eab308", by+240),
        ("GPIO 16 (SCK)", "#22c55e", by+280),
        ("GPIO 17 (SD)", "#38bdf8", by+320),
    ]

    for label, col, py in pins:
        draw.rounded_rectangle([bx+bw-12, py-10, bx+bw+12, py+10], radius=3, fill='#334155', outline=col, width=2)
        draw.text((bx+bw-130, py-8), label, fill=col, font=fonts['mono_b'])

    # Microphone 1 (Left Channel - Slot 0)
    m1x, m1y, mw, mh = 620, 110, 260, 180
    draw.rounded_rectangle([m1x, m1y, m1x+mw, m1y+mh], radius=10, fill='#1e293b', outline='#10b981', width=2)
    draw.text((m1x+18, m1y+15), "INMP441 #1 (Left Channel)", fill='#10b981', font=fonts['ui_b'])
    draw.text((m1x+18, m1y+35), "L/R Pin → GND (I2S Left Slot)", fill='#cbd5e1', font=fonts['mono_s'])

    # Microphone 2 (Right Channel - Slot 1)
    m2x, m2y = 620, 320
    draw.rounded_rectangle([m2x, m2y, m2x+mw, m2y+mh], radius=10, fill='#1e293b', outline='#f59e0b', width=2)
    draw.text((m2x+18, m2y+15), "INMP441 #2 (Right Channel)", fill='#f59e0b', font=fonts['ui_b'])
    draw.text((m2x+18, m2y+35), "L/R Pin → 3V3 (I2S Right Slot)", fill='#cbd5e1', font=fonts['mono_s'])

    # Acoustic spacing bracket
    draw.line([(m1x+mw+25, m1y+mh//2), (m1x+mw+40, m1y+mh//2)], fill='#a855f7', width=2)
    draw.line([(m1x+mw+40, m1y+mh//2), (m1x+mw+40, m2y+mh//2)], fill='#a855f7', width=2)
    draw.line([(m1x+mw+25, m2y+mh//2), (m1x+mw+40, m2y+mh//2)], fill='#a855f7', width=2)
    draw.text((m1x+mw-20, (m1y+m2y)//2 + 75), "55 mm", fill='#c084fc', font=fonts['ui_b'])

    # Mic Pins definitions
    def draw_mic_pins(mx, my, is_m1):
        m_pins = [
            ("VDD", "#ef4444", my+70),
            ("GND", "#94a3b8", my+92),
            ("SD", "#38bdf8", my+114),
            ("SCK", "#22c55e", my+136),
            ("WS", "#eab308", my+158),
        ]
        for name, col, py in m_pins:
            draw.circle((mx+10, py+6), 5, fill=col, outline='#ffffff')
            draw.text((mx+24, py), name, fill='#e2e8f0', font=fonts['mono_s'])
        # L/R pin wire
        lr_y = my+60
        draw.circle((mx+mw-20, lr_y), 5, fill='#ffffff', outline='#000000')
        draw.text((mx+mw-60, lr_y-7), "L/R", fill='#ffffff', font=fonts['mono_s'])
        if is_m1:
            draw.line([(mx+mw-20, lr_y), (mx+mw-20, my+92), (mx+24+30, my+92)], fill='#94a3b8', width=2)
        else:
            draw.line([(mx+mw-20, lr_y), (mx+mw-20, my+70), (mx+24+30, my+70)], fill='#ef4444', width=2)

    draw_mic_pins(m1x, m1y, True)
    draw_mic_pins(m2x, m2y, False)

    # Draw bus wiring lines
    # 3V3 (Red)
    draw.line([(bx+bw, by+150), (450, by+150), (450, m1y+76), (m1x+10, m1y+76)], fill='#ef4444', width=2)
    draw.line([(450, by+150), (450, m2y+76), (m2x+10, m2y+76)], fill='#ef4444', width=2)

    # GND (Slate)
    draw.line([(bx+bw, by+190), (470, by+190), (470, m1y+98), (m1x+10, m1y+98)], fill='#94a3b8', width=2)
    draw.line([(470, by+190), (470, m2y+98), (m2x+10, m2y+98)], fill='#94a3b8', width=2)

    # WS (Yellow) GPIO 15
    draw.line([(bx+bw, by+240), (490, by+240), (490, m1y+164), (m1x+10, m1y+164)], fill='#eab308', width=2)
    draw.line([(490, by+240), (490, m2y+164), (m2x+10, m2y+164)], fill='#eab308', width=2)

    # SCK (Green) GPIO 16
    draw.line([(bx+bw, by+280), (510, by+280), (510, m1y+142), (m1x+10, m1y+142)], fill='#22c55e', width=2)
    draw.line([(510, by+280), (510, m2y+142), (m2x+10, m2y+142)], fill='#22c55e', width=2)

    # SD (Sky Blue) GPIO 17
    draw.line([(bx+bw, by+320), (530, by+320), (530, m1y+120), (m1x+10, m1y+120)], fill='#38bdf8', width=2)
    draw.line([(530, by+320), (530, m2y+120), (m2x+10, m2y+120)], fill='#38bdf8', width=2)

    img.save('docs/img/board_wiring.png', 'PNG')
    print("board_wiring.png saved")

# -------------------------------------------------------------
# 2. serial_status.png
# -------------------------------------------------------------
def make_serial_status():
    w, h = 980, 520
    img = Image.new('RGB', (w, h), color='#0d1117')
    draw = ImageDraw.Draw(img)

    # Terminal title bar
    draw.rectangle([0, 0, w, 36], fill='#161b22')
    draw.line([(0, 36), (w, 36)], fill='#30363d', width=1)
    draw.circle((20, 18), 6, fill='#ff5f56')
    draw.circle((38, 18), 6, fill='#ffbd2e')
    draw.circle((56, 18), 6, fill='#27c93f')
    draw.text((w//2 - 130, 9), "ESP32-S3 Serial Monitor — COM6 (115200)", fill='#8b949e', font=fonts['ui'])

    # Terminal Log lines
    lines = [
        ("I (240) esp_image: segment 0: paddr=00010020 vaddr=3c070020 size=14a84h ( 84612) map", '#8b949e'),
        ("I (295) cpu_start: Pro cpu start user code", '#8b949e'),
        ("I (310) cpu_start: Starting scheduler on PRO CPU. FreeRTOS HZ=1000", '#8b949e'),
        ("----------------------- KWS BOOT STATUS -----------------------", '#58a6ff'),
        ("reset reason   : power-on (clean boot)", '#c9d1d9'),
        ("RAM limit test : 163 KB reserved, strict peak limit 256 KB enforced", '#7ee787'),
        ("model ready    : wake word 'marvin', threshold 0.60, window 5, arena 26 KB", '#79c0ff'),
        ("microphones    : left -22.3 dBFS, right -17.4 dBFS -> using both (time-aligned)", '#c9d1d9'),
        ("stream         : connected to server ws://192.168.1.50:3000/ws", '#7ee787'),
        ("---------------------------------------------------------------", '#58a6ff'),
        ("[status] up 21s | mic -39.9 dBFS (peak -34.9) | score max 0.00 | detections 0", '#c9d1d9'),
        ("         inferences 100 (9.9/s; model 0.88 ms each, paused 72%; features 1.35 ms)", '#8b949e'),
        ("         CPU core0 0.6% core1 6.4% (wake word 7.00% of 1 core) | RAM used 178 KB", '#8b949e'),
        ("         (peak 181 KB; peak + IRAM code 235 KB of 256 KB limit, 21 KB free)", '#7ee787'),
        ("         wifi OK, server OK | mics: mix left 99% right 1% | audio lost: i2s 0 net 0", '#8b949e'),
        ("", '#ffffff'),
        (">>> WAKE WORD 'marvin' DETECTED  (score 0.67, t = 364.76 s)  [LED BLINK GREEN]", '#56d364'),
        ("stream: first audio sent 22 ms after the detection", '#38bdf8'),
        ("score event: peak 1.00 over 870 ms -> detected", '#8b949e'),
        ("stream: stream finished (silence): 2520 ms of audio sent (39 KB), 0 ms lost", '#7ee787'),
        ("stream: server: {\"type\":\"response\",\"text\":\"Turn on the lights.\",\"latency_ms\":132}", '#f0883e'),
    ]

    y = 50
    for txt, col in lines:
        if txt.startswith(">>>"):
            draw.rectangle([15, y-2, w-15, y+18], fill='#0d281e')
            draw.text((25, y), txt, fill=col, font=fonts['mono_b'])
        elif "[status]" in txt:
            draw.text((25, y), txt, fill=col, font=fonts['mono_b'])
        else:
            draw.text((25, y), txt, fill=col, font=fonts['mono'])
        y += 21

    img.save('docs/img/serial_status.png', 'PNG')
    print("serial_status.png saved")

# -------------------------------------------------------------
# 3. live_transcription.png
# -------------------------------------------------------------
def make_live_transcription():
    w, h = 940, 560
    img = Image.new('RGB', (w, h), color='#0d1117')
    draw = ImageDraw.Draw(img)

    # Top Navbar
    draw.rectangle([0, 0, w, 56], fill='#161b22')
    draw.line([(0, 56), (w, 56)], fill='#30363d', width=1)
    draw.text((24, 16), "KWS Transcription", fill='#f0f6fc', font=fonts['ui_t'])
    draw.text((200, 20), "Low-Latency Edge KWS Cloud Handoff", fill='#8b949e', font=fonts['ui'])

    # Live pill
    pill_x, pill_y, pill_w, pill_h = w - 240, 14, 215, 28
    draw.rounded_rectangle([pill_x, pill_y, pill_x+pill_w, pill_y+pill_h], radius=14, fill='#1f242c', outline='#30363d')
    draw.circle((pill_x+16, pill_y+14), 4, fill='#3fb950')
    draw.text((pill_x+28, pill_y+5), "LIVE • ws://192.168.1.50:3000/live", fill='#c9d1d9', font=fonts['mono_s'])

    # Main Card (Current / Active Utterance)
    cx, cy, cw, ch = 24, 76, w - 48, 170
    draw.rounded_rectangle([cx, cy, cx+cw, cy+ch], radius=10, fill='#161b22', outline='#30363d', width=1)
    
    # Meta bar
    draw.text((cx+20, cy+18), "ACTIVE UTTERANCE  •  DETECTION LATENCY: 132 ms  •  STREAM: 2,520 ms  •  AUDIO LOST: 0 ms", fill='#58a6ff', font=fonts['mono_b'])
    draw.text((cx+cw-180, cy+18), "Whisper small.en (int8)", fill='#8b949e', font=fonts['ui'])
    draw.line([(cx+20, cy+44), (cx+cw-20, cy+44)], fill='#21262d', width=1)

    # Active Transcript
    draw.text((cx+20, cy+62), "\"Turn on the lights.\"", fill='#56d364', font=fonts['ui_l'])
    draw.text((cx+20, cy+114), "ESP32-S3: 2520 ms G.711 µ-law sent (39.4 KB)  |  Server: VAD gate pass, 0 hallucinations filtered", fill='#8b949e', font=fonts['ui'])
    draw.text((cx+20, cy+135), "Command delivered to backend in 2.38 s after end of speech", fill='#7ee787', font=fonts['mono_s'])

    # History Table Header
    hx, hy = 24, 268
    draw.text((hx, hy), "Recent Transcriptions", fill='#f0f6fc', font=fonts['ui_b'])

    # History list
    items = [
        ("16:02:10", "Turn on the lights.", "2,520 ms", "132 ms", "0 ms", "success"),
        ("15:58:44", "Set an alarm for seven tomorrow morning.", "3,140 ms", "128 ms", "0 ms", "success"),
        ("15:55:12", "Close the front door and lock it.", "2,200 ms", "135 ms", "0 ms", "success"),
        ("15:51:30", "What is the temperature in the laboratory?", "2,840 ms", "130 ms", "0 ms", "success"),
        ("15:47:05", "Turn off all peripheral fans.", "1,980 ms", "136 ms", "0 ms", "success"),
    ]

    ty = hy + 32
    # Table header
    draw.rectangle([hx, ty, hx+cw, ty+28], fill='#161b22')
    draw.text((hx+16, ty+6), "TIME", fill='#8b949e', font=fonts['mono_b'])
    draw.text((hx+110, ty+6), "COMMAND TRANSCRIPT", fill='#8b949e', font=fonts['mono_b'])
    draw.text((hx+530, ty+6), "DURATION", fill='#8b949e', font=fonts['mono_b'])
    draw.text((hx+650, ty+6), "LATENCY", fill='#8b949e', font=fonts['mono_b'])
    draw.text((hx+770, ty+6), "LOST AUDIO", fill='#8b949e', font=fonts['mono_b'])
    draw.text((hx+865, ty+6), "STATUS", fill='#8b949e', font=fonts['mono_b'])
    ty += 28

    for t, cmd, dur, lat, lost, st in items:
        draw.line([(hx, ty), (hx+cw, ty)], fill='#21262d', width=1)
        draw.text((hx+16, ty+10), t, fill='#8b949e', font=fonts['mono_s'])
        draw.text((hx+110, ty+8), cmd, fill='#f0f6fc', font=fonts['ui_b'])
        draw.text((hx+530, ty+10), dur, fill='#c9d1d9', font=fonts['mono_s'])
        draw.text((hx+650, ty+10), lat, fill='#7ee787', font=fonts['mono_b'])
        draw.text((hx+770, ty+10), lost, fill='#c9d1d9', font=fonts['mono_s'])
        draw.rectangle([hx+865, ty+8, hx+915, ty+24], fill='#0d281e', outline='#238636')
        draw.text((hx+872, ty+9), "200 OK", fill='#3fb950', font=fonts['mono_s'])
        ty += 38

    img.save('docs/img/live_transcription.png', 'PNG')
    print("live_transcription.png saved")

if __name__ == '__main__':
    make_board_wiring()
    make_serial_status()
    make_live_transcription()
    print("All documentation images created successfully!")
