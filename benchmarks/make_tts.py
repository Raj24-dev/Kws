"""SYNTHETIC audio with the Windows speech voices (SAPI, 16 kHz 16-bit mono). Everything this writes is
synthetic and is labelled so in every result that uses it.

    python make_tts.py idle      -> audio/synthetic_idle_speech.wav  (~5 min of neutral talk, no wake word; keeps
                                    the device's quiet-room gate open during the CPU test)
    python make_tts.py longspeech -> audio/synthetic_longspeech_*.wav (~2 h of read-aloud English from the ESP-IDF
                                    documentation, both voices, 3 rates) for a HOST-ONLY false-accepts-per-hour estimate
    python make_tts.py confusable -> audio/synthetic_confusable/*.wav (sound-alike words, voices x rates) and
                                    audio/synthetic_marvin/*.wav (the wake word, same voices), for the synthetic test
"""
import os
import sys

import numpy as np
import soundfile as sf
import win32com.client as w

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIO = os.path.join(HERE, "audio")
SAFT16kHz16BitMono = 18

IDLE_TEXT = """The weather this week will stay warm and dry, with light winds from the west and clear skies in the evening.
Farmers in the region expect a good harvest of rice and wheat if the rains arrive on time next month.
To make a simple vegetable soup, chop two onions, three carrots and a handful of beans, then boil them with salt and pepper.
The library opens at nine in the morning and closes at six, except on public holidays when it stays closed all day.
Students should submit their project reports before Friday, and each team will present for ten minutes on Monday.
The train to the capital leaves from platform four, and passengers are requested to keep their tickets ready.
A healthy breakfast gives you the energy to focus in class, so try to eat fruit, eggs or a bowl of oats every day.
The football match ended in a draw after both teams scored twice in the second half, and the crowd cheered loudly.
Please remember to switch off the lights and the fans when you leave the room, because saving power helps everyone.
Our city will plant ten thousand trees this year along the main roads, the river bank and near every school.
The museum has a new gallery of old coins, maps and letters, and entry is free for children under twelve.
If the internet connection is slow, restart the router, wait for a minute, and then try to load the page again."""

CONFUSABLES = ["Pravin", "Navin", "Kevin", "Melvin", "Morgan", "Martin", "Marvel", "Margin", "Marion", "Marlin",
               "Garvin", "Mervyn", "Marvelous", "Martian", "Harvard", "Carbon", "Mark", "Marble", "Starving", "Arvind"]
RATES = [-3, 0, 3]


def voices():
    v = w.Dispatch("SAPI.SpVoice")
    return {t.GetDescription().split()[1]: t for t in v.GetVoices()}  # David, Zira


def speak(text, path, voice, rate=0):
    stream = w.Dispatch("SAPI.SpFileStream")
    fmt = w.Dispatch("SAPI.SpAudioFormat")
    fmt.Type = SAFT16kHz16BitMono
    stream.Format = fmt
    stream.Open(path, 3)
    sp = w.Dispatch("SAPI.SpVoice")
    sp.AudioOutputStream = stream
    sp.Voice = voice
    sp.Rate = rate
    sp.Speak(text)
    stream.Close()


def main(what):
    os.makedirs(AUDIO, exist_ok=True)
    vs = voices()
    if what == "idle":
        parts = []
        for k, (name, tok) in enumerate(sorted(vs.items())):
            p = os.path.join(AUDIO, f"_tmp_{name}.wav")
            speak(IDLE_TEXT, p, tok, rate=0)
            x, sr = sf.read(p, dtype="int16")
            os.remove(p)
            parts += [x, np.zeros(sr // 2, np.int16)]
        x = np.concatenate(parts)
        sf.write(os.path.join(AUDIO, "synthetic_idle_speech.wav"), x, 16000, subtype="PCM_16")
        print(f"synthetic_idle_speech.wav: {len(x) / 16000:.0f} s")
    elif what == "longspeech":
        longspeech()
    elif what == "confusable":
        for sub, words in (("synthetic_confusable", CONFUSABLES), ("synthetic_marvin", ["Marvin"])):
            d = os.path.join(AUDIO, sub)
            os.makedirs(d, exist_ok=True)
            n = 0
            for name, tok in sorted(vs.items()):
                for r in RATES:
                    for word in words:
                        speak(word, os.path.join(d, f"{name}_r{r}_{word}.wav"), tok, r)
                        n += 1
            print(f"{sub}: {n} clips")


def long_text(max_words):
    """Plain English sentences from the ESP-IDF docs (.rst), markup stripped; any sentence with a Marvin-like word dropped."""
    import glob
    import re
    idf = os.environ.get("IDF_PATH") or "C:/esp/v5.5.5/esp-idf"
    words, out = 0, []
    for f in sorted(glob.glob(os.path.join(idf, "docs", "en", "**", "*.rst"), recursive=True)):
        for line in open(f, encoding="utf-8", errors="ignore"):
            line = line.strip()
            if not line or line[0] in ".:=-*#|+`" or "::" in line or "_" in line or "http" in line:
                continue
            line = re.sub(r"[`*<>\[\]{}|]", " ", line)
            if len(line.split()) < 6 or re.search(r"\b(mar|garv|harv|arv|erv|elv)", line, re.I):
                continue
            out.append(line)
            words += len(line.split())
            if words >= max_words:
                return out
    return out


def longspeech(hours=2.0):
    vs = voices()
    lines = long_text(int(hours * 3600 * 2.6))  # ~2.6 words/s
    chunk = 400  # lines per file
    k = 0
    for i in range(0, len(lines), chunk):
        name, tok = sorted(vs.items())[k % len(vs)]
        rate = RATES[(k // len(vs)) % len(RATES)]
        p = os.path.join(AUDIO, f"synthetic_longspeech_{k:02d}_{name}_r{rate}.wav")
        speak(" ".join(lines[i:i + chunk]), p, tok, rate)
        x, _ = sf.read(p, dtype="int16")
        print(os.path.basename(p), f"{len(x) / 16000 / 60:.1f} min", flush=True)
        k += 1


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "idle")
