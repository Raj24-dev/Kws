"""[HOST-ONLY] Which rule removes the wake word from a 250 ms pre-roll transcript without cutting the command?

Whisper runs once per recording (word timestamps, the server's own transcription code); every rule is then applied
to the cached words. Sets:
  val : real recordings with the whole wake word (host validation split), stream = x[d - 0.25 s:], d from the
        firmware mirror, reference = transcript of the audio from the annotated keyword end (as eval_preroll.py)
  f10 : what the board itself streamed with a 250 ms pre-roll (run F10), reference = the clip's known command
"""
import csv
import json
import os
import pickle
import sys
import tempfile

import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(ROOT, "benchmarks")
for p in (os.path.join(ROOT, "training"), os.path.join(ROOT, "cloudServer", "server", "stt-services"), BENCH):
    sys.path.insert(0, p)
from summarize_acoustic import wer, words  # noqa: E402

REC = os.path.join(ROOT, "cloudServer", "server", "recordings")
CACHE = os.path.join(tempfile.gettempdir(), "kws_strip_rules_cache.pkl")  # Whisper words, reused between runs
P = 0.25
stt = None


def whisper_words(pcm: bytes):
    """None = no speech (the server returns ""), else [(word, start, end, probability)]"""
    audio = stt.to_float(pcm)
    if not len(audio) or not stt.get_speech_timestamps(audio, stt.VadOptions()):
        return None
    return [(w.word, w.start, w.end, w.probability)
            for s in stt.segments_of(audio, stt.PUNCTUATION_PROMPT, words=True) for w in s.words]


def build_val():
    from check_model import Detector, LcgNoise, run
    model = os.path.join(ROOT, "kws_s3", "model", "marvin.tflite")
    det = Detector(model, json.load(open(model.replace(".tflite", ".json")))["micro"]["probability_cutoff"], 5)
    items = []
    for r in csv.DictReader(open(os.path.join(BENCH, "sets", "real_val.csv"), encoding="utf-8")):
        if r["label"] != "1" or not r["kw_end"]:
            continue
        x, _ = sf.read(os.path.join(REC, r["file"]), dtype="int16")
        found, _ = run(det, [LcgNoise(1).take(1200 * 16), x])
        if not found:
            continue
        d = (found[0][0] - 1200 * 16) / 16000
        ref = stt.transcribe_command(x[int(float(r["kw_end"]) * 16000):].tobytes())
        if not ref:
            continue
        p = min(P, d)
        items.append({"set": "val", "file": r["file"], "ref": ref, "detect_s": p,
                      "words": whisper_words(x[int((d - p) * 16000):].tobytes())})
        print(items[-1]["file"], items[-1]["ref"], flush=True)
    return items


def build_f10():
    cmds = {r["file"]: r["command"] for r in csv.DictReader(open(os.path.join(BENCH, "sets", "synthetic_latency.csv")))}
    trials = []
    for name in ("F10_preroll250_acoustic_latency_synth_pass1", "F10_preroll250_acoustic_latency_synth_pass2",
                 "trig_F10_preroll250_acoustic_trigger"):
        for t in json.load(open(os.path.join(BENCH, "results", name + ".json")))["trials"]:
            if t["detected"]:
                trials.append((cmds.get(t["file"], ""), t["transcript"]))
    recs = [json.loads(l) for l in open(os.path.join(REC, "index.jsonl"), encoding="utf-8") if l.strip()]
    recs = [r for r in recs if r.get("preroll_ms") == 250]
    items, i = [], 0
    for cmd, said in trials:  # align trials with the saved recordings by their transcript (both in time order)
        while i < len(recs) and (recs[i].get("transcript") or "") != (said or ""):
            i += 1
        if i == len(recs):
            break
        x, _ = sf.read(os.path.join(REC, recs[i]["file"]), dtype="int16")
        items.append({"set": "f10", "file": recs[i]["file"], "ref": cmd, "detect_s": P, "words": whisper_words(x.tobytes()),
                      "server_then": said})
        print(items[-1]["file"], repr(cmd), repr(said), flush=True)
        i += 1
    return items


def text(ws):
    return stt.tidy("".join(w[0] for w in ws)) if ws else ""


def rule_current(ws, d):  # stt-services/main.py today
    ws = [w for w in ws if w[2] > d - 0.05]
    while ws and ws[0][1] < d and stt.is_wake_word_part(ws[0][0]):
        ws.pop(0)
    return ws


def rule_end(delta):  # also drop a leading word that ends before the detection point + delta, whatever it sounds like
    def f(ws, d):
        ws = [w for w in ws if w[2] > d - 0.05]
        while ws and ws[0][1] < d and (stt.is_wake_word_part(ws[0][0]) or ws[0][2] <= d + delta):
            ws.pop(0)
        return ws
    return f


def rule_prob(p0, delta=None):  # also drop a leading word from before the detection that Whisper is unsure of
    def f(ws, d):
        ws = [w for w in ws if w[2] > d - 0.05]
        while ws and ws[0][1] < d and (stt.is_wake_word_part(ws[0][0]) or ws[0][3] < p0
                                       or (delta is not None and ws[0][2] <= d + delta)):
            ws.pop(0)
        return ws
    return f


RULES = {"current": rule_current, "end+0.00": rule_end(0.0), "end+0.05": rule_end(0.05), "end+0.10": rule_end(0.10),
         "prob<0.5": rule_prob(0.5), "prob<0.7": rule_prob(0.7), "prob<0.5|end+0.05": rule_prob(0.5, 0.05)}


def score(items, rule):
    errs = n = leaks = cut = 0
    bad = []
    for it in items:
        hyp = text(rule(list(it["words"]), it["detect_s"])) if it["words"] else ""
        rw, hw = words(it["ref"]), words(hyp)
        e, m = wer(it["ref"], hyp)
        errs, n = errs + e, n + max(m, 1)
        leak = (bool(hw) and not rw) or (len(hw) > 1 and bool(rw) and hw[0] != rw[0] and hw[1] == rw[0])
        lost_first = bool(rw) and bool(hw) and hw[0] == (rw[1] if len(rw) > 1 else None)
        leaks, cut = leaks + leak, cut + lost_first
        if leak or lost_first:
            bad.append((it["ref"], hyp))
    return errs / n, leaks, cut, bad


if __name__ == "__main__":
    if os.path.exists(CACHE):
        items = pickle.load(open(CACHE, "rb"))
    else:
        import main as stt_module
        stt = stt_module
        items = build_val() + build_f10()
        pickle.dump(items, open(CACHE, "wb"))
    if stt is None:
        import main as stt_module
        stt = stt_module
    for s in ("val", "f10"):
        sub = [it for it in items if it["set"] == s]
        print(f"\n== {s}: {len(sub)} recordings")
        for name, rule in RULES.items():
            w, leaks, cut, bad = score(sub, rule)
            print(f"  {name:18s} WER {w:.3f}  leaks {leaks}  first word cut {cut}  " + "; ".join(f"{r!r}->{h!r}" for r, h in bad[:4]))
