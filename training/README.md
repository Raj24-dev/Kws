# Wake word model ("marvin")

| file | what it is |
|---|---|
| `SIH_marvin_retrain_v2.ipynb` | training notebook (Google Colab, T4 GPU, 2.5-4 h); exports `marvin.tflite` + `marvin.json` |
| `check_model.py` | runs a model on WAV files with the firmware's exact pipeline (features, int8 input, streaming, sliding window, cool-down, quiet gate) on a PC or in Colab |
| `export_training_clips.py` | cuts the wake word out of board recordings (made with a pre-roll) into `device_positives.zip` / `device_hard_negatives.zip` for the notebook |

## The deployed model (v2)

`kws_s3/model/marvin.tflite` (60,896 bytes, sha256 `62eef78f56a92dcf3048...`) came from this notebook.

* Framework: microWakeWord @ `4665173c` (Apache-2.0), MixedNet streaming model trained **from scratch**, int8 export.
  Piper sample generator @ `2971426a` (MIT) for synthetic voices.
* Positives: Google Speech Commands v2 "marvin" (CC-BY-4.0, official speaker split) + 2,000 **synthetic** Piper TTS
  clips (augmentation; disclosed).
* Negatives: other Speech Commands words (20,000), 16 sound-alike phrases × 400 **synthetic** Piper clips,
  14 real false triggers recorded by the board (each used 10×), microWakeWord's negative feature sets and noise sets
  (some **CC-BY-NC**: fine for SIH, not for a commercial product).
* Board recordings used in training were made on 27 Sep between 04:18 and 06:00. The frozen benchmark sets
  (`benchmarks/sets/real_*.csv`) only hold recordings from 27 Sep 08:12 onward: no overlap.
* Cutoff 0.6, chosen on the frozen validation set (`benchmarks/sweep_window.py`), not on microWakeWord's own
  testing split (which Step 12 of the notebook suggests). For a retrain, pick the cutoff the same way and report
  the frozen test set once.

## Next model (v3)

The current model still fires on sound-alike words (Martin, Marvel, Melvin, Kevin, Morgan...). The notebook already
has `CONFUSABLE_PHRASES` and `MY_NEGATIVES_ZIP`: add those words as negatives, feed it the board's recorded false
triggers from `export_training_clips.py`, and give the run a new `RUN_NAME` (reusing an old Drive folder resumes the
old checkpoint).

## Checking a model on the PC

```
pip install numpy pymicro-features ai-edge-litert soundfile
python check_model.py ../kws_s3/model/marvin.tflite recording.wav [more.wav ...]
```
It prints the highest averaged score and when a detection fired, as the board's boot self-test does. The scores
agree with the board within ~0.03; for exact on-device numbers use `benchmarks/run_injected.py`.
