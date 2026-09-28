# Training: the deployed "marvin" model (v2)

`SIH_marvin_retrain_v2.ipynb` (Google Colab, T4 GPU, 2.5-4 h) produced `kws_s3/model/marvin.tflite`
(sha256 `62eef78f56a92dcf3048...`, 60,896 bytes, identical to `marvin_export2/marvin.tflite`).

* Framework: microWakeWord @ `4665173c` (Apache-2.0), MixedNet streaming model trained **from scratch**, int8 export.
  Piper sample generator @ `2971426a` (MIT) for synthetic voices.
* Positives: Google Speech Commands v2 "marvin" (CC-BY-4.0, official speaker split) + 2,000 **synthetic** Piper TTS
  clips (augmentation; disclosed).
* Negatives: other Speech Commands words (20,000), 16 sound-alike phrases x 400 **synthetic** Piper clips,
  14 real false triggers recorded by the board (each used 10x), microWakeWord's negative feature sets and noise sets
  (some **CC-BY-NC**: fine for SIH, not for a commercial product).
* Board recordings used in training were made on 27 Sep between 04:18 and 06:00. The frozen benchmark sets
  (`benchmarks/sets/real_*.csv`) only hold recordings from 27 Sep 08:12 onward: no overlap.

Known flaw: Step 12 suggests the cutoff from microWakeWord's ROC on its **testing** split. The deployed cutoff (0.6) was
re-chosen on the frozen validation set instead (`benchmarks/sweep_window.py`, fix F5). For a retrain, pick the cutoff
the same way and report the frozen test set once.
