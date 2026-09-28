# Put your trained model here

From the zip that Colab **Step 13** downloads (`<name>_export.zip`), copy **two files** into this folder:

| file | what it is | needed? |
|---|---|---|
| `<name>.tflite` | the trained model | **yes** |
| `<name>.json` | threshold + window size + wake word name | recommended |

Optional: one recording of the wake word as `<anything>.wav` (**16 kHz, 16-bit**). The new notebook exports
`test_0.wav`; alternatively take any file from the `marvin` folder of the Speech Commands dataset, or from Colab's
`/content/work/clips/positives/testing/` folder. The firmware then runs a **self-test** at boot on that recording.

Keep only **one** `.tflite`, one `.json` and one `.wav` here. Then rebuild (`idf.py build flash monitor`).

With no `.tflite` here, the firmware still builds and runs in **microphone-test mode** (level meter +
manual streaming with the BOOT button), which is handy for checking the hardware before the model is ready.

Do **not** use the old `model_data.cc` from the first project: that model is a different kind
(a 1-second classifier with an unknown feature recipe), and this firmware cannot run it.
