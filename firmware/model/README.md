# Model folder

The firmware embeds the files in this folder at build time (`main/embed_model.cmake`).

## Deployed model: `marvin.tflite` + `marvin.json`

| | |
|---|---|
| type | microWakeWord MixedNet, streaming, int8, 60,896 bytes |
| input | `[1, 3, 40]` int8 (3 feature frames of 40 mel bands; scale 0.10196, zero point -128) |
| output | `[1, 1]` uint8 probability (scale 1/256) |
| operations (13) | ASSIGN_VARIABLE, CALL_ONCE, CONCATENATION, CONV_2D, DEPTHWISE_CONV_2D, FULLY_CONNECTED, LOGISTIC, QUANTIZE, READ_VARIABLE, RESHAPE, SPLIT_V, STRIDED_SLICE, VAR_HANDLE |
| settings (`marvin.json`) | cutoff 0.6, sliding window 5, 10 ms feature step, tensor arena 30,000 bytes (the board uses 25,964) |

How it was trained: [`../../training`](../../training).

## Using another model

From the zip that the training notebook's last step downloads (`<name>_export.zip`), copy **two files** here:

| file | what it is | needed? |
|---|---|---|
| `<name>.tflite` | the trained model | **yes** |
| `<name>.json` | threshold + window size + wake word name | recommended |

Optional: one recording of the wake word as `<anything>.wav` (**16 kHz, 16-bit**); the firmware then runs a
**self-test** on it at boot. Keep only **one** `.tflite`, one `.json` and one `.wav` here, then rebuild.

With no `.tflite` here, the firmware still builds and runs in **microphone-test mode** (level meter + manual
streaming with the BOOT button), which is handy for checking the hardware before a model is ready.

The model must be a *streaming* microWakeWord model (input `[1, stride, 40]` int8, 10 ms feature steps); a
1-second classifier with an input like `[1, 98, 40, 1]` needs a different feature pipeline and is rejected at boot.
