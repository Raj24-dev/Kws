# Full re-verification of the flashed (production) firmware, Wi-Fi + server up. Same protocols as the baseline.
#   1) idle CPU/RAM, quiet room and continuous (synthetic) speech, 6 min each
#   2) acoustic end-to-end: synthetic latency clips twice + the frozen real test set once (latency, TPR/FA, transcripts,
#      RAM while streaming)
# The on-device accuracy of the final model/cutoff comes from run_injected.py (no Wi-Fi needed, see README).
# -Level/-SpeechGain: keep the RECEIVED level equal to the baseline run (check with calibrate_level.py first).
param([string]$Label = 'final', [double]$Level = -24, [double]$SpeechGain = 0.5)
$py = if ($env:PYTHON) { $env:PYTHON } else { 'python' }
Set-Location $PSScriptRoot
& $py run_idle.py --label $Label --cond quiet --minutes 6
& $py run_idle.py --label $Label --cond speech --minutes 6 --gain $SpeechGain
& $py run_acoustic.py --label $Label --level-db $Level --name latency_synth_pass1 sets/synthetic_latency.csv
& $py run_acoustic.py --label $Label --level-db $Level --name latency_synth_pass2 sets/synthetic_latency.csv
& $py run_acoustic.py --label $Label --level-db $Level sets/real_test.csv
& $py summarize_acoustic.py "results/${Label}_acoustic_latency_synth_pass1.json" "results/${Label}_acoustic_latency_synth_pass2.json" > "results/${Label}_latency_synth_summary.json"
& $py summarize_acoustic.py "results/${Label}_acoustic_latency_synth_pass1.json" "results/${Label}_acoustic_latency_synth_pass2.json" "results/${Label}_acoustic_real_test.json" > "results/${Label}_latency_all_summary.json"
