# Full re-verification of the flashed (production) firmware, Wi-Fi + server up. Same protocols as the baseline.
#   1) idle CPU/RAM, quiet room and continuous (synthetic) speech, 6 min each
#   2) acoustic end-to-end: synthetic latency clips twice + the frozen real test set once (latency, TPR/FA, transcripts,
#      RAM while streaming)
# The on-device accuracy of the final model/cutoff comes from run_injected.py (no Wi-Fi needed, see README).
param([string]$Label = 'final')
Set-Location $PSScriptRoot
C:\Python314\python.exe run_idle.py --label $Label --cond quiet --minutes 6
C:\Python314\python.exe run_idle.py --label $Label --cond speech --minutes 6 --gain 0.5
C:\Python314\python.exe run_acoustic.py --label $Label --level-db -20 --name latency_synth_pass1 sets/synthetic_latency.csv
C:\Python314\python.exe run_acoustic.py --label $Label --level-db -20 --name latency_synth_pass2 sets/synthetic_latency.csv
C:\Python314\python.exe run_acoustic.py --label $Label --level-db -20 sets/real_test.csv
C:\Python314\python.exe summarize_acoustic.py "results/${Label}_acoustic_latency_synth_pass1.json" "results/${Label}_acoustic_latency_synth_pass2.json" > "results/${Label}_latency_synth_summary.json"
C:\Python314\python.exe summarize_acoustic.py "results/${Label}_acoustic_latency_synth_pass1.json" "results/${Label}_acoustic_latency_synth_pass2.json" "results/${Label}_acoustic_real_test.json" > "results/${Label}_latency_all_summary.json"
