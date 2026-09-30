# Streaming-path check: plays the 6 synthetic "Marvin" clips N times at a loud level (reliable detections) and prints
# the board's "detection -> first audio sent" times, network drops and RAM peak. Isolates the firmware's streaming path
# from acoustic conditions. Usage: .\run_trigger.ps1 -Label expX [-Repeats 3] [-Level -18]
param([Parameter(Mandatory)][string]$Label, [int]$Repeats = 3, [int]$Level = -18)
$py = if ($env:PYTHON) { $env:PYTHON } else { 'python' }
Set-Location $PSScriptRoot
$srcs = @('audio/synthetic_marvin') * $Repeats
& $py run_acoustic.py --label "trig_$Label" --level-db $Level --name trigger @srcs 2>&1 | Out-Null
$log = "results/trig_${Label}_acoustic_trigger.log"
$ms = Select-String -Path $log -Pattern "first audio sent (-?\d+)" | ForEach-Object { [int]$_.Matches[0].Groups[1].Value } | Sort-Object
"$Label  streams $($ms.Count)  first-send ms: $($ms -join ' ')"
(Select-String -Path $log -Pattern "\[status\]" | Select-Object -Last 1).Line -replace '.*\| RAM', 'RAM'
& $py -c "import json; s=json.load(open('results/trig_${Label}_acoustic_trigger.json'))['summary']; print('latency', s['latency_ms'], 'ping', s['ping_rtt_ms'])"
