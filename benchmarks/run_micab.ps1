# Dual-microphone A/B ([DEVICE-ACOUSTIC], synthetic clips): both microphones time-aligned vs not aligned ("average",
# still mixed by their noise) vs the left one only, interleaved passes at a low playback level where detection is near
# its knee. Results: results/micab_*.json (micab_both_*: the plain 50/50 average before either existed).
# Check first that the status line shows "mix left 50% right 50%": with a faulty microphone all three are "left only".
# Put the speaker off to one side of the pair (in line with the two microphones): that is where alignment matters.
param([int]$Level = -26, [int]$Passes = 2)
# resolved before the IDF profile below puts its own python first on PATH
$py = if ($env:PYTHON) { $env:PYTHON } else { (Get-Command python).Source }
if (-not $env:IDF_PATH) {
    $prof = if ($env:IDF_PS_PROFILE) { $env:IDF_PS_PROFILE } else { 'C:\Espressif\tools\Microsoft.v5.5.5.PowerShell_profile.ps1' }
    . $prof *> $null
}
$fw = Join-Path $PSScriptRoot '..\firmware'
$variants = [ordered]@{
    aligned = @('KWS_MIC_SELECT=0', 'KWS_MIC_SPACING_MM=55')
    average = @('KWS_MIC_SELECT=0', 'KWS_MIC_SPACING_MM=0')
    left    = @('KWS_MIC_SELECT=1', 'KWS_MIC_SPACING_MM=55')
}
for ($p = 1; $p -le $Passes; $p++) {
    foreach ($name in $variants.Keys) {
        Set-Location $fw
        & $py ..\benchmarks\set_config.py @($variants[$name]) | Out-Null
        idf.py reconfigure 2>&1 | Out-Null
        ninja -C build -j 3 2>&1 | Out-Null
        idf.py -p COM6 -b 921600 flash 2>&1 | Out-Null
        Set-Location $PSScriptRoot
        & $py run_acoustic.py --label "micab_$name" --offline --level-db $Level --name "pass$p" audio/micab 2>&1 | Out-Null
        & $py -c "import json; s=json.load(open('results/micab_${name}_acoustic_pass$p.json'))['summary']; print('$name pass $p', s['detected'], '/', s['positives'])"
    }
}
Set-Location $fw
& $py ..\benchmarks\set_config.py KWS_MIC_SELECT=0 KWS_MIC_SPACING_MM=55 | Out-Null
