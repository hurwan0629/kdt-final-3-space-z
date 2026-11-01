param(
    [string]$ExperimentName = '003_body_stability',
    [string]$RunNumber = '003',
    [int]$InitialStep = 300000,
    [int]$Blocks = 5
)
$ErrorActionPreference = 'Stop'
$experimentRoot = Join-Path $PSScriptRoot '.'
$env:PYTHONPATH = "$PWD/src"
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUTF8 = '1'
$env:PYGAME_HIDE_SUPPORT_PROMPT = '1'
$env:MPLCONFIGDIR = "$experimentRoot/.cache/matplotlib"
foreach ($block in 1..$Blocks) {
    $step = $InitialStep + $block * 100000
    if (Test-Path -LiteralPath "$experimentRoot/$($RunNumber)_$($step / 1000)k_broader_development_comparison.json") { continue }
    $checkpoint = "$experimentRoot/experiments/$ExperimentName/checkpoints/evaluated_$('{0:D9}' -f $step).zip"
    while (-not (Test-Path -LiteralPath $checkpoint)) {
        if (Test-Path -LiteralPath "$experimentRoot/$($RunNumber)_exit_code.txt") { exit 0 }
        Start-Sleep -Seconds 5
    }
    $label = "$($RunNumber)_$($step / 1000)k"
    $extra = "$experimentRoot/$($label)_extra_development.json"
    if (-not (Test-Path -LiteralPath $extra)) {
        & "$PSScriptRoot/process_slot.ps1" -PythonArguments @('-m', 'space_z.rl.gpt_training_rotation_continuous.evaluate', $checkpoint, '--seeds', '43506', '44506', '45506', '46506', '--max-seconds', '200', '--output', $extra) *> "$experimentRoot/$($label)_extra_development.log"
    }
    do {
        $matching = @(Get-Content -LiteralPath "$experimentRoot/experiments/$ExperimentName/evaluations.jsonl" | ForEach-Object { $_ | ConvertFrom-Json } | Where-Object { $_.timesteps -eq $step })
        if ($matching.Count -eq 0) { Start-Sleep -Seconds 5 }
    } while ($matching.Count -eq 0)
    $env:ROTATION_EXTRA_REPORT = $extra
    $env:ROTATION_EVALUATED_STEP = "$step"
    $env:ROTATION_REPORT_LABEL = "$($label)_broader_development"
    $env:ROTATION_EXPERIMENT_NAME = $ExperimentName
    $mergeCode = @'
import os
from space_z.rl.gpt_training_rotation_continuous.report import merge_development_evaluation
merge_development_evaluation(os.environ['ROTATION_EXPERIMENT_NAME'], int(os.environ['ROTATION_EVALUATED_STEP']), os.environ['ROTATION_EXTRA_REPORT'], os.environ['ROTATION_REPORT_LABEL'])
'@
    & "$PSScriptRoot/process_slot.ps1" -PythonArguments @('-c', $mergeCode) *> "$experimentRoot/$($label)_comparison.log"
}
Set-Content -LiteralPath "$experimentRoot/$($RunNumber)_extra_validation_exit_code.txt" '0'
