$ErrorActionPreference = 'Stop'
$experimentRoot = $PSScriptRoot
while (-not ((Test-Path -LiteralPath "$experimentRoot/005_exit_code.txt") -and (Test-Path -LiteralPath "$experimentRoot/005_extra_validation_exit_code.txt"))) {
    Start-Sleep -Seconds 5
}
$phase005 = Get-Content -LiteralPath "$experimentRoot/experiments/005_aim_precision/experiment.json" -Raw | ConvertFrom-Json
if ($phase005.status -ne 'completed' -or $phase005.additional_steps -ne 500000) { throw 'Final block did not complete its budget' }
& "$experimentRoot/process_slot.ps1" -PythonArguments @('-m', 'space_z.rl.gpt_training_rotation_continuous.select_model') *> "$experimentRoot/candidate_selection.log"
$protocol = Get-Content -LiteralPath "$experimentRoot/protocol.json" -Raw | ConvertFrom-Json
$evaluationArguments = @('-m', 'space_z.rl.gpt_training_rotation_continuous.evaluate', "$experimentRoot/selected_model.zip", '--seeds')
$evaluationArguments += @($protocol.heldout_seeds | ForEach-Object { "$_" })
$evaluationArguments += @('--max-seconds', '400', '--output', "$experimentRoot/heldout_candidate.json")
& "$experimentRoot/process_slot.ps1" -PythonArguments $evaluationArguments *> "$experimentRoot/heldout_candidate.log"
& "$experimentRoot/process_slot.ps1" -PythonArguments @('-m', 'space_z.rl.gpt_training_rotation_continuous.report') *> "$experimentRoot/final_comparison.log"
Set-Content -LiteralPath "$experimentRoot/final_comparison_exit_code.txt" '0'
