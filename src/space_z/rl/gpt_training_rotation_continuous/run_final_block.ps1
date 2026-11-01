$ErrorActionPreference = 'Stop'
$experimentRoot = $PSScriptRoot
while (-not ((Test-Path -LiteralPath "$experimentRoot/004_exit_code.txt") -and (Test-Path -LiteralPath "$experimentRoot/004_extra_validation_exit_code.txt"))) {
    Start-Sleep -Seconds 5
}
$phase004 = Get-Content -LiteralPath "$experimentRoot/experiments/004_body_continuation/experiment.json" -Raw | ConvertFrom-Json
if ($phase004.status -ne 'completed' -or $phase004.additional_steps -ne 500000) { throw 'Phase 004 did not complete its training budget' }
$comparisonPaths = & rg --files $experimentRoot -g '*_broader_development_comparison.json'
$candidates = foreach ($comparisonPath in $comparisonPaths) {
    $comparison = Get-Content -LiteralPath $comparisonPath -Raw | ConvertFrom-Json
    if (-not $comparison.all_guards_passed) { continue }
    $reportPath = $comparisonPath.Replace('_comparison.json', '.json')
    $report = (Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json)[0]
    if ($null -eq $report.timesteps) { throw "Missing model step count: $reportPath" }
    [pscustomobject]@{
        report = $reportPath
        checkpoint = $report.checkpoint
        checkpoint_sha256 = $report.checkpoint_sha256
        new_action_lineage_steps = $report.timesteps
        ranking = $comparison.candidate.score_at_120_seconds.mean + $comparison.candidate.score_at_120_seconds.p25
        final_score = $comparison.candidate.score.mean
        survival = $comparison.candidate.survival_seconds.mean
        development_summary = $comparison.candidate
    }
}
$parent = $candidates | Sort-Object @{Expression='ranking';Descending=$true}, @{Expression='final_score';Descending=$true}, @{Expression='survival';Descending=$true} | Select-Object -First 1
if (-not $parent) { throw 'No eligible development checkpoint for the final block' }
if ((Get-FileHash -LiteralPath $parent.checkpoint -Algorithm SHA256).Hash.ToLowerInvariant() -ne $parent.checkpoint_sha256) { throw 'Selected parent hash mismatch' }
$selection = [ordered]@{
    recorded_at = [DateTime]::UtcNow.ToString('o')
    purpose = 'Initialization of final aim-only block; deployment candidate will be chosen after the full research budget'
    research_steps_used_before_phase005 = 1500000
    parent = $parent
    all_eligible_candidates = @($candidates)
    steps = 500000
    learning_rate = 0.00002
    entropy = 0.001
    freeze_body = $true
    seed = 8105
    replay_mode = 'greedy'
}
[System.IO.File]::WriteAllText("$experimentRoot/phase005_parent_selection.json", ($selection | ConvertTo-Json -Depth 40), [System.Text.UTF8Encoding]::new($false))
& "$experimentRoot/process_slot.ps1" -RequiredSlots 6 -TrainingLogBase "$experimentRoot/005_train" -PythonArguments @(
    '-m', 'space_z.rl.gpt_training_rotation_continuous.train',
    '--name', '005_aim_precision',
    '--hypothesis', 'Adapt continuous rotation and fire with PPO in full game while preserving learned movement, growth and CNN weights',
    '--steps', '500000', '--stage', 'full', '--reward-profile', 'score_v7',
    '--learning-rate', '0.00002', '--entropy', '0.001', '--freeze-body',
    '--eval-seeds', '43001', '44001', '45001', '46001', '--seed', '8105', '--resume', $parent.checkpoint
)
Set-Content -LiteralPath "$experimentRoot/005_exit_code.txt" '0'
