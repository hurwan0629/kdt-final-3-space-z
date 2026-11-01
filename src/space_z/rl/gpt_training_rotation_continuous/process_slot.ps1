param(
    [Parameter(Mandatory = $true)][string[]]$PythonArguments,
    [ValidateSet(2, 6, 8)][int]$RequiredSlots = 2,
    [string]$TrainingLogBase
)
$ErrorActionPreference = 'Stop'
$jobMutex = [System.Threading.Mutex]::new($false, 'Local\SpaceZRotationOptionalJob')
$ownsMutex = $false
try {
    try { $ownsMutex = $jobMutex.WaitOne() }
    catch [System.Threading.AbandonedMutexException] { $ownsMutex = $true }
    do {
        $pythonProcesses = @(Get-CimInstance Win32_Process -Filter "name = 'python.exe'")
        $owned = @($pythonProcesses | Where-Object { $_.CommandLine -like '*space_z.rl.gpt_training_rotation_continuous*' })
        $ownedIds = @($owned.ProcessId)
        do {
            $children = @($pythonProcesses | Where-Object { $_.ParentProcessId -in $ownedIds -and $_.ProcessId -notin $ownedIds })
            $owned += $children
            $ownedIds = @($owned.ProcessId)
        } while ($children.Count -gt 0)
        if ($owned.Count + $RequiredSlots -gt 8) { Start-Sleep -Seconds 3 }
    } while ($owned.Count + $RequiredSlots -gt 8)
    $env:PYTHONPATH = "$PWD/src"
    $env:PYTHONDONTWRITEBYTECODE = '1'
    $env:PYTHONUTF8 = '1'
    $env:PYGAME_HIDE_SUPPORT_PROMPT = '1'
    $env:MPLCONFIGDIR = "$PSScriptRoot/.cache/matplotlib"
    [ordered]@{utc=[DateTime]::UtcNow.ToString('o'); state='started'; existing_python=$owned.Count; reserved_slots=$RequiredSlots; arguments=$PythonArguments} | ConvertTo-Json -Compress | Add-Content -LiteralPath "$PSScriptRoot/process_slot.jsonl" -Encoding utf8
    if ($RequiredSlots -eq 2) {
        $savedErrorPreference = $ErrorActionPreference
        try {
            # PNG 등의 stderr 경고는 기록하고 실제 프로세스 종료 코드로 실패를 판정한다.
            $ErrorActionPreference = 'Continue'
            & src/space_z/rl/gpt_training/.venv/Scripts/python.exe -B @PythonArguments 2>&1 | ForEach-Object { $_.ToString() }
            $jobExit = $LASTEXITCODE
        }
        finally { $ErrorActionPreference = $savedErrorPreference }
    }
    else {
        if (-not $TrainingLogBase -or ($PythonArguments | Where-Object { $_.Contains('"') -or $_.EndsWith('\') })) {
            throw 'Training needs a log base and arguments without embedded quotes or trailing backslashes'
        }
        $quotedArguments = @('-B') + $PythonArguments | ForEach-Object { '"' + $_ + '"' }
        $trainingProcess = Start-Process -FilePath (Resolve-Path 'src/space_z/rl/gpt_training/.venv/Scripts/python.exe').Path -WorkingDirectory "$PWD" -ArgumentList $quotedArguments -WindowStyle Hidden -PassThru -RedirectStandardOutput "$TrainingLogBase.log" -RedirectStandardError "$TrainingLogBase.errors.log"
        # Windows는 종료 후 핸들을 처음 열면 ExitCode가 사라질 수 있어 미리 보관한다.
        $trainingHandle = $trainingProcess.Handle
        do {
            $trainingProcess.Refresh()
            $pythonProcesses = @(Get-CimInstance Win32_Process -Filter "name = 'python.exe'")
            $trainingIds = @($trainingProcess.Id)
            do {
                $children = @($pythonProcesses | Where-Object { $_.ParentProcessId -in $trainingIds -and $_.ProcessId -notin $trainingIds })
                $trainingIds += @($children.ProcessId)
            } while ($children.Count -gt 0)
            if ($trainingIds.Count -lt $RequiredSlots -and -not $trainingProcess.HasExited) { Start-Sleep -Seconds 1 }
        } while ($trainingIds.Count -lt $RequiredSlots -and -not $trainingProcess.HasExited)
        # 학습 프로세스가 모두 생성된 뒤에만 보조 작업의 잠금을 풀어 시작 경쟁을 막는다.
        $jobMutex.ReleaseMutex()
        $ownsMutex = $false
        $trainingProcess.WaitForExit()
        $trainingProcess.Refresh()
        $jobExit = $trainingProcess.ExitCode
        if ($null -eq $jobExit) { throw 'The training process exited but Windows did not return its exit code' }
    }
    [ordered]@{utc=[DateTime]::UtcNow.ToString('o'); state='finished'; exit_code=$jobExit} | ConvertTo-Json -Compress | Add-Content -LiteralPath "$PSScriptRoot/process_slot.jsonl" -Encoding utf8
    if ($jobExit -ne 0) { throw "Python job failed with exit code $jobExit" }
}
finally {
    if ($ownsMutex) { $jobMutex.ReleaseMutex() }
    $jobMutex.Dispose()
}
