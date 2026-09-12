# Freeze / unfreeze the training process without losing progress.
#
# Killing a run mid-seed throws away that seed - the CSV is only written once
# both arms finish. Suspending stops it burning GPU cycles while keeping the
# process and its state intact, so it picks up exactly where it left off.
#
# The VRAM stays allocated while suspended (a few GB). That's usually fine for
# gaming alongside it; if you need the card completely free, stop the run
# between seeds instead and restart at the next seed.
#
#   .\experiments\pause_run.ps1 -Action pause
#   .\experiments\pause_run.ps1 -Action resume
#   .\experiments\pause_run.ps1 -Action status

param(
    [ValidateSet("pause", "resume", "status")]
    [string]$Action = "status",
    [int]$ProcId = 0     # defaults to every running python process ($Pid is reserved)
)

Add-Type -Namespace Win32 -Name Proc -MemberDefinition @'
[DllImport("ntdll.dll")] public static extern uint NtSuspendProcess(IntPtr h);
[DllImport("ntdll.dll")] public static extern uint NtResumeProcess(IntPtr h);
'@

$targets = if ($ProcId) { Get-Process -Id $ProcId } else { Get-Process python -ErrorAction SilentlyContinue }

if (-not $targets) { "No python process running."; exit }

foreach ($p in $targets) {
    switch ($Action) {
        "pause" {
            [void][Win32.Proc]::NtSuspendProcess($p.Handle)
            "PAUSED  pid $($p.Id)  ($([math]::Round($p.WorkingSet64/1GB,2)) GB RAM held)"
        }
        "resume" {
            [void][Win32.Proc]::NtResumeProcess($p.Handle)
            "RESUMED pid $($p.Id)"
        }
        "status" {
            # a frozen process stops accumulating CPU time; sample it twice
            $t0 = $p.CPU; Start-Sleep -Milliseconds 700; $p.Refresh()
            $state = if (($p.CPU - $t0) -gt 0.05) { "RUNNING" } else { "PAUSED (or idle)" }
            "pid $($p.Id): $state  -  CPU $([math]::Round($p.CPU/60,1)) min, " +
            "$([math]::Round($p.WorkingSet64/1GB,2)) GB RAM"
        }
    }
}
