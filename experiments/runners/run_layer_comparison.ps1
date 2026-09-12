# Read-out comparison: single layer vs fixed blend vs learned blend vs a loss
# applied at several layers at once (multiloss = one RSR term each at L7, L9 and
# L12, equally weighted). The first three change where ONE loss reads from; the
# fourth changes how many losses there are. All four evaluate on the last layer.
#
# Each job is "<arm>:<seed>". Runs them in the order given, so put one seed of
# every arm first - a run that gets interrupted then still leaves a complete
# comparison at whatever n it reached, rather than all seeds of one arm.
#
# Use -DryRun first. Building the argument list is the fiddly part: PowerShell's
# `switch` unwraps a one-element array into a bare string, and splatting a
# string spreads it one character per argument, which python then rejects with
# "unrecognized arguments: - - l a y e r - m i x". The [string[]] casts below
# are what stop that.
#
#   .\experiments\run_layer_comparison.ps1 -Jobs learnedmix:1 -DryRun
#   .\experiments\run_layer_comparison.ps1 -Jobs learnedmix:1,layer12:2

param(
    [Parameter(Mandatory = $true)][string[]]$Jobs,
    [string]$SentencesDir = "artifacts/sentences_strict",
    [int]$Steps = 800,
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
$script = "experiments/rsr_pretraining/bert_rsr_continued_pretraining.py"

# `powershell -File` hands every argument over as a plain string, so a
# comma-separated list arrives as one element rather than several. Re-split so
# the script behaves the same whether it's called that way or from a live shell.
$Jobs = $Jobs | ForEach-Object { $_ -split "," } | Where-Object { $_.Trim() }

function Get-ArmArgs([string]$arm) {
    switch ($arm) {
        "layer12"    { return [string[]]@() }
        "frozenmix"  { return [string[]]@("--layer-mix", "--freeze-layer-mix") }
        "learnedmix" { return [string[]]@("--layer-mix") }
        "multiloss"  { return [string[]]@("--rsr-loss-layers", "7,9,12") }
        "decay"      { return [string[]]@("--lambda-decay", "linear") }
        default      { throw "unknown arm: $arm" }
    }
}

$env:USE_TF = "0"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

foreach ($job in $Jobs) {
    $arm, $seed = $job.Split(":")
    [string[]]$armArgs = Get-ArmArgs $arm
    [string[]]$all = @(
        "-u", $script,
        "--sentences-dir", $SentencesDir,
        "--combine", "weighted",
        "--lambda", "0.7",
        "--steps", "$Steps",
        "--eval-every", "100",
        "--seed", "$seed"
    ) + $armArgs

    Write-Output "===== ARM $arm SEED $seed ====="
    if ($DryRun) {
        Write-Output ("would run: python " + ($all -join " "))
        continue
    }
    $started = Get-Date
    & python @all
    Write-Output ("===== DONE $arm seed $seed, exit $LASTEXITCODE, " +
                  "$([math]::Round(((Get-Date) - $started).TotalMinutes,1)) min =====")
}
