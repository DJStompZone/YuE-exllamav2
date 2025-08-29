<#
.SYNOPSIS
Invoke YuE inference with interactively prompted + cached args

.DESCRIPTION
Runs:
  python src/yue/infer.py [flags...]

Features:
- Interactive prompts with JSON caching (Cache = $HOME\.yue_infer_cache.json)
- Preflight: Verify Torch/CUDA/nvidia-smi/ninja/MSVC; OMP duplicate detection
- PS 5.1+ compatible

What's New:
- Added CUDA preflight to preserve my sanity
- PATH sanitization to avoid libiomp5md.dll duplication issues

TODO:
- Remind me to active the damn conda env or activate it automatically (?)
- Check encoding on input txt files since utf-16-le seems to break everything

.PARAMETER UseDefaults
(Switch) Automatically use the default values for any unspecified params.

.PARAMETER Interactive
(Switch) Force interactive prompts even if cache exists.

.PARAMETER ResetCache
(Switch) Delete the cache file before running.

.PARAMETER DryRun
(Switch) Show the final command without executing.

.PARAMETER SkipPreflight
(Switch) Skip CUDA/Torch/OMP preflight checks.

.PARAMETER AllowOmpDuplicate
(Switch) Set KMP_DUPLICATE_LIB_OK=TRUE (last resort, here be dragons).

.PARAMETER Python
(String) Python executable. Default: "$env:CONDA_PREFIX/python.exe".

.PARAMETER ScriptPath
(String) Path to src/yue/infer.py. Default: "src/yue/infer.py".

.PARAMETER Stage1UseExl2
(Bool) Enable --stage1_use_exl2.

.PARAMETER Stage2UseExl2
(Bool) Enable --stage2_use_exl2.

.PARAMETER Stage2CacheSize
(Int) Value for --stage2_cache_size.

.PARAMETER CudaIdx
(Int) Value for --cuda_idx.

.PARAMETER Stage1Model
(String) Value for --stage1_model.

.PARAMETER Stage2Model
(String) Value for --stage2_model.

.PARAMETER RunNSegments
(Int) Value for --run_n_segments.

.PARAMETER Stage2BatchSize
(Int) Value for --stage2_batch_size.

.PARAMETER OutputDir
(String) Value for --output_dir.

.PARAMETER MaxNewTokens
(Int) Value for --max_new_tokens.

.PARAMETER RepetitionPenalty
(Double) Value for --repetition_penalty.

.PARAMETER GenreTxt
(String) Path for --genre_txt.

.PARAMETER LyricsTxt
(String) Path for --lyrics_txt.

.EXAMPLE
.\yue-inference.ps1 -UseDefaults # "Just do it"

.EXAMPLE
.\yue-inference.ps1 -DryRun # "Just don't it"

.EXAMPLE
.\yue-inference.ps1 -AllowOmpDuplicate # May the odds be ever in your favor
#>

[CmdletBinding(SupportsShouldProcess)]
param(
  [string]$Python = "$env:CONDA_PREFIX/python.exe",
  [string]$ScriptPath = "src/yue/infer.py",

  [bool]$Stage1UseExl2 = $true,
  [bool]$Stage2UseExl2 = $true,
  [int]$Stage2CacheSize = 32768,
  [int]$CudaIdx = 0,

  # Model DL
  # hf download m-a-p/YuE-s1-7B-anneal-en-cot --local-dir .\models\YuE-s1
  [string]$Stage1Model = "./models/YuE-s1",

  # Model DL
  # hf download m-a-p/YuE-s2-1B-general --local-dir .\models\YuE-s2
  [string]$Stage2Model = "./models/YuE-s2",
  
  [int]$RunNSegments = 2,
  [int]$Stage2BatchSize = 2,

  [string]$OutputDir = "./output",

  [int]$MaxNewTokens = 3000,
  [double]$RepetitionPenalty = 1.1,

  [string]$GenreTxt = "./genre.txt",
  [string]$LyricsTxt = "./lyrics.txt",
  
  [switch]$UseDefaults,
  [switch]$Interactive,
  [switch]$ResetCache,
  [switch]$DryRun,
  [switch]$SkipPreflight,
  [switch]$AllowOmpDuplicate
)

$ErrorActionPreference = "Stop"
$CachePath = Join-Path $HOME ".yue_infer_cache.json"

function Write-Info($msg) { Write-Host "[INFO] $msg" }
function Write-Warn($msg) { Write-Warning $msg }
function Write-ErrLine($msg) { Write-Host "[ERROR] $msg" -ForegroundColor Red }

function Coalesce([object]$A, [object]$B) { if ($null -ne $A -and $A -ne "") { $A } else { $B } }

function Read-Default([string]$Prompt, [string]$Default) {
  $suffix = if ($Default -ne "") { " [$Default]" } else { "" }
  $input = Read-Host "$Prompt$suffix"
  if ([string]::IsNullOrWhiteSpace($input)) { return $Default }
  $input
}
function Read-DefaultInt([string]$Prompt, [int]$Default) {
  while ($true) { $v = Read-Default $Prompt ([string]$Default); $out=0; if ([int]::TryParse($v,[ref]$out)) { return $out }; Write-Warn "Enter an integer." }
}
function Read-DefaultDouble([string]$Prompt, [double]$Default) {
  while ($true) { $v = Read-Default $Prompt ([string]$Default); $out=0.0; if ([double]::TryParse($v,[ref]$out)) { return $out }; Write-Warn "Enter a number." }
}
function Read-DefaultBool([string]$Prompt, [bool]$Default) {
  $d = if ($Default) { "Y" } else { "N" }
  while ($true) {
    $v = Read-Default "$Prompt (Y/N)" $d
    switch -Regex ($v) { '^(?i:y|yes)$' { return $true } '^(?i:n|no)$' { return $false } default { Write-Warn "Enter Y or N." } }
  }
}

function Load-Cache { if (Test-Path -LiteralPath $CachePath) { try { return Get-Content -LiteralPath $CachePath -Raw | ConvertFrom-Json } catch { Write-Warn "Failed to read cache; ignoring: $($_.Exception.Message)" } } @{} }
function Save-Cache($obj) { try { $obj | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $CachePath -Encoding UTF8; Write-Info "Saved cache to $CachePath" } catch { Write-Warn "Failed to write cache: $($_.Exception.Message)" } }

function Resolve-PathStrict([string]$PathLike, [switch]$AllowMissing, [switch]$CreateDir) {
  $expanded = [Environment]::ExpandEnvironmentVariables($PathLike)
  if ($expanded -match '^[~]') { $expanded = $expanded -replace '^[~]', $HOME }
  $full = Resolve-Path -LiteralPath $expanded -ErrorAction SilentlyContinue
  if ($full) { return $full.Path }
  $candidate = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $expanded))
  if ($CreateDir) { New-Item -ItemType Directory -Force -Path $candidate | Out-Null; return $candidate }
  if ($AllowMissing) { return $candidate }
  throw "Path not found: $candidate"
}

function Test-Command([string]$name) { [bool](Get-Command $name -ErrorAction SilentlyContinue) }

function Test-TorchProbe([string]$python) {
  $code = @'
import os, json, sys
try:
    import torch
    out = {
        "torch_version": getattr(torch, "__version__", None),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cuda_is_available": torch.cuda.is_available(),
        "cuda_home": os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH"),
    }
    print(json.dumps(out))
except Exception as e:
    print(json.dumps({"error": str(e)}))
    sys.exit(11)
'@
  $all = & $python -c $code 2>&1
  $exit = $LASTEXITCODE
  [pscustomobject]@{ exit=$exit; output=$all }
}

function Test-Ninja([string]$python) {
  & $python -c 'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("ninja") else 1)' *> $null
  ($LASTEXITCODE -eq 0)
}

function Detect-OmpDupFromOutput([string]$s) {
  if (-not $s) { return $false }
  return ($s -match 'Initializing libiomp5md\.dll, but found libiomp5md\.dll already initialized')
}

function Get-CondaContext() {
  $envPrefix = $env:CONDA_PREFIX
  if (-not $envPrefix) { return $null }
  $envsDir   = Split-Path $envPrefix      # ...\envs
  $condaRoot = Split-Path $envsDir        # ...\conda
  [pscustomobject]@{
    EnvPrefix = $envPrefix
    CondaRoot = $condaRoot
  }
}

function Sanitize-Path-ForChild([string]$currentPath, [string]$envPrefix, [string]$condaRoot) {
  # Drop: entries under the "base" conda root
  $parts = $currentPath -split ';'
  $filtered = foreach ($p in $parts) {
    if ($p -eq '') { continue }
    $norm = $p.TrimEnd('\')
    if ($norm -like "$envPrefix*") { $p; continue }
    if ($norm -like "$condaRoot*") { continue }
    $p
  }
  ($filtered -join ';')
}

function Preflight([string]$python, [switch]$skipCudaChecks=$false) {
  Write-Info "Preflight: Torch/CUDA/ninja/MSVC/OMP"

  $probe = Test-TorchProbe $python
  $ompDup = Detect-OmpDupFromOutput $probe.output

  if ($ompDup) {
    Write-ErrLine "Detected OpenMP duplicate runtime (libiomp5md.dll). Will attempt PATH sanitization for child process."
  }

  if (-not $skipCudaChecks) {
    if ($probe.exit -eq 0) {
      try {
        $j = $probe.output | ConvertFrom-Json
        Write-Info ("torch={0}, torch.cuda={1}, is_available={2}, CUDA_HOME={3}" -f $j.torch_version, $j.torch_cuda_version, $j.cuda_is_available, $j.cuda_home)
        if (-not $j.torch_cuda_version) { Write-ErrLine "Your PyTorch build is CPU-only. Install CUDA-enabled torch." ; return $false }
        if (-not $j.cuda_is_available)  { Write-ErrLine "torch.cuda.is_available() is False. Driver/CUDA toolkit missing or mismatched." ; return $false }
        if (-not $j.cuda_home)          { Write-ErrLine "CUDA_HOME not set. Set it to your CUDA install root (e.g., C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x)." ; return $false }
      } catch {
        Write-Warn "Failed to parse torch probe JSON; continuing..."
      }
    } else {
      if (-not $ompDup) {
        Write-ErrLine "Python/torch probe failed: $($probe.output)"
        return $false
      }
    }

    if (-not (Test-Command "nvidia-smi")) {
      Write-ErrLine "nvidia-smi not found. Install NVIDIA driver or add it to PATH."
      return $false
    }
  }

  if (-not (Test-Ninja $python)) {
    Write-ErrLine "Python package 'ninja' not found. Install with: pip install ninja"
    return $false
  }

  $hasCL = (Test-Command "cl.exe")
  if (-not $hasCL) {
    Write-ErrLine "MSVC Build Tools not detected (cl.exe). Install Visual Studio Build Tools: Desktop development with C++, MSVC v143, Windows 10/11 SDK."
    return $false
  }

  return $true
}

if ($ResetCache -and (Test-Path -LiteralPath $CachePath)) {
  Remove-Item -LiteralPath $CachePath -Force
  Write-Info "Cache cleared: $CachePath"
}

$cache = Load-Cache

function Get-Value {
  param(
    [string]$Name,
    [object]$Default,
    [ValidateSet('string','int','double','bool')] [string]$Type = 'string',
    [switch]$PromptUser
  )
  if ($PSBoundParameters.ContainsKey($Name)) { return $PSBoundParameters[$Name] }
  if ($UseDefaults) {
      return $Default
  }
  $cached = $null
  if ($cache -and $cache.PSObject.Properties.Name -contains $Name) { $cached = $cache.$Name }
  if ($Interactive -or $PromptUser) {
    switch ($Type) {
      'bool'   { return Read-DefaultBool   $Name (Coalesce $cached $Default) }
      'int'    { return Read-DefaultInt    $Name (Coalesce $cached $Default) }
      'double' { return Read-DefaultDouble $Name (Coalesce $cached $Default) }
      default  { return Read-Default       $Name ([string](Coalesce $cached $Default)) }
    }
  }
  if ($null -ne $cached) { return $cached }
  $Default
}

# --- Param condensation ---
$vals = [ordered]@{}
$vals.Python            = Get-Value -Name "Python"            -Default $Python
$vals.ScriptPath        = Get-Value -Name "ScriptPath"        -Default $ScriptPath -PromptUser

$vals.Stage1UseExl2     = Get-Value -Name "Stage1UseExl2"     -Default $Stage1UseExl2 -Type bool -PromptUser
$vals.Stage2UseExl2     = Get-Value -Name "Stage2UseExl2"     -Default $Stage2UseExl2 -Type bool -PromptUser
$vals.Stage2CacheSize   = Get-Value -Name "Stage2CacheSize"   -Default $Stage2CacheSize -Type int -PromptUser
$vals.CudaIdx           = Get-Value -Name "CudaIdx"           -Default $CudaIdx -Type int -PromptUser

$vals.Stage1Model       = Get-Value -Name "Stage1Model"       -Default $Stage1Model -PromptUser
$vals.Stage2Model       = Get-Value -Name "Stage2Model"       -Default $Stage2Model -PromptUser

$vals.RunNSegments      = Get-Value -Name "RunNSegments"      -Default $RunNSegments -Type int -PromptUser
$vals.Stage2BatchSize   = Get-Value -Name "Stage2BatchSize"   -Default $Stage2BatchSize -Type int -PromptUser

$vals.OutputDir         = Get-Value -Name "OutputDir"         -Default $OutputDir -PromptUser
$vals.MaxNewTokens      = Get-Value -Name "MaxNewTokens"      -Default $MaxNewTokens -Type int -PromptUser
$vals.RepetitionPenalty = Get-Value -Name "RepetitionPenalty" -Default $RepetitionPenalty -Type double -PromptUser

$vals.GenreTxt          = Get-Value -Name "GenreTxt"          -Default $GenreTxt -PromptUser
$vals.LyricsTxt         = Get-Value -Name "LyricsTxt"         -Default $LyricsTxt -PromptUser

# --- Path resolution ---
$pythonPath   = $vals.Python
$scriptAbs    = Resolve-PathStrict -PathLike $vals.ScriptPath
$genreAbs     = Resolve-PathStrict -PathLike $vals.GenreTxt
$lyricsAbs    = Resolve-PathStrict -PathLike $vals.LyricsTxt
$outputAbs    = Resolve-PathStrict -PathLike $vals.OutputDir -CreateDir

# --- Preflight ---
if (-not $SkipPreflight) {
  $pf = Preflight -python $pythonPath
  if (-not $pf) {
    Write-ErrLine "Environment not ready. Fix the issues above or pass -SkipPreflight to bypass."
    exit 1
  }
}

# --- Build args ---
$argv = @()
$argv += $scriptAbs
if ($vals.Stage1UseExl2)     { $argv += "--stage1_use_exl2" }
if ($vals.Stage2UseExl2)     { $argv += "--stage2_use_exl2" }
$argv += @(
  "--stage2_cache_size", [string]$vals.Stage2CacheSize,
  "--cuda_idx",          [string]$vals.CudaIdx,
  "--stage1_model",      $vals.Stage1Model,
  "--stage2_model",      $vals.Stage2Model,
  "--run_n_segments",    [string]$vals.RunNSegments,
  "--stage2_batch_size", [string]$vals.Stage2BatchSize,
  "--output_dir",        $outputAbs,
  "--max_new_tokens",    [string]$vals.MaxNewTokens,
  "--repetition_penalty",[string]$vals.RepetitionPenalty,
  "--genre_txt",         $genreAbs,
  "--lyrics_txt",        $lyricsAbs
)

function QuoteIfNeeded([string]$s) {
  if ($s -match '\s' -or $s -match '"') { return '"' + ($s -replace '"','`"') + '"' }
  $s
}

Write-Info "Python: $pythonPath"
Write-Info "Script: $scriptAbs"
Write-Info "Output: $outputAbs"
Write-Host ""
Write-Host "Final command:" -ForegroundColor Cyan
$cmdPreview = @((QuoteIfNeeded $pythonPath)) + ($argv | ForEach-Object { QuoteIfNeeded $_ })
Write-Host ($cmdPreview -join ' ')
Write-Host ""

if ($DryRun) { Write-Info "DryRun: not executing."; return }
$oldPath = $env:Path
$oldKmp  = $env:KMP_DUPLICATE_LIB_OK
$conda = Get-CondaContext

try {
  if ($conda) {
    $sanitized = Sanitize-Path-ForChild -currentPath $env:Path -envPrefix $conda.EnvPrefix -condaRoot $conda.CondaRoot
    if ($sanitized -ne $env:Path) {
      Write-Info "Sanitizing PATH for child process to avoid base-Conda OpenMP duplication."
      $env:Path = $sanitized
    }
  }

  if ($AllowOmpDuplicate) {
    Write-Warn "AllowOmpDuplicate set: enabling KMP_DUPLICATE_LIB_OK=TRUE for child only."
    $env:KMP_DUPLICATE_LIB_OK = "TRUE"
  }

  if ($PSCmdlet.ShouldProcess("$pythonPath", "Run YuE inference")) {
    & $pythonPath @argv
    $exit = $LASTEXITCODE
    if ($exit -ne 0) { throw "YuE inference failed with exit code $exit." }
    Save-Cache $vals
    Write-Info "Done."
  }
}
finally {
  $env:Path = $oldPath
  $env:KMP_DUPLICATE_LIB_OK = $oldKmp
}
