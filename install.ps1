param(
    [string]$RuntimeDir = 'D:\Models\Drama3Runtime',
    [string]$CheckpointDir = 'D:\Models\drama-3',
    [switch]$SkipDownload
)
$ErrorActionPreference = 'Stop'
$repoDir = $PSScriptRoot
$RuntimeDir = [IO.Path]::GetFullPath($RuntimeDir)
$CheckpointDir = [IO.Path]::GetFullPath($CheckpointDir)
$upstreamRevision = '1ff3c3ed2f80d3106a83909dd25b83129c3d816f'
$checkpointRevision = 'dd64a85cd82e9521f7202610ede97b367fc83946'

function Run-Checked {
    param([string]$Program, [Parameter(ValueFromRemainingArguments=$true)][string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program failed (exit $LASTEXITCODE)." }
}

Get-Command uv, git -ErrorAction Stop | Out-Null
New-Item -ItemType Directory -Force $RuntimeDir, $CheckpointDir | Out-Null
$env:UV_CACHE_DIR = Join-Path $RuntimeDir 'uv-cache'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $RuntimeDir 'python'
$env:HF_HOME = Join-Path $CheckpointDir '.download-cache'
$env:HF_XET_CACHE = Join-Path $env:HF_HOME 'xet'
$env:HF_XET_CHUNK_CACHE_SIZE_BYTES = '0'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:UV_HTTP_TIMEOUT = '120'
$upstream = Join-Path $RuntimeDir 'fish-speech'
if (-not (Test-Path -LiteralPath (Join-Path $upstream '.git'))) {
    Run-Checked 'git' @('clone', '--filter=blob:none', '--no-checkout', 'https://github.com/fishaudio/fish-speech.git', $upstream)
    Run-Checked 'git' @('-C', $upstream, 'sparse-checkout', 'set', 'fish_speech', 'tools')
    Run-Checked 'git' @('-C', $upstream, 'checkout', $upstreamRevision)
}
$currentRevision = (& git -C $upstream rev-parse HEAD).Trim()
if ($currentRevision -ne $upstreamRevision) { throw 'Existing Fish Speech checkout has a different revision; use a fresh RuntimeDir.' }
$patch = Join-Path $repoDir 'runtime\streaming.patch'
& git -C $upstream apply --reverse --check $patch 2>$null
if ($LASTEXITCODE -ne 0) {
    Run-Checked 'git' @('-C', $upstream, 'apply', '--check', $patch)
    Run-Checked 'git' @('-C', $upstream, 'apply', $patch)
}
$venv = Join-Path $RuntimeDir 'drama3-env'
$python = Join-Path $venv 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    Run-Checked 'uv' @('venv', '--python', '3.12', $venv)
}
Run-Checked 'uv' @('pip','install','--python',$python,'torch==2.8.0','torchaudio==2.8.0','--index-url','https://download.pytorch.org/whl/cu129')
Run-Checked 'uv' @('pip','install','--python',$python,'--override',(Join-Path $repoDir 'runtime\overrides.txt'),'-r',(Join-Path $repoDir 'runtime\requirements.txt'))
Copy-Item -LiteralPath (Join-Path $repoDir 'runtime\run_drama3_full.py'), (Join-Path $repoDir 'runtime\fp8_offload.py'), (Join-Path $repoDir 'runtime\take_io.py') -Destination $RuntimeDir -Force
@{checkpoint_dir=$CheckpointDir} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $RuntimeDir 'model-location.json') -Encoding utf8
$revisionFile = Join-Path $CheckpointDir 'revision.txt'
if (Test-Path -LiteralPath $revisionFile) {
    if ((Get-Content -Raw -LiteralPath $revisionFile).Trim() -ne $checkpointRevision) { throw 'Checkpoint revision.txt differs from the tested checkpoint. Choose another CheckpointDir.' }
}
if (-not $SkipDownload) {
    Run-Checked $python @((Join-Path $repoDir 'runtime\download.py'), '--checkpoint-dir', $CheckpointDir)
} else {
    $index = Join-Path $CheckpointDir 'model.safetensors.index.json'
    $files = @('codec.pth','config.json','tokenizer.json','model.safetensors.index.json')
    if (Test-Path -LiteralPath $index) {
        $map = (Get-Content -Raw -LiteralPath $index | ConvertFrom-Json).weight_map
        $files += @($map.PSObject.Properties.Value | Sort-Object -Unique)
    }
    foreach ($file in $files) {
        if (-not (Test-Path -LiteralPath (Join-Path $CheckpointDir $file))) { throw "Missing checkpoint file: $file" }
    }
    if (-not (Test-Path -LiteralPath $revisionFile)) { throw 'SkipDownload requires revision.txt from the pinned checkpoint download.' }
}
@{runtime_folder=$RuntimeDir} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $repoDir 'runtime.json') -Encoding utf8
Write-Host 'Ready. Restart ComfyUI and open an example from the ComfyUI-Drama3-FP8 template category.'
Write-Host 'For demo workflows, copy examples/Drama3_demo_* into ComfyUI/input.'
