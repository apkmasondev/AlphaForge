# Fetches the native runtime files that are bundled with the installer but not stored in git:
#   * ONNX Runtime 1.28.3 (CPU build) from the official GitHub release, verified by SHA-256
#   * Microsoft Visual C++ runtime DLLs (app-local deployment) from the local Visual Studio install
#
# Usage (from the repository root):  powershell -ExecutionPolicy Bypass -File tools\fetch_runtime.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$res = Join-Path $root "app\src-tauri\resources"

# --- ONNX Runtime -----------------------------------------------------------------------------
$ortDir = Join-Path $res "onnxruntime"
New-Item -ItemType Directory -Force $ortDir | Out-Null
$expected = @{
    "onnxruntime.dll"                  = "4d2774a5f64e4a230b16b74b167171e92f65b0683c19680c1d04fb284b92a6e2"
    "onnxruntime_providers_shared.dll" = "6582418469facb25b33bc212d8efdc8f7ee57d0c8161e57a59434377780a38b4"
}
$ok = $true
foreach ($name in $expected.Keys) {
    $p = Join-Path $ortDir $name
    if (-not (Test-Path $p) -or (Get-FileHash $p -Algorithm SHA256).Hash.ToLower() -ne $expected[$name]) { $ok = $false }
}
if (-not $ok) {
    $url = "https://github.com/microsoft/onnxruntime/releases/download/v1.28.3/onnxruntime-win-x64-1.28.3.zip"
    $tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("ort-" + [guid]::NewGuid())
    New-Item -ItemType Directory $tmp | Out-Null
    try {
        Write-Host "Downloading $url"
        Invoke-WebRequest $url -OutFile (Join-Path $tmp "ort.zip")
        Expand-Archive (Join-Path $tmp "ort.zip") (Join-Path $tmp "x")
        $lib = Join-Path $tmp "x\onnxruntime-win-x64-1.28.3\lib"
        foreach ($name in $expected.Keys) {
            $src = Join-Path $lib $name
            $hash = (Get-FileHash $src -Algorithm SHA256).Hash.ToLower()
            if ($hash -ne $expected[$name]) { throw "Checksum mismatch for $name" }
            Copy-Item $src $ortDir -Force
        }
        $lic = Join-Path $res "licenses"
        Copy-Item (Join-Path $tmp "x\onnxruntime-win-x64-1.28.3\LICENSE") (Join-Path $lic "onnxruntime-LICENSE.txt") -Force
        Copy-Item (Join-Path $tmp "x\onnxruntime-win-x64-1.28.3\ThirdPartyNotices.txt") (Join-Path $lic "onnxruntime-ThirdPartyNotices.txt") -Force
    }
    finally { Remove-Item -Recurse -Force $tmp }
}
Write-Host "ONNX Runtime: OK"

# --- Visual C++ runtime -------------------------------------------------------------------------
$vcDir = Join-Path $res "vcruntime"
New-Item -ItemType Directory -Force $vcDir | Out-Null
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$crt = Get-ChildItem (Join-Path $vs "VC\Redist\MSVC") -Directory | Sort-Object Name -Descending |
    ForEach-Object { Join-Path $_.FullName "x64\Microsoft.VC143.CRT" } | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $crt) { throw "Visual C++ redistributable folder not found (install the MSVC build tools)." }
foreach ($dll in "msvcp140.dll", "msvcp140_1.dll", "vcruntime140.dll", "vcruntime140_1.dll") {
    Copy-Item (Join-Path $crt $dll) $vcDir -Force
}
Write-Host "VC++ runtime: OK ($crt)"
