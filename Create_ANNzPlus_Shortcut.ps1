$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$launcher = Join-Path $projectRoot "Run_ANNzPlus.bat"

if (-not (Test-Path -LiteralPath $launcher)) {
    throw "Run_ANNzPlus.bat was not found in $projectRoot"
}

$desktop = [Environment]::GetFolderPath("Desktop")
$startMenu = [Environment]::GetFolderPath("Programs")
$menuFolder = Join-Path $startMenu "ANNzPlus Python"
New-Item -ItemType Directory -Path $menuFolder -Force | Out-Null

$wsh = New-Object -ComObject WScript.Shell

function New-ANNzShortcut([string]$shortcutPath) {
    $shortcut = $wsh.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = Join-Path $env:SystemRoot "System32\cmd.exe"
    $shortcut.Arguments = '/c "' + $launcher + '"'
    $shortcut.WorkingDirectory = $projectRoot
    $shortcut.Description = "Launch the ANNzPlus Python galaxy redshift analysis app"
    $pythonIcon = Join-Path $projectRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $pythonIcon) {
        $shortcut.IconLocation = "$pythonIcon,0"
    }
    $shortcut.Save()
}

New-ANNzShortcut (Join-Path $desktop "ANNzPlus Python.lnk")
New-ANNzShortcut (Join-Path $menuFolder "ANNzPlus Python.lnk")

Write-Host "Created desktop and Start Menu shortcuts."
