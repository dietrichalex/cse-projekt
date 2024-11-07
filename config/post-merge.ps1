$scriptDir = $PSScriptRoot
$packagePath = Join-Path -Path $scriptDir -ChildPath "requirements.txt"

pip install -r $packagePath