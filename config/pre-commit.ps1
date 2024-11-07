$scriptDir = $PSScriptRoot
$packagePath = Join-Path -Path $scriptDir -ChildPath "requirements.txt"

pip freeze > $packagePath

git add $packagePath

exit