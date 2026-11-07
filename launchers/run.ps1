param([Parameter(Mandatory=$true)][ValidateSet('pretraining','couplet','poetry')][string]$Task, [Parameter(Mandatory=$true)][string]$Text, [ValidateSet('rules','unconstrained')][string]$Mode='unconstrained', [switch]$ForwardOnly)
$taskRoot=Split-Path -Parent $PSScriptRoot
$taskPython=Join-Path $taskRoot '.venv_teacher_v4/Scripts/python.exe'
if(-not(Test-Path -LiteralPath $taskPython)){$taskPython='python'}
$taskArguments=@((Join-Path $taskRoot 'scripts/generate.py'),$Task,'--text',$Text,'--mode',$Mode)
if($ForwardOnly){$taskArguments+='--forward-only'}
& $taskPython @taskArguments
