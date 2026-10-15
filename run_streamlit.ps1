$python = Join-Path $env:LOCALAPPDATA 'Python\bin\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    Write-Error "Python was not found at $python"
    exit 1
}

Set-Location -LiteralPath $PSScriptRoot
& $python -m streamlit run streamlit_app.py
