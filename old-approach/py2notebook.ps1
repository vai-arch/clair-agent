param(
    [Parameter(Mandatory=$true)]
    [string]$PythonFileName
)

$codeFolder = "code"
$notebookFolder = "notebooks"

# Full path to input .py file
$inputPath = Join-Path $codeFolder $PythonFileName

if (-Not (Test-Path $inputPath)) {
    Write-Host "ERROR: Python file '$PythonFileName' not found in '$codeFolder'"
    exit 1
}

Write-Host "Converting Python file: $inputPath"

# Build output notebook name
$notebookName = [System.IO.Path]::GetFileNameWithoutExtension($PythonFileName) + ".ipynb"
$outputPath = Join-Path $notebookFolder $notebookName

# Ensure output folder exists
if (-Not (Test-Path $notebookFolder)) {
    New-Item -ItemType Directory -Path $notebookFolder | Out-Null
}

# Convert using jupytext
jupytext --to ipynb "$inputPath" --output "$outputPath"

if (-Not (Test-Path $outputPath)) {
    Write-Host "ERROR: Conversion failed. Notebook not generated."
    exit 1
}

Write-Host "Notebook created at: $outputPath"
Write-Host "Done."
