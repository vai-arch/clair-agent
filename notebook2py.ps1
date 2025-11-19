param(
    [Parameter(Mandatory=$true)]
    [string]$NotebookName
)

$notebookFolder = "notebooks"
$codeFolder = "code"

# Compute paths
$inputPath = Join-Path $notebookFolder $NotebookName

if (-Not (Test-Path $inputPath)) {
    Write-Host "ERROR: Notebook '$NotebookName' not found in '$notebookFolder'"
    exit 1
}

Write-Host "Converting notebook: $inputPath"

# Ensure PowerShell outputs UTF-8
$OutputEncoding = [System.Text.UTF8Encoding]::UTF8

# Compute output path first
$outputPath = Join-Path $codeFolder ([System.IO.Path]::GetFileNameWithoutExtension($inputPath) + ".py")

# Convert notebook → py preserving emojis
python -m jupytext --to py "$inputPath" --output "$outputPath"

# ==== Locate generated .py file ====

# Remove .ipynb → add .py
$expectedPyName = [System.IO.Path]::GetFileNameWithoutExtension($NotebookName) + ".py"
$generatedPyPath = Join-Path $codeFolder $expectedPyName

Write-Host "Looking for generated file: $generatedPyPath"

if (-Not (Test-Path $generatedPyPath)) {
    Write-Host "ERROR: nbconvert did not generate the expected file."
    Write-Host "Listing files in notebooks/:"
    Get-ChildItem $notebookFolder
    exit 1
}

Write-Host "Generated file found."
