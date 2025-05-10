# PowerShell script to download Kaggle data
$ErrorActionPreference = "Stop"
$RepoDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$DataDir = Join-Path $RepoDir "data"

# Create data directory if it doesn't exist
if (-not (Test-Path $DataDir)) {
    New-Item -Path $DataDir -ItemType Directory -Force | Out-Null
}

# ---------------------------------------------------------------------
# 1. Make sure Kaggle token exists
# ---------------------------------------------------------------------
$KaggleDir = Join-Path $env:USERPROFILE ".kaggle"
$KaggleJson = Join-Path $KaggleDir "kaggle.json"

if (-not (Test-Path $KaggleJson)) {
    Write-Host "First-time setup – paste the **contents** of kaggle.json then press <Ctrl-Z> followed by Enter 👇"
    
    # Create .kaggle directory if it doesn't exist
    if (-not (Test-Path $KaggleDir)) {
        New-Item -Path $KaggleDir -ItemType Directory -Force | Out-Null
    }
    
    # Read the Kaggle JSON from user input
    $jsonContent = @()
    while ($true) {
        $line = Read-Host
        if ($line -eq "") { break }
        $jsonContent += $line
    }
    
    # Write to file
    $jsonContent | Out-File -FilePath $KaggleJson -Encoding utf8
    
    # Set permissions (similar to chmod 600 in Unix)
    $acl = Get-Acl $KaggleJson
    $accessRule = New-Object System.Security.AccessControl.FileSystemAccessRule(
        [System.Security.Principal.WindowsIdentity]::GetCurrent().Name,
        "FullControl",
        "Allow"
    )
    $acl.SetAccessRule($accessRule)
    Set-Acl $KaggleJson $acl
    
    Write-Host "✅ Saved token → $KaggleJson"
}

# ---------------------------------------------------------------------
# 2. Skip if we already have train.csv
# ---------------------------------------------------------------------
$TrainCsv = Join-Path $DataDir "train.csv"
if (Test-Path $TrainCsv) {
    Write-Host "📝 Dataset already present → $DataDir (skipping download)"
    exit 0
}

# ---------------------------------------------------------------------
# 3. Download & unzip (≈ 720 MB)
# ---------------------------------------------------------------------
Write-Host "📥 Downloading Civil Comments ..."
kaggle competitions download -c jigsaw-unintended-bias-in-toxicity-classification -p $DataDir

Write-Host "📦 Extracting ..."
$zipFile = Join-Path $DataDir "jigsaw-unintended-bias-in-toxicity-classification.zip"
Expand-Archive -Path $zipFile -DestinationPath $DataDir -Force

Write-Host "🎉 Dataset ready in $DataDir" 