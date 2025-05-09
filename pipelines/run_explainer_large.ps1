# PowerShell script to run the explainer with larger sample
# Usage: .\run_explainer_large.ps1

# Activate virtual environment
Write-Host "🔵 Activating virtual environment..."
& ".\.venv_new\Scripts\activate.ps1"

# Show current mode
Write-Host "🔍 RUNNING ENHANCED EXPLAINER: Larger sample for improved model explanations"

# Create output directories
$outputDir = "output/large_attributions"
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force
}

# Run attribution generation
Write-Host "`n🔵 [1/2] Generating model attributions with larger sample..."
python generate_mock_shap.py --ckpt "output/checkpoints/distilbert_headtail_fold0.pth" --out-dir "$outputDir" --sample 40

# Run SHAP fairness analysis
Write-Host "`n🔵 [2/2] Running SHAP fairness analysis..."
python fairness/shap_report.py --shap-dir "$outputDir" --data data/valid.csv --out-dir "$outputDir/fairness"

# Display summary
Write-Host "`n✅ ENHANCED EXPLAINER RUN COMPLETE!"
Write-Host "Explanation results stored in: $outputDir" 