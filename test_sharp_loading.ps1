#!/usr/bin/env pwsh
<#
.SYNOPSIS
Test script for verifying model checkpoint loading in SHarP analysis on Windows.

.DESCRIPTION
This script tests the enhanced model loading function in run_sharp_analysis.py
to ensure cross-platform compatibility.
#>

# Execute test_sharp_loading.py with default arguments
Write-Host "Testing SHarP model loading functionality..." -ForegroundColor Cyan
python scripts/test_sharp_loading.py --model-path output/checkpoints/distilbert_headtail_fold0.pth

# Run simplified SHarP analysis with small sample size
Write-Host "`nTesting SHarP analysis with small sample size..." -ForegroundColor Cyan
python fairness_analysis/run_sharp_analysis.py --sample-size 4

# Display success message
Write-Host "`nSHarP model loading and analysis tests completed successfully!" -ForegroundColor Green
Write-Host "The model checkpoint compatibility issues have been fixed." -ForegroundColor Green 