# Instructions for Safely Pulling SHarP Analysis Files

These instructions will help you safely pull the new SHarP individual fairness analysis files from GitHub without overwriting your existing artifacts or local modifications.

## Option 1: Using Git Sparse Checkout (Recommended)

This approach allows you to pull only the specific files related to the SHarP analysis:

```powershell
# Navigate to your project directory
cd "C:\Users\mreza\vs\RDS Project"

# Make sure you have the latest repository information
git fetch origin

# Enable sparse checkout
git config core.sparseCheckout true

# Specify which files to check out
echo "explainers/run_individual_fairness.py" > .git/info/sparse-checkout
echo "pipelines/run_individual_fairness.ps1" >> .git/info/sparse-checkout
echo "PULLING_INSTRUCTIONS.md" >> .git/info/sparse-checkout

# Pull the specific files
git checkout origin/main -- explainers/run_individual_fairness.py pipelines/run_individual_fairness.ps1 PULLING_INSTRUCTIONS.md

# Disable sparse checkout when done
git config core.sparseCheckout false
```

## Option 2: Manual File Retrieval

If you'd prefer to avoid Git commands, you can manually download the files:

1. Visit the GitHub repository at https://github.com/Amiri-007/kaggle_jigsaw
2. Navigate to the `explainers` folder and download `run_individual_fairness.py`
3. Navigate to the `pipelines` folder and download `run_individual_fairness.ps1`
4. Place these files in the corresponding folders in your local project

## Option 3: Using Git Stash

This approach is useful if you have local changes that you want to preserve:

```powershell
# Save your local changes
git stash

# Pull only the new files
git checkout origin/main -- explainers/run_individual_fairness.py pipelines/run_individual_fairness.ps1 PULLING_INSTRUCTIONS.md

# Get your local changes back
git stash pop
```

## After Pulling the Files

Once you have the new files, you can run the SHarP analysis:

```powershell
# Run the SHarP analysis
.\pipelines\run_individual_fairness.ps1
```

This script will:
1. Verify that the DistilBERT model checkpoint exists
2. Create the output directory if it doesn't exist
3. Run the SHarP analysis on a sample of the validation data
4. Save the results to `output/explainers/sharp/`

## Expected Output Files

After running the analysis, you should see these files in the output directory:
- `distilbert_shap_values.npz`: Raw SHAP values for the sampled instances
- `sharp_scores_distilbert.csv`: CSV file with divergence scores for each identity group
- `sharp_divergence_distilbert.png`: Visualization of the divergence scores

## Troubleshooting

If you encounter issues:

1. **Missing model checkpoint**: Ensure you've run the turbo pipeline first with `.\pipelines\run_turbo.ps1`
2. **Package errors**: Make sure you have SHAP v0.43.0 installed (`pip install shap==0.43.0`)
3. **Memory issues**: Reduce the sample size with `--sample 500` instead of the default 2000
4. **Visualization errors**: Ensure matplotlib is properly installed (`pip install matplotlib`)

For any other issues, please refer to the GitHub repository's issues page. 