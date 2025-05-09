.PHONY: train predict figures figures-fast help blend test clean setup full-run dev-run turbo-run explainers-fast explainers-dev eda bias-aucs

help:
	@echo "Available targets:"
	@echo "  setup       - Install dependencies"
	@echo "  train       - Train a model"
	@echo "  train-dry   - Run training in dry-run mode"
	@echo "  predict     - Generate predictions"
	@echo "  figures     - Generate fairness figures"
	@echo "  figures-fast - Generate fairness figures in fast mode"
	@echo "  explainers-fast - Run SHAP + SHARP explainer on dev model"
	@echo "  explainers-dev  - SHAP+SHARP on dev DistilBERT checkpoint (2k rows)"
	@echo "  eda         - Run exploratory data analysis on identity distribution"
	@echo "  bias-aucs   - Calculate bias AUCs (AUC, BPSN, BNSP) for each identity subgroup"
	@echo "  blend       - Blend multiple model predictions"
	@echo "  test        - Run tests"
	@echo "  clean       - Clean output directories"
	@echo "  full-run    - Run a full end-to-end run"
	@echo "  dev-run     - 10% subset, 1 epoch each, ~12 min"
	@echo "  turbo-run   - 5% subset with progress bars, ~5 min"

setup:
	pip install -r requirements.txt

train:
	python -m src.train --model bert_headtail

train-dry:
	python -m src.train --model bert_headtail --dry_run

predict:
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "Error: CHECKPOINT is required. Usage: make predict CHECKPOINT=output/bert_headtail_fold0.pth"; \
		exit 1; \
	fi
	python -m src.predict --checkpoint_path $(CHECKPOINT) --test_file data/test_public_expanded.csv

figures:
	jupytext --to notebook notebooks/04_generate_figures.py -o notebooks/tmp_figs.ipynb
	jupyter nbconvert --execute notebooks/tmp_figs.ipynb --to html --output figs_run.html --ExecutePreprocessor.timeout=600

figures-fast:
eda:  ## run exploratory-data-analysis plots
	python scripts/eda_identity.py --csv data/train.csv

	@if [ -d "results" ]; then \
		jupytext --to notebook notebooks/04_generate_figures.py -o notebooks/tmp_figs.ipynb; \
		jupyter nbconvert --execute notebooks/tmp_figs.ipynb --to html --output figs_run.html --ExecutePreprocessor.timeout=60 --ExecutePreprocessor.kernel_name=python3; \
	else \
		echo "No metrics files found in results/ directory. Run predictions and write_metrics.py first."; \
		exit 1; \
	fi

explainers-fast:
	python scripts/run_explainers_shap.py --sample 2000

explainers-dev:   ## SHAP+SHARP on dev DistilBERT checkpoint (2 k rows)
	python scripts/explainers_distilbert.py \
	       --ckpt output/checkpoints/distilbert_headtail_fold0.pth \
	       --sample 2000

blend:
	@if [ -z "$(GROUND_TRUTH)" ]; then \
		echo "Error: GROUND_TRUTH is required. Usage: make blend GROUND_TRUTH=data/valid.csv"; \
		exit 1; \
	fi
	python -m src.blend_optuna --ground_truth $(GROUND_TRUTH)

test:
	pytest

clean:
	rm -rf output/*/
	rm -rf results/
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	find . -name "*.pyc" -delete

full-run: ## heavyweight run
	@echo "running full pipeline ..."
	$(MAKE) clean
	python -m src.train --model bert_headtail --config configs/bert_headtail.yaml --epochs 2 --save-checkpoint $(ARGS)
	python scripts/pseudo_label.py --base-model output/checkpoints/bert_headtail_fold0.pth --unlabeled-csv data/train.csv --out-csv output/pseudo_bert.csv
	python -m src.train --model lstm_caps --config configs/lstm_caps.yaml --epochs 6 $(ARGS)
	python -m src.train --model gpt2_headtail --config configs/gpt2_headtail.yaml --epochs 2 $(ARGS)
	python -m src.blend_optuna --pred-dir output/preds --ground-truth data/valid.csv --n-trials 200 --out-csv output/preds/blend_ensemble.csv
	python scripts/write_metrics.py --predictions output/preds/blend_ensemble.csv --model-name blend_ensemble
	$(MAKE) figures
	python scripts/run_explainers.py --model-path output/checkpoints/bert_headtail_fold0.pth --n-samples 500

dev-run:    ## 10% subset, 1 epoch each, ~12 min
	python -m src.train --model bert_headtail --config configs/bert_headtail_dev.yaml --fp16 --sample-frac 0.1
	python -m src.train --model lstm_caps     --config configs/lstm_caps_dev.yaml     --sample-frac 0.1
	python -m src.train --model gpt2_headtail --config configs/gpt2_headtail_dev.yaml --fp16 --sample-frac 0.1
	python -m src.blend_optuna --pred-dir output/preds --ground-truth data/valid.csv --n-trials 25 --out-csv output/preds/blend_dev.csv
	python scripts/write_metrics.py --pred output/preds/blend_dev.csv --model-name blend_dev
	make figures-fast

turbo-run:  ## Ultra-fast mode: 5% subset, progress bars, ~5 min
	@echo "🚀 TURBO MODE: Ultra-fast training pipeline with progress bars"
	python -m src.train --model bert_headtail --config configs/bert_headtail_turbo.yaml --fp16 --turbo
	python -m src.train --model lstm_caps     --config configs/lstm_caps_turbo.yaml     --turbo
	python -m src.train --model gpt2_headtail --config configs/gpt2_headtail_turbo.yaml --fp16 --turbo
	python -m src.blend_optuna --pred-dir output/preds --ground-truth data/valid.csv --n-trials 10 --out-csv output/preds/blend_turbo.csv
	python scripts/write_metrics.py --pred output/preds/blend_turbo.csv --model-name blend_turbo
	make figures-fast 

bias-aucs:  ## Calculate bias AUCs (AUC, BPSN AUC, BNSP AUC) for each identity subgroup
	python scripts/bias_auc_metrics.py --validation-csv data/valid.csv --predictions-csv output/preds/simplest_preds.csv --model-name simplest_model
