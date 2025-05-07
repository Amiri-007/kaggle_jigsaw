.PHONY: train predict figures figures-fast help blend test clean setup

help:
	@echo "Available targets:"
	@echo "  setup       - Install dependencies"
	@echo "  train       - Train a model"
	@echo "  train-dry   - Run training in dry-run mode"
	@echo "  predict     - Generate predictions"
	@echo "  figures     - Generate fairness figures"
	@echo "  figures-fast - Generate fairness figures in fast mode"
	@echo "  blend       - Blend multiple model predictions"
	@echo "  test        - Run tests"
	@echo "  clean       - Clean output directories"

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
	@if [ -d "results" ]; then \
		jupytext --to notebook notebooks/04_generate_figures.py -o notebooks/tmp_figs.ipynb; \
		jupyter nbconvert --execute notebooks/tmp_figs.ipynb --to html --output figs_run.html --ExecutePreprocessor.timeout=60 --ExecutePreprocessor.kernel_name=python3; \
	else \
		echo "No metrics files found in results/ directory. Run predictions and write_metrics.py first."; \
		exit 1; \
	fi

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