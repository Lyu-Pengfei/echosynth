PYTHON ?= python
CONFIG ?= configs/experiment/duke_two_step_allview.yaml
MANIFEST ?= manifests/duke_master.csv

.PHONY: install lint test test-fast ingest qc preprocess views splits gen sample train experiment fidelity report clean

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e . --no-deps

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .
	$(PYTHON) -m mypy src/echosynth

test:
	$(PYTHON) -m pytest -m "not reproduction"

test-fast:
	$(PYTHON) -m pytest tests/unit -m "not slow and not reproduction"

ingest:
	$(PYTHON) scripts/00_ingest_dicom.py --help

qc:
	$(PYTHON) scripts/01_run_qc.py --help

preprocess:
	$(PYTHON) scripts/02_preprocess.py --help

views:
	$(PYTHON) scripts/03_classify_views.py --help

splits:
	$(PYTHON) scripts/04_make_splits.py --manifest $(MANIFEST)

gen:
	$(PYTHON) scripts/05_train_diffusion.py --config $(CONFIG)

sample:
	$(PYTHON) scripts/06_sample_synthetic.py --config $(CONFIG)

train:
	$(PYTHON) scripts/07_train_classifier.py --config $(CONFIG)

experiment:
	$(PYTHON) scripts/08_run_experiment.py --config $(CONFIG)

fidelity:
	$(PYTHON) scripts/09_fidelity.py --help

report:
	$(PYTHON) scripts/12_make_report.py --runs results/runs --out-dir results/report

clean:
	find . -type d -name __pycache__ -prune -exec rm -r {} +
	find . -type d -name .pytest_cache -prune -exec rm -r {} +
