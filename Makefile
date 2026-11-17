PY = .venv/bin/python

.PHONY: setup data data-gmdcsa pose processed naive test gmdcsa

setup:
	uv venv --python 3.12 .venv
	uv pip install --python $(PY) -r requirements.txt

data:
	$(PY) src/download.py

data-gmdcsa:
	$(PY) src/download.py --gmdcsa

pose:
	$(PY) src/extract_pose.py

processed:
	$(PY) src/preprocess.py
	$(PY) src/features.py

naive:
	$(PY) src/run_naive.py
	$(PY) src/evaluate.py results/naive_oof.parquet

test:
	$(PY) tests/test_metrics.py

gmdcsa:
	cd data/raw/gmdcsa24 && unzip -qn GMDCSA24-v2.1.zip
	$(PY) src/external.py gmdcsa24
