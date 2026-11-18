PY = .venv/bin/python

.PHONY: all models compare impact setup data data-gmdcsa pose processed naive test gmdcsa le2i
ALGOS = naive rules logreg cart boosting nn

# Everything from the poses already in the repo (no download): train the 6 models (5-fold CV),
# score them, compare, confirmation, external tests, impact. ~6 min on a laptop (results identical).
all: models compare impact

models:
	for a in $(ALGOS); do $(PY) src/run_$$a.py && $(PY) src/evaluate.py results/$${a}_oof.parquet || exit 1; done

compare:
	$(PY) src/compare.py
	$(PY) src/confirm.py
	$(PY) src/confirm_long.py

impact:
	$(PY) src/impact.py

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

le2i:
	$(PY) src/le2i_convert.py
	$(PY) src/external.py le2i
