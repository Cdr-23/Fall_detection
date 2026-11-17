PY = .venv/bin/python

.PHONY: setup data data-gmdcsa pose

setup:
	uv venv --python 3.12 .venv
	uv pip install --python $(PY) -r requirements.txt

data:
	$(PY) src/download.py

data-gmdcsa:
	$(PY) src/download.py --gmdcsa

pose:
	$(PY) src/extract_pose.py
