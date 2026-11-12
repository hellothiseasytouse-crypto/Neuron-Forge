.PHONY: install test sample train generate serve
install:
	python -m pip install -e ".[dev]"
test:
	pytest -q
sample:
	python scripts/prepare_data.py
train:
	neuronforge train --config configs/quick.yaml
generate:
	neuronforge generate --checkpoint checkpoints/latest.pt --prompt "Hello"
serve:
	python -m pip install -e ".[web]" && neuronforge serve --checkpoint checkpoints/latest.pt
