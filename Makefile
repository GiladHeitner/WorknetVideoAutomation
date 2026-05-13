.PHONY: setup run preview clean

PY := .venv/bin/python
SCRIPT ?= examples/sample_script.txt
VIDEO  ?=

setup:
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	@test -f .env || cp .env.example .env
	@mkdir -p assets inputs out
	@echo "Setup complete. Edit .env to add OPENAI_API_KEY."

preview:
	$(PY) -m src $(SCRIPT) $(VIDEO) --dry-run

run:
	@test -n "$(VIDEO)" || (echo "usage: make run VIDEO=path/to/video.mp4 [SCRIPT=examples/sample_script.txt]" && exit 1)
	$(PY) -m src $(SCRIPT) $(VIDEO)

clean:
	rm -rf out
