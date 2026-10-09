# Every assignment shares the single virtual environment in ./.venv.
# Scripts use paths relative to their own assignment folder, so each target cd's there first.

VENV := .venv
PY   := $(CURDIR)/$(VENV)/bin/python
RUFF := $(CURDIR)/$(VENV)/bin/ruff

# Headless by default so `make run` never blocks on plt.show(); override for windows:
#   make run-4 MPLBACKEND=TkAgg
export MPLBACKEND ?= Agg

REPORTS := report-1 report-2 report-3 report-4

.PHONY: help venv run run-1 run-2 run-3 run-4 reports $(REPORTS) clean-reports lint format

help:
	@echo "make venv      create ./.venv and install requirements.txt"
	@echo "make run       run every assignment (regenerates figures)"
	@echo "make run-N     run assignment N (1-4)"
	@echo "make reports   build every LaTeX report (assignment-N/report/main.pdf)"
	@echo "make report-N  build the report of assignment N (1-4)"
	@echo "make lint      ruff check"
	@echo "make format    ruff format"

$(PY):
	python3 -m venv $(VENV)

venv: $(PY)
	$(PY) -m pip install -r requirements.txt

run: run-1 run-2 run-3 run-4

run-1: $(PY)
	cd assignment-1 && $(PY) problem1_er_graphs.py

run-2: $(PY)
	cd assignment-2 && $(PY) problem2_wmsr.py

run-3: $(PY)
	cd assignment-3 && $(PY) problem1_formation.py
	cd assignment-3 && $(PY) problem2_tracking.py
	cd assignment-3 && $(PY) problem3_admm.py

run-4: $(PY)
	cd assignment-4 && $(PY) -W ignore::UserWarning problem1.py

# Each report is assignment-N/report/main.tex on top of the shared latex/mas.sty.
reports: $(REPORTS)

$(REPORTS): report-%:
	@cd assignment-$*/report && \
	for pass in 1 2; do \
	    pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null || \
	    { grep -A8 '^!' main.log; exit 1; }; \
	done
	@echo "built assignment-$*/report/main.pdf"

clean-reports:
	rm -f assignment-*/report/main.aux assignment-*/report/main.log assignment-*/report/main.out assignment-*/report/main.toc assignment-*/report/main.pdf

lint: $(PY)
	$(RUFF) check .

format: $(PY)
	$(RUFF) format .
