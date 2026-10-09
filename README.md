# Multi-Agent Systems

Assignments for the Multi-Agent Systems (MAS) course at IIT Hyderabad.

**Author:** Nevil C Prince (EP23BTECH11019)

## Repository layout

```
.
├── assignment-1/   # Erdos-Renyi graphs, linear systems
├── assignment-2/   # consensus dynamics, W-MSR resilient consensus
├── assignment-3/   # formation control, drone tracking (DGD), ADMM task allocation, LASSO
├── assignment-4/   # SIR epidemic on a network, temporal logic
├── latex/mas.sty   # the style shared by every report
├── Makefile        # setup / run / reports / lint, all through the shared .venv
├── pyproject.toml  # ruff configuration
└── requirements.txt
```

Every assignment folder has the same structure:

```
assignment-N/
├── ep23btech11019_assignmentN.pdf   # the submitted report
├── problem_statement.pdf            # the assignment sheet from the course
├── problemX_*.py                    # code, one script per coding problem
├── data/                            # input data, if any (assignment 4)
├── figures/                         # figures written by the scripts, used by the report
└── report/main.tex                  # LaTeX source of the report
```

Problems without code (proofs, temporal logic, ...) live only in the report.

## Setup

All assignments share the single virtual environment in `.venv/`.

```bash
make venv                          # creates .venv and installs requirements.txt
# or by hand:
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

`assignment-3` also needs `ffmpeg` on the `PATH` to save the animation.
Building the reports needs a LaTeX installation (`pdflatex`).

## Running the code

```bash
make run          # every assignment (headless; regenerates figures/)
make run-3        # a single assignment (1-4)
make run-4 MPLBACKEND=TkAgg   # show plots in windows instead of headless
make lint         # ruff check
```

Scripts use paths relative to their assignment folder, so when running one by hand,
`cd` into that folder first (e.g. `cd assignment-3 && ../.venv/bin/python problem2_tracking.py`).
`make` does this for you. Runs are seeded, so results match the reports.

## Building the reports

```bash
make reports      # all four -> assignment-N/report/main.pdf
make report-2     # one of them
make clean-reports
```

The reports are built from the code in the repository: each one lists its scripts straight from
the `.py` files and loads its figures from `figures/`, so they cannot drift from the code.
`report/main.pdf` is a build artefact and is git-ignored; the PDF next to the code is the
submitted version.
