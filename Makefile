# QDE — quantum vs classical reservoir computing on chaotic time series.
#
# Run `make` (or `make help`) for the target list. Everything is meant to be run
# from the repository root; scripts write into results/ and results_de/ relative
# to the working directory.
#
# Interpreter: local ./.venv if present, else system python3. Override with `make PY=...`.
PY ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)

.DEFAULT_GOAL := help

.PHONY: help
help:  ## show this help
	@echo ""
	@echo "  QDE benchmark  (PY=$(PY))"
	@echo ""
	@grep -hE '^[a-zA-Z0-9_.-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk -F':.*?## ' '{printf "  \033[1m%-16s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

.PHONY: install
install:  ## install the package plus the extras everything in this repo needs
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[orchestration,registry,tools,dev]"

.PHONY: install-pretrained
install-pretrained:  ## additionally install the zero-shot context baseline (downloads a checkpoint at run time)
	$(PY) -m pip install -e ".[pretrained]"

.PHONY: install-pinned
install-pinned:  ## install the exact validated versions (requirements.lock)
	$(PY) -m pip install -r requirements.lock
	$(PY) -m pip install -e . --no-deps

# ---------------------------------------------------------------------------
# Check it works  (~2 minutes, no quantum simulation)
# ---------------------------------------------------------------------------

.PHONY: smoke
smoke:  ## end-to-end sanity run of the pipeline on one classical model
	$(PY) smoke_test.py

.PHONY: test
test:  ## unit and data-quality test suite
	$(PY) -m pytest tests/ -q

.PHONY: verify
verify:  ## smoke + full test suite + fault-injection gates (the 2-minute check)
	$(MAKE) smoke
	$(MAKE) test
	$(PY) tests/test_unit_gates.py

# ---------------------------------------------------------------------------
# Look around  (reads committed results, computes nothing)
# ---------------------------------------------------------------------------

.PHONY: demo
demo:  ## guided walkthrough of the pipeline's five guarantees
	$(PY) scripts/demo.py

.PHONY: browse
browse:  ## interactive terminal browser over every experiment and its results
	$(PY) scripts/qde_ui.py --tui

.PHONY: experiments
experiments:  ## list every experiment family, its script and its result files
	$(PY) scripts/experiments.py

.PHONY: trace
trace:  ## trace a reported number back to the artifact and run that produced it
	@echo "usage: $(PY) scripts/trace.py <value>   e.g.  $(PY) scripts/trace.py 0.02003424617616511"

# ---------------------------------------------------------------------------
# Reproduce the science
# ---------------------------------------------------------------------------

.PHONY: baseline
baseline:  ## baseline leaderboard across all models and systems
	$(PY) experiments/synthesize.py

.PHONY: matched
matched:  ## matched feature-budget comparison, ZZ ablation, climate battery
	$(PY) experiments/experiments_advanced.py --seeds 5

.PHONY: significance
significance:  ## Diebold-Mariano and Model Confidence Set verdicts
	$(PY) experiments/significance_run.py --system henon --seeds 5

.PHONY: scaling
scaling:  ## qubit-scaling family: the 214-test result
	$(PY) experiments/run_scaling_proof.py

.PHONY: shots
shots:  ## finite-shot degradation and concentration analysis
	$(PY) experiments/concentration_run.py

.PHONY: mechanism
mechanism:  ## entanglement ablation, encoding sweep, leaky integration
	$(PY) experiments/run_entanglement.py
	$(PY) experiments/run_leaky.py

.PHONY: rescue
rescue:  ## the five attempts to make a quantum model win
	$(PY) experiments/run_followup_tricks.py
	$(PY) experiments/run_cheb_encoding.py
	$(PY) experiments/run_rfqrc.py
	$(PY) experiments/run_dissipative_qrc.py
	$(PY) experiments/run_mc_tuned_esn.py

.PHONY: engineering
engineering:  ## the six data-engineering experiments (chapter 8)
	$(PY) experiments/run_de_volume.py
	$(PY) experiments/run_de_storage.py
	$(PY) experiments/run_de_parallel.py
	$(PY) experiments/run_de_incremental.py
	$(PY) experiments/run_de_reproducibility.py
	$(PY) experiments/run_de_failure.py

.PHONY: plots
plots:  ## regenerate every figure from the committed result files
	$(PY) -m qdepipe.plots

.PHONY: reproduce
reproduce:  ## everything, end to end (resumable; see docs/REPRODUCTION.md for runtimes)
	$(MAKE) verify
	$(MAKE) baseline
	$(MAKE) matched
	$(MAKE) significance
	$(MAKE) shots
	$(MAKE) scaling
	$(MAKE) mechanism
	$(PY) experiments/run_ic_study.py
	$(PY) experiments/run_lorenz96.py
	$(MAKE) rescue
	$(MAKE) engineering
	$(MAKE) plots

# ---------------------------------------------------------------------------
# Containerised cross-environment check
# ---------------------------------------------------------------------------

IMAGE ?= qde-repro
TAG   ?= local
DOCKER_RUN = docker run --rm -t -v "$(PWD)":/work -e QDE_LOCKFILE=/opt/requirements.lock.txt $(IMAGE):$(TAG)

.PHONY: docker-build
docker-build:  ## build the pinned reproduction image
	docker build -f docker/Dockerfile -t $(IMAGE):$(TAG) .

.PHONY: docker-env
docker-env:  ## print the environment fingerprint inside the image
	$(DOCKER_RUN) python -m qdepipe.envid

.PHONY: docker-lock
docker-lock:  ## export the image's resolved lockfile to docker/requirements.lock.txt
	docker run --rm $(IMAGE):$(TAG) cat /opt/requirements.lock.txt > docker/requirements.lock.txt
	@echo "wrote docker/requirements.lock.txt ($$(wc -l < docker/requirements.lock.txt) packages)"

.PHONY: docker-pin
docker-pin:  ## resolve the base-image digest and print the FROM line to pin
	@docker image inspect python:3.9-slim-bookworm \
	  --format 'FROM python@{{index .RepoDigests 0}}' 2>/dev/null | sed 's|python@python:.*@|python@|' || \
	  echo "pull python:3.9-slim-bookworm first"

.PHONY: docker-test
docker-test:  ## run the test suite inside the image
	$(DOCKER_RUN) python -m pytest tests/ -q

.PHONY: docker-verify
docker-verify:  ## regenerate results in the image and diff against the committed ones
	$(PY) scripts/container_verify.py --image $(IMAGE):$(TAG)

.PHONY: docker-shell
docker-shell:  ## interactive shell inside the image
	docker run --rm -it -v "$(PWD)":/work $(IMAGE):$(TAG) bash

# ---------------------------------------------------------------------------

.PHONY: clean
clean:  ## remove caches and build junk (results/ is committed evidence, untouched)
	rm -rf .pytest_cache qdepipe.egg-info build dist
	find . -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
