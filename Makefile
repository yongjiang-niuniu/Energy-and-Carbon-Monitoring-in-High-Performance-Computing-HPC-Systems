PYTHON ?= python3
.PHONY: test build-simulator
test:
	$(PYTHON) -m pytest -q
build-simulator:
	docker build -t slurm-sim-scheduler-fix:latest -f simulator/slurm_simulator_source_legacy_desktop/Dockerfile.local-sim simulator/slurm_simulator_source_legacy_desktop
