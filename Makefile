PYTHON ?= python
BASE_REF ?= HEAD

.PHONY: chore-lint test

chore-lint:
	$(PYTHON) scripts/quality_gate.py --base-ref $(BASE_REF)

test:
	$(PYTHON) -m pytest tests
