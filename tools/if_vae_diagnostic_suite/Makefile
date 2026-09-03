.PHONY: test chore-lint demo

test:
	PYTHONPATH=src python -m unittest discover -s tests -v

chore-lint:
	PYTHONPATH=src python scripts/quality_gate.py

demo:
	PYTHONPATH=src python -m ifvae_diag simulate --out build/demo

