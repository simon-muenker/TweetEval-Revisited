.PHONY: ruff
ruff:
	@poetry run ruff check --fix
	@poetry run ruff format


.PHONY: baseline
baseline:
	@poetry run python experiments/prompt_baseline.py


.PHONY: finetune
finetune:
	@poetry run python experiments/finetuned_head.py