.PHONY: ruff
ruff:
	@poetry run ruff check --fix
	@poetry run ruff format

