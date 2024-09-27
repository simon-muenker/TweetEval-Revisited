.PHONY: ruff
ruff:
	@poetry run ruff check --fix
	@poetry run ruff format


.PHONY: run
run:
	@poetry run python src/tweeteval_revisited/