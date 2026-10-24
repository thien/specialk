UV ?= uv

.PHONY: download-deps tests-light tests-heavy tests-bench tests lint format

download-deps:
	$(UV) run python -m nltk.downloader cmudict averaged_perceptron_tagger averaged_perceptron_tagger_eng stopwords punkt_tab

lint:
	$(UV) run ruff check .

format:
	$(UV) run ruff format .

tests-light:
	$(UV) run pytest -v -m "not heavyweight and not benchmark"

# Benchmark tests saturate the MPS queue and return bad numbers when several
# xdist workers share the GPU; keep them out of the parallel suite.
tests-heavy:
	$(UV) run pytest -v -m "not lightweight and not benchmark" -n auto

tests-bench:
	$(UV) run pytest -v -m benchmark -n 0

tests:
	$(UV) run pytest -v -m "not benchmark" -n auto
