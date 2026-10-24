UV ?= uv

.PHONY: download-deps tests-light tests-heavy tests

download-deps:
	$(UV) run python -m nltk.downloader cmudict averaged_perceptron_tagger averaged_perceptron_tagger_eng stopwords punkt_tab

tests-light:
	$(UV) run pytest -v -m "not heavyweight"

tests-heavy:
	$(UV) run pytest -v -m "not lightweight" -n auto

tests:
	$(UV) run pytest -v -n auto
