download-deps:
	python -m nltk.downloader cmudict averaged_perceptron_tagger

tests-light:
	pytest -v -m "not heavyweight"

tests-heavy:
	pytest -v -m "not lightweight" -n auto

tests:
	pytest -v -n auto
