from metrics import Metrics
import spacy
import os
import json


def test_init():
    metrics = Metrics()
    assert metrics != None

