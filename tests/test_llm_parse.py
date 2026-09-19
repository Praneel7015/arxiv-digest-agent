import json
from llm import parse_json_object


def test_parse_json_object_raw():
    assert parse_json_object('{"a": 1}')["a"] == 1


def test_parse_json_object_fenced():
    text = 'Here you go:\n```json\n{"a": 2, "b": [1]}\n```\n'
    assert parse_json_object(text)["a"] == 2
