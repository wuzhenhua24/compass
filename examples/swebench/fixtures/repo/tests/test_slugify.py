from slugify import slugify


def test_lowercases():
    assert slugify("Hello") == "hello"


def test_replaces_spaces():
    assert slugify("hello world") == "hello-world"


def test_strips_edges():
    assert slugify("  hello  ") == "hello"
