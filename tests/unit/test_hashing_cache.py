"""Tests for hashing cache behavior."""

from code_explainer.utils.hashing import fast_hash_str


def test_fast_hash_str_cache():
    a = "repeated-key"
    h1 = fast_hash_str(a)
    h2 = fast_hash_str(a)
    assert h1 == h2
*** End Patch