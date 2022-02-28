import pytest
from ._vars import Shared, Storage, Var


# TODO: UPDATE TEST

class TestVar:

    def test_get_value(self):

        var = Var(2)
        assert var.val == 2

    def test_set_value(self):

        var = Var(3)
        var.val = 4
        assert var.val == 4

    def test_set_value_with_str(self):

        var = Var("x")
        var.val = "hi"
        assert var.val == "hi"


class TestStorage:

    def test_get_value(self):
        var = 4
        storage = Storage()
        storage["key"] = var

        assert storage["key"].val == var

    def test_storage_contains(self):

        storage = Storage()
        storage["key1"] = 2
        storage["key2"] = 3
        assert "key1" in storage
        assert "key2" in storage

    def test_storage_not_contains(self):

        storage = Storage()
        storage["key1"] = 2
        storage["key2"] = 3
        assert "key3" not in storage

    def test_keys_contains_all_items(self):

        storage = Storage()
        storage["key1"] = 2
        storage["key2"] = 3
        keys = set(storage.keys())
        assert "key1" in keys
        assert "key2" in keys

    def test_vars_contains_all_vars(self):

        storage = Storage()
        storage["key1"] = 2
        storage["key2"] = 3
        keys = set(var.val for var in storage.vars(recursive=True))
        assert 2 in keys
        assert 3 in keys

    def test_items_contains_all_items(self):

        storage = Storage()
        storage["key1"] = 2
        storage["key2"] = 3
        keys = set((key, var.val) for key, var in storage.items())
        assert ("key1", 2) in keys
        assert ("key2", 3) in keys


class TestHierarchicalStorage:

    def test_get_value(self):
        parent = Storage(dict(y=2, k=2))
        child = Storage(dict(k=1), parent)
        assert child["k"].val == 1

    def test_storage_contains(self):

        parent = Storage(data=dict(y=2, k=2))
        storage = Storage(data=dict(k=1), parent=parent)
        assert storage.contains('y', True)

    def test_with_three_levels(self):

        grandparent = Storage(dict(z=2, k=4))
        parent = Storage(dict(y=2, k=2), grandparent)
        child = Storage(dict(k=1), parent)
        assert child.contains("z", recursive=True)

    def test_storage_not_contains(self):

        grandparent = Storage(dict(z=2, k=4))
        parent = Storage(dict(y=2, k=2), grandparent)
        child = Storage(dict(k=1), parent)
        assert not child.contains("x")

    def test_keys_contains_all_items(self):

        grandparent = Storage(dict(z=2, k=4))
        parent = Storage(dict(y=2, k=2), grandparent)
        child = Storage(dict(k=1), parent)

        keys = set(child.keys(True))
        assert "z" in keys
        assert "y" in keys

    def test_vars_contains_all_vars(self):
        grandparent = Storage(dict(z=2, k=4))
        parent = Storage(dict(y=2, k=2), grandparent)
        child = Storage(dict(k=1), parent)
        assert "x" not in child

        keys = set(var.val for var in child.vars(True))
        assert 2 in keys
        assert 1 in keys
