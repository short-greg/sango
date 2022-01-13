import pytest
from .vars import Shared, Storage, Var


# TODO: UPDATE TEST

class TestVar:

    def test_get_value(self):

        var = Var(2)
        assert var.value == 2

    def test_set_value(self):

        var = Var(3)
        var.value = 4
        assert var.value == 4

    def test_set_value_with_str(self):

        var = Var("x")
        var.value = "hi"
        assert var.value == "hi"


class TestStorage:

    def test_get_value(self):
        var = 4
        storage = Storage()
        storage["key"] = var

        assert storage["key"].value == var

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
        keys = set(var.value for var in storage.vars())
        assert 2 in keys
        assert 3 in keys

    def test_items_contains_all_items(self):

        storage = Storage()
        storage["key1"] = 2
        storage["key2"] = 3
        keys = set((key, var.value) for key, var in storage.items())
        assert ("key1", 2) in keys
        assert ("key2", 3) in keys


class TestHierarchicalStorage:

    def test_get_value(self):
        child = Storage(k=1)
        parent = Storage(y=2, k=2)
        storage = Storage(child, parent)
        assert storage["k"].value == 1

    def test_storage_contains(self):


        parent = Storage(data=dict(y=2, k=2))
        storage = Storage(data=dict(k=1), parent=parent)
        assert "y" in storage

    def test_with_three_levels(self):

        grandparent = Storage(dict(z=2, k=4))
        parent = Storage(dict(y=2, k=2), grandparent)
        child = Storage(dict(k=1), parent)
        assert "z" in child

    def test_storage_not_contains(self):

        grandparent = Storage(z=2, k=4)
        parent = Storage(y=2, k=2)
        child = Storage(dict(k=1))
        assert "x" not in child

    def test_keys_contains_all_items(self):

        grandparent = Storage(dict(z=2, k=4))
        parent = Storage(dict(y=2, k=2), grandparent)
        child = Storage(dict(k=1), parent)

        keys = set(child.keys())
        assert "z" in keys
        assert "y" in keys

    def test_vars_contains_all_vars(self):


        grandparent = Storage(dict(z=2, k=4))
        parent = Storage(dict(y=2, k=2), grandparent)
        child = Storage(dict(k=1), parent)
        assert "x" not in child

        keys = set(var.val for var in child.vars())
        assert 2 in keys
        assert 1 in keys
