import pytest
from .vars import HierarchicalStorage, Shared, Storage, Var


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
        storage = HierarchicalStorage(child, parent)
        assert storage["k"].value == 1

    def test_storage_contains(self):

        child = Storage(k=1)
        parent = Storage(y=2, k=2)
        storage = HierarchicalStorage(child, parent)
        assert "y" in storage

    def test_with_three_levels(self):

        child = Storage(k=1)
        parent = Storage(y=2, k=2)
        grandparent = Storage(z=2, k=4)
        hierarchical = HierarchicalStorage(parent, grandparent)
        storage = HierarchicalStorage(child, hierarchical)
        assert "z" in storage

    def test_storage_not_contains(self):

        child = Storage(k=1)
        parent = Storage(y=2, k=2)
        grandparent = Storage(z=2, k=4)
        hierarchical = HierarchicalStorage(parent, grandparent)
        storage = HierarchicalStorage(child, hierarchical)
        assert "x" not in storage

    def test_keys_contains_all_items(self):

        child = Storage(k=1)
        parent = Storage(y=2, k=2)
        grandparent = Storage(z=2, k=4)
        hierarchical = HierarchicalStorage(parent, grandparent)
        storage = HierarchicalStorage(child, hierarchical)

        keys = set(storage.keys())
        assert "z" in keys
        assert "y" in keys

    def test_vars_contains_all_vars(self):

        child = Storage(k=1)
        parent = Storage(y=2, k=2)
        grandparent = Storage(z=2, k=4)
        hierarchical = HierarchicalStorage(parent, grandparent)
        storage = HierarchicalStorage(child, hierarchical)
        assert "x" not in storage

        keys = set(var.value for var in storage.vars())
        assert 2 in keys
        assert 1 in keys
