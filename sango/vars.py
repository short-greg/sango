
from abc import ABC, abstractmethod
from functools import singledispatchmethod
from typing import TypeVar, Generic
from itertools import chain


class BaseVar(ABC):
    
    @abstractmethod
    def value(self):
        raise NotImplementedError

# TODO: Make Var use a generic

class Var(BaseVar):
    
    def __init__(self, value):

        self._value = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value


class Shared(BaseVar):

    def __init__(self, var: Var):

        self._var = var
    
    @property
    def value(self):
        return self._var.value

    @value.setter
    def value(self, value):
        self._var.value = value


class AbstractStorage(ABC):

    @abstractmethod
    def __setitem__(self, key, value):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, key) -> Var:
        raise NotImplementedError
    
    def items(self):
        raise NotImplementedError

    def keys(self):
        raise NotImplementedError

    def vars(self):
        raise NotImplementedError

    def get(self, key, default):
        raise NotImplementedError


class Storage(AbstractStorage):

    def __init__(self, **kwargs):
        self._data = {}
        for k, v in kwargs.items():
            self[k] = v

    @singledispatchmethod
    def __setitem__(self, key, value):
        self._data[key] = Var(value)

    @__setitem__.register
    def _(self, key, value: Var):
        self._data[key] = value

    def __getitem__(self, key) -> Var:
        return self._data[key]
    
    def items(self):
        return self._data.items()

    def keys(self):
        return self._data.keys()

    def vars(self):
        return self._data.values()

    def get(self, key, default):
        return self._data.get(key, default)
    
    def __contains__(self, key: str):
        return key in self._data


class HierarchicalStorage(AbstractStorage):

    def __init__(self, child: AbstractStorage, parent: AbstractStorage):

        self._parent = parent
        self._child = child

    def __setitem__(self, key, value):
        
        self._child[key] = value

    def __getitem__(self, key) -> Var:
        
        if key in self._child:
            return self._child[key]
        elif key in self._parent:
            return self._parent[key]
        raise ValueError(f"Key {key} not in child or parent")
    
    def items(self):
        for key, val in chain(self._child.items(), self._parent.items()):
            yield key, val

    def keys(self):
        for key in chain(self._child.keys(), self._parent.keys()):
            yield key

    def vars(self):
        for var in chain(self._child.vars(), self._parent.vars()):
            yield var

    def get(self, key, default=None):
        
        if key in self._child:
            return self._child[key]
        elif key in self._parent:
            return self._parent[key]
        return default

    def __contains__(self, key: str):
        return key in self._child or key in self._parent


class Ref(object):

    def __init__(self, var_name: str):

        self._var_name = var_name

    @property
    def shared(self, storage: Storage) -> Shared:

        return Shared(storage[self._var_name])
    
    @property
    def var(self, storage: Storage) -> Var:
        return Var(storage[self._var_name].value)
