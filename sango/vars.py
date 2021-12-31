
from abc import ABC, abstractmethod
from functools import partialmethod, singledispatchmethod
from typing import TypeVar, Generic
from itertools import chain
import typing
from dataclasses import dataclass


class _Undefined:
    pass

UNDEFINED = _Undefined()

V = TypeVar('V')

class StoreVar(Generic[V]):
    
    @abstractmethod
    def value(self):
        raise NotImplementedError


T = TypeVar('T')

class Var(StoreVar[T]):
    
    def __init__(self, value: T):

        self._value = value

    @property
    def value(self) -> T:
        return self._value

    @value.setter
    def value(self, value):
        self._value = value


class Shared(StoreVar):

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
    
    @abstractmethod
    def items(self):
        raise NotImplementedError

    @abstractmethod
    def keys(self):
        raise NotImplementedError

    @abstractmethod
    def vars(self):
        raise NotImplementedError

    @abstractmethod
    def get(self, key, default):
        raise NotImplementedError
    
    @abstractmethod
    def add(self, var: Var):
        raise NotImplementedError

    def contains(self, key: str):
        return key in self._data


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
        return self.contains(key)
    
    def add(self, key: str, var: Var):
        self._data[key] = var

    def contains(self, key: str):
        return key in self._data

Storage.__contains__ = Storage.contains


class NullStorage(AbstractStorage):

    def __init__(self):
        self._data = {}

    def __setitem__(self, key, value):
        pass

    def __getitem__(self, key) -> Var:
        pass
    
    def items(self):
        return self._data.items()

    def keys(self):
        return self._data.keys()

    def vars(self):
        return self._data.values()

    def get(self, key, default):
        return default
    
    # def update_references(self, parent: AbstractStorage):
    #     pass

    def add(self, key: str, var: Var):
        pass

    def contains(self, key: str):
        return False


NullStorage.__contains__ = NullStorage.contains


class Ref(object):

    def __init__(self, var_name: str, store: Storage=None):

        self._var_name = var_name
        self.store = store

    @property
    def name(self):
        return self._var_name
    
    def value(self):
        if self.store is None:
            raise AttributeError("Storage to reference has not been set.")
        return self.store[self._var_name]

    def shared(self, storage: Storage) -> Shared:

        return Shared(storage[self._var_name])
    
    def var(self, storage: Storage) -> Var:
        return Var(storage[self._var_name].value)


class HierarchicalStorage(AbstractStorage):

    def __init__(self, child: AbstractStorage, parent: AbstractStorage=None):

        self._parent = parent or NullStorage()
        self._child = child
        # self._child.update_references(self._parent)

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

    def get(self, key, default=None, recursive=False):
        
        if key in self._child:
            return self._child[key]
        elif recursive and key in self._parent:
            return self._parent[key]
        return default

    def contains(self, key: str, recursive: bool=False):
        if key in self._child:
            return True
        
        if recursive:
            return key in self._parent
        return False

    @singledispatchmethod
    def add(self, key: str, var):
        self._child.add(key, Var(var))

    @add.register
    def _(self, key: str, var: Ref):
        self._child.add(
            key, var.shared(self._parent)
        )

    @add.register
    def _(self, key: str, var: Var):
        self._child.add(key, var)


HierarchicalStorage.__contains__ = partialmethod(HierarchicalStorage.contains, recursive=False)


@dataclass
class Condition:
    value: str


class ConditionSet(object):

    def __init__(self, conditions: typing.Iterable[Condition]=None):
        if conditions is not None:
            self._conditions = set(condition for condition in conditions)
        else:
            self._conditions = set()
        
    def satisfies(self, condition: Condition):
        return condition in self._conditions
    
    def add(self, condition: Condition):
        self._conditions.add(condition)
    
    def remove(self, condition: Condition):
        self._conditions.remove(condition)

    def __iter__(self):
        for condition in self._conditions:
            yield condition


class Args(object):

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
