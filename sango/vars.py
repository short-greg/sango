
from abc import ABC, abstractmethod
from functools import partialmethod, singledispatchmethod
from typing import TypeVar, Generic
from itertools import chain
import typing
from dataclasses import dataclass


UNDEFINED = object()

V = TypeVar('V')

class StoreVar(Generic[V]):
    
    @abstractmethod
    def val(self):
        raise NotImplementedError

    @abstractmethod
    def is_empty(self):
        raise NotImplementedError

T = TypeVar('T')

class Var(StoreVar[T]):
    
    def __init__(self, val: T):

        self._val = val

    @property
    def val(self) -> T:
        return self._val

    @val.setter
    def val(self, val):
        self._val = val
    
    def empty(self):
        self._val = None

    def is_empty(self):
        return self._val is None


class Const(StoreVar[T]):
    
    def __init__(self, val: T):
        self._val = val

    @property
    def val(self) -> T:
        return self._val

    def is_empty(self):
        return self._val is None

class Shared(StoreVar[T]):

    def __init__(self, var: Var[T]):

        self._var = var
    
    @property
    def val(self):
        return self._var.val

    @val.setter
    def val(self, val) -> T:
        self._var.val = val

    def is_empty(self):
        return self._var.is_empty()

    def empty(self):
        return self._var.empty()


class ConstShared(StoreVar):

    def __init__(self, var: Var[T]):
        self._var = var
    
    @property
    def val(self) -> T:
        return self._var.val

    def is_empty(self):
        return self._var.is_empty()


class Storage(ABC):

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
    
    def contains(self, key: str):
        return key in self._data

    @abstractmethod
    def get_or_add(self, key, default=None):
        raise NotImplementedError

class SingleStorage(Storage):

    def __init__(self, **kwargs):
        self._data = {}
        for k, v in kwargs.items():
            
            self[k] = v

    @singledispatchmethod
    def __setitem__(self, key, value):
        if isinstance(value, Var):
            self._data[key] = value
        else:
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

    def get(self, key, default=None):
        return self._data.get(key, default)
                
    def __contains__(self, key: str):
        return self.contains(key)

    def contains(self, key: str):
        return key in self._data

    def get_or_add(self, key, default=None):

        if not self.contains(key):
            self._data[key] = Var(default)
        
        return self.get(key)

Storage.__contains__ = Storage.contains


class NullStorage(Storage):

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

    def contains(self, key: str):
        return False

    def get_or_add(self, key, default=None):
        return default

NullStorage.__contains__ = NullStorage.contains


class Ref(ABC):
    
    @abstractmethod
    def val(self, store: Storage):
        raise NotImplementedError

    @abstractmethod
    def shared(self, store: Storage) -> Shared:
        raise NotImplementedError
    
    @abstractmethod
    def var(self, store: Storage) -> Var:
        raise NotImplementedError

STORE_REF = object()


class HierarchicalStorage(Storage):

    def __init__(self, child: Storage, parent: Storage=None):

        self._parent = parent or NullStorage()
        self._child = child

    @singledispatchmethod
    def __setitem__(self, key, value):
        #TODO: Find out why this doesn't work
        if isinstance(value, Var):
            self._child[key] = value
        else:
            self._child[key] = Var(value)

    @__setitem__.register
    def _(self, key, value: Var):
        self._child[key] = value

    @__setitem__.register
    def _(self, key, value: Ref):
        self._child[key] = value.shared(self._parent)

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
    
    def get_or_add(self, key, default=None, recursive=False):

        if (
            key not in self._child 
            and (not recursive or not self._parent.contains(key, recursive=True))
        ):
            self._child[key] = default
        
        return self.get(key, recursive=recursive)

    def contains(self, key: str, recursive: bool=False):
        if key in self._child:
            return True
        
        if recursive:
            return key in self._parent
        return False


HierarchicalStorage.__contains__ = partialmethod(HierarchicalStorage.contains, recursive=False)


class VarRef(Ref):

    def __init__(self, var_name: str, store: Storage=None):

        self._var_name = var_name
        self.store = store

    @property
    def name(self):
        return self._var_name
    
    def val(self, store: Storage):
        if store is None:
            raise AttributeError("Storage to reference has not been set.")
        return store[self._var_name]

    def shared(self, store: Storage) -> Shared:

        return Shared(store[self._var_name])
    
    def val(self, store: Storage) -> Var:
        return Var(store[self._var_name].val)


class ConstRef(Ref):

    def __init__(self, var_name: str):

        self._var_name = var_name

    @property
    def name(self):
        return self._var_name
    
    def val(self, store: Storage):
        if store is None:
            raise AttributeError("Storage to reference has not been set.")
        return store[self._var_name]

    def shared(self, store: Storage) -> Shared:
        return ConstShared(store[self._var_name])
    
    def val(self, store: Storage) -> Var:
        return Const(store[self._var_name].val)


@dataclass
class Condition:
    value: str


class _ref(object):

    def __setattr__(self, __name: str, __value) -> None:
        raise AttributeError('Cannot set {__name} for ref object')

    def __getattr__(self, key):

        return VarRef(key)


class _const_ref(object):

    def __setattr__(self, __name: str, __value) -> None:
        raise AttributeError('Cannot set {__name} for ref object')

    def __getattr__(self, key):

        return ConstRef(key)


ref = _ref()
cref = _const_ref()


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
    
    def _update(self, a, store):
        if isinstance(a, Ref):
            return a.shared(store)
        if a == STORE_REF:
            return store
        return a

    def update_refs(self, store: Storage):
        args = [self._update(v, store) for v in self.args]
        kwargs = {k: self._update(v, store) for k, v in self.kwargs.items()}

        return Args(*args, **kwargs)
