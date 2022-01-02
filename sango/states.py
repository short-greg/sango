from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import singledispatch
import typing

from sango.vars import UNDEFINED, Args, HierarchicalStorage, Ref, Shared, Storage
from .nodes import ClassArgFilter, Loader, Status, Task, TaskMeta, TypeFilter
from typing import Any, Generic, TypeVar


class StateMeta(TaskMeta):

    def __call__(cls, *args, **kw):
        self = cls.__new__(cls, *args, **kw)
        store = cls._update_var_stores(kw)
        cls.__pre_init__(self, store)
        cls.__init__(self, *args, **kw)
        return self


V = TypeVar('V')

Emission = TypeVar('Emission')


class State(Generic[V], metaclass=StateMeta):

    def __init__(self, name: str=''):
        self._name = name
        
    def __pre_init__(self, store: Storage):
        self._store = store


    def __getattribute__(self, key: str) -> Any:
        try:
            store: Storage = super().__getattribute__('_store')
            if isinstance(store, HierarchicalStorage):
                if store.contains(key, recursive=False):
                    v = store.get(key, recursive=False)
                    return v
            else:
                if store.contains(key):
                    v = store.get(key)
                    return v

        except AttributeError:
            pass
        return super().__getattribute__(key)

    @property
    def name(self):
        return self._name

    @abstractmethod
    def update(self) -> Emission:
        raise NotImplementedError
    
    @abstractmethod
    def enter(self):
        raise NotImplementedError
    
    def reset(self):
        pass


class StateType(Enum):

    SUCCESS = 0
    FAILURE = 1
    READY = 2
    RUNNING = 3

    @property
    def final(self):
        return self == StateType.SUCCESS or self == StateType.FAILURE

    @property
    def status(self):
        if self == StateType.SUCCESS:
            return Status.SUCCESS
        if self == StateType.FAILURE:
            return Status.FAILURE
        if self == StateType.READY:
            return Status.READY
        return Status.RUNNING


class Discrete(State[V], metaclass=StateMeta):

    def __init__(self, status: StateType=StateType.RUNNING, name: str=''):
        self._name = name
        self._status = status
    
    @property
    def status(self) -> StateType:
        return self._status

    @abstractmethod
    def update(self) -> Emission:
        raise NotImplementedError
    
    def enter(self):
        self._status = StateType.READY
    
    def reset(self):
        self.enter()


class StateRef(object):

    def __init__(self, ref: str):

        self._ref = ref

    def lookup(self, states: typing.Dict[str, State]):

        return states[self._ref]


StateVar = typing.Union[State, StateRef]


class Emission(Generic[V]):

    def __init__(self, next_state: StateVar, value: V=None):
        self._next_state = next_state
        self._value = value
    
    def emit(self, states: typing.Dict[str, State]=None):
        states = states or {}

        if isinstance(self._next_state, StateRef):
            state = self._next_state.lookup(states)
        else:
            state = self._next_state
        state.enter()
        return state, self._value



# @singledispatch
# def process_state_var(state: State, states: typing.Dict[str, State]):
#     return state


# @process_state_var.register
# def _(state: StateRef, states: typing.Dict[str, State]):
#     return state.lookup(states)


class StateLoader(Loader):

    def __init__(self, state_cls: typing.Type[State]=UNDEFINED, args: Args=None, decorators=None):
        super().__init__(state_cls, args, decorators)

    def __call__(self, state: State):
        self._state = state
        return self


class StateMachineMeta(TaskMeta):

    def _load_states(cls, store, kw):
        states = {}
        for name, loader in ClassArgFilter([TypeFilter(StateLoader)]).filter(cls).items():
            if name in kw:
                loader(kw[name])
                del kw[name]
            states[name] = loader.load(store, name)
        return states

    def __call__(cls, *args, **kw):
        self = cls.__new__(cls, *args, **kw)
        store = cls._update_var_stores(kw)
        states = cls._load_states(store, kw)
        start = states['start']
        del states['start']
        cls.__pre_init__(self, start, states, store)
        cls.__init__(self, *args, **kw)
        return self


class StateMachine(Task, metaclass=StateMachineMeta):
    
    def __pre_init__(self, start: State, states: typing.Dict[str, State], store: Storage):

        self._start = start
        self._states = states
        self._store = store
    
    def reset(self):
        pass

    def tick(self):
        pass


def decorate(state_loader: StateLoader, decorators):
    state_loader.add_decorators(decorators)


def state_(state_cls, *args, **kwargs):
    return StateLoader(state_cls, Args(*args, **kwargs) )


def state(*args, **kwargs):
    return StateLoader(args=Args(*args, **kwargs))


class FSM(StateMachine):

    def __pre_init__(self, start: Discrete, states: typing.Dict[str, Discrete], store: Storage):

        super().__pre_init__(start, states, store)
        self._cur_state = self._start
        self._cur_state.reset()
    
    def __init__(self, name: str=''):
        
        self._name = name
    
    def reset(self):
        
        self._cur_state = self._start
        self._start.reset()

    def tick(self):

        emission = self._cur_state.update()
        next_state, value = emission.emit(self._states)
        if next_state.status.final:
            self._cur_state = next_state
            return self._cur_state.status
        elif next_state != self._cur_state:
            self._cur_state = next_state
            self._cur_state.reset()
        return Status.RUNNING

    @property
    def cur_state(self):
        return self._cur_state

    @property
    def status(self):
        return self._cur_state.status.status

# class PlayState(State):

#     pause_clicked: Shared = Ref("pause_clicked")
#     stop_clicked: Shared = Ref("stop_clicked")

#     def __init__(self, stopped: State, paused: State, store: Storage, name: str):
#         super().__init__(store, name)
#         self._stopped = stopped
#         self._paused = paused

#     def enter(self):
#         # self.play
#         pass

#     def update(self) -> State:
#         if self.pause_clicked.value is True:
#             return self._paused
#         if self.stop_clicked.value is True:
#             return self._stopped
#         if self.finished is True:
#             return self._stopped
#         return self
