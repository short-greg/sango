from abc import ABC, abstractmethod
import typing

from sango.vars import Ref, Shared, Storage
from .nodes import ClassArgFilter, Status, Task, TaskMeta, TypeFilter


class StateMeta(TaskMeta):

    def __call__(cls, *args, **kw):
        self = cls.__new__(cls, *args, **kw)
        kw['store'] = cls._update_var_stores(kw)
        cls.__init__(self, *args, **kw)
        return self


class State(ABC, metaclass=StateMeta):

    def __init__(self, store: Storage, name: str):

        self._store = store
        self._name = name

    @abstractmethod
    def update(self):
        raise NotImplementedError


class StateLoader(object):

    pass


class StateMachineMeta(TaskMeta):

    def _load_states(cls, store, kw):
        tasks = []
        for name, loader in ClassArgFilter([TypeFilter(StateLoader)]).filter(cls).items():
            if name in kw:
                loader(kw[name])
                del kw[name]
            tasks.append(loader.load(store, name))
        return tasks

    def __call__(cls, *args, **kw):
        self = cls.__new__(cls, *args, **kw)
        kw['store'] = cls._update_var_stores(kw)
        kw['states'] = cls._load_states(kw['store'], kw)
        cls.__init__(self, *args, **kw)
        return self


class StateMachine(Task, metaclass=StateMachineMeta):
    
    def reset(self):
        pass

    def tick(self):
        pass



def state(self):
    pass


def final(self, status=Status.SUCCESS):
    pass


class FSM(StateMachine):

    def __init__(self, start: State, states: typing.List[State], store: Storage=None, name: str=''):
        
        self._start = start
        self._states = states
        self._cur_state = self._start
    
    def reset(self):
        pass

    def tick(self):
        pass

    @property
    def cur_status(self):

        if self._cur_state.final:
            return self._cur_state.status
        return Status.RUNNING


class PlayState(State):

    pause_clicked: Shared = Ref("pause_clicked")
    stop_clicked: Shared = Ref("stop_clicked")

    def __init__(self, stopped: State, paused: State, store: Storage, name: str):
        super().__init__(store, name)
        self._stopped = stopped
        self._paused = paused

    def enter(self):
        pass

    def update(self) -> State:
        if self.pause_clicked.value is True:
            return self._paused
        if self.stop_clicked.value is True:
            return self._stopped
        return self

        


class T(FSM):

    start = state()



    pass

class Factorized(StateMachine):

    pass


class PushDown(StateMachine):

    pass



class Discrete(State):

    pass
