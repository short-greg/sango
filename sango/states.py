"""
State machine classes to use in conjunction with the Behavior Tree. This
makes it possible to build more complex state machines, such as
ones that execute in parallel or ones that are pause the execution of others
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import singledispatch
import typing
from sango.vars import UNDEFINED, Args, Storage
from .nodes import ClassArgFilter, Loader, MemberRefFactory, Status, Task, TaskLoader, TaskMeta, TypeFilter, task
from typing import Any, Generic, TypeVar


class StateMeta(TaskMeta):

    def __call__(cls, *args, **kw):
        self = cls.__new__(cls, *args, **kw)
        store = cls._update_var_stores(kw)
        reference = cls._get_reference(kw)
        cls.__pre_init__(self, store, reference)
        cls.__init__(self, *args, **kw)
        return self


V = TypeVar('V')
 
Emission = TypeVar('Emission')


class State(Generic[V], metaclass=StateMeta):

    def __init__(self, name: str=''):
        self._name = name
        
    def __pre_init__(self, store: Storage, reference):
        self._store = store
        self._reference = reference

    def __getattribute__(self, key: str) -> Any:
        try:
            store: Storage = super().__getattribute__('_store')
            if store.contains(key, recursive=False):
                v = store.get(key, recursive=False)
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


class Discrete(State[V], metaclass=StateMeta):

    def __init__(self, name: str=''):
        """[summary]

        Args:
            status (StateType, optional): Whether the state is a running state, final state, etc. 
            Defaults to StateType.RUNNING.
            name (str, optional): Name of the state. Defaults to ''.
        """
        self._name = name
    
    @property
    def status(self) -> Status:
        raise NotImplementedError

    @abstractmethod
    def update(self) -> Emission:
        raise NotImplementedError
    
    def enter(self):
        """Method called when entering the state. By default it does nothing"""
        pass
    
    def reset(self):
        self.enter()


class Running(Discrete[V]):

    @property
    def status(self) -> Status:
        return Status.RUNNING


class Start(Discrete[V]):

    @property
    def status(self) -> Status:
        return Status.READY


class Failure(Discrete[V]):

    @property
    def status(self) -> Status:
        return Status.FAILURE
    
    @abstractmethod
    def emit_value(self):
        raise NotImplementedError

    def update(self) -> Emission:
        return Emission(self, self.emit_value())


class Success(Discrete[V]):

    @property
    def status(self) -> Status:
        return Status.SUCCESS

    @abstractmethod
    def emit_value(self):
        raise NotImplementedError
    
    def update(self) -> Emission:
        return Emission(self, self.emit_value())


class StateID(object):

    def __init__(self, ref: str):

        self._ref = ref
    
    @property
    def ref(self):
        return self._ref

    def lookup(self, states: typing.Dict[str, State]):

        try: 
            return states[self._ref]
        except KeyError:
            raise KeyError(
                f"State {self._ref} not in states in"
                f"argument states {list(states.keys())}"
            )

StateVar = typing.Union[State, StateID]


@dataclass
class Emission(Generic[V]):
    """
    Emission of a state. An emission consists of the next state and the
    value emitted
    """
    next_state: StateVar
    value: V=None
    
    def emit(self, states: typing.Dict[str, State]=None):
        """Emit the result 

        Args:
            states (typing.Dict[str, State], optional): States in the parent state machine. Defaults to None.

        Returns:
            [type]: The next state and the value emitted
        """
        states = states or {}

        if isinstance(self.next_state, StateID):
            state = self.next_state.lookup(states)
        else:
            state = self.next_state
        state.enter()
        return Emission(state, self.value)


class StateLoader(Loader):
     
    # TODO: add in state decorators later
    def __init__(self, cls: typing.Type=UNDEFINED, args: Args=None):

        super().__init__(cls, args)


class StateMachineMeta(TaskMeta):

    def _load_states(cls, store, kw, reference):
        states = {}
        for name, loader in ClassArgFilter([TypeFilter(StateLoader)]).filter(cls).items():
            if name in kw:
                loader(kw[name])
                del kw[name]
            states[name] = loader.load(store, name, reference)
        return states

    def __call__(cls, *args, **kw):
        self = cls.__new__(cls, *args, **kw)
        store = cls._update_var_stores(kw)
        reference = cls._get_reference(kw)
        states = cls._load_states(store, kw, reference)
        start = states['start']
        del states['start']
        cls.__pre_init__(self, start, states, store, reference)
        cls.__init__(self, *args, **kw)
        return self


class StateMachine(Task, metaclass=StateMachineMeta):
    
    def __pre_init__(self, start: State, states: typing.Dict[str, State], store: Storage, reference):

        self._start = start
        self._states = states
        self._store = store
        self._reference = reference
    
    def reset(self):
        pass

    @abstractmethod
    def tick(self):
        raise NotImplementedError


class TaskState(Discrete):
    
    def __init__(self, task: Task, failure_to: StateVar, success_to: StateVar):

        self._task = task
        self._failure_to = failure_to
        self._success_to = success_to

    def enter(self):

        self._task.reset()

    def update(self):

        if self._task.status == Status.DONE:
            return None

        result = self._task.tick()
        if result == Status.FAILURE:
            return self._failure_to
        elif result == Status.SUCCESS:
            return self._success_to
        
        return self


class TaskStateLoader(StateLoader):

    def __init__(self, task_loader: TaskLoader, failure_to: StateVar, success_to: StateVar):

        def load(store, name, *args, **kwargs):
            return TaskState(task_loader.load(store, name, *args, **kwargs), failure_to, success_to)
        super().__init__(load)

    def __call__(self, state: State):
        self._state = state
        return self

@singledispatch
def state_(s: typing.Union[State, TaskLoader], *args, **kwargs):

    if issubclass(s, State):
        return StateLoader(s, Args(*args, **kwargs))
    
    return StateLoader(TaskStateLoader(s))

@state_.register
def _(s: str):
    return StateLoader(DiscreteStateRef, args=Args(member_ref=s))


def state(*args, **kwargs):
    return StateLoader(args=Args(*args, **kwargs))


class FSM(StateMachine):

    # TODO: check status of start state and final states
    def __pre_init__(self, start: Discrete, states: typing.Dict[str, Discrete], store: Storage, reference):

        super().__pre_init__(start, states, store, reference)
        self._cur_state: Discrete = self._start
        self._cur_state.reset()
    
    def __init__(self, name: str=''):
        
        self._name = name

    @property
    def states(self):
        return [self._start] + [*self._states.values()]
    
    def reset(self):
        
        self._cur_state = self._start
        self._start.reset()

    def tick(self):

        emission = self._cur_state.update().emit(self._states)
        self._cur_state = emission.next_state

        if emission.next_state != self._cur_state:
            self._cur_state.reset()

        return self._cur_state.status

    @property
    def cur_state(self):
        return self._cur_state

    @property
    def status(self) -> Status:
        return self._cur_state.status


class StateLink(object):

    def __init__(self, **state_map: typing.Dict[str, str]):

        self._state_map = state_map

    def __getitem__(self, state: State) -> str:
        return StateID(self._state_map[state.name])


class FSMState(Discrete):
    
    def __init__(self, name: str, machine: FSM, state_link: StateLink):

        super().__init__(name)
        self._machine = machine
        self._state_link = state_link

    def enter(self):

        self._machine.reset()

    def update(self):

        if self._machine.status == Status.DONE:
            return None

        result = self._machine.tick()
        # TODO: Determine what to use for emission
        if result.done:
            return Emission(self._state_link[self._machine.cur_state], None)

        return Emission(self, None)

    @property
    def status(self):
        return self._machine.status


class DiscreteStateRef(Discrete):

    def __init__(self, name: str, member_factory: MemberRefFactory):
        super().__init__(name)
        self._member_ref = member_factory.produce(self._store, self._reference)
    
    @property
    def state(self) -> Discrete:
        return self._member_ref.get()

    def enter(self):
        self.state.enter()
    
    def reset(self):
        super().reset()
        self.state.reset()
    
    def update(self):
        return self.state.update()


def to_state(**state_map: typing.Dict[str, str]):
    
    def _(states: typing.List[State]):

        _states = set()
        
        for state in states:
            if state.name in state_map:
                if not state.status.done:
                    raise ValueError(f"State {state.name} is not a final state.")
                # _state_map[state.name] = state_map[state.name]
            elif state.status.done:
                    raise ValueError(
                        f"State {state.name} is a final state" 
                        "but does not map to another state."
                    )
            _states.add(state.name)

        difference = set(state_map.keys()).difference(_states)
        if len(difference) > 0:
            raise ValueError(
                f"Mapping is not defined for {difference}"
            )

        return StateLink(**state_map)
    return _


def to_status(failure: typing.Optional[str] = None, success: typing.Optional[str]=None):

    def _(states: typing.List[State]):

        _state_map: typing.Dict[Discrete, str] = {}
        
        for state in states:
            if state.status == Status.FAILURE:
                if failure is None:
                    raise ValueError(
                        f"There is a failure state but no mapping for success"
                    )
                _state_map[state.name] = failure
            elif state.status == Status.SUCCESS:
                _state_map[state.name] = success
        
                if success is None:
                    raise ValueError(
                        f"There is a success state but no mapping for success"
                    )
        return StateLink(**_state_map)
    return _


LinkFunc = typing.Callable[[typing.List[Discrete]], StateLink]


class FSMStateLoader(StateLoader):

    def __init__(self, fsm_factory, link_f: LinkFunc, args: Args=None):

        self._args = args or Args()
        self._fsm_factory = fsm_factory
        def load(store, name='', *args, **kwargs):

            if self._fsm_factory is None:
                raise ValueError("The finite state machine factory has not been defined.")
            fsm: FSM = self._fsm_factory(store=store, *args, **kwargs)
            
            return FSMState(
                name=name, machine=fsm, state_link=link_f(fsm.states)
            )

        super().__init__(load, args)

    def __call__(self, fsm_factory):
        self._fsm_factory = fsm_factory
        return self


@singledispatch
def fsmstate_(s: typing.Type, map_to: typing.Dict[str, State], *args, **kwargs):

    return FSMStateLoader(s, map_to, Args(*args, **kwargs))


@singledispatch
def fsmstate(map_to: typing.Dict[str, State], *args, **kwargs):

    return FSMStateLoader(None, map_to, Args(*args, **kwargs))


# TODO: Determine if this is necessary
# What are the use cases?
class FSMRef(FSM):
    pass
