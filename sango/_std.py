"""
Nodes for buidling a Behavior Tree. A tree can be built hierarchically within
Python's class system by specifying which members are tasks and which
are variables to store.

Also nodes for building a state machine to use in conjunction with the Behavior Tree. This
makes it possible to build more complex state machines, such as
ones that execute in parallel or ones that are pause the execution of others

"""

from enum import Enum
import typing
from abc import ABC, abstractmethod, abstractproperty
from functools import wraps
from ._utils import coalesce
from typing import Iterator
from typing import Generic, TypeVar
from dataclasses import dataclass


class Status(Enum):
    """The status of a task
    """

    FAILURE = 0  # the task failed
    SUCCESS = 1  # the task succeeded
    RUNNING = 2  # the task is runnning
    READY = 3  # the task has not started
    DONE = 4  # the task has already finished
    NONE = 5  # the task has no state

    @property
    def done(self):
        """Whether the task has finished
        """
        return self == Status.FAILURE or self == Status.SUCCESS or self == Status.DONE


class Filter(ABC):
    """
    Use to filter nodes in a tree
    """
    @abstractmethod
    def check(self, node):
        raise NotImplementedError


class NullFilter(Filter):

    def check(self, node):
        return True


class Task(object):
    """The base class for a Task node

    A task node has a 'store' and a 'reference' object which may 
    be None.

    Attributes in the 'store' can be accessed through the attribute
    operator.
    """

    def __init__(self, name: str=''):
        self._name = name
        self._cur_status: Status = Status.READY
    
    def reset(self):
        self._cur_status = Status.READY
    
    @abstractmethod
    def tick(self) -> Status:
        """Execute the task

        Returns:
            Status
        """
        raise NotImplementedError

    @property
    def status(self) -> Status:
        """Current status of the 

        Returns:
            Status
        """
        return self._cur_status
    
    @property
    def name(self) -> str:
        return self._name

    def __lshift__(self, decorator):
        """
        Args:
            decorator: TickDecorator or functional decorator

        Returns:
            Decorator
        """

        if isinstance(TickDecorator, decorator):
            return decorator.decorate(self)
        return decorator(self)
    
    def iterate(self, filter: Filter=None, deep: bool=True):

        # hack to iterate on any task
        if False:
            yield None


Task.__call__ = Task.tick


class Planner(ABC):
    """Chooses the order which to execute the subtasks of a composite task
    """

    @abstractmethod
    def adv(self):
        pass

    @abstractmethod
    def rev(self):
        pass

    @abstractmethod
    def end(self):
        pass
    
    @abstractmethod
    def reset(self):
        pass

    @abstractproperty
    def idx(self):
        pass
    
    @idx.setter
    def idx(self, idx: int):
        pass

    @abstractproperty
    def cur(self):
        pass

    @abstractmethod
    def clone(self):
        pass

    @abstractmethod
    def __len__(self):
        pass


class LinearPlanner(object):
    """Planner that executes sequentially
    """

    def __init__(self, items: typing.List[Task]):
        """initializer

        Args:
            items (typing.List[Task])
        """
        self._items = items
        self._idx = 0

    def reset(self, items: list=None):
        """Reset the planner to the start state

        Args:
            items (list, optional): Updated items to set the planner to. Defaults to None.
        """
        self._idx = 0
        self._items = coalesce(items, self._items)
    
    @property
    def idx(self):
        return self._idx
    
    @idx.setter
    def idx(self, idx):
        if not (0 <= idx <= len(self._items)):
            raise IndexError(f"Index {idx} is out of range in {len(self._items)}")
        self._idx = idx

    def end(self) -> bool:
        """Advance the planner

        Returns:
            bool
        """
        if self._idx == len(self._items):
            return True
        return False
    
    def adv(self) -> bool:
        """Advance the planner

        Returns:
            bool
        """
        if self._idx == len(self._items):
            return False
        self._idx += 1
        return True
    
    @property
    def cur(self) -> Task:
        """
        Returns:
            Task
        """
        if self._idx == len(self._items):
            return None
        return self._items[self._idx]

    def rev(self) -> bool:
        """Reverse the planner

        Returns:
            bool
        """
        if self._idx == 0:
            return False
        self._idx -= 1
        return True

    def clone(self):
        """Clone the linear planner

        Returns:
            LinearPlanner
        """
        return LinearPlanner([*self._items])
    
    def __len__(self) -> int:
        return len(self._items)


def iterate_planner(planner: Planner) -> Iterator[Task]:
    """
    Convenience function to iterate over a planner
    """

    planner.reset()
    
    while planner.end() is False:
        yield planner.cur
        planner.adv()


class PlannerDecorator(Planner):
    """Decorate the planner
    """

    def __init__(self, planner: Planner):
        """Planner

        Args:
            planner (Planner)
        """
        self._planner = planner

    def adv(self):
        self._planner.adv()

    def rev(self):
        self._planner.rev()

    def end(self):
        return self._planner.end()
    
    def reset(self):
        self._planner.reset()

    @property
    def cur(self):
        return self._planner.cur

    def clone(self):
        return self.__class__(self._planner.clone())


class Composite(Task):
    """Task composed of subtasks
    """

    def __init__(
        self, tasks: typing.List[Task], name: str='',  planner: Planner=None
    ):
        super().__init__(name)
        self._planner = planner or LinearPlanner(tasks)
        self._sub_status = Status.READY
        self._tasks = tasks

    @property
    def n(self):
        """The number of subtasks"""
        return len(self._tasks)
    
    @property
    def tasks(self):
        """The subtasks"""
        return [*self._tasks]

    @abstractmethod
    def subtick(self) -> Status:
        """Tick each subtask. Implement when implementing a new Composite task"""
        raise NotImplementedError

    @property
    def status(self):
        return self._cur_status

    def tick(self):
        
        if self._cur_status.done:
            return Status.DONE

        status = self.subtick()
        self._cur_status = status
        return status
    
    def reset(self):
        super().reset()
        self._planner.reset()
        for task in self._tasks:
            task.reset()

    def iterate(self, filter: Filter=None, deep: bool=True):
        filter = filter or NullFilter()    
        for task in self._tasks:
            if filter.check(task):
                yield task
                if deep:
                    for subtask in task.iterate(filter, deep):
                        yield subtask


class Tree(Task):
    """The base behavior tree task. Use the behavior tree like a regular class
    The reference object for subtasks will refer to the 'tree' object
    """

    def __init__(self, name: str, entry: Task):
        super().__init__(name)
        self.entry = entry
    
    def tick(self) -> Status:        
        self._cur_status = self.entry.tick()
        return self._cur_status
    
    def reset(self):
        self._cur_status = Status.READY
        return self.entry.reset()

    def iterate(self, filter: Filter=None, deep: bool=True):
        filter = filter or NullFilter()
        
        if filter.check(self.entry):
            yield self.entry
            if deep:
                for subtask in self.entry.iterate(filter, deep):
                    yield subtask


class Action(Task):
    """Use to execute an action. Implement the 'act' method for subclasses"""

    @abstractmethod
    def act(self):
        raise NotImplementedError

    def tick(self):
        if self._cur_status.done:
            return Status.DONE
        self._cur_status = self.act()
        return self._cur_status


class Conditional(Task):
    """Use to check a condition. Implement the 'check' method for subclasses"""

    @abstractmethod
    def check(self) -> bool:
        raise NotImplementedError

    def tick(self):
        if self._cur_status.done:
            return Status.DONE
        self._cur_status = Status.SUCCESS if self.check() else Status.FAILURE
        return self._cur_status


class Sequence(Composite):
    """
    Executes the subtasks in sequential order
    Succeeds when all subtasks have succeeded
    """

    def reset(self):
        super().reset()
        self._planner.reset()
    
    def _plan(self):
        return self._tasks

    def subtick(self) -> Status:

        if self._planner.end() is True:
            return Status.NONE
        
        status = self._planner.cur.tick()
        if status == Status.FAILURE:
            return Status.FAILURE
        
        if status == Status.SUCCESS:
            self._planner.adv()
        if self._planner.end():
            return Status.SUCCESS
        return Status.RUNNING


class Fallback(Composite):
    """
    Executes the subtasks in sequential order
    Succeeds when one subtask has succeeded
    """

    def subtick(self) -> Status:

        if self._planner.end() is True:
            return Status.NONE
        
        status = self._planner.cur.tick()
        if status == Status.SUCCESS:
            return Status.SUCCESS
        
        if status == Status.FAILURE:
            self._planner.adv()
        
        if self._planner.end():
            return Status.FAILURE
        return Status.RUNNING


class Parallel(Composite):
    """
    Executes the subtasks in parallel
    Succeeds when all subtasks have succeeded
    Fails when all subtasks have finished and one fails
    """

    def __init__(
        self, tasks: typing.List[Task], name: str='', planner: Planner=None
    ):
        super().__init__(tasks, name, planner=planner)
        self._statuses = []
    
    def status_total(self, status: Status):
        total = 0
        for s in self._statuses:
            if s == status:
                total += 1
        return total
    
    def reset(self):
        super().reset()
        self._statuses = []

    def subtick(self):

        for i, task in enumerate(iterate_planner(self._planner)):
            if i > len(self._statuses) - 1:
                self._statuses.append(Status.RUNNING)
            elif self._statuses[i] != Status.RUNNING:
                continue
            self._statuses[i] = task.tick()
    
        if Status.RUNNING in self._statuses:
            return Status.RUNNING 

        return Status.SUCCESS if Status.FAILURE not in self._statuses else Status.FAILURE


class TaskDecorator(Task):
    """A 'task' that decorates another task
    """
    
    def __init__(self, task: Task):
        """initializer

        Args:
            task (Task): Task to decorate
        """
        super().__init__('')
        self._task = task
    
    @abstractmethod
    def decorate(self):
        raise NotImplementedError

    def tick(self):

        if self._cur_status.done:
            return Status.DONE

        return self.decorate()


class TickDecorator(object):
    """
    Wraps the 'tick' method of a class with anohter function

    When inheriting, implement the decorate_tick method
    """

    def __init__(self, node: typing.Type[Task]=None):
        self._node = node

    @abstractmethod
    def decorate_tick(self, node):
        """Decorate the tick method of the argument node

        Args:
            node (Task)
        """
        raise NotImplementedError
    
    def decorate(self, node: Task):
        node.tick = wraps(node.tick)(self.decorate_tick(node))
        return node

    def __call__(self, *args, **kwargs):
        """
        Instantiate the node class with args, kwargs and
        then decorate it with the decorate_tick method

        Returns:
            Decorated node
        """
        if self._node is None:
            raise AttributeError(f'Member node has not been instantiated')
        
        return self.decorate(self._node(*args, **kwargs))


class TickDecorator2nd(TickDecorator):
    """
    2nd order decorator. Wraps the 'tick' method of a class with another function

    When inheriting, implement the decorate method
    """
    def __init__(self, *args, **kwargs):
        """
        Args:
            args : The args into the network
            kwargs: The kwargs
        """
        super().__init__()
        self._args = args
        self._kwargs = kwargs

    @abstractmethod
    def decorate_tick(self, node):
        """Decorate the tick method of the argument node

        Args:
            node (Task)
        """
        raise NotImplementedError

    def decorate(self, node: Task):
        node.tick = wraps(node.tick)(self.decorate_tick(node))
        return node
    
    def __call__(self, node_cls: typing.Type[Task]):
        """Return a method to instantiate the node passed in

        Args:
            node_cls (typing.Type[Task]): [description]
        """
        
        def instantiator(*args, **kwargs):

            node = node_cls(*args, **kwargs)
            return self.decorate(node)
        
        return instantiator


V = TypeVar('V')


class State(Generic[V]):
    """Standard state in a machine
    """

    def __init__(self, name: str=''):
        self._name = name

    @property
    def name(self) -> str:
        """
        Returns:
            str: Name of the state
        """
        return self._name

    @abstractmethod
    def update(self):
        raise NotImplementedError
    
    @abstractmethod
    def enter(self):
        raise NotImplementedError
    
    def reset(self):
        pass

    def iterate(self, filter: Filter=None, deep: bool=True):        
        pass


class StateID(object):
    """A reference to a state. Used in emission as
    part of the update function
    """

    def __init__(self, ref: str):
        """initializer
        Args:
            ref (str): The name of the state 
        """
        self._ref = ref
    
    @property
    def ref(self) -> str:
        """Reference of the ID

        Returns:
            str: _description_
        """
        return self._ref

    def lookup(self, states: typing.Dict[str, State]) -> State:
        """Access the state referred to by this ID

        Args:
            states (typing.Dict[str, State]): States that are part of the caller

        Raises:
            KeyError: The state does not exist

        Returns:
            State
        """
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


class StateMachine(Task):
    """StateMachine - Use to create 

    Args:
        Task (_type_): _description_
    """
    
    def __init__(self, start: State, states: typing.Dict[str, State], name: str=''):
        """

        Args:
            start (State): Start state
            states (typing.Dict[str, State]): Dict of states and their names
            name (str, optional): Name of the task. Defaults to ''.
        """
        self._start = start
        self._states = states
        self._name = name
    
    def reset(self):
        pass

    @abstractmethod
    def tick(self):
        raise NotImplementedError

    def iterate(self, filter: Filter=None, deep: bool=True):
        """iterate over the nodes in the state machine

        Args:
            filter (Filter, optional): Filter to use for iterating. Defaults to None.
            deep (bool, optional): Whether to iterate to sub tasks. Defaults to True.

        Yields:
            State
        """
        
        filter = filter or NullFilter()
        for state in self._states:
           if filter.check(state):
               yield state
               if deep:
                   for substate in state.iterate(filter, deep):
                       yield substate


class StateLink(object):

    def __init__(self, **state_map: typing.Dict[str, str]):

        self._state_map = state_map

    def __getitem__(self, state: State) -> str:
        return StateID(self._state_map[state.name])


class Discrete(State[V]):

    def __init__(self, name: str=''):
        """Discrete state

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


class FSM(StateMachine):
    """Finite State Machine
    """

    def __init__(self, start: Discrete, states: typing.Dict[str, Discrete], name: str=''):
        """initializer

        Args:
            start (Discrete): Start state for the FSM
            states (typing.Dict[str, Discrete]): Non-start states for the FSM
            name (str, optional): _description_. Defaults to ''.
        """
        super().__init__(start, states, name)
        self._cur_state: Discrete = self._start
        self._cur_state.reset()

    @property
    def states(self):
        return [self._start] + [*self._states.values()]
    
    def reset(self):
        """Reset the state machine"""
        
        self._cur_state = self._start
        self._start.reset()

    def tick(self):
        """Execute the state machine"""

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


class FSMState(Discrete):
    """FSM that acts as a state
    """
    
    def __init__(self, name: str, machine: FSM, state_link: StateLink):
        """_summary_

        Args:
            name (str): Name of the state
            machine (FSM): The FSM to execute
            state_link (StateLink): A link between the states to execute
        """

        super().__init__(name)
        self._machine = machine
        self._state_link = state_link

    def enter(self):
        """Enter the state machine
        """

        self._machine.reset()

    def update(self):
        """Execute the state machine
        """

        if self._machine.status == Status.DONE:
            return None

        result = self._machine.tick()
        # TODO: Determine what to use for emission
        if result.done:
            return Emission(self._state_link[self._machine.cur_state], None)

        return Emission(self, None)

    @property
    def status(self) -> Status:
        """
        Returns:
            Status: Current status of the state machine
        """
        return self._machine.status

    def iterate(self, filter: Filter=None, deep: bool=True):
        
        filter = filter or NullFilter()
        for state in self._machine.iterate(filter, deep):
           if filter.check(state):
               yield state


class TaskState(Discrete):
    """
    Allows for a task to be treated as a state
    """
    
    def __init__(self, task: Task, failure_to: StateVar, success_to: StateVar):
        """initializer

        Args:
            task (Task): Task to treat as a state
            failure_to (StateVar): Mapping from failed status to another state
            success_to (StateVar): Mapping from success status to another state
        """

        self._task = task
        self._failure_to = failure_to
        self._success_to = success_to

    def enter(self):
        """Enter the state
        """
        self._task.reset()

    def update(self) -> Status:
        """Update the 'state' by ticking the task

        Returns:
            Status
        """

        if self._task.status == Status.DONE:
            return None

        result = self._task.tick()
        if result == Status.FAILURE:
            return self._failure_to
        elif result == Status.SUCCESS:
            return self._success_to
        
        return self

    def iterate(self, filter: Filter=None, deep: bool=True):
        
        filter = filter or NullFilter()
        for task in self._task.iterate(filter, deep):
           if filter.check(task):
               yield task
