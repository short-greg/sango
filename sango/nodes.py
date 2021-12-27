from abc import ABC, abstractmethod, abstractproperty
from dataclasses import InitVar
from enum import Enum
import typing
from functools import singledispatch
from typing import Generic, TypeVar

from sango.vars import Storage


class Status(Enum):

    FAILURE = 0
    SUCCESS = 1
    RUNNING = 2


class Task(ABC):

    @abstractmethod
    def tick(self) -> Status:
        pass

    @abstractproperty
    def cost(self) -> float:
        pass

    def __post_init__(self):
        pass


Task.__call__ = Task.tick


class Tree(Task):

    def __new__(cls):

        # 1) get all items in the tree
        # 2) make sure enter is specified
        # 3) pass resuls to __init__ method for setting the object

        pass

    def __init__(self, enter: Task, data: dict, **kwargs):

        # pass kwargs to post init

        # call set attribute on all data
        pass
    
    @abstractmethod
    def post_condition(self):
        pass

    @abstractmethod
    def pre_condition(self):
        pass

    @abstractmethod
    def __post_init__(self):
        raise NotImplementedError

    @abstractmethod
    def tick(self) -> Status:
        pass

    @abstractproperty
    def cost(self) -> float:
        pass


@singledispatch
def task(obj):
    pass


@task.register
def _(obj: Task):
    pass


@task.register
def _(obj: function):
    pass


@task.register
def _(obj: Tree):
    pass


class Composite(Task):

    @abstractproperty
    def n(self) -> int:
        pass


class Action(Task):
    
    @abstractmethod
    def tick(self):
        raise NotImplementedError


class Conditional(Task):

    @abstractmethod
    def tick(self):
        raise NotImplementedError

    @abstractmethod
    def value(self):
        pass


def vdir(obj):
    for x in dir(obj):
        if not x.startswith('__'):
            yield x, obj.__dict__[x]


# def annotations(obj):
#    return obj.__annotations__ if __annotations__ in obj else {}

def _func():
    pass


def is_task(annotation: typing.Type, val):
    return isinstance(val, Task) or issubclass(annotation, Task) or issubclass(val, Task):


def vals(cls):

    annotations = cls.__annotations__

    for var in [x for x in dir(cls) if not x.startswith('__')]:
        annotation = annotations.get(var, None)
        val = cls.__dict__[var]
        is_task_ = is_task(annotation, val)
        if is_task_:
            yield var, annotation, val, True
        yield var, annotation, val, False


def to_task(val) -> Task:

    pass

# def data(cls):
#     for var in [x for x in dir(cls) if not x.startswith('__')]:
#         pass


T = TypeVar('T')

class StoreVar(Generic[T]):
    pass


class Composite(Task):

    @property
    def n(self):
        return len(self._tasks)
    
    @classmethod
    def __new__(cls, *args, **kwargs):

        obj: Composite = super().__new__(cls)
        
        tasks = []
        data = {}
        init_vars = {}
        for i, (name, type_, value, is_task) in enumerate(vals(cls)):
            
            if i < len(args):
                val = args[i]
            else:
                val = kwargs.get(name, value)
            if is_task:
                tasks[name] = to_task(val)
            elif type_ is not None:
                if issubclass(type_, InitVar):
                    init_vars[name] = val
                elif issubclass(type_, StoreVar):
                    data[name] = val
                else:
                    obj.__setattr__(name, val)

        obj._store = Storage(**data)
        obj._tasks = tasks
        obj._n = len(obj._tasks)
        obj.__post_init__(**init_vars)

        return obj


def storage(task: Task):
    return task.__storage__


def tasks(composite: Composite):
    return composite.__tasks__


class Sequence(Composite):

    # TODO: Determine how to work in the planner
    def __init__(self):
        self._cur = None
        self._cur_plan: typing.Iterable = None

    def reset(self):
        self._cur_plan: typing.Iterable = None
    
    def _plan(self):
        return self._tasks

    def tick(self):
        
        if self._cur_plan is None:
            self._cur_plan = self._plan()
        
        i = self._plan.adv()
        if i is None:
            return Status.SUCCESS
        status = self._tasks[i].tick()
        if status == Status.FAILURE:
            return Status.FAILURE
        
        if self._plan.end():
            return Status.SUCCESS
        return Status.RUNNING


class Planner(object):
    
    def __init__(self):
        pass

    def adv(self):
        pass

    def end(self):
        pass


class LinearPlanner(object):

    def __init__(self, items: list):

        self._items = None
        self._iter = None
        self._ended = False
        self._cur = None
        self.reset(items)
    
    def reset(self, items: list=None):

        self._items = items if items is not None else self._items
        self._iter = iter(self._items)
        try:
            self._cur = iter(self._items)
            self._ended = False
        except StopIteration:
            self._cur = None
            self._ended = True    

    def end(self):
        if self._ended:
            return True
        return False
    
    def adv(self):
        cur = self._cur
        try:
            self._cur = next(self._iter)
        except StopIteration:
            self._cur = None
            self._ended = True    
        return cur
    

class SequenceTicker(ABC):

    @abstractmethod
    def tick(self):
        raise NotImplementedError


class RandomSequenceTicker(SequenceTicker):

    def _plan(self):
        pass


class Fallback(Composite):

    def _plan(self):
        return Plan(self._tasks)

    def tick(self):
        
        if self._cur_plan is None:
            self._cur_plan = self._plan()
        
        i = self._plan.adv()
        if i is None:
            return Status.SUCCESS
        status = self._tasks[i].tick()
        if status == Status.SUCCESS:
            return Status.SUCCESS
        
        if self._plan.end():
            return Status.FAILURE
        return Status.RUNNING


class Parallel(Composite):

    def __init__(self):
        super().__init__()
        self._n_running = None
        self._succeeded = None
        self._failed = None

    def reset(self):
        pass

    def n_running(self):
        return self._n_running

    def n_successes(self) -> int:
        return self._succeeded

    def n_failures(self) -> int:
        # need to add
        return self._failed
    
    def tick(self):
        success = True
        finished = True
        n_running = 0
        if self._succeeded is None:
            self._succeeded = 0
            self._failed = 0

        for task in self._tasks:
            status = task.tick()
            if status != Status.SUCCESS:
                success = False
            if status == Status.RUNNING:
                finished = False
                n_running += 1
            if status == Status.SUCCESS:
                self._succeeded += 1
            if status == Status.FAILURE:
                self._failed += 1
        
        self._n_running = n_running
        if finished is False:
            return Status.RUNNING 
            
        if success:
            return Status.SUCCESS
        return Status.FAILURE
        

def success_on_first(parallel: Parallel):

    def tick(self):
        result = parallel.tick()
        if result == Status.SUCCESS and parallel.n_successes > 0:
            return Status.SUCCESS
        return result

    parallel.tick = tick


def fail_on_first(parallel: Parallel):

    def tick(self):
        result = parallel.tick()
        if result == Status.FAILURE and parallel.n_failures > 0: 
            return Status.FAILURE
        return result

    parallel.tick = tick


class Fallback(Task):

    def __init__(self, nodes: typing.List[Task]):

        self._nodes = nodes
        self._cur: int = None

    def __choose__(self) -> int:
        pass

    def tick(self):
        
        # TODO: Find this out
        cur = self.__choose__()

        status = self._nodes[self._cur].tick()
        if status == Status.SUCCESS:
            return Status.SUCCESS
        return Status.RUNNING


# need to think about how to do this more
# class State(Enum):
#     pass


# use for post conditioxn / pre condition
# class Condition:

#     def check(self, states: typing.List[State]):
#         pass


class Decorator(Task):

    def __init__(self, node: Task):
        self._node = node
    
    def wrapped(self):
        return self._node
    
    @abstractproperty
    def wrapped(self):
        pass


class Neg(Decorator):

    def tick(self):
        status = self._node.tick()
        if status == Status.SUCCESS:
            return Status.FAILURE
        elif status == Status.FAILURE:
            return Status.SUCCESS
        return Status.RUNNING


class Fail(Decorator):

    def tick(self):
        self._node.tick()
        return Status.FAILURE


class Success(Decorator):

    def tick(self):
        self._node.tick()
        return Status.SUCCESS


class Until(Decorator):

    def tick(self):

        status = self._node.tick()
        if status == Status.SUCCESS:
            return status
        return Status.RUNNING


def neg(node: Task):
    
    neg_ = Neg(node)
    
    node.tick = neg_.tick


def fail(node: Task):
    
    fail_ = Fail(node)
    node.tick = fail_.tick


def success(node: Task):
    
    success_ = Success
    node.tick = success_.tick


def until(node: Task):
    
    until_ = Until(node)
    node.tick = until_.tick


