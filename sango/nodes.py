from abc import ABC, abstractmethod, abstractproperty
from enum import Enum
import typing
from functools import singledispatch


class Status(Enum):

    Failure = 0
    Success = 1
    Running = 2


class Task(ABC):

    @abstractmethod
    def tick(self) -> Status:
        pass

    @abstractproperty
    def cost(self) -> float:
        pass


Task.__call__ = Task.tick


class Tree(ABC):

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
    pass


class Var(object):

    @abstractmethod
    def value(self):
        pass


class Conditional(Task, Var):

    @abstractmethod
    def value(self):
        pass




# need to think about how to do this more
class State(Enum):
    pass



# use for post condition / pre condition
class Condition:

    def check(self, states: typing.List[State]):
        pass


class Shared(Var):

    def __init__(self, var):

        self._var = var

    @abstractmethod
    def value(self):
        return self._var.value


class Decorator(Task):

    @abstractproperty
    def wrapped(self):
        pass


class Not(Decorator):

    def __init__(self, node: Task):

        self._node = node
    
    def wrapped(self):
        return self._node

    def tick(self):
        return not self._node.tick()


def not_(node: Task):
    return Not(node)



