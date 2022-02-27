from .std import (
    TickDecorator, Status, PlannerDecorator, Action, Conditional, Planner,
    Parallel
)
from .vars import Args
from abc import abstractproperty
import typing
import random


class neg(TickDecorator):
    """Converts the decorated nodes status from SUCCESS to FAILURE and vice versa
    """

    def decorate_tick(self, node):
        
        tick = node.tick
        def _(*args, **kwargs):
            status = tick(*args, **kwargs)
            if status == Status.SUCCESS:
                return Status.FAILURE
            elif status == Status.FAILURE:
                return Status.SUCCESS
            return status
        return _


class fail(TickDecorator):
    """Converts the decorated nodes status to FAILURE
    """

    def decorate_tick(self, node):
        
        tick = node.tick
        def _(*args, **kwargs):
            status = tick(*args, **kwargs)
            return Status.FAILURE
        return _


class succeed(TickDecorator):
    """Converts the decorated nodes status to SUCCESS
    """

    def decorate_tick(self, node):
        
        tick = node.tick
        def _(*args, **kwargs):
            status = tick(*args, **kwargs)
            return Status.SUCCESS
        return _


class until(TickDecorator):
    """Repeats the task until the result is SUCCESS
    """

    def decorate_tick(self, node):
        
        tick = node.tick
        def _(*args, **kwargs):
            status = tick(*args, **kwargs)
            if status == Status.SUCCESS:
                return status
            elif status == Status.FAILURE:
                node.reset()
            return Status.RUNNING
        return _


class succeed_on_first(TickDecorator):
    """
    ParallelTask decorator. Returns SUCCESS if one of the
    subtasks returns success
    """

    def decorate_tick(self, node: Parallel):
        
        tick = node.tick
        def _( *args, **kwargs):
            status = tick(*args, **kwargs)
            status_total = node.status_total(Status.SUCCESS)

            if status == Status.RUNNING and status_total > 0:
                return Status.SUCCESS
            return status
        return _


class fail_on_first(TickDecorator):
    """
    ParallelTask decorator. Returns FAILURE if one of the
    subtasks returns FAILURE
    """

    def decorate_tick(self, node: Parallel):
        
        tick = node.tick
        def _(*args, **kwargs):
            status = tick(*args, **kwargs)
            status_total = node.status_total(Status.FAILURE)
            if status == Status.RUNNING and status_total > 0:
                return Status.FAILURE
            return status
        return _


class StatusMixin(object):
    """Mixin for specfifying the status of a node
    """

    @abstractproperty
    def status(self) -> Status:
        raise NotImplementedError


class Running(StatusMixin):
    """Running status mixin
    """

    @property
    def status(self) -> Status:
        return Status.RUNNING


class Ready(StatusMixin):
    """Ready status mixin
    """

    @property
    def status(self) -> Status:
        return Status.READY


class Failure(StatusMixin):
    """Failure status mixin
    """

    @property
    def status(self) -> Status:
        return Status.FAILURE


class Success(StatusMixin):
    """Success status mixin
    """

    @property
    def status(self) -> Status:
        return Status.SUCCESS


class ShufflePlanner(PlannerDecorator):

    def __init__(self, planner: Planner):
        super().__init__(planner)
        self._order = self._shuffle()
        self._idx = 0
        self._planner.idx = self._order[self._idx]

    def _shuffle(self):
        order = list(range(len(self._planner)))
        random.shuffle(order)
        return order

    def reset(self):
        self._planner.reset()
        self._order = self._shuffle()
    
    @property
    def idx(self):
        return self._idx

    @idx.setter
    def idx(self, idx: int):
        self._idx = idx
        self._planner.idx = self._order[idx]

    def adv(self):
        if self._idx == len(self._order):
            return False
        self._idx += 1
        if self._idx == len(self._order):
            self._planner.idx = len(self._order)
        else: self._planner.idx = self._order[self._idx]
        return True
    
    @property
    def cur(self):
        return self._planner.cur

    def rev(self):
        if self._idx == 0:
            return False
        self._idx -= 1
        self._planner.idx = self._order[self._idx]
        return True
    
    def __len__(self):
        return len(self._order)


class ActionFunc(Action):
    """Action that executes a function passed in
    """
    
    def __init__(self, name: str, f: typing.Callable, args: Args, status_override: Status=None):
        super().__init__(name)
        self._f = f
        self._args = args
        self._status_override = status_override
    
    def act(self):
        res = self._f(*self._args.args, **self._args.kwargs)
        if self._status_override is not None:
            return self._status_override
        return res


class ConditionalFunc(Conditional):
    """Conditional that executes a function passed in
    """
    
    def __init__(self, name: str, f: typing.Callable, args: Args):
        super().__init__(name)
        self._f = f
        self._args = args

    def check(self):
        return self._f(*self._args.args, **self._args.kwargs)
