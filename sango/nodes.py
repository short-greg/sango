from abc import ABC, abstractmethod, abstractproperty
from enum import Enum
import functools
import itertools
import typing
from functools import partial, singledispatch, singledispatchmethod
from typing import Any, Generic, Iterator, TypeVar
from sango.vars import UNDEFINED, ConditionSet, HierarchicalStorage, Storage
from .vars import AbstractStorage, Args, HierarchicalStorage, NullStorage, Ref, Shared, StoreVar, Var
from .utils import coalesce
import random
from functools import wraps


class Status(Enum):

    FAILURE = 0
    SUCCESS = 1
    RUNNING = 2
    READY = 3
    DONE = 4
    NONE = 5

    @property
    def done(self):
        return self == Status.FAILURE or self == Status.SUCCESS or self == Status.DONE


# TODO: Probably remove.. I don't think I need this now
def is_task(annotation: typing.Type, val):
    if annotation is None:
        return False
    return isinstance(val, TaskLoader)


def vals(cls):

    try:
        annotations = cls.__annotations__
    except AttributeError:
        annotations = {}
    d  = getattr(cls, '__dict__', {})

    for var in [x for x in d.keys() if not x.startswith('__')]:
        annotation = annotations.get(var, None)
        val = getattr(cls, var)
        is_task_ = is_task(annotation, val)
        if is_task_:
            yield var, annotation, val, True
        yield var, annotation, val, False


@singledispatch
def _update_var(val, storage: AbstractStorage):
    return Var[Any](val)


@_update_var.register
def _(val: Ref, storage: AbstractStorage):
    return val.shared(storage)


@_update_var.register
def _(val: StoreVar, storage: AbstractStorage):
    return val


class ArgFilter(ABC):

    @abstractmethod
    def filter(self, name, annotation, value):
        raise NotImplementedError


class TypeFilter(ArgFilter):

    def __init__(self, arg_type: typing.Type):
        self._arg_type = arg_type
    
    def filter(self, name: str, annotation: typing.Type, value):
        return isinstance(value, self._arg_type)


class ClassArgFilter(object):

    def __init__(self, filters: typing.List[ArgFilter]):
        
        self._filters = filters

    def _run_filters(self, name, type_, value):

        for filter in self._filters:
            if filter.filter(name, type_, value):
                return True

        return False

    def filter(self, cls):
        result_kwargs = {}
        for (name, type_, value, is_task) in vals(cls):
            if self._run_filters(name, type_, value):
                result_kwargs[name] = value
        return result_kwargs


class TaskMeta(type):

    def _update_var_stores(cls, kw):
        
        var_stores = ClassArgFilter([TypeFilter(VarStorer)]).filter(cls)
        store = kw.get('store')
        if store is None:
            store = HierarchicalStorage(Storage())
        
        for name, storer in var_stores.items():
            if name in kw:
                storer(kw[name])
                del kw[name]
            
            store.add(name, storer.value)
        return store

    # def __call__(cls, *args, **kw):
    #     self = cls.__new__(cls, *args, *kw)
    #     cls.__init__(self, *args, **kw)
    #     cls.__post_init__(self, *args, **kw)
    #     return self


class AtomicMeta(TaskMeta):

    def __call__(cls, *args, **kw):

        self = cls.__new__(cls, *args, **kw)
        kw['store'] = cls._update_var_stores(kw)
        cls.__init__(self, *args, **kw)
        return self
    

class Task(object, metaclass=TaskMeta):

    def __init__(self, store: Storage=None, name: str=''):
        self._store = store
        self._name = name
        self._cur_status = Status.READY
    
    def reset(self):
        self._cur_status = Status.READY
    
    @abstractmethod
    def tick(self) -> Status:
        raise NotImplementedError

    @property
    def status(self) -> Status:
        return self._cur_status
    
    @property
    def name(self) -> str:
        return self._name

    def __getattribute__(self, key: str) -> Any:
        try:
            store: HierarchicalStorage = super().__getattribute__('_store')
            print(store)

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


Task.__call__ = Task.tick


def _func():
    pass


class Atomic(Task, metaclass=AtomicMeta):
    pass


class Planner(ABC):
    
    def __init__(self):
        pass

    def adv(self):
        pass

    def end(self):
        pass
    
    def reset(self):
        pass

    @abstractproperty
    def cur(self):
        pass


class LinearPlanner(object):

    def __init__(self, items: typing.List[Task]):
        self._items = items
        self._idx = 0

    def reset(self, items: list=None):
        self._idx = 0
        self._items = coalesce(items, self._items)
        
    def idx(self):
        return self._idx

    def end(self):
        if self._idx == len(self._items):
            return True
        return False
    
    def adv(self):
        if self._idx == len(self._items):
            return False
        self._idx += 1
        return True
    
    @property
    def cur(self):
        if self._idx == len(self._items):
            return None
        return self._items[self._idx]

    def rev(self):
        if self._idx == 0:
            return False
        self._idx -= 1
        return True


def iterate_planner(planner: Planner) -> Iterator[Task]:

    planner.reset()
    
    while planner.end() is False:
        yield planner.cur
        planner.adv()


def shuffle(linear: LinearPlanner):

    old_reset = linear.reset
    def shuffle(items):
        if items is not None:
            items = [*items]
            random.shuffle(items)
        return items

    def reset(self, items: list=None):
        old_reset(self, shuffle(items))

    linear.reset = reset
    linear.reset()
    return linear


class CompositeMeta(TaskMeta):

    def _load_tasks(cls, store, kw):
        tasks = []
        for name, loader in ClassArgFilter([TypeFilter(TaskLoader)]).filter(cls).items():
            if name in kw:
                loader(kw[name])
                del kw[name]
            tasks.append(loader.load(store, name))
        return tasks

    def __call__(cls, *args, **kw):
        self = cls.__new__(cls, *args, **kw)
        kw['store'] = cls._update_var_stores(kw)
        kw['tasks'] = cls._load_tasks(kw['store'], kw)
        cls.__init__(self, *args, **kw)
        return self


class Composite(Task, metaclass=CompositeMeta):

    def __init__(
        self, tasks: typing.List[Task], store: Storage=None, name: str='', planner: Planner=None
    ):
        super().__init__(store, name)
        self._tasks = tasks
        self._planner = planner or LinearPlanner(tasks)

    @property
    def n(self):
        return len(self._tasks)
    
    @property
    def tasks(self):
        return list(**self._tasks)

    @abstractmethod
    def subtick(self) -> Status:
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
        for task in self._tasks:
            task.reset()


class TreeMeta(TaskMeta, metaclass=TaskMeta):

    def _load_entry(cls, store, kw):
        entry = ClassArgFilter([TypeFilter(TaskLoader)]).filter(cls)['entry']
        if entry in kw:
            entry(kw['entry'])
        entry = entry.load(store, 'entry')
        print('Entry: ', entry)
        return entry

    def __call__(cls, *args, **kw):

        self = cls.__new__(cls, *args, **kw)
        kw['store'] = cls._update_var_stores(kw)
        kw['entry'] = cls._load_entry(kw['store'], kw)
        cls.__init__(self, *args, **kw)
        return self


class Tree(Task, metaclass=TreeMeta):

    def __init__(self, entry: Task, store: Storage=None, name: str=''):
        super().__init__(store, name)
        self.entry = entry

    def tick(self) -> Status:        
        return self.entry.tick()


class Action(Atomic):

    @abstractmethod
    def act(self):
        raise NotImplementedError

    def tick(self):
        raise NotImplementedError


@singledispatch
def action(action, args: Args):
    
    def _(store: Storage):
        return action(*args.args, **args.kwargs, _store=store)

    return _


@action.register
def _(args: Args):
    
    def _(cls):
        x = cls
        def __new__(cls, store: Storage):
            
            x.__new__(*args.args, **args.kwargs, _store=store)

        cls.__new__ = __new__

    return _
 

class Conditional(Atomic):

    @abstractmethod
    def check(self) -> bool:
        raise NotImplementedError

    def tick(self):
        if self._cur_status.done:
            return Status.DONE
        self._cur_status = Status.SUCCESS if self.check() else Status.FAILURE
        return self._cur_status


# class ConditionalVar(Task, Var)

# action(Args(Ref(''), Ref('')))

# class ActionFunc(Action):

#     func = InitVar()

#     def __post_init__(self, func):
#         return super().__post_init__()


class TaskLoader(object):

    def __init__(self, task: typing.Type[Task]=UNDEFINED, args: Args=None, decorators=None):

        self._task = task
        self._args = args or Args()
        self._decorators = decorators or []
    
    def load(self, storage: Storage, name: str=''):
        storage = HierarchicalStorage(Storage(), storage)
        task = self._task(
            store=storage, name=name, *self._args.args, **self._args.kwargs
        )
        for decorator in self._decorators:
            task = decorator(task)
        return task
    
    def add_decorator(self, decorator):
        self._decorators.append(decorator)

    def __call__(self, task: Task):
        self._task = task
        return self


@singledispatch
def task(t=None, args: Args=None, decorators=None):
    return TaskLoader(t, args, decorators=decorators)


# TODO: DEFINE THIS.
# @task.register
# def _(t: function, args: Args=None, decorators=None):
#     return TaskLoader(t, args, decorators=decorators)


@task.register
def _(args: Args, decorators=None):
    return TaskLoader(args=args, decorators=decorators)


class VarStorer(object):

    def __init__(self, val):

        if isinstance(val, Var):
            self._val = val
        else:
            self._val = Var(val)

    @property
    def value(self):
        return self._val

    @singledispatchmethod
    def __call__(self, val):

        self._val = Var(val)
        return self

    @__call__.register
    def _(self, val: Ref):
        self._val = val
        return self

    @__call__.register
    def _(self, val: Var):
        self._val = val
        return self


# TODO: Error handling if passing a ref to a regular storage



def var(val=UNDEFINED):    
    return VarStorer(Var(val))


class Sequence(Composite):

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
        
        self._planner.adv()
        if self._planner.end():
            return Status.SUCCESS
        return Status.RUNNING


class Fallback(Composite):

    def subtick(self) -> Status:

        if self._planner.end() is True:
            return Status.NONE
        
        status = self._planner.cur.tick()
        if status == Status.SUCCESS:
            return Status.SUCCESS
        
        self._planner.adv()
        if self._planner.end():
            return Status.FAILURE
        return Status.RUNNING


class Parallel(Composite):

    def __init__(
        self, tasks, store: Storage
    ):
        super().__init__(tasks, store)
        self._statuses = []
    
    def status_total(self, status: Status):
        return functools.reduce(lambda x, y: x + (1 if y == status else 0), self._statuses)

    def reset(self):
        super().reset()
        self._statuses = []

    def subtick(self):

        for i, task in enumerate(iterate_planner(self._planner)):
            if i > len(self._statuses) - 1:
                self._statuses.append(Status.RUNNING)
            elif self._statuses[i] != Status.RUNNING:
                continue
            self._statuses.append(task.tick())
        
        if Status.RUNNING in self._statuses:
            return Status.RUNNING 

        return Status.SUCCESS if Status.FAILURE not in self._statuses else Status.FAILURE


def neg(node: Task):

    old_tick = node.tick
    @wraps(node.tick)
    def tick(self):
        status = old_tick(self)
        if status == Status.SUCCESS:
            return Status.FAILURE
        elif status == Status.FAILURE:
            return Status.SUCCESS
        return status
    
    node.tick = tick
    return node


def fail(node: Task):

    old_tick = node.tick
    @wraps(node.tick)
    def tick(self):
        old_tick(self)
        return Status.FAILURE
    
    node.tick = tick
    return node


def succeed(node: Task):

    old_tick = node.tick
    @wraps(node.tick)
    def tick(self):
        old_tick(self)
        return Status.SUCCESS
    
    node.tick = tick
    return node


def until(node: Task):

    old_tick = node.tick
    @wraps(node.tick)
    def tick(self):
        status = old_tick(self)
        if status == Status.SUCCESS:
            return status
        return Status.RUNNING
    
    node.tick = tick
    return node


def succeed_on_first(node: Parallel):

    old_tick = node.tick
    @wraps(node.tick)
    def tick(self):
        status = old_tick(self)
        if status == Status.SUCCESS and node.status_total(Status.SUCCESS) > 0:
            return Status.SUCCESS
        return status
    
    node.tick = tick
    return node


def succeed_on_first(node: Parallel):

    old_tick = node.tick
    @wraps(node.tick)
    def tick(self):
        status = old_tick(self)
        if status == Status.FAILURE and node.status_total(Status.FAILURE) > 0:
            return Status.FAILURE
        return status
    
    node.tick = tick
    return node


def loads(decorator):
    def _(loader: TaskLoader):

        loader.add_decorator(decorator)
        return loader
    return _


# class t(Tree):
    
#     class entry(Sequence):
#         
#         # TODO: Add in meta
#         class meta:
#           postcondition = ConditionSet()
#           cost
#         x: Action = task(act, ref(""), ref(""))
#         y: Action = task(act, ref(""), ref(""))
#         x: Conditional = UNDEFINED
#         xx: Conditional = False
        
#         @action
#         def z(storage: Storage):
#             pass
            
#         class y(Action):
#             pass


# def vdir(obj):
#     for x in dir(obj):
#         if not x.startswith('__'):
#             yield x, obj.__dict__[x]


# def ttt(cls):

#     __old_init__ = cls.__init__
#     def __init__(self):
#         self.x = 1
#         __old_init__(self)
    
#     cls.__init__ = __init__
#     return cls

# @ttt
# class Y:
#     pass

# class Fallback(Task):

#     def __init__(self, nodes: typing.List[Task]):

#         self._nodes = nodes
#         self._cur: int = None

#     def subtisk(self):

#         status = self._nodes[self._cur].tick()
#         if status == Status.SUCCESS:
#             return Status.SUCCESS
#         return Status.RUNNING


# class Decorator(Task):

#     def __init__(self, node: Task):
#         self._node = node
    
#     def wrapped(self):
#         return self._node

#     @abstractmethod
#     def tick(self):
#         raise NotImplementedError


# class Neg(Decorator):

#     def tick(self):
#         status = self._node.tick()
#         if status == Status.SUCCESS:
#             return Status.FAILURE
#         elif status == Status.FAILURE:
#             return Status.SUCCESS
#         return Status.RUNNING


# class Fail(Decorator):

#     def tick(self):
#         self._node.tick()
#         return Status.FAILURE


# class Success(Decorator):

#     def tick(self):
#         self._node.tick()
#         return Status.SUCCESS


# class Until(Decorator):

#     def tick(self):

#         status = self._node.tick()
#         if status == Status.SUCCESS:
#             return status
#         return Status.RUNNING


# class DecoratorWrapper(ABC):

#     @abstractmethod
#     def tick(self, node: Task):
#         pass

#     @singledispatchmethod
#     def __call__(self, node: Task):
#         tick = partial(self.tick, node.tick)
#         node.tick = tick
#         return node

#     @__call__.register
#     def _(self, node: TaskLoader):
#         node.add_decorator(self)
#         return node

# T = typing.Union[TaskLoader, Task]


# @singledispatch
# def process_decorator(node: Task, tick):
#     # old_tick = node.tick
#     node.tick = tick
#     return node

# @process_decorator.register
# def _(node: TaskLoader, tick):
#     node.add_decorator(tick)

