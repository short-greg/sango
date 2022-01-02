from abc import ABC, abstractmethod, abstractproperty
from enum import Enum
import typing
from functools import partial, singledispatch, singledispatchmethod
from typing import Any, Iterator
from sango.vars import UNDEFINED, HierarchicalStorage, Storage
from .vars import AbstractStorage, Args, HierarchicalStorage, Ref, StoreVar, Var
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


class TaskMeta(type):

    def _update_var_stores(cls, kw):
        
        var_stores = ClassArgFilter([TypeFilter(VarStorer)]).filter(cls)
        if 'store' in kw:
            store = kw.get('store')
            del kw['store']
        else:
            store = HierarchicalStorage(Storage())
        
        for name, storer in var_stores.items():
            if name in kw:
                storer(kw[name])
                del kw[name]
            
            store[name] = storer.value
        return store


class Task(object, metaclass=TaskMeta):

    def __init__(self, name: str=''):
        self._name = name
        self._cur_status = Status.READY
    
    def __pre_init__(self, store: Storage):
        self._store = store

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


Task.__call__ = Task.tick


def _func():
    pass


class AtomicMeta(TaskMeta):

    def __call__(cls, *args, **kw):

        self = cls.__new__(cls, *args, **kw)
        store = cls._update_var_stores(kw)
        cls.__pre_init__(self, store)
        cls.__init__(self, *args, **kw)
        return self
    

class Atomic(Task, metaclass=AtomicMeta):
    
    def __pre_init__(self, store: Storage):

        self._store = store


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
        store = cls._update_var_stores(kw)
        tasks = cls._load_tasks(store, kw)
        cls.__pre_init__(self, tasks, store)
        cls.__init__(self, *args, **kw)
        return self


class Composite(Task, metaclass=CompositeMeta):

    def __init__(
        self, name: str='', planner: Planner=None
    ):
        super().__init__(name)
        self._planner = planner or LinearPlanner(self._tasks)

    def __pre_init__(self, tasks: typing.List[Task], store: Storage=None):
        super().__pre_init__(store)
        self._tasks = tasks

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


class TreeMeta(TaskMeta):

    def _load_entry(cls, store, kw):
        entry = ClassArgFilter([TypeFilter(TaskLoader)]).filter(cls)['entry']
        if entry in kw:
            entry(kw['entry'])
            del kw['entry']
        entry = entry.load(store, 'entry')
        return entry

    def __call__(cls, *args, **kw):

        self = cls.__new__(cls, *args, **kw)
        store = cls._update_var_stores(kw)
        entry = cls._load_entry(store, kw)
        cls.__pre_init__(self, entry, store)
        cls.__init__(self, *args, **kw)
        return self


class Tree(Task, metaclass=TreeMeta):

    def __pre_init__(self, entry: Task, store: Storage):
        super().__pre_init__(store)
        self.entry = entry
    
    def tick(self) -> Status:        
        return self.entry.tick()


class Action(Atomic):

    @abstractmethod
    def act(self):
        raise NotImplementedError

    def tick(self):
        if self._cur_status.done:
            return Status.DONE
        self._cur_status = self.act()
        return self._cur_status


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


class Loader(object):

    def __init__(self, cls: typing.Type=UNDEFINED, args: Args=None, decorators=None):

        self._item_cls = cls
        self._args = args or Args()
        self._decorators: typing.List = decorators or []
    
    def load(self, storage: Storage, name: str=''):
        storage = HierarchicalStorage(Storage(), storage)
        item_factory = self._item_cls
        for decorator in self._decorators:
            item_factory = decorator(item_factory)
        
        item = item_factory(
            store=storage, name=name, *self._args.args, **self._args.kwargs
        )
        return item
    
    def add_decorator(self, decorator):
        self._decorators.append(decorator)
        
    def add_decorators(self, decorators):
        self._decorators.extend(decorators)

    def __call__(self, task: Task):
        self._task = task
        return self



class TaskLoader(Loader):

    def __init__(self, task_cls: typing.Type[Task]=UNDEFINED, args: Args=None, decorators=None):
        super().__init__(task_cls, args, decorators)

    def __call__(self, task: Task):
        self._task = task
        return self


def decorate(loader: Loader, decorators=None):
    loader.add_decorators(decorators)


def task_(loader: TaskLoader=None, *args, **kwargs):
    return TaskLoader(loader, Args(*args, **kwargs))


def task(*args, **kwargs):
    return TaskLoader(args=Args(*args, **kwargs))


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
        self, name: str='', planner: Planner=None
    ):
        super().__init__(name=name, planner=planner)
        self._statuses = []
    
    def status_total(self, status: Status):
        total = 0
        for s in self._statuses:
            if s == status:
                total += 1
        return total
        # return functools.reduce(lambda x, y: x + (1 if y == status else 0), self._statuses)

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


def neg(node: Task):

    def tick(node: Task, wrapped_tick):
        status = wrapped_tick()
        if status == Status.SUCCESS:
            return Status.FAILURE
        elif status == Status.FAILURE:
            return Status.SUCCESS
        return status
    
    return TickDecorator(node, tick)


def fail(node: Task):

    def tick(node: Task, wrapped_tick):
        wrapped_tick()
        return Status.FAILURE
    
    return TickDecorator(node, tick)


def succeed(node: Task):

    def tick(node: Task, wrapped_tick):
        wrapped_tick()
        return Status.SUCCESS
    
    return TickDecorator(node, tick)


def until(node: Task):

    def tick(node: Task, wrapped_tick):
        status = wrapped_tick()
        if status == Status.SUCCESS:
            return status
        elif status == Status.FAILURE:
            node.reset()
        return Status.RUNNING
    
    return TickDecorator(node, tick)


def succeed_on_first(node: Parallel):

    def tick(node: Parallel, wrapped_tick):
        status = wrapped_tick()

        status_total = node.status_total(Status.SUCCESS)

        if status == Status.RUNNING and status_total > 0:
            return Status.SUCCESS
        return status
    
    return TickDecorator(node, tick)


def fail_on_first(node: Parallel):

    def tick(node: Parallel, wrapped_tick):
        status = wrapped_tick()
        status_total = node.status_total(Status.FAILURE)
        if status == Status.RUNNING and status_total > 0:
            return Status.FAILURE
        return status
    
    return TickDecorator(node, tick)


class TickDecorator(object):

    def __init__(self, node_cls: typing.Type[Task], tick):
        self._node_cls = node_cls
        self._tick = tick
    
    def __call__(self, *args, **kwargs):
        node = self._node_cls(*args, **kwargs)
        node.tick = wraps(node.tick)(partial(self._tick, node, node.tick))
        return node


def loads(decorator):
    def _(loader: TaskLoader):

        loader.add_decorator(decorator)
        return loader
    return _
