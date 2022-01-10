from abc import ABC, abstractmethod, abstractproperty
from enum import Enum
import typing
from functools import partial, singledispatch, singledispatchmethod
from typing import Any, Iterator, Type
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


def ref_is_external(task_cls, default=True):
    
    if not isinstance(task_cls, Type): return True
    if not issubclass(task_cls, Tree): return True

    return getattr(task_cls, '__external_ref__', default)


class TaskMeta(type):

    def _update_var_stores(cls, kw):
        
        var_stores = ClassArgFilter([TypeFilter(VarStorer)]).filter(cls)
        if 'store' in kw:
            store = kw['store']
            del kw['store']
        else:
            store = HierarchicalStorage(Storage())
        
        for name, storer in var_stores.items():
            if name in kw:
                storer(kw[name])
                del kw[name]
            
            store[name] = storer.value
        return store

    def _get_reference(cls, kw):
        if 'reference' in kw:
            reference = kw['reference']
            del kw['reference']
            return reference


class Task(object, metaclass=TaskMeta):

    def __init__(self, name: str=''):
        self._name = name
        self._cur_status: Status = Status.READY
    
    def __pre_init__(self, store: Storage, reference):
        self._store = store
        self._reference = reference

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
        reference = cls._get_reference(kw)
        cls.__pre_init__(self, store, reference)
        cls.__init__(self, *args, **kw)
        return self
    

class Atomic(Task, metaclass=AtomicMeta):
    
    def __pre_init__(self, store: Storage, reference):
        super().__pre_init__(store, reference)


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
        reference = cls._get_reference(kw)
        cls.__pre_init__(self, tasks, store, reference)
        cls.__init__(self, *args, **kw)
        return self


class Composite(Task, metaclass=CompositeMeta):

    def __init__(
        self, name: str='', planner: Planner=None
    ):
        super().__init__(name)
        self._planner = planner or LinearPlanner(self._tasks)

    def __pre_init__(self, tasks: typing.List[Task], store: Storage, reference):
        super().__pre_init__(store, reference)
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

    def _load_entry(cls, store, reference, kw):
        entry = ClassArgFilter([TypeFilter(TaskLoader)]).filter(cls)['entry']
        if entry in kw:
            entry(kw['entry'])
            del kw['entry']
        entry = entry.load(store, 'entry', reference)
        return entry
    
    def _get_reference(cls, self, kw, external_default=True):

        if not ref_is_external(cls, external_default):
            if 'reference' in kw:
                raise ValueError(f'Must not define ref object for tree {cls.__qualname__} because ref is external')

            return self
        if 'reference' not in kw:
            raise ValueError(f'Must pass in reference object to tree {cls.__qualname__} with external ref')
        
        reference = kw['reference']
        del kw['reference']

        if reference is None:
            raise ValueError(f'Value of None is not valid for reference object for tree {cls.__qualname__}')

        return reference

    def __call__(cls, *args, **kw):

        self = cls.__new__(cls, *args, **kw)
        store = cls._update_var_stores(kw)

        reference = cls._get_reference(self, kw, False)
        entry = cls._load_entry(store, reference, kw)
        cls.__pre_init__(self, entry, store, reference)
        cls.__init__(self, *args, **kw)
        return self


class Tree(Task, metaclass=TreeMeta):

    def __pre_init__(self, entry: Task, store: Storage, reference):
        super().__pre_init__(store, reference)
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


class Conditional(Atomic):

    @abstractmethod
    def check(self) -> bool:
        raise NotImplementedError

    def tick(self):
        if self._cur_status.done:
            return Status.DONE
        self._cur_status = Status.SUCCESS if self.check() else Status.FAILURE
        return self._cur_status


class DecoratorLoader(object):

    def __call__(self, loader):
        pass

    def __lshift__(self, other):
        pass

    def append(self, other):
        pass

    def prepend(self, other):
        pass

    def decorate(self, item):
        pass


class Loader(object):

    def __init__(self, cls: typing.Type=UNDEFINED, args: Args=None):

        self._cls = cls
        self._args = args or Args()
        self.decorator: DecoratorLoader = None
    
    def load(self, storage: Storage, name: str='', reference=None):
        storage = HierarchicalStorage(Storage(), storage)
        if self._cls is UNDEFINED:
            raise ValueError(f"Cls to load for {type(self).__name__} has not been defined")
        
        kwargs = {}
        if not ref_is_external(self._cls):
            kwargs['reference'] = reference

        item = self._cls(
            store=storage, name=name, *self._args.args, **self._args.kwargs, **kwargs
        )
        if self.decorator is not None:
            item = self.decorator.decorate(item)
        return item
    
    def add_decorator(self, decorator, prepend=True):

        if self.decorator is None:
            self.decorator = decorator
        elif prepend:
            self.decorator = self.decorator.prepend(decorator)
        else:
            self.decorator = self.decorator.append(decorator)

    def __call__(self, cls: typing.Type):
        self._cls = cls
        return self

    def __lshift__(self, decorator: DecoratorLoader):

        self.add_decorator(decorator)
        return self


class TaskLoader(Loader):

    pass

    # def __init__(self, task_cls: typing.Type[Task]=UNDEFINED, args: Args=None, decorators=None):
    #     super().__init__(task_cls, args, decorators)

    # def __call__(self, cls: typing.Type[Task]):
    #     self._item_cls = cls
    #     return self


def task(cls: typing.Type[Task]):
    return TaskLoader(cls)


def task_(cls: typing.Type[Task], *args, **kwargs):
    return TaskLoader(cls, args=Args(*args, **kwargs))


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


class TickDecorator(object):

    def __init__(self, node: typing.Type[Task]=None):
        self._node = node

    def decorate_tick(self, node):
        raise NotImplementedError
    
    def decorate(self, node: Task):
        node.tick = wraps(node.tick)(self.decorate_tick(node))
        return node

    def __call__(self, *args, **kwargs):

        if self._node is None:
            raise AttributeError(f'Member node has not been instantiated')
        
        return self.decorate(self._node(*args, **kwargs))


class TickDecorator2nd(TickDecorator):

    def __init__(self, *args, **kwargs):
        super().__init__()
        
        self._args = args
        self._kwargs = kwargs

    def decorate(self, node: Task):
        node.tick = wraps(node.tick)(self.decorate_tick(node))
        return node
    
    def __call__(self, node_cls: typing.Type[Task]):
        
        def instantiator(*args, **kwargs):

            node = node_cls(*args, **kwargs)
            return self.decorate(node)
        
        return instantiator


class TaskDecorator(Task):
    
    def __init__(self, task: Task):
        super().__init__('')
        self._task = task
    
    @abstractmethod
    def decorate(self):
        raise NotImplementedError

    def tick(self):

        if self._cur_status.done:
            return Status.DONE

        return self.decorate()


class DecoratorSequenceLoader(DecoratorLoader):

    def __init__(self, decorators: typing.List[DecoratorLoader]):

        super().__init__()
        self._decorators = decorators

    @singledispatchmethod
    def __call__(self, loader):
        return self.prepend(loader)

    @__call__.register
    def _(self, loader: Loader):
        loader.add_decorator(self)
        return loader
    
    def __lshift__(self, other):
        return self.prepend(other)

    def append(self, other):
        return DecoratorSequenceLoader.from_pair(self, other)

    def prepend(self, other):
        return DecoratorSequenceLoader.from_pair(other, self)

    def decorate(self, item):
        
        for decorator in reversed(self._decorators):
            item = decorator.decorate(item)
        return item
    
    @property
    def decorators(self):
        return [*self._decorators]
    
    @classmethod
    def from_pair(cls, first, second):

        loaders = []
        if isinstance(first, DecoratorSequenceLoader):
            loaders.extend(first._decorators)
        else:
            loaders.append(first)
        
        if isinstance(second, DecoratorSequenceLoader):
            loaders.extend(second._decorators)
        else:
            loaders.append(second)
        return DecoratorSequenceLoader(loaders)


class AtomicDecoratorLoader(DecoratorLoader):

    @singledispatchmethod
    def __call__(self, loader):
        raise f'__call__ not defined for type {type(loader)}'

    @__call__.register
    def _(self, loader: DecoratorLoader):
        return self.prepend(loader)

    @__call__.register
    def _(self, loader: Loader):
        loader.add_decorator(self)
        return loader
    
    def __lshift__(self, other):
        return self.prepend(other)

    def append(self, other):
        return DecoratorSequenceLoader.from_pair(self, other)

    def prepend(self, other):
        return DecoratorSequenceLoader.from_pair(other, self)

    def decorate(self, item):
        raise NotImplementedError


class TaskDecoratorLoader(AtomicDecoratorLoader):

    def __init__(self, decorator_cls: typing.Type[TaskDecorator], name: str=None):

        super().__init__()
        self._decorator_cls = decorator_cls
        self._name = name or self._decorator_cls.__name__
    
    @property
    def decorator_cls(self):
        return self._decorator_cls

    def decorate(self, item):
        return self._decorator_cls(self._name, item)


class TickDecoratorLoader(AtomicDecoratorLoader):

    def __init__(self, decorator: TickDecorator):

        super().__init__()
        self._decorator = decorator

    @property
    def tick_decorator(self):
        return self._decorator

    def decorate(self, item):
        return self._decorator.decorate(item)


class neg(TickDecorator):

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

    def decorate_tick(self, node):
        
        tick = node.tick
        def _(*args, **kwargs):
            status = tick(*args, **kwargs)
            return Status.FAILURE
        return _


class succeed(TickDecorator):

    def decorate_tick(self, node):
        
        tick = node.tick
        def _(*args, **kwargs):
            status = tick(*args, **kwargs)
            return Status.SUCCESS
        return _


class until(TickDecorator):

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

    def decorate_tick(self, node: Parallel):
        
        tick = node.tick
        def _(*args, **kwargs):
            status = tick(*args, **kwargs)
            status_total = node.status_total(Status.FAILURE)
            if status == Status.RUNNING and status_total > 0:
                return Status.FAILURE
            return status
        return _


STORE_REF = object()


class RefMixin(object):

    @classmethod
    def _process_ref_arg(cls, arg, store):

        if isinstance(arg, Ref):
            return arg.shared(store).value

        elif arg == STORE_REF:
            return store
        
        return arg

    @classmethod
    def _process_ref_args(cls, args: Args, store):

        return Args(
            *[cls._process_ref_arg(arg, store) for arg in args.args],
            **{k: cls._process_ref_arg(arg, store) for k, arg in args.kwargs}
        )

    @classmethod
    def _get_ref(cls, reference, member):
        return getattr(reference, member)

    @classmethod
    def _execute_ref(cls, reference, member, args: Args, store):
        args = cls._process_ref_args(args, store)
        return getattr(reference, member)(*args.args, **args.kwargs)


class ActionRef(Action, RefMixin):
    
    def __init__(self, name: str, action: str, args: Args):
        super().__init__(name)

        if self._reference is None:
            raise ValueError('Reference object must be defined to create an ActionReference')

        self._action_str = action
        self._args = args

    def act(self):
        return self._execute_ref(self._reference, self._action_str, self._args, self._store)


class ConditionalVarRef(Conditional, RefMixin):

    def __init__(self, name: str, condition: str):
        super().__init__(name)

        if self._reference is None:
            raise ValueError('Reference object must be defined to create a ConditionalReference')
        
        self._condition_str = condition

    def check(self):
        return self._get_ref(self._reference, self._condition_str)


class ConditionalRef(Conditional, RefMixin):
    
    def __init__(self, name: str, condition: str, args: Args):
        super().__init__(name)

        if self._reference is None:
            raise ValueError('Reference object must be defined to create a ConditionalReference')
        
        self._condition_str = condition
        self._args = args

    def check(self):
        return self._execute_ref(self._reference, self._condition_str, self._args, self._store)


class TaskRefDecorator(TaskDecorator, RefMixin):

    def __init__(self, name: str, decoration: str, args: Args, task: Task):
        super().__init__(name, task)

        if self._reference is None:
            raise ValueError('Reference object must be defined to create a Decorator')
        
        self._decoration_str = decoration
        self._args = args
        self._args.kwargs['task'] = task

    def decorate(self):
        return self._execute_ref(self._reference, self._decoration_str, self._args, self._store)


class TaskRefDecoratorLoader(AtomicDecoratorLoader):

    def __init__(self, decoration: str, args: Args, name: str=None):

        super().__init__()
        self._decoration = decoration
        self._name = name or decoration
        self._args = args

    def decorate(self, item):
        return TaskRefDecorator(self._name, self._decoration, self._args, item)


def action(act: str, *args, **kwargs):
    return TaskLoader(ActionRef, Args(act, *args, **kwargs))


def cond(check: str, *args, **kwargs):
    return TaskLoader(ConditionalRef, Args(check, *args, **kwargs))


def condvar(check: str):
    return TaskLoader(ConditionalVarRef, Args(check))


def _issubclassinstance(obj, cls):

    return isinstance(obj, type) and issubclass(obj, cls)


@singledispatch
def loads(decorator, name: str=None):
    """Loads a first order decorator

    Args:
        decorator: The tick decorator to load

    Returns:
        DecoratorLoader
    """

    if _issubclassinstance(decorator, TaskDecorator):
        return TaskDecoratorLoader(decorator, name)

    elif _issubclassinstance(decorator, TickDecorator):
        return TickDecoratorLoader(decorator())

    elif isinstance(decorator, TickDecorator):
        return TickDecoratorLoader(decorator)

    raise ValueError

@singledispatch
def loads_(decorator, *args, **kwargs):
    """Loads a decorator that takes arguments (2nd order decorator)

    Args:
        decorator: The tick decorator to load

    Returns:
        DecoratorLoader
    """
    if isinstance(decorator, str):
        return TaskRefDecoratorLoader(decorator, Args(*args, **kwargs))

    elif _issubclassinstance(decorator, TickDecorator2nd):
        return TickDecoratorLoader(decorator(*args, **kwargs))

    raise ValueError


# @loads_.register
# def _(decorator: str, *args, **kwargs):
#     return TaskDecoratorLoader('', TaskRefDecorator('', decorator, Args(*args, **kwargs)))


# @loads_.register
# def _(decorator: typing.Type, *args, **kwargs):
#     return TaskDecoratorLoader(decorator(*args, **kwargs))


class TaskFunc(object):
    
    def __init__(self, task_cls: typing.Type, func_vars: typing.List[str], init_vars: typing.List[str]):
        super().__init__()
        
        self._task_cls = task_cls
        self._func_vars = func_vars
        self._init_vars = init_vars
        self._task: Task = None
    
    def load(self, *args, **kwargs):

        for a, _a in zip(self._init_vars, args):
            kwargs[a] = _a

        self._task = self._task_cls(**kwargs)
    
    def __call__(self, *args, **kwargs):
        
        for a, _a in zip(self._init_vars, args):
            kwargs[a] = _a
        
        # TODO: Class would need to override the 
        # tick function and provide optional args.. 
        # I think this is the best approach
        self._task.tick(**kwargs)


def func(func_vars: typing.List[str], init_vars: typing.List[str]):
    
    def _(task_cls: typing.Type[Task]):
        return TaskFunc(task_cls, func_vars, init_vars)


def func_(task_cls: typing.Type[Task], func_vars: typing.List[str], init_vars: typing.List[str]):
    return TaskFunc(task_cls, func_vars, init_vars)


# def decorate(loader: Loader, decorators=None):
#     loader.add_decorators(decorators)


# @singledispatch
# def action(action, args: Args):
    
#     def _(store: Storage):
#         return action(*args.args, **args.kwargs, _store=store)

#     return _


# @action.register
# def _(args: Args):
    
#     def _(cls):
#         x = cls
#         def __new__(cls, store: Storage):
            
#             x.__new__(*args.args, **args.kwargs, _store=store)

#         cls.__new__ = __new__

#     return _


# class DecoratorRef(Task):

#     def __init__(self, name: str, decoration: str, args: Args, task: Task):
#         super().__init__(name)

#         if self._reference is None:
#             raise ValueError('Reference object must be defined to create a Decorator')
        
#         self._decoration_str = decoration

#     def decorate(self):
#         return self._decoration()

# def decorator(decoration: str, *args, **kwargs):

#     def _(node: Task):
#         return DecoratorRef(
#             '', decoration, Args(args, kwargs), node
#         )
#     # return DecoratorLoader(_)


# def context(cont: str, *args, **kwargs):

#     def _(node: Task):
#         return ContextRef(
#             '', cont, Args(args, kwargs), node
#         )
#     # return DecoratorLoader(_)

# decorator('progress_bar') << action('train')
# need to think a bit more about this

# def decorator(decorate: str, use_store: bool=False):

#     def decorate()
#     # Determine how to do this
#     return TaskLoader(DecoratorRef, Args())
#     # return DecoratorLoader(DecoratorRef)


# class DecoratorMeta(TaskMeta):

#     def _load_tasks(cls, store, kw):
#         tasks = []
#         for name, loader in ClassArgFilter([TypeFilter(TaskLoader)]).filter(cls).items():
#             if name in kw:
#                 loader(kw[name])
#                 del kw[name]
#             tasks.append(loader.load(store, name))
#         return tasks

#     def __call__(cls, *args, **kw):
#         self = cls.__new__(cls, *args, **kw)
#         store = cls._update_var_stores(kw)
#         task = cls._load_tasks(store, kw)['task']
#         reference = cls._get_reference(kw)
#         cls.__pre_init__(self, task, store, reference)
#         cls.__init__(self, *args, **kw)
#         return self


# class Decorator(Task, metaclass=DecoratorMeta):

#     def __pre_init__(self, task: Task, store: Storage=None):
#         super().__pre_init__(store)
#         self._task = task

#     @property
#     def tasks(self):
#         return self._task

#     @property
#     def status(self):
#         return self._cur_status
    
#     @abstractmethod
#     def decorate(self):
#         raise NotImplementedError

#     def tick(self):
#         if self._cur_status.done:
#             return Status.DONE

#         status = self.decorate()
#         self._cur_status = status
#         return status
    
#     def reset(self):
#         super().reset()
#         self._task.reset()
