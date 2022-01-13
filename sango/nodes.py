"""
Nodes for buidling a Behavior Tree. A tree can be built hierarchically within
Python's class system by specifying which members are tasks and which
are variables to store.

example:
class tree(Tree):

    @task # specifies to load the following task
    class entry(Sequence):

        # variables in the store
        finished = var(True)

        # tasks - sequence of tasks to execute
        save = action('save')
        @task
        class finished(Conditional):
            def check(self):
                # will refer to the variable 'finished'
                return self.finished

    def save(self):
        # save operations
        pass

"""
from abc import ABC, abstractmethod, abstractproperty
from enum import Enum
import typing
from functools import singledispatch, singledispatchmethod
from typing import Any, Iterator
from .vars import STORE_REF, Args, Const, Ref, Storage, Store, Var, UNDEFINED
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


def vals(cls):

    try:
        annotations = cls.__annotations__
    except AttributeError:
        annotations = {}
    d  = getattr(cls, '__dict__', {})

    for var in [x for x in d.keys() if not x.startswith('__')]:
        annotation = annotations.get(var, None)
        val = getattr(cls, var)
        yield var, annotation, val


@singledispatch
def _update_var(val, storage: Storage):
    return Var[Any](val)


@_update_var.register
def _(val: Ref, storage: Storage):
    return val.shared(storage)


@_update_var.register
def _(val: Store, storage: Storage):
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
    """Use to extract args from class members
    """

    def __init__(self, filters: typing.List[ArgFilter]):
        self._filters = filters

    def _run_filters(self, name, type_, value):

        for filter in self._filters:
            if filter.filter(name, type_, value):
                return True

        return False
    
    def _filter_helper(self, cls, result_kwargs):
        
        for (name, type_, value) in vals(cls):
            if self._run_filters(name, type_, value):
                result_kwargs[name] = value
        return result_kwargs
        

    def filter(self, cls):
        """Run the arg filter
        """
        result_kwargs = {}
        for _cls in reversed(cls.mro()):
            self._filter_helper(_cls, result_kwargs)
        return result_kwargs


# TODO: Consider refactoring and simplifying

class Storer(ABC):
    """Used to specify which variables are stored
    """
    @abstractproperty
    def val(self):
        raise NotImplementedError

    @abstractproperty
    def __call__(self, val):
        raise NotImplementedError


class VarStorer(Storer):
    """Used to specify which variables are stored
    """

    def __init__(self, val):

        if isinstance(val, Var):
            self._val = val
        else:
            self._val = Var(val)

    @property
    def val(self):
        return self._val

    @singledispatchmethod
    def __call__(self, val):
        if isinstance(val, Const):
            raise ValueError('Cannot convert constant to var')
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


class ConstStorer(Storer):
    """Used to specify which variables are stored
    """

    def __init__(self, val):

        self._val = Const(val)

    @property
    def val(self):
        return self._val

    @singledispatchmethod
    def __call__(self, val):
        if isinstance(val, Var):
            val = val.val
        self._val = Const(val)
        return self

    # TODO: Decide what to do here
    @__call__.register
    def _(self, val: Ref):
        self._val = val
        return self

    @__call__.register
    def _(self, val: Const):
        self._val = val
        return self


def var_(val=UNDEFINED):    
    """Convenience function to create a VarStorer
    """
    return VarStorer(val)


def const_(val=UNDEFINED):    
    """Convenience function to create a ConstStorer
    """
    return ConstStorer(val)

def ref_is_external(task_cls, default=True):
    
    if _issubclassinstance(task_cls, Tree): return False

    return getattr(task_cls, '__external_ref__', default)


class TaskMeta(type):

    def _update_var_stores(cls, kw):
        
        var_stores = ClassArgFilter([TypeFilter(Storer)]).filter(cls)
        if 'store' in kw:
            store = kw['store']
            del kw['store']
        else:
            store = Storage()
        
        for name, storer in var_stores.items():
            if name in kw:
                storer(kw[name])
                del kw[name]
            
            store[name] = storer.val
        return store

    def _get_reference(cls, kw):
        if 'reference' in kw:
            reference = kw['reference']
            del kw['reference']
            return reference


class Task(object, metaclass=TaskMeta):
    """The base class for a Task node

    A task node has a 'store' and a 'reference' object which may 
    be None.

    Attributes in the 'store' can be accessed through the attribute
    operator.
    """

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

            if store.contains(key, recursive=False):
                v = store.get(key, recursive=False)
                return v

        except AttributeError:
            pass
        return super().__getattribute__(key)


Task.__call__ = Task.tick


class AtomicMeta(TaskMeta):
    """MetaClass for an atomic Task
    """

    def __call__(cls, *args, **kw):

        self = cls.__new__(cls, *args, **kw)
        
        store = cls._update_var_stores(kw)
        reference = cls._get_reference(kw)
        cls.__pre_init__(self, store, reference)
        cls.__init__(self, *args, **kw)
        return self
    

class Atomic(Task, metaclass=AtomicMeta):
    """Base class for an Atomic Class
    """
    
    def __pre_init__(self, store: Storage, reference):
        super().__pre_init__(store, reference)


class Planner(ABC):
    """Chooses the order which to execute the subtasks of a composite task
    """

    @abstractmethod
    def adv(self):
        pass

    @abstractmethod
    def end(self):
        pass
    
    @abstractmethod
    def reset(self):
        pass

    @abstractproperty
    def cur(self):
        pass


class LinearPlanner(object):
    """Planner that executes sequentially
    """

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
    """
    Convenience function to iterate over a planner
    """

    planner.reset()
    
    while planner.end() is False:
        yield planner.cur
        planner.adv()


def shuffle(linear: LinearPlanner):
    """Decorator for a linear planner that will shuffle the order
    in which tasks get executed
    """

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

    def _load_tasks(cls, store, kw, reference):
        tasks = []
        for name, loader in ClassArgFilter([TypeFilter(TaskLoader)]).filter(cls).items():
            loader: Loader = loader
            if name in kw:
                loader(kw[name])
                del kw[name]
            tasks.append(loader.load(store, name, reference))
        return tasks

    def __call__(cls, *args, **kw):
        self = cls.__new__(cls, *args, **kw)
        store = cls._update_var_stores(kw)
        reference = cls._get_reference(kw)
        tasks = cls._load_tasks(store, kw, reference)
        cls.__pre_init__(self, tasks, store, reference)
        cls.__init__(self, *args, **kw)
        return self


class Composite(Task, metaclass=CompositeMeta):
    """Task composed of subtasks
    """

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
        for task in self._tasks:
            task.reset()


class TreeMeta(TaskMeta):

    def _load_entry(cls, store, kw, reference):

        tasks = ClassArgFilter([TypeFilter(TaskLoader)]).filter(cls)
        if 'entry' not in tasks:
            raise AttributeError(f'Task entry not defined for tree')
        entry = tasks['entry']
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
        entry = cls._load_entry(store, kw, reference)
        cls.__pre_init__(self, entry, store, reference)
        cls.__init__(self, *args, **kw)
        return self


class Tree(Task, metaclass=TreeMeta):
    """The base behavior tree task. Use the behavior tree like a regular class
    The reference object for subtasks will refer to the 'tree' object
    """

    def __pre_init__(self, entry: Task, store: Storage, reference):
        super().__pre_init__(store, reference)
        self.entry = entry
    
    def tick(self) -> Status:        
        return self.entry.tick()


class Action(Atomic):
    """Use to execute an action. Implement the 'act' method for subclasses"""

    @abstractmethod
    def act(self):
        raise NotImplementedError

    def tick(self):
        if self._cur_status.done:
            return Status.DONE
        self._cur_status = self.act()
        return self._cur_status


class Conditional(Atomic):
    """Use to check a condition. Implement the 'check' method for subclasses"""

    @abstractmethod
    def check(self) -> bool:
        raise NotImplementedError

    def tick(self):
        if self._cur_status.done:
            return Status.DONE
        self._cur_status = Status.SUCCESS if self.check() else Status.FAILURE
        return self._cur_status


class DecoratorLoader(ABC):
    """Use to check a condition. Implement the 'check' method for subclasses"""

    @abstractmethod
    def __call__(self, loader):
        """Concatenate two decorator loaders or decorate a task loader"""
        pass

    @abstractmethod
    def __lshift__(self, other):
        """Concatenate two decorator loaders"""
        pass

    @abstractmethod
    def append(self, other):
        """Concatenate two decorator loaders"""
        pass

    @abstractmethod
    def prepend(self, other):
        """Concatenate two decorator loaders"""
        pass

    @abstractmethod
    def decorate(self, item):
        """Decorate the TaskLoader"""
        pass


class Loader(object):

    def __init__(self, cls: typing.Type=UNDEFINED, args: Args=None):

        self._cls = cls
        self._args = args or Args()
        self.decorator: DecoratorLoader = None
    
    def load(self, storage: Storage, name: str='', reference=None):
        storage = Storage(parent=storage)
        args = self._args.update_refs(storage)

        if self._cls is UNDEFINED:
            raise ValueError(f"Cls to load for {type(self).__name__} has not been defined")
        
        kwargs = {}
        if reference is not None and not isinstance(self._cls, Tree):
            kwargs['reference'] = reference

        print(isinstance(self._cls, Task), self._cls, *args.kwargs.values())
        item = self._cls(
            store=storage, name=name, *args.args, **args.kwargs, **kwargs
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


# TODO: Decide how to use the task loader
class TaskLoader(Loader):

    pass


def task(cls: typing.Type[Task]):
    """Convenience method to create a TaskLoader"""
    return TaskLoader(cls)


# TODO: Determine whether this is necessary. Does not look
# necessary right now
def task_(cls: typing.Type[Task], *args, **kwargs):
    """Convenience method to create a TaskLoader"""
    return TaskLoader(cls, args=Args(*args, **kwargs))


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

    When inheriting, implement the decorate_tick method
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


class DecoratorSequenceLoader(DecoratorLoader):
    """A sequence of decorators
    """

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
    def from_pair(cls, first: DecoratorLoader, second: DecoratorLoader):
        """Create a sequence from two decorators
        Args:
            first (DecoratorLoader): The first decorator loader
            second (DecoratorLoader): The second decorator loader

        Returns:
            [type]: [description]
        """

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
    """A single decorator loader
    """

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

    def __init__(self, decorator_cls: typing.Type[TaskDecorator], args: Args=None, name: str=None):

        super().__init__()
        self._args = args or Args()
        self._decorator_cls = decorator_cls
        self._name = name or self._decorator_cls.__name__
    
    @property
    def decorator_cls(self):
        return self._decorator_cls

    def decorate(self, item):
        return self._decorator_cls(self._name, item, *self._args.args, **self._args.kwargs)

@singledispatch
def decorate(decorator: typing.Type[TaskDecorator], *args, **kwargs):

    return TaskDecoratorLoader(
        decorator, Args(*args, **kwargs)
    )

@decorate.register
def _(decorator: str, *args, **kwargs):
    return TaskRefDecoratorLoader(
        decorator, Args(*args, **kwargs)
    )


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


class RefMixin(object):
    """Mixin for 'Reference' tasks. 'Reference' tasks call the 'Reference' object
    """

    @classmethod
    def _process_ref_arg(cls, arg, store):

        if isinstance(arg, Ref):
            return arg.shared(store).val

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
    """Task that executes an action on the reference object
    """
    
    def __init__(self, name: str, action: str, args: Args):
        super().__init__(name)
        args = args.update_refs(self._store)
        
        if self._reference is None:
            raise ValueError('Reference object must be defined to create an ActionReference')

        self._action_str = action
        self._args = args

    def act(self):
        return self._execute_ref(self._reference, self._action_str, self._args, self._store)


class ConditionalVarRef(Conditional, RefMixin):
    """Task that retrieves a boolean variable
    """

    def __init__(self, name: str, condition: str):
        super().__init__(name)

        if self._reference is None:
            raise ValueError('Reference object must be defined to create a ConditionalReference')
        
        self._condition_str = condition

    def check(self):
        return self._get_ref(self._reference, self._condition_str)


class ConditionalRef(Conditional, RefMixin):
    """Task that retrieves a function
    """
    
    def __init__(self, name: str, condition: str, args: Args):
        super().__init__(name)

        args = args.update_refs(self._store)
        if self._reference is None:
            raise ValueError('Reference object must be defined to create a ConditionalReference')
        
        self._condition_str = condition
        self._args = args

    def check(self):
        return self._execute_ref(self._reference, self._condition_str, self._args, self._store)


class TaskRefDecorator(TaskDecorator, RefMixin):
    """Decorator that uses a reference function. The reference function
    Must take in a task
    """

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


def action(act: str, *args, **kwargs) -> TaskLoader:
    """Convenience function for creating an ActionRef

    Args:
        act (str): The name fo the actino function

    Returns:
        TaskLoader
    """
    args = Args(*args, **kwargs)
    return TaskLoader(
        ActionRef, args=Args(args=args, action=act)
    )

def cond(check: str, *args, **kwargs) -> TaskLoader:
    """Convenience function for creating a ConditionalRef

    Args:
        check (str): The name of the conditional function

    Returns:
        TaskLoader
    """
    args = Args(*args, **kwargs)
    return TaskLoader(
        ConditionalRef, 
        args=Args(args=args, condition=check)
    )

def condvar(check: str) -> TaskLoader:
    """Convenience function for creating a ConditionalRef

    Args:
        check (str): The name of the conditional function

    Returns:
        TaskLoader
    """

    return TaskLoader(ConditionalVarRef, args=Args(condition=check))


def _issubclassinstance(obj, cls):
    """Convenience function to check if an object is a class and if
    so a subclass of cls
    """

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
