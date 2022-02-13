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
from typing import Any, Generic, TypeVar
from .vars import STORE_REF, Args, Ref, Shared, Storage, Store, Var, UNDEFINED
from .utils import coalesce
import random
from functools import wraps

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
# from .nodes import ClassArgFilter, Loader, MemberRef, MemberRefFactory, Status, Task, TaskLoader, TaskMeta, TypeFilter, task
from typing import Any, Generic, TypeVar
from . import std
from .std import Status, Emission, Running, Ready, Success, Failure, StateVar, StateID, TaskDecorator, TickDecorator

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


class TaskClassFilter(ArgFilter):

    def __init__(self):
        pass
        # self._arg_type = arg_type
    
    def filter(self, name: str, annotation: typing.Type, value):
        print(name, value, type(value))
        return (
            (isinstance(value, TaskMeta) and issubclass(value, std.Task)) or
            isinstance(value, TickDecorator) or isinstance(value, TaskDecorator)
        )


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


T = TypeVar('T')


class VarStorer(Generic[T]):
    """Used to specify which variables are stored
    """

    def __init__(self, val: T=UNDEFINED):

        if isinstance(val, Store):
            self._val = val
        else:
            self._val = Var(val)

    @property
    def val(self):
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
    def _(self, val: Store):
        self._val = val
        return self


var_ = VarStorer


def ref_is_external(task_cls, default=True):
    
    if _issubclassinstance(task_cls, Tree): return False

    return getattr(task_cls, '__external_ref__', default)


class TaskMeta(type):

    def _update_var_stores(cls, kw):
        
        var_stores = ClassArgFilter([TypeFilter(VarStorer)]).filter(cls)
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



class Ext(object):
        
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


class Task(Ext, std.Task, metaclass=TaskMeta):
    pass


class ExtComposite(Ext):
        
    def __pre_init__(self, tasks: typing.List[Task], store: Storage, reference):
        self._store = store
        self._reference = reference
        self.__tasks__ = tasks


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


class CompositeMeta(TaskMeta):

    def _load_tasks(cls, store, kw, reference):
        tasks = []
        for name, loader in ClassArgFilter([TypeFilter(TaskLoader), TaskClassFilter()]).filter(cls).items():
            if not isinstance(loader, TaskLoader):
                loader = TaskLoader(loader)
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


class ExtTree(Ext):
        
    def __pre_init__(self, entry: Task, store: Storage, reference):
        self._store = store
        self._reference = reference
        self.__entry__ = entry
        

class TreeMeta(TaskMeta):

    def _load_entry(cls, store, kw, reference):

        tasks = ClassArgFilter([TypeFilter(TaskLoader), TaskClassFilter()]).filter(cls)

        if 'entry' not in tasks:
            raise AttributeError(f'Task entry not defined for tree')
        entry = tasks['entry']
        if not isinstance(entry, TaskLoader):
            entry = TaskLoader(entry)
        if entry in kw:
            entry(kw['entry'])
            del kw['entry']
        if isinstance(entry, TaskMeta) and issubclass(entry, Task):
            entry = TaskLoader(entry)
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


class Tree(ExtTree, std.Tree, metaclass=TreeMeta):
    
    def __init__(self, name: str=''):
        std.Tree.__init__(
            self, name, self.__entry__
        )


class Action(Ext, std.Action, metaclass=AtomicMeta):
    pass


class Conditional(Ext, std.Conditional, metaclass=AtomicMeta):
    pass


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

    def __init__(self, factory: typing.Callable=UNDEFINED, args: Args=None):

        self._factory = factory
        self._args = args or Args()
    
    def load(self, storage: Storage, name: str='', reference=None):
        storage = Storage(parent=storage)
        args = self._args.update_refs(storage)

        if self._factory is UNDEFINED:
            raise ValueError(f"Factory to load for {type(self).__name__} has not been defined")
        
        kwargs = {}
        if reference is not None and not isinstance(self._factory, Tree):
            kwargs['reference'] = reference

        item = self._factory(
            store=storage, name=name, *args.args, **args.kwargs, **kwargs
        )
        return item

    def __call__(self, factory):
        self._factory = factory
        return self


class TaskLoader(Loader):

    def __init__(self, cls: typing.Type=UNDEFINED, args: Args=None):

        super().__init__(cls, args)
        self.decorator: DecoratorLoader = None

    def add_decorator(self, decorator, prepend=True):

        if self.decorator is None:
            self.decorator = decorator
        elif prepend:
            self.decorator = self.decorator.prepend(decorator)
        else:
            self.decorator = self.decorator.append(decorator)
    
    def load(self, storage: Storage, name: str='', reference=None):

        task = super().load(storage, name, reference)
        if self.decorator is not None:
            task = self.decorator.decorate(task)
        return task

    def __lshift__(self, decorator: DecoratorLoader):

        self.add_decorator(decorator)
        return self


def task(cls: typing.Type[Task]):
    """Convenience method to create a TaskLoader"""
    return TaskLoader(cls)


# TODO: Determine whether this is necessary. Does not look
# necessary right now
def task_(cls: typing.Type[Task], *args, **kwargs):
    """Convenience method to create a TaskLoader"""
    return TaskLoader(cls, args=Args(*args, **kwargs))



class Sequence(ExtComposite, std.Sequence, metaclass=CompositeMeta):
    
    def __init__(self, name: str='', planner: std.Planner=None):
        std.Sequence.__init__(self, self.__tasks__, name, planner)


class Fallback(ExtComposite, std.Fallback, metaclass=CompositeMeta):

    def __init__(self, name: str='', planner: std.Planner=None):
        std.Fallback.__init__(self, self.__tasks__, name, planner)


class Parallel(ExtComposite, std.Parallel, metaclass=CompositeMeta):

    def __init__(self, name: str='', planner: std.Planner=None):
        std.Parallel.__init__(self, self.__tasks__, name, planner)


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

    def __init__(self, decorator_cls: typing.Type[std.TaskDecorator], args: Args=None, name: str=None):

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
def decorate(decorator: typing.Type[std.TaskDecorator], *args, **kwargs):

    return TaskDecoratorLoader(
        decorator, Args(*args, **kwargs)
    )


class TickDecoratorLoader(AtomicDecoratorLoader):

    def __init__(self, decorator: std.TickDecorator):

        super().__init__()
        self._decorator = decorator

    @property
    def tick_decorator(self):
        return self._decorator

    def decorate(self, item):
        return self._decorator.decorate(item)


class MemberRef(object):

    def __init__(self, member: str, args: Args, store: Store, reference):

        self._member = member
        self._args = args
        self._store = store
        self._reference = reference

    def _process_ref_arg(self, arg):

        if isinstance(arg, Ref):
            return arg.shared(self._store).val

        elif arg == STORE_REF:
            return self._store
        
        return arg

    def _process_ref_args(self):

        return Args(
            *[self._process_ref_arg(arg) for arg in self._args.args],
            **{k: self._process_ref_arg(arg) for k, arg in self._args.kwargs.items()}
        )
    
    def execute(self):
        args = self._process_ref_args()
        return getattr(self._reference, self._member)(*args.args, **args.kwargs)

    def get(self):
        return getattr(self._reference, self._member)


class MemberRefFactory(object):

    def __init__(self, member: str, args: Args=None):

        self._member = member
        self._args = args if args is not None else Args()
    
    def produce(self, store: Store, reference):

        if reference is None:
            raise ValueError('Reference object must be defined to create an ActionReference')
        args = self._args.update_refs(store)
        return MemberRef(self._member, args, store, reference)


class ActionFuncRef(Action):
    """Task that executes an action on the reference object
    """
    
    def __init__(self, name: str, member_factory: MemberRefFactory):
        super().__init__(name)
        self._member_ref = member_factory.produce(self._store, self._reference)

    def act(self):
        return self._member_ref.execute()


class ConditionalFuncRef(Conditional):
    """Task that retrieves a function
    """
    
    def __init__(self, name: str, member_factory: MemberRefFactory):
        super().__init__(name)
        self._member_ref = member_factory.produce(self._store, self._reference)

    def check(self):
        return self._member_ref.execute()


class ActionRef(Action):
    """Task that executes an action on the reference object
    """
    
    def __init__(self, name: str, member_factory: MemberRefFactory, status_override: Status=None):
        super().__init__(name)
        self._member_ref = member_factory.produce(self._store, self._reference)
        self._status_override = status_override

    def reset(self):
        super().reset()
        self._member_ref.get().reset()

    def act(self):
        res = self._member_ref.get().act()
        if self._status_override is not None:
            return self._status_override
        return res


class ConditionalRef(Conditional):
    """Task that executes an action on the reference object
    """
    
    def __init__(self, name: str, member_factory: MemberRefFactory):
        super().__init__(name)
        self._member_ref = member_factory.produce(self._store, self._reference)

    def reset(self):
        super().reset()
        self._member_ref.get().reset()

    def check(self):
        return self._member_ref.get().check()


class ConditionalVarRef(Conditional):
    """Task that retrieves a boolean variable
    """

    def __init__(self, name: str, member_factory: MemberRefFactory):
        super().__init__(name)
        self._member_ref = member_factory.produce(self._store, self._reference)

    def check(self):
        return self._member_ref.get()


class ActionFunc(Action):
    
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
    """Task that executes an action on the reference object
    """
    
    def __init__(self, name: str, f: typing.Callable, args: Args):
        super().__init__(name)
        self._f = f
        self._args = args

    def check(self):
        return self._f(*self._args.args, **self._args.kwargs)


def action(act: str) -> TaskLoader:
    """Convenience function for creating an ActionRef

    Args:
        act (str): The name fo the actino function

    Returns:
        TaskLoader
    """
    factory  = MemberRefFactory(act)
    return TaskLoader(ActionRef, args=Args(member_factory=factory))


@singledispatch
def actionf(act, *args, **kwargs) -> TaskLoader:
    args = Args(*args, **kwargs)
    return TaskLoader(ActionFunc, args=Args(f=act, args=args))


@actionf.register
def _(act: str, *args, **kwargs) -> TaskLoader:
    """Convenience function for creating an ActionRef

    Args:
        act (str): The name fo the actino function

    Returns:
        TaskLoader
    """
    factory  = MemberRefFactory(act, Args(*args, **kwargs))
    return TaskLoader(
        ActionFuncRef, args=Args(member_factory=factory)
    )


@singledispatch
def success(act, *args, **kwargs):

    args = Args(*args, **kwargs)
    return TaskLoader(ActionFunc, args=Args(f=act, args=args, status_override=Status.SUCCESS))

@success.register
def _(act: str, *args, **kwargs):
    factory  = MemberRefFactory(act, Args(*args, **kwargs))
    return TaskLoader(
        ActionFuncRef, args=Args(member_factory=factory, status_override=Status.SUCCESS)
    )

@singledispatch
def failure(act, *args, **kwargs):
    args = Args(*args, **kwargs)
    return TaskLoader(ActionFunc, args=Args(f=act, args=args, status_override=Status.FAILURE))


@failure.register
def _(act: str, *args, **kwargs):
    factory  = MemberRefFactory(act, Args(*args, **kwargs))
    return TaskLoader(
        ActionFuncRef, args=Args(member_factory=factory, status_override=Status.FAILURE)
    )


def cond(check: str) -> TaskLoader:
    """Convenience function for creating an ConditionalRef

    Args:
        act (str): The name fo the actino function

    Returns:
        TaskLoader
    """
    factory = MemberRefFactory(check)
    return TaskLoader(
        ConditionalRef, args=Args(member_factory=factory)
    )


@singledispatch
def condf(check, *args, **kwargs) -> TaskLoader:
    args = Args(*args, **kwargs)
    return TaskLoader(ConditionalFunc, args=Args(f=check, args=args))


@condf.register
def _(check: str, *args, **kwargs) -> TaskLoader:
    """Convenience function for creating a ConditionalRef

    Args:
        check (str): The name of the conditional function

    Returns:
        TaskLoader
    """
    factory  = MemberRefFactory(check, Args(*args, **kwargs))
    return TaskLoader(
        ConditionalFuncRef, args=Args(member_factory=factory)
    )


@singledispatch
def condvar(check: str) -> TaskLoader:
    """Convenience function for creating a ConditionalRef

    Args:
        check (str): The name of the conditional function

    Returns:
        TaskLoader
    """
    factory  = MemberRefFactory(check, Args())
    return TaskLoader(ConditionalVarRef, args=Args(member_factory=factory))


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

    if _issubclassinstance(decorator, std.TaskDecorator):
        return TaskDecoratorLoader(decorator, name)

    elif _issubclassinstance(decorator, std.TickDecorator):
        return TickDecoratorLoader(decorator())

    elif isinstance(decorator, std.TickDecorator):
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

    if _issubclassinstance(decorator, std.TickDecorator2nd):
        return TickDecoratorLoader(decorator(*args, **kwargs))

    raise ValueError


class StateMeta(TaskMeta):

    def __call__(cls, *args, **kw):
        self = cls.__new__(cls, *args, **kw)
        store = cls._update_var_stores(kw)
        reference = cls._get_reference(kw)
        cls.__pre_init__(self, store, reference)
        cls.__init__(self, *args, **kw)
        return self


V = TypeVar('V')
 
# Emission = TypeVar('Emission')


class Discrete(Ext, std.Discrete[V], metaclass=StateMeta):
    pass


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


class ExtStateMachine(Ext):
        
    def __pre_init__(self, start, states, store: Storage, reference):
        self._store = store
        self._reference = reference
        self.__start__ = start
        self.__states__ = states


# TODO: Need to edit to use preinit
class StateMachine(Ext, std.StateMachine, metaclass=StateMachineMeta):
    pass


class TaskState(Ext, std.Discrete):
    pass


class TaskStateLoader(StateLoader):

    def __init__(self, task_loader: TaskLoader, failure_to: StateVar, success_to: StateVar):

        def load(store, name, *args, **kwargs):
            return TaskState(task_loader.load(store, name, *args, **kwargs), failure_to, success_to)
        super().__init__(load)

    def __call__(self, state: std.State):
        self._state = state
        return self


@singledispatch
def state_(s: typing.Union[std.State, TaskLoader], *args, **kwargs):

    if issubclass(s, std.State):
        return StateLoader(s, Args(*args, **kwargs))
    
    return StateLoader(TaskStateLoader(s))

@state_.register
def _(s: str):
    return StateLoader(DiscreteStateRef, args=Args(member_ref=s))


def state(*args, **kwargs):
    return StateLoader(args=Args(*args, **kwargs))


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
    
    def _(states: typing.List[std.State]):

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

        return std.StateLink(**state_map)
    return _


def to_status(failure: typing.Optional[str] = None, success: typing.Optional[str]=None):

    def _(states: typing.List[std.State]):

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
        return std.StateLink(**_state_map)
    return _


LinkFunc = typing.Callable[[typing.List[Discrete]], std.StateLink]


class FSM(ExtStateMachine, std.FSM, metaclass=StateMachineMeta):
    
    def __init__(self, name: str=''):
        std.FSM.__init__(
            self, self.__start__, self.__states__, name
        )


class FSMStateLoader(StateLoader):

    def __init__(self, fsm_factory, link_f: LinkFunc, args: Args=None):

        self._args = args or Args()
        self._fsm_factory = fsm_factory
        def load(store, name='', *args, **kwargs):

            if self._fsm_factory is None:
                raise ValueError("The finite state machine factory has not been defined.")
            fsm: std.FSM = self._fsm_factory(store=store, *args, **kwargs)
            
            return std.FSMState(
                name=name, machine=fsm, state_link=link_f(fsm.states)
            )

        super().__init__(load, args)

    def __call__(self, fsm_factory):
        self._fsm_factory = fsm_factory
        return self


@singledispatch
def fsmstate_(s: typing.Type, map_to: typing.Dict[str, std.State], *args, **kwargs):

    return FSMStateLoader(s, map_to, Args(*args, **kwargs))


@singledispatch
def fsmstate(map_to: typing.Dict[str, std.State], *args, **kwargs):

    return FSMStateLoader(None, map_to, Args(*args, **kwargs))


class FSMRef(std.FSM):

    def __init__(self, name: str, member_ref: MemberRef):
        super().__init__(name)
        self._member_ref = member_ref

    def enter(self):
        self._member_ref.get().enter()

    def reset(self):

        self._member_ref.get().reset()

    def update(self):

        return self._member_ref.get().update()

    @property
    def status(self):
        return self._member_ref.get().status


class StatusMixin(ABC):

    @abstractproperty
    def status(self) -> Status:
        raise NotImplementedError



# class Tree(Task, metaclass=TreeMeta):
#     """The base behavior tree task. Use the behavior tree like a regular class
#     The reference object for subtasks will refer to the 'tree' object
#     """

#     def __pre_init__(self, entry: Task, store: Storage, reference):
#         super().__pre_init__(store, reference)
#         self.entry = entry
    
#     def tick(self) -> Status:        
#         self._cur_status = self.entry.tick()
#         return self._cur_status
    
#     def reset(self):
#         self._cur_status = Status.READY
#         return self.entry.reset()

# class Fallback(Composite):
#     """
#     Executes the subtasks in sequential order
#     Succeeds when one subtask has succeeded
#     """

#     def subtick(self) -> Status:

#         if self._planner.end() is True:
#             return Status.NONE
        
#         status = self._planner.cur.tick()
#         if status == Status.SUCCESS:
#             return Status.SUCCESS
        
#         if status == Status.FAILURE:
#             self._planner.adv()
        
#         if self._planner.end():
#             return Status.FAILURE
#         return Status.RUNNING


# class Composite(Task, metaclass=CompositeMeta):
#     """Task composed of subtasks
#     """

#     def __init__(
#         self, name: str='', planner: Planner=None
#     ):
#         super().__init__(name)
#         self._planner = planner or LinearPlanner(self._tasks)
#         self._sub_status = Status.READY

#     def __pre_init__(self, tasks: typing.List[Task], store: Storage, reference):
#         super().__pre_init__(store, reference)
#         self._tasks = tasks

#     @property
#     def n(self):
#         """The number of subtasks"""
#         return len(self._tasks)
    
#     @property
#     def tasks(self):
#         """The subtasks"""
#         return [*self._tasks]

#     @abstractmethod
#     def subtick(self) -> Status:
#         """Tick each subtask. Implement when implementing a new Composite task"""
#         raise NotImplementedError

#     @property
#     def status(self):
#         return self._cur_status

#     def tick(self):
        
#         if self._cur_status.done:
#             return Status.DONE

#         status = self.subtick()
#         self._cur_status = status
#         return status
    
#     def reset(self):
#         super().reset()
#         self._planner.reset()
#         for task in self._tasks:
#             task.reset()




# class Sequence(Composite):
#     """
#     Executes the subtasks in sequential order
#     Succeeds when all subtasks have succeeded
#     """

#     def reset(self):
#         super().reset()
#         self._planner.reset()
    
#     def _plan(self):
#         return self._tasks

#     def subtick(self) -> Status:

#         if self._planner.end() is True:
#             return Status.NONE
        
#         status = self._planner.cur.tick()
#         if status == Status.FAILURE:
#             return Status.FAILURE
        
#         if status == Status.SUCCESS:
#             self._planner.adv()
#         if self._planner.end():
#             return Status.SUCCESS
#         return Status.RUNNING


# class Conditional(Task, metaclass=AtomicMeta):
#     """Use to check a condition. Implement the 'check' method for subclasses"""

#     @abstractmethod
#     def check(self) -> bool:
#         raise NotImplementedError

#     def tick(self):
#         if self._cur_status.done:
#             return Status.DONE
#         self._cur_status = Status.SUCCESS if self.check() else Status.FAILURE
#         return self._cur_status


# class Action(Task, metaclass=AtomicMeta):
#     """Use to execute an action. Implement the 'act' method for subclasses"""

#     @abstractmethod
#     def act(self):
#         raise NotImplementedError

#     def tick(self):
#         if self._cur_status.done:
#             return Status.DONE
#         self._cur_status = self.act()
#         return self._cur_status



# class Parallel(Composite):
#     """
#     Executes the subtasks in parallel
#     Succeeds when all subtasks have succeeded
#     Fails when all subtasks have finished and one fails
#     """

#     def __init__(
#         self, name: str='', planner: Planner=None
#     ):
#         super().__init__(name=name, planner=planner)
#         self._statuses = []
    
#     def status_total(self, status: Status):
#         total = 0
#         for s in self._statuses:
#             if s == status:
#                 total += 1
#         return total
    
#     def reset(self):
#         super().reset()
#         self._statuses = []

#     def subtick(self):

#         for i, task in enumerate(iterate_planner(self._planner)):
#             if i > len(self._statuses) - 1:
#                 self._statuses.append(Status.RUNNING)
#             elif self._statuses[i] != Status.RUNNING:
#                 continue
#             self._statuses[i] = task.tick()
    
#         if Status.RUNNING in self._statuses:
#             return Status.RUNNING 

#         return Status.SUCCESS if Status.FAILURE not in self._statuses else Status.FAILURE


# class Task(object, metaclass=TaskMeta):
#     """The base class for a Task node

#     A task node has a 'store' and a 'reference' object which may 
#     be None.

#     Attributes in the 'store' can be accessed through the attribute
#     operator.
#     """

#     def __init__(self, name: str=''):
#         self._name = name
#         self._cur_status: Status = Status.READY
    
#     def __pre_init__(self, store: Storage, reference):
#         self._store = store
#         self._reference = reference

#     def reset(self):
#         self._cur_status = Status.READY
    
#     @abstractmethod
#     def tick(self) -> Status:
#         raise NotImplementedError

#     @property
#     def status(self) -> Status:
#         return self._cur_status
    
#     @property
#     def name(self) -> str:
#         return self._name

#     def __getattribute__(self, key: str) -> Any:
#         try:
#             store: Storage = super().__getattribute__('_store')

#             if store.contains(key, recursive=False):
#                 v = store.get(key, recursive=False)
#                 return v

#         except AttributeError:
#             pass
#         return super().__getattribute__(key)



# TODO: Think about how to define this... will be called
# on ready and on 

# @singledispatch
# def context(f, *args, **kwargs):
#     factory  = MemberRefFactory(f, Args(*args, **kwargs))


# @context.register
# def _(f: str, *args, **kwargs):
#     factory  = MemberRefFactory(f, Args(*args, **kwargs))



# class TickDecorator(object):
#     """
#     Wraps the 'tick' method of a class with anohter function

#     When inheriting, implement the decorate_tick method
#     """

#     def __init__(self, node: typing.Type[Task]=None):
#         self._node = node

#     @abstractmethod
#     def decorate_tick(self, node):
#         """Decorate the tick method of the argument node

#         Args:
#             node (Task)
#         """
#         raise NotImplementedError
    
#     def decorate(self, node: Task):
#         node.tick = wraps(node.tick)(self.decorate_tick(node))
#         return node

#     def __call__(self, *args, **kwargs):
#         """
#         Instantiate the node class with args, kwargs and
#         then decorate it with the decorate_tick method

#         Returns:
#             Decorated node
#         """
#         if self._node is None:
#             raise AttributeError(f'Member node has not been instantiated')
        
#         return self.decorate(self._node(*args, **kwargs))


# class TickDecorator2nd(TickDecorator):
#     """
#     2nd order decorator. Wraps the 'tick' method of a class with another function

#     When inheriting, implement the decorate method
#     """
#     def __init__(self, *args, **kwargs):
#         """
#         Args:
#             args : The args into the network
#             kwargs: The kwargs
#         """
#         super().__init__()
#         self._args = args
#         self._kwargs = kwargs

#     @abstractmethod
#     def decorate_tick(self, node):
#         """Decorate the tick method of the argument node

#         Args:
#             node (Task)
#         """
#         raise NotImplementedError

#     def decorate(self, node: Task):
#         node.tick = wraps(node.tick)(self.decorate_tick(node))
#         return node
    
#     def __call__(self, node_cls: typing.Type[Task]):
#         """Return a method to instantiate the node passed in

#         Args:
#             node_cls (typing.Type[Task]): [description]
#         """
        
#         def instantiator(*args, **kwargs):

#             node = node_cls(*args, **kwargs)
#             return self.decorate(node)
        
#         return instantiator


# class TaskDecorator(Task):
#     """A 'task' that decorates another task
#     """
    
#     def __init__(self, task: Task):
#         """initializer

#         Args:
#             task (Task): Task to decorate
#         """
#         super().__init__('')
#         self._task = task
    
#     @abstractmethod
#     def decorate(self):
#         raise NotImplementedError

#     def tick(self):

#         if self._cur_status.done:
#             return Status.DONE

#         return self.decorate()

