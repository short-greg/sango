from abc import ABC, abstractmethod, abstractproperty
from enum import Enum
import functools
import itertools
import typing
from functools import partial, singledispatch, singledispatchmethod
from typing import Any, Generic, Iterator, TypeVar
from sango.vars import UNDEFINED, ConditionSet, HierarchicalStorage, Storage
from .vars import AbstractStorage, Args, HierarchicalStorage, NullStorage, Ref, Shared, StoreVar, Var, InitVar
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
        return self == Status.FAILURE or Status.SUCCESS or Status.DONE


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

    for var in [x for x in dir(cls) if not x.startswith('__')]:
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


class ArgManager(object):

    def __init__(self, filters: typing.List[ArgFilter]):
        
        self._filters = filters


        # self._init_vars = {}
        # self._args = []
        # self._args_by_name = {}
        # self._tasks: typing.Dict[str, TaskLoader] = {}
        # self._vars: typing.Dict[str, VarStorer] = {}
        # i = 0
        # # parent_storage = kwargs.get('_store') or NullStorage()
        # self._storage = HierarchicalStorage(Storage(), parent_store)
        # for (name, type_, value, is_task) in vals(cls):
            

        #     val = kwargs.get(name, value)



        #     self._args.append((name, type_, value, is_task))
        #     self._args_by_name[name] = (name, type_, value, is_task)
        #     if i < len(args):
        #         val = args[i]
        #     else:
        #         val = kwargs.get(name, value)

        #     if val == UNDEFINED:
        #         raise AttributeError(f"Value for {name} was not given.")
        #     if isinstance(value, TaskLoader) and value is not val:
        #         self._tasks[name] = value(val)
        #     elif isinstance(value, TaskLoader):
        #         self._tasks[name] = val
        #     elif isinstance(val, InitVar):
        #         self._init_vars[name] = val.value
        #     elif isinstance(value, VarStorer):
        #         self._vars[name] = value
        #         # self._storage.add(name, value(val).val)
        #         # value(val).store(name, self._storage)
        #     i += 1

        # if i < len(self._args):
        #     raise TypeError(f"{cls}() takes {i} positional arguments but {len(self._args)} were given") 

    
        # data_defined = {key: _update_var(datum, parent_storage) for key, datum in data.items()}
        # self._storage = HierarchicalStorage(Storage(**data_defined), parent_storage)
        # self._tasks = {k: task.load(self._storage) for k, task in tasks.items()}

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

            # val = kwargs.get(name, value)
            # result_kwargs[]
            


    # @property
    # def vars(self):
    #     return self._vars

    # @property
    # def tasks(self):
    #     return self._tasks

    # @property
    # def init_vars(self):
    #     return self._init_vars

    # # @property
    # # def store(self) -> Storage:
    # #     return self._storage
    
    # def get(self, key):
    #     return self._args_by_name[key]['value']

# post_init <- use args/kwargs
# get all vars/tasks etc from kwargs only


class AtomicMeta(type):

    def __call__(cls, *args, **kw):

        self = cls.__new__(*args, **kw)
        kw['store'] = HierarchicalStorage(Storage(**vars), kw.get('store'))
        kw['tasks'] = ArgManager(TypeFilter(TaskLoader)).filter(cls)
        cls.__init__(self, *args, **kw)
    

class TreeMeta(type):

    def __call__(cls, *args, **kw):

        self = cls.__new__(*args, **kw)
        kw['store'] = HierarchicalStorage(Storage(**vars), kw.get('store'))
        kw['entry'] = ArgManager(TypeFilter(TaskLoader)).filter(cls)['entry']
        cls.__init__(self, *args, **kw)


class CompositeMeta(type):

    def __call__(cls, *args, **kw):
        self = cls.__new__(*args, **kw)
        vars = ArgManager(TypeFilter(VarStorer)).filter(cls)
        kw['store'] = HierarchicalStorage(Storage(**vars), kw.get('store'))
        kw['tasks'] = ArgManager(TypeFilter(TaskLoader)).filter(cls)
        cls.__init__(self, *args, **kw)


class Task(ABC):

    def __init__(self, store: Storage):
        self._store = store
    
    @abstractmethod
    def reset(self):
        pass
    
    @abstractmethod
    def tick(self) -> Status:
        raise NotImplementedError

    def __post_init__(self):
        pass

    def __getattribute__(self, key: str) -> Any:
        try:
            store: HierarchicalStorage = super().__getattribute__('_store')
            if store.contains(key, recursive=False):
                v = store.get(key, recursive=False)
                return v
        except AttributeError:
            pass
        return super().__getattribute__(key)


Task.__call__ = Task.tick


def _func():
    pass


class Atomic(Task):

    __metaclass__ = AtomicMeta

    # def __new__(cls, *args, **kwargs):
    #     arg_vars = ArgVars(cls, args, kwargs)
    #     obj: Atomic = object.__new__(cls)
    #     print('Before init')
    #     # obj.__init__(store=arg_vars.store)
    #     print('After init')
    #     # obj.__post_init__(**arg_vars.init_vars)
    #     print('After post init')

    #     return obj


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
        if len(self._idx) == len(self._items):
            return True
        return False
    
    def adv(self):
        if self._idx == len(self._items):
            return False
        self._idx += 1
        return True
    
    @property
    def cur(self):
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


class Composite(Task):

    __metaclass__ = CompositeMeta

    def __init__(
        self, tasks, store: Storage, planner: Planner=None
    ):
        super().__init__(store)
        self._tasks = tasks
        self._planner = planner or LinearPlanner(tasks)

    @property
    def n(self):
        return len(self._tasks)

    # def __init__(
    #     self, tasks: typing.List[Task], store: Storage
    # ):
    #     self._tasks: typing.List[Task] = tasks
    #     self._store = store
    #     self._n = len(tasks)
    
    @property
    def tasks(self):
        return list(**self._tasks)
    
    @property
    def n(self):
        return self._n

    @abstractmethod
    def subtick(self) -> Status:
        raise NotImplementedError

    @property
    def status(self):
        return self._cur_status

    def tick(self):
        if self._cur_status.done():
            return Status.DONE

        status = self.subtick()
        self._cur_status = status
        return status
    
    def reset(self):
        for task in self._tasks:
            task.reset()
    
    # def __new__(cls, *args, **kwargs):

    #     arg_vars = ArgVars(cls, args, kwargs)
    #     obj: Composite = super().__new__(cls, tasks=arg_vars.tasks, store=arg_vars.store)
    #     obj.__post_init__(**arg_vars.init_vars)
    #     return obj


class Tree(Task):

    __metaclass__ = TreeMeta
    # def __new__(cls, *args, **kwargs):

    #     arg_vars = ArgVars(cls, args, kwargs)
    #     entry = arg_vars.get('entry')        
    #     if entry is None:
    #         try:
    #             entry = vals["entry"]
    #         except KeyError:
    #             raise KeyError("Field entry was not defined.")
    #     obj: Tree = super().__new__(cls, entry, store=arg_vars.store)
    #     obj.__post_init__(**arg_vars.init_vars)
    #     return obj
    
    def __init__(self, entry: Task, store: Storage):
        self._entry = entry
        self._store = store

    def __post_init__(self):
        pass

    def tick(self) -> Status:        
        self._entry.tick()


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
    
    def reset(self):
        pass

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
    
    def load(self, storage: Storage):
        print('Task: ', self._task)
        task = self._task(
            _store=storage, *self._args.args, **self._args.kwargs
        )
        print('Loaded')
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

    def __init__(self, val: Var):

        self._val = val

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
    def _(self, val: Var):
        self._val = val
        return self


def var(val=UNDEFINED):    
    return VarStorer(Var(val))


class Sequence(Composite):

    def reset(self):
        self._planner.reset()
    
    def _plan(self):
        return self._tasks

    def subtick(self) -> Status:
        if self._planner.end() is True:
            return Status.NONE

        status = self._planner.cur.tick()
        if status == Status.FAILURE:
            return Status.FAILURE
        
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

