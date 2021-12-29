from .vars import InitVar
import pytest

from sango.vars import Args
from .nodes import Action, ArgVars, Conditional, Status, TaskLoader, vals

class TestStatus:

    def test_done_if_successful(self):
        status = Status.SUCCESS
        assert status.done 

    def test_done_if_failed(self):
        status = Status.SUCCESS
        assert status.done 

    def test_done_if_done(self):
        status = Status.DONE
        assert status.done 


class TestVals:

    def test_gets_value_with_no_annotation(self):

        class T:
            x = 2
            # x: int = 3
            # x: float
        
        _iter = vals(T)
        var, annotation, val, is_task = next(_iter)
        assert var == 'x'
        assert annotation is None
        assert val == 2
        assert is_task is False
            
    def test_gets_value_with_annotation(self):

        class T:
            x = 2
            x2: int = 3
            # x: float
        
        _iter = vals(T)
        var, annotation, val, is_task = next(_iter)
        var, annotation, val, is_task = next(_iter)
        assert var == 'x2'
        assert annotation == int
        assert val == 3
        assert is_task is False
            
    def test_value_with_only_annotation_doesnt_exist(self):

        class T:
            x: float
        
        _iter = vals(T)
        with pytest.raises(StopIteration):
            next(_iter)


class DummyAction(Action):

    def act(self):
        return Status.RUNNING


class DummyNegative(Conditional):

    def check(self):
        return False


class DummyPositive(Conditional):

    def check(self):
        return True


class TestArgVars:

    def test_with_two_init_vars(self):
        
        class T:
            x = InitVar(2)
            y = InitVar(3)
        vars = ArgVars(T, [], {})
        print(vars.init_vars)
        x = vars.init_vars['x']
        y = vars.init_vars['y']
        assert x == 2
        assert y == 3

    def test_with_no_init_vars(self):
        
        class T:
            x = 3
        vars = ArgVars(T, [], {})
        assert len(vars.init_vars) == 0
        assert len(vars.tasks) == 0

    def test_with_task_loader(self):
        
        class T:
            x = TaskLoader(DummyPositive)
    
        vars = ArgVars(T, [], {})
        assert len(vars.tasks) == 1
        task = vars.tasks['x']
        assert isinstance(task, DummyPositive)


class AtomicMeta(type):

    def __call__(cls, *args, **kw):
        self = cls.__new__(*args, **kw)
        arg_vars = ArgVars(*args, **kw)
        cls.__init__(self, vars=arg_vars.vars, *args, **kw)


class CompositeMeta(type):

    def __call__(cls, *args, **kw):
        self = cls.__new__(*args, **kw)
        arg_vars = ArgVars(*args, **kw)
        cls.__init__(self, task_loaders=arg_vars.tasks vars=arg_vars.vars, *args, **kw)


class T:

    def __init__(self, *args, **kwargs):
        print('init called')
    
    def __new__(cls):
        print("new called")
        obj = object.__new__(cls)
        return obj

t = T()
