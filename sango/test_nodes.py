import pytest

from sango.vars import Args
from .nodes import Action, Conditional, LinearPlanner, Sequence, Status, TaskLoader, TypeFilter, VarStorer, task, vals, ArgFilter, ClassArgFilter

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


class TestArgFilter:

    pass


class TestArgManager:

    def test_with_two_init_vars(self):
        
        class T:
            x = 2
            y = 3
        vars = ClassArgFilter([TypeFilter(int)]).filter(T)
        assert vars['x'] == 2
        assert vars['y'] == 3

    def test_with_no_init_vars(self):
        
        class T:
            x = 3
        vars = ClassArgFilter([TypeFilter(float)]).filter(T)
        assert len(vars) == 0


class TestCreateAtomicTask:

    def test_conditional_tick(self):

        conditional = DummyPositive(name='x')
        assert conditional.tick() == Status.SUCCESS
    
    def test_conditional_check(self):

        conditional = DummyPositive(name='x')
        assert conditional.check() == True
    
    def test_action_tick(self):

        actor = DummyAction(name='x')
        assert actor.act() == Status.RUNNING

    def test_action_start_status(self):

        actor = DummyAction(name='x')
        assert actor.status == Status.READY


class TestCreateSequenceTask:

    def test_num_elements_with_one_element(self):

        class Pos(Sequence):
            pos = task(DummyPositive)
        
        seq = Pos()
        assert seq.n == 1

    def test_tick_with_one_element(self):

        class Pos(Sequence):
            pos = task(DummyPositive)
        
        seq = Pos()
        status = seq.tick()
        assert status == Status.SUCCESS

    def test_tick_once_with_two_elements(self):

        class Pos(Sequence):
            pos = task(DummyPositive)
            neg = task(DummyNegative)
        
        seq = Pos()
        status = seq.tick()
        assert status == Status.RUNNING

    def test_tick_twice_with_two_elements(self):

        class Pos(Sequence):
            pos = task(DummyPositive)
            neg = task(DummyNegative)
        
        seq = Pos()
        status = seq.tick()
        status = seq.tick()
        assert status == Status.FAILURE
    
    def test_tick_twice_with_first_fail(self):

        class Pos(Sequence):
            neg = task(DummyNegative)
            pos = task(DummyPositive)
        
        seq = Pos()
        status = seq.tick()
        assert status == Status.FAILURE


class TestLinearPlanner:

    def test_loop_over_linear_planner(self):

        planner = LinearPlanner([DummyPositive(name='x'), DummyNegative(name='y')])
        assert isinstance(planner.cur, DummyPositive)
        planner.adv()
        assert isinstance(planner.cur, DummyNegative)
        
    def test_loop_over_linear_planner(self):

        planner = LinearPlanner([DummyPositive(name='x'), DummyNegative(name='y')])
        assert isinstance(planner.cur, DummyPositive)
        planner.adv()
        assert isinstance(planner.cur, DummyNegative)


# class T:

#     def __init__(self, *args, **kwargs):
#         print('init called')
    
#     def __new__(cls):
#         print("new called")
#         obj = object.__new__(cls)
#         return obj

# t = T()
