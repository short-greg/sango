from functools import wraps
import pytest

from sango.vars import Args, Storage
from .nodes import Action, Conditional, Fallback, LinearPlanner, Parallel, Sequence, Status, Task, TaskLoader, TickDecorator, TickDecorator2nd, Tree, TypeFilter, VarStorer, action, cond, condvar, fail, fail_on_first, loads, loads_, neg, succeed, succeed_on_first, task, task_, until, vals, ArgFilter, ClassArgFilter, var


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
        var, annotation, val = next(_iter)
        assert var == 'x'
        assert annotation is None
        assert val == 2
            
    def test_gets_value_with_annotation(self):

        class T:
            x = 2
            x2: int = 3
            # x: float
        
        _iter = vals(T)
        var, annotation, val = next(_iter)
        var, annotation, val = next(_iter)
        assert var == 'x2'
        assert annotation == int
        assert val == 3
            
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

class DummyRange(Conditional):

    def __init__(self, name: str = ''):
        super().__init__(name=name)
        self._idx = 0

    def check(self):
        if self._idx < 2:
            self._idx += 1
            return False
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

    def test_tick_with_no_tasks(self):

        class Pos(Sequence):
            pass
        
        seq = Pos()
        status = seq.tick()
        assert status == Status.NONE

    def test_pos_with_store(self):

        class Pos(Sequence):
            neg = task(DummyPositive)
            data = var(2)
        
        seq = Pos()
        assert seq.data.val == 2

    def test_pos_with_store_and_new_val(self):

        class Pos(Sequence):
            neg = task(DummyPositive)
            data = var(2)
        
        seq = Pos(data=3)
        assert seq.data.val == 3



class TestFallbackTask:

    def test_num_elements_with_one_element(self):

        class Pos(Fallback):
            pos = task(DummyPositive)
        
        seq = Pos()
        assert seq.n == 1

    def test_tick_with_one_element(self):

        class Pos(Fallback):
            pos = task(DummyPositive)
        
        seq = Pos()
        status = seq.tick()
        assert status == Status.SUCCESS

    def test_tick_once_with_two_elements(self):

        class Pos(Fallback):
            pos = task(DummyPositive)
            neg = task(DummyNegative)
        
        seq = Pos()
        status = seq.tick()
        assert status == Status.SUCCESS

    def test_tick_twice_with_two_elements(self):

        class Pos(Fallback):
            neg = task(DummyNegative)
            pos = task(DummyPositive)
        
        seq = Pos()
        status = seq.tick()
        assert status == Status.RUNNING
    
    def test_tick_twice_with_first_fail(self):

        class Pos(Fallback):
            neg = task(DummyNegative)
            pos = task(DummyNegative)
        
        seq = Pos()
        status = seq.tick()
        status = seq.tick()
        assert status == Status.FAILURE

    def test_tick_with_no_tasks(self):

        class Pos(Fallback):
            pass
        
        seq = Pos()
        status = seq.tick()
        assert status == Status.NONE


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


class TestTree:

    def test_tree_with_one_sequence(self):

        class X(Tree):
            entry = task_(DummyPositive)
        
        seq = X()
        assert isinstance(seq, Tree)
        assert isinstance(seq.entry, DummyPositive)

    def test_tree_with_one_sequence(self):

        class X(Tree):
            @task
            class entry(Fallback):
                pos = task_(DummyPositive)
                neg = task_(DummyPositive)
        
        seq = X()
        assert isinstance(seq, Tree)
        assert isinstance(seq.entry, Fallback)

    def test_tick_with_one_element(self):

        class X(Tree):
            @task
            class entry(Fallback):
                pos = task_(DummyPositive)
        
        tree = X()
        status = tree.tick()
        assert status == Status.SUCCESS


class TestDecorators:

    def test_neg_with_one_sequence(self):

        class X(Tree):
            @loads(neg)
            @task
            class entry(Fallback):
                pos = task_(DummyPositive)
                neg = task_(DummyNegative)
        
        tree = X()
        status = tree.tick()
        assert status == Status.FAILURE

    def test_neg_preceding_dirctor_with_one_sequence(self):

        class X(Tree):
            @task
            @neg
            class entry(Fallback):
                pos = task_(DummyPositive)
                neg = task_(DummyNegative)
        
        tree = X()
        status = tree.tick()
        assert status == Status.FAILURE

    def test_fail_preceding_decorator_with_one_sequence(self):

        class X(Tree):
            @task
            @succeed
            class entry(Sequence):
                neg = task_(DummyNegative)
                pos = task_(DummyPositive)
        
        tree = X()
        status = tree.tick()
        assert status == Status.SUCCESS

    def test_until_with_one_sequence(self):

        class X(Tree):
            @task
            @until
            class entry(Sequence):
                range = task_(DummyRange)
                pos = task_(DummyPositive)
        
        tree = X()
        status = tree.tick()
        assert status == Status.RUNNING
        status = tree.tick()
        status = tree.tick()
        status = tree.tick()
        assert status == Status.SUCCESS


class TestParallelTask:

    def test_num_elements_with_one_element(self):

        class Pos(Parallel):
            pos = task_(DummyPositive)
        
        seq = Pos()
        assert seq.n == 1

    def test_tick_with_one_element(self):

        class Pos(Parallel):
            pos = task_(DummyPositive)
        
        seq = Pos()
        status = seq.tick()
        assert status == Status.SUCCESS

    def test_fail_on_first_with_two_elements(self):

        class X(Tree):
            @task
            @fail_on_first
            class entry(Parallel):
                pos = task_(DummyAction)
                neg = task_(DummyNegative)

        
        tree = X()
        status = tree.tick()
        assert status == Status.FAILURE


    def test_succeed_on_first_with_two_elements(self):

        class X(Tree):
            @task
            @succeed_on_first
            class entry(Parallel):
                pos = task_(DummyAction)
                neg = task_(DummyPositive)

        tree = X()
        status = tree.tick()
        assert status == Status.SUCCESS


class iterate_over(TickDecorator2nd):

    def __init__(self, iterations: int):
        super().__init__()
        self._iterations = iterations

    def decorate_tick(self, node):
        i = 0

        tick = node.tick
        def _(*args, **kwargs):
            nonlocal i
            status = tick(*args, **kwargs)
            i += 1
            if i >= self._iterations:
                return Status.SUCCESS
            return Status.FAILURE
        return _


class TestDecoratorLoader:

    def test_with_one_decorator_and_loader(self):

        loader = task_(DummyPositive) << loads(fail)
        assert isinstance(loader, TaskLoader)
    
    def test_with_two_decorators(self):

        loader = loads(fail) << loads(succeed)
        assert isinstance(loader.decorators[0].tick_decorator, succeed)
        assert isinstance(loader.decorators[1].tick_decorator, fail)

    def test_with_two_decorators(self):

        loader = task_(DummyPositive) << loads(fail) << loads(succeed)
        sequence = loader.decorator

        assert isinstance(sequence.decorators[0].tick_decorator, succeed)
        assert isinstance(sequence.decorators[1].tick_decorator, fail)

    def test_tick_with_two_decorators(self):

        loader = task_(DummyPositive) << loads(fail) << loads(succeed)
        task = loader.load(Storage(), "dummy")
        status = task.tick()
        assert status == Status.SUCCESS

    def test_tick_with_two_decorators_is_success(self):

        loader =  task_(DummyPositive) << loads(succeed) << loads(fail)
        task = loader.load(Storage(), "dummy")
        status = task.tick()
        assert status == Status.FAILURE

    def test_tick_with_two_decorators_and_second_order_is_success(self):

        loader = task_(DummyPositive)<< loads(fail) <<loads_(iterate_over, 1) 
        task = loader.load(Storage(), "dummy")
        status = task.tick()
        assert status == Status.SUCCESS

    def test_tick_with_two_decorators_and_second_order_is_failure(self):

        loader =task_(DummyPositive) << loads(fail) << loads_(iterate_over, 2)
        task = loader.load(Storage(), "dummy")
        status = task.tick()
        assert status == Status.FAILURE


class TestTreeReference:

    def test_trial_tree_x(self):

        class TrialTree(Tree):

            @task
            class entry(Sequence):
                x = action("x")

            def x(self):
                return Status.SUCCESS
            
        tree = TrialTree()
        assert tree.tick() == Status.SUCCESS

    def test_trial_tree_with_condition(self):
        
        class TrialTree(Tree):

            @task
            class entry(Sequence):
                x = cond("x")

            def x(self):
                return True
            
        tree = TrialTree()
        assert tree.tick() == Status.SUCCESS   

    def test_trial_tree_with_condvar(self):
        
        class TrialTree(Tree):

            def __init__(self, name: str=''):
                super().__init__(name)
                self.x = True

            @task
            class entry(Sequence):
                x = condvar("x")
            
        tree = TrialTree()
        assert tree.tick() == Status.SUCCESS   
